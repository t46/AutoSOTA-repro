"""Typer CLI for AutoSOTA with Rich output.

Usage examples::

    # Run optimisation on a paper
    autosota run --paper-pdf paper.pdf --repo-url https://github.com/user/repo \\
        --metric-name accuracy --metric-value 0.85

    # Check status
    autosota status abc123

    # Resume interrupted run
    autosota resume abc123
"""

from __future__ import annotations

import asyncio
import hashlib
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from autosota.config import Settings, load_settings
from autosota.constants import MetricDirection
from autosota.memory.scores import ScoreTracker
from autosota.models.task import TaskConfig
from autosota.pipeline.runner import PipelineRunner
from autosota.pipeline.state import PipelineState

app = typer.Typer(name="autosota", help="AutoSOTA: Automated SOTA optimization")
console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_config(config: Path | None, workspace: Path) -> Settings:
    """Load settings from YAML config, applying overrides."""
    if config is not None:
        settings = load_settings(config_dir=config.parent, profile=config.stem)
    else:
        settings = load_settings()
    settings.workspace_root = workspace
    return settings


def _make_task_config(
    paper_pdf: Path,
    repo_url: str,
    metric_name: str,
    metric_value: float,
    metric_direction: str,
) -> TaskConfig:
    """Build a TaskConfig from CLI arguments."""
    direction = (
        MetricDirection.HIGHER_IS_BETTER
        if metric_direction == "higher"
        else MetricDirection.LOWER_IS_BETTER
    )

    raw = f"{repo_url}:{metric_name}"
    task_id = hashlib.sha256(raw.encode()).hexdigest()[:12]

    return TaskConfig(
        task_id=task_id,
        paper_pdf=paper_pdf,
        repo_url=repo_url,
        target_metric=metric_name,
        metric_direction=direction,
        baseline_score=metric_value,
    )


def _run_async(coro):
    """Run an async coroutine, handling the event loop."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None:
        # Already inside an event loop (e.g. Jupyter)
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def run(
    paper_pdf: Path = typer.Option(..., help="Path to paper PDF"),
    repo_url: str = typer.Option(..., help="GitHub repository URL"),
    metric_name: str = typer.Option(..., help="Primary metric name (e.g., accuracy, FID)"),
    metric_value: float = typer.Option(..., help="Baseline metric value from paper"),
    metric_direction: str = typer.Option("higher", help="'higher' or 'lower' is better"),
    max_iterations: int = typer.Option(15, help="Max optimization iterations"),
    config: Path = typer.Option(None, help="Custom config YAML"),
    workspace: Path = typer.Option("./workspace", help="Workspace directory"),
) -> None:
    """Run AutoSOTA optimization on a single paper."""
    # Validate metric_direction
    if metric_direction not in ("higher", "lower"):
        console.print(
            "[red]Error:[/red] --metric-direction must be 'higher' or 'lower'",
        )
        raise typer.Exit(code=1)

    # Validate paper PDF exists
    if not paper_pdf.exists():
        console.print(f"[red]Error:[/red] Paper PDF not found: {paper_pdf}")
        raise typer.Exit(code=1)

    # Load config
    settings = _load_config(config, workspace)
    settings.pipeline.max_iterations = max_iterations

    # Create task config
    task = _make_task_config(paper_pdf, repo_url, metric_name, metric_value, metric_direction)

    console.print(
        Panel(
            f"[bold]Paper:[/bold] {paper_pdf.name}\n"
            f"[bold]Repo:[/bold] {repo_url}\n"
            f"[bold]Metric:[/bold] {metric_name} = {metric_value} ({metric_direction} is better)\n"
            f"[bold]Max iterations:[/bold] {max_iterations}\n"
            f"[bold]Workspace:[/bold] {workspace}",
            title="AutoSOTA Optimisation",
            border_style="blue",
        )
    )

    # Run pipeline
    runner = PipelineRunner(task=task, settings=settings, workspace_dir=workspace)

    with console.status("[bold green]Running optimisation pipeline...") as _status:
        try:
            state = _run_async(runner.run())
        except Exception as exc:
            console.print(f"\n[red]Pipeline failed:[/red] {exc}")
            raise typer.Exit(code=1) from exc

    # Show results
    _print_summary(state, metric_direction)


@app.command()
def status(
    paper_id: str = typer.Argument(..., help="Paper ID to check"),
    workspace: Path = typer.Option("./workspace", help="Workspace directory"),
) -> None:
    """Check optimization status for a paper."""
    try:
        state = PipelineState.load(workspace, paper_id)
    except FileNotFoundError:
        console.print(f"[red]No pipeline state found for paper_id=[/red]{paper_id}")
        raise typer.Exit(code=1)

    console.print(
        Panel(
            f"[bold]Paper ID:[/bold] {state.paper_id}\n"
            f"[bold]Phase:[/bold] {state.current_phase.value}\n"
            f"[bold]Iteration:[/bold] {state.current_iteration}\n"
            f"[bold]Baseline:[/bold] {state.baseline_score}\n"
            f"[bold]Best:[/bold] {state.best_score}\n"
            f"[bold]Complete:[/bold] {state.is_complete}\n"
            f"[bold]LLM calls:[/bold] {state.llm_calls_used}",
            title=f"Status: {paper_id}",
            border_style="cyan",
        )
    )

    # Show recent scores
    paper_dir = workspace / paper_id
    tracker = ScoreTracker(paper_dir)
    recent = tracker.get_recent(5)
    if recent:
        table = Table(title="Recent Scores")
        table.add_column("Iteration", style="cyan")
        table.add_column("Metric Value", style="green")
        table.add_column("Hypothesis", style="yellow")
        table.add_column("Improvement", style="magenta")

        for entry in recent:
            value = f"{entry.primary_metric_value:.4f}" if entry.primary_metric_value is not None else "N/A"
            table.add_row(
                str(entry.iteration),
                value,
                entry.hypothesis_title or "-",
                "Yes" if entry.is_improvement else "No",
            )
        console.print(table)


@app.command()
def resume(
    paper_id: str = typer.Argument(..., help="Paper ID to resume"),
    workspace: Path = typer.Option("./workspace", help="Workspace directory"),
) -> None:
    """Resume a previously interrupted optimization."""
    # Check that state exists
    try:
        state = PipelineState.load(workspace, paper_id)
    except FileNotFoundError:
        console.print(f"[red]No pipeline state found for paper_id=[/red]{paper_id}")
        raise typer.Exit(code=1)

    if state.is_complete:
        console.print(f"[yellow]Pipeline for {paper_id} is already complete.[/yellow]")
        raise typer.Exit(code=0)

    console.print(
        f"[bold]Resuming[/bold] {paper_id} from "
        f"phase={state.current_phase.value}, iteration={state.current_iteration}"
    )

    # We need settings -- load defaults
    settings = load_settings()
    settings.workspace_root = workspace

    # Build a minimal TaskConfig (the real task details are in the state)
    task = TaskConfig(task_id=paper_id)

    runner = PipelineRunner(task=task, settings=settings, workspace_dir=workspace)

    with console.status("[bold green]Resuming optimisation pipeline...") as _status:
        try:
            final_state = _run_async(runner.resume(paper_id))
        except Exception as exc:
            console.print(f"\n[red]Pipeline failed:[/red] {exc}")
            raise typer.Exit(code=1) from exc

    _print_summary(final_state, "higher")


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _print_summary(state: PipelineState, metric_direction: str) -> None:
    """Print a Rich summary table of the final pipeline state."""
    table = Table(title="Optimisation Summary")
    table.add_column("Property", style="bold")
    table.add_column("Value")

    table.add_row("Paper ID", state.paper_id)
    table.add_row("Final Phase", state.current_phase.value)
    table.add_row("Iterations", str(state.current_iteration))
    table.add_row("Baseline Score", str(state.baseline_score) if state.baseline_score is not None else "N/A")
    table.add_row("Best Score", str(state.best_score) if state.best_score is not None else "N/A")

    if state.baseline_score is not None and state.best_score is not None:
        delta = state.best_score - state.baseline_score
        sign = "+" if delta >= 0 else ""
        table.add_row("Improvement", f"{sign}{delta:.4f}")

    table.add_row("LLM Calls", str(state.llm_calls_used))
    hours = state.wall_clock_seconds / 3600
    table.add_row("Wall Clock", f"{hours:.2f} hours")
    table.add_row("Complete", str(state.is_complete))

    if state.abort_reason:
        table.add_row("Abort Reason", state.abort_reason)

    console.print(table)

    # Score history
    if state.score_history:
        score_table = Table(title="Score History")
        score_table.add_column("Iter", style="cyan")
        score_table.add_column("Value", style="green")
        score_table.add_column("Hypothesis", style="yellow")
        score_table.add_column("Improved", style="magenta")

        for entry in state.score_history:
            value = f"{entry.primary_metric_value:.4f}" if entry.primary_metric_value is not None else "N/A"
            score_table.add_row(
                str(entry.iteration),
                value,
                entry.hypothesis_title or "-",
                "Yes" if entry.is_improvement else "No",
            )
        console.print(score_table)
