"""Main pipeline orchestrator for AutoSOTA.

The :class:`PipelineRunner` ties together all agents, memory managers, and
execution infrastructure to run the full four-phase optimisation loop:

1. **Phase 0 -- Baseline**: AgentResource gathers information,
   AgentObjective defines success criteria, AgentInit reproduces
   the baseline result.
2. **Phase 1 -- Understanding**: Deep code analysis produces
   ``code_analysis.md`` and ``research_report.md``.
3. **Phase 2 -- Idea Library**: AgentIdeator generates and ranks
   improvement hypotheses in ``idea_library.md``.
4. **Phase 3 -- Iterative Optimisation**: AgentScheduler orchestrates
   the iteration loop (Monitor -> Fix -> Ideator -> Supervisor).
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

import structlog

from autosota.config import Settings, load_settings
from autosota.memory.code_analysis import CodeAnalysisManager
from autosota.memory.failure_memory import FailureMemoryManager
from autosota.memory.idea_library import IdeaLibraryManager
from autosota.memory.research_report import ResearchReportManager
from autosota.memory.scores import ScoreTracker
from autosota.models.task import TaskConfig
from autosota.pipeline.phases import advance_phase, can_advance, is_terminal
from autosota.pipeline.state import OptimizationPhase, PipelineState

logger = structlog.get_logger(__name__)


def _paper_id_from_task(task: TaskConfig) -> str:
    """Derive a deterministic paper_id from the task configuration."""
    if task.task_id:
        return task.task_id
    # Fall back to a hash of repo_url + metric
    raw = f"{task.repo_url}:{task.target_metric}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


class PipelineRunner:
    """Main orchestrator that runs the full AutoSOTA pipeline."""

    def __init__(
        self,
        task: TaskConfig,
        settings: Settings,
        workspace_dir: Path,
    ) -> None:
        self.task = task
        self.settings = settings
        self.workspace_dir = workspace_dir
        self.state: PipelineState | None = None

        # Memory managers (initialised lazily in _setup_workspace)
        self.scores: ScoreTracker | None = None
        self.code_analysis: CodeAnalysisManager | None = None
        self.idea_library: IdeaLibraryManager | None = None
        self.research_report: ResearchReportManager | None = None
        self.failure_memory: FailureMemoryManager | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self) -> PipelineState:
        """Execute full pipeline: Resource -> Objective -> Init -> Scheduler.

        Returns the final :class:`PipelineState`.
        """
        paper_dir = self._setup_workspace()
        paper_id = paper_dir.name

        self.state = PipelineState(
            paper_id=paper_id,
            current_phase=OptimizationPhase.PHASE_0_BASELINE,
            wall_clock_start=datetime.utcnow(),
        )
        self.state.save(self.workspace_dir)
        logger.info("pipeline.start", paper_id=paper_id)

        try:
            # Stage 1: Resource gathering + Objective definition + Baseline
            await self._run_phase_0_baseline()

            # Stage 2: Deep understanding
            await self._run_phase_1_understanding()

            # Stage 3: Idea library construction
            await self._run_phase_2_idea_library()

            # Stage 4: Iterative optimisation
            await self._run_phase_3_optimization()

        except Exception:
            logger.exception("pipeline.error", paper_id=paper_id)
            raise
        finally:
            if self.state is not None:
                self.state.wall_clock_seconds = (
                    datetime.utcnow() - (self.state.wall_clock_start or datetime.utcnow())
                ).total_seconds()
                self.state.save(self.workspace_dir)

        # Generate final report
        report = await self._generate_report(self.state)
        report_path = paper_dir / "OPTIMIZATION.md"
        report_path.write_text(report)
        logger.info("pipeline.complete", paper_id=paper_id, report=str(report_path))

        return self.state

    async def resume(self, paper_id: str) -> PipelineState:
        """Resume from saved state.

        Loads the checkpoint from ``<workspace_dir>/<paper_id>/pipeline_state.json``
        and continues from wherever execution was interrupted.
        """
        self.state = PipelineState.load(self.workspace_dir, paper_id)
        paper_dir = self.workspace_dir / paper_id
        self._init_memory_managers(paper_dir)

        logger.info(
            "pipeline.resume",
            paper_id=paper_id,
            phase=self.state.current_phase,
            iteration=self.state.current_iteration,
        )

        try:
            phase = self.state.current_phase

            if phase == OptimizationPhase.PHASE_0_BASELINE:
                await self._run_phase_0_baseline()
                await self._run_phase_1_understanding()
                await self._run_phase_2_idea_library()
                await self._run_phase_3_optimization()

            elif phase == OptimizationPhase.PHASE_1_UNDERSTANDING:
                await self._run_phase_1_understanding()
                await self._run_phase_2_idea_library()
                await self._run_phase_3_optimization()

            elif phase == OptimizationPhase.PHASE_2_IDEA_LIBRARY:
                await self._run_phase_2_idea_library()
                await self._run_phase_3_optimization()

            elif phase == OptimizationPhase.PHASE_3_OPTIMIZATION:
                await self._run_phase_3_optimization()

        except Exception:
            logger.exception("pipeline.resume_error", paper_id=paper_id)
            raise
        finally:
            if self.state is not None:
                self.state.save(self.workspace_dir)

        return self.state

    # ------------------------------------------------------------------
    # Phase implementations (stubs -- agents will be wired in later)
    # ------------------------------------------------------------------

    async def _run_phase_0_baseline(self) -> None:
        """Phase 0: Resource gathering, objective definition, baseline reproduction."""
        assert self.state is not None
        logger.info("phase.start", phase="phase_0_baseline")

        # TODO: Wire AgentResource to gather paper info and clone repo
        # TODO: Wire AgentObjective to define success criteria
        # TODO: Wire AgentInit to reproduce the baseline

        # Placeholder: mark baseline as recorded so the phase can advance
        # In the real implementation, the agents will set this.
        logger.info("phase.complete", phase="phase_0_baseline")

    async def _run_phase_1_understanding(self) -> None:
        """Phase 1: Deep code understanding."""
        assert self.state is not None

        ok, reason = can_advance(self.state, self.workspace_dir)
        if ok:
            advance_phase(self.state, self.workspace_dir)
        else:
            logger.warning("phase.skip", phase="phase_1_understanding", reason=reason)
            return

        logger.info("phase.start", phase="phase_1_understanding")

        # TODO: Wire AgentResource for deep code analysis -> code_analysis.md
        # TODO: Wire AgentResource for research report -> research_report.md

        logger.info("phase.complete", phase="phase_1_understanding")

    async def _run_phase_2_idea_library(self) -> None:
        """Phase 2: Hypothesis generation and ranking."""
        assert self.state is not None

        ok, reason = can_advance(self.state, self.workspace_dir)
        if ok:
            advance_phase(self.state, self.workspace_dir)
        else:
            logger.warning("phase.skip", phase="phase_2_idea_library", reason=reason)
            return

        logger.info("phase.start", phase="phase_2_idea_library")

        # TODO: Wire AgentIdeator to generate hypotheses -> idea_library.md
        # TODO: Wire AgentSupervisor for red-line audit of hypotheses

        logger.info("phase.complete", phase="phase_2_idea_library")

    async def _run_phase_3_optimization(self) -> None:
        """Phase 3: Iterative optimisation loop.

        The 7-step iteration loop:
        1. AgentScheduler selects next hypothesis
        2. Apply hypothesis (code changes)
        3. Execute in Docker container
        4. AgentMonitor parses results
        5. AgentFix handles failures (up to max_fix_attempts)
        6. AgentSupervisor red-line audit
        7. Record outcome, update idea library, checkpoint
        """
        assert self.state is not None
        assert self.scores is not None
        assert self.idea_library is not None

        ok, reason = can_advance(self.state, self.workspace_dir)
        if ok:
            advance_phase(self.state, self.workspace_dir)
        else:
            logger.warning("phase.skip", phase="phase_3_optimization", reason=reason)
            return

        logger.info("phase.start", phase="phase_3_optimization")

        max_iters = self.settings.pipeline.max_iterations
        stagnation_window = self.settings.pipeline.stagnation_window

        for iteration in range(self.state.current_iteration, max_iters):
            self.state.current_iteration = iteration
            logger.info("iteration.start", iteration=iteration)

            # Budget check
            if self._budget_exhausted():
                logger.warning("iteration.budget_exhausted", iteration=iteration)
                self.state.abort_reason = "Budget exhausted"
                break

            # Stagnation check
            if self._is_stagnating(stagnation_window):
                logger.warning("iteration.stagnating", iteration=iteration)
                # Continue but log -- the scheduler should switch strategy

            # TODO: Wire the 7-step iteration loop with actual agents
            # For now, the loop structure is in place for agents to fill in.

            # Checkpoint after each iteration
            self.state.save(self.workspace_dir)
            logger.info("iteration.complete", iteration=iteration)

        self.state.is_complete = True
        logger.info("phase.complete", phase="phase_3_optimization")

    # ------------------------------------------------------------------
    # Workspace setup
    # ------------------------------------------------------------------

    def _setup_workspace(self) -> Path:
        """Create workspace/<paper_id>/ directory structure.

        Returns the paper-specific workspace directory.
        """
        paper_id = _paper_id_from_task(self.task)
        paper_dir = self.workspace_dir / paper_id

        # Create standard subdirectories
        for subdir in ["repo", "logs", "diffs", "checkpoints"]:
            (paper_dir / subdir).mkdir(parents=True, exist_ok=True)

        self._init_memory_managers(paper_dir)
        return paper_dir

    def _init_memory_managers(self, paper_dir: Path) -> None:
        """Initialise all memory managers for a paper workspace."""
        self.scores = ScoreTracker(paper_dir)
        self.code_analysis = CodeAnalysisManager(paper_dir)
        self.idea_library = IdeaLibraryManager(paper_dir)
        self.research_report = ResearchReportManager(paper_dir)
        self.failure_memory = FailureMemoryManager(self.workspace_dir)

    # ------------------------------------------------------------------
    # Budget & stagnation helpers
    # ------------------------------------------------------------------

    def _budget_exhausted(self) -> bool:
        """Check whether the wall-clock or LLM-call budget has been exceeded."""
        assert self.state is not None
        settings = self.settings.pipeline

        # Wall-clock budget
        if self.state.wall_clock_start is not None:
            elapsed = (datetime.utcnow() - self.state.wall_clock_start).total_seconds()
            if elapsed > settings.budget_wall_clock_hours * 3600:
                return True

        # LLM call budget
        if self.state.llm_calls_used >= settings.budget_llm_calls:
            return True

        return False

    def _is_stagnating(self, window: int) -> bool:
        """Check whether the last *window* iterations showed no improvement."""
        assert self.state is not None
        recent = self.state.score_history[-window:]
        if len(recent) < window:
            return False
        return all(not entry.is_improvement for entry in recent)

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    async def _generate_report(self, state: PipelineState) -> str:
        """Generate OPTIMIZATION.md from final state."""
        lines = [
            f"# Optimisation Report: {state.paper_id}",
            "",
            f"**Phase**: {state.current_phase.value}",
            f"**Iterations completed**: {state.current_iteration}",
            f"**Baseline score**: {state.baseline_score}",
            f"**Best score**: {state.best_score}",
            "",
        ]

        if state.baseline_score is not None and state.best_score is not None:
            delta = state.best_score - state.baseline_score
            sign = "+" if delta >= 0 else ""
            lines.append(f"**Improvement**: {sign}{delta:.4f}")
            lines.append("")

        if state.abort_reason:
            lines.append(f"**Abort reason**: {state.abort_reason}")
            lines.append("")

        # Score history table
        if state.score_history:
            lines.append("## Score History")
            lines.append("")
            lines.append("| Iteration | Metric Value | Hypothesis | Improvement |")
            lines.append("|-----------|-------------|------------|-------------|")
            for entry in state.score_history:
                value = f"{entry.primary_metric_value:.4f}" if entry.primary_metric_value is not None else "N/A"
                improved = "Yes" if entry.is_improvement else "No"
                lines.append(
                    f"| {entry.iteration} | {value} | {entry.hypothesis_title} | {improved} |"
                )
            lines.append("")

        # Wall clock
        lines.append("## Resource Usage")
        lines.append("")
        hours = state.wall_clock_seconds / 3600
        lines.append(f"- **Wall clock**: {hours:.2f} hours")
        lines.append(f"- **LLM calls**: {state.llm_calls_used}")
        lines.append("")

        return "\n".join(lines)
