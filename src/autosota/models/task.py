"""Data models for top-level task configuration."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from autosota.constants import MetricDirection


class TaskConfig(BaseModel):
    """Top-level input that describes a single AutoSOTA optimisation task.

    This is the user-facing entry point: it specifies the paper to reproduce,
    where its code lives, and which metric to optimise.
    """

    task_id: str = ""
    paper_pdf: Path | None = None
    repo_url: str = ""
    repo_local_path: Path | None = None

    # Metric
    target_metric: str = ""
    metric_direction: MetricDirection = MetricDirection.HIGHER_IS_BETTER
    baseline_score: float | None = None

    # Execution
    eval_command: str = ""
    output_pattern: str = ""
    dataset_path: str = ""
    python_version: str = "3.10"
    setup_commands: list[str] = Field(default_factory=list)

    # Constraints
    hard_constraints: list[str] = Field(default_factory=list)
    notes: str = ""
