"""Data models for paper metadata and code units."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from autosota.constants import Conference, MetricDirection


class PaperMetadata(BaseModel):
    """Metadata extracted from a research paper."""

    title: str
    authors: list[str] = Field(default_factory=list)
    conference: Conference | None = None
    year: int | None = None
    arxiv_id: str | None = None
    abstract: str = ""
    keywords: list[str] = Field(default_factory=list)
    pdf_path: Path | None = None
    repo_url: str | None = None

    # Metric information
    target_metric: str = ""
    metric_direction: MetricDirection = MetricDirection.HIGHER_IS_BETTER
    baseline_score: float | None = None


class PaperCodeUnit(BaseModel):
    """An execution-ready task unit combining paper + code information.

    This is the primary input to the optimization pipeline: it bundles
    everything the system needs to reproduce and improve a paper's results.
    """

    paper: PaperMetadata
    repo_url: str
    repo_local_path: Path | None = None
    entry_script: str = ""
    eval_command: str = ""
    dataset_path: str = ""
    setup_commands: list[str] = Field(default_factory=list)
    requirements_file: str = "requirements.txt"
    python_version: str = "3.10"
    additional_files: dict[str, str] = Field(default_factory=dict)
