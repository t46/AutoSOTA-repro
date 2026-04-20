"""Data models for iteration score tracking."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class ScoreEntry(BaseModel):
    """Record of a single optimisation iteration's outcome."""

    iteration: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metrics: dict[str, float] = Field(default_factory=dict)
    primary_metric_value: float | None = None
    hypothesis_id: str = ""
    hypothesis_title: str = ""
    git_hash: str = ""
    is_improvement: bool = False
    notes: str = ""
