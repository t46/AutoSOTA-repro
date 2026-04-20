"""Data models for execution state and supervisory actions."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from autosota.constants import ExecutionPhase


class ExecutionState(BaseModel):
    """Snapshot of the current execution within a Docker container."""

    phase: ExecutionPhase = ExecutionPhase.SETUP
    started_at: datetime | None = None
    finished_at: datetime | None = None
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    elapsed_seconds: float = 0.0
    is_success: bool = False


class SupervisoryAction(BaseModel):
    """An action recommended (or enforced) by the supervisor agent.

    Used when the supervisor detects red-line violations or suggests
    corrective measures mid-pipeline.
    """

    action_type: str = ""  # e.g. "halt", "rollback", "warn", "approve"
    guidance: str = ""
    reason: str = ""
    triggered_by_rule: str = ""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
