"""Data models for failure diagnosis and repair."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class FailureSignal(BaseModel):
    """Structured representation of a single execution failure."""

    error_type: str = ""
    message: str = ""
    traceback: str = ""
    normalized_signature: str = ""
    file_path: str = ""
    line_number: int | None = None


class RepairAction(BaseModel):
    """A concrete repair that was attempted or applied."""

    description: str = ""
    commands: list[str] = Field(default_factory=list)
    file_patches: dict[str, str] = Field(default_factory=dict)
    skill_used: str = ""
    succeeded: bool = False


class FailureMemoryEntry(BaseModel):
    """A single entry in the failure memory bank.

    Links a failure signal to the repairs that were attempted and whether
    they resolved the issue.
    """

    failure: FailureSignal
    repairs_attempted: list[RepairAction] = Field(default_factory=list)
    resolved: bool = False
    first_seen: datetime = Field(default_factory=datetime.utcnow)
    last_seen: datetime = Field(default_factory=datetime.utcnow)
    occurrences: int = 1


class FailureMemory(BaseModel):
    """Accumulated memory of failures keyed by their normalised signatures.

    The pipeline consults this memory to avoid repeating unsuccessful
    repairs and to accelerate diagnosis of recurring issues.
    """

    entries: dict[str, FailureMemoryEntry] = Field(default_factory=dict)

    def record(self, failure: FailureSignal, repair: RepairAction | None = None) -> None:
        """Record a failure occurrence, optionally with a repair attempt."""
        sig = failure.normalized_signature
        if sig in self.entries:
            entry = self.entries[sig]
            entry.occurrences += 1
            entry.last_seen = datetime.utcnow()
            if repair is not None:
                entry.repairs_attempted.append(repair)
                if repair.succeeded:
                    entry.resolved = True
        else:
            entry = FailureMemoryEntry(
                failure=failure,
                repairs_attempted=[repair] if repair is not None else [],
            )
            self.entries[sig] = entry

    def is_known(self, signature: str) -> bool:
        return signature in self.entries

    def successful_repairs_for(self, signature: str) -> list[RepairAction]:
        entry = self.entries.get(signature)
        if entry is None:
            return []
        return [r for r in entry.repairs_attempted if r.succeeded]
