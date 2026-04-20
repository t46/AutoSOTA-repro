"""Score tracking for optimisation progress.

Manages ``scores.jsonl`` -- a newline-delimited JSON file where each line
is a serialised :class:`ScoreEntry`.  This format is append-friendly and
easy to tail / grep from the command line.
"""

from __future__ import annotations

import json
from pathlib import Path

from autosota.models.iteration import ScoreEntry


class ScoreTracker:
    """Manages scores.jsonl for tracking optimization progress."""

    def __init__(self, workspace_dir: Path) -> None:
        self.scores_file = workspace_dir / "scores.jsonl"

    def record(self, entry: ScoreEntry) -> None:
        """Append a score entry to scores.jsonl."""
        self.scores_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.scores_file, "a") as fh:
            fh.write(entry.model_dump_json() + "\n")

    def get_all(self) -> list[ScoreEntry]:
        """Read all score entries."""
        if not self.scores_file.exists():
            return []
        entries: list[ScoreEntry] = []
        for line in self.scores_file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            entries.append(ScoreEntry.model_validate(json.loads(line)))
        return entries

    def get_best(self, metric_direction: str) -> ScoreEntry | None:
        """Get the best scoring entry.

        Parameters
        ----------
        metric_direction:
            ``"higher"`` or ``"lower"`` -- determines whether the best entry
            is the one with the highest or lowest ``primary_metric_value``.
        """
        entries = [e for e in self.get_all() if e.primary_metric_value is not None]
        if not entries:
            return None
        if metric_direction == "lower":
            return min(entries, key=lambda e: e.primary_metric_value)  # type: ignore[arg-type]
        return max(entries, key=lambda e: e.primary_metric_value)  # type: ignore[arg-type]

    def get_recent(self, n: int = 3) -> list[ScoreEntry]:
        """Get last *n* entries."""
        all_entries = self.get_all()
        return all_entries[-n:]
