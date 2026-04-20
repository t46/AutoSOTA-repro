"""Cross-paper failure memory persisted as JSON.

Unlike the per-paper memory files (scores, code_analysis, idea_library),
the failure memory is shared across all papers so that common setup
failures and their repairs can be reused.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class FailureMemoryManager:
    """Cross-paper failure memory persisted as JSON.

    The file lives at ``<workspace_root>/failure_memory.json`` and is keyed
    by normalised error signatures.  Each entry stores a list of repair
    attempts and whether they succeeded.
    """

    def __init__(self, workspace_root: Path) -> None:
        self.file_path = workspace_root / "failure_memory.json"

    def load(self) -> dict[str, Any]:
        """Load the entire failure memory from disk.

        Returns an empty dict if the file does not exist.
        """
        if not self.file_path.exists():
            return {}
        try:
            data = json.loads(self.file_path.read_text())
            if not isinstance(data, dict):
                return {}
            return data
        except (json.JSONDecodeError, OSError):
            return {}

    def save(self, data: dict[str, Any]) -> None:
        """Write the full failure memory to disk."""
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.file_path.write_text(json.dumps(data, indent=2, default=str))

    def lookup(self, signature: str) -> list[dict[str, Any]] | None:
        """Look up repair attempts for a given error signature.

        Returns ``None`` if the signature has never been seen, otherwise
        returns a list of repair attempt dicts.
        """
        data = self.load()
        entry = data.get(signature)
        if entry is None:
            return None
        return entry.get("repairs", [])

    def record(self, signature: str, repair: dict[str, Any], success: bool) -> None:
        """Record a repair attempt against a failure signature.

        Parameters
        ----------
        signature:
            Normalised error signature (e.g. ``"ModuleNotFoundError:torch"``).
        repair:
            Dict describing the repair that was attempted.
        success:
            Whether the repair resolved the failure.
        """
        data = self.load()
        now = datetime.utcnow().isoformat()

        if signature not in data:
            data[signature] = {
                "first_seen": now,
                "last_seen": now,
                "occurrences": 0,
                "repairs": [],
            }

        entry = data[signature]
        entry["occurrences"] = entry.get("occurrences", 0) + 1
        entry["last_seen"] = now
        entry["repairs"].append(
            {
                "repair": repair,
                "success": success,
                "timestamp": now,
            }
        )

        self.save(data)
