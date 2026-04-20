"""Persistent memory for the idea / hypothesis library.

``idea_library.md`` stores the ranked set of hypotheses and their outcomes.
It is updated by AgentIdeator (Phase 2) and by the optimisation loop
(Phase 3) as hypotheses are tested.
"""

from __future__ import annotations

import re
from pathlib import Path


class IdeaLibraryManager:
    """Manages idea_library.md persistent memory."""

    def __init__(self, workspace_dir: Path) -> None:
        self.file_path = workspace_dir / "idea_library.md"

    def read(self) -> str:
        """Read the full contents of idea_library.md.

        Returns an empty string if the file does not exist.
        """
        if not self.file_path.exists():
            return ""
        return self.file_path.read_text()

    def write(self, content: str) -> None:
        """Write (overwrite) idea_library.md."""
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.file_path.write_text(content)

    def append_iteration_result(
        self,
        iteration: int,
        hypothesis_id: str,
        result: str,
        delta: float,
    ) -> None:
        """Append an iteration result to the end of idea_library.md.

        Parameters
        ----------
        iteration:
            Iteration number.
        hypothesis_id:
            ID of the hypothesis that was tested.
        result:
            Short description of the outcome (e.g. "improved", "regressed",
            "no change").
        delta:
            Change in the primary metric.
        """
        sign = "+" if delta >= 0 else ""
        entry = (
            f"\n### Iteration {iteration} -- {hypothesis_id}\n"
            f"- **Result**: {result}\n"
            f"- **Delta**: {sign}{delta:.4f}\n"
        )
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.file_path, "a") as fh:
            fh.write(entry)

    def mark_hypothesis_status(self, hypothesis_id: str, status: str) -> None:
        """Update the status marker of a hypothesis in-place.

        Looks for a line matching ``**Status**: ...`` under the hypothesis
        heading and replaces it.  If no such line exists, the status is
        appended after the heading line.
        """
        text = self.read()
        if not text:
            return

        # Build a pattern that matches the hypothesis heading and its status line
        heading_pattern = re.compile(
            rf"(###?\s+.*{re.escape(hypothesis_id)}.*\n)",
            re.IGNORECASE,
        )
        status_pattern = re.compile(r"- \*\*Status\*\*:.*\n?")

        # Find the heading
        heading_match = heading_pattern.search(text)
        if not heading_match:
            return

        heading_end = heading_match.end()
        # Check if there's an existing status line right after the heading area
        rest = text[heading_end:]
        status_match = status_pattern.search(rest)

        new_status_line = f"- **Status**: {status}\n"

        if status_match and status_match.start() < 200:
            # Replace existing status line (within 200 chars of heading)
            abs_start = heading_end + status_match.start()
            abs_end = heading_end + status_match.end()
            text = text[:abs_start] + new_status_line + text[abs_end:]
        else:
            # Insert status after heading
            text = text[:heading_end] + new_status_line + text[heading_end:]

        self.write(text)
