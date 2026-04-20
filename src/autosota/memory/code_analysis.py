"""Persistent memory for code analysis results.

``code_analysis.md`` stores the deep understanding of a paper's repository
produced by AgentResource during Phase 1.  It typically contains sections
for architecture overview, key files, hard constraints, and evaluation
pipeline description.
"""

from __future__ import annotations

import re
from pathlib import Path


class CodeAnalysisManager:
    """Manages code_analysis.md persistent memory."""

    def __init__(self, workspace_dir: Path) -> None:
        self.file_path = workspace_dir / "code_analysis.md"

    def exists(self) -> bool:
        """Return True if the code analysis file has been written."""
        return self.file_path.exists()

    def read(self) -> str:
        """Read the full contents of code_analysis.md.

        Returns an empty string if the file does not exist.
        """
        if not self.file_path.exists():
            return ""
        return self.file_path.read_text()

    def write(self, content: str) -> None:
        """Write (overwrite) code_analysis.md."""
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.file_path.write_text(content)

    def get_hard_constraints(self) -> list[str]:
        """Extract hard constraints section from code_analysis.md.

        Looks for a section headed ``## Hard Constraints`` (case-insensitive)
        and returns each bullet or numbered item as a string.  Returns an
        empty list if no such section exists.
        """
        text = self.read()
        if not text:
            return []

        # Find the hard constraints section
        pattern = re.compile(
            r"##\s*Hard\s+Constraints\s*\n(.*?)(?=\n##|\Z)",
            re.IGNORECASE | re.DOTALL,
        )
        match = pattern.search(text)
        if not match:
            return []

        section = match.group(1)
        constraints: list[str] = []
        for line in section.splitlines():
            line = line.strip()
            # Match bullet points (-, *, +) or numbered lists (1., 2.)
            cleaned = re.sub(r"^[-*+]\s+", "", line)
            cleaned = re.sub(r"^\d+[.)]\s+", "", cleaned)
            if cleaned and cleaned != line.strip():
                # Only include lines that actually had a list marker
                constraints.append(cleaned)
            elif cleaned and not line.startswith("#"):
                # Also include plain text lines (non-empty, non-heading)
                constraints.append(cleaned)
        return constraints
