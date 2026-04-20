"""Persistent memory for the research report.

``research_report.md`` stores the comprehensive research report produced
by AgentResource.  It captures the paper's contributions, methodology,
experimental setup, and key results that the optimisation pipeline needs
to understand.
"""

from __future__ import annotations

from pathlib import Path


class ResearchReportManager:
    """Manages research_report.md persistent memory."""

    def __init__(self, workspace_dir: Path) -> None:
        self.file_path = workspace_dir / "research_report.md"

    def exists(self) -> bool:
        """Return True if the research report has been written."""
        return self.file_path.exists()

    def read(self) -> str:
        """Read the full contents of research_report.md.

        Returns an empty string if the file does not exist.
        """
        if not self.file_path.exists():
            return ""
        return self.file_path.read_text()

    def write(self, content: str) -> None:
        """Write (overwrite) research_report.md."""
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.file_path.write_text(content)
