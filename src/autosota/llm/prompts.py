"""Prompt template management for AutoSOTA agents.

Loads system prompts from ``prompts/*.md`` files and renders Jinja2 templates
with variable substitution.
"""

from __future__ import annotations

import logging
from pathlib import Path

from jinja2 import BaseLoader, Environment, TemplateNotFound

logger = logging.getLogger(__name__)


class PromptManager:
    """Manage and render agent prompt templates.

    Parameters
    ----------
    prompts_dir:
        Directory containing ``{agent_name}_system.md`` and other template files.
    """

    def __init__(self, prompts_dir: Path) -> None:
        self.prompts_dir = Path(prompts_dir)
        self._env = Environment(
            loader=_DirectoryLoader(self.prompts_dir),
            keep_trailing_newline=True,
            autoescape=False,
        )
        # In-memory template cache keyed by file stem.
        self._cache: dict[str, str] = {}

    def get_system_prompt(self, agent_name: str) -> str:
        """Load and return the raw content of ``prompts/{agent_name}_system.md``.

        Raises :class:`FileNotFoundError` if the prompt file does not exist.
        """
        filename = f"{agent_name}_system.md"
        return self._load_file(filename)

    def render(self, template_name: str, **kwargs: object) -> str:
        """Render a Jinja2 template file with the given variables.

        Parameters
        ----------
        template_name:
            Name of the template file inside *prompts_dir* (e.g. ``"analyze.md"``).
        **kwargs:
            Variables to inject into the template.

        Returns
        -------
        str
            The rendered template string.
        """
        try:
            template = self._env.get_template(template_name)
        except TemplateNotFound:
            raise FileNotFoundError(
                f"Prompt template '{template_name}' not found in {self.prompts_dir}"
            )
        return template.render(**kwargs)

    def list_templates(self) -> list[str]:
        """Return a sorted list of template filenames in the prompts directory."""
        if not self.prompts_dir.is_dir():
            return []
        return sorted(p.name for p in self.prompts_dir.iterdir() if p.is_file())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_file(self, filename: str) -> str:
        """Read a file from the prompts directory, with simple caching."""
        if filename in self._cache:
            return self._cache[filename]

        filepath = self.prompts_dir / filename
        if not filepath.is_file():
            raise FileNotFoundError(
                f"Prompt file '{filename}' not found in {self.prompts_dir}"
            )
        content = filepath.read_text(encoding="utf-8")
        self._cache[filename] = content
        return content


class _DirectoryLoader(BaseLoader):
    """A simple Jinja2 loader that reads templates from a filesystem directory."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def get_source(
        self, environment: Environment, template: str
    ) -> tuple[str, str, callable]:
        filepath = self.directory / template
        if not filepath.is_file():
            raise TemplateNotFound(template)
        source = filepath.read_text(encoding="utf-8")
        mtime = filepath.stat().st_mtime

        def uptodate() -> bool:
            try:
                return filepath.stat().st_mtime == mtime
            except OSError:
                return False

        return source, str(filepath), uptodate
