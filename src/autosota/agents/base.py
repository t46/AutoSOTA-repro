"""Base agent class for all AutoSOTA agents.

Every agent inherits from ``BaseAgent`` and implements the ``run`` method.
Agents communicate with the LLM through ``_call_llm`` which automatically
prepends the agent's system prompt loaded from ``prompts/<name>_system.md``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import structlog
from pydantic import BaseModel, Field


class AgentResult(BaseModel):
    """Standard result returned by every agent's ``run`` method."""

    success: bool
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    llm_calls_used: int = 0


class BaseAgent(ABC):
    """Base class for all AutoSOTA agents.

    Parameters
    ----------
    llm:
        An LLM provider instance exposing ``complete`` and
        ``complete_structured`` async methods.
    workspace_dir:
        Root directory where the pipeline stores artefacts.
    config:
        Parsed pipeline configuration dictionary (from ``configs/*.yaml``).
    """

    def __init__(self, llm: Any, workspace_dir: Path, config: dict[str, Any]) -> None:
        self.llm = llm
        self.workspace_dir = workspace_dir
        self.config = config
        self.logger = structlog.get_logger(agent=self.__class__.__name__)
        self._system_prompt = self._load_system_prompt()

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    def _load_system_prompt(self) -> str:
        """Load from ``prompts/{agent_name}_system.md``.

        The agent name is derived by lower-casing the class name and
        stripping a trailing ``agent`` suffix.  For example
        ``AgentResource`` becomes ``resource``.
        """
        prompts_dir = Path(__file__).parent.parent.parent.parent / "prompts"
        name = self.__class__.__name__.lower().replace("agent", "")
        prompt_file = prompts_dir / f"{name}_system.md"
        if prompt_file.exists():
            return prompt_file.read_text()
        return f"You are {self.__class__.__name__}."

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------

    async def _call_llm(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> Any:
        """Convenience wrapper that prepends the system prompt.

        Parameters
        ----------
        messages:
            User/assistant conversation messages.
        tools:
            Optional tool definitions for function-calling.
        response_model:
            If given, the LLM will be asked to return structured output
            conforming to this Pydantic model.
        """
        full_messages = [{"role": "system", "content": self._system_prompt}] + messages
        if response_model:
            return await self.llm.complete_structured(full_messages, response_model)
        return await self.llm.complete(full_messages, tools=tools)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    async def run(self, **kwargs: Any) -> AgentResult:
        """Execute the agent's primary task and return a result."""
        ...
