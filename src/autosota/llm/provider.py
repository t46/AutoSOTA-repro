"""Unified LLM provider wrapping litellm with retry, fallback, and usage tracking."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import litellm
from dotenv import load_dotenv
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Load API keys from the secrets .env file if it exists, then from process env.
_SECRETS_ENV = Path.home() / "Library" / "CloudStorage" / "Dropbox" / "secrets" / ".env"
if _SECRETS_ENV.exists():
    load_dotenv(_SECRETS_ENV, override=False)

# Suppress litellm's noisy debug output by default.
litellm.suppress_debug_info = True


class LLMUsage(BaseModel):
    """Accumulated LLM usage statistics."""

    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_calls: int = 0
    total_cost_usd: float = 0.0


class LLMError(Exception):
    """Raised when all LLM call attempts (including fallbacks) are exhausted."""


class LLMProvider:
    """Async LLM client with automatic retry, multi-model fallback, and cost tracking.

    Parameters
    ----------
    primary_model:
        Model identifier understood by litellm (e.g. ``"anthropic/claude-sonnet-4-20250514"``).
    fallback_model:
        Optional secondary model tried when the primary model fails after all retries.
    temperature:
        Sampling temperature passed to the model.
    max_tokens:
        Maximum tokens in the completion.
    timeout:
        Per-request timeout in seconds.
    max_retries:
        Number of retry attempts (per model) before moving to the fallback.
    """

    def __init__(
        self,
        primary_model: str,
        fallback_model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 16384,
        timeout: int = 120,
        max_retries: int = 3,
    ) -> None:
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self._usage = LLMUsage()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        response_format: dict | None = None,
        model_override: str | None = None,
    ) -> dict:
        """Call the LLM with retry and fallback logic.

        Returns the ``message`` dict from the first choice in the response.

        Raises :class:`LLMError` if all attempts (primary + fallback) fail.
        """
        models_to_try: list[str] = []
        if model_override:
            models_to_try.append(model_override)
        else:
            models_to_try.append(self.primary_model)
            if self.fallback_model:
                models_to_try.append(self.fallback_model)

        last_error: Exception | None = None

        for model in models_to_try:
            try:
                return await self._call_with_retries(
                    model=model,
                    messages=messages,
                    tools=tools,
                    response_format=response_format,
                )
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "LLM call failed for model %s: %s. Trying next model if available.",
                    model,
                    exc,
                )

        raise LLMError(
            f"All LLM attempts exhausted. Last error: {last_error}"
        ) from last_error

    async def complete_structured(
        self,
        messages: list[dict],
        response_model: type[BaseModel],
        model_override: str | None = None,
    ) -> BaseModel:
        """Get structured output parsed into a Pydantic model.

        Uses JSON mode and instructs the LLM to return a JSON object matching
        the schema of *response_model*.
        """
        schema = response_model.model_json_schema()
        system_addition = (
            "You MUST respond with a JSON object conforming to this schema:\n"
            f"```json\n{json.dumps(schema, indent=2)}\n```\n"
            "Return ONLY valid JSON. No markdown fences, no extra text."
        )

        # Prepend or augment the system message.
        augmented_messages = _augment_system_message(messages, system_addition)

        response_format = {"type": "json_object"}

        msg = await self.complete(
            messages=augmented_messages,
            response_format=response_format,
            model_override=model_override,
        )

        content = msg.get("content", "")
        # Strip potential markdown code fences the model might still produce.
        content = _strip_code_fences(content)
        data = json.loads(content)
        return response_model.model_validate(data)

    @property
    def usage(self) -> LLMUsage:
        """Return accumulated usage statistics."""
        return self._usage.model_copy()

    def reset_usage(self) -> None:
        """Reset accumulated usage counters to zero."""
        self._usage = LLMUsage()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _call_with_retries(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict] | None,
        response_format: dict | None,
    ) -> dict:
        """Attempt the LLM call up to *max_retries* times with exponential backoff."""
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                return await self._single_call(
                    model=model,
                    messages=messages,
                    tools=tools,
                    response_format=response_format,
                )
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    wait = min(2 ** attempt, 30)
                    logger.info(
                        "Retry %d/%d for model %s after %.1fs (error: %s)",
                        attempt,
                        self.max_retries,
                        model,
                        wait,
                        exc,
                    )
                    await asyncio.sleep(wait)

        raise last_error  # type: ignore[misc]

    async def _single_call(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict] | None,
        response_format: dict | None,
    ) -> dict:
        """Execute a single litellm.acompletion call and update usage stats."""
        kwargs: dict = {
            "model": model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
        }
        if tools:
            kwargs["tools"] = tools
        if response_format:
            kwargs["response_format"] = response_format

        response = await litellm.acompletion(**kwargs)

        # Update usage tracking.
        self._update_usage(response, model)

        choice = response.choices[0]
        msg = choice.message

        # Return a plain dict for broad compatibility.
        return _message_to_dict(msg)

    def _update_usage(self, response: object, model: str) -> None:
        """Accumulate token and cost statistics from a litellm response."""
        usage = getattr(response, "usage", None)
        if usage:
            self._usage.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            self._usage.completion_tokens += getattr(usage, "completion_tokens", 0) or 0
            self._usage.total_tokens += getattr(usage, "total_tokens", 0) or 0

        self._usage.total_calls += 1

        # litellm.completion_cost can estimate cost from the response.
        try:
            cost = litellm.completion_cost(completion_response=response)
            self._usage.total_cost_usd += cost
        except Exception:
            # Cost estimation may fail for certain models; ignore gracefully.
            pass


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _message_to_dict(msg: object) -> dict:
    """Convert a litellm/openai message object to a plain dict."""
    result: dict = {}
    for attr in ("role", "content", "tool_calls", "function_call"):
        val = getattr(msg, attr, None)
        if val is not None:
            # tool_calls are objects; convert each to dict.
            if attr == "tool_calls" and isinstance(val, list):
                val = [_tool_call_to_dict(tc) for tc in val]
            result[attr] = val
    return result


def _tool_call_to_dict(tc: object) -> dict:
    """Convert a single tool-call object to a JSON-friendly dict."""
    result: dict = {}
    for attr in ("id", "type"):
        val = getattr(tc, attr, None)
        if val is not None:
            result[attr] = val
    fn = getattr(tc, "function", None)
    if fn is not None:
        result["function"] = {
            "name": getattr(fn, "name", ""),
            "arguments": getattr(fn, "arguments", ""),
        }
    return result


def _augment_system_message(messages: list[dict], addition: str) -> list[dict]:
    """Return a copy of *messages* with *addition* appended to the system message."""
    messages = [m.copy() for m in messages]
    for m in messages:
        if m.get("role") == "system":
            m["content"] = m.get("content", "") + "\n\n" + addition
            return messages
    # No system message found; insert one.
    messages.insert(0, {"role": "system", "content": addition})
    return messages


def _strip_code_fences(text: str) -> str:
    """Remove leading/trailing markdown code fences from *text*."""
    text = text.strip()
    if text.startswith("```"):
        # Remove first line (```json or ```)
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
    if text.endswith("```"):
        text = text[: -3]
    return text.strip()
