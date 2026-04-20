"""Unit tests for autosota.llm.provider."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from autosota.llm.provider import (
    LLMError,
    LLMProvider,
    LLMUsage,
    _augment_system_message,
    _strip_code_fences,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_response(content: str = "Hello", tool_calls=None):
    """Build a mock litellm response object."""
    msg = MagicMock()
    msg.role = "assistant"
    msg.content = content
    msg.tool_calls = tool_calls
    msg.function_call = None

    choice = MagicMock()
    choice.message = msg

    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 5
    usage.total_tokens = 15

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


# ---------------------------------------------------------------------------
# Tests: LLMUsage
# ---------------------------------------------------------------------------


class TestLLMUsage:
    def test_defaults(self):
        u = LLMUsage()
        assert u.total_tokens == 0
        assert u.prompt_tokens == 0
        assert u.completion_tokens == 0
        assert u.total_calls == 0
        assert u.total_cost_usd == 0.0

    def test_fields(self):
        u = LLMUsage(total_tokens=100, total_calls=2, total_cost_usd=0.05)
        assert u.total_tokens == 100
        assert u.total_calls == 2
        assert u.total_cost_usd == 0.05


# ---------------------------------------------------------------------------
# Tests: LLMProvider init
# ---------------------------------------------------------------------------


class TestLLMProviderInit:
    def test_default_init(self):
        p = LLMProvider(primary_model="test/model")
        assert p.primary_model == "test/model"
        assert p.fallback_model is None
        assert p.temperature == 0.3
        assert p.max_tokens == 16384
        assert p.timeout == 120
        assert p.max_retries == 3

    def test_custom_init(self):
        p = LLMProvider(
            primary_model="a/b",
            fallback_model="c/d",
            temperature=0.7,
            max_tokens=4096,
            timeout=60,
            max_retries=5,
        )
        assert p.primary_model == "a/b"
        assert p.fallback_model == "c/d"
        assert p.temperature == 0.7
        assert p.max_tokens == 4096

    def test_usage_starts_at_zero(self):
        p = LLMProvider(primary_model="x")
        u = p.usage
        assert u.total_calls == 0


# ---------------------------------------------------------------------------
# Tests: complete()
# ---------------------------------------------------------------------------


class TestComplete:
    @pytest.fixture
    def provider(self):
        return LLMProvider(
            primary_model="test/primary",
            fallback_model="test/fallback",
            max_retries=2,
        )

    @patch("autosota.llm.provider.litellm.acompletion")
    async def test_basic_complete(self, mock_acompletion, provider):
        mock_acompletion.return_value = _make_mock_response("world")

        result = await provider.complete(
            messages=[{"role": "user", "content": "hello"}]
        )
        assert result["role"] == "assistant"
        assert result["content"] == "world"
        mock_acompletion.assert_called_once()

    @patch("autosota.llm.provider.litellm.acompletion")
    async def test_usage_tracking(self, mock_acompletion, provider):
        mock_acompletion.return_value = _make_mock_response()

        await provider.complete(messages=[{"role": "user", "content": "hi"}])
        u = provider.usage
        assert u.total_calls == 1
        assert u.prompt_tokens == 10
        assert u.completion_tokens == 5
        assert u.total_tokens == 15

    @patch("autosota.llm.provider.litellm.acompletion")
    async def test_model_override(self, mock_acompletion, provider):
        mock_acompletion.return_value = _make_mock_response()

        await provider.complete(
            messages=[{"role": "user", "content": "hi"}],
            model_override="override/model",
        )
        call_kwargs = mock_acompletion.call_args[1]
        assert call_kwargs["model"] == "override/model"

    @patch("autosota.llm.provider.litellm.acompletion")
    async def test_fallback_on_primary_failure(self, mock_acompletion, provider):
        # First two calls (primary retries) fail, third (fallback) succeeds.
        mock_acompletion.side_effect = [
            Exception("fail1"),
            Exception("fail2"),
            _make_mock_response("fallback worked"),
        ]
        # Patch sleep to avoid waiting.
        with patch("autosota.llm.provider.asyncio.sleep", new_callable=AsyncMock):
            result = await provider.complete(
                messages=[{"role": "user", "content": "test"}]
            )
        assert result["content"] == "fallback worked"

    @patch("autosota.llm.provider.litellm.acompletion")
    async def test_all_models_fail_raises_error(self, mock_acompletion, provider):
        mock_acompletion.side_effect = Exception("always fails")
        with patch("autosota.llm.provider.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(LLMError, match="All LLM attempts exhausted"):
                await provider.complete(
                    messages=[{"role": "user", "content": "test"}]
                )

    @patch("autosota.llm.provider.litellm.acompletion")
    async def test_tools_passed_through(self, mock_acompletion, provider):
        mock_acompletion.return_value = _make_mock_response()
        tools = [{"type": "function", "function": {"name": "foo"}}]

        await provider.complete(
            messages=[{"role": "user", "content": "hi"}],
            tools=tools,
        )
        call_kwargs = mock_acompletion.call_args[1]
        assert call_kwargs["tools"] == tools

    @patch("autosota.llm.provider.litellm.acompletion")
    async def test_response_format_passed_through(self, mock_acompletion, provider):
        mock_acompletion.return_value = _make_mock_response()

        await provider.complete(
            messages=[{"role": "user", "content": "hi"}],
            response_format={"type": "json_object"},
        )
        call_kwargs = mock_acompletion.call_args[1]
        assert call_kwargs["response_format"] == {"type": "json_object"}


# ---------------------------------------------------------------------------
# Tests: complete_structured()
# ---------------------------------------------------------------------------


class TestCompleteStructured:
    class SampleModel(BaseModel):
        name: str
        score: float

    @patch("autosota.llm.provider.litellm.acompletion")
    async def test_structured_output(self, mock_acompletion):
        provider = LLMProvider(primary_model="test/model", max_retries=1)
        json_str = json.dumps({"name": "test", "score": 0.95})
        mock_acompletion.return_value = _make_mock_response(json_str)

        result = await provider.complete_structured(
            messages=[{"role": "user", "content": "get data"}],
            response_model=self.SampleModel,
        )
        assert isinstance(result, self.SampleModel)
        assert result.name == "test"
        assert result.score == 0.95

    @patch("autosota.llm.provider.litellm.acompletion")
    async def test_structured_strips_code_fences(self, mock_acompletion):
        provider = LLMProvider(primary_model="test/model", max_retries=1)
        fenced = '```json\n{"name": "fenced", "score": 1.0}\n```'
        mock_acompletion.return_value = _make_mock_response(fenced)

        result = await provider.complete_structured(
            messages=[{"role": "user", "content": "get data"}],
            response_model=self.SampleModel,
        )
        assert result.name == "fenced"


# ---------------------------------------------------------------------------
# Tests: reset_usage()
# ---------------------------------------------------------------------------


class TestResetUsage:
    @patch("autosota.llm.provider.litellm.acompletion")
    async def test_reset(self, mock_acompletion):
        provider = LLMProvider(primary_model="test/model", max_retries=1)
        mock_acompletion.return_value = _make_mock_response()

        await provider.complete(messages=[{"role": "user", "content": "hi"}])
        assert provider.usage.total_calls == 1

        provider.reset_usage()
        assert provider.usage.total_calls == 0
        assert provider.usage.total_tokens == 0


# ---------------------------------------------------------------------------
# Tests: helper functions
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_strip_code_fences_plain(self):
        assert _strip_code_fences('{"a": 1}') == '{"a": 1}'

    def test_strip_code_fences_json(self):
        assert _strip_code_fences('```json\n{"a": 1}\n```') == '{"a": 1}'

    def test_strip_code_fences_bare(self):
        assert _strip_code_fences('```\nfoo\n```') == "foo"

    def test_augment_system_message_existing(self):
        msgs = [
            {"role": "system", "content": "Base prompt"},
            {"role": "user", "content": "hello"},
        ]
        result = _augment_system_message(msgs, "EXTRA")
        assert "Base prompt" in result[0]["content"]
        assert "EXTRA" in result[0]["content"]
        # Original must not be mutated.
        assert "EXTRA" not in msgs[0]["content"]

    def test_augment_system_message_missing(self):
        msgs = [{"role": "user", "content": "hello"}]
        result = _augment_system_message(msgs, "EXTRA")
        assert result[0]["role"] == "system"
        assert "EXTRA" in result[0]["content"]
