"""Unit tests for autosota.tools.download_tools."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from autosota.tools.download_tools import DownloadError, DownloadTools


class TestDownloadUrl:
    async def test_downloads_file(self, tmp_path):
        dest = tmp_path / "file.txt"
        content = b"file content here"

        # Mock httpx streaming client.
        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_bytes = MagicMock(return_value=_async_iter([content]))

        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream.__aexit__ = AsyncMock(return_value=False)

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=mock_stream)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("autosota.tools.download_tools.httpx.AsyncClient", return_value=mock_client):
            result = await DownloadTools.download_url("https://example.com/file.txt", dest)

        assert result == dest
        assert dest.read_bytes() == content

    async def test_raises_on_failure(self, tmp_path):
        dest = tmp_path / "fail.txt"

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.stream = MagicMock(
            side_effect=httpx.HTTPError("download failed")
        )

        with patch("autosota.tools.download_tools.httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(DownloadError, match="Failed to download"):
                await DownloadTools.download_url("https://example.com/bad", dest)


class TestDownloadHuggingface:
    async def test_downloads_single_file(self, tmp_path):
        with patch("autosota.tools.download_tools.asyncio.to_thread") as mock_thread:
            expected = tmp_path / "model.bin"
            mock_thread.return_value = expected

            result = await DownloadTools.download_huggingface(
                repo_id="test/model", dest=tmp_path, filename="model.bin"
            )
            assert result == expected

    async def test_raises_on_failure(self, tmp_path):
        with patch("autosota.tools.download_tools.asyncio.to_thread") as mock_thread:
            mock_thread.side_effect = Exception("HF error")

            with pytest.raises(DownloadError, match="Failed to download"):
                await DownloadTools.download_huggingface(
                    repo_id="bad/repo", dest=tmp_path
                )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _async_iter:
    """Turn a list into an async iterator for mocking aiter_bytes."""

    def __init__(self, items):
        self._items = items
        self._idx = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._idx >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._idx]
        self._idx += 1
        return item
