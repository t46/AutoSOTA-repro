"""Unit tests for autosota.tools.web_tools."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from autosota.tools.web_tools import WebError, WebTools


@pytest.fixture
def mock_client():
    """Return a mock httpx.AsyncClient."""
    client = AsyncMock(spec=httpx.AsyncClient)
    return client


class TestParseGithubUrl:
    def test_full_url(self):
        owner, repo = WebTools._parse_github_url("https://github.com/owner/repo")
        assert owner == "owner"
        assert repo == "repo"

    def test_url_with_git_suffix(self):
        owner, repo = WebTools._parse_github_url("https://github.com/owner/repo.git")
        assert repo == "repo"

    def test_shorthand(self):
        owner, repo = WebTools._parse_github_url("owner/repo")
        assert owner == "owner"
        assert repo == "repo"

    def test_invalid_url(self):
        with pytest.raises(WebError, match="Cannot parse"):
            WebTools._parse_github_url("https://github.com/only_owner")

    def test_url_with_extra_path(self):
        owner, repo = WebTools._parse_github_url(
            "https://github.com/owner/repo/tree/main/src"
        )
        assert owner == "owner"
        assert repo == "repo"


class TestSearchGithubRepos:
    async def test_search_returns_results(self, mock_client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "items": [
                {
                    "full_name": "owner/repo",
                    "html_url": "https://github.com/owner/repo",
                    "description": "A test repo",
                    "stargazers_count": 100,
                    "language": "Python",
                    "updated_at": "2024-01-01T00:00:00Z",
                }
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_resp)

        tools = WebTools(httpx_client=mock_client)
        results = await tools.search_github_repos("test query")
        assert len(results) == 1
        assert results[0]["full_name"] == "owner/repo"
        assert results[0]["stars"] == 100


class TestFetchUrl:
    async def test_fetch_returns_text(self, mock_client):
        mock_resp = MagicMock()
        mock_resp.text = "<html>hello</html>"
        mock_resp.raise_for_status = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_resp)

        tools = WebTools(httpx_client=mock_client)
        text = await tools.fetch_url("https://example.com")
        assert "hello" in text

    async def test_fetch_raises_on_error(self, mock_client):
        mock_client.get = AsyncMock(
            side_effect=httpx.HTTPError("connection failed")
        )
        tools = WebTools(httpx_client=mock_client)
        with pytest.raises(WebError, match="Failed to fetch"):
            await tools.fetch_url("https://example.com/bad")


class TestCrawlGithubReadme:
    async def test_crawl_readme_via_api(self, mock_client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "# README content"
        mock_client.get = AsyncMock(return_value=mock_resp)

        tools = WebTools(httpx_client=mock_client)
        readme = await tools.crawl_github_readme("https://github.com/owner/repo")
        assert "README content" in readme
