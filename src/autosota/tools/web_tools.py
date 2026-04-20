"""Web search and GitHub crawling utilities.

Provides async methods for searching GitHub repositories, fetching URLs,
and crawling README content from GitHub repos.
"""

from __future__ import annotations

import logging
import os
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)


class WebError(Exception):
    """Raised when a web operation fails."""


class WebTools:
    """Async web utilities for search and content retrieval.

    Parameters
    ----------
    httpx_client:
        Optional pre-configured :class:`httpx.AsyncClient`. If ``None``,
        a default client is created.
    github_token:
        GitHub personal access token for API calls. Falls back to
        the ``GITHUB_TOKEN`` environment variable.
    """

    def __init__(
        self,
        httpx_client: httpx.AsyncClient | None = None,
        github_token: str | None = None,
    ) -> None:
        self._client = httpx_client
        self._owns_client = httpx_client is None
        self.github_token = github_token or os.environ.get("GITHUB_TOKEN", "")

    async def _get_client(self) -> httpx.AsyncClient:
        """Return the HTTP client, creating one if needed."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=60.0,
                follow_redirects=True,
                headers={"User-Agent": "AutoSOTA/0.1"},
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client if we own it."""
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------
    # GitHub search
    # ------------------------------------------------------------------

    async def search_github_repos(self, query: str, max_results: int = 10) -> list[dict]:
        """Search GitHub for repositories matching *query*.

        Returns a list of dicts with keys: ``full_name``, ``html_url``,
        ``description``, ``stars``, ``language``, ``updated_at``.
        """
        client = await self._get_client()
        headers = self._github_headers()

        try:
            resp = await client.get(
                "https://api.github.com/search/repositories",
                params={
                    "q": query,
                    "sort": "stars",
                    "order": "desc",
                    "per_page": min(max_results, 100),
                },
                headers=headers,
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise WebError(f"GitHub search failed: {exc}") from exc

        data = resp.json()
        results: list[dict] = []
        for item in data.get("items", [])[:max_results]:
            results.append(
                {
                    "full_name": item.get("full_name", ""),
                    "html_url": item.get("html_url", ""),
                    "description": item.get("description", ""),
                    "stars": item.get("stargazers_count", 0),
                    "language": item.get("language", ""),
                    "updated_at": item.get("updated_at", ""),
                }
            )
        return results

    # ------------------------------------------------------------------
    # URL fetching
    # ------------------------------------------------------------------

    async def fetch_url(self, url: str) -> str:
        """Fetch the content of a URL and return the response text.

        Raises :class:`WebError` on failure.
        """
        client = await self._get_client()
        try:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.text
        except httpx.HTTPError as exc:
            raise WebError(f"Failed to fetch {url}: {exc}") from exc

    # ------------------------------------------------------------------
    # GitHub README crawling
    # ------------------------------------------------------------------

    async def crawl_github_readme(self, repo_url: str) -> str:
        """Fetch the README content from a GitHub repository.

        Accepts URLs in the form ``https://github.com/owner/repo``.
        Tries the GitHub API first, falling back to raw file fetching.
        """
        owner, repo_name = self._parse_github_url(repo_url)
        client = await self._get_client()
        headers = self._github_headers()
        headers["Accept"] = "application/vnd.github.v3.raw"

        # Try the API endpoint for the README.
        readme_url = f"https://api.github.com/repos/{owner}/{repo_name}/readme"
        try:
            resp = await client.get(readme_url, headers=headers)
            if resp.status_code == 200:
                return resp.text
        except httpx.HTTPError:
            pass

        # Fallback: try common raw README file names.
        for name in ("README.md", "README.rst", "README.txt", "README"):
            raw_url = f"https://raw.githubusercontent.com/{owner}/{repo_name}/main/{name}"
            try:
                resp = await client.get(raw_url)
                if resp.status_code == 200:
                    return resp.text
            except httpx.HTTPError:
                continue

            # Also try 'master' branch.
            raw_url_master = f"https://raw.githubusercontent.com/{owner}/{repo_name}/master/{name}"
            try:
                resp = await client.get(raw_url_master)
                if resp.status_code == 200:
                    return resp.text
            except httpx.HTTPError:
                continue

        raise WebError(f"Could not find README for {repo_url}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _github_headers(self) -> dict[str, str]:
        """Build headers for GitHub API requests."""
        headers: dict[str, str] = {"Accept": "application/vnd.github.v3+json"}
        if self.github_token:
            headers["Authorization"] = f"token {self.github_token}"
        return headers

    @staticmethod
    def _parse_github_url(url: str) -> tuple[str, str]:
        """Extract ``(owner, repo)`` from a GitHub URL.

        Raises :class:`WebError` if the URL cannot be parsed.
        """
        # Handle "owner/repo" shorthand.
        if not url.startswith("http"):
            parts = url.strip("/").split("/")
            if len(parts) >= 2:
                return parts[0], parts[1]
            raise WebError(f"Cannot parse GitHub URL: {url}")

        parsed = urlparse(url)
        path_parts = [p for p in parsed.path.strip("/").split("/") if p]
        if len(path_parts) < 2:
            raise WebError(f"Cannot parse GitHub URL: {url}")
        owner = path_parts[0]
        repo_name = path_parts[1].removesuffix(".git")
        return owner, repo_name
