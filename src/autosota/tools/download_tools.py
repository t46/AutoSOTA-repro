"""Resource downloading utilities.

Provides async methods for downloading files from URLs and Hugging Face Hub.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)


class DownloadError(Exception):
    """Raised when a download operation fails."""


class DownloadTools:
    """Static methods for downloading files and model artifacts."""

    @staticmethod
    async def download_url(
        url: str,
        dest: Path,
        timeout: int = 300,
        chunk_size: int = 8192,
    ) -> Path:
        """Download a file from *url* to *dest*.

        Parameters
        ----------
        url:
            The URL to download from.
        dest:
            Local destination file path.
        timeout:
            Request timeout in seconds.
        chunk_size:
            Streaming chunk size in bytes.

        Returns
        -------
        Path
            The destination path (same as *dest*).
        """
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Downloading %s -> %s", url, dest)

        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(timeout, connect=30.0),
                follow_redirects=True,
            ) as client:
                async with client.stream("GET", url) as response:
                    response.raise_for_status()
                    with open(dest, "wb") as f:
                        async for chunk in response.aiter_bytes(chunk_size):
                            f.write(chunk)
        except httpx.HTTPError as exc:
            # Clean up partial file on failure.
            dest.unlink(missing_ok=True)
            raise DownloadError(f"Failed to download {url}: {exc}") from exc

        logger.info("Downloaded %s (%.2f MB)", dest.name, dest.stat().st_size / 1_048_576)
        return dest

    @staticmethod
    async def download_huggingface(
        repo_id: str,
        dest: Path,
        filename: str | None = None,
        revision: str | None = None,
        token: str | None = None,
    ) -> Path:
        """Download a file or snapshot from the Hugging Face Hub.

        Parameters
        ----------
        repo_id:
            Repository identifier (e.g. ``"facebook/opt-1.3b"``).
        dest:
            Local destination directory.
        filename:
            If provided, download only this single file from the repo.
            Otherwise, download the entire repository snapshot.
        revision:
            Branch or commit hash to download from.
        token:
            Hugging Face API token. Falls back to ``HF_TOKEN`` env var
            (handled automatically by ``huggingface_hub``).

        Returns
        -------
        Path
            Path to the downloaded file or snapshot directory.
        """
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)

        def _download() -> Path:
            from huggingface_hub import hf_hub_download, snapshot_download

            kwargs: dict = {"repo_id": repo_id}
            if revision:
                kwargs["revision"] = revision
            if token:
                kwargs["token"] = token

            if filename:
                logger.info(
                    "Downloading %s/%s from Hugging Face", repo_id, filename
                )
                result = hf_hub_download(
                    **kwargs,
                    filename=filename,
                    local_dir=str(dest),
                )
                return Path(result)
            else:
                logger.info(
                    "Downloading snapshot of %s from Hugging Face", repo_id
                )
                result = snapshot_download(
                    **kwargs,
                    local_dir=str(dest),
                )
                return Path(result)

        try:
            return await asyncio.to_thread(_download)
        except Exception as exc:
            raise DownloadError(
                f"Failed to download from Hugging Face ({repo_id}): {exc}"
            ) from exc
