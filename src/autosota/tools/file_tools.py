"""Async file operations for reading, writing, searching, and listing files."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import aiofiles

logger = logging.getLogger(__name__)


class FileError(Exception):
    """Raised when a file operation fails."""


class FileTools:
    """Collection of async static methods for common file system operations."""

    @staticmethod
    async def read_file(path: Path) -> str:
        """Read and return the full text content of a file.

        Raises :class:`FileError` if the file does not exist or cannot be read.
        """
        path = Path(path)
        if not path.is_file():
            raise FileError(f"File not found: {path}")
        try:
            async with aiofiles.open(path, mode="r", encoding="utf-8") as f:
                return await f.read()
        except Exception as exc:
            raise FileError(f"Failed to read {path}: {exc}") from exc

    @staticmethod
    async def write_file(path: Path, content: str) -> None:
        """Write *content* to *path*, creating parent directories as needed.

        Overwrites the file if it already exists.
        """
        path = Path(path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(path, mode="w", encoding="utf-8") as f:
                await f.write(content)
        except Exception as exc:
            raise FileError(f"Failed to write {path}: {exc}") from exc

    @staticmethod
    async def search_in_files(
        directory: Path,
        pattern: str,
        glob: str = "**/*",
    ) -> list[tuple[Path, int, str]]:
        """Search for *pattern* in files matching *glob* under *directory*.

        Parameters
        ----------
        directory:
            Root directory to search.
        pattern:
            Regular expression pattern to match against each line.
        glob:
            Glob pattern for selecting files (default: all files recursively).

        Returns
        -------
        list[tuple[Path, int, str]]
            A list of ``(file_path, line_number, line_text)`` for every match.
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise FileError(f"Directory not found: {directory}")

        compiled = re.compile(pattern)
        results: list[tuple[Path, int, str]] = []

        for filepath in sorted(directory.glob(glob)):
            if not filepath.is_file():
                continue
            try:
                async with aiofiles.open(filepath, mode="r", encoding="utf-8", errors="replace") as f:
                    line_num = 0
                    async for line in f:
                        line_num += 1
                        if compiled.search(line):
                            results.append((filepath, line_num, line.rstrip("\n")))
            except Exception:
                # Skip binary / unreadable files silently.
                continue

        return results

    @staticmethod
    async def list_directory(path: Path, recursive: bool = False) -> list[Path]:
        """List files (and directories) under *path*.

        Parameters
        ----------
        path:
            Directory to list.
        recursive:
            If ``True``, list all entries recursively; otherwise list immediate children.

        Returns
        -------
        list[Path]
            Sorted list of paths.
        """
        path = Path(path)
        if not path.is_dir():
            raise FileError(f"Directory not found: {path}")

        if recursive:
            entries = sorted(p for p in path.rglob("*"))
        else:
            entries = sorted(p for p in path.iterdir())

        return entries

    @staticmethod
    async def append_file(path: Path, content: str) -> None:
        """Append *content* to *path*, creating the file if it does not exist."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            async with aiofiles.open(path, mode="a", encoding="utf-8") as f:
                await f.write(content)
        except Exception as exc:
            raise FileError(f"Failed to append to {path}: {exc}") from exc

    @staticmethod
    async def file_exists(path: Path) -> bool:
        """Return ``True`` if *path* exists and is a regular file."""
        return Path(path).is_file()
