"""Unit tests for autosota.tools.file_tools."""

from __future__ import annotations

from pathlib import Path

import pytest

from autosota.tools.file_tools import FileError, FileTools


@pytest.fixture
def sample_dir(tmp_path):
    """Create a directory with some sample files."""
    (tmp_path / "hello.txt").write_text("hello world\n")
    (tmp_path / "data.csv").write_text("a,b\n1,2\n3,4\n")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "nested.py").write_text("def foo():\n    return 42\n")
    return tmp_path


class TestReadFile:
    async def test_reads_existing_file(self, sample_dir):
        text = await FileTools.read_file(sample_dir / "hello.txt")
        assert text == "hello world\n"

    async def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileError, match="File not found"):
            await FileTools.read_file(tmp_path / "nope.txt")


class TestWriteFile:
    async def test_writes_new_file(self, tmp_path):
        path = tmp_path / "new.txt"
        await FileTools.write_file(path, "content")
        assert path.read_text() == "content"

    async def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "a" / "b" / "c.txt"
        await FileTools.write_file(path, "deep")
        assert path.read_text() == "deep"

    async def test_overwrites_existing(self, sample_dir):
        path = sample_dir / "hello.txt"
        await FileTools.write_file(path, "goodbye")
        assert path.read_text() == "goodbye"


class TestSearchInFiles:
    async def test_finds_pattern(self, sample_dir):
        results = await FileTools.search_in_files(sample_dir, r"hello")
        assert len(results) == 1
        assert results[0][2] == "hello world"

    async def test_finds_in_nested_files(self, sample_dir):
        results = await FileTools.search_in_files(sample_dir, r"def foo")
        assert len(results) == 1
        filepath, line_num, line = results[0]
        assert filepath.name == "nested.py"
        assert line_num == 1

    async def test_glob_filter(self, sample_dir):
        results = await FileTools.search_in_files(
            sample_dir, r"\d", glob="**/*.csv"
        )
        # Should find lines with digits only in .csv files.
        assert all(r[0].suffix == ".csv" for r in results)

    async def test_empty_results(self, sample_dir):
        results = await FileTools.search_in_files(sample_dir, r"ZZZZZ")
        assert results == []

    async def test_raises_on_missing_dir(self, tmp_path):
        with pytest.raises(FileError, match="Directory not found"):
            await FileTools.search_in_files(tmp_path / "nope", r"x")


class TestListDirectory:
    async def test_non_recursive(self, sample_dir):
        entries = await FileTools.list_directory(sample_dir, recursive=False)
        names = [e.name for e in entries]
        assert "hello.txt" in names
        assert "sub" in names

    async def test_recursive(self, sample_dir):
        entries = await FileTools.list_directory(sample_dir, recursive=True)
        names = [e.name for e in entries]
        assert "nested.py" in names

    async def test_raises_on_missing_dir(self, tmp_path):
        with pytest.raises(FileError, match="Directory not found"):
            await FileTools.list_directory(tmp_path / "nope")


class TestAppendFile:
    async def test_appends_to_existing(self, sample_dir):
        path = sample_dir / "hello.txt"
        await FileTools.append_file(path, "extra\n")
        text = path.read_text()
        assert text == "hello world\nextra\n"

    async def test_creates_new_file(self, tmp_path):
        path = tmp_path / "new.txt"
        await FileTools.append_file(path, "first")
        assert path.read_text() == "first"


class TestFileExists:
    async def test_existing_file(self, sample_dir):
        assert await FileTools.file_exists(sample_dir / "hello.txt") is True

    async def test_missing_file(self, tmp_path):
        assert await FileTools.file_exists(tmp_path / "nope") is False
