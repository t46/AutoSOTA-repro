"""Unit tests for autosota.tools.git_tools."""

from __future__ import annotations

from pathlib import Path

import pytest

from autosota.tools.git_tools import GitError, GitManager


@pytest.fixture
def git_repo(tmp_path):
    """Create and return a GitManager with an initialised repo."""
    mgr = GitManager()
    mgr.init(tmp_path / "repo")
    # Create an initial file and commit.
    (mgr.repo_path / "README.md").write_text("# Test\n")
    mgr.commit("Initial commit")
    return mgr


class TestInit:
    def test_init_creates_repo(self, tmp_path):
        mgr = GitManager()
        mgr.init(tmp_path / "new_repo")
        assert (tmp_path / "new_repo" / ".git").is_dir()

    def test_repo_property_raises_when_none(self):
        mgr = GitManager()
        with pytest.raises(GitError, match="No repository"):
            _ = mgr.repo


class TestCommit:
    def test_commit_returns_hash(self, git_repo):
        (git_repo.repo_path / "file.txt").write_text("test")
        sha = git_repo.commit("Add file.txt")
        assert len(sha) == 40
        assert sha == git_repo.get_current_hash()

    def test_commit_no_changes(self, git_repo):
        # No changes; should return current hash without error.
        sha = git_repo.commit("Nothing to commit")
        assert len(sha) == 40


class TestTag:
    def test_create_tag(self, git_repo):
        git_repo.tag("v1.0")
        tags = [t.name for t in git_repo.repo.tags]
        assert "v1.0" in tags


class TestCheckout:
    def test_checkout_branch(self, git_repo):
        # Create a new branch and check it out.
        git_repo.repo.create_head("feature")
        git_repo.checkout("feature")
        assert git_repo.repo.active_branch.name == "feature"


class TestGetDiff:
    def test_diff_uncommitted(self, git_repo):
        (git_repo.repo_path / "README.md").write_text("# Updated\n")
        diff = git_repo.get_diff()
        assert "Updated" in diff

    def test_diff_between_refs(self, git_repo):
        hash1 = git_repo.get_current_hash()
        (git_repo.repo_path / "new.txt").write_text("content")
        hash2 = git_repo.commit("Add new.txt")
        diff = git_repo.get_diff(from_ref=hash1, to_ref=hash2)
        assert "content" in diff


class TestSnapshot:
    def test_snapshot_returns_hash(self, git_repo):
        (git_repo.repo_path / "snap.txt").write_text("snapshot data")
        sha = git_repo.snapshot("before risky change")
        assert len(sha) == 40


class TestLog:
    def test_log_entries(self, git_repo):
        (git_repo.repo_path / "a.txt").write_text("a")
        git_repo.commit("Add a")
        entries = git_repo.log(max_count=5)
        assert len(entries) >= 2
        assert entries[0]["message"] == "Add a"
        assert "hash" in entries[0]
        assert "date" in entries[0]


class TestIsDirty:
    def test_clean_repo(self, git_repo):
        assert git_repo.is_dirty() is False

    def test_dirty_repo(self, git_repo):
        (git_repo.repo_path / "dirty.txt").write_text("!")
        assert git_repo.is_dirty() is True


class TestClone:
    def test_clone_local_repo(self, git_repo, tmp_path):
        dest = tmp_path / "cloned"
        mgr = GitManager()
        cloned = mgr.clone(url=str(git_repo.repo_path), dest=dest, shallow=False)
        assert (dest / "README.md").is_file()
        assert len(cloned.get_current_hash()) == 40


class TestConstructorWithPath:
    def test_open_existing_repo(self, git_repo):
        mgr2 = GitManager(repo_path=git_repo.repo_path)
        assert mgr2.get_current_hash() == git_repo.get_current_hash()

    def test_open_invalid_path(self, tmp_path):
        with pytest.raises(GitError, match="Not a valid git repository"):
            GitManager(repo_path=tmp_path)
