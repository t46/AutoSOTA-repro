"""Git operations for version control within the optimisation loop.

Uses GitPython for repository management, supporting clone, commit,
diff, tag, and snapshot workflows.
"""

from __future__ import annotations

import logging
from pathlib import Path

from git import InvalidGitRepositoryError, Repo

logger = logging.getLogger(__name__)


class GitError(Exception):
    """Raised when a git operation fails."""


class GitManager:
    """High-level git helper wrapping :class:`git.Repo`.

    Parameters
    ----------
    repo_path:
        Path to an existing git repository. Set to ``None`` if the repo
        will be created later via :meth:`clone` or :meth:`init`.
    """

    def __init__(self, repo_path: Path | None = None) -> None:
        self._repo: Repo | None = None
        if repo_path is not None:
            self._repo = self._open(repo_path)

    @property
    def repo(self) -> Repo:
        """Return the underlying :class:`git.Repo`, raising if unset."""
        if self._repo is None:
            raise GitError("No repository is loaded. Call clone() or init() first.")
        return self._repo

    @property
    def repo_path(self) -> Path:
        """Return the working directory path."""
        return Path(self.repo.working_dir)

    # ------------------------------------------------------------------
    # Repository setup
    # ------------------------------------------------------------------

    def clone(self, url: str, dest: Path, shallow: bool = True) -> "GitManager":
        """Clone a remote repository and return a new :class:`GitManager`.

        Parameters
        ----------
        url:
            Remote repository URL.
        dest:
            Local destination directory.
        shallow:
            If ``True``, perform a ``--depth 1`` clone.

        Returns
        -------
        GitManager
            A new manager instance pointing at the cloned repository.
        """
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        kwargs: dict = {}
        if shallow:
            kwargs["depth"] = 1
        logger.info("Cloning %s -> %s (shallow=%s)", url, dest, shallow)
        repo = Repo.clone_from(url, str(dest), **kwargs)
        mgr = GitManager.__new__(GitManager)
        mgr._repo = repo
        return mgr

    def init(self, path: Path) -> "GitManager":
        """Initialise a new git repository at *path*.

        Returns
        -------
        GitManager
            ``self`` with the newly-initialised repository loaded.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._repo = Repo.init(str(path))
        return self

    # ------------------------------------------------------------------
    # Basic operations
    # ------------------------------------------------------------------

    def commit(self, message: str, add_all: bool = True) -> str:
        """Create a commit and return its hash.

        Parameters
        ----------
        message:
            Commit message.
        add_all:
            If ``True``, stage all changes (tracked + untracked) before committing.
        """
        if add_all:
            self.repo.git.add(A=True)

        # Ensure there is something to commit.
        if not self.repo.is_dirty(untracked_files=True) and not self.repo.index.diff("HEAD"):
            logger.info("Nothing to commit.")
            return self.get_current_hash()

        self.repo.index.commit(message)
        new_hash = self.get_current_hash()
        logger.info("Committed %s: %s", new_hash[:8], message)
        return new_hash

    def tag(self, tag_name: str) -> None:
        """Create a lightweight tag at HEAD."""
        self.repo.create_tag(tag_name)
        logger.info("Tagged HEAD as %s", tag_name)

    def checkout(self, ref: str) -> None:
        """Checkout a branch, tag, or commit hash."""
        self.repo.git.checkout(ref)
        logger.info("Checked out %s", ref)

    def get_diff(
        self, from_ref: str | None = None, to_ref: str | None = None
    ) -> str:
        """Return a unified diff string.

        * No arguments: diff of uncommitted changes against HEAD.
        * ``from_ref`` only: diff from that ref to the working tree.
        * Both ``from_ref`` and ``to_ref``: diff between the two refs.
        """
        if from_ref and to_ref:
            return self.repo.git.diff(from_ref, to_ref)
        if from_ref:
            return self.repo.git.diff(from_ref)
        # Default: show unstaged + staged changes against HEAD.
        return self.repo.git.diff("HEAD")

    def get_current_hash(self) -> str:
        """Return the full SHA-1 hash of HEAD."""
        return self.repo.head.commit.hexsha

    def snapshot(self, message: str = "pre-commit snapshot") -> str:
        """Commit the current working-tree state and return its hash.

        This is intended for creating rollback points before risky operations.
        """
        return self.commit(message=message, add_all=True)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def log(self, max_count: int = 10) -> list[dict]:
        """Return recent commit log entries as dicts.

        Each dict has keys: ``hash``, ``short_hash``, ``message``, ``author``, ``date``.
        """
        entries: list[dict] = []
        for commit in self.repo.iter_commits(max_count=max_count):
            entries.append(
                {
                    "hash": commit.hexsha,
                    "short_hash": commit.hexsha[:8],
                    "message": commit.message.strip(),
                    "author": str(commit.author),
                    "date": commit.committed_datetime.isoformat(),
                }
            )
        return entries

    def is_dirty(self) -> bool:
        """Return ``True`` if the working tree has uncommitted changes."""
        return self.repo.is_dirty(untracked_files=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _open(path: Path) -> Repo:
        """Open an existing repository, raising :class:`GitError` on failure."""
        try:
            return Repo(str(path))
        except InvalidGitRepositoryError:
            raise GitError(f"Not a valid git repository: {path}")
