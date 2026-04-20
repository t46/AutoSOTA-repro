"""AgentResource -- paper ingestion and repository discovery.

Takes a paper PDF or arxiv URL, finds the associated GitHub repo,
downloads datasets and models, and produces a ``PaperCodeUnit``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from autosota.agents.base import AgentResult, BaseAgent
from autosota.models.paper import PaperCodeUnit, PaperMetadata


# ------------------------------------------------------------------
# Structured LLM response models
# ------------------------------------------------------------------


class PaperClassification(BaseModel):
    """LLM output: whether the paper is methodological or theoretical."""

    classification: str  # "methodological" | "theoretical" | "empirical"
    reasoning: str
    has_code: bool


class RepoCandidate(BaseModel):
    """A candidate GitHub repository linked to the paper."""

    url: str
    confidence: float  # 0-1
    reasoning: str


class ReadinessAssessment(BaseModel):
    """LLM assessment of reproduction readiness."""

    readiness_score: float  # 0-1
    has_requirements: bool
    has_eval_script: bool
    has_dataset_instructions: bool
    issues: list[str]


# ------------------------------------------------------------------
# Agent
# ------------------------------------------------------------------


class AgentResource(BaseAgent):
    """Paper ingestion and repository acquisition agent.

    For MVP this supports a *manual* mode where the user provides
    ``repo_url`` directly in the task config, skipping paper parsing
    and GitHub search.
    """

    async def run(self, *, task: dict[str, Any]) -> AgentResult:
        """Produce a ``PaperCodeUnit`` from a task configuration.

        Parameters
        ----------
        task:
            Dictionary containing at least ``paper_url`` or ``repo_url``.
            Optional keys: ``pdf_path``, ``target_metric``, ``eval_command``.
        """
        llm_calls = 0

        try:
            # ----- Step 1: extract / build metadata -------------------
            pdf_path = task.get("pdf_path")
            if pdf_path:
                metadata, calls = await self._extract_metadata(Path(pdf_path))
                llm_calls += calls
            else:
                # Minimal metadata from what the user provided
                metadata = PaperMetadata(
                    title=task.get("title", "Unknown"),
                    arxiv_id=task.get("arxiv_id"),
                    repo_url=task.get("repo_url"),
                    target_metric=task.get("target_metric", ""),
                )

            # ----- Step 2: classify paper -----------------------------
            classification, calls = await self._classify_paper(metadata)
            llm_calls += calls
            self.logger.info(
                "paper_classified",
                classification=classification.classification,
                has_code=classification.has_code,
            )

            # ----- Step 3: locate repository --------------------------
            repo_url = task.get("repo_url") or metadata.repo_url
            if not repo_url:
                repo_url, calls = await self._locate_repository(metadata)
                llm_calls += calls

            if not repo_url:
                return AgentResult(
                    success=False,
                    error="Could not locate a repository for this paper.",
                    llm_calls_used=llm_calls,
                )

            # ----- Step 4: clone / reference local path ---------------
            repo_local = self._resolve_local_repo(repo_url, task)

            # ----- Step 5: assess readiness ---------------------------
            readiness, calls = await self._assess_readiness(metadata, repo_local)
            llm_calls += calls
            self.logger.info("readiness_assessed", score=readiness.readiness_score)

            # ----- Step 6: discover & download resources (simplified) --
            resources, calls = await self._discover_resources(repo_local)
            llm_calls += calls

            # ----- Build PaperCodeUnit --------------------------------
            unit = PaperCodeUnit(
                paper=metadata,
                repo_url=repo_url,
                repo_local_path=repo_local,
                eval_command=task.get("eval_command", ""),
                entry_script=task.get("entry_script", ""),
                dataset_path=task.get("dataset_path", ""),
            )

            return AgentResult(
                success=True,
                data={
                    "paper_code_unit": unit.model_dump(mode="json"),
                    "classification": classification.model_dump(),
                    "readiness": readiness.model_dump(),
                    "resources": resources,
                },
                llm_calls_used=llm_calls,
            )

        except Exception as exc:
            self.logger.error("resource_agent_failed", error=str(exc))
            return AgentResult(
                success=False,
                error=str(exc),
                llm_calls_used=llm_calls,
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _extract_metadata(self, pdf_path: Path) -> tuple[PaperMetadata, int]:
        """Parse a PDF for title, abstract, links, etc.

        Uses ``pymupdf4llm`` to convert PDF to markdown, then asks the LLM
        to extract structured metadata.
        """
        # TODO: full implementation with pymupdf4llm
        # For MVP, build minimal metadata and return
        self.logger.info("extracting_metadata", pdf_path=str(pdf_path))

        messages = [
            {
                "role": "user",
                "content": (
                    f"Extract metadata from this paper located at {pdf_path}. "
                    "Return the title, authors, abstract, arxiv_id if visible, "
                    "and any GitHub repository URLs mentioned."
                ),
            }
        ]
        response = await self._call_llm(messages)
        # For MVP, parse LLM response heuristically
        metadata = PaperMetadata(
            title=str(pdf_path.stem).replace("_", " "),
            pdf_path=pdf_path,
        )
        return metadata, 1

    async def _classify_paper(
        self, metadata: PaperMetadata
    ) -> tuple[PaperClassification, int]:
        """LLM call to classify the paper type."""
        messages = [
            {
                "role": "user",
                "content": (
                    f"Classify this research paper:\n"
                    f"Title: {metadata.title}\n"
                    f"Abstract: {metadata.abstract}\n\n"
                    "Is it methodological (proposes a new method with code), "
                    "theoretical (proofs, no code), or empirical (experiments "
                    "on existing methods)? Does it likely have associated code?"
                ),
            }
        ]
        result = await self._call_llm(messages, response_model=PaperClassification)
        return result, 1

    async def _locate_repository(
        self, metadata: PaperMetadata
    ) -> tuple[str | None, int]:
        """Find the GitHub repo for a paper.

        Strategy:
        1. Extract URLs from the PDF text.
        2. Search GitHub by paper title.
        3. LLM selects the best match.
        """
        # Heuristic: check if arxiv_id links to paperswithcode
        messages = [
            {
                "role": "user",
                "content": (
                    f"Find the most likely GitHub repository for this paper:\n"
                    f"Title: {metadata.title}\n"
                    f"ArXiv ID: {metadata.arxiv_id}\n"
                    f"Known repo URL: {metadata.repo_url}\n\n"
                    "Return the best repository URL, or 'NOT_FOUND' if you "
                    "cannot determine one."
                ),
            }
        ]
        response = await self._call_llm(messages)
        content = response if isinstance(response, str) else str(response)

        # Try to extract a GitHub URL from the response
        match = re.search(r"https?://github\.com/[\w\-]+/[\w\-]+", content)
        url = match.group(0) if match else None
        return url, 1

    def _resolve_local_repo(self, repo_url: str, task: dict[str, Any]) -> Path | None:
        """Resolve or clone the repository locally.

        For MVP: if a local path is provided in the task, use that.
        Otherwise return a workspace sub-path (actual clone is deferred
        to AgentInit which has Docker context).
        """
        local = task.get("repo_local_path")
        if local:
            return Path(local)
        # Derive a local path under workspace
        repo_name = repo_url.rstrip("/").split("/")[-1]
        return self.workspace_dir / "repos" / repo_name

    async def _assess_readiness(
        self, metadata: PaperMetadata, repo_path: Path | None
    ) -> tuple[ReadinessAssessment, int]:
        """LLM judges how ready the repository is for reproduction."""
        repo_info = ""
        if repo_path and repo_path.exists():
            # Summarise repo contents
            files = list(repo_path.iterdir()) if repo_path.is_dir() else []
            file_names = [f.name for f in files[:50]]
            repo_info = f"Repository files: {file_names}"

        messages = [
            {
                "role": "user",
                "content": (
                    f"Assess the reproduction readiness of this repository:\n"
                    f"Paper: {metadata.title}\n"
                    f"{repo_info}\n\n"
                    "Score from 0 (impossible) to 1 (turnkey reproducible)."
                ),
            }
        ]
        result = await self._call_llm(messages, response_model=ReadinessAssessment)
        return result, 1

    async def _discover_resources(
        self, repo_path: Path | None
    ) -> tuple[list[dict[str, str]], int]:
        """Scan the repository for dataset / model download URLs.

        Returns a list of dicts with keys ``type``, ``url``, ``description``.
        """
        if not repo_path or not repo_path.exists():
            return [], 0

        # Look for common resource patterns in README
        readme_candidates = ["README.md", "readme.md", "README.rst", "README"]
        readme_text = ""
        for name in readme_candidates:
            p = repo_path / name
            if p.exists():
                readme_text = p.read_text(errors="replace")[:5000]
                break

        if not readme_text:
            return [], 0

        messages = [
            {
                "role": "user",
                "content": (
                    "Extract download URLs for datasets, pretrained models, "
                    "and other resources from this README:\n\n"
                    f"{readme_text}\n\n"
                    "Return a JSON list of objects with keys: type, url, description. "
                    "If none found, return an empty list []."
                ),
            }
        ]
        response = await self._call_llm(messages)
        # TODO: parse JSON from response for full implementation
        return [], 1

    async def _download_resources(
        self, resources: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """Download resources to workspace. Placeholder for MVP."""
        # TODO: implement actual download via httpx
        self.logger.info("download_resources_skipped", count=len(resources))
        return resources
