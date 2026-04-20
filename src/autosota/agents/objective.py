"""AgentObjective -- evaluation rubric construction.

Reads the paper (PDF markdown) and repository to build a ``RubricTree``
that defines the primary metric, baseline value, and evaluation criteria.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel

from autosota.agents.base import AgentResult, BaseAgent
from autosota.constants import MetricDirection
from autosota.models.rubric import RubricNode, RubricTree


# ------------------------------------------------------------------
# Structured LLM response models
# ------------------------------------------------------------------


class TargetMetricInfo(BaseModel):
    """LLM-extracted primary evaluation metric."""

    metric_name: str
    baseline_value: float | None = None
    direction: str = "higher_is_better"  # "higher_is_better" | "lower_is_better"
    unit: str = ""
    reasoning: str = ""


class RubricCriterion(BaseModel):
    """A single evaluation criterion identified by the LLM."""

    name: str
    description: str
    weight: float = 1.0
    is_hard_constraint: bool = False


# ------------------------------------------------------------------
# Agent
# ------------------------------------------------------------------


class AgentObjective(BaseAgent):
    """Builds an evaluation rubric from paper content and repository.

    For MVP: extracts the primary metric and builds a flat rubric
    (one root node with leaf children).  Full version would construct
    a hierarchical BFS tree.
    """

    async def run(
        self,
        *,
        paper_unit: dict[str, Any],
        pdf_markdown: str = "",
    ) -> AgentResult:
        """Build a ``RubricTree`` for the given paper.

        Parameters
        ----------
        paper_unit:
            Serialised ``PaperCodeUnit`` dictionary.
        pdf_markdown:
            Full paper text converted to markdown (from pymupdf4llm).
        """
        llm_calls = 0

        try:
            repo_path = paper_unit.get("repo_local_path")
            title = paper_unit.get("paper", {}).get("title", "Unknown")

            # ----- Step 1: extract target metric ----------------------
            metric_info, calls = await self._extract_target_metric(
                pdf_markdown, repo_path, title
            )
            llm_calls += calls
            self.logger.info(
                "target_metric_extracted",
                metric=metric_info.metric_name,
                baseline=metric_info.baseline_value,
            )

            # ----- Step 2: build rubric tree --------------------------
            direction = (
                MetricDirection.LOWER_IS_BETTER
                if metric_info.direction == "lower_is_better"
                else MetricDirection.HIGHER_IS_BETTER
            )

            rubric, calls = await self._build_rubric_tree(
                metric_info, pdf_markdown, title
            )
            llm_calls += calls

            rubric.target_metric = metric_info.metric_name
            rubric.metric_direction = direction

            return AgentResult(
                success=True,
                data={
                    "rubric": rubric.model_dump(),
                    "target_metric": metric_info.model_dump(),
                },
                llm_calls_used=llm_calls,
            )

        except Exception as exc:
            self.logger.error("objective_agent_failed", error=str(exc))
            return AgentResult(
                success=False,
                error=str(exc),
                llm_calls_used=llm_calls,
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _extract_target_metric(
        self,
        pdf_markdown: str,
        repo_path: str | None,
        title: str,
    ) -> tuple[TargetMetricInfo, int]:
        """Ask the LLM to identify the primary evaluation metric."""
        context_parts = [f"Paper title: {title}"]
        if pdf_markdown:
            # Truncate to fit context window
            context_parts.append(f"Paper text (truncated):\n{pdf_markdown[:8000]}")

        if repo_path:
            # Try reading eval scripts for metric clues
            rp = Path(repo_path)
            for candidate in ["eval.py", "evaluate.py", "test.py", "run_eval.sh"]:
                p = rp / candidate
                if p.exists():
                    content = p.read_text(errors="replace")[:3000]
                    context_parts.append(f"Eval script ({candidate}):\n{content}")
                    break

        messages = [
            {
                "role": "user",
                "content": (
                    "Identify the PRIMARY evaluation metric for this paper.\n\n"
                    + "\n\n".join(context_parts)
                    + "\n\nReturn:\n"
                    "- metric_name: the exact metric name (e.g. 'PSNR', 'mAP', 'BLEU')\n"
                    "- baseline_value: the reported baseline value if available\n"
                    "- direction: 'higher_is_better' or 'lower_is_better'\n"
                    "- unit: measurement unit if applicable\n"
                    "- reasoning: brief explanation of your choice"
                ),
            }
        ]
        result = await self._call_llm(messages, response_model=TargetMetricInfo)
        return result, 1

    async def _build_rubric_tree(
        self,
        metric_info: TargetMetricInfo,
        pdf_markdown: str,
        title: str,
    ) -> tuple[RubricTree, int]:
        """Construct the evaluation rubric.

        For MVP: flat list of criteria under a single root node.
        Full version: hierarchical BFS tree with sub-goals.
        """
        messages = [
            {
                "role": "user",
                "content": (
                    f"Build evaluation criteria for reproducing this paper:\n"
                    f"Title: {title}\n"
                    f"Primary metric: {metric_info.metric_name}\n"
                    f"Baseline: {metric_info.baseline_value}\n\n"
                    "List 3-7 concrete, measurable criteria that together "
                    "determine whether reproduction is successful. Include:\n"
                    "1. Primary metric matches or exceeds baseline\n"
                    "2. No integrity violations\n"
                    "3. Reproducibility (deterministic results)\n"
                    "Return as a list of criteria with name, description, weight, "
                    "and whether it's a hard constraint."
                ),
            }
        ]
        # TODO: for full version, parse structured list from LLM
        # For MVP, build a reasonable default rubric
        response = await self._call_llm(messages)

        # Construct rubric tree with sensible defaults
        children = [
            RubricNode(
                name="primary_metric",
                description=(
                    f"{metric_info.metric_name} meets or exceeds baseline "
                    f"({metric_info.baseline_value})"
                ),
                weight=3.0,
            ),
            RubricNode(
                name="reproducibility",
                description="Results are reproducible across runs within tolerance",
                weight=2.0,
            ),
            RubricNode(
                name="integrity",
                description="No red-line violations (R1-R7)",
                weight=2.0,
            ),
            RubricNode(
                name="code_quality",
                description="Changes are clean, minimal, and well-documented",
                weight=1.0,
            ),
        ]

        root = RubricNode(
            name="reproduction_quality",
            description=f"Overall reproduction quality for: {title}",
            children=children,
        )

        rubric = RubricTree(
            root=root,
            target_metric=metric_info.metric_name,
            notes=f"Auto-generated rubric. LLM reasoning: {metric_info.reasoning}",
        )
        return rubric, 1
