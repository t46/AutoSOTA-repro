"""AgentIdeator -- hypothesis generation for optimization.

Generates, aligns, audits, and prioritises optimization hypotheses
across PARAM / CODE / ALGO categories.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from autosota.agents.base import AgentResult, BaseAgent
from autosota.constants import HypothesisType, RedLineVerdict, RiskLevel


# ------------------------------------------------------------------
# Structured models
# ------------------------------------------------------------------


class Hypothesis(BaseModel):
    """A single optimization hypothesis."""

    id: str = ""
    title: str
    description: str
    category: HypothesisType = HypothesisType.PARAM
    risk: RiskLevel = RiskLevel.LOW
    expected_gain: str = ""
    implementation_steps: list[str] = Field(default_factory=list)
    files_to_modify: list[str] = Field(default_factory=list)
    red_line_verdict: RedLineVerdict = RedLineVerdict.CLEARED
    red_line_notes: str = ""
    alignment_score: float = 0.0  # 0-1 composite alignment


class AlignmentScores(BaseModel):
    """Four-dimensional alignment assessment for a hypothesis."""

    metric_relevance: float = 0.0  # How directly does it target the metric?
    feasibility: float = 0.0  # How likely to implement correctly?
    safety: float = 0.0  # How unlikely to violate red lines?
    novelty: float = 0.0  # How different from previous attempts?
    composite: float = 0.0


class HypothesisBatch(BaseModel):
    """LLM output: a batch of generated hypotheses."""

    hypotheses: list[Hypothesis] = Field(default_factory=list)


# ------------------------------------------------------------------
# Agent
# ------------------------------------------------------------------


class AgentIdeator(BaseAgent):
    """Hypothesis generation and prioritisation agent.

    Workflow:
    1. Integrate prior knowledge from research report.
    2. Generate 10+ candidate hypotheses across PARAM/CODE/ALGO.
    3. Align each hypothesis on 4 dimensions.
    4. Red-line audit each hypothesis against R1-R7.
    5. Prioritise by composite alignment and feasibility.
    """

    async def run(
        self,
        *,
        code_analysis: str = "",
        research_report: str = "",
        score_history: list[dict[str, Any]] | None = None,
        hard_constraints: list[str] | None = None,
    ) -> AgentResult:
        """Generate and prioritise optimization hypotheses.

        Parameters
        ----------
        code_analysis:
            Content of ``code_analysis.md``.
        research_report:
            Content of ``research_report.md``.
        score_history:
            List of dicts with ``iteration``, ``score``, ``hypothesis`` keys.
        hard_constraints:
            Paper-specific hard constraints (for R7).
        """
        llm_calls = 0
        score_history = score_history or []
        hard_constraints = hard_constraints or []

        try:
            # ----- Step 1: integrate prior knowledge ------------------
            priors, calls = await self._integrate_prior_knowledge(research_report)
            llm_calls += calls

            # ----- Step 2: generate hypotheses ------------------------
            context = self._build_generation_context(
                code_analysis, priors, score_history
            )
            hypotheses, calls = await self._generate_hypotheses(context, priors)
            llm_calls += calls
            self.logger.info("hypotheses_generated", count=len(hypotheses))

            # ----- Step 3: align each hypothesis ----------------------
            for hyp in hypotheses:
                alignment, calls = await self._align_hypothesis(hyp, context)
                llm_calls += calls
                hyp.alignment_score = alignment.composite

            # ----- Step 4: red-line audit -----------------------------
            cleared = []
            for hyp in hypotheses:
                verdict, calls = await self._red_line_audit(hyp, hard_constraints)
                llm_calls += calls
                hyp.red_line_verdict = verdict
                if verdict == RedLineVerdict.CLEARED:
                    cleared.append(hyp)
                else:
                    self.logger.info(
                        "hypothesis_rejected",
                        title=hyp.title,
                        notes=hyp.red_line_notes,
                    )

            # ----- Step 5: prioritise ---------------------------------
            prioritised = self._prioritize(cleared)

            return AgentResult(
                success=True,
                data={
                    "hypotheses": [h.model_dump() for h in prioritised],
                    "total_generated": len(hypotheses),
                    "total_cleared": len(cleared),
                },
                llm_calls_used=llm_calls,
            )

        except Exception as exc:
            self.logger.error("ideator_agent_failed", error=str(exc))
            return AgentResult(
                success=False,
                error=str(exc),
                llm_calls_used=llm_calls,
            )

    # ------------------------------------------------------------------
    # Prior knowledge integration
    # ------------------------------------------------------------------

    async def _integrate_prior_knowledge(
        self, research_report: str
    ) -> tuple[str, int]:
        """Extract SOTA techniques from the research report."""
        if not research_report:
            return "", 0

        messages = [
            {
                "role": "user",
                "content": (
                    "Extract the key SOTA techniques and optimization tricks "
                    "from this research report that could be applied:\n\n"
                    f"{research_report[:6000]}\n\n"
                    "List concrete, actionable techniques with brief descriptions."
                ),
            }
        ]
        response = await self._call_llm(messages)
        content = response if isinstance(response, str) else str(response)
        return content, 1

    # ------------------------------------------------------------------
    # Hypothesis generation
    # ------------------------------------------------------------------

    def _build_generation_context(
        self,
        code_analysis: str,
        priors: str,
        score_history: list[dict[str, Any]],
    ) -> str:
        """Assemble context for hypothesis generation."""
        parts = []
        if code_analysis:
            parts.append(f"## Code Analysis\n{code_analysis[:4000]}")
        if priors:
            parts.append(f"## SOTA Techniques\n{priors[:2000]}")
        if score_history:
            recent = score_history[-5:]
            history_str = "\n".join(
                f"- Iter {h.get('iteration', '?')}: "
                f"score={h.get('score', '?')}, "
                f"hyp={h.get('hypothesis', '?')}"
                for h in recent
            )
            parts.append(f"## Recent Score History\n{history_str}")
        return "\n\n".join(parts)

    async def _generate_hypotheses(
        self, context: str, priors: str
    ) -> tuple[list[Hypothesis], int]:
        """LLM generates 10+ candidate hypotheses."""
        messages = [
            {
                "role": "user",
                "content": (
                    "Generate at least 10 optimization hypotheses for improving "
                    "the target metric. Cover all three categories:\n"
                    "- PARAM: hyperparameter changes (learning rate, batch size, etc.)\n"
                    "- CODE: implementation improvements (efficiency, bugs, etc.)\n"
                    "- ALGO: algorithmic changes (loss functions, architectures, etc.)\n\n"
                    f"{context}\n\n"
                    "For each hypothesis provide:\n"
                    "- title: short descriptive name\n"
                    "- description: what to change and why\n"
                    "- category: PARAM, CODE, or ALGO\n"
                    "- risk: low, medium, or high\n"
                    "- expected_gain: estimated improvement\n"
                    "- implementation_steps: concrete steps\n"
                    "- files_to_modify: which files need changes"
                ),
            }
        ]
        result = await self._call_llm(messages, response_model=HypothesisBatch)
        hypotheses = result.hypotheses if isinstance(result, HypothesisBatch) else []

        # Assign IDs
        for i, h in enumerate(hypotheses):
            h.id = f"H{i+1:03d}"

        return hypotheses, 1

    # ------------------------------------------------------------------
    # Alignment
    # ------------------------------------------------------------------

    async def _align_hypothesis(
        self, hypothesis: Hypothesis, context: str
    ) -> tuple[AlignmentScores, int]:
        """Score hypothesis on 4 dimensions via LLM."""
        messages = [
            {
                "role": "user",
                "content": (
                    "Score this optimization hypothesis on 4 dimensions (0-1):\n\n"
                    f"Hypothesis: {hypothesis.title}\n"
                    f"Description: {hypothesis.description}\n"
                    f"Category: {hypothesis.category}\n"
                    f"Risk: {hypothesis.risk}\n\n"
                    f"Context:\n{context[:2000]}\n\n"
                    "Score:\n"
                    "- metric_relevance: How directly does it target the metric?\n"
                    "- feasibility: How likely to implement correctly?\n"
                    "- safety: How unlikely to violate integrity constraints?\n"
                    "- novelty: How different from standard approaches?"
                ),
            }
        ]
        result = await self._call_llm(messages, response_model=AlignmentScores)
        if isinstance(result, AlignmentScores):
            result.composite = (
                result.metric_relevance * 0.35
                + result.feasibility * 0.30
                + result.safety * 0.25
                + result.novelty * 0.10
            )
        return result, 1

    # ------------------------------------------------------------------
    # Red-line audit
    # ------------------------------------------------------------------

    async def _red_line_audit(
        self, hypothesis: Hypothesis, hard_constraints: list[str]
    ) -> tuple[RedLineVerdict, int]:
        """Verify hypothesis against R1-R7.

        Returns CLEARED or REJECTED.
        """
        constraints_text = "\n".join(
            f"- {c}" for c in hard_constraints
        ) if hard_constraints else "None specified."

        messages = [
            {
                "role": "user",
                "content": (
                    "Audit this optimization hypothesis for red-line violations.\n\n"
                    f"Hypothesis: {hypothesis.title}\n"
                    f"Description: {hypothesis.description}\n"
                    f"Steps: {hypothesis.implementation_steps}\n\n"
                    "Red-line rules:\n"
                    "R1: Evaluation metric parameters must not change\n"
                    "R2: Evaluation script integrity must be maintained\n"
                    "R3: Integrity of algorithm outputs must not be violated\n"
                    "R4: No unfair trade-offs between metric dimensions\n"
                    "R5: Dataset integrity must be preserved\n"
                    "R6: No modification of the dataset\n"
                    "R7: Paper-specific constraints must be followed\n\n"
                    f"Paper-specific constraints:\n{constraints_text}\n\n"
                    "Would implementing this hypothesis violate any rule? "
                    "Answer CLEARED or REJECTED with reasoning."
                ),
            }
        ]
        response = await self._call_llm(messages)
        content = response if isinstance(response, str) else str(response)

        if "REJECTED" in content.upper():
            hypothesis.red_line_notes = content[:500]
            return RedLineVerdict.REJECTED, 1

        return RedLineVerdict.CLEARED, 1

    # ------------------------------------------------------------------
    # Prioritisation
    # ------------------------------------------------------------------

    def _prioritize(self, hypotheses: list[Hypothesis]) -> list[Hypothesis]:
        """Sort hypotheses by composite alignment score (descending).

        Within tied scores, prefer lower risk.
        """
        risk_order = {RiskLevel.LOW: 0, RiskLevel.MEDIUM: 1, RiskLevel.HIGH: 2}
        return sorted(
            hypotheses,
            key=lambda h: (-h.alignment_score, risk_order.get(h.risk, 1)),
        )
