"""Data models for improvement hypotheses and the hypothesis library."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from autosota.constants import HypothesisType, RedLineVerdict, RiskLevel


class FourDimAlignment(BaseModel):
    """Four-dimensional alignment score for a hypothesis.

    Each dimension is scored 0.0 -- 1.0:
    - novelty: how original / non-obvious the idea is
    - feasibility: probability of successful implementation
    - impact: expected improvement magnitude
    - safety: likelihood of not violating red-lines
    """

    novelty: float = 0.0
    feasibility: float = 0.0
    impact: float = 0.0
    safety: float = 1.0

    def composite_score(self) -> float:
        """Weighted composite; safety acts as a gate (multiplied, not averaged)."""
        return (0.3 * self.novelty + 0.3 * self.feasibility + 0.4 * self.impact) * self.safety


class RedLineAudit(BaseModel):
    """Result of a red-line check on a hypothesis."""

    verdict: RedLineVerdict = RedLineVerdict.CLEARED
    checked_rules: list[str] = Field(default_factory=list)
    violations: list[str] = Field(default_factory=list)
    notes: str = ""


class Hypothesis(BaseModel):
    """A single improvement hypothesis, fully annotated for tracking."""

    id: str = ""
    hypothesis_type: HypothesisType = HypothesisType.PARAM
    title: str = ""
    description: str = ""
    rationale: str = ""

    # Alignment & audit
    alignment: FourDimAlignment = Field(default_factory=FourDimAlignment)
    audit: RedLineAudit = Field(default_factory=RedLineAudit)
    risk_level: RiskLevel = RiskLevel.LOW

    # Tracking
    iteration_proposed: int | None = None
    iteration_applied: int | None = None
    applied: bool = False
    result_delta: float | None = None
    git_hash: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Implementation details
    code_diff: str = ""
    rollback_diff: str = ""


class HypothesisLibrary(BaseModel):
    """Ordered collection of hypotheses plus hard constraints.

    Hypotheses are stored in priority order (highest composite score first).
    ``hard_constraints`` list red-line rule IDs that apply to the current task.
    """

    hypotheses: list[Hypothesis] = Field(default_factory=list)
    hard_constraints: list[str] = Field(default_factory=list)

    def add(self, hypothesis: Hypothesis) -> None:
        self.hypotheses.append(hypothesis)
        self._sort()

    def pending(self) -> list[Hypothesis]:
        """Return hypotheses that have not yet been applied."""
        return [h for h in self.hypotheses if not h.applied]

    def cleared(self) -> list[Hypothesis]:
        """Return hypotheses that passed the red-line audit."""
        return [h for h in self.hypotheses if h.audit.verdict == RedLineVerdict.CLEARED]

    def _sort(self) -> None:
        self.hypotheses.sort(key=lambda h: h.alignment.composite_score(), reverse=True)
