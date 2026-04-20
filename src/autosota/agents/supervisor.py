"""AgentSupervisor -- Red Line enforcement.

Examines code diffs, evaluation output, and metrics to ensure no
integrity constraint (R1-R7) is violated by an optimization step.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from autosota.agents.base import AgentResult, BaseAgent
from autosota.constants import RED_LINE_RULES, RedLineVerdict


# ------------------------------------------------------------------
# Structured models
# ------------------------------------------------------------------


class RedLineViolation(BaseModel):
    """A detected red-line violation."""

    rule: str  # e.g. "R1"
    description: str
    severity: str = "critical"  # "critical" | "warning"
    evidence: str = ""


class SupervisorVerdict(BaseModel):
    """Overall verdict from red-line checking."""

    verdict: RedLineVerdict = RedLineVerdict.CLEARED
    violations: list[RedLineViolation] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    notes: str = ""


# ------------------------------------------------------------------
# Agent
# ------------------------------------------------------------------


class AgentSupervisor(BaseAgent):
    """Red Line enforcement agent.

    Checks every optimization step against R1-R7 integrity constraints.
    Each check method uses LLM analysis of the code diff, eval output,
    and metrics.
    """

    async def run(
        self,
        *,
        code_diff: str = "",
        hypothesis: dict[str, Any] | None = None,
        eval_output: str = "",
        hard_constraints: list[str] | None = None,
        baseline_metrics: dict[str, float] | None = None,
        current_metrics: dict[str, float] | None = None,
    ) -> AgentResult:
        """Check all red-line rules and return a verdict.

        Parameters
        ----------
        code_diff:
            Git diff of the changes made in this iteration.
        hypothesis:
            The hypothesis being tested.
        eval_output:
            Raw output from the evaluation command.
        hard_constraints:
            Paper-specific constraints for R7.
        baseline_metrics:
            Metric values from baseline (iteration 0).
        current_metrics:
            Metric values from the current iteration.
        """
        llm_calls = 0
        hard_constraints = hard_constraints or []
        baseline_metrics = baseline_metrics or {}
        current_metrics = current_metrics or {}
        violations: list[RedLineViolation] = []
        warnings: list[str] = []

        try:
            # Run all checks
            checks = [
                ("R1", self._check_r1_metric_params(code_diff, eval_output)),
                ("R2", self._check_r2_eval_script(code_diff)),
                ("R3", self._check_r3_output_integrity(code_diff)),
                ("R4", self._check_r4_no_unfair_tradeoffs(
                    baseline_metrics, current_metrics
                )),
                ("R5", self._check_r5_dataset_integrity(code_diff)),
                ("R6", self._check_r6_no_data_modification(code_diff)),
                ("R7", self._check_r7_paper_specific(code_diff, hard_constraints)),
            ]

            for rule_id, coro in checks:
                violation, calls = await coro
                llm_calls += calls
                if violation:
                    violations.append(violation)
                    self.logger.warning(
                        "red_line_violation",
                        rule=rule_id,
                        description=violation.description,
                    )

            verdict = (
                RedLineVerdict.REJECTED if violations else RedLineVerdict.CLEARED
            )

            result = SupervisorVerdict(
                verdict=verdict,
                violations=violations,
                warnings=warnings,
            )

            return AgentResult(
                success=True,
                data={"supervisor_verdict": result.model_dump()},
                llm_calls_used=llm_calls,
            )

        except Exception as exc:
            self.logger.error("supervisor_agent_failed", error=str(exc))
            return AgentResult(
                success=False,
                error=str(exc),
                llm_calls_used=llm_calls,
            )

    # ------------------------------------------------------------------
    # R1: Evaluation metric parameters must not change
    # ------------------------------------------------------------------

    async def _check_r1_metric_params(
        self, code_diff: str, eval_output: str
    ) -> tuple[RedLineViolation | None, int]:
        """Check that evaluation metric parameters are unchanged."""
        if not code_diff:
            return None, 0

        messages = [
            {
                "role": "user",
                "content": (
                    "Check if this code diff changes any evaluation metric "
                    "parameters (e.g., IoU thresholds, confidence thresholds, "
                    "top-k values, distance metrics, etc.).\n\n"
                    f"Code diff:\n{code_diff[:4000]}\n\n"
                    "Answer YES if metric parameters were changed, NO otherwise. "
                    "Explain your reasoning."
                ),
            }
        ]
        response = await self._call_llm(messages)
        content = response if isinstance(response, str) else str(response)

        if content.strip().upper().startswith("YES"):
            return RedLineViolation(
                rule="R1",
                description="Evaluation metric parameters were modified",
                evidence=content[:500],
            ), 1

        return None, 1

    # ------------------------------------------------------------------
    # R2: Evaluation script integrity must be maintained
    # ------------------------------------------------------------------

    async def _check_r2_eval_script(
        self, code_diff: str
    ) -> tuple[RedLineViolation | None, int]:
        """Check that evaluation scripts are not improperly modified."""
        if not code_diff:
            return None, 0

        # Quick heuristic: check if eval files are in the diff
        eval_markers = ["eval.py", "evaluate.py", "test.py", "run_eval", "metrics.py"]
        has_eval_changes = any(marker in code_diff for marker in eval_markers)
        if not has_eval_changes:
            return None, 0

        messages = [
            {
                "role": "user",
                "content": (
                    "Check if this code diff improperly modifies evaluation "
                    "scripts. Allowed changes: logging, formatting, comments. "
                    "Disallowed: changing metric computation, skipping samples, "
                    "changing data preprocessing in eval.\n\n"
                    f"Code diff:\n{code_diff[:4000]}\n\n"
                    "Answer YES if the eval script integrity is violated, NO otherwise."
                ),
            }
        ]
        response = await self._call_llm(messages)
        content = response if isinstance(response, str) else str(response)

        if content.strip().upper().startswith("YES"):
            return RedLineViolation(
                rule="R2",
                description="Evaluation script integrity violated",
                evidence=content[:500],
            ), 1

        return None, 1

    # ------------------------------------------------------------------
    # R3: Integrity of algorithm outputs must not be violated
    # ------------------------------------------------------------------

    async def _check_r3_output_integrity(
        self, code_diff: str
    ) -> tuple[RedLineViolation | None, int]:
        """Check for fabricated or hardcoded outputs."""
        if not code_diff:
            return None, 0

        messages = [
            {
                "role": "user",
                "content": (
                    "Check if this code diff fabricates, hardcodes, or manipulates "
                    "algorithm outputs (e.g., hardcoded predictions, cached results "
                    "used as fresh outputs, skipped computation steps).\n\n"
                    f"Code diff:\n{code_diff[:4000]}\n\n"
                    "Answer YES if output integrity is violated, NO otherwise."
                ),
            }
        ]
        response = await self._call_llm(messages)
        content = response if isinstance(response, str) else str(response)

        if content.strip().upper().startswith("YES"):
            return RedLineViolation(
                rule="R3",
                description="Algorithm output integrity violated",
                evidence=content[:500],
            ), 1

        return None, 1

    # ------------------------------------------------------------------
    # R4: No unfair trade-offs between metric dimensions
    # ------------------------------------------------------------------

    async def _check_r4_no_unfair_tradeoffs(
        self,
        baseline: dict[str, float],
        current: dict[str, float],
    ) -> tuple[RedLineViolation | None, int]:
        """Check that improving one metric doesn't severely degrade others."""
        if not baseline or not current:
            return None, 0

        degraded = []
        for metric, base_val in baseline.items():
            curr_val = current.get(metric)
            if curr_val is None:
                continue
            # Check for significant degradation (> 10%)
            if base_val != 0:
                change_pct = (curr_val - base_val) / abs(base_val) * 100
                if change_pct < -10:
                    degraded.append(
                        f"{metric}: {base_val:.4f} -> {curr_val:.4f} "
                        f"({change_pct:.1f}%)"
                    )

        if degraded:
            messages = [
                {
                    "role": "user",
                    "content": (
                        "These metrics have degraded significantly:\n"
                        + "\n".join(f"- {d}" for d in degraded)
                        + "\n\nIs this an unfair trade-off? Consider whether the "
                        "degradation is acceptable given the primary metric gain. "
                        "Answer YES (unfair) or NO (acceptable)."
                    ),
                }
            ]
            response = await self._call_llm(messages)
            content = response if isinstance(response, str) else str(response)

            if content.strip().upper().startswith("YES"):
                return RedLineViolation(
                    rule="R4",
                    description="Unfair trade-off between metrics",
                    evidence="\n".join(degraded),
                ), 1

            return None, 1

        return None, 0

    # ------------------------------------------------------------------
    # R5: Dataset integrity must be preserved
    # ------------------------------------------------------------------

    async def _check_r5_dataset_integrity(
        self, code_diff: str
    ) -> tuple[RedLineViolation | None, int]:
        """Check that train/test split is not violated."""
        if not code_diff:
            return None, 0

        # Quick heuristic: check for data-related changes
        data_markers = [
            "train", "test", "val", "split", "dataset", "dataloader",
            "data_dir", "shuffle", "random_split",
        ]
        has_data_changes = any(marker in code_diff.lower() for marker in data_markers)
        if not has_data_changes:
            return None, 0

        messages = [
            {
                "role": "user",
                "content": (
                    "Check if this code diff violates dataset integrity:\n"
                    "- Leaking test data into training\n"
                    "- Changing train/test/val split ratios\n"
                    "- Using test set for hyperparameter tuning\n"
                    "- Duplicating data samples\n\n"
                    f"Code diff:\n{code_diff[:4000]}\n\n"
                    "Answer YES if dataset integrity is violated, NO otherwise."
                ),
            }
        ]
        response = await self._call_llm(messages)
        content = response if isinstance(response, str) else str(response)

        if content.strip().upper().startswith("YES"):
            return RedLineViolation(
                rule="R5",
                description="Dataset integrity violated",
                evidence=content[:500],
            ), 1

        return None, 1

    # ------------------------------------------------------------------
    # R6: No modification of the dataset
    # ------------------------------------------------------------------

    async def _check_r6_no_data_modification(
        self, code_diff: str
    ) -> tuple[RedLineViolation | None, int]:
        """Check that the dataset files themselves are not modified."""
        if not code_diff:
            return None, 0

        # Check for data file modifications in the diff
        data_extensions = [".csv", ".json", ".jsonl", ".tsv", ".txt", ".h5",
                          ".hdf5", ".npy", ".npz", ".pkl", ".pickle"]
        has_data_file_changes = any(ext in code_diff for ext in data_extensions)
        if not has_data_file_changes:
            return None, 0

        messages = [
            {
                "role": "user",
                "content": (
                    "Check if this code diff modifies dataset files (not just "
                    "code that reads data, but actual data files).\n\n"
                    f"Code diff:\n{code_diff[:4000]}\n\n"
                    "Answer YES if data files are modified, NO otherwise."
                ),
            }
        ]
        response = await self._call_llm(messages)
        content = response if isinstance(response, str) else str(response)

        if content.strip().upper().startswith("YES"):
            return RedLineViolation(
                rule="R6",
                description="Dataset files were modified",
                evidence=content[:500],
            ), 1

        return None, 1

    # ------------------------------------------------------------------
    # R7: Paper-specific constraints
    # ------------------------------------------------------------------

    async def _check_r7_paper_specific(
        self, code_diff: str, hard_constraints: list[str]
    ) -> tuple[RedLineViolation | None, int]:
        """Check paper-specific constraints."""
        if not code_diff or not hard_constraints:
            return None, 0

        constraints_text = "\n".join(f"- {c}" for c in hard_constraints)

        messages = [
            {
                "role": "user",
                "content": (
                    "Check if this code diff violates any of these paper-specific "
                    "constraints:\n\n"
                    f"{constraints_text}\n\n"
                    f"Code diff:\n{code_diff[:4000]}\n\n"
                    "Answer YES if any constraint is violated, NO otherwise. "
                    "Explain which constraint and why."
                ),
            }
        ]
        response = await self._call_llm(messages)
        content = response if isinstance(response, str) else str(response)

        if content.strip().upper().startswith("YES"):
            return RedLineViolation(
                rule="R7",
                description="Paper-specific constraint violated",
                evidence=content[:500],
            ), 1

        return None, 1
