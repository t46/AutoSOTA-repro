"""AgentMonitor -- execution monitoring and memory management.

Watches pipeline execution, detects deadlocks, manages memory files
(``code_analysis.md``, ``research_report.md``, ``idea_library.md``),
and enforces budget limits.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from autosota.agents.base import AgentResult, BaseAgent


# ------------------------------------------------------------------
# Structured models
# ------------------------------------------------------------------


class SupervisoryAction(BaseModel):
    """Action recommended by the monitor."""

    action: str = "continue"  # "continue" | "pause" | "abort" | "guidance"
    reason: str = ""
    guidance: str = ""
    budget_remaining_pct: float = 100.0


class DeadlockDetection(BaseModel):
    """Result of deadlock / stagnation pattern detection."""

    is_deadlocked: bool = False
    pattern: str = ""
    suggestion: str = ""


# ------------------------------------------------------------------
# Agent
# ------------------------------------------------------------------


class AgentMonitor(BaseAgent):
    """Execution monitor and memory manager.

    Runs periodically during the optimisation loop to:
    1. Detect stagnation / deadlock patterns in recent trace history.
    2. Enforce wall-clock and LLM-call budgets.
    3. Generate corrective guidance when issues are detected.
    4. Build and maintain memory files used by other agents.
    """

    async def run(
        self,
        *,
        trace_lines: list[str] | None = None,
        state: dict[str, Any] | None = None,
    ) -> AgentResult:
        """Evaluate current execution state and return a supervisory action.

        Parameters
        ----------
        trace_lines:
            Recent log / trace lines from the pipeline.
        state:
            Dictionary with keys like ``elapsed_seconds``, ``llm_calls``,
            ``iteration``, ``recent_scores``, etc.
        """
        llm_calls = 0
        state = state or {}
        trace_lines = trace_lines or []

        try:
            # ----- Budget check (no LLM needed) -----------------------
            budget_action = self._check_budget(
                elapsed=state.get("elapsed_seconds", 0),
                llm_calls_used=state.get("llm_calls", 0),
            )
            if budget_action.action == "abort":
                return AgentResult(
                    success=True,
                    data={"supervisory_action": budget_action.model_dump()},
                    llm_calls_used=0,
                )

            # ----- Deadlock detection ---------------------------------
            deadlock = await self._detect_deadlock(trace_lines, state)
            if deadlock.is_deadlocked:
                llm_calls += 1
                self.logger.warning(
                    "deadlock_detected",
                    pattern=deadlock.pattern,
                )
                guidance, calls = await self._generate_guidance(
                    state, deadlock.suggestion
                )
                llm_calls += calls

                action = SupervisoryAction(
                    action="guidance",
                    reason=deadlock.pattern,
                    guidance=guidance,
                    budget_remaining_pct=budget_action.budget_remaining_pct,
                )
                return AgentResult(
                    success=True,
                    data={"supervisory_action": action.model_dump()},
                    llm_calls_used=llm_calls,
                )

            # ----- Normal: continue -----------------------------------
            action = SupervisoryAction(
                action="continue",
                budget_remaining_pct=budget_action.budget_remaining_pct,
            )
            return AgentResult(
                success=True,
                data={"supervisory_action": action.model_dump()},
                llm_calls_used=llm_calls,
            )

        except Exception as exc:
            self.logger.error("monitor_agent_failed", error=str(exc))
            return AgentResult(
                success=False,
                error=str(exc),
                llm_calls_used=llm_calls,
            )

    # ------------------------------------------------------------------
    # Deadlock detection
    # ------------------------------------------------------------------

    async def _detect_deadlock(
        self,
        trace_history: list[str],
        state: dict[str, Any],
    ) -> DeadlockDetection:
        """Pattern detection for stagnation and deadlocks.

        Checks:
        - Repeated identical errors in trace history.
        - Score stagnation over ``stagnation_window`` iterations.
        - Fix loop (more than ``max_fix_attempts`` consecutive failures).
        """
        pipeline_cfg = self.config.get("pipeline", {})
        stagnation_window = pipeline_cfg.get("stagnation_window", 3)

        # Check score stagnation
        recent_scores = state.get("recent_scores", [])
        if len(recent_scores) >= stagnation_window:
            window = recent_scores[-stagnation_window:]
            if len(set(round(s, 6) for s in window)) == 1:
                return DeadlockDetection(
                    is_deadlocked=True,
                    pattern="score_stagnation",
                    suggestion=(
                        f"Score has not changed in {stagnation_window} iterations. "
                        "Consider switching to Leap path or trying a different "
                        "hypothesis category."
                    ),
                )

        # Check repeated errors
        if len(trace_history) >= 3:
            recent = trace_history[-3:]
            if len(set(recent)) == 1:
                return DeadlockDetection(
                    is_deadlocked=True,
                    pattern="repeated_error",
                    suggestion=f"Same output repeated 3 times: {recent[0][:100]}",
                )

        # Check fix loop
        fix_attempts = state.get("consecutive_fix_attempts", 0)
        max_fix = pipeline_cfg.get("max_fix_attempts", 5)
        if fix_attempts >= max_fix:
            return DeadlockDetection(
                is_deadlocked=True,
                pattern="fix_loop",
                suggestion=(
                    f"Fix agent has failed {fix_attempts} consecutive times. "
                    "Rolling back and trying a different approach."
                ),
            )

        return DeadlockDetection()

    # ------------------------------------------------------------------
    # Guidance generation
    # ------------------------------------------------------------------

    async def _generate_guidance(
        self,
        state: dict[str, Any],
        issue: str,
    ) -> tuple[str, int]:
        """Ask the LLM for high-level corrective guidance."""
        messages = [
            {
                "role": "user",
                "content": (
                    "The optimization pipeline has hit an issue:\n"
                    f"{issue}\n\n"
                    f"Current state: iteration {state.get('iteration', '?')}, "
                    f"best score {state.get('best_score', '?')}\n\n"
                    "Provide concise, actionable guidance to resolve this. "
                    "Focus on what to try differently, not debugging details."
                ),
            }
        ]
        response = await self._call_llm(messages)
        content = response if isinstance(response, str) else str(response)
        return content, 1

    # ------------------------------------------------------------------
    # Budget enforcement
    # ------------------------------------------------------------------

    def _check_budget(self, elapsed: float, llm_calls_used: int) -> SupervisoryAction:
        """Enforce wall-clock and LLM-call limits (no LLM needed)."""
        pipeline_cfg = self.config.get("pipeline", {})
        max_hours = pipeline_cfg.get("budget_wall_clock_hours", 8.0)
        max_calls = pipeline_cfg.get("budget_llm_calls", 500)

        max_seconds = max_hours * 3600
        time_pct = (elapsed / max_seconds * 100) if max_seconds > 0 else 0
        calls_pct = (llm_calls_used / max_calls * 100) if max_calls > 0 else 0
        remaining_pct = 100 - max(time_pct, calls_pct)

        if remaining_pct <= 0:
            reason = []
            if time_pct >= 100:
                reason.append(f"wall-clock budget exceeded ({elapsed:.0f}s)")
            if calls_pct >= 100:
                reason.append(f"LLM call budget exceeded ({llm_calls_used} calls)")
            return SupervisoryAction(
                action="abort",
                reason="; ".join(reason),
                budget_remaining_pct=0,
            )

        if remaining_pct < 10:
            return SupervisoryAction(
                action="continue",
                reason=f"Budget low ({remaining_pct:.0f}% remaining)",
                budget_remaining_pct=remaining_pct,
            )

        return SupervisoryAction(
            action="continue",
            budget_remaining_pct=remaining_pct,
        )

    # ------------------------------------------------------------------
    # Memory file builders
    # ------------------------------------------------------------------

    async def build_code_analysis(
        self, repo_path: Path, env: dict[str, Any] | None = None
    ) -> tuple[str, int]:
        """Generate ``code_analysis.md`` by having the LLM explore the repo.

        Returns the markdown content and the number of LLM calls used.
        """
        context_parts: list[str] = []
        if repo_path.is_dir():
            # Top-level listing
            files = sorted(f.name for f in repo_path.iterdir())[:80]
            context_parts.append(f"Top-level files: {files}")

            # Read key source files
            for pattern in ["*.py", "**/*.py"]:
                for p in sorted(repo_path.glob(pattern))[:10]:
                    try:
                        content = p.read_text(errors="replace")[:2000]
                        rel = p.relative_to(repo_path)
                        context_parts.append(f"=== {rel} ===\n{content}")
                    except Exception:
                        continue

        messages = [
            {
                "role": "user",
                "content": (
                    "Analyse this repository's code and produce a structured "
                    "code analysis document (markdown).\n\n"
                    + "\n\n".join(context_parts[:15])
                    + "\n\nInclude:\n"
                    "1. Architecture overview\n"
                    "2. Key modules and their responsibilities\n"
                    "3. Data flow\n"
                    "4. Model architecture details\n"
                    "5. Training / evaluation pipeline\n"
                    "6. Potential optimization points\n"
                    "7. Known limitations"
                ),
            }
        ]
        response = await self._call_llm(messages)
        content = response if isinstance(response, str) else str(response)

        # Write to workspace
        output_path = self.workspace_dir / "memory" / "code_analysis.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content)
        self.logger.info("code_analysis_written", path=str(output_path))

        return content, 1

    async def build_research_report(
        self, paper_unit: dict[str, Any]
    ) -> tuple[str, int]:
        """Query LLM for SOTA insights related to the paper.

        Returns the markdown content and the number of LLM calls used.
        """
        title = paper_unit.get("paper", {}).get("title", "Unknown")
        abstract = paper_unit.get("paper", {}).get("abstract", "")

        messages = [
            {
                "role": "user",
                "content": (
                    "Provide a research report on the state-of-the-art for "
                    f"this paper's topic:\n\n"
                    f"Title: {title}\n"
                    f"Abstract: {abstract}\n\n"
                    "Include:\n"
                    "1. Recent SOTA methods and their key improvements\n"
                    "2. Common optimization techniques in this domain\n"
                    "3. Known tricks that improve performance\n"
                    "4. Potential directions for improvement"
                ),
            }
        ]
        response = await self._call_llm(messages)
        content = response if isinstance(response, str) else str(response)

        output_path = self.workspace_dir / "memory" / "research_report.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content)
        self.logger.info("research_report_written", path=str(output_path))

        return content, 1

    async def update_idea_library(
        self, iteration: int, result: dict[str, Any]
    ) -> None:
        """Append iteration outcome to the idea library log."""
        output_path = self.workspace_dir / "memory" / "idea_library.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        entry = (
            f"\n## Iteration {iteration}\n"
            f"- Hypothesis: {result.get('hypothesis', 'N/A')}\n"
            f"- Score: {result.get('score', 'N/A')}\n"
            f"- Success: {result.get('success', False)}\n"
            f"- Notes: {result.get('notes', '')}\n"
        )

        mode = "a" if output_path.exists() else "w"
        with open(output_path, mode) as f:
            if mode == "w":
                f.write("# Idea Library\n\nLog of optimization attempts.\n")
            f.write(entry)

        self.logger.info("idea_library_updated", iteration=iteration)
