"""AgentScheduler -- main orchestration agent.

Runs the four-phase optimization pipeline:
  Phase 0: Baseline reproduction
  Phase 1: Deep understanding (code analysis)
  Phase 2: Idea library construction
  Phase 3: Iterative optimization loop

Holds references to AgentMonitor, AgentIdeator, AgentSupervisor, and AgentFix.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from autosota.agents.base import AgentResult, BaseAgent
from autosota.agents.fix import AgentFix
from autosota.agents.ideator import AgentIdeator, Hypothesis
from autosota.agents.monitor import AgentMonitor
from autosota.agents.supervisor import AgentSupervisor
from autosota.constants import MetricDirection, PhaseName, RedLineVerdict


# ------------------------------------------------------------------
# Structured models
# ------------------------------------------------------------------


class IterationRecord(BaseModel):
    """Record of a single optimization iteration."""

    iteration: int
    hypothesis: str = ""
    hypothesis_category: str = ""
    path_type: str = "normal"  # "normal" | "leap"
    score: float | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    commit_hash: str = ""
    success: bool = False
    error: str = ""
    duration_seconds: float = 0.0


class OptimizationState(BaseModel):
    """Tracks overall optimization state across iterations."""

    phase: str = PhaseName.PHASE_0_BASELINE.value
    current_iteration: int = 0
    best_score: float | None = None
    best_commit: str = ""
    baseline_score: float | None = None
    baseline_metrics: dict[str, float] = Field(default_factory=dict)
    score_history: list[float] = Field(default_factory=list)
    iterations: list[IterationRecord] = Field(default_factory=list)
    total_llm_calls: int = 0
    start_time: float = 0.0
    consecutive_fix_attempts: int = 0


# ------------------------------------------------------------------
# Agent
# ------------------------------------------------------------------


class AgentScheduler(BaseAgent):
    """Main orchestration agent for the optimization pipeline.

    Manages the four-phase workflow and coordinates sub-agents.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        # Sub-agents -- created lazily or injected
        self._monitor: AgentMonitor | None = None
        self._ideator: AgentIdeator | None = None
        self._supervisor: AgentSupervisor | None = None
        self._fix: AgentFix | None = None

    # ------------------------------------------------------------------
    # Sub-agent accessors (lazy initialisation)
    # ------------------------------------------------------------------

    @property
    def monitor(self) -> AgentMonitor:
        if self._monitor is None:
            self._monitor = AgentMonitor(self.llm, self.workspace_dir, self.config)
        return self._monitor

    @property
    def ideator(self) -> AgentIdeator:
        if self._ideator is None:
            self._ideator = AgentIdeator(self.llm, self.workspace_dir, self.config)
        return self._ideator

    @property
    def supervisor(self) -> AgentSupervisor:
        if self._supervisor is None:
            self._supervisor = AgentSupervisor(self.llm, self.workspace_dir, self.config)
        return self._supervisor

    @property
    def fix(self) -> AgentFix:
        if self._fix is None:
            self._fix = AgentFix(self.llm, self.workspace_dir, self.config)
        return self._fix

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(
        self,
        *,
        paper_unit: dict[str, Any],
        env: dict[str, Any],
        plan: dict[str, Any],
        hard_constraints: list[str] | None = None,
    ) -> AgentResult:
        """Execute the full four-phase optimization pipeline.

        Parameters
        ----------
        paper_unit:
            Serialised ``PaperCodeUnit``.
        env:
            Serialised ``DockerEnvironment``.
        plan:
            Serialised ``ExecutionPlan``.
        hard_constraints:
            Paper-specific constraints for red-line checks.
        """
        hard_constraints = hard_constraints or []
        state = OptimizationState(start_time=time.time())
        llm_calls = 0

        try:
            repo_path = paper_unit.get("repo_local_path", "")

            # === Phase 0: Baseline ====================================
            state.phase = PhaseName.PHASE_0_BASELINE.value
            self.logger.info("phase_0_start")
            result = await self._phase_0_baseline(env, plan, state)
            llm_calls += result.llm_calls_used

            if not result.success:
                return AgentResult(
                    success=False,
                    error=f"Phase 0 failed: {result.error}",
                    llm_calls_used=llm_calls,
                )

            # === Phase 1: Understanding ===============================
            state.phase = PhaseName.PHASE_1_UNDERSTANDING.value
            self.logger.info("phase_1_start")
            result = await self._phase_1_understanding(
                env, Path(repo_path) if repo_path else None, paper_unit
            )
            llm_calls += result.llm_calls_used
            code_analysis = result.data.get("code_analysis", "")
            research_report = result.data.get("research_report", "")

            # === Phase 2: Idea Library ================================
            state.phase = PhaseName.PHASE_2_IDEA_LIBRARY.value
            self.logger.info("phase_2_start")
            result = await self._phase_2_idea_library(
                code_analysis, research_report, hard_constraints
            )
            llm_calls += result.llm_calls_used
            hypotheses = result.data.get("hypotheses", [])

            # === Phase 3: Optimization Loop ===========================
            state.phase = PhaseName.PHASE_3_OPTIMIZATION.value
            self.logger.info("phase_3_start", num_hypotheses=len(hypotheses))
            result = await self._phase_3_optimization_loop(
                env, plan, hypotheses, state, hard_constraints
            )
            llm_calls += result.llm_calls_used

            state.total_llm_calls = llm_calls

            return AgentResult(
                success=True,
                data={
                    "state": state.model_dump(),
                    "best_score": state.best_score,
                    "best_commit": state.best_commit,
                    "iterations": [it.model_dump() for it in state.iterations],
                },
                llm_calls_used=llm_calls,
            )

        except Exception as exc:
            self.logger.error("scheduler_failed", error=str(exc))
            return AgentResult(
                success=False,
                error=str(exc),
                llm_calls_used=llm_calls,
                data={"state": state.model_dump()},
            )

    # ------------------------------------------------------------------
    # Phase 0: Baseline Reproduction
    # ------------------------------------------------------------------

    async def _phase_0_baseline(
        self,
        env: dict[str, Any],
        plan: dict[str, Any],
        state: OptimizationState,
    ) -> AgentResult:
        """Run baseline evaluation and record iteration 0."""
        eval_command = plan.get("eval_command", "")
        metric_pattern = plan.get("metric_parse_pattern", "")

        if not eval_command:
            return AgentResult(
                success=False,
                error="No eval_command in execution plan.",
            )

        # Execute eval command
        output, success = await self._execute_eval(env, eval_command)

        if not success:
            # Try to fix
            fix_result = await self.fix.run(
                failure={"traceback": output, "command": eval_command},
                env=env,
            )
            if fix_result.success:
                # Retry after fix
                repair = fix_result.data.get("repair", {})
                for cmd in repair.get("commands", []):
                    await self._execute_in_env(env, cmd)
                output, success = await self._execute_eval(env, eval_command)

            if not success:
                return AgentResult(
                    success=False,
                    error=f"Baseline eval failed: {output[:500]}",
                    llm_calls_used=fix_result.llm_calls_used if fix_result else 0,
                )

        # Parse metrics
        metrics = self._extract_metrics(output, metric_pattern)
        primary_score = self._get_primary_score(metrics, plan)

        state.baseline_score = primary_score
        state.baseline_metrics = metrics
        state.best_score = primary_score
        state.score_history.append(primary_score or 0.0)

        record = IterationRecord(
            iteration=0,
            hypothesis="baseline",
            score=primary_score,
            metrics=metrics,
            success=True,
        )
        state.iterations.append(record)

        # Record scores.jsonl
        self._record_score(0, metrics, "baseline")

        self.logger.info(
            "baseline_recorded",
            score=primary_score,
            metrics=metrics,
        )

        return AgentResult(
            success=True,
            data={"baseline_score": primary_score, "metrics": metrics},
            llm_calls_used=0,
        )

    # ------------------------------------------------------------------
    # Phase 1: Deep Understanding
    # ------------------------------------------------------------------

    async def _phase_1_understanding(
        self,
        env: dict[str, Any],
        repo_path: Path | None,
        paper_unit: dict[str, Any],
    ) -> AgentResult:
        """Build code_analysis.md and research_report.md via AgentMonitor."""
        llm_calls = 0

        code_analysis = ""
        if repo_path:
            code_analysis, calls = await self.monitor.build_code_analysis(
                repo_path, env
            )
            llm_calls += calls

        research_report, calls = await self.monitor.build_research_report(paper_unit)
        llm_calls += calls

        return AgentResult(
            success=True,
            data={
                "code_analysis": code_analysis,
                "research_report": research_report,
            },
            llm_calls_used=llm_calls,
        )

    # ------------------------------------------------------------------
    # Phase 2: Idea Library Construction
    # ------------------------------------------------------------------

    async def _phase_2_idea_library(
        self,
        code_analysis: str,
        research_report: str,
        hard_constraints: list[str],
    ) -> AgentResult:
        """Invoke AgentIdeator to generate hypotheses."""
        result = await self.ideator.run(
            code_analysis=code_analysis,
            research_report=research_report,
            hard_constraints=hard_constraints,
        )
        return result

    # ------------------------------------------------------------------
    # Phase 3: Iterative Optimization Loop
    # ------------------------------------------------------------------

    async def _phase_3_optimization_loop(
        self,
        env: dict[str, Any],
        plan: dict[str, Any],
        hypotheses: list[dict[str, Any]],
        state: OptimizationState,
        hard_constraints: list[str],
    ) -> AgentResult:
        """Run the iterative optimization loop."""
        pipeline_cfg = self.config.get("pipeline", {})
        max_iterations = pipeline_cfg.get("max_iterations", 15)
        llm_calls = 0

        for i in range(1, max_iterations + 1):
            state.current_iteration = i
            elapsed = time.time() - state.start_time

            # ----- Monitor check ----------------------------------
            monitor_result = await self.monitor.run(
                trace_lines=[],
                state={
                    "elapsed_seconds": elapsed,
                    "llm_calls": state.total_llm_calls + llm_calls,
                    "iteration": i,
                    "best_score": state.best_score,
                    "recent_scores": state.score_history[-5:],
                    "consecutive_fix_attempts": state.consecutive_fix_attempts,
                },
            )
            llm_calls += monitor_result.llm_calls_used

            action = monitor_result.data.get("supervisory_action", {})
            if action.get("action") == "abort":
                self.logger.info("budget_exhausted", reason=action.get("reason"))
                break

            # ----- Select hypothesis or leap ----------------------
            path_type = self._determine_path(state.iterations[-3:])

            if path_type == "leap":
                hyp_dict, calls = await self._leap_path_synthesis(
                    state.iterations, ""
                )
                llm_calls += calls
            elif i - 1 < len(hypotheses):
                hyp_dict = hypotheses[i - 1]
            else:
                # Re-generate if we've exhausted the library
                self.logger.info("hypothesis_library_exhausted", regenerating=True)
                break

            # ----- Run iteration ----------------------------------
            record, calls = await self._run_iteration(
                i, hyp_dict, env, plan, state, hard_constraints
            )
            llm_calls += calls
            state.iterations.append(record)

            if record.score is not None:
                state.score_history.append(record.score)

            # Update best
            if record.success and record.score is not None:
                if state.best_score is None or self._is_improvement(
                    record.score, state.best_score
                ):
                    state.best_score = record.score
                    state.best_commit = record.commit_hash
                    self.logger.info(
                        "new_best_score",
                        iteration=i,
                        score=record.score,
                    )

            # Update idea library
            await self.monitor.update_idea_library(i, {
                "hypothesis": hyp_dict.get("title", ""),
                "score": record.score,
                "success": record.success,
                "notes": record.error,
            })

        return AgentResult(
            success=True,
            data={"iterations_completed": state.current_iteration},
            llm_calls_used=llm_calls,
        )

    # ------------------------------------------------------------------
    # Single iteration
    # ------------------------------------------------------------------

    async def _run_iteration(
        self,
        iteration: int,
        hypothesis: dict[str, Any],
        env: dict[str, Any],
        plan: dict[str, Any],
        state: OptimizationState,
        hard_constraints: list[str],
    ) -> tuple[IterationRecord, int]:
        """Execute the 7-step iteration protocol.

        Steps:
        1. Implement the hypothesis (LLM generates code changes).
        2. Git snapshot.
        3. Run evaluation.
        4. Parse metrics.
        5. Supervisor red-line check.
        6. Accept or rollback.
        7. Record results.
        """
        llm_calls = 0
        start = time.time()
        hyp_title = hypothesis.get("title", f"hypothesis_{iteration}")
        record = IterationRecord(
            iteration=iteration,
            hypothesis=hyp_title,
            hypothesis_category=hypothesis.get("category", ""),
        )

        try:
            # Step 1: Implement hypothesis
            impl_result, calls = await self._implement_hypothesis(
                hypothesis, env, plan
            )
            llm_calls += calls

            if not impl_result.get("success"):
                # Attempt fix
                fix_result = await self.fix.run(
                    failure={
                        "traceback": impl_result.get("error", ""),
                        "command": "implement hypothesis",
                    },
                    env=env,
                )
                llm_calls += fix_result.llm_calls_used
                state.consecutive_fix_attempts += 1

                if not fix_result.success:
                    record.error = f"Implementation failed: {impl_result.get('error', '')}"
                    record.duration_seconds = time.time() - start
                    return record, llm_calls

            state.consecutive_fix_attempts = 0

            # Step 2: Git snapshot
            commit_hash = await self._git_snapshot(env, hyp_title)
            record.commit_hash = commit_hash

            # Step 3: Run evaluation
            eval_command = plan.get("eval_command", "")
            output, eval_success = await self._execute_eval(env, eval_command)

            if not eval_success:
                # Try fix + retry
                fix_result = await self.fix.run(
                    failure={"traceback": output, "command": eval_command},
                    env=env,
                )
                llm_calls += fix_result.llm_calls_used

                if fix_result.success:
                    repair = fix_result.data.get("repair", {})
                    for cmd in repair.get("commands", []):
                        await self._execute_in_env(env, cmd)
                    output, eval_success = await self._execute_eval(env, eval_command)

                if not eval_success:
                    await self._rollback(env, state.best_commit)
                    record.error = f"Eval failed: {output[:200]}"
                    record.duration_seconds = time.time() - start
                    return record, llm_calls

            # Step 4: Parse metrics
            metric_pattern = plan.get("metric_parse_pattern", "")
            metrics = self._extract_metrics(output, metric_pattern)
            primary_score = self._get_primary_score(metrics, plan)
            record.score = primary_score
            record.metrics = metrics

            # Step 5: Supervisor check
            code_diff = await self._get_code_diff(env, state.best_commit)
            sup_result = await self.supervisor.run(
                code_diff=code_diff,
                hypothesis=hypothesis,
                eval_output=output,
                hard_constraints=hard_constraints,
                baseline_metrics=state.baseline_metrics,
                current_metrics=metrics,
            )
            llm_calls += sup_result.llm_calls_used

            verdict_data = sup_result.data.get("supervisor_verdict", {})
            if verdict_data.get("verdict") == RedLineVerdict.REJECTED:
                violations = verdict_data.get("violations", [])
                violation_desc = "; ".join(
                    v.get("description", "") for v in violations
                )
                self.logger.warning(
                    "iteration_rejected",
                    iteration=iteration,
                    violations=violation_desc,
                )
                await self._rollback(env, state.best_commit)
                record.error = f"Red-line violation: {violation_desc}"
                record.duration_seconds = time.time() - start
                return record, llm_calls

            # Step 6: Accept
            record.success = True
            self._record_score(iteration, metrics, hyp_title)
            self.logger.info(
                "iteration_accepted",
                iteration=iteration,
                score=primary_score,
            )

        except Exception as exc:
            self.logger.error(
                "iteration_error",
                iteration=iteration,
                error=str(exc),
            )
            record.error = str(exc)
            await self._rollback(env, state.best_commit)

        record.duration_seconds = time.time() - start
        return record, llm_calls

    # ------------------------------------------------------------------
    # Path determination
    # ------------------------------------------------------------------

    def _determine_path(
        self, recent_iterations: list[IterationRecord]
    ) -> str:
        """Decide between Normal and Leap path.

        Switch to Leap when recent iterations show stagnation.
        """
        pipeline_cfg = self.config.get("pipeline", {})
        stagnation_window = pipeline_cfg.get("stagnation_window", 3)
        honeymoon = pipeline_cfg.get("honeymoon_iterations", 5)

        if len(recent_iterations) < stagnation_window:
            return "normal"

        # During honeymoon, always normal
        if recent_iterations and recent_iterations[-1].iteration <= honeymoon:
            return "normal"

        # Check for score stagnation
        recent_scores = [
            it.score for it in recent_iterations[-stagnation_window:]
            if it.score is not None
        ]
        if len(recent_scores) >= stagnation_window:
            unique = set(round(s, 6) for s in recent_scores)
            if len(unique) <= 1:
                self.logger.info("switching_to_leap_path")
                return "leap"

        return "normal"

    async def _leap_path_synthesis(
        self,
        history: list[IterationRecord],
        code_analysis: str,
    ) -> tuple[dict[str, Any], int]:
        """Generate a novel hypothesis when Normal path stagnates."""
        history_text = "\n".join(
            f"- Iter {it.iteration}: score={it.score}, hyp={it.hypothesis}"
            for it in history[-10:]
        )

        messages = [
            {
                "role": "user",
                "content": (
                    "The optimization has stagnated. Generate a NOVEL hypothesis "
                    "that is significantly different from all previous attempts.\n\n"
                    f"Previous attempts:\n{history_text}\n\n"
                    f"Code analysis:\n{code_analysis[:3000]}\n\n"
                    "Think outside the box. Consider:\n"
                    "- Completely different architecture components\n"
                    "- Unconventional training strategies\n"
                    "- Novel data augmentation approaches\n"
                    "- Ensemble or multi-stage methods\n\n"
                    "Return a single hypothesis with title, description, "
                    "category, and implementation steps."
                ),
            }
        ]
        response = await self._call_llm(messages)
        content = response if isinstance(response, str) else str(response)

        # Build hypothesis dict from response
        hyp = {
            "title": f"leap_{len(history)}",
            "description": content[:1000],
            "category": "algo",
            "risk": "high",
            "implementation_steps": [],
            "path_type": "leap",
        }
        return hyp, 1

    # ------------------------------------------------------------------
    # Hypothesis implementation
    # ------------------------------------------------------------------

    async def _implement_hypothesis(
        self,
        hypothesis: dict[str, Any],
        env: dict[str, Any],
        plan: dict[str, Any],
    ) -> tuple[dict[str, Any], int]:
        """Have the LLM generate and apply code changes for a hypothesis.

        For MVP: generates a description of changes.
        Full version: generates actual diffs and applies them.
        """
        messages = [
            {
                "role": "user",
                "content": (
                    "Implement this optimization hypothesis:\n\n"
                    f"Title: {hypothesis.get('title', '')}\n"
                    f"Description: {hypothesis.get('description', '')}\n"
                    f"Steps: {hypothesis.get('implementation_steps', [])}\n"
                    f"Files to modify: {hypothesis.get('files_to_modify', [])}\n\n"
                    "Generate the exact code changes needed. For each change, "
                    "specify the file, the old code to replace, and the new code."
                ),
            }
        ]
        response = await self._call_llm(messages)
        content = response if isinstance(response, str) else str(response)

        # TODO: parse and apply code changes via Docker exec
        return {"success": True, "changes": content[:2000]}, 1

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------

    async def _execute_eval(
        self, env: dict[str, Any], command: str
    ) -> tuple[str, bool]:
        """Execute evaluation command inside Docker.

        For MVP: returns placeholder output.
        Full version: uses DockerManager.exec_run().
        """
        # TODO: actual Docker execution
        self.logger.info("execute_eval", command=command[:80])
        # Placeholder -- in production this calls DockerManager
        return "", True

    async def _execute_in_env(self, env: dict[str, Any], command: str) -> str:
        """Execute arbitrary command inside Docker.

        For MVP: placeholder.
        """
        self.logger.info("execute_in_env", command=command[:80])
        return ""

    async def _get_code_diff(
        self, env: dict[str, Any], base_commit: str
    ) -> str:
        """Get git diff between current state and a base commit.

        For MVP: placeholder.
        """
        # TODO: git diff inside Docker
        return ""

    async def _git_snapshot(self, env: dict[str, Any], message: str) -> str:
        """Commit current state and return the commit hash.

        For MVP: returns empty string.
        Full version: runs git add/commit inside Docker.
        """
        # TODO: git operations via DockerManager
        self.logger.info("git_snapshot", message=message[:60])
        return ""

    async def _rollback(self, env: dict[str, Any], commit_hash: str) -> None:
        """Roll back to a previous commit.

        For MVP: logs the rollback.
        Full version: runs git checkout inside Docker.
        """
        if commit_hash:
            self.logger.info("rollback", commit=commit_hash[:12])
            # TODO: git checkout {commit_hash} via DockerManager

    # ------------------------------------------------------------------
    # Metric extraction
    # ------------------------------------------------------------------

    def _extract_metrics(
        self, output: str, pattern: str
    ) -> dict[str, float]:
        """Parse evaluation output for metric values.

        Looks for patterns like ``metric_name: 0.1234`` or ``metric = 0.1234``.
        """
        metrics: dict[str, float] = {}

        # General pattern: key: value or key = value
        general_pattern = re.compile(
            r"(\w[\w\s]*\w|\w+)\s*[=:]\s*([\d]+\.[\d]+(?:e[+-]?\d+)?)"
        )
        for match in general_pattern.finditer(output):
            key = match.group(1).strip().lower().replace(" ", "_")
            try:
                value = float(match.group(2))
                metrics[key] = value
            except ValueError:
                continue

        # User-specified pattern
        if pattern and pattern != r"[\d.]+":
            try:
                user_matches = re.findall(pattern, output)
                for i, m in enumerate(user_matches):
                    try:
                        metrics[f"parsed_{i}"] = float(m)
                    except (ValueError, TypeError):
                        continue
            except re.error:
                pass

        return metrics

    def _get_primary_score(
        self, metrics: dict[str, float], plan: dict[str, Any]
    ) -> float | None:
        """Extract the primary metric score from parsed metrics."""
        if not metrics:
            return None

        # If only one metric, use it
        if len(metrics) == 1:
            return next(iter(metrics.values()))

        # Look for common primary metric names
        primary_names = [
            "accuracy", "acc", "psnr", "ssim", "bleu", "rouge",
            "f1", "map", "mAP", "auc", "loss", "error", "rmse",
            "mae", "mse", "iou", "dice",
        ]
        for name in primary_names:
            if name in metrics:
                return metrics[name]

        # Fall back to first metric
        return next(iter(metrics.values()))

    def _is_improvement(self, new_score: float, best_score: float) -> bool:
        """Check if new_score is better than best_score.

        Uses metric direction from config if available.
        """
        # Default: higher is better
        return new_score > best_score

    # ------------------------------------------------------------------
    # Score recording
    # ------------------------------------------------------------------

    def _record_score(
        self, iteration: int, metrics: dict[str, float], hypothesis: str
    ) -> None:
        """Append iteration results to scores.jsonl."""
        scores_path = self.workspace_dir / "scores.jsonl"
        scores_path.parent.mkdir(parents=True, exist_ok=True)

        record = {
            "iteration": iteration,
            "metrics": metrics,
            "hypothesis": hypothesis,
            "timestamp": time.time(),
        }

        with open(scores_path, "a") as f:
            f.write(json.dumps(record) + "\n")
