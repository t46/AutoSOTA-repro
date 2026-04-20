"""Unit tests for all AutoSOTA agents.

Uses a mock LLM provider to test agent logic without real API calls.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from autosota.agents.base import AgentResult, BaseAgent
from autosota.agents.fix import (
    AgentFix,
    FailureSignature,
    KNOWN_PATTERNS,
    RepairAction,
)
from autosota.agents.ideator import (
    AgentIdeator,
    AlignmentScores,
    Hypothesis,
    HypothesisBatch,
)
from autosota.agents.init_env import (
    AgentInit,
    DockerEnvironment,
    ExecutionPlan,
    RepositoryAnalysis,
)
from autosota.agents.monitor import AgentMonitor, DeadlockDetection, SupervisoryAction
from autosota.agents.objective import AgentObjective, TargetMetricInfo
from autosota.agents.resource import (
    AgentResource,
    PaperClassification,
    ReadinessAssessment,
)
from autosota.agents.scheduler import AgentScheduler, IterationRecord, OptimizationState
from autosota.agents.supervisor import AgentSupervisor, RedLineViolation, SupervisorVerdict
from autosota.constants import HypothesisType, MetricDirection, RedLineVerdict, RiskLevel


# ------------------------------------------------------------------
# Mock LLM provider
# ------------------------------------------------------------------


class MockLLM:
    """Minimal mock LLM that returns predictable responses."""

    def __init__(self, response: Any = "mock response"):
        self._response = response
        self.call_count = 0

    async def complete(self, messages, tools=None):
        self.call_count += 1
        return self._response

    async def complete_structured(self, messages, response_model):
        self.call_count += 1
        if isinstance(self._response, response_model):
            return self._response
        # Build a default instance of the response model
        return self._build_default(response_model)

    def _build_default(self, model_cls):
        """Build a default instance with required fields filled."""
        fields = model_cls.model_fields
        kwargs = {}
        for name, field in fields.items():
            if field.is_required():
                annotation = field.annotation
                if annotation is str:
                    kwargs[name] = "mock"
                elif annotation is float:
                    kwargs[name] = 0.5
                elif annotation is int:
                    kwargs[name] = 0
                elif annotation is bool:
                    kwargs[name] = True
                elif annotation is list:
                    kwargs[name] = []
        return model_cls(**kwargs)


@pytest.fixture
def workspace(tmp_path):
    return tmp_path


@pytest.fixture
def config():
    return {
        "llm": {"primary_model": "test/mock"},
        "docker": {
            "base_image": "python:3.10-slim",
            "platform": "linux/amd64",
        },
        "pipeline": {
            "max_iterations": 5,
            "max_fix_attempts": 3,
            "stagnation_window": 3,
            "honeymoon_iterations": 2,
            "budget_wall_clock_hours": 1.0,
            "budget_llm_calls": 100,
        },
        "supervisor": {
            "red_lines": [
                "R1: Evaluation metric parameters must not change",
            ],
        },
    }


# ------------------------------------------------------------------
# BaseAgent tests
# ------------------------------------------------------------------


class TestBaseAgent:

    def test_agent_result_creation(self):
        r = AgentResult(success=True, data={"key": "value"}, llm_calls_used=3)
        assert r.success is True
        assert r.data == {"key": "value"}
        assert r.llm_calls_used == 3
        assert r.error is None

    def test_agent_result_failure(self):
        r = AgentResult(success=False, error="something broke")
        assert r.success is False
        assert r.error == "something broke"
        assert r.data == {}

    def test_base_agent_is_abstract(self):
        with pytest.raises(TypeError):
            BaseAgent(MockLLM(), Path("/tmp"), {})

    def test_system_prompt_fallback(self, workspace, config):
        """When no prompt file exists, uses class name as prompt."""
        agent = AgentMonitor(MockLLM(), workspace, config)
        assert "AgentMonitor" in agent._system_prompt

    def test_system_prompt_from_file(self, workspace, config, tmp_path):
        """When prompt file exists, loads it."""
        prompts_dir = Path(__file__).parent.parent.parent / "prompts"
        prompts_dir.mkdir(exist_ok=True)
        prompt_file = prompts_dir / "monitor_system.md"
        prompt_file.write_text("You are the monitor agent. Be vigilant.")
        try:
            agent = AgentMonitor(MockLLM(), workspace, config)
            assert "Be vigilant" in agent._system_prompt
        finally:
            prompt_file.unlink(missing_ok=True)


# ------------------------------------------------------------------
# AgentResource tests
# ------------------------------------------------------------------


class TestAgentResource:

    @pytest.mark.asyncio
    async def test_run_with_repo_url(self, workspace, config):
        llm = MockLLM()
        llm_classify = PaperClassification(
            classification="methodological", reasoning="test", has_code=True
        )
        llm_readiness = ReadinessAssessment(
            readiness_score=0.8,
            has_requirements=True,
            has_eval_script=True,
            has_dataset_instructions=False,
            issues=[],
        )

        # We need a smarter mock that returns different models
        class SmartMock:
            call_count = 0

            async def complete(self, messages, tools=None):
                self.call_count += 1
                return "mock response"

            async def complete_structured(self, messages, response_model):
                self.call_count += 1
                if response_model is PaperClassification:
                    return llm_classify
                if response_model is ReadinessAssessment:
                    return llm_readiness
                return response_model(
                    **{
                        f.alias or name: "mock" if f.annotation is str else 0.5
                        for name, f in response_model.model_fields.items()
                        if f.is_required()
                    }
                )

        agent = AgentResource(SmartMock(), workspace, config)
        result = await agent.run(task={
            "title": "Test Paper",
            "repo_url": "https://github.com/test/repo",
        })

        assert result.success is True
        assert "paper_code_unit" in result.data
        assert result.data["paper_code_unit"]["repo_url"] == "https://github.com/test/repo"
        assert result.llm_calls_used > 0

    @pytest.mark.asyncio
    async def test_run_without_repo_url(self, workspace, config):
        """When no repo URL provided and LLM can't find one, fails gracefully."""
        llm = MockLLM("NOT_FOUND - no repository")
        agent = AgentResource(llm, workspace, config)
        result = await agent.run(task={"title": "Abstract Theory Paper"})
        # May succeed or fail depending on LLM mock behavior for classify/readiness
        # The key test is that it doesn't crash
        assert isinstance(result, AgentResult)


# ------------------------------------------------------------------
# AgentObjective tests
# ------------------------------------------------------------------


class TestAgentObjective:

    @pytest.mark.asyncio
    async def test_run_produces_rubric(self, workspace, config):
        metric = TargetMetricInfo(
            metric_name="PSNR",
            baseline_value=30.5,
            direction="higher_is_better",
            unit="dB",
            reasoning="Primary metric in the paper",
        )

        class MetricMock:
            call_count = 0

            async def complete(self, messages, tools=None):
                self.call_count += 1
                return "mock rubric criteria"

            async def complete_structured(self, messages, response_model):
                self.call_count += 1
                if response_model is TargetMetricInfo:
                    return metric
                return MockLLM()._build_default(response_model)

        agent = AgentObjective(MetricMock(), workspace, config)
        result = await agent.run(
            paper_unit={"paper": {"title": "Test"}, "repo_local_path": None},
            pdf_markdown="We evaluate using PSNR...",
        )

        assert result.success is True
        rubric = result.data["rubric"]
        assert rubric["target_metric"] == "PSNR"
        assert rubric["metric_direction"] == MetricDirection.HIGHER_IS_BETTER
        assert len(rubric["root"]["children"]) >= 3


# ------------------------------------------------------------------
# AgentInit tests
# ------------------------------------------------------------------


class TestAgentInit:

    @pytest.mark.asyncio
    async def test_run_without_repo_path(self, workspace, config):
        agent = AgentInit(MockLLM(), workspace, config)
        result = await agent.run(paper_unit={})
        assert result.success is False
        assert "repo_local_path" in result.error

    @pytest.mark.asyncio
    async def test_run_with_repo_path(self, workspace, config):
        repo = workspace / "test_repo"
        repo.mkdir()
        (repo / "requirements.txt").write_text("torch>=2.0\n")
        (repo / "README.md").write_text("# Test\nRun: python eval.py\n")

        analysis = RepositoryAnalysis(
            python_version="3.10",
            requirements_files=["requirements.txt"],
            gpu_required=False,
        )

        class InitMock:
            call_count = 0

            async def complete(self, messages, tools=None):
                self.call_count += 1
                return "python eval.py --data ./data"

            async def complete_structured(self, messages, response_model):
                self.call_count += 1
                if response_model is RepositoryAnalysis:
                    return analysis
                return MockLLM()._build_default(response_model)

        agent = AgentInit(InitMock(), workspace, config)
        result = await agent.run(
            paper_unit={"repo_local_path": str(repo)},
            rubric={"target_metric": "PSNR"},
        )

        assert result.success is True
        assert "docker_env" in result.data
        assert "execution_plan" in result.data


# ------------------------------------------------------------------
# AgentMonitor tests
# ------------------------------------------------------------------


class TestAgentMonitor:

    @pytest.mark.asyncio
    async def test_budget_abort(self, workspace, config):
        """When budget is exhausted, returns abort action."""
        agent = AgentMonitor(MockLLM(), workspace, config)
        result = await agent.run(
            state={"elapsed_seconds": 999999, "llm_calls": 0},
        )
        assert result.success is True
        action = result.data["supervisory_action"]
        assert action["action"] == "abort"

    @pytest.mark.asyncio
    async def test_normal_continue(self, workspace, config):
        agent = AgentMonitor(MockLLM(), workspace, config)
        result = await agent.run(
            state={"elapsed_seconds": 10, "llm_calls": 5},
        )
        assert result.success is True
        action = result.data["supervisory_action"]
        assert action["action"] == "continue"
        assert action["budget_remaining_pct"] > 90

    @pytest.mark.asyncio
    async def test_score_stagnation_detection(self, workspace, config):
        agent = AgentMonitor(MockLLM(), workspace, config)
        result = await agent.run(
            state={
                "elapsed_seconds": 10,
                "llm_calls": 5,
                "recent_scores": [0.5, 0.5, 0.5],
            },
        )
        assert result.success is True
        action = result.data["supervisory_action"]
        assert action["action"] == "guidance"
        assert "stagnation" in action["reason"]

    @pytest.mark.asyncio
    async def test_repeated_error_detection(self, workspace, config):
        agent = AgentMonitor(MockLLM(), workspace, config)
        result = await agent.run(
            trace_lines=["error X", "error X", "error X"],
            state={"elapsed_seconds": 10, "llm_calls": 5},
        )
        assert result.success is True
        action = result.data["supervisory_action"]
        assert action["action"] == "guidance"
        assert "repeated_error" in action["reason"]

    def test_budget_check_no_llm(self, workspace, config):
        agent = AgentMonitor(MockLLM(), workspace, config)
        action = agent._check_budget(elapsed=10, llm_calls_used=5)
        assert action.action == "continue"
        assert action.budget_remaining_pct > 90

    def test_budget_check_llm_exceeded(self, workspace, config):
        agent = AgentMonitor(MockLLM(), workspace, config)
        action = agent._check_budget(elapsed=0, llm_calls_used=200)
        assert action.action == "abort"

    @pytest.mark.asyncio
    async def test_build_code_analysis(self, workspace, config):
        repo = workspace / "repo"
        repo.mkdir()
        (repo / "model.py").write_text("class Model:\n    pass\n")

        agent = AgentMonitor(MockLLM(), workspace, config)
        content, calls = await agent.build_code_analysis(repo)
        assert calls == 1
        assert (workspace / "memory" / "code_analysis.md").exists()

    @pytest.mark.asyncio
    async def test_update_idea_library(self, workspace, config):
        agent = AgentMonitor(MockLLM(), workspace, config)
        await agent.update_idea_library(1, {
            "hypothesis": "increase lr",
            "score": 0.85,
            "success": True,
        })
        lib_path = workspace / "memory" / "idea_library.md"
        assert lib_path.exists()
        content = lib_path.read_text()
        assert "Iteration 1" in content
        assert "increase lr" in content


# ------------------------------------------------------------------
# AgentFix tests
# ------------------------------------------------------------------


class TestAgentFix:

    def test_normalize_error(self, workspace, config):
        agent = AgentFix(MockLLM(), workspace, config)
        tb = (
            'Traceback (most recent call last):\n'
            '  File "/app/train.py", line 42, in main\n'
            '    import missing_module\n'
            "ModuleNotFoundError: No module named 'missing_module'"
        )
        sig = agent._normalize_error(tb)
        assert sig.error_type == "ModuleNotFoundError"
        assert "missing_module" in sig.error_message
        assert sig.file_path == "/app/train.py"
        assert sig.line_number == 42
        assert len(sig.hash) == 12

    def test_known_pattern_import(self, workspace, config):
        agent = AgentFix(MockLLM(), workspace, config)
        tb = "ModuleNotFoundError: No module named 'scipy'"
        sig = agent._normalize_error(tb)
        repair = agent._lookup_skill(tb, sig)
        assert repair is not None
        assert "pip install scipy" in repair.commands[0]

    def test_known_pattern_file_not_found(self, workspace, config):
        agent = AgentFix(MockLLM(), workspace, config)
        tb = "FileNotFoundError: [Errno 2] No such file or directory: '/data/train.csv'"
        sig = agent._normalize_error(tb)
        repair = agent._lookup_skill(tb, sig)
        assert repair is not None
        assert "mkdir" in repair.commands[0]

    def test_known_pattern_no_match(self, workspace, config):
        agent = AgentFix(MockLLM(), workspace, config)
        tb = "SomeWeirdError: something unexpected"
        sig = agent._normalize_error(tb)
        repair = agent._lookup_skill(tb, sig)
        assert repair is None

    @pytest.mark.asyncio
    async def test_run_with_known_pattern(self, workspace, config):
        agent = AgentFix(MockLLM(), workspace, config)
        result = await agent.run(
            failure={
                "traceback": "ModuleNotFoundError: No module named 'numpy'",
                "command": "python train.py",
            },
        )
        assert result.success is True
        assert result.data["source"] == "known_pattern"
        assert result.llm_calls_used == 0

    @pytest.mark.asyncio
    async def test_run_with_llm_repair(self, workspace, config):
        repair = RepairAction(
            description="Fix the configuration",
            commands=["echo fix"],
            confidence=0.7,
        )
        agent = AgentFix(MockLLM(repair), workspace, config)
        result = await agent.run(
            failure={
                "traceback": "KeyError: 'nonexistent_key'",
                "command": "python train.py",
            },
        )
        assert result.success is True
        assert result.data["source"] == "llm_generated"
        assert result.llm_calls_used == 1

    def test_memory_persistence(self, workspace, config):
        agent = AgentFix(MockLLM(), workspace, config)
        sig = FailureSignature(
            error_type="TestError",
            error_message="test",
            hash="abc123def456",
        )
        repair = RepairAction(description="test fix", confidence=0.9)
        agent._update_memory(sig, repair, success=True)

        # Retrieve
        found = agent._retrieve_from_memory(sig)
        assert found is not None
        assert found["repair"]["description"] == "test fix"


# ------------------------------------------------------------------
# AgentIdeator tests
# ------------------------------------------------------------------


class TestAgentIdeator:

    @pytest.mark.asyncio
    async def test_run_generates_hypotheses(self, workspace, config):
        batch = HypothesisBatch(
            hypotheses=[
                Hypothesis(
                    title="Increase learning rate",
                    description="Try lr=0.01",
                    category=HypothesisType.PARAM,
                ),
                Hypothesis(
                    title="Add dropout",
                    description="Add dropout=0.3",
                    category=HypothesisType.CODE,
                ),
            ]
        )

        class IdeatorMock:
            call_count = 0

            async def complete(self, messages, tools=None):
                self.call_count += 1
                return "CLEARED - no violations"

            async def complete_structured(self, messages, response_model):
                self.call_count += 1
                if response_model is HypothesisBatch:
                    return batch
                if response_model is AlignmentScores:
                    return AlignmentScores(
                        metric_relevance=0.8,
                        feasibility=0.9,
                        safety=0.9,
                        novelty=0.5,
                    )
                return MockLLM()._build_default(response_model)

        agent = AgentIdeator(IdeatorMock(), workspace, config)
        result = await agent.run(
            code_analysis="Simple CNN model...",
            research_report="SOTA uses transformer...",
        )

        assert result.success is True
        assert result.data["total_generated"] == 2
        assert len(result.data["hypotheses"]) > 0

    def test_prioritize(self, workspace, config):
        agent = AgentIdeator(MockLLM(), workspace, config)
        hypotheses = [
            Hypothesis(
                title="A", description="x",
                alignment_score=0.3, risk=RiskLevel.LOW,
            ),
            Hypothesis(
                title="B", description="x",
                alignment_score=0.9, risk=RiskLevel.HIGH,
            ),
            Hypothesis(
                title="C", description="x",
                alignment_score=0.9, risk=RiskLevel.LOW,
            ),
        ]
        ordered = agent._prioritize(hypotheses)
        assert ordered[0].title == "C"  # highest score, lowest risk
        assert ordered[1].title == "B"  # same score, higher risk
        assert ordered[2].title == "A"  # lowest score


# ------------------------------------------------------------------
# AgentSupervisor tests
# ------------------------------------------------------------------


class TestAgentSupervisor:

    @pytest.mark.asyncio
    async def test_all_clear(self, workspace, config):
        agent = AgentSupervisor(MockLLM("NO - no violations"), workspace, config)
        result = await agent.run(
            code_diff="--- a/model.py\n+++ b/model.py\n+dropout=0.3",
            baseline_metrics={"accuracy": 0.90},
            current_metrics={"accuracy": 0.92},
        )
        assert result.success is True
        verdict = result.data["supervisor_verdict"]
        assert verdict["verdict"] == RedLineVerdict.CLEARED
        assert len(verdict["violations"]) == 0

    @pytest.mark.asyncio
    async def test_violation_detected(self, workspace, config):
        agent = AgentSupervisor(MockLLM("YES - metric threshold was changed"), workspace, config)
        result = await agent.run(
            code_diff="--- a/eval.py\n+++ b/eval.py\n-iou_threshold=0.5\n+iou_threshold=0.3",
        )
        assert result.success is True
        verdict = result.data["supervisor_verdict"]
        assert verdict["verdict"] == RedLineVerdict.REJECTED
        assert len(verdict["violations"]) > 0

    @pytest.mark.asyncio
    async def test_no_diff_no_checks(self, workspace, config):
        agent = AgentSupervisor(MockLLM(), workspace, config)
        result = await agent.run(code_diff="")
        assert result.success is True
        verdict = result.data["supervisor_verdict"]
        assert verdict["verdict"] == RedLineVerdict.CLEARED

    @pytest.mark.asyncio
    async def test_metric_tradeoff_check(self, workspace, config):
        agent = AgentSupervisor(MockLLM("YES - unfair tradeoff"), workspace, config)
        result = await agent.run(
            code_diff="",
            baseline_metrics={"accuracy": 0.90, "f1": 0.85},
            current_metrics={"accuracy": 0.95, "f1": 0.50},  # f1 dropped 41%
        )
        assert result.success is True


# ------------------------------------------------------------------
# AgentScheduler tests
# ------------------------------------------------------------------


class TestAgentScheduler:

    def test_determine_path_normal(self, workspace, config):
        agent = AgentScheduler(MockLLM(), workspace, config)
        records = [IterationRecord(iteration=1, score=0.5)]
        assert agent._determine_path(records) == "normal"

    def test_determine_path_leap_stagnation(self, workspace, config):
        agent = AgentScheduler(MockLLM(), workspace, config)
        records = [
            IterationRecord(iteration=5, score=0.8),
            IterationRecord(iteration=6, score=0.8),
            IterationRecord(iteration=7, score=0.8),
        ]
        assert agent._determine_path(records) == "leap"

    def test_determine_path_honeymoon(self, workspace, config):
        """During honeymoon period, always normal even if stagnant."""
        agent = AgentScheduler(MockLLM(), workspace, config)
        records = [
            IterationRecord(iteration=1, score=0.5),
            IterationRecord(iteration=2, score=0.5),
            IterationRecord(iteration=2, score=0.5),
        ]
        assert agent._determine_path(records) == "normal"

    def test_extract_metrics(self, workspace, config):
        agent = AgentScheduler(MockLLM(), workspace, config)
        output = "accuracy: 0.9523\nloss = 0.1234\nPSNR: 32.45"
        metrics = agent._extract_metrics(output, "")
        assert "accuracy" in metrics
        assert abs(metrics["accuracy"] - 0.9523) < 1e-5
        assert "loss" in metrics
        assert "psnr" in metrics

    def test_extract_metrics_empty(self, workspace, config):
        agent = AgentScheduler(MockLLM(), workspace, config)
        metrics = agent._extract_metrics("", "")
        assert metrics == {}

    def test_get_primary_score(self, workspace, config):
        agent = AgentScheduler(MockLLM(), workspace, config)
        metrics = {"accuracy": 0.95, "loss": 0.05}
        score = agent._get_primary_score(metrics, {})
        assert score == 0.95  # "accuracy" is a known primary name

    def test_get_primary_score_single(self, workspace, config):
        agent = AgentScheduler(MockLLM(), workspace, config)
        metrics = {"custom_metric": 42.0}
        score = agent._get_primary_score(metrics, {})
        assert score == 42.0

    def test_is_improvement(self, workspace, config):
        agent = AgentScheduler(MockLLM(), workspace, config)
        assert agent._is_improvement(0.95, 0.90) is True
        assert agent._is_improvement(0.85, 0.90) is False

    def test_record_score(self, workspace, config):
        agent = AgentScheduler(MockLLM(), workspace, config)
        agent._record_score(0, {"accuracy": 0.9}, "baseline")
        agent._record_score(1, {"accuracy": 0.92}, "increase_lr")

        scores_path = workspace / "scores.jsonl"
        assert scores_path.exists()
        lines = scores_path.read_text().strip().splitlines()
        assert len(lines) == 2
        record = json.loads(lines[0])
        assert record["iteration"] == 0
        assert record["metrics"]["accuracy"] == 0.9

    def test_sub_agent_lazy_init(self, workspace, config):
        agent = AgentScheduler(MockLLM(), workspace, config)
        assert agent._monitor is None
        monitor = agent.monitor
        assert isinstance(monitor, AgentMonitor)
        assert agent._monitor is monitor  # cached

    def test_optimization_state(self):
        state = OptimizationState()
        assert state.current_iteration == 0
        assert state.best_score is None
        assert state.score_history == []

    def test_iteration_record(self):
        record = IterationRecord(iteration=1, hypothesis="test", score=0.9)
        assert record.success is False  # default
        assert record.path_type == "normal"
