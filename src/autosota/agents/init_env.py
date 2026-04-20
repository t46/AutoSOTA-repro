"""AgentInit -- Docker environment setup.

Analyses the repository, builds a Docker image, installs dependencies,
discovers the evaluation command, and validates the environment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from autosota.agents.base import AgentResult, BaseAgent


# ------------------------------------------------------------------
# Structured models
# ------------------------------------------------------------------


class RepositoryAnalysis(BaseModel):
    """LLM analysis of a repository's setup requirements."""

    python_version: str = "3.10"
    requirements_files: list[str] = Field(default_factory=list)
    setup_commands: list[str] = Field(default_factory=list)
    system_packages: list[str] = Field(default_factory=list)
    gpu_required: bool = False
    notes: str = ""


class DockerEnvironment(BaseModel):
    """Represents a running Docker environment."""

    container_id: str = ""
    image_name: str = ""
    repo_mount_path: str = "/workspace/repo"
    working_dir: str = "/workspace/repo"
    is_ready: bool = False


class ExecutionPlan(BaseModel):
    """Plan for running evaluation inside the container."""

    eval_command: str = ""
    metric_parse_pattern: str = ""
    expected_runtime_seconds: int = 300
    setup_commands: list[str] = Field(default_factory=list)
    env_vars: dict[str, str] = Field(default_factory=dict)


# ------------------------------------------------------------------
# Agent
# ------------------------------------------------------------------


class AgentInit(BaseAgent):
    """Environment setup agent.

    Builds a Docker container, installs dependencies, and validates
    that the evaluation command runs successfully.
    """

    async def run(
        self,
        *,
        paper_unit: dict[str, Any],
        rubric: dict[str, Any] | None = None,
    ) -> AgentResult:
        """Set up Docker environment and execution plan.

        Parameters
        ----------
        paper_unit:
            Serialised ``PaperCodeUnit``.
        rubric:
            Serialised ``RubricTree`` (used to inform eval command discovery).
        """
        llm_calls = 0

        try:
            repo_path = paper_unit.get("repo_local_path")
            if not repo_path:
                return AgentResult(
                    success=False,
                    error="No repo_local_path in paper_unit.",
                    llm_calls_used=0,
                )

            repo_path = Path(repo_path)

            # ----- Step 1: analyse repository -------------------------
            analysis, calls = await self._analyze_repository(repo_path)
            llm_calls += calls
            self.logger.info(
                "repo_analyzed",
                python=analysis.python_version,
                gpu_required=analysis.gpu_required,
            )

            # ----- Step 2: build Docker environment -------------------
            env, calls = await self._build_docker_environment(repo_path, analysis)
            llm_calls += calls

            # ----- Step 3: install dependencies -----------------------
            install_ok, calls = await self._install_dependencies(env, repo_path, analysis)
            llm_calls += calls
            if not install_ok:
                self.logger.warning("dependency_install_issues")

            # ----- Step 4: discover eval command ----------------------
            target_metric = ""
            if rubric:
                target_metric = rubric.get("target_metric", "")

            plan, calls = await self._discover_eval_command(
                repo_path, target_metric, paper_unit
            )
            llm_calls += calls
            self.logger.info("eval_command_discovered", command=plan.eval_command)

            # ----- Step 5: validate environment -----------------------
            valid, calls = await self._validate_environment(env, plan)
            llm_calls += calls
            env.is_ready = valid

            return AgentResult(
                success=True,
                data={
                    "docker_env": env.model_dump(),
                    "execution_plan": plan.model_dump(),
                    "analysis": analysis.model_dump(),
                },
                llm_calls_used=llm_calls,
            )

        except Exception as exc:
            self.logger.error("init_agent_failed", error=str(exc))
            return AgentResult(
                success=False,
                error=str(exc),
                llm_calls_used=llm_calls,
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _analyze_repository(
        self, repo_path: Path
    ) -> tuple[RepositoryAnalysis, int]:
        """LLM reads README, requirements.txt, setup.py to understand setup."""
        context_parts: list[str] = []

        # Gather key files
        for filename in [
            "README.md",
            "readme.md",
            "requirements.txt",
            "setup.py",
            "setup.cfg",
            "pyproject.toml",
            "environment.yml",
            "Dockerfile",
            "Makefile",
        ]:
            p = repo_path / filename
            if p.exists():
                content = p.read_text(errors="replace")[:3000]
                context_parts.append(f"=== {filename} ===\n{content}")

        # File listing
        if repo_path.is_dir():
            files = sorted(f.name for f in repo_path.iterdir())[:80]
            context_parts.append(f"=== Directory listing ===\n{files}")

        messages = [
            {
                "role": "user",
                "content": (
                    "Analyse this repository and determine setup requirements.\n\n"
                    + "\n\n".join(context_parts)
                    + "\n\nReturn:\n"
                    "- python_version\n"
                    "- requirements_files (list of files to pip install)\n"
                    "- setup_commands (shell commands needed before pip install)\n"
                    "- system_packages (apt packages needed)\n"
                    "- gpu_required (bool)\n"
                    "- notes"
                ),
            }
        ]
        result = await self._call_llm(messages, response_model=RepositoryAnalysis)
        return result, 1

    async def _build_docker_environment(
        self, repo_path: Path, analysis: RepositoryAnalysis
    ) -> tuple[DockerEnvironment, int]:
        """Build or pull the Docker image and create a container.

        For MVP: returns a DockerEnvironment placeholder.  Actual Docker
        operations are delegated to ``tools.docker_tools.DockerManager``
        when available.
        """
        docker_config = self.config.get("docker", {})
        base_image = docker_config.get("base_image", "python:3.10-slim")
        platform = docker_config.get("platform", "linux/amd64")

        # Derive image name
        repo_name = repo_path.name if repo_path else "default"
        image_name = f"autosota-{repo_name}:latest"

        self.logger.info(
            "building_docker_env",
            base_image=base_image,
            platform=platform,
            image=image_name,
        )

        # TODO: actual Docker build via DockerManager
        # For MVP, create placeholder environment
        env = DockerEnvironment(
            container_id="",  # will be filled by DockerManager
            image_name=image_name,
            repo_mount_path="/workspace/repo",
            working_dir="/workspace/repo",
        )
        return env, 0

    async def _install_dependencies(
        self,
        env: DockerEnvironment,
        repo_path: Path,
        analysis: RepositoryAnalysis,
    ) -> tuple[bool, int]:
        """Install pip dependencies inside the container.

        For MVP: generates the install commands but does not execute them.
        """
        commands = []

        # System packages
        if analysis.system_packages:
            pkgs = " ".join(analysis.system_packages)
            commands.append(f"apt-get update && apt-get install -y {pkgs}")

        # Python packages
        for req_file in analysis.requirements_files:
            commands.append(f"pip install -r {req_file}")

        # Setup commands
        commands.extend(analysis.setup_commands)

        self.logger.info(
            "install_commands_generated",
            count=len(commands),
            commands=commands[:5],
        )

        # TODO: execute via DockerManager
        return True, 0

    async def _discover_eval_command(
        self,
        repo_path: Path,
        target_metric: str,
        paper_unit: dict[str, Any],
    ) -> tuple[ExecutionPlan, int]:
        """Find the evaluation command from README, scripts, or config.

        If ``eval_command`` is already provided in paper_unit, use that.
        """
        explicit_cmd = paper_unit.get("eval_command", "")
        if explicit_cmd:
            return ExecutionPlan(eval_command=explicit_cmd), 0

        # Gather clues from the repository
        context_parts: list[str] = []
        for filename in ["README.md", "readme.md", "Makefile", "run.sh", "eval.py",
                         "evaluate.py", "test.py", "scripts/eval.sh"]:
            p = repo_path / filename
            if p.exists():
                content = p.read_text(errors="replace")[:3000]
                context_parts.append(f"=== {filename} ===\n{content}")

        messages = [
            {
                "role": "user",
                "content": (
                    "Determine the evaluation command for this repository.\n"
                    f"Target metric: {target_metric}\n\n"
                    + "\n\n".join(context_parts)
                    + "\n\nReturn the exact shell command to run evaluation, "
                    "and a regex pattern to parse the metric value from output."
                ),
            }
        ]
        response = await self._call_llm(messages)
        content = response if isinstance(response, str) else str(response)

        plan = ExecutionPlan(
            eval_command=content.strip()[:500],
            metric_parse_pattern=r"[\d.]+",
        )
        return plan, 1

    async def _validate_environment(
        self, env: DockerEnvironment, plan: ExecutionPlan
    ) -> tuple[bool, int]:
        """Quick test that the eval command can at least parse.

        For MVP: validates the plan structure without actually running
        inside Docker.
        """
        if not plan.eval_command:
            self.logger.warning("no_eval_command_to_validate")
            return False, 0

        self.logger.info("environment_validated", command=plan.eval_command[:80])
        # TODO: run a quick test inside Docker container
        return True, 0
