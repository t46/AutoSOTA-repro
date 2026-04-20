"""Data models for Docker execution environments."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DockerEnvironment(BaseModel):
    """State of a Docker container used for code execution."""

    container_id: str | None = None
    image: str = "python:3.10-slim"
    python_version: str = "3.10"
    gpu_available: bool = False
    platform: str = "linux/arm64"
    memory_limit: str = "16g"
    network_mode: str = "bridge"
    working_dir: str = "/workspace"
    is_running: bool = False


class ExecutionPlan(BaseModel):
    """A concrete plan for setting up and evaluating a paper's code inside a container."""

    setup_commands: list[str] = Field(default_factory=list)
    install_commands: list[str] = Field(default_factory=list)
    eval_command: str = ""
    output_pattern: str = ""
    timeout_seconds: int = 7200
    environment_vars: dict[str, str] = Field(default_factory=dict)
