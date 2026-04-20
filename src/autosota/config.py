"""Settings management for AutoSOTA.

Configuration is resolved in the following order (later sources override earlier):
1. Built-in defaults (coded in the Pydantic models below)
2. configs/default.yaml
3. configs/m1_local.yaml  (optional platform-specific overrides)
4. Environment variables   (prefix: ``AUTOSOTA_``)
5. Secrets .env file       (API keys loaded via dotenv)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from autosota.constants import RED_LINE_RULES

# ---------------------------------------------------------------------------
# Project root detection
# ---------------------------------------------------------------------------

_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parent.parent.parent  # src/autosota/config.py -> repo root


# ---------------------------------------------------------------------------
# Sub-settings
# ---------------------------------------------------------------------------


class LLMSettings(BaseModel):
    primary_model: str = "anthropic/claude-sonnet-4-20250514"
    fallback_model: str = "openai/gpt-4o"
    research_model: str = "google/gemini-2.5-pro"
    temperature: float = 0.3
    max_tokens: int = 16384
    timeout_seconds: int = 120
    max_retries: int = 3


class DockerSettings(BaseModel):
    base_image: str = "python:3.10-slim"
    gpu_runtime: str | None = None
    memory_limit: str = "16g"
    timeout_seconds: int = 7200
    network_mode: str = "bridge"
    platform: str = "linux/arm64"


class PipelineSettings(BaseModel):
    max_iterations: int = 15
    max_fix_attempts: int = 5
    stagnation_window: int = 3
    honeymoon_iterations: int = 5
    budget_wall_clock_hours: float = 8.0
    budget_llm_calls: int = 500


class SupervisorSettings(BaseModel):
    red_lines: list[str] = Field(default_factory=lambda: list(RED_LINE_RULES))


# ---------------------------------------------------------------------------
# Top-level settings
# ---------------------------------------------------------------------------

_SECRETS_ENV_FILE = Path(
    "/Users/kumacmini/Library/CloudStorage/Dropbox/secrets/.env"
)


class Settings(BaseSettings):
    """Top-level application settings.

    Resolved from YAML files, environment variables (``AUTOSOTA_`` prefix),
    and the secrets ``.env`` file.
    """

    model_config = SettingsConfigDict(
        env_prefix="AUTOSOTA_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    llm: LLMSettings = Field(default_factory=LLMSettings)
    docker: DockerSettings = Field(default_factory=DockerSettings)
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)
    supervisor: SupervisorSettings = Field(default_factory=SupervisorSettings)
    workspace_root: Path = Path("./workspace")
    secrets_env_file: Path = _SECRETS_ENV_FILE


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into *base* (mutates *base*)."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r") as fh:
        data = yaml.safe_load(fh)
    return data if isinstance(data, dict) else {}


def load_settings(
    *,
    config_dir: Path | None = None,
    profile: str | None = None,
) -> Settings:
    """Build a ``Settings`` instance from YAML + env.

    Parameters
    ----------
    config_dir:
        Directory containing ``default.yaml`` and optional profile YAML.
        Defaults to ``<project_root>/configs``.
    profile:
        Name of the override profile (e.g. ``"m1_local"``).  The file
        ``<config_dir>/<profile>.yaml`` will be merged on top of defaults.
    """
    if config_dir is None:
        config_dir = PROJECT_ROOT / "configs"

    data: dict[str, Any] = _load_yaml(config_dir / "default.yaml")

    if profile is not None:
        override = _load_yaml(config_dir / f"{profile}.yaml")
        _deep_merge(data, override)

    return Settings(**data)
