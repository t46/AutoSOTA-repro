"""Pipeline state management.

``PipelineState`` is the persistent checkpoint of the optimisation loop.
It is serialised to / deserialised from JSON so the pipeline can be
resumed after interruptions.
"""

from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from autosota.models.hypothesis import HypothesisLibrary
from autosota.models.iteration import ScoreEntry
from autosota.models.repair import FailureMemory


class OptimizationPhase(str, Enum):
    PHASE_0_BASELINE = "phase_0_baseline"
    PHASE_1_UNDERSTANDING = "phase_1_understanding"
    PHASE_2_IDEA_LIBRARY = "phase_2_idea_library"
    PHASE_3_OPTIMIZATION = "phase_3_optimization"


class PipelineState(BaseModel):
    """Full snapshot of a running optimisation pipeline.

    This is the single source of truth that gets checkpointed after every
    iteration so the pipeline can resume from any point.
    """

    paper_id: str
    current_phase: OptimizationPhase = OptimizationPhase.PHASE_0_BASELINE
    current_iteration: int = 0

    # Scores
    baseline_score: float | None = None
    best_score: float | None = None
    score_history: list[ScoreEntry] = Field(default_factory=list)

    # Hypotheses
    hypothesis_library: HypothesisLibrary = Field(default_factory=HypothesisLibrary)

    # Failure memory
    failure_memory: FailureMemory = Field(default_factory=FailureMemory)

    # Budget tracking
    llm_calls_used: int = 0
    wall_clock_start: datetime | None = None
    wall_clock_seconds: float = 0.0

    # Git tracking
    baseline_git_hash: str = ""
    current_git_hash: str = ""

    # Flags
    is_complete: bool = False
    abort_reason: str = ""

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _state_path(self, workspace_dir: Path) -> Path:
        return workspace_dir / self.paper_id / "pipeline_state.json"

    def save(self, workspace_dir: Path) -> None:
        """Serialise state to ``<workspace_dir>/<paper_id>/pipeline_state.json``."""
        path = self._state_path(workspace_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, workspace_dir: Path, paper_id: str) -> PipelineState:
        """Load state from disk.

        Raises ``FileNotFoundError`` if no checkpoint exists.
        """
        path = workspace_dir / paper_id / "pipeline_state.json"
        if not path.exists():
            raise FileNotFoundError(f"No pipeline state found at {path}")
        data = json.loads(path.read_text())
        return cls.model_validate(data)
