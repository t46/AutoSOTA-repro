"""Helper functions for phase transitions and validation.

Provides utilities that the :class:`PipelineRunner` uses to determine
whether the pipeline can advance from one phase to the next, and to
validate that required artefacts exist before proceeding.
"""

from __future__ import annotations

from pathlib import Path

from autosota.pipeline.state import OptimizationPhase, PipelineState

# ---------------------------------------------------------------------------
# Phase ordering
# ---------------------------------------------------------------------------

_PHASE_ORDER: list[OptimizationPhase] = [
    OptimizationPhase.PHASE_0_BASELINE,
    OptimizationPhase.PHASE_1_UNDERSTANDING,
    OptimizationPhase.PHASE_2_IDEA_LIBRARY,
    OptimizationPhase.PHASE_3_OPTIMIZATION,
]


def next_phase(phase: OptimizationPhase) -> OptimizationPhase | None:
    """Return the phase that follows *phase*, or ``None`` if it is the last."""
    idx = _PHASE_ORDER.index(phase)
    if idx + 1 >= len(_PHASE_ORDER):
        return None
    return _PHASE_ORDER[idx + 1]


def is_terminal(phase: OptimizationPhase) -> bool:
    """Return True if the phase is the final one."""
    return phase == _PHASE_ORDER[-1]


# ---------------------------------------------------------------------------
# Phase-transition guards
# ---------------------------------------------------------------------------


def can_advance(state: PipelineState, workspace_dir: Path) -> tuple[bool, str]:
    """Check whether the pipeline can advance from its current phase.

    Returns ``(True, "")`` when the advance is allowed, or
    ``(False, reason)`` with a human-readable explanation otherwise.
    """
    phase = state.current_phase
    paper_dir = workspace_dir / state.paper_id

    if phase == OptimizationPhase.PHASE_0_BASELINE:
        # Need a baseline score
        if state.baseline_score is None:
            return False, "Baseline score has not been recorded yet."
        return True, ""

    if phase == OptimizationPhase.PHASE_1_UNDERSTANDING:
        # code_analysis.md must exist
        code_analysis = paper_dir / "code_analysis.md"
        if not code_analysis.exists():
            return False, "code_analysis.md has not been written."
        return True, ""

    if phase == OptimizationPhase.PHASE_2_IDEA_LIBRARY:
        # idea_library.md must exist
        idea_library = paper_dir / "idea_library.md"
        if not idea_library.exists():
            return False, "idea_library.md has not been written."
        # At least one hypothesis must be present
        if len(state.hypothesis_library.hypotheses) == 0:
            return False, "No hypotheses have been generated."
        return True, ""

    if phase == OptimizationPhase.PHASE_3_OPTIMIZATION:
        # Terminal phase -- cannot advance further
        return False, "Already in the final optimisation phase."

    return False, f"Unknown phase: {phase}"


def advance_phase(state: PipelineState, workspace_dir: Path) -> PipelineState:
    """Advance the pipeline to the next phase if allowed.

    Raises :class:`RuntimeError` if the transition guard fails.
    """
    ok, reason = can_advance(state, workspace_dir)
    if not ok:
        raise RuntimeError(f"Cannot advance from {state.current_phase}: {reason}")

    nxt = next_phase(state.current_phase)
    if nxt is None:
        raise RuntimeError("No next phase (already terminal).")

    state.current_phase = nxt
    return state


# ---------------------------------------------------------------------------
# Artefact validation
# ---------------------------------------------------------------------------


def validate_workspace(workspace_dir: Path, paper_id: str) -> list[str]:
    """Return a list of warnings about missing workspace artefacts.

    An empty list means everything expected is present.
    """
    paper_dir = workspace_dir / paper_id
    warnings: list[str] = []

    expected = [
        "pipeline_state.json",
        "scores.jsonl",
    ]
    for name in expected:
        if not (paper_dir / name).exists():
            warnings.append(f"Missing: {paper_dir / name}")

    return warnings
