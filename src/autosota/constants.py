"""Constants for the AutoSOTA system.

Red-line rules, pipeline phase names, hypothesis types, and other
project-wide constants live here so every module can import a single
source of truth.
"""

from __future__ import annotations

from enum import Enum

# ---------------------------------------------------------------------------
# Red-line rules  (integrity constraints that must never be violated)
# ---------------------------------------------------------------------------

RED_LINE_RULES: list[str] = [
    "R1: Evaluation metric parameters must not change",
    "R2: Evaluation script integrity must be maintained",
    "R3: Integrity of algorithm outputs must not be violated",
    "R4: No unfair trade-offs between metric dimensions",
    "R5: Dataset integrity must be preserved",
    "R6: No modification of the dataset",
    "R7: Paper-specific constraints must be explicitly identified and followed",
]

# ---------------------------------------------------------------------------
# Pipeline phases
# ---------------------------------------------------------------------------


class PhaseName(str, Enum):
    """Human-readable names for the four optimization phases."""

    PHASE_0_BASELINE = "Phase 0: Baseline Reproduction"
    PHASE_1_UNDERSTANDING = "Phase 1: Deep Understanding"
    PHASE_2_IDEA_LIBRARY = "Phase 2: Idea Library Construction"
    PHASE_3_OPTIMIZATION = "Phase 3: Iterative Optimization"


# ---------------------------------------------------------------------------
# Hypothesis types
# ---------------------------------------------------------------------------


class HypothesisType(str, Enum):
    """Categories of improvement hypotheses."""

    PARAM = "param"
    CODE = "code"
    ALGO = "algo"


# ---------------------------------------------------------------------------
# Risk levels
# ---------------------------------------------------------------------------


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ---------------------------------------------------------------------------
# Metric direction
# ---------------------------------------------------------------------------


class MetricDirection(str, Enum):
    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"


# ---------------------------------------------------------------------------
# Execution phases
# ---------------------------------------------------------------------------


class ExecutionPhase(str, Enum):
    SETUP = "setup"
    INSTALL = "install"
    LAUNCH = "launch"
    EVAL = "eval"
    REPORT = "report"
    FAILURE = "failure"


# ---------------------------------------------------------------------------
# Red-line verdicts
# ---------------------------------------------------------------------------


class RedLineVerdict(str, Enum):
    CLEARED = "cleared"
    REJECTED = "rejected"


# ---------------------------------------------------------------------------
# Conferences
# ---------------------------------------------------------------------------


class Conference(str, Enum):
    NEURIPS = "NeurIPS"
    ICML = "ICML"
    ICLR = "ICLR"
    ACL = "ACL"
    AAAI = "AAAI"
    NAACL = "NAACL"
    ICCV = "ICCV"
    CVPR = "CVPR"
