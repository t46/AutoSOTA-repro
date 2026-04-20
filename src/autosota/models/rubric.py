"""Data models for evaluation rubrics.

A rubric is a tree of criteria used to verify that a paper's code has been
correctly reproduced and that improvements do not violate constraints.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from autosota.constants import MetricDirection


class RubricNode(BaseModel):
    """A single node in the evaluation rubric tree.

    Leaf nodes carry a ``pass_fail`` verdict; internal nodes aggregate their
    children according to ``weight``.
    """

    name: str
    description: str = ""
    weight: float = 1.0
    pass_fail: bool | None = None
    details: str = ""
    children: list[RubricNode] = Field(default_factory=list)

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def weighted_score(self) -> float:
        """Return a normalised score in [0, 1].

        Leaf nodes return 1.0 if passing, 0.0 otherwise.
        Internal nodes return the weighted average of their children.
        """
        if self.is_leaf():
            if self.pass_fail is None:
                return 0.0
            return 1.0 if self.pass_fail else 0.0

        total_weight = sum(c.weight for c in self.children)
        if total_weight == 0.0:
            return 0.0
        return sum(c.weight * c.weighted_score() for c in self.children) / total_weight


class RubricTree(BaseModel):
    """Root of a rubric evaluation tree, tied to a specific target metric."""

    root: RubricNode
    target_metric: str = ""
    metric_direction: MetricDirection = MetricDirection.HIGHER_IS_BETTER
    notes: str = ""

    def overall_score(self) -> float:
        return self.root.weighted_score()
