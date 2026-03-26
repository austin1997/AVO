"""Abstract scoring function interface for AVO.

Users implement this per-domain to define correctness checking
and performance evaluation for their optimization target.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from avo.core.types import Score, Solution


class ScoringFunction(ABC):
    """Abstract base class for domain-specific scoring.

    The scoring function f evaluates candidates along two dimensions:
    1. Numerical correctness against a reference implementation
    2. Performance (throughput / latency / quality) on the target

    f(x) = (f1(x), f2(x), ..., fn(x)) is an n-dimensional vector where
    fj represents the score for test configuration j.
    A candidate that fails correctness is assigned zero score.
    """

    @abstractmethod
    def evaluate(self, source_code: str, source_file: str = "solution.py") -> Score:
        """Evaluate a candidate solution.

        This should:
        1. Write the source code to a file
        2. Check correctness against a reference
        3. If correct, measure performance across all configurations
        4. Return a Score with per-configuration values

        Returns Score with passes_correctness=False and zero values on failure.
        """

    @abstractmethod
    def get_configurations(self) -> list[str]:
        """Return the list of benchmark configuration names."""

    def get_reference_description(self) -> str:
        """Optional: return a description of the reference implementation."""
        return ""

    def get_scoring_context(self) -> str:
        """Return a human-readable description of what is being optimized.

        This is provided to the agent so it understands the optimization target.
        """
        configs = self.get_configurations()
        return (
            f"Scoring function with {len(configs)} configurations: "
            + ", ".join(configs)
        )
