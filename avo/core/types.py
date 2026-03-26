"""Core data types for the AVO evolutionary search framework.

Implements the fundamental structures from the AVO paper (arXiv:2603.24517):
- Solution: a candidate program x_i
- Score: multi-dimensional score vector f(x) = (f1(x), ..., fn(x))
- LineageEntry: a (solution, score) pair in the population
- Lineage: the full ordered sequence P_t
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Score:
    """Multi-dimensional score vector f(x) = (f1(x), ..., fn(x)).

    A candidate that fails correctness gets all-zero values.
    Each dimension corresponds to a benchmark configuration.
    """

    values: dict[str, float] = field(default_factory=dict)
    passes_correctness: bool = False
    correctness_message: str = ""

    @property
    def geomean(self) -> float:
        """Geometric mean across all configurations (paper's primary metric)."""
        if not self.values or not self.passes_correctness:
            return 0.0
        vals = [v for v in self.values.values() if v > 0]
        if not vals:
            return 0.0
        product = 1.0
        for v in vals:
            product *= v
        return product ** (1.0 / len(vals))

    @property
    def is_zero(self) -> bool:
        return not self.passes_correctness or all(
            v == 0.0 for v in self.values.values()
        )

    def dominates(self, other: Score) -> bool:
        """True if this score is >= other on all configs and > on at least one."""
        if self.is_zero:
            return False
        if other.is_zero:
            return True
        at_least_one_better = False
        for key in self.values:
            if key not in other.values:
                continue
            if self.values[key] < other.values[key]:
                return False
            if self.values[key] > other.values[key]:
                at_least_one_better = True
        return at_least_one_better

    def improves_over(self, other: Score) -> bool:
        """True if geomean improved (the paper's commit criterion)."""
        return self.geomean > other.geomean

    def to_dict(self) -> dict[str, Any]:
        return {
            "values": self.values,
            "passes_correctness": self.passes_correctness,
            "correctness_message": self.correctness_message,
            "geomean": self.geomean,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Score:
        return cls(
            values=data.get("values", {}),
            passes_correctness=data.get("passes_correctness", False),
            correctness_message=data.get("correctness_message", ""),
        )

    def __repr__(self) -> str:
        return (
            f"Score(geomean={self.geomean:.4f}, "
            f"correct={self.passes_correctness}, "
            f"configs={len(self.values)})"
        )


@dataclass
class Solution:
    """A candidate program in the evolutionary search.

    In AVO, each x_i is a source code implementation (e.g. a CUDA kernel,
    a Python function) along with metadata about its origin.
    """

    source_code: str
    version: int = 0
    source_file: str = "solution.py"
    commit_hash: str = ""
    parent_version: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "source_file": self.source_file,
            "commit_hash": self.commit_hash,
            "parent_version": self.parent_version,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "source_code": self.source_code,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Solution:
        return cls(
            source_code=data.get("source_code", ""),
            version=data.get("version", 0),
            source_file=data.get("source_file", "solution.py"),
            commit_hash=data.get("commit_hash", ""),
            parent_version=data.get("parent_version"),
            metadata=data.get("metadata", {}),
            timestamp=data.get("timestamp", time.time()),
        )

    def __repr__(self) -> str:
        return (
            f"Solution(v{self.version}, "
            f"file={self.source_file}, "
            f"len={len(self.source_code)})"
        )


@dataclass
class LineageEntry:
    """A (solution, score) pair in the population P_t."""

    solution: Solution
    score: Score

    def to_dict(self) -> dict[str, Any]:
        return {
            "solution": self.solution.to_dict(),
            "score": self.score.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LineageEntry:
        return cls(
            solution=Solution.from_dict(data["solution"]),
            score=Score.from_dict(data["score"]),
        )


@dataclass
class Lineage:
    """Ordered sequence of committed (solution, score) pairs: P_t.

    In AVO's single-lineage mode, this is a linear chain of improvements.
    """

    entries: list[LineageEntry] = field(default_factory=list)

    @property
    def current_version(self) -> int:
        if not self.entries:
            return -1
        return self.entries[-1].solution.version

    @property
    def best_entry(self) -> LineageEntry | None:
        if not self.entries:
            return None
        return max(self.entries, key=lambda e: e.score.geomean)

    @property
    def best_score(self) -> Score:
        entry = self.best_entry
        if entry is None:
            return Score()
        return entry.score

    @property
    def latest_entry(self) -> LineageEntry | None:
        if not self.entries:
            return None
        return self.entries[-1]

    def add(self, solution: Solution, score: Score) -> None:
        self.entries.append(LineageEntry(solution=solution, score=score))

    def recent(self, n: int = 5) -> list[LineageEntry]:
        return self.entries[-n:]

    def summary(self) -> str:
        """Human-readable summary of the lineage for agent context."""
        if not self.entries:
            return "No solutions committed yet."
        lines = [
            f"Lineage: {len(self.entries)} committed versions",
            f"Best geomean: {self.best_score.geomean:.4f}",
            "",
        ]
        for entry in self.entries:
            marker = " *BEST*" if entry == self.best_entry else ""
            lines.append(
                f"  v{entry.solution.version}: "
                f"geomean={entry.score.geomean:.4f}{marker}"
            )
            for cfg, val in sorted(entry.score.values.items()):
                lines.append(f"    {cfg}: {val:.4f}")
        return "\n".join(lines)

    def to_list(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self.entries]

    @classmethod
    def from_list(cls, data: list[dict[str, Any]]) -> Lineage:
        return cls(entries=[LineageEntry.from_dict(d) for d in data])

    def __len__(self) -> int:
        return len(self.entries)
