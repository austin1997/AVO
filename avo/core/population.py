"""Population / lineage management for AVO.

Implements P_t = {(x1, f(x1)), ..., (xt, f(xt))} from the paper.
Currently supports single-lineage mode (the configuration studied in the paper).
Architecture supports future extension to island-based or MAP-Elites archives.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from avo.core.types import Lineage, LineageEntry, Score, Solution

logger = logging.getLogger(__name__)


class Population:
    """Single-lineage population manager.

    Maintains a linear chain of committed improvements. Only persists
    versions that pass correctness and match/improve the best score.
    """

    def __init__(self, workspace_dir: Path | str | None = None) -> None:
        self._lineage = Lineage()
        self._failed_attempts: list[dict[str, Any]] = []
        self._workspace_dir = Path(workspace_dir) if workspace_dir else None

    @property
    def lineage(self) -> Lineage:
        return self._lineage

    @property
    def size(self) -> int:
        return len(self._lineage)

    @property
    def best_score(self) -> Score:
        return self._lineage.best_score

    @property
    def best_entry(self) -> LineageEntry | None:
        return self._lineage.best_entry

    @property
    def latest_entry(self) -> LineageEntry | None:
        return self._lineage.latest_entry

    @property
    def current_version(self) -> int:
        return self._lineage.current_version

    @property
    def failed_attempt_count(self) -> int:
        return len(self._failed_attempts)

    def initialize_seed(self, source_code: str, source_file: str = "solution.py") -> Solution:
        """Create and commit the seed solution (v0) with a zero score."""
        seed = Solution(
            source_code=source_code,
            version=0,
            source_file=source_file,
        )
        seed_score = Score(passes_correctness=True, correctness_message="seed")
        self._lineage.add(seed, seed_score)
        logger.info("Initialized seed solution v0")
        return seed

    def try_commit(self, source_code: str, score: Score, metadata: dict[str, Any] | None = None) -> Solution | None:
        """Attempt to commit a new solution.

        Only commits if the candidate passes correctness and its geomean
        matches or improves the best committed version (paper Section 3.2).

        Returns the committed Solution on success, None on rejection.
        """
        next_version = self.current_version + 1

        if not score.passes_correctness:
            self._record_failed_attempt(source_code, score, "correctness_failure")
            logger.info("v%d rejected: failed correctness check", next_version)
            return None

        if self.size > 1 and not score.improves_over(self.best_score):
            self._record_failed_attempt(source_code, score, "no_improvement")
            logger.info(
                "v%d rejected: geomean %.4f <= best %.4f",
                next_version,
                score.geomean,
                self.best_score.geomean,
            )
            return None

        solution = Solution(
            source_code=source_code,
            version=next_version,
            source_file=self.latest_entry.solution.source_file if self.latest_entry else "solution.py",
            parent_version=self.current_version if self.current_version >= 0 else None,
            metadata=metadata or {},
        )
        self._lineage.add(solution, score)
        self._failed_attempts.clear()
        logger.info(
            "Committed v%d: geomean=%.4f (prev best=%.4f)",
            next_version,
            score.geomean,
            self.best_score.geomean,
        )
        return solution

    def _record_failed_attempt(self, source_code: str, score: Score, reason: str) -> None:
        self._failed_attempts.append({
            "source_code_len": len(source_code),
            "score": score.to_dict(),
            "reason": reason,
        })

    def recent_entries(self, n: int = 5) -> list[LineageEntry]:
        return self._lineage.recent(n)

    def get_entry(self, version: int) -> LineageEntry | None:
        for entry in self._lineage.entries:
            if entry.solution.version == version:
                return entry
        return None

    def summary(self) -> str:
        return self._lineage.summary()

    def save_state(self, path: Path | str) -> None:
        """Persist the full lineage to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "lineage": self._lineage.to_list(),
            "failed_attempt_count": len(self._failed_attempts),
        }
        path.write_text(json.dumps(data, indent=2))

    def load_state(self, path: Path | str) -> None:
        """Restore lineage from a JSON file."""
        path = Path(path)
        if not path.exists():
            return
        data = json.loads(path.read_text())
        self._lineage = Lineage.from_list(data.get("lineage", []))
        logger.info("Loaded lineage with %d entries", len(self._lineage))
