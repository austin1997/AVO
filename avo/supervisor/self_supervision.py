"""Self-supervision mechanism for AVO.

Implements Section 3.3's stagnation detection: monitors the evolution
trajectory for stalling and unproductive cycles, and intervenes by
reviewing the trajectory and steering toward fresh optimization directions.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from avo.agent.llm_client import BaseLLMClient
from avo.agent.prompts import SUPERVISOR_REDIRECT_PROMPT
from avo.config import SupervisorConfig
from avo.core.population import Population

logger = logging.getLogger(__name__)


class SelfSupervisor:
    """Detects stagnation and redirects exploration.

    Two failure modes are monitored:
    1. Stalling: no new commits for max_failed_attempts consecutive attempts
    2. Unproductive cycles: geomean has not improved over cycle_detection_window
       committed versions
    """

    def __init__(self, config: SupervisorConfig, llm_client: BaseLLMClient | None = None) -> None:
        self._config = config
        self._llm = llm_client
        self._consecutive_failures = 0
        self._last_committed_version = -1
        self._intervention_count = 0

    def record_attempt(self, committed: bool) -> None:
        """Record the outcome of a variation step."""
        if committed:
            self._consecutive_failures = 0
        else:
            self._consecutive_failures += 1

    def should_intervene(self, population: Population) -> bool:
        """Check whether the supervisor should redirect exploration."""
        if not self._config.enabled:
            return False

        if self._consecutive_failures >= self._config.max_failed_attempts:
            logger.info(
                "Stagnation detected: %d consecutive failures",
                self._consecutive_failures,
            )
            return True

        if population.size >= self._config.cycle_detection_window:
            recent = population.recent_entries(self._config.cycle_detection_window)
            geomeans = [e.score.geomean for e in recent]
            if len(geomeans) >= 2 and max(geomeans) - min(geomeans) < 0.001:
                logger.info("Stagnation detected: no improvement over last %d versions", len(geomeans))
                return True

        return False

    def get_redirect_directions(self, population: Population, num_directions: int = 3) -> list[str]:
        """Generate fresh optimization directions via LLM or heuristics.

        If an LLM client is available and redirect_with_llm is enabled,
        asks the LLM to analyze the trajectory and propose new directions.
        Otherwise, falls back to heuristic suggestions.
        """
        self._intervention_count += 1
        self._consecutive_failures = 0

        if self._llm and self._config.redirect_with_llm:
            return self._llm_redirect(population, num_directions)
        return self._heuristic_redirect(population, num_directions)

    def _llm_redirect(self, population: Population, num_directions: int) -> list[str]:
        """Use the LLM to analyze the trajectory and propose directions."""
        stagnation_reason = (
            f"{self._consecutive_failures} consecutive failed attempts "
            f"without producing a new committed version."
        )

        prompt = SUPERVISOR_REDIRECT_PROMPT.format(
            stagnation_reason=stagnation_reason,
            lineage_summary=population.summary(),
            failed_attempt_count=population.failed_attempt_count,
            num_directions=num_directions,
        )

        try:
            response = self._llm.chat([
                {"role": "system", "content": "You are a technical advisor for code optimization."},
                {"role": "user", "content": prompt},
            ])
            content = response.get("content", "")
            directions = self._parse_directions(content, num_directions)
            if directions:
                logger.info("Supervisor generated %d redirect directions via LLM", len(directions))
                return directions
        except Exception as e:
            logger.warning("LLM redirect failed: %s, falling back to heuristics", e)

        return self._heuristic_redirect(population, num_directions)

    def _heuristic_redirect(self, population: Population, num_directions: int) -> list[str]:
        """Generate directions from predefined heuristics."""
        directions = [
            "Try a fundamentally different algorithmic approach. Review the knowledge base "
            "for alternative algorithms or data structures that could replace the current approach.",

            "Focus on memory access patterns and cache efficiency. Profile the current solution "
            "to identify cache misses or unnecessary memory allocations, then optimize data layout.",

            "Look for parallelism opportunities. Identify independent computations that could "
            "be restructured for better concurrent execution or vectorization.",

            "Simplify the hot path. Remove unnecessary branches, function calls, or abstractions "
            "in the performance-critical section. Inline operations where beneficial.",

            "Revisit an earlier version that took a different approach. Compare its structure "
            "with the current best and combine the strengths of both.",
        ]
        return directions[:num_directions]

    @staticmethod
    def _parse_directions(content: str, max_directions: int) -> list[str]:
        """Parse numbered directions from LLM output."""
        directions: list[str] = []
        current: list[str] = []

        for line in content.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            is_new = False
            for prefix in ("1.", "2.", "3.", "4.", "5.", "- ", "* "):
                if stripped.startswith(prefix):
                    is_new = True
                    break

            if is_new and current:
                directions.append(" ".join(current))
                current = [stripped.lstrip("0123456789.-* ")]
            elif is_new:
                current = [stripped.lstrip("0123456789.-* ")]
            else:
                current.append(stripped)

        if current:
            directions.append(" ".join(current))

        return directions[:max_directions]

    @property
    def intervention_count(self) -> int:
        return self._intervention_count

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures
