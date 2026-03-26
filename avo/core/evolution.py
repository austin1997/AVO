"""Main continuous evolution loop for AVO.

Orchestrates the evolutionary process described in Section 3.3:
the AVO agent operates as a continuous loop that periodically produces
new solutions without human intervention. Each committed version is
persisted as a git commit along with its score.
"""

from __future__ import annotations

import importlib
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any

from avo.agent.knowledge_base import KnowledgeBase
from avo.agent.llm_client import create_llm_client
from avo.agent.variation_operator import AVOAgent
from avo.config import EvolutionConfig
from avo.core.population import Population
from avo.core.scoring import ScoringFunction
from avo.core.types import Solution
from avo.persistence.git_backend import GitBackend
from avo.supervisor.self_supervision import SelfSupervisor
from avo.utils.logging import setup_logging

logger = logging.getLogger(__name__)


class EvolutionRunner:
    """Orchestrates continuous AVO evolution.

    Manages the lifecycle: seed initialization, variation loop, self-supervision,
    git persistence, and graceful shutdown.
    """

    def __init__(self, config: EvolutionConfig) -> None:
        self._config = config
        self._shutdown_requested = False

        setup_logging(
            log_dir=config.log_path(),
            level=config.log_level,
            json_logs=True,
        )

        self._workspace = config.workspace_path()
        self._workspace.mkdir(parents=True, exist_ok=True)

        self._llm = create_llm_client(config.llm)
        self._kb = self._load_knowledge_base()
        self._scoring_fn = self._load_scoring_function()
        self._population = Population(self._workspace)
        self._git = GitBackend(self._workspace) if config.git_persist else None
        self._supervisor = SelfSupervisor(config.supervisor, llm_client=self._llm)

        self._agent = AVOAgent(
            config=config,
            population=self._population,
            scoring_fn=self._scoring_fn,
            knowledge_base=self._kb,
            llm_client=self._llm,
        )

    def _load_knowledge_base(self) -> KnowledgeBase:
        kb_path = self._config.knowledge_base_path()
        if kb_path and kb_path.exists():
            return KnowledgeBase.from_directory(kb_path)
        return KnowledgeBase()

    def _load_scoring_function(self) -> ScoringFunction:
        """Dynamically load the scoring function from config."""
        module_path = self._config.scoring_module
        class_name = self._config.scoring_class

        if not module_path:
            raise ValueError("scoring_module must be set in config")

        spec_path = Path(module_path)
        if spec_path.exists():
            spec = importlib.util.spec_from_file_location("scorer_module", spec_path)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return getattr(mod, class_name)(workspace_dir=self._workspace)

        mod = importlib.import_module(module_path)
        return getattr(mod, class_name)(workspace_dir=self._workspace)

    def initialize_seed(self) -> Solution:
        """Load the seed program and commit it as v0."""
        seed_file = self._config.seed_file
        if not seed_file:
            raise ValueError("seed_file must be set in config")

        seed_path = Path(seed_file)
        if not seed_path.is_absolute():
            seed_path = Path.cwd() / seed_path
        if not seed_path.exists():
            raise FileNotFoundError(f"Seed file not found: {seed_path}")

        source_code = seed_path.read_text()

        sol_path = self._workspace / self._config.solution_file
        sol_path.parent.mkdir(parents=True, exist_ok=True)
        sol_path.write_text(source_code)

        seed = self._population.initialize_seed(source_code, self._config.solution_file)

        if self._git:
            commit_hash = self._git.persist(seed, self._population.best_score)
            seed.commit_hash = commit_hash

        logger.info("Initialized seed from %s", seed_path)
        return seed

    def run(self) -> None:
        """Run the continuous evolution loop.

        Runs until one of:
        - max_versions reached
        - max_wall_time_hours exceeded
        - SIGINT/SIGTERM received
        """
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        if self._population.size == 0:
            self.initialize_seed()

        start_time = time.time()
        max_seconds = self._config.max_wall_time_hours * 3600

        logger.info(
            "Starting AVO evolution: max_versions=%d, max_hours=%.1f",
            self._config.max_versions,
            self._config.max_wall_time_hours,
        )

        while not self._should_stop(start_time, max_seconds):
            redirect_direction = None
            if self._supervisor.should_intervene(self._population):
                directions = self._supervisor.get_redirect_directions(self._population)
                if directions:
                    redirect_direction = directions[0]
                    logger.info("Supervisor redirecting: %s", redirect_direction[:100])

            committed = self._agent.variation_step(redirect_direction)
            self._supervisor.record_attempt(committed is not None)

            if committed:
                if self._git:
                    commit_hash = self._git.persist(committed, self._population.best_score)
                    committed.commit_hash = commit_hash

                state_path = self._workspace / ".avo_population.json"
                self._population.save_state(state_path)

                elapsed = time.time() - start_time
                logger.info(
                    "Evolution progress: v%d committed, "
                    "geomean=%.4f, elapsed=%.1fh, "
                    "supervisor_interventions=%d",
                    committed.version,
                    self._population.best_score.geomean,
                    elapsed / 3600,
                    self._supervisor.intervention_count,
                )

        self._finalize()

    def _should_stop(self, start_time: float, max_seconds: float) -> bool:
        if self._shutdown_requested:
            logger.info("Shutdown requested")
            return True
        if self._population.current_version >= self._config.max_versions:
            logger.info("Reached max versions: %d", self._config.max_versions)
            return True
        elapsed = time.time() - start_time
        if elapsed >= max_seconds:
            logger.info("Reached max wall time: %.1fh", elapsed / 3600)
            return True
        return False

    def _handle_signal(self, signum: int, frame: Any) -> None:
        logger.info("Received signal %d, requesting graceful shutdown", signum)
        self._shutdown_requested = True

    def _finalize(self) -> None:
        """Log final results and clean up."""
        best = self._population.best_entry
        logger.info("=== Evolution Complete ===")
        logger.info("Total committed versions: %d", self._population.size)
        if best:
            logger.info("Best version: v%d, geomean=%.4f", best.solution.version, best.score.geomean)
            for cfg, val in sorted(best.score.values.items()):
                logger.info("  %s: %.4f", cfg, val)
        logger.info("Supervisor interventions: %d", self._supervisor.intervention_count)

        self._agent.close()

    def get_results(self) -> dict[str, Any]:
        """Return a summary of the evolution results."""
        best = self._population.best_entry
        return {
            "total_versions": self._population.size,
            "best_version": best.solution.version if best else None,
            "best_geomean": best.score.geomean if best else 0.0,
            "best_scores": best.score.values if best else {},
            "supervisor_interventions": self._supervisor.intervention_count,
            "agent_steps": self._agent.step_count,
        }


def run_evolution(config: EvolutionConfig) -> dict[str, Any]:
    """Convenience function to run a full evolution from config."""
    runner = EvolutionRunner(config)
    runner.run()
    return runner.get_results()


def run_from_yaml(config_path: str) -> dict[str, Any]:
    """Run evolution from a YAML config file."""
    config = EvolutionConfig.from_yaml(config_path)
    return run_evolution(config)
