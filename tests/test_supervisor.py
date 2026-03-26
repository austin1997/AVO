"""Tests for avo.supervisor.self_supervision."""

import pytest

from avo.config import SupervisorConfig
from avo.core.population import Population
from avo.core.types import Score
from avo.supervisor.self_supervision import SelfSupervisor


class TestSelfSupervisor:
    def test_no_intervention_when_disabled(self):
        config = SupervisorConfig(enabled=False)
        sv = SelfSupervisor(config)
        pop = Population()
        pop.initialize_seed("v0")

        for _ in range(20):
            sv.record_attempt(False)
        assert not sv.should_intervene(pop)

    def test_intervention_on_stall(self):
        config = SupervisorConfig(enabled=True, max_failed_attempts=3)
        sv = SelfSupervisor(config)
        pop = Population()
        pop.initialize_seed("v0")

        sv.record_attempt(False)
        sv.record_attempt(False)
        assert not sv.should_intervene(pop)

        sv.record_attempt(False)
        assert sv.should_intervene(pop)

    def test_reset_on_success(self):
        config = SupervisorConfig(enabled=True, max_failed_attempts=3)
        sv = SelfSupervisor(config)
        pop = Population()
        pop.initialize_seed("v0")

        sv.record_attempt(False)
        sv.record_attempt(False)
        sv.record_attempt(True)
        assert sv.consecutive_failures == 0
        assert not sv.should_intervene(pop)

    def test_heuristic_directions(self):
        config = SupervisorConfig(enabled=True, max_failed_attempts=2)
        sv = SelfSupervisor(config)
        pop = Population()
        pop.initialize_seed("v0")

        sv.record_attempt(False)
        sv.record_attempt(False)
        assert sv.should_intervene(pop)

        directions = sv.get_redirect_directions(pop, num_directions=3)
        assert len(directions) == 3
        assert sv.intervention_count == 1
        assert sv.consecutive_failures == 0

    def test_intervention_on_plateau(self):
        config = SupervisorConfig(enabled=True, cycle_detection_window=3)
        sv = SelfSupervisor(config)
        pop = Population()
        pop.initialize_seed("v0")

        s = Score(values={"a": 10.0}, passes_correctness=True)
        pop.try_commit("v1", s)
        s2 = Score(values={"a": 10.0001}, passes_correctness=True)
        pop.try_commit("v2", s2)
        s3 = Score(values={"a": 10.0002}, passes_correctness=True)
        pop.try_commit("v3", s3)

        assert sv.should_intervene(pop)
