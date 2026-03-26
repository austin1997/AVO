"""Tests for avo.core.population."""

import json
import tempfile
from pathlib import Path

import pytest

from avo.core.population import Population
from avo.core.types import Score


class TestPopulation:
    def test_initialize_seed(self):
        pop = Population()
        seed = pop.initialize_seed("def f(): pass")
        assert seed.version == 0
        assert pop.size == 1
        assert pop.current_version == 0

    def test_commit_success(self):
        pop = Population()
        pop.initialize_seed("v0")

        score = Score(values={"a": 10.0}, passes_correctness=True)
        sol = pop.try_commit("v1", score)
        assert sol is not None
        assert sol.version == 1
        assert pop.size == 2

    def test_commit_reject_correctness(self):
        pop = Population()
        pop.initialize_seed("v0")

        score = Score(values={"a": 10.0}, passes_correctness=False)
        sol = pop.try_commit("bad", score)
        assert sol is None
        assert pop.size == 1

    def test_commit_reject_no_improvement(self):
        pop = Population()
        pop.initialize_seed("v0")

        good = Score(values={"a": 20.0}, passes_correctness=True)
        pop.try_commit("v1", good)

        worse = Score(values={"a": 10.0}, passes_correctness=True)
        sol = pop.try_commit("v2_worse", worse)
        assert sol is None
        assert pop.size == 2

    def test_commit_accept_improvement(self):
        pop = Population()
        pop.initialize_seed("v0")

        s1 = Score(values={"a": 10.0}, passes_correctness=True)
        pop.try_commit("v1", s1)

        s2 = Score(values={"a": 20.0}, passes_correctness=True)
        sol = pop.try_commit("v2", s2)
        assert sol is not None
        assert sol.version == 2
        assert pop.best_score.geomean == 20.0

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pop = Population()
            pop.initialize_seed("seed_code")
            s = Score(values={"a": 5.0}, passes_correctness=True)
            pop.try_commit("v1_code", s)

            path = Path(tmpdir) / "state.json"
            pop.save_state(path)

            pop2 = Population()
            pop2.load_state(path)
            assert pop2.size == 2
            assert pop2.best_score.geomean == 5.0

    def test_failed_attempt_tracking(self):
        pop = Population()
        pop.initialize_seed("v0")
        s1 = Score(values={"a": 10.0}, passes_correctness=True)
        pop.try_commit("v1", s1)

        bad = Score(values={"a": 5.0}, passes_correctness=True)
        pop.try_commit("bad1", bad)
        pop.try_commit("bad2", bad)
        assert pop.failed_attempt_count == 2

        better = Score(values={"a": 20.0}, passes_correctness=True)
        pop.try_commit("v2", better)
        assert pop.failed_attempt_count == 0
