"""Tests for avo.core.scoring (interface contract)."""

import pytest

from avo.core.scoring import ScoringFunction
from avo.core.types import Score


class DummyScorer(ScoringFunction):
    """Minimal scorer for testing the interface."""

    def evaluate(self, source_code: str, source_file: str = "solution.py") -> Score:
        if "error" in source_code:
            return Score(passes_correctness=False, correctness_message="Contains error")
        return Score(
            values={"test": float(len(source_code))},
            passes_correctness=True,
            correctness_message="OK",
        )

    def get_configurations(self) -> list[str]:
        return ["test"]


class TestScoringInterface:
    def test_evaluate_correct(self):
        scorer = DummyScorer()
        score = scorer.evaluate("hello world")
        assert score.passes_correctness
        assert score.values["test"] == 11.0

    def test_evaluate_incorrect(self):
        scorer = DummyScorer()
        score = scorer.evaluate("has error in code")
        assert not score.passes_correctness

    def test_get_configurations(self):
        scorer = DummyScorer()
        assert scorer.get_configurations() == ["test"]

    def test_scoring_context(self):
        scorer = DummyScorer()
        ctx = scorer.get_scoring_context()
        assert "1 configurations" in ctx
