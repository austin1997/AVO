"""Tests for avo.core.types."""

import pytest

from avo.core.types import Lineage, LineageEntry, Score, Solution


class TestScore:
    def test_zero_score(self):
        s = Score()
        assert s.is_zero
        assert s.geomean == 0.0
        assert not s.passes_correctness

    def test_geomean_single(self):
        s = Score(values={"a": 100.0}, passes_correctness=True)
        assert s.geomean == 100.0

    def test_geomean_multiple(self):
        s = Score(values={"a": 4.0, "b": 16.0}, passes_correctness=True)
        assert abs(s.geomean - 8.0) < 1e-6

    def test_geomean_zero_when_incorrect(self):
        s = Score(values={"a": 100.0}, passes_correctness=False)
        assert s.geomean == 0.0

    def test_dominates(self):
        s1 = Score(values={"a": 10.0, "b": 20.0}, passes_correctness=True)
        s2 = Score(values={"a": 8.0, "b": 15.0}, passes_correctness=True)
        assert s1.dominates(s2)
        assert not s2.dominates(s1)

    def test_dominates_equal_not_dominant(self):
        s1 = Score(values={"a": 10.0, "b": 20.0}, passes_correctness=True)
        s2 = Score(values={"a": 10.0, "b": 20.0}, passes_correctness=True)
        assert not s1.dominates(s2)

    def test_improves_over(self):
        s1 = Score(values={"a": 20.0}, passes_correctness=True)
        s2 = Score(values={"a": 10.0}, passes_correctness=True)
        assert s1.improves_over(s2)
        assert not s2.improves_over(s1)

    def test_serialization(self):
        s = Score(values={"x": 42.0}, passes_correctness=True, correctness_message="OK")
        d = s.to_dict()
        s2 = Score.from_dict(d)
        assert s2.values == s.values
        assert s2.passes_correctness == s.passes_correctness
        assert s2.correctness_message == s.correctness_message


class TestSolution:
    def test_creation(self):
        sol = Solution(source_code="print('hello')", version=1)
        assert sol.version == 1
        assert sol.source_code == "print('hello')"

    def test_serialization(self):
        sol = Solution(
            source_code="x = 1", version=3, source_file="test.py",
            parent_version=2, metadata={"note": "test"},
        )
        d = sol.to_dict()
        sol2 = Solution.from_dict(d)
        assert sol2.version == sol.version
        assert sol2.source_code == sol.source_code
        assert sol2.parent_version == sol.parent_version
        assert sol2.metadata == sol.metadata


class TestLineage:
    def test_empty(self):
        lin = Lineage()
        assert len(lin) == 0
        assert lin.current_version == -1
        assert lin.best_entry is None

    def test_add_and_query(self):
        lin = Lineage()
        sol1 = Solution(source_code="v1", version=1)
        score1 = Score(values={"a": 10.0}, passes_correctness=True)
        lin.add(sol1, score1)

        sol2 = Solution(source_code="v2", version=2)
        score2 = Score(values={"a": 20.0}, passes_correctness=True)
        lin.add(sol2, score2)

        assert len(lin) == 2
        assert lin.current_version == 2
        assert lin.best_entry.solution.version == 2

    def test_recent(self):
        lin = Lineage()
        for i in range(10):
            lin.add(
                Solution(source_code=f"v{i}", version=i),
                Score(values={"a": float(i)}, passes_correctness=True),
            )
        recent = lin.recent(3)
        assert len(recent) == 3
        assert recent[0].solution.version == 7

    def test_serialization(self):
        lin = Lineage()
        lin.add(
            Solution(source_code="code", version=0),
            Score(values={"x": 5.0}, passes_correctness=True),
        )
        data = lin.to_list()
        lin2 = Lineage.from_list(data)
        assert len(lin2) == 1
        assert lin2.entries[0].solution.source_code == "code"
