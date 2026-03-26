"""Tests for avo.persistence.git_backend."""

import tempfile
from pathlib import Path

import pytest

from avo.core.types import Score, Solution
from avo.persistence.git_backend import GitBackend


class TestGitBackend:
    def test_init_new_repo(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gb = GitBackend(tmpdir)
            assert (Path(tmpdir) / ".git").exists()
            assert (Path(tmpdir) / ".gitignore").exists()

    def test_persist_and_read(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gb = GitBackend(tmpdir)
            sol = Solution(source_code="def f(): return 1", version=0, source_file="solution.py")
            score = Score(values={"a": 10.0}, passes_correctness=True)

            commit_hash = gb.persist(sol, score)
            assert len(commit_hash) == 40

            code = gb.read_solution_at_version(0)
            assert code == "def f(): return 1"

    def test_multiple_versions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gb = GitBackend(tmpdir)

            for i in range(3):
                sol = Solution(source_code=f"version_{i}", version=i, source_file="solution.py")
                score = Score(values={"a": float(i * 10)}, passes_correctness=True)
                gb.persist(sol, score)

            assert gb.read_solution_at_version(0) == "version_0"
            assert gb.read_solution_at_version(1) == "version_1"
            assert gb.read_solution_at_version(2) == "version_2"

    def test_commit_history(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gb = GitBackend(tmpdir)
            sol = Solution(source_code="code", version=0, source_file="solution.py")
            score = Score(values={"a": 5.0}, passes_correctness=True)
            gb.persist(sol, score)

            history = gb.get_commit_history()
            assert len(history) == 1
            assert "v0" in history[0]["message"]

    def test_load_lineage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gb = GitBackend(tmpdir)

            for i in range(3):
                sol = Solution(source_code=f"v{i}_code", version=i, source_file="solution.py")
                score = Score(values={"x": float(i + 1)}, passes_correctness=True)
                gb.persist(sol, score)

            lineage = gb.load_lineage()
            assert len(lineage) == 3
            assert lineage.entries[0].solution.version == 0
            assert lineage.entries[2].score.values["x"] == 3.0
