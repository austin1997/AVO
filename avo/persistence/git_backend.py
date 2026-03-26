"""Git-based lineage persistence for AVO.

Each committed version x_i is persisted as a git commit along with its
score metadata, maintaining full state continuity across the evolutionary
process (paper Section 3.3).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from git import Repo
from git.exc import InvalidGitRepositoryError

from avo.core.types import Lineage, LineageEntry, Score, Solution

logger = logging.getLogger(__name__)

SCORE_METADATA_FILE = ".avo_scores.json"
LINEAGE_METADATA_FILE = ".avo_lineage.json"


class GitBackend:
    """Persist each committed solution as a git commit with score metadata."""

    def __init__(self, workspace_dir: Path | str) -> None:
        self._workspace = Path(workspace_dir).resolve()
        self._workspace.mkdir(parents=True, exist_ok=True)
        self._repo = self._init_or_open_repo()

    def _init_or_open_repo(self) -> Repo:
        try:
            repo = Repo(self._workspace)
            logger.info("Opened existing git repo at %s", self._workspace)
            return repo
        except InvalidGitRepositoryError:
            repo = Repo.init(self._workspace)
            logger.info("Initialized new git repo at %s", self._workspace)
            gitignore = self._workspace / ".gitignore"
            gitignore.write_text("__pycache__/\n*.pyc\n.env\n")
            repo.index.add([".gitignore"])
            repo.index.commit("Initial AVO repository")
            return repo

    @property
    def repo(self) -> Repo:
        return self._repo

    @property
    def workspace(self) -> Path:
        return self._workspace

    def persist(self, solution: Solution, score: Score) -> str:
        """Commit a solution and its score to git.

        Returns the commit hash.
        """
        solution_path = self._workspace / solution.source_file
        solution_path.parent.mkdir(parents=True, exist_ok=True)
        solution_path.write_text(solution.source_code)

        score_path = self._workspace / SCORE_METADATA_FILE
        score_data = {
            "version": solution.version,
            "score": score.to_dict(),
            "parent_version": solution.parent_version,
            "metadata": solution.metadata,
        }
        score_path.write_text(json.dumps(score_data, indent=2))

        self._update_lineage_file(solution, score)

        self._repo.index.add([solution.source_file, SCORE_METADATA_FILE, LINEAGE_METADATA_FILE])

        commit_msg = (
            f"AVO v{solution.version}: geomean={score.geomean:.4f}\n\n"
            f"Correctness: {'PASS' if score.passes_correctness else 'FAIL'}\n"
            f"Scores: {json.dumps(score.values, indent=2)}"
        )
        commit = self._repo.index.commit(commit_msg)
        commit_hash = commit.hexsha

        tag_name = f"v{solution.version}"
        try:
            self._repo.create_tag(tag_name, ref=commit, message=f"Version {solution.version}")
        except Exception:
            logger.warning("Tag %s already exists, skipping", tag_name)

        logger.info("Persisted v%d as commit %s", solution.version, commit_hash[:8])
        return commit_hash

    def _update_lineage_file(self, solution: Solution, score: Score) -> None:
        """Append to the lineage metadata file."""
        lineage_path = self._workspace / LINEAGE_METADATA_FILE
        existing: list[dict[str, Any]] = []
        if lineage_path.exists():
            try:
                existing = json.loads(lineage_path.read_text())
            except (json.JSONDecodeError, ValueError):
                existing = []

        existing.append({
            "version": solution.version,
            "score": score.to_dict(),
            "source_file": solution.source_file,
            "timestamp": solution.timestamp,
            "parent_version": solution.parent_version,
        })
        lineage_path.write_text(json.dumps(existing, indent=2))

    def read_solution_at_version(self, version: int, source_file: str = "solution.py") -> str | None:
        """Read a solution's source code at a specific version tag."""
        tag_name = f"v{version}"
        try:
            tag = self._repo.tags[tag_name]
            blob = tag.commit.tree / source_file
            return blob.data_stream.read().decode("utf-8")
        except (IndexError, KeyError):
            logger.warning("Could not read v%d from git", version)
            return None

    def get_commit_history(self) -> list[dict[str, Any]]:
        """Return list of all AVO commits with metadata."""
        history = []
        for commit in self._repo.iter_commits():
            if commit.message.startswith("AVO v"):
                history.append({
                    "hash": commit.hexsha,
                    "message": commit.message.strip(),
                    "timestamp": commit.committed_date,
                })
        return list(reversed(history))

    def load_lineage(self, source_file: str = "solution.py") -> Lineage:
        """Reconstruct the full Lineage from git history."""
        lineage_path = self._workspace / LINEAGE_METADATA_FILE
        if not lineage_path.exists():
            return Lineage()

        try:
            data = json.loads(lineage_path.read_text())
        except (json.JSONDecodeError, ValueError):
            return Lineage()

        lineage = Lineage()
        for entry_data in data:
            version = entry_data["version"]
            source_code = self.read_solution_at_version(version, source_file) or ""
            solution = Solution(
                source_code=source_code,
                version=version,
                source_file=entry_data.get("source_file", source_file),
                parent_version=entry_data.get("parent_version"),
                timestamp=entry_data.get("timestamp", 0),
            )
            score = Score.from_dict(entry_data["score"])
            lineage.add(solution, score)

        return lineage
