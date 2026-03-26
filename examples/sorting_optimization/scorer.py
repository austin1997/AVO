"""Scoring function for the sorting optimization example.

Evaluates candidates on:
1. Correctness: output must match Python's built-in sorted()
2. Performance: wall-clock time across multiple array sizes
"""

from __future__ import annotations

import importlib.util
import random
import time
from pathlib import Path
from typing import Any

from avo.core.scoring import ScoringFunction
from avo.core.types import Score

CONFIGURATIONS = {
    "n_100": 100,
    "n_1000": 1000,
    "n_5000": 5000,
    "n_10000": 10000,
}

NUM_TRIALS = 5
RANDOM_SEED = 42


class Scorer(ScoringFunction):
    """Sorting performance scorer."""

    def __init__(self, workspace_dir: Path | str | None = None) -> None:
        self._workspace = Path(workspace_dir) if workspace_dir else Path(".")

    def evaluate(self, source_code: str, source_file: str = "solution.py") -> Score:
        sol_path = self._workspace / source_file
        sol_path.parent.mkdir(parents=True, exist_ok=True)
        sol_path.write_text(source_code)

        try:
            spec = importlib.util.spec_from_file_location("candidate", sol_path)
            if spec is None or spec.loader is None:
                return Score(correctness_message="Failed to load module")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            sort_fn = getattr(mod, "sort_array", None)
            if sort_fn is None:
                return Score(correctness_message="No sort_array function found")
        except Exception as e:
            return Score(correctness_message=f"Import error: {e}")

        correct, msg = self._check_correctness(sort_fn)
        if not correct:
            return Score(passes_correctness=False, correctness_message=msg)

        values = self._measure_performance(sort_fn)
        return Score(values=values, passes_correctness=True, correctness_message="OK")

    def _check_correctness(self, sort_fn: Any) -> tuple[bool, str]:
        rng = random.Random(RANDOM_SEED)
        test_cases = [
            [],
            [1],
            [3, 1, 2],
            [5, 4, 3, 2, 1],
            list(range(100)),
            list(range(100, 0, -1)),
            [rng.randint(-10000, 10000) for _ in range(500)],
            [rng.randint(-10000, 10000) for _ in range(1000)],
        ]
        for i, case in enumerate(test_cases):
            try:
                result = sort_fn(case)
                expected = sorted(case)
                if result != expected:
                    return False, f"Test case {i}: expected {expected[:5]}... got {result[:5]}..."
            except Exception as e:
                return False, f"Test case {i} raised: {e}"
        return True, "OK"

    def _measure_performance(self, sort_fn: Any) -> dict[str, float]:
        """Measure throughput (elements/second) for each configuration."""
        values = {}
        for config_name, size in CONFIGURATIONS.items():
            rng = random.Random(RANDOM_SEED + hash(config_name))
            arrays = [[rng.randint(-100000, 100000) for _ in range(size)] for _ in range(NUM_TRIALS)]

            total_time = 0.0
            total_elements = 0
            for arr in arrays:
                start = time.perf_counter()
                sort_fn(arr)
                elapsed = time.perf_counter() - start
                total_time += elapsed
                total_elements += len(arr)

            throughput = total_elements / total_time if total_time > 0 else 0
            values[config_name] = throughput

        return values

    def get_configurations(self) -> list[str]:
        return list(CONFIGURATIONS.keys())

    def get_reference_description(self) -> str:
        return "Reference: Python's built-in sorted() function"

    def get_scoring_context(self) -> str:
        return (
            "Optimize a sorting function for integer arrays. "
            "Performance is measured as elements/second across array sizes: "
            + ", ".join(f"{k}={v}" for k, v in CONFIGURATIONS.items())
            + f". Each configuration runs {NUM_TRIALS} trials. "
            "The function must be named sort_array(arr) and return a sorted copy."
        )
