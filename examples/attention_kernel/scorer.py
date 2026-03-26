"""Scoring function for the attention kernel optimization example.

Evaluates candidates on:
1. Correctness: numerical agreement with torch SDPA reference
2. Performance: TFLOPS throughput across benchmark configurations

Follows the paper's benchmark setup:
- head_dim=128, BF16 precision
- seq_len in {4096, 8192, 16384, 32768}
- batch_size * seq_len = 32768 total tokens
- 16 heads for MHA
"""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from avo.core.scoring import ScoringFunction
from avo.core.types import Score

HEAD_DIM = 128
NUM_HEADS = 16
KV_HEADS = 16  # MHA: kv_heads == num_heads
TOTAL_TOKENS = 32768
DTYPE = torch.bfloat16

CONFIGURATIONS: dict[str, dict[str, Any]] = {
    "mha_causal_seq4k": {"seq_len": 4096, "batch_size": 8, "causal": True},
    "mha_causal_seq8k": {"seq_len": 8192, "batch_size": 4, "causal": True},
    "mha_causal_seq16k": {"seq_len": 16384, "batch_size": 2, "causal": True},
    "mha_causal_seq32k": {"seq_len": 32768, "batch_size": 1, "causal": True},
    "mha_noncausal_seq4k": {"seq_len": 4096, "batch_size": 8, "causal": False},
    "mha_noncausal_seq8k": {"seq_len": 8192, "batch_size": 4, "causal": False},
    "mha_noncausal_seq16k": {"seq_len": 16384, "batch_size": 2, "causal": False},
    "mha_noncausal_seq32k": {"seq_len": 32768, "batch_size": 1, "causal": False},
}

WARMUP_ROUNDS = 5
BENCHMARK_ROUNDS = 10
CORRECTNESS_ATOL = 1e-2
CORRECTNESS_RTOL = 1e-2


def _compute_flops(batch: int, heads: int, seq_len: int, head_dim: int, causal: bool) -> float:
    """Compute FLOPs for attention forward pass."""
    flops = 4 * batch * heads * seq_len * seq_len * head_dim
    if causal:
        flops //= 2
    return float(flops)


class Scorer(ScoringFunction):
    """Attention kernel performance scorer."""

    def __init__(self, workspace_dir: Path | str | None = None) -> None:
        self._workspace = Path(workspace_dir) if workspace_dir else Path(".")
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

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
            attn_fn = getattr(mod, "attention_forward", None)
            if attn_fn is None:
                return Score(correctness_message="No attention_forward function found")
        except Exception as e:
            return Score(correctness_message=f"Import error: {e}")

        if self._device == "cpu":
            return self._evaluate_cpu_only(attn_fn)

        return self._evaluate_gpu(attn_fn)

    def _evaluate_cpu_only(self, attn_fn: Any) -> Score:
        """Lightweight evaluation for CPU-only environments."""
        correct, msg = self._check_correctness_small(attn_fn)
        if not correct:
            return Score(passes_correctness=False, correctness_message=msg)
        return Score(
            values={"cpu_correctness": 1.0},
            passes_correctness=True,
            correctness_message="CPU-only: correctness verified, no GPU benchmarks",
        )

    def _evaluate_gpu(self, attn_fn: Any) -> Score:
        """Full GPU evaluation with correctness + TFLOPS benchmarks."""
        correct, msg = self._check_correctness(attn_fn)
        if not correct:
            return Score(passes_correctness=False, correctness_message=msg)

        values = {}
        for config_name, cfg in CONFIGURATIONS.items():
            tflops = self._benchmark_config(attn_fn, cfg)
            values[config_name] = tflops

        return Score(values=values, passes_correctness=True, correctness_message="OK")

    def _check_correctness_small(self, attn_fn: Any) -> tuple[bool, str]:
        """Quick correctness check with small tensors (works on CPU)."""
        for causal in [False, True]:
            batch, seq, heads, dim = 1, 64, 4, 32
            Q = torch.randn(batch, heads, seq, dim)
            K = torch.randn(batch, heads, seq, dim)
            V = torch.randn(batch, heads, seq, dim)

            try:
                result = attn_fn(Q, K, V, causal=causal)
            except Exception as e:
                return False, f"Execution error (causal={causal}): {e}"

            ref = F.scaled_dot_product_attention(Q, K, V, is_causal=causal)
            if not torch.allclose(result, ref, atol=0.05, rtol=0.05):
                max_diff = (result - ref).abs().max().item()
                return False, f"Correctness failure (causal={causal}): max_diff={max_diff:.6f}"

        return True, "OK"

    def _check_correctness(self, attn_fn: Any) -> tuple[bool, str]:
        """Full correctness check with benchmark-sized tensors on GPU."""
        for causal in [False, True]:
            batch, seq = 2, 1024
            Q = torch.randn(batch, NUM_HEADS, seq, HEAD_DIM, device=self._device, dtype=DTYPE)
            K = torch.randn(batch, KV_HEADS, seq, HEAD_DIM, device=self._device, dtype=DTYPE)
            V = torch.randn(batch, KV_HEADS, seq, HEAD_DIM, device=self._device, dtype=DTYPE)

            try:
                result = attn_fn(Q, K, V, causal=causal)
            except Exception as e:
                return False, f"Execution error (causal={causal}): {e}"

            ref = F.scaled_dot_product_attention(
                Q, K, V, is_causal=causal
            )
            if not torch.allclose(result, ref, atol=CORRECTNESS_ATOL, rtol=CORRECTNESS_RTOL):
                max_diff = (result - ref).abs().max().item()
                return False, f"Correctness failure (causal={causal}): max_diff={max_diff:.6f}"

        return True, "OK"

    def _benchmark_config(self, attn_fn: Any, cfg: dict[str, Any]) -> float:
        """Benchmark a single configuration, returning TFLOPS."""
        batch = cfg["batch_size"]
        seq = cfg["seq_len"]
        causal = cfg["causal"]

        Q = torch.randn(batch, NUM_HEADS, seq, HEAD_DIM, device=self._device, dtype=DTYPE)
        K = torch.randn(batch, KV_HEADS, seq, HEAD_DIM, device=self._device, dtype=DTYPE)
        V = torch.randn(batch, KV_HEADS, seq, HEAD_DIM, device=self._device, dtype=DTYPE)

        for _ in range(WARMUP_ROUNDS):
            attn_fn(Q, K, V, causal=causal)
        torch.cuda.synchronize()

        times = []
        for _ in range(BENCHMARK_ROUNDS):
            torch.cuda.synchronize()
            start = time.perf_counter()
            attn_fn(Q, K, V, causal=causal)
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)

        avg_time = sum(times) / len(times)
        flops = _compute_flops(batch, NUM_HEADS, seq, HEAD_DIM, causal)
        tflops = flops / avg_time / 1e12

        return tflops

    def get_configurations(self) -> list[str]:
        return list(CONFIGURATIONS.keys())

    def get_reference_description(self) -> str:
        return (
            "Reference: torch.nn.functional.scaled_dot_product_attention\n"
            f"Hardware: GPU ({self._device})\n"
            f"Precision: BF16, head_dim={HEAD_DIM}, heads={NUM_HEADS}"
        )

    def get_scoring_context(self) -> str:
        return (
            f"Optimize attention forward pass. head_dim={HEAD_DIM}, "
            f"num_heads={NUM_HEADS}, kv_heads={KV_HEADS}, BF16 precision.\n"
            f"Configurations: {', '.join(CONFIGURATIONS.keys())}\n"
            f"Total tokens fixed at {TOTAL_TOKENS} (batch_size * seq_len).\n"
            f"Scored by TFLOPS. Correctness checked against torch SDPA."
        )
