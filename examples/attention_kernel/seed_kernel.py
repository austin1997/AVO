"""Seed attention kernel: naive scaled dot-product attention in PyTorch.

This is the starting point (x0) for AVO evolution on attention kernels.
The agent will iteratively optimize this toward FlashAttention-level performance.

Matches the paper's benchmark: head_dim=128, BF16, variable seq_len and batch_size.
"""

import torch
import torch.nn.functional as F


def attention_forward(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    """Compute scaled dot-product attention.

    Args:
        Q: Query tensor of shape (batch, heads, seq_len, head_dim)
        K: Key tensor of shape (batch, kv_heads, seq_len, head_dim)
        V: Value tensor of shape (batch, kv_heads, seq_len, head_dim)
        causal: Whether to apply causal masking

    Returns:
        Output tensor of shape (batch, heads, seq_len, head_dim)
    """
    batch, heads, seq_len, head_dim = Q.shape
    kv_heads = K.shape[1]

    if kv_heads < heads:
        group_size = heads // kv_heads
        K = K.repeat_interleave(group_size, dim=1)
        V = V.repeat_interleave(group_size, dim=1)

    scale = head_dim ** -0.5
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

    if causal:
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=Q.device, dtype=torch.bool),
            diagonal=1,
        )
        scores.masked_fill_(mask, float("-inf"))

    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)

    return output
