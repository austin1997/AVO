# FlashAttention Algorithm Reference

## Core Idea

FlashAttention computes exact attention without materializing the full N×N score matrix.
It processes Q, K, V in tiles, maintaining a running softmax (running row-maximum and
row-sum) and accumulating the output O incrementally.

## Online Softmax Algorithm

For each query row:
1. Initialize: m = -inf (running max), l = 0 (running sum), O = 0
2. For each key block j:
   a. Compute scores: S_j = Q * K_j^T / sqrt(d)
   b. Compute block max: m_j = max(S_j)
   c. Update running max: m_new = max(m, m_j)
   d. Rescale accumulator: O = O * exp(m - m_new)
   e. Compute weights: P_j = exp(S_j - m_new)
   f. Update sum: l = l * exp(m - m_new) + sum(P_j)
   g. Accumulate: O = O + P_j * V_j
   h. Update max: m = m_new
3. Normalize: O = O / l

## Key Optimizations

### Tiling
- Process K, V in blocks to fit in SRAM/shared memory
- Reduces HBM reads from O(N^2) to O(N)

### Warp Specialization (Blackwell)
- MMA warps: execute QK GEMM and PV GEMM via tensor cores
- Softmax warps: compute attention weights from scores
- Correction warps: rescale output when running max changes
- Load/epilogue warps: handle data movement via TMA

### Dual Q-Stage Pipeline
- Process two Q-tiles concurrently
- Overlap computation and memory operations

### Causal Masking
- Fully masked K-blocks (no valid entries): skip entirely
- Partially masked K-blocks: apply mask during softmax
- Fully unmasked K-blocks: different (faster) execution path

## Performance Considerations

1. **Memory bandwidth**: Attention is memory-bound at long sequences
2. **Compute utilization**: At shorter sequences, becoming compute-bound
3. **Register pressure**: Accumulator rescaling requires extra registers
4. **Synchronization overhead**: Warp barriers between pipeline stages
5. **Branch divergence**: Causal mask creates different execution paths
