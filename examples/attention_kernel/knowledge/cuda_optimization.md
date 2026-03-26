# CUDA / GPU Kernel Optimization Reference

## Memory Hierarchy

1. **Global Memory (HBM)**: ~2TB/s bandwidth, high latency (~400 cycles)
2. **Shared Memory (SRAM)**: ~19TB/s bandwidth, low latency (~20 cycles)
3. **Registers**: Fastest, but limited (255 per thread, or 2048 per SM for warp-registers)
4. **L1/L2 Cache**: Automatic caching, configurable split with shared memory

## Key Optimization Strategies

### Memory Access Patterns
- Coalesced memory access: threads in a warp access consecutive addresses
- Avoid bank conflicts in shared memory
- Use vectorized loads (float4, int4) for higher bandwidth utilization
- Prefetch data using async copy (cp.async, TMA on Blackwell)

### Compute Optimization
- Use Tensor Cores for matrix operations (WMMA, MMA instructions)
- Minimize non-matmul operations (softmax, rescaling)
- Software-emulated exponential and conditional operations
- Packed arithmetic for half-precision operations

### Parallelism
- Thread-level parallelism within warps
- Warp-level parallelism via warp specialization
- Block-level parallelism across SMs
- Overlap computation and memory transfers

### Register Optimization
- Balance register allocation across warp groups
- Avoid register spilling to local memory (very expensive)
- Use register rebalancing: redistribute from underutilized groups
- Packed operations to reduce register footprint

### Synchronization
- Minimize barrier usage
- Use lightweight fences when possible
- Branchless operations to avoid warp divergence
- Predicated execution over conditional branches

### Pipeline Optimization
- Overlap independent operations (MMA + correction, load + compute)
- Double/triple buffering for shared memory
- Asynchronous data movement
- Instruction-level scheduling for optimal throughput

## PyTorch/Triton Optimization

For Python-level attention optimization:
- Use torch.compile() for kernel fusion
- Leverage SDPA backend selection
- Custom Triton kernels for specific patterns
- Memory-efficient attention patterns (chunked, tiled)
- Use contiguous tensors to avoid stride overhead
