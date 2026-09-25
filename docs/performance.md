# Performance notes

Methodology: `bench/jax_bench.py` (same jitted programs on every JAX backend,
warmup, median of 10 with `block_until_ready`), `bench/mlx_bench.py` (same
workloads in MLX with `mx.compile`), `bench/latency.py` (fixed per-call
cost), `bench/run_all.sh` to produce `bench/results/table.md`. Machine: M3,
10-core GPU, 8 GB. Per-command-buffer GPU timing: `METAL_PJRT_TRACE=1`.

## Findings so far (2026-09-24)

1. **Buffer page faults dominated everything.** A fresh `MTLBuffer` costs
   about 60 us per MB on first touch (3.8 ms for 64 MB) plus 0.8 ms to
   release. With the plain platform allocator every output was a fresh
   buffer. Switching the PJRT client to XLA's BFC pool (grow on demand, no
   preallocation) took the nanoGPT training step from 3257 ms to 203 ms and
   attention forward from 58 ms to 7 ms.
2. **Emitted elementwise kernels run at memory bandwidth.** Replaying the
   emitted `x*2` kernel standalone at XLA's launch geometry gives 1.58 ms for
   128 MB of traffic (81 GB/s), the same as hand-written MSL. Threadgroup
   size, fast-math and the `char*` pointer convention make no difference.
   XLA bakes the block size into the index math, so launch geometry must
   match what XLA chose.
3. **Fixed per-execution overhead is about 0.4 ms** on top of GPU time, and
   about 165 us for tiny programs, the same as jax-mps and MLX. One execution
   is one command buffer with all its dispatches, plus an empty command
   buffer on the event stream.
4. **cumsum is 5 memory passes**: XLA's ReduceWindowRewriter builds a
   hierarchical scan out of slices and adds. MLX has a single-pass scan
   kernel. A Metal scan emitter is the fix (about 3x on this case).
5. **softmax is 3 passes** (max reduce, sum reduce, elementwise); MLX fuses
   it. XLA's fused softmax path is Triton-only. A Metal-specific rewrite to a
   single row kernel would close the 1.5x gap.
6. **GEMM parity with MLX on f32** via MPS; bf16 is slower because MPS has
   no bf16 and we stage through f32 conversions.

## Where we stand vs MLX (medians, ms; lower is better)

See `bench/results/table.md` for the current full table. Highlights after
the allocator fix: nanoGPT train step 203 (MLX 252, jax-mps 276), attention
fwd+bwd 12.9 (MLX 18.6), layernorm fwd+bwd 3.6 (MLX 13.4), reductions ahead;
elementwise chain and transpose within 1.3x; cumsum 3x behind; bf16 GEMM
1.2x behind.

## Next levers, in order

1. Scan emitter (cumsum, cumprod, associative scans).
2. Fused row softmax / log-softmax kernel.
3. bf16 GEMM without staging (MLX-style steel kernels or Metal 4 primitives).
4. Untracked hazard mode with explicit barriers (CPU-side encode cost).
5. XLA launch-dimension and tiling heuristics tuned for Apple GPUs via a
   proper Apple compute capability.
