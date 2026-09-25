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
4. **cumsum was 5 memory passes**: XLA's ReduceWindowRewriter builds a
   hierarchical scan out of slices and adds. `MetalScanRewriter` now turns
   minor-dim cumsum/cumprod/cummax/cummin into one `metal$scan` FFI kernel.
   GPU time for cumsum 4096x4096 f32 (METAL_PJRT_TRACE): 3.5 ms (5 kernels)
   -> 1.46 ms (1 kernel, ~88 GB/s). Threadgroup size 64-512 made no
   difference; 1024 exceeded the pipeline's limit (now clamped).
5. **softmax was 3 passes** (max reduce, sum reduce, elementwise).
   `MetalSoftmaxRewriter` now emits one `metal$softmax` kernel (online
   max/sum, row re-read from cache). GPU time for softmax 8192x1024 f32:
   1.45 ms -> 0.75 ms. Wall-clock numbers for both still need a rerun on an
   idle GPU (the 2026-09-24 runs were disturbed by other GPU jobs timing out).
6. **GEMM parity with MLX on f32** via MPS; bf16 is slower because MPS has
   no bf16 and we stage through f32 conversions.

## Where we stand vs MLX (medians, ms; lower is better)

See `bench/results/table.md` for the current full table. Highlights after
the allocator fix: nanoGPT train step 203 (MLX 252, jax-mps 276), attention
fwd+bwd 12.9 (MLX 18.6), layernorm fwd+bwd 3.6 (MLX 13.4), reductions ahead;
elementwise chain and transpose within 1.3x; cumsum 3x behind; bf16 GEMM
1.2x behind.

## Next levers, in order

1. Scans over non-minor axes and few-long-row scans (a decoupled
   look-back scan across threadgroups); logcumsumexp.
2. Softmax with a fused producer/consumer or masking (attention).
3. bf16 GEMM without staging (MLX-style steel kernels or Metal 4 primitives).
4. Untracked hazard mode with explicit barriers (CPU-side encode cost).
5. XLA launch-dimension and tiling heuristics tuned for Apple GPUs via a
   proper Apple compute capability.

## Update 2026-09-25 (early morning)

- Fused softmax and single-pass scan via FFI custom calls: kernel time 1.45 ->
  0.75 ms (softmax 8192x1024) and 3.5 -> 1.46 ms (cumsum 4096x4096). Wall
  clock not re-measured yet (see below).
- Native bf16/f16 GEMM (simdgroup-matrix kernels ported from MLX's steel
  design) is implemented but unverified on device.
- **Caveat:** killing a `pytest -n 2` run of JAX's lax_test with SIGALRM left
  worker processes stuck exiting inside the GPU driver, after which every
  Metal command buffer in every process timed out (`kIOGPUCommandBuffer
  CallbackErrorTimeout`). Measurements taken after that point are invalid; a
  reboot clears it. Lesson: never kill GPU-using test workers with signals
  while command buffers are in flight; use pytest timeouts per test instead.
- Runtime changes prompted by this: a failed command buffer is now reported
  once and the stream recovers (a stuck error state made whole test runs
  fail), and command buffers are committed after 2^29 dispatched threads so a
  batch of heavy kernels cannot approach the watchdog.
