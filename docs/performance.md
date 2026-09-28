# Performance

Where metal-pjrt-plugin stands, how it is measured, and what was tried and dropped.
The dated measurements behind every line here, including superseded ones,
are in `docs/archive/performance-log-2026-09.md`. The runtime policies the
numbers depend on (command-buffer batching, the buffer cache) are described
in `docs/design.md`, "Runtime".

## Methodology

- `bench/jax_bench.py` runs the same jitted programs on any JAX backend
  (3 warmup calls, median of 10 with `block_until_ready`);
  `bench/mlx_bench.py` runs the same workloads in MLX with `mx.compile`;
  `bench/linalg_bench.py` and `bench/tinygp_bench.py` cover linear algebra
  and tinygp's solvers; `bench/dispatch_bound.py` times the fixed per-call
  cost and chains of tiny kernels.
- `bench/run_all.sh` runs `BENCH_ROUNDS` (default 3) rounds with the
  backends (`BENCH_BACKENDS`, default `metal metal-gpu cpu mlx`) interleaved
  inside each round, and `bench/report.py` writes the median over rounds to
  `bench/results/table.md`. Rows record the commit, JAX/MLX versions, the
  plugin's platform version and the `METAL_PJRT_*`, `JAX_MTL_*`,
  `XLA_FLAGS` and `JAX_PLATFORMS` settings. It refuses to run after a GPU
  reset since boot (`scripts/gpu_health.py --strict`;
  `BENCH_ALLOW_DEGRADED=1` overrides): a GPU that has been reset several
  times stays ~10x slower per dispatch until a reboot.
- GPU time: the `metal-gpu` arm is a second pass with `METAL_PJRT_TRACE=1`,
  which logs every command buffer's GPU start/end time; the trace adds
  ~0.1-0.3 ms to sub-millisecond wall times, hence a separate pass. TFLOPS
  are computed at the best wall time.
- Run `bazel shutdown` first (or set `BENCH_BAZEL_SHUTDOWN=1`; `run_all.sh`
  no longer does it by default, as it would kill another session's build).
  The Bazel server's memory makes the system memory guard refuse large
  allocations (see "Runtime" in `docs/design.md`).
- A/Bs of sub-millisecond programs interleave the arms and report p10 /
  median / p90. Tiny kernels run in one of two GPU performance states
  (~4.2 vs ~2.3 us of GPU time per dispatch), which macOS picks from the
  duty cycle and no public API pins: `dispatch_bound.py`'s 64-step chain
  flips between them from call to call (~410 vs ~720 us), so a single
  median is bimodal. The command-buffer pacing is not the cause (both
  modes appear with an early commit on every op or none).
- Machine: M3, 10-core GPU, 8 GB, macOS 26.2. Every number here is from it.

## Headline numbers

`bench/results/table.md` has the full table (3 interleaved rounds at
64488ff, before the platform rename; MLX's column is an older run without
metadata). Medians, ms:

| case | mtl | CPU | MLX |
|---|---|---|---|
| nanoGPT train step | 187 | 1258 | 232 |
| nanoGPT forward (loss) | 60 | 408 | 69 |
| attention fwd+bwd 8x8x512x64 | 11.9 | 43.9 | 15.4 |
| layernorm fwd+bwd 8192x1024 | 2.6 | 1.5 | 10.6 |
| adam, 50x1M parameters | 15.7 | 27.8 | 16.0 |
| matmul f32 4096 | 48.7 | 315 | 52.2 |
| matmul bf16 2048 | 6.1 | 89.4 | 5.9 |
| softmax 8192x1024 | 1.91 | 1.33 | 0.97 |
| cumsum 4096x4096 rows | 1.79 | 11.6 | 1.76 |
| cnn fwd+bwd 32x32x32 | 6.2 | 3.8 | 2.3 |
| cholesky 128 | 0.86 | 0.03 | - |
| cholesky 2048 | 6.8 | 8.6 | - |

Later runs agree: the nanoGPT train step measured 180-185 ms at 23bc8fd
(one outlier at 200 ms), the forward 58-59 ms. Other current numbers:

- Fixed cost of a tiny program (`jit(x*2+1)` on 1024 floats, dispatch to
  result): ~170-210 us median. A long chain of tiny kernels costs ~2-4 us
  of GPU time per dispatch; a 2000-step scan with tiny state ~8.5 ms.
- Sort (`jnp.sort` f32, radix): 20k 0.31 ms, 1M 2.6 ms, 16M 41 ms; 28-66x
  faster than the bitonic network it replaced (e93e731). Rows of <= 64
  stay on the bitonic network.
- tinygp value+grad, parallel solver: n = 20000 7 ms, n = 200000 50 ms
  (CPU 77 and 703 ms at the time of the loop work, fd27d3c).
- Memory: a finished nanoGPT run idles at ~0.52 GB footprint (the process
  itself; the cache is released after ~2 s), peak ~1.9 GB.
- Small dense linear algebra above 32x32 runs in host LAPACK after a full
  stream synchronization: cholesky 128 costs ~0.3-0.9 ms against 0.03 ms on
  CPU, two dependent GPU round trips (~130 us each) around the LAPACK call.
  From n ~ 2048 it matches CPU.

Where the time goes, in brief: memory-bound kernels run at memory bandwidth
(the emitted `x*2` kernel reaches the same ~81 GB/s as hand-written MSL);
fused reductions and optimizer updates are where XLA beats MLX; the gaps are
convolutions (naive loop emitter, no library path), standalone softmax
(XLA's two fusions vs MLX's one kernel) and anything bound by per-dispatch
latency.

## Measured and rejected

One line each; the numbers and reasoning are in the archived log and the
commit messages.

- Indirect command buffers (cc0c5e2): replay costs 1.2-1.4 us of GPU time
  per dependent dispatch against 0.8 us for direct encoding.
- XLA command buffers as software replay (ab0dfcc, removed bdb9c4c): nanoGPT
  184 vs 196 ms, but loops 5-30% slower (scan 8.1 vs 6.2 ms); the 30-40 us
  per-launch cost it targeted was a 1 ms Synchronize sleep and a
  command-buffer submission storm, both fixed in the runtime.
- The `metal$softmax` rewriter (fcbf5ce): 1.5-1.8x on standalone softmax,
  but it never fires under autodiff (0 of 111 custom calls in the nanoGPT
  train step) and gained nothing where it fired. Standalone softmax
  8192x1024 went 1.24 -> 1.91 ms (MLX 0.97).
- f32 GEMMs with an epilogue on steel (7704d02, reverted 29869ef): steel f32
  is 2-15% slower than MPS from batch*m*n*k ~ 1.7e10, 0-7% faster on mid
  shapes; nanoGPT unchanged. f32 stays MPS + a second pass (bitwise
  identical output).
- XLA's BFC allocator (adopted 6f98ea9, replaced 1f81e01, option removed
  b0019d2): the runtime's size-class cache matches its step time (nanoGPT
  178.7-180.0 vs 179.6-181.7 ms) and returns memory (idle 0.5 vs 2.5-2.8 GB).
  The plain platform allocator with no cache: 217-245 ms (first-touch page
  faults, ~60 us per MB).
- Host LAPACK as a stream host task (59ec97d): cholesky 128 median 313 ->
  299 us with the same p90, a chain of 100 `cho_solve` 64 +12%. The two GPU
  round trips around the call remain, and the hold rule keeps the dispatch
  thread waiting anyway. Python callbacks stay synchronous too (not
  measured: a commit that host-waits for a callback task needing the GIL
  could deadlock).
- Radix sort on short rows (e93e731): one threadgroup per row lost 3-180x
  (1000x32 942 vs 307 us, 100000x4 65 vs 0.35 ms); such sorts are
  expanded to the bitonic network before SortRewriter sees them.
- Host staging of H2D transfers (e133b93): 100 MB `device_put` 9.4 vs
  8.7 ms, and it never triggered anyway (numpy memory counts as pinned).
- `const` on read-only kernel arguments (roadmap 5.4, not committed):
  nanoGPT 178.9-179.7 vs 178.3-179.5 ms.
- An on-disk kernel cache / `MTLBinaryArchive` (roadmap 5.2): Metal's
  system shader cache already compiles repeated MSL in ~1.3 ms (186 ms the
  first time), and JAX's persistent compilation cache skips it entirely.
- Per-encoder blame via `EncoderExecutionStatus` (d90c267): free (96-107 vs
  97-111 us per round trip), but a command buffer is mostly one compute
  encoder, so it cannot narrow a watchdog reset to a kernel.
- A configurable early-commit interval (`METAL_PJRT_EARLY_COMMIT_US`,
  removed f7ec606): 0 and 1e6 leave chain64's bimodality as it is (above);
  the 500 us constant stays.
