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
  plugin's platform version and the `METAL_PJRT_*`, `XLA_FLAGS` and
  `JAX_PLATFORMS` settings. It refuses to run after a GPU reset since boot
  (`scripts/gpu_health.py --strict`; `BENCH_ALLOW_DEGRADED=1` overrides): a
  GPU that has been reset several times stays ~10x slower per dispatch
  until a reboot.
- The arms keep their names from before the platform rename: `metal` runs
  `jax_bench.py` with `JAX_PLATFORMS=mtl`, `metal-gpu` the same with the
  trace below, `cpu` with `JAX_PLATFORMS=cpu`.
- GPU time: the `metal-gpu` arm is a second pass with `METAL_PJRT_TRACE=1`,
  which logs every command buffer's GPU start/end time; the trace adds
  ~0.1-0.3 ms to sub-millisecond wall times, hence a separate pass. TFLOPS
  are computed at the best wall time.
- Run `bazel shutdown` first (or set `BENCH_BAZEL_SHUTDOWN=1`; `run_all.sh`
  does not do it by default, as it would stop a build running elsewhere).
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

Historical: this table and `bench/results/table.md` (the full table) are
from one run (3 interleaved rounds at 64488ff, before the platform rename
and later runtime work; MLX's column is an older run without metadata).
They are not regenerated; the bullets below are more recent. Medians, ms:

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

Later runs agree: the nanoGPT train step measured 180-185 ms (one outlier
at 200 ms), the forward 58-59 ms. Other current numbers:

- Fixed cost of a tiny program (`jit(x*2+1)` on 1024 floats, dispatch to
  result): ~170-210 us median. A long chain of tiny kernels costs ~2-4 us
  of GPU time per dispatch; a 2000-step scan with tiny state ~8.5 ms.
- Sort (`jnp.sort` f32, radix): 20k 0.31 ms, 1M 2.6 ms, 16M 41 ms; 28-66x
  faster than the bitonic network it replaced. Rows of <= 64
  stay on the bitonic network.
- tinygp value+grad, parallel solver: n = 20000 7 ms, n = 200000 50 ms
  (CPU 77 and 703 ms when measured).
- Memory: a finished nanoGPT run idles at ~0.52 GB footprint (the process
  itself; the cache is released after ~2 s), peak ~1.9 GB.
- Small dense linear algebra above 32x32 runs in host LAPACK after a full
  stream synchronization: cholesky 128 costs ~0.3 ms median (313 us, p90
  680 us, in the later host-task A/B below; the historical table's 0.86 ms
  is the older run) against 0.03 ms on CPU, two dependent GPU round trips
  (~130 us each) around the LAPACK call. From n ~ 2048 it matches CPU.
- Persistent compilation cache (README, "Compilation cache"): with it on,
  the first call of the nanoGPT training step in `bench/jax_bench.py` in a
  new process takes ~0.42 s from the cache instead of ~0.75 s (MLP train
  step: 41 ms instead of ~66 ms).

- Convolutions run on MLX's steel kernels (`metal$conv`, `metal_pjrt/conv`;
  docs/integration-notes.md). The cnn bench (f32, batch 32; bursts of 30
  calls, 6 interleaved rounds, p10 / median / p90 ms; "loop" is the same
  build with `METAL_PJRT_DISABLE_REWRITES=conv`, i.e. XLA's loop emitter as
  before):

  | case | metal$conv | loop emitter | MLX |
  |---|---|---|---|
  | cnn fwd+bwd | 1.555 / 1.562 / 1.631 | 5.509 / 5.552 / 5.593 | 1.416 / 1.581 / 2.191 |
  | cnn fwd 32x32x32 | 0.274 / 0.275 / 0.278 | 1.643 / 1.648 / 1.696 | 0.384 / 0.386 / 0.396 |
  | cnn fwd+bwd, NCHW | 1.573 / 1.575 / 1.582 | 30.5 / 30.7 / 30.7 | - |
  | cnn fwd, NCHW | 0.280 / 0.281 / 0.292 | 6.74 / 6.74 / 6.75 | - |

  NCHW programs pay only the rewriter's transposes to NHWC (~0.01-0.02
  ms here). The loop emitter spent ~97-100% of the step's GPU time in the
  convolutions. Per kernel (`bench/conv_bench` vs `bench/conv_bench_mlx.py`,
  same method, p10 ms): conv1 (3->32) 0.092 vs MLX 0.098, conv2 (32->64,
  stride 2) 0.134 vs 0.128, conv2's input gradient (input dilation 2,
  flipped) 0.220 vs 0.218, the weight gradients (patches x dY, split-K)
  0.322 vs 0.252 and 0.576 vs 0.525. f16: 0.84-1.10x MLX (forward).
- Small convolutions stay on the loop emitter: under 4 Mflop a custom call
  plus a separate relu kernel lost to the loop emitter's one fused kernel
  (forward + relu, bursts of 30, p10 ms, two interleaved runs: 0.3-2.4
  Mflop 0.030-0.058 vs 0.029-0.037; C = 1 at 3.6 Mflop 0.121 vs 0.084),
  from ~4.7 Mflop metal$conv wins (0.038 vs 0.046; 19 Mflop 0.036-0.087 vs
  0.121-0.199). MLX runs these in 0.01-0.03 ms: the fixed cost of a
  dispatch-bound program, not the kernels.

- FFTs run on MLX's FFT kernels (`metal$fft`, `metal_pjrt/fft`).
  `bench/fft_bench.py` (one process; bursts of 10 calls, 15 interleaved
  rounds; p10 ms per call; "DFT" is the `METAL_PJRT_DISABLE_FFT=1`
  lowering, the dense DFT used before):

  | case | metal$fft | DFT | MLX |
  |---|---|---|---|
  | fft 17 x 4096 rows (Rader) | 0.062 | 0.157 | 0.065 |
  | fft 1000 x 256 (Stockham) | 0.059 | 1.265 | 0.051 |
  | fft 1024 x 256 | 0.057 | 1.300 | 0.049 |
  | fft 1021 x 256 (fused Bluestein) | 0.099 | 1.319 | 0.128 |
  | rfft 4096 x 64 | 0.054 | 1.831 | 0.042 |
  | irfft 4096 x 64 | 0.056 | 1.953 | 0.044 |
  | fft 2053 x 16 (multi-upload Bluestein) | 0.128 | 2.193 | 0.209 |
  | fft2 512 x 512 | 0.131 | 1.174 | 0.113 |
  | fft 2^16 x 16 (four-step) | 0.473 | - | 0.447 |
  | fft 2^20 (four-step) | 0.461 | - | 0.451 |
  | rfft 2^20 (four-step) | 0.372 | - | 0.360 |

  0.6-1.3x MLX's time: the small transforms (~0.05 ms) are
  dispatch-bound, and the Bluestein ones beat MLX (its multi-upload path
  chains ~8 elementwise ops, fused here into 3). The DFT (O(n^2), n x n
  twiddles, so not run above 4096) was 2.5-44x slower.

- Buffer donation: JAX dropped `donate_argnums` on mtl (copying the
  donated input) until the plugin added "mtl" to JAX's list of platforms
  with donation. Qwen3-0.6B decode with a 4096-slot KV cache donated each
  step: 39.8 -> 27.2 ms/token (measured in the "metal: case studies"
  work, which found it).

Where the time goes, in brief: memory-bound kernels run at memory bandwidth
(the emitted `x*2` kernel reaches the same ~81 GB/s as hand-written MSL);
fused reductions and optimizer updates are where XLA beats MLX; the gaps are
grouped / depthwise and 3-D convolutions (still on the loop emitter),
standalone softmax
(XLA's two fusions vs MLX's one kernel) and anything bound by per-dispatch
latency.

## Measured and rejected

One line each; the numbers and reasoning are in the archived log and the
commit messages.

- Indirect command buffers (cc0c5e2): replay costs 1.2-1.4 us of GPU time
  per dependent dispatch against 0.8 us for direct encoding.
- XLA command buffers as software replay (ab0dfcc): nanoGPT
  184 vs 196 ms, but loops 5-30% slower (scan 8.1 vs 6.2 ms); the 30-40 us
  per-launch cost it targeted was a 1 ms Synchronize sleep and a
  command-buffer submission storm, both fixed in the runtime.
- The `metal$softmax` rewriter (fcbf5ce): 1.5-1.8x on standalone softmax,
  but it never fires under autodiff (0 of 111 custom calls in the nanoGPT
  train step) and gained nothing where it fired. Standalone softmax
  8192x1024 went 1.24 -> 1.91 ms (MLX 0.97).
- f32 GEMMs with an epilogue on steel (7704d02, reverted): steel f32
  is 2-15% slower than MPS from batch*m*n*k ~ 1.7e10, 0-7% faster on mid
  shapes; nanoGPT unchanged. f32 stays MPS + a second pass (bitwise
  identical output).
- XLA's BFC allocator (6f98ea9, later replaced and removed): the runtime's size-class cache matches its step time (nanoGPT
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
- `const` on read-only kernel arguments (not committed):
  nanoGPT 178.9-179.7 vs 178.3-179.5 ms.
- An on-disk kernel cache / `MTLBinaryArchive`: Metal's
  system shader cache already compiles repeated MSL in ~1.3 ms (186 ms the
  first time), and JAX's persistent compilation cache skips it entirely.
- Per-encoder blame via `EncoderExecutionStatus` (d90c267): free (96-107 vs
  97-111 us per round trip), but a command buffer is mostly one compute
  encoder, so it cannot narrow a watchdog reset to a kernel.
- A configurable early-commit interval (`METAL_PJRT_EARLY_COMMIT_US`,
  f7ec606 removed it): 0 and 1e6 leave chain64's bimodality as it is (above);
  the 500 us constant stays.
