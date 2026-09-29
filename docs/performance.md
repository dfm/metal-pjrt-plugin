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
- Weight gradients at CNN-training size (bf16, N = 1024, 3x3 SAME; the
  CIFAR case study's airbench94 layers, `examples/cifar/conv_jax.py`,
  median ms): a vectorized unfold (be3f9e7) and the GEMM tile and split-K
  sized together (f0aca7b: 64 x 64 tiles, >= 512 threadgroups, f32
  partials capped at 8 MiB). Forward and input gradients unchanged.

  | layer | before | after |
  |---|---|---|
  | 31x31, 24->64 | 53.8 | 18.2 |
  | 15x15, 64->64 | 28.3 | 10.3 |
  | 15x15, 64->256 | 53.0 | 26.3 |
  | 7x7, 256->256 | 38.9 | 23.3 |
  | 3x3, 256->256 | 7.3 | 4.5 |

  airbench94 end to end (M3 8 GB, nothing else on the GPU): mtl bf16
  94.03% in 228.8 s and 93.93% in 259.1 s (2 seeds; the system memory
  guard refused the other three, at 1.1-1.4 GB free), peak footprint 3.3
  GB; before, a steady-state step of 545 ms, ~259 s per run. PyTorch 2.14
  MPS fp16 in the same window: 93.93% +/- 0.11%, 253.9 +/- 20.0 s (5
  seeds), peak 5.9 GB. Not built: an implicit-GEMM weight gradient (the
  GEMM loads patches straight from the input); it would save only the
  unfold, at most a third of the 31x31 layer's time and under 17%
  elsewhere, for a new loader and kernel variant (~200 lines).
- Small convolutions stay on the loop emitter: under 4 Mflop a custom call
  plus a separate relu kernel lost to the loop emitter's one fused kernel
  (forward + relu, bursts of 30, p10 ms, two interleaved runs: 0.3-2.4
  Mflop 0.030-0.058 vs 0.029-0.037; C = 1 at 3.6 Mflop 0.121 vs 0.084),
  from ~4.7 Mflop metal$conv wins (0.038 vs 0.046; 19 Mflop 0.036-0.087 vs
  0.121-0.199). MLX runs these in 0.01-0.03 ms: the fixed cost of a
  dispatch-bound program, not the kernels.

- Few-row f16/bf16 GEMMs (small-batch LLM decode). Three changes:
  x W^T with 2..8 rows (either side) and K >= 512 runs on MLX's wide gemv
  (`blas:gemv`), 9..48 rows (or K < 512) on a 16-row steel tile, and
  XLA's DotMerger is off. Before, 28 bf16 GEMMs of [M, 1024] x [1024, 6144] ran at ~23
  GB/s for every M from 2 to 64. Half of that was DotMerger: the GEMMs
  share x, so it concatenated the 28 weights into one operand, a 336 MB
  copy on every call. The other half was steel's 64-row tile. p10 ms per
  call, 3 interleaved rounds of 3 bursts of 10 (the case study's
  `skinny_jax.py` / `skinny_mlx.py`; "chained" feeds each GEMM's output to
  the next, so neither side can overlap or merge them):

  | M | before | after | MLX | chained: before | after | MLX |
  |---|---|---|---|---|---|---|
  | 1 | 4.16 | 4.31 | 4.11 | 3.91 | 3.91 | 4.32 |
  | 2 | 15.03 | 4.47 | 4.07 | 8.13 | 4.03 | 4.49 |
  | 4 | 15.06 | 4.56 | 4.54 | 8.15 | 4.39 | 4.83 |
  | 8 | 15.11 | 4.91 | 4.86 | 8.12 | 4.30 | 5.46 |
  | 16 | 15.14 | 4.95 | 7.83 | 8.19 | 4.67 | 8.60 |
  | 32 | 15.36 | 4.98 | 7.54 | 8.22 | 5.49 | 8.40 |
  | 64 | 15.71 | 8.33 | 7.64 | 8.41 | 8.54 | 8.49 |

  M = 1 is a reduction fusion, unchanged; M = 64 keeps its 64-row tile.
  MLX runs 2..15 rows on the gemv and uses its 64-row tile from 16. Per
  kernel (`bench/gemm_bench`, 28 cycled weights), the 16-row tile is
  1.3-1.9x faster than the 64-row one up to M = 48, for NT and NN, and for
  [N, 1024], [1024, 3072], [4096, 4096] and [288, 32768] weights. The gemv
  beats it by 6-40% up to 8 rows, and loses from 12 rows on: by 23% on a
  151936-row vocabulary matrix, and by 5% there already at 8. Hence the
  8-row limit, not MLX's 15.
  Qwen3-0.6B bf16 decode (`examples/llm/bench.py --max-len 256 --cases
  decode_fused`, p10 ms per step, 3 interleaved rounds), batch 1 / 2 / 4
  / 8: before 13.26 / 26.71 / 27.47 / 28.96, after 13.20 / 13.88 / 15.11 /
  17.11. DotMerger does not touch that model (no difference with it off),
  and `bench/jax_bench.py` did not move with DotMerger off (4 interleaved
  rounds, every case within noise).
  The 16-row rule ignores the batch count, so it was also measured on
  decode attention: 8 independent bf16 GEMMs of batch 1024 (batch x
  heads), [m, 128] x [128, 128], as q K^T (NT) and p V (NN). p10 ms per
  call, 4 interleaved rounds of 3 bursts of 10, without the rule, with it,
  and with it but the gemv off:

  | m | NT: no rule | rule | rule, no gemv | NN: no rule | rule |
  |---|---|---|---|---|---|
  | 1 | 3.28 | 3.41 | 3.40 | 3.31 | 3.13 |
  | 2 | 3.70 | 3.63 | 3.37 | 6.51 | 3.65 |
  | 4 | 5.47 | 6.29 | 3.73 | 6.68 | 3.51 |
  | 8 | 6.47 | 6.55 | 3.66 | 6.56 | 3.60 |
  | 16 | 6.01 | 4.01 | 4.01 | 6.64 | 3.99 |

  m = 1 is a reduction fusion and NT 2..8 is the gemv in the first two
  columns, so those differences are noise. Where the rule applies (NN 2..16,
  NT 16) it is 1.5-1.9x faster, so it keeps no batch condition. The gemv
  was not: on 1024 batches of a 128-row matrix it was 1.7-1.8x slower than
  the 16-row tile at 4 and 8 vectors (and 8% at 2), about where the
  64-row tile was before.
  A sweep (NT, bf16, 8 independent GEMMs per call with distinct weights,
  fewer when they pass 512 MB; p10 ms, 3 interleaved rounds, gemv vs the
  16-row tile) shows K decides, not the batch or the rows per batch:

  | batch x [rows, K] | m = 2 | 4 | 8 |
  |---|---|---|---|
  | 1 x [128, 128] | 0.29 / 0.37 | 0.61 / 0.73 | 0.66 / 0.75 |
  | 1 x [1024, 128] | 0.39 / 0.39 | 0.80 / 0.57 | 0.75 / 0.67 |
  | 64 x [128, 128] | 1.13 / 0.52 | 1.42 / 0.86 | 1.35 / 1.06 |
  | 64 x [1024, 128] | 2.65 / 1.81 | 5.42 / 2.44 | 7.33 / 2.59 |
  | 64 x [256, 256] | 2.15 / 1.11 | 1.82 / 1.61 | 2.33 / 1.58 |
  | 1024 x [128, 128] | 3.74 / 3.30 | 5.45 / 3.38 | 6.46 / 3.59 |
  | 1024 x [256, 256] | 6.29 / 6.08 | 6.96 / 6.30 | 8.51 / 6.59 |
  | 1 x [128, 1024] | 0.33 / 1.03 | 0.67 / 1.38 | 0.65 / 1.39 |
  | 1 x [512, 512] | 0.36 / 0.65 | 0.65 / 1.01 | 0.74 / 0.92 |
  | 64 x [128, 512] | 1.11 / 1.22 | 1.44 / 1.86 | 1.95 / 1.95 |
  | 64 x [512, 512] | 3.14 / 3.41 | 3.72 / 4.02 | 4.22 / 4.24 |
  | 1024 x [512, 512] | 6.20 / 6.08 | 6.47 / 6.21 | 6.46 / 6.44 |
  | 1024 x [128, 1024] | 5.95 / 6.14 | 6.16 / 6.26 | 6.57 / 6.46 |
  | 64 x [1024, 1024] | 5.84 / 6.23 | 6.11 / 6.46 | 6.36 / 6.64 |
  | 64 x [6144, 1024] | 8.68 / 9.13 | 8.89 / 9.34 | 9.30 / 9.76 |

  With K = 128 or 256 the gemv loses up to 2.8x once there is work to
  fill the GPU (it wins only single tiny GEMMs, by 0.01-0.02 ms each);
  from K = 512 it ties (within 8%) or wins, at every batch. Hence the
  gemv needs K >= 512 (`kGemvMinK`), one condition. NN never takes the
  gemv (the same sweep over NN only shows noise). The decode attention
  above, NT: 6.90 / 6.82 ms -> 3.44 / 3.58 at m = 4 / 8 (4.19 -> 3.67 at
  2); the case-study bench (K = 1024) and Qwen3-0.6B decode at batch 1 /
  2 / 4 / 8 (13.26 / 13.95 / 15.04 / 17.11 -> 13.26 / 13.98 / 15.11 /
  17.09) do not change (3 interleaved rounds).

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

## Host memory per executable

A compiled executable holds host memory for as long as it lives, and JAX
keeps every executable a live jitted function compiled (pxla's
`_cached_compilation`, a weak-keyed LRU behind the function's trace cache):
`del` of the compiled object frees nothing while the function lives. It
comes back when the function is deleted or after `jax.clear_caches()`; XLA:CPU
behaves the same. Measured (2026-09-29) with `heap -s` (malloc'd bytes in
use), `malloc_history -callTree` under `MallocStackLogging=lite`, `vmmap
--summary` and `footprint`, on a Qwen3-0.6B LoRA train step (bf16, grad
checkpointing, batch 4, lengths 97/129/161: ~3150 kernel thunks each, of
which XLA's KernelReuseCache leaves 159 distinct kernels at length 97):

| per executable (length 97) | MB |
|---|---|
| MSL, the kernel thunks' copies (one per thunk) | 66 |
| MSL, `GpuExecutable`'s serialized thunks (kept for serialization) | 65 |
| rest of XLA: HLO module with fusion computations (~50), `ModuleAnnotations` (22), MLIR dialects in pooled contexts (18), GEMM thunks (8), buffer assignment (4), ... | ~120 |
| outside XLA's compile (JAX lowering, Python objects) | ~40 |
| total malloc in use (272 on average over the three lengths) | 292 |

With the prelude out of each kernel's binary (`docs/design.md`,
Compiler), the two MSL rows drop by ~40 MB: 3 lengths compiled, malloc
in use 889 -> 769 MB and footprint 2619 -> 2503 MB (1714 MB after loading
the model). Once an executable has run, its kernels (pipeline, library,
MSL cache key: ~40 KB per distinct kernel) are cached by the runtime until
the executable goes; before 2026-09-29 they stayed for the process (400
kernels after the 3 lengths; 1200 in a synthetic 6 x 300-fusion test).
One-time costs: Metal's shader-cache index (~30 MB, on the first
compile) and XLA's pooled MLIR contexts (~15 MB).

`footprint` overstates what is live: after `jax.clear_caches()` malloc is
back to 133 MB (72 at load) while footprint stays ~300 MB above the load
point, because the malloc zone keeps the compile's freed pages dirty (82%
fragmentation; `malloc_zone_pressure_relief` returns nothing). XLA:CPU
shows the same (a synthetic program: malloc back to +3 MB, footprint +73
MB). The persistent compilation cache is off by default and not involved.
For training loops over many batch shapes, bucket the lengths: each
distinct shape is an executable (~230 MB for this model).

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
