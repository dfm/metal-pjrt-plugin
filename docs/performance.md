# Performance

How the plugin is measured, what the numbers are, and what was tried and
dropped. The runtime policies behind the numbers (command-buffer batching,
the buffer cache) are in [`design.md`](design.md#runtime).

Every number here comes from one machine: an M3 MacBook Air (10-core GPU,
8 GB, no fan) on macOS 26.2.

## Methodology

- **Benchmarks.** `bench/jax_bench.py` runs the same jitted programs on any
  JAX backend (3 warm-up calls, then the median of 10 with
  `block_until_ready`). `bench/mlx_bench.py` runs the same workloads in MLX
  with `mx.compile`. `bench/linalg_bench.py` and `bench/tinygp_bench.py`
  cover linear algebra and tinygp's solvers, and `bench/dispatch_bound.py`
  times the fixed cost per call and chains of tiny kernels.
- **Rounds.** `bench/run_all.sh` runs `BENCH_ROUNDS` rounds (default 3),
  interleaving the backends (`BENCH_BACKENDS`, default
  `metal metal-gpu cpu mlx`) inside each round. `bench/report.py` writes
  the median over rounds to `bench/results/table.md`, which is generated
  locally and not tracked. Each row records the commit, the JAX and MLX
  versions, the plugin's platform version and the `METAL_PJRT_*`,
  `XLA_FLAGS` and `JAX_PLATFORMS` settings. The arms keep their names from
  before the platform rename: `metal` runs with `JAX_PLATFORMS=mtl`,
  `metal-gpu` is the same with the GPU trace, and `cpu` runs on CPU.
- **GPU time.** The `metal-gpu` arm is a second pass with
  `METAL_PJRT_TRACE=1`, which logs each command buffer's GPU start and end.
  The trace adds ~0.1-0.3 ms to sub-millisecond wall times, hence the
  separate pass. TFLOPS are computed from the best wall time.
- **A healthy GPU.** `run_all.sh` refuses to run after a GPU reset since
  boot (`scripts/gpu_health.py --strict`; `BENCH_ALLOW_DEGRADED=1`
  overrides), because a GPU that has been reset several times stays ~10x
  slower per dispatch until a reboot.
- **A quiet machine.** Run `bazel shutdown` first, or set
  `BENCH_BAZEL_SHUTDOWN=1` (`run_all.sh` doesn't do it by default, since it
  would stop a build running elsewhere). The Bazel server's memory makes
  macOS compress and swap, which skews timings.
- **Sub-millisecond A/Bs** interleave the arms and report p10 / median /
  p90. Tiny kernels run in one of two GPU performance states (~4.2 or
  ~2.3 us of GPU time per dispatch). macOS picks the state from the duty
  cycle, and no public API pins it: `dispatch_bound.py`'s 64-step chain
  flips between ~410 and ~720 us from call to call, so a single median is
  bimodal. The command-buffer pacing isn't the cause (both modes appear
  whether every op commits early or none does).
- **Thermals.** This laptop holds its top GPU performance state (~9 W)
  for about 3 minutes of sustained load, then settles near 5-6 W. Long
  comparisons therefore start with an untimed warm-up and interleave the
  arms, so that each side runs in the same thermal state.

## A snapshot across workloads

One run of `bench/run_all.sh` on 2026-09-27 (3 interleaved rounds; MLX's
column is from an earlier run without metadata). It hasn't been
regenerated since; the sections below are more recent. Medians, ms:

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

Later runs of the nanoGPT train step took 180-185 ms (one outlier at
200 ms) and its forward 58-59 ms. The cnn row predates the steel
convolutions (below).

In broad strokes: memory-bound kernels run at memory bandwidth (the
emitted `x*2` kernel reaches the same ~81 GB/s as hand-written MSL), and
XLA's fusion helps most on reductions fused with their producers and on
optimizer updates. The slow spots are grouped, depthwise and 3-D
convolutions (still on XLA's loop emitter), standalone softmax (two
fusions here, one kernel in MLX), and anything bound by per-dispatch
latency.

## Dispatch, sorting, solvers and memory

- **Fixed cost.** A tiny program (`jit(x*2+1)` on 1024 floats) takes
  ~170-210 us median from dispatch to result. A long chain of tiny kernels
  costs ~2-4 us of GPU time per dispatch; a 2000-step `scan` with tiny
  state takes ~8.5 ms.
- **Sort** (`jnp.sort`, f32, radix): 20k elements 0.31 ms, 1M 2.6 ms, 16M
  41 ms, 28-66x less time than the bitonic network it replaced. Rows of 64
  or fewer stay on the bitonic network.
- **tinygp** value and gradient, parallel solver: n = 20,000 in 7 ms,
  n = 200,000 in 50 ms (CPU: 77 and 703 ms when measured).
- **Small dense linear algebra.** Above 32x32 it runs in Accelerate's
  LAPACK on the host, after a full GPU synchronization. For n up to a few
  hundred that is ~10x slower than JAX on CPU: `cholesky` at n = 128 takes
  ~0.3 ms median (313 us, p90 680 us) against 0.03 ms on CPU, mostly two
  dependent GPU round trips (~130 us each) around the LAPACK call. From
  n ~ 2048 the two are about equal. Keep small factorizations on CPU if
  they dominate a program.
- **Memory.** A finished nanoGPT run idles at ~0.52 GB footprint (the
  buffer cache is released after ~2 s idle); the peak is ~1.9 GB.
- **Persistent compilation cache.** With it on, the first call of the
  nanoGPT training step in a new process takes ~0.42 s from the cache
  instead of ~0.75 s (an MLP train step: 41 ms instead of ~66 ms). Most
  mtl compiles take under a second, which is why the README suggests
  `JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0`.
- **Buffer donation.** Until the plugin added "mtl" to JAX's list of
  platforms with donation, JAX copied donated inputs on mtl. Qwen3-0.6B
  decode with a 4096-slot KV cache donated each step went from 39.8 to
  27.2 ms per token.

## Convolutions

Convolutions of 4 Mflop and more run on MLX's steel kernels (`metal$conv`,
`metal_pjrt/conv`; see [`integration-notes.md`](integration-notes.md)).

**Small CNN** (f32, batch 32; bursts of 30 calls, 6 interleaved rounds;
p10 / median / p90 ms). "Loop emitter" is the same build with
`METAL_PJRT_DISABLE_REWRITES=conv`, i.e. XLA's generic kernels:

| case | metal$conv | loop emitter | MLX |
|---|---|---|---|
| cnn fwd+bwd | 1.555 / 1.562 / 1.631 | 5.509 / 5.552 / 5.593 | 1.416 / 1.581 / 2.191 |
| cnn fwd 32x32x32 | 0.274 / 0.275 / 0.278 | 1.643 / 1.648 / 1.696 | 0.384 / 0.386 / 0.396 |
| cnn fwd+bwd, NCHW | 1.573 / 1.575 / 1.582 | 30.5 / 30.7 / 30.7 | - |
| cnn fwd, NCHW | 0.280 / 0.281 / 0.292 | 6.74 / 6.74 / 6.75 | - |

NCHW programs pay only for the rewriter's transposes to NHWC (~0.01-0.02
ms here). Per kernel (`bench/conv_bench` and `bench/conv_bench_mlx.py`,
p10 ms, plugin / MLX): conv1 (3->32) 0.092 / 0.098; conv2 (32->64,
stride 2) 0.134 / 0.128; conv2's input gradient 0.220 / 0.218; the two
weight gradients 0.322 / 0.252 and 0.576 / 0.525. In f16 the forward
takes 0.84-1.10x MLX's time.

**Training-size layers** (bf16, N = 1024, 3x3 SAME: the layers of the
CIFAR example's airbench94, `examples/cifar/conv_jax.py`, median ms). Two
changes on 2026-09-29 sped up the weight gradients: a vectorized unfold,
and sizing the GEMM tile and split-K together (64 x 64 tiles, at least
512 threadgroups, f32 partials capped at 8 MiB). Re-measured within 1% on
2026-10-01.

| weight gradient | before | after |
|---|---|---|
| 31x31, 24->64 | 53.8 | 18.2 |
| 15x15, 64->64 | 28.3 | 10.3 |
| 15x15, 64->256 | 53.0 | 26.3 |
| 7x7, 256->256 | 38.9 | 23.3 |
| 3x3, 256->256 | 7.3 | 4.5 |

Two later changes (2026-10-01, p10 of 5 interleaved rounds):

- Output channels that don't fill a column tile now use the specialized
  implicit-GEMM kernel with the weight's rows zero-padded to a whole tile.
  The 31x31 64->24 input gradient went from 20.66 to 12.79 ms.
- Max-pool gradients run as `metal$pool_max_bwd`, one pass over x, dy and
  dx, instead of XLA's select-and-scatter expansion. airbench94's four
  pools went from 25.3 to 7.9 ms per step (the largest, 31x31x64 2x2,
  from 11.8 to 3.24 ms, ~88 GB/s).

**Where an airbench94 step goes** (bf16, batch 1024, without
rematerialization, `METAL_PJRT_TRACE=1`, GPU at full clock, 2026-10-01):
308 ms of GPU time per step, with the GPU busy 99.9% of it. The
convolutions take 275 ms (89%): forwards 86.7 ms, input gradients 83.5 ms
and weight gradients 104.5 ms. The weight gradients run at 1.5-2.6
TFLOP/s and the rest at ~3.0. The remaining ~34 ms are the pools,
batch-norm statistics, GELU, the head and the optimizer.

The single command queue (2026-09-30) removed a gap in which the GPU
waited for the host at the start of each step: with a queue per stream
the GPU sat idle 60-67 ms per step (87% busy, 489-496 ms per step); with
one queue it's busy 99.2-100% (428-432 ms per step, both with
rematerialization). Once the laptop throttles, step time follows energy
per step, which this doesn't change. End-to-end training numbers are in
[`examples/cifar`](../examples/cifar).

**Small convolutions** (under 4 Mflop) stay on the loop emitter, which
fuses the convolution with what follows it into one kernel. A custom call
plus a separate activation kernel took about as long or longer there
(forward + relu, p10: 0.030-0.058 ms vs 0.029-0.037 for 0.3-2.4 Mflop).
From ~4.7 Mflop up, `metal$conv` takes less time (0.038 vs 0.046 ms; at 19
Mflop 0.036-0.087 vs 0.121-0.199). MLX runs these in 0.01-0.03 ms: the
difference is the fixed cost of a dispatch-bound program, not the kernels.

Not built: an implicit-GEMM weight gradient, which loads patches straight
from the input. It would save only the unfold (at most a third of the
31x31 layer's time, under 17% elsewhere) for a new loader and kernel
variant.

## Few-row GEMMs (small-batch decode)

For f16/bf16 x W^T with few rows, three rules apply: 2-8 rows with
K >= 512 run on MLX's wide gemv (`blas:gemv`); 9-48 rows (or K < 512) run
on a 16-row steel tile; and XLA's DotMerger pass is off.

Before these changes, 28 bf16 GEMMs of [M, 1024] x [1024, 6144] ran at
~23 GB/s for every M from 2 to 64. Half of that was DotMerger: the GEMMs
share x, so it concatenated the 28 weights into one operand, a 336 MB
copy on every call. The other half was steel's 64-row tile.

Method: p10 ms per call, 3 interleaved rounds of 3 bursts of 10
(`examples/llm/skinny_jax.py` and `skinny_mlx.py`). "List" runs the 28
GEMMs independently; "chained" feeds each one's output into the next, so
nothing can overlap or merge. The "after" and MLX list columns were
re-measured on 2026-10-01; the chained MLX column is from 2026-09-29.

| M | list: before | after | MLX | chained: before | after | MLX |
|---|---|---|---|---|---|---|
| 1 | 4.16 | 4.09 | 4.08 | 3.91 | 3.87 | 4.32 |
| 2 | 15.03 | 3.95 | 3.85 | 8.13 | 3.96 | 4.49 |
| 4 | 15.06 | 4.00 | 3.89 | 8.15 | 4.06 | 4.83 |
| 8 | 15.11 | 4.28 | 4.04 | 8.12 | 4.21 | 5.46 |
| 16 | 15.14 | 4.41 | 7.28 | 8.19 | 4.60 | 8.60 |
| 32 | 15.36 | 4.80 | 7.33 | 8.22 | 5.09 | 8.40 |
| 64 | 15.71 | 7.62 | 7.41 | 8.41 | 8.28 | 8.49 |

M = 1 is a reduction fusion and didn't change; M = 64 keeps the 64-row
tile. MLX uses its gemv for 2-15 rows and its 64-row tile from 16.

Why the limits are where they are:

- **16-row tile.** Per kernel (`bench/gemm_bench`, 28 cycled weights) it
  takes 1.3-1.9x less time than the 64-row tile up to M = 48, for NT and
  NN and for [N, 1024], [1024, 3072], [4096, 4096] and [288, 32768]
  weights.
- **8 rows for the gemv.** Against the 16-row tile, the gemv takes 6-40%
  less time up to 8 rows and more from 12 rows on (23% more on a
  151,936-row vocabulary matrix, and already 5% more there at 8). MLX
  uses it up to 15 rows.
- **No batch condition for the tile.** On decode attention (8 independent
  bf16 GEMMs of batch 1024, [m, 128] x [128, 128], as q K^T (NT) and p V
  (NN); 4 interleaved rounds), the tile rule takes 1.5-1.9x less time
  wherever it applies (NN 2-16, NT 16):

  | m | NT: no rule | rule | rule, no gemv | NN: no rule | rule |
  |---|---|---|---|---|---|
  | 1 | 3.28 | 3.41 | 3.40 | 3.31 | 3.13 |
  | 2 | 3.70 | 3.63 | 3.37 | 6.51 | 3.65 |
  | 4 | 5.47 | 6.29 | 3.73 | 6.68 | 3.51 |
  | 8 | 6.47 | 6.55 | 3.66 | 6.56 | 3.60 |
  | 16 | 6.01 | 4.01 | 4.01 | 6.64 | 3.99 |

  (m = 1 is a reduction fusion, and NT 2-8 is the gemv in the first two
  columns, so those differences are noise.)
- **K >= 512 for the gemv.** A sweep (NT, bf16, 8 independent GEMMs per
  call with distinct weights; p10 ms, 3 interleaved rounds; gemv / 16-row
  tile) shows that K decides, not the batch or the rows per batch:

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

  With K = 128 or 256 the gemv takes up to 2.8x longer once there's
  enough work to fill the GPU; from K = 512 the two are within 8% or the
  gemv is ahead, at every batch. So the gemv requires K >= 512
  (`kGemvMinK`). NN never takes the gemv (the same sweep over NN shows
  only noise). With this rule, the decode attention above (NT) went from
  6.90 / 6.82 to 3.44 / 3.58 ms at m = 4 / 8.

Qwen3-0.6B bf16 decode (`examples/llm/bench.py --max-len 256 --cases
decode_fused`, p10 ms per step, 3 interleaved rounds) at batch 1 / 2 / 4 /
8: 13.26 / 26.71 / 27.47 / 28.96 before, 13.20 / 13.88 / 15.11 / 17.11
after. Turning DotMerger off changed neither this model nor
`bench/jax_bench.py` (every case within noise).

## int4 LLM decode

Qwen3-1.7B with 4-bit weights in groups of 64 (`examples/llm`,
`Engine(max_len=256)`, context 128, 64 tokens; 2 rounds interleaved with
mlx-lm, GPU at full clock; 2026-10-01): 11.1-11.3 ms per token, and
mlx-lm 4-bit 11.6-11.7 ms. The GPU is busy 99.6% of the time (571 ops and
about one command buffer per token). XLA's row reductions over the int4
weights (the matvecs) take 10.6-10.8 of the 11.2 ms, at 62-96 GB/s per
shape; the other ~430 kernels (norms, RoPE, attention, cache updates,
sampling) take ~0.5 ms. Reading ~0.99 GB per token at the ~90 GB/s the
best shapes reach puts the floor at ~10.8 ms.

Not built: a dedicated quantized matvec (a port of MLX's qmv). At most
~0.4 ms per token is left above the floor, and MLX's kernel, timed on the
same chains of matvecs, took no less time than XLA's reductions on any of
the five shapes.

## FFT

FFTs run on MLX's FFT kernels (`metal$fft`, `metal_pjrt/fft`).
`bench/fft_bench.py`, one process, bursts of 10 calls, 15 interleaved
rounds, p10 ms per call. "DFT" is the dense O(n^2) lowering used before
(`METAL_PJRT_DISABLE_FFT=1`; not run above 4096):

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

The plugin takes 0.6-1.3x MLX's time. The small transforms (~0.05 ms) are
dispatch-bound. The Bluestein cases differ most: MLX's multi-upload path
chains ~8 elementwise ops, which XLA fuses into 3 here.

## Host memory per executable

A compiled executable holds host memory for as long as it lives, and JAX
keeps every executable of a live jitted function (a weak-keyed LRU behind
the function's trace cache). So `del` on the compiled object frees
nothing while the function lives; the memory comes back when the function
is deleted or after `jax.clear_caches()`. XLA:CPU behaves the same.

Measured on 2026-09-29 (`heap -s`, `malloc_history`, `vmmap` and
`footprint`) on a Qwen3-0.6B LoRA train step (bf16, gradient
checkpointing, batch 4, sequence length 97; ~3150 kernel thunks, 159
distinct kernels):

| per executable | MB |
|---|---|
| MSL, the kernel thunks' copies (one per thunk) | 66 |
| MSL, the executable's serialized thunks | 65 |
| rest of XLA: HLO module with fusion computations (~50), `ModuleAnnotations` (22), MLIR contexts (18), GEMM thunks (8), buffer assignment (4), ... | ~120 |
| outside XLA's compile (JAX lowering, Python objects) | ~40 |
| total malloc in use | 292 |

Since then emitted kernels carry a one-line stand-in for the shared MSL
prelude ([`design.md`](design.md#codegen)), which takes ~40 MB off the two
MSL rows: for 3 lengths, malloc in use went from 889 to 769 MB and
footprint from 2619 to 2503 MB. A loaded kernel (~40 KB) stays cached
only as long as its executable. One-time costs are Metal's shader-cache
index (~30 MB, on the first compile) and XLA's pooled MLIR contexts
(~15 MB).

`footprint` overstates what's live: after `jax.clear_caches()` malloc is
back to 133 MB (72 at load), while footprint stays ~300 MB above the load
point, because the malloc zone keeps the compile's freed pages dirty.
XLA:CPU shows the same. For training loops over many input shapes,
bucket the lengths: each distinct shape is its own executable (~230 MB
for this model).

## Measured and dropped

- **Indirect command buffers** (2026-09-26): replay cost 1.2-1.4 us of GPU
  time per dependent dispatch, against 0.8 us for direct encoding.
- **XLA command buffers as software replay** (2026-09-26): nanoGPT 184 vs
  196 ms, but loops 5-30% slower (scan 8.1 vs 6.2 ms). The per-launch cost
  it targeted was fixed in the runtime instead.
- **A `metal$softmax` rewriter** (2026-09-27): 1.5-1.8x on standalone
  softmax, but it never fired under autodiff (0 of 111 custom calls in
  the nanoGPT train step) and gained nothing where it did.
- **f32 GEMMs with an epilogue on steel** (2026-09-27): 2-15% slower than
  MPS on large shapes, 0-7% faster on mid-sized ones, nanoGPT unchanged.
  f32 stays on MPS plus a second pass (bitwise identical output).
- **XLA's BFC allocator** (2026-09-24): the runtime's size-class cache
  matches its step time (nanoGPT 178.7-180.0 vs 179.6-181.7 ms) and gives
  memory back (idle 0.5 vs 2.5-2.8 GB). With no cache at all, first-touch
  page faults (~60 us per MB) made steps 217-245 ms.
- **Host LAPACK as a stream host task** (2026-09-27): `cholesky` 128
  median 313 -> 299 us with the same p90, and a chain of 100 `cho_solve`
  64 12% slower. The two GPU round trips around the call remain.
- **Radix sort on short rows** (2026-09-27): one threadgroup per row was
  3-180x slower (1000x32: 942 vs 307 us), so short rows go to the bitonic
  network.
- **Host staging of host-to-device transfers** (2026-09-27): a 100 MB
  `device_put` took 9.4 vs 8.7 ms, and the path never triggered anyway.
- **`const` on read-only kernel arguments**: nanoGPT 178.9-179.7 vs
  178.3-179.5 ms.
- **An on-disk kernel cache** (`MTLBinaryArchive`): Metal's system shader
  cache already compiles repeated MSL in ~1.3 ms (186 ms the first time),
  and JAX's persistent cache skips compilation entirely.
- **Per-encoder blame** via `EncoderExecutionStatus` (2026-09-26): free,
  but a command buffer is mostly one compute encoder, so it can't narrow a
  watchdog reset down to a kernel.
- **A configurable early-commit interval** (2026-09-27): neither extreme
  changed the bimodal chain timings above, so the 500 us constant stays.
