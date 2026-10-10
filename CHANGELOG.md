# Changelog

This file keeps the renames and dated decisions, so the reference docs in
`docs/` can describe the present only. Dates in parentheses are when a
change was made.

## Unreleased

### 2026-10-09

- mtl is JAX's default backend once the plugin is installed and a Metal
  device can be created, as JAX's other GPU plugins are (it was opt-in, a
  priority below CPU's). Programs it can't run raise an error, as on cuda
  or tpu; `JAX_PLATFORMS=cpu` keeps CPU. Without a usable Metal device JAX
  still falls back to CPU quietly.

## 0.0.2 (2026-10-09)

`metal-pjrt-plugin` 0.0.2 and `metal-pjrt-core` 0.0.2. Silent wrong
answers fixed (8-bit bit counts, gathers with clamped indices, short-row
argsort feeding a gather, `tridiagonal_solve` needing pivoting), complex64
scatters with repeated indices, and `jax.Device.memory_stats()`. Tested on
the GPU with jax 0.10.0, 0.10.2, 0.11.2 and nightly (2026-10-09): this
package's suite and 44 of JAX's test files, whose remaining failures are
listed in `scripts/jax_known_failures/`. Known: nested `dynamic_slice` with
out-of-range starts gives XLA's pre-fix answer, as XLA:CPU does in jax 0.11.2
(docs/accuracy.md).

### 2026-10-09

- float32 `lax.linalg.tridiagonal_solve` runs Accelerate's `sgtsv`
  (`metal$lapack_gtsv`), which pivots, as JAX's CPU and CUDA lowerings do.
  It used JAX's generic Thomas algorithm, which does not, so a system
  needing pivoting (a zero or small leading pivot) silently returned NaN.
  complex64 still takes the generic path.
- docs/accuracy.md: converting a float32 subnormal to bfloat16 flushes it to
  zero on Metal (so `jnp.spacing` of bfloat16 subnormals is 0).

### 2026-10-08

- Short-row sorts (the bitonic network, rows of up to 64) run one kernel
  per network step, as designed: XLA had fused the whole network into the
  sort's consumer, which then recomputed every earlier step per element
  (2^steps work), in a form the Metal compiler miscompiled.
  `v[:, jnp.argsort(s, descending=True)]` picked wrong columns, and with
  it complex64 `jnp.linalg.svd` (JAX's QDWH path) returned wrong factors.
  An optimization barrier after each step keeps them apart.
- That change broke `jnp.argsort(x, stable=True, descending=True)` of
  8-bit keys (bool, int8) in rows whose length is not a power of two, and
  so `jnp.compress`/`jnp.extract` with `size=` on such rows: the first
  network step read its padded, reversed input through two inlined copies
  of one helper (self and partner), which the Metal compiler miscompiled.
  The network's inputs are now materialized too, so every step reads
  plain arrays.
- Three silent wrong answers fixed, found by running JAX's own tests:
  `lax.population_count`/`clz` (and `jnp.bitwise_count`) on 8-bit
  integers counted the sign-extended 32-bit value; and a gather whose
  computed indices are clamped (`x[::-2]` through JAX's gather indexing,
  `jnp.compress`/`jnp.extract` with `size=`) read zeros past its first row,
  from a Metal compiler miscompile of the clamp's compare and select.
  Integer min/max and bit counts now go through the MSL prelude's wrappers
  of Metal's builtins.
- `jax.Device.memory_stats()` works on mtl (it returned `None`): bytes in
  use and their peak, allocations, the budget as `bytes_limit`, and the
  cached buffers in `pool_bytes`. The plugin answers
  `PJRT_Device_MemoryStats` from its runtime; no XLA patch. The LLM
  example reports peak device memory with it, and the CIFAR example's
  memory profile no longer uses the `metal_pjrt_memory_stats` test hook.
- complex64 scatters with repeated indices that add or overwrite work
  (they were refused: XLA combines complex elements with a 64-bit
  compare-and-swap). An add becomes two f32 scatter-adds
  (`MetalComplexScatterSplitter`); an overwrite becomes one 8-byte store
  in the kernel lowering. So complex LU and what is built on it (`solve`,
  `inv`, `det`, `slogdet`), densifying complex sparse arrays and gradients
  through complex indexing work. Other
  combiners (multiply) are still refused without `unique_indices`.

## 0.0.1 (2026-10-03)

The first release: `metal-pjrt-plugin` 0.0.1 and `metal-pjrt-core` 0.0.1
on PyPI. Everything below led up to it.

### 2026-10-02

- Two packages: `metal-pjrt-plugin`, the pure-Python frontend, and
  `metal-pjrt-core`, the dylib alone. They are separate wheels,
  checked against each other at startup by a frontend ABI version
  (`docs/development.md`, "Packages and releases"). The frontend pins the
  one core it was tested with, and requires `jax`/`jaxlib >=0.10.0` (no
  upper bound) instead of exactly 0.11.2: today's dylib
  passed the test suite with every JAX from 0.10.0 to a 0.12 nightly, once
  host callbacks handled jax 0.10.0's `mlir.set_sharding`. The development
  link to the dylib moved to `core/metal_pjrt_core/`.

### 2026-10-01

- Convolutions whose output channels do not fill a kernel column tile
  (e.g. 24) take the specialized implicit-GEMM kernel, with the weight's
  rows zero-padded in the workspace, instead of the general one: the
  airbench94 31x31 64->24 input gradient 20.7 -> 12.8 ms.
- Max-pool gradients over non-overlapping windows (JAX's `reduce_window`
  max with stride = window, VALID) run as one kernel, `metal$pool_max_bwd`,
  bit-identical to the expansion it replaces: airbench94's four pools 25.3
  -> 7.9 ms. `METAL_PJRT_DISABLE_REWRITES=pool` turns it off.
- Fixed: `jax.device_put(x, s)` inside `jit`, with `s` a `pinned_host` (or
  `device`) sharding, ignored the memory kind: the result came back in
  device memory. It now lowers as on CUDA, to XLA's host offloading.

### 2026-09-30

From a pre-release review of the whole tree.

- Fixed: `jax.grad` with respect to the kernel of a grouped or depthwise
  convolution was wrong (relative error ~0.5). JAX expresses that gradient
  as a convolution with `batch_group_count`, which XLA's loop emitter sums
  over every batch group (on CUDA cuDNN takes these). They are now
  converted to ordinary convolutions first, as on XLA:CPU.
- Fixed: `expm1` returned `inf` for x between 84.3 and 88.72.
- XLA's dynamic-slice fusion is off: `x.at[:n].set(x[:n] @ r)` compiled to
  a matmul writing in place over its own operand.
- Refused by name at compile time, where they failed later or worse:
  host offloading (`jax.experimental.compute_on("device_host")`, a process
  abort), `jax.lax.rng_uniform`, and a matmul bias that XLA fused through a
  slice or reshape of the matmul's result.
- Complex64 convolutions are expanded into real ones (JAX's cpu/gpu rule),
  so they reach the convolution kernels; a complex64-result matmul of two
  real operands works.
- Licensing: the kernel prelude's `erfc` (from Numerical Recipes) is gone;
  `THIRD_PARTY_NOTICES` carries the license text of every project linked
  into the library (`scripts/gen_third_party_notices.py`); every source
  file has an SPDX header; the copyright holder is "The metal-pjrt-plugin
  Authors" (`AUTHORS`).
- `docs/archive/` is no longer in the tree, and the docs cite dates
  instead of commit hashes.
- Fixed: a transfer queued behind a long backlog of computation could keep
  a GPU command buffer waiting for that whole backlog, which counts against
  the GPU watchdog (a GPU reset if it runs long enough). Seen with
  `jax.device_put(x, may_alias=False)` of an array still being computed,
  and the same of an empty array. Transfers now wait for earlier work on
  the CPU instead.
- The runtime keeps one Metal command queue per device (it had one per
  XLA stream): no command buffer ever waits on the GPU for another one, so
  no transfer, event or copy can sit on the GPU watchdog's clock behind
  queued work. After a GPU failure, pending host transfers are skipped.
  The reset quarantine is now opt-in (`METAL_PJRT_QUARANTINE_STRIKES=n`),
  and a command buffer that ran over 1 s is logged with its kernels.
- Fixed: `max` / `min` (and `relu`, max/min reductions) picked their left
  operand on ties, so `relu(-0.0)` was `-0.0`; they now follow IEEE
  754-2019 like XLA:CPU (-0 < +0, NaN propagates).
- `jax.lax.mulhi` on 64-bit integers (which XLA computes in 128-bit
  integers) is refused by name at compile time; it failed with an obscure
  emitter error.
- Host linear algebra uses Accelerate's current LAPACK (3.9+, the
  `$NEWLAPACK` symbols) instead of the deprecated LAPACK 3.2.1; results
  now equal JAX's CPU backend in the eigh tests. `eigh` and `svd` of an
  input with NaN or inf return NaN without calling LAPACK. The LAPACK
  handlers check their result buffers' types and sizes (a mismatched
  `ffi_call` wrote past a buffer).

### 2026-09-29

- Memory limits work like PyTorch MPS: the per-process budget (half of
  RAM, capped by the GPU's recommended working set) is the limit, and
  within it macOS compresses and swaps as for any program. The system
  memory guard, which refused allocations leaving less than 512 MB of
  free pages (25 of 28 airbench94 runs on an 8 GB Mac at 50-70% free), is
  gone; an allocation is now refused only while the system is at critical
  memory pressure, and the first allocation at warning logs a warning.
  `METAL_PJRT_SYSTEM_MEMORY_RESERVE_MB` is no longer read. The test
  hook `metal_pjrt_memory_pressure` is now
  `metal_pjrt_testing_memory_pressure` and also fakes the level
  allocations see.
- Convolution weight gradients are 1.6-3x faster at CNN-training sizes
  (bf16, batch 1024: 31x31 24->64 53.8 -> 18.2 ms): the patch unfold is
  vectorized and the split-K GEMM's tile and part count are chosen
  together. airbench94 (CIFAR-10 to 94%) now runs in 229-259 s, on par
  with PyTorch MPS (254 s) at 3.3 vs 5.9 GB peak. docs/performance.md has
  the per-layer numbers.
- Out-of-memory errors name the computation whose allocation was refused
  (`in jit_f: ...`), also when the error surfaces in a later operation
  that consumed its result; before, XLA's message named that later
  operation.
- Kernels the compiler emitted now go with their executables: dropping
  an executable releases its Metal pipelines, libraries and the MSL text
  the kernel cache kept (~40 KB per kernel; before, they stayed until the
  process exited). JAX's caches hold every executable a live jitted
  function compiled, so the memory comes back when the function is
  deleted or after `jax.clear_caches()`. Library kernels (steel, FFI,
  built-ins) still stay for the process. `metal_pjrt_memory_stats` (test
  hook) also reports the cached kernels and their MSL bytes.
- Each compiled executable takes ~15% less host memory: emitted kernels
  carry a one-line `#include <metal_pjrt/msl_prelude.metal>` instead of
  the ~9 KB MSL prelude, which the runtime puts back when it compiles the
  kernel. XLA keeps two copies of every kernel thunk's source per
  executable; a Qwen3-0.6B LoRA train step (~3150 kernels) drops from 272
  to 232 MB of malloc'd memory. MSL dumps (`--xla_dump_to`) have the
  prelude expanded, so a dumped `.metal` file compiles on its own.
- Accuracy policy decided: mtl matches XLA:CPU, including its flushing of
  subnormal inputs and outputs. The subnormal `log` / `log2` / `log10`
  fix (2026-09-27) is reverted: `log(1e-40)` is `-inf` again, as on CPU, and
  an ALU-bound chain of 16 logs is back to 0.79 ms p10 from 0.95.
  `cbrt` of a subnormal returns it unchanged, as CPU does (it gave the
  true cube root, 4.6e-14 for 1e-40), and `pow(-inf, 0.5)` is inf, as on
  CPU (NaN before). The small-argument exp / sin / cos polynomials and
  cbrt's Newton step stay: they bring mtl closer to CPU.
  `docs/accuracy.md` lists the remaining differences.
- complex64 matmul and sort work (they were refused at compile time):
  a new pass, `MetalComplexDotExpander`, turns each complex dot into four
  real f32 dots on the usual GEMM / loop paths (as accurate as CPU), and
  complex sorts run on the bitonic network, bit-identical to CPU. A
  1024^2 c64 matmul takes 3.9 ms (3.9x the f32 one; CPU 17 ms), a
  1024 x 1024 c64 sort 26 ms (CPU 21 ms). Complex convolutions, cholesky,
  triangular_solve, qr and ragged_dot work with them; complex LU (solve,
  inv, det) is still refused (its scatter). JAX's lax_test: the 3 complex
  known failures now pass.
- Few-row f16/bf16 matmuls (small-batch LLM decode) are 1.8-3.4x faster,
  at or above MLX's speed. x W^T with 2..8 rows runs on MLX's wide gemv,
  9..48 rows on a 16-row steel tile, and XLA's DotMerger is off: it
  copied the weights of dots sharing an input into one operand on every
  call. Qwen3-0.6B bf16 decode at batch 2: 26.7 -> 13.9 ms per step
  (batch 1: 13.3). docs/performance.md has the numbers.
- The wide gemv needs K >= 512; below that (e.g. decode attention's
  head_dim of 128) the 16-row steel tile runs it: batched q K^T, 1024 x
  [4..8, 128] x [128, 128]^T, 6.9 -> 3.5 ms per 8 GEMMs.
- FFTs run on MLX's FFT kernels (`metal$fft`, one call per transformed
  axis) instead of the dense O(n^2) DFT: every length up to 2^24 (powers
  of two) or 2^23 - 1, complex64 / float32, 0.6-1.3x MLX's time and 2.5-44x
  faster than the DFT on `bench/fft_bench.py`. Longer lengths raise
  NotImplementedError (the DFT would need an n x n matrix); the DFT stays
  for `METAL_PJRT_DISABLE_FFT=1` (every axis), complex128 and symbolic
  batch dims (`jax.export`), up to n = 46340. The HLO `fft` op (from
  programs lowered for another platform) is refused at compile time
  instead of reaching XLA's cuFFT-only thunk; a module exported for
  ("cpu", "mtl") runs, since the check comes after its platform
  conditional is folded.
- Clearer messages for a complex scatter without `unique_indices` and for
  complex128 data movement.

### 2026-09-28

- complex64 works: the MSL emitter carries it as `float2` in buffers,
  constants, shared memory and loop-carried values, so complex arrays can
  be returned from `jit`, passed between kernels, reduced and transferred,
  and FFT results and gradients work. Complex matmuls and sorts are refused
  at compile time naming the op (they used to fail in the kernel
  translator), as is a complex scatter without `unique_indices`;
  complex128 is refused with f64. 46 of JAX's 62 known `lax_test.py`
  failures now pass.
- f16/bf16 conv weight gradients could be wrong (off by up to 1e26) when
  the f32 partials buffer was misaligned; fixed by aligning it to 256
  bytes (2026-09-28).
- Convolutions run on MLX's steel convolution kernels (`metal$conv`,
  MetalConvRewriter): 1-D and 2-D, f32/f16/bf16 (f16/bf16 accumulate in
  f32), forward, input and weight gradients, of 4 Mflop and more. The cnn
  fwd+bwd bench went from 5.5 to 1.56 ms (MLX 1.42).
  `METAL_PJRT_DISABLE_REWRITES=conv` restores the loop emitter.
- Buffer donation works on mtl: JAX lowers `donate_argnums` only for the
  platforms in a private list (`mlir._platforms_with_donation`) and before
  this silently copied donated inputs on mtl; the plugin now adds "mtl" to
  it at initialization. Qwen3-0.6B decode with a 4096-slot KV cache: 39.8
  -> 27.2 ms/token.
- Renamed for release (2026-09-28): JAX platform `"mtl"` (was `"openmetal"`),
  Python package `metal_pjrt_plugin` (was `jax_plugins/openmetal`), PyPI
  dist `metal-pjrt-plugin` (was `openmetal_pjrt_plugin`), plugin library
  `pjrt_c_api_mtl_plugin.dylib`, environment variables `JAX_MTL_*` (were
  `JAX_OPENMETAL_*`; `METAL_PJRT_*` unchanged). The C++ Bazel tree moved
  from `metal_pjrt_plugin/` to `metal_pjrt/` (2026-09-27), and the state
  directory from `~/.cache/openmetal/` to `~/.cache/metal-pjrt/` (2026-09-28).
  The XLA-internal names stay "METAL".
- Every environment variable the plugin defines is `METAL_PJRT_*`:
  `JAX_MTL_MEMORY_FRACTION` and `JAX_MTL_DEVICE_LOCK_HELD` are now
  `METAL_PJRT_MEMORY_FRACTION` and `METAL_PJRT_DEVICE_LOCK_HELD`, with no
  alias (the old names are ignored) (2026-09-28).
- Decisions (`docs/roadmap.md`): a jax/jaxlib version mismatch warns and
  loads; the softmax rewriter stays deleted; int8 GEMM stays refused; the
  system memory guard stays strict; Metal's transcendental bias is
  accepted as documented; quarantine stays per boot. The no-wait
  `CheckInFlight` is deferred until a profile shows it matters.
- The FFI handlers' work moved into libraries with no XLA
  (`ffi:scan`, `ffi:radix_sort`, `linalg:small_linalg`,
  `linalg:lapack_host`, over `runtime:kernel_launch`), each with its own
  test: `scan_test`, `radix_sort_test` (replaces `cub_sort_test`, which
  drove the handlers through XLA's FFI), `small_linalg_test` and the host
  `lapack_host_test`. `xla_free_test` keeps them XLA-free (2026-09-28). The Cholesky, triangular-solve and LU handlers now look up the
  stream's Metal context before the empty-batch early return, so a call on
  a non-Metal stream fails even with an empty batch (2026-09-28).
- The hand-written MSL moved out of C++ string literals into `.metal` files
  under `metal_pjrt/kernels/` (2026-09-28), with a device test that compiles
  every kernel (2026-09-28).
- Error messages say what to do, in user terms and in MB; environment
  variables share one parser for booleans and numbers (2026-09-28). With `JAX_PLATFORMS` unset, a failure to start the mtl backend
  no longer stops CPU programs (2026-09-28). The implementation modules are
  private (`_callbacks`, `_lowerings`, `_linalg_lowerings`).

### 2026-09-27

- Platform renamed from `"metal"` to `"openmetal"` (2026-09-27): `"metal"` is
  Apple's closed-source jax-metal plugin. `"openmetal"` was replaced by the
  names above before any release. The state directory was
  `~/.cache/jax_metal/` under `"metal"`; `scripts/device_lock.py` (every
  version since 2026-09-27) also takes the old `~/.cache/jax_metal` lock, so a
  `git bisect` to older commits still excludes a current checkout. That
  lock goes at the next pin bump, not before 2026-10-31.
- Decisions:
  - The platform is opt-in: CPU stays JAX's default backend (2026-09-27).
  - The plugin never sets `jax_compilation_cache_dir` or the cache
    thresholds (2026-09-27): the setting is process-wide and `initialize()`
    runs for every installed plugin, so a default directory turned the
    cache on for CPU-only users too.
  - Land the minimum viable product before numerical side quests; known
    accuracy gaps are listed in `docs/accuracy.md`, not chased.
  - The size-class buffer cache is the only device allocator; the BFC
    option is gone (2026-09-27). f16/bf16 GEMMs run only on steel, with its
    out-of-range shapes refused at compile time (2026-09-27). MLX is the one
    benchmark reference; the jax-mps arm is gone (2026-09-27).
  - The `metal$softmax` rewriter was removed after an end-to-end A/B
    (2026-09-27; `docs/performance.md`, "Measured and rejected").
- A scripted wheel with the dylib inside (2026-09-27).

### 2026-09-23 to 2026-09-26

The original milestones: XLA's GPU compiler built on macOS without CUDA, a
StreamExecutor platform over metal-cpp (2026-09-23), MLIR -> EmitC -> MSL
codegen, GEMM via MPS and steel, JAX-side lowerings (2026-09-24), XLA FFI and
Python callbacks on Metal (2026-09-25), JAX's persistent compilation
cache without an XLA patch (2026-09-26), and XLA command buffers, built and
then removed to keep the platform small (2026-09-26).
