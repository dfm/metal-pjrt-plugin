# Changelog

Nothing is released yet (version 0.0.1, no PyPI release). This file keeps
the renames and dated decisions, so the reference docs in `docs/` can
describe the present only. Commit hashes point at the change.

## Unreleased

### 2026-09-29

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
  bytes (ab3e2d0).
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
- Renamed for release (c652306): JAX platform `"mtl"` (was `"openmetal"`),
  Python package `metal_pjrt_plugin` (was `jax_plugins/openmetal`), PyPI
  dist `metal-pjrt-plugin` (was `openmetal_pjrt_plugin`), plugin library
  `pjrt_c_api_mtl_plugin.dylib`, environment variables `JAX_MTL_*` (were
  `JAX_OPENMETAL_*`; `METAL_PJRT_*` unchanged). The C++ Bazel tree moved
  from `metal_pjrt_plugin/` to `metal_pjrt/` (6aa0587), and the state
  directory from `~/.cache/openmetal/` to `~/.cache/metal-pjrt/` (881a2f8).
  The XLA-internal names stay "METAL".
- Every environment variable the plugin defines is `METAL_PJRT_*`:
  `JAX_MTL_MEMORY_FRACTION` and `JAX_MTL_DEVICE_LOCK_HELD` are now
  `METAL_PJRT_MEMORY_FRACTION` and `METAL_PJRT_DEVICE_LOCK_HELD`, with no
  alias (the old names are ignored) (6d28c62).
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
  `lapack_host_test`. `xla_free_test` keeps them XLA-free (88be444 to
  3a80152). The Cholesky, triangular-solve and LU handlers now look up the
  stream's Metal context before the empty-batch early return, so a call on
  a non-Metal stream fails even with an empty batch (880c55b).
- The hand-written MSL moved out of C++ string literals into `.metal` files
  under `metal_pjrt/kernels/` (f59719c), with a device test that compiles
  every kernel (318c2ba).
- Error messages say what to do, in user terms and in MB; environment
  variables share one parser for booleans and numbers (ce99e8f to
  d2f4ccf). With `JAX_PLATFORMS` unset, a failure to start the mtl backend
  no longer stops CPU programs (d2f4ccf). The implementation modules are
  private (`_callbacks`, `_lowerings`, `_linalg_lowerings`).

### 2026-09-27

- Platform renamed from `"metal"` to `"openmetal"` (36cc906): `"metal"` is
  Apple's closed-source jax-metal plugin. `"openmetal"` was replaced by the
  names above before any release. The state directory was
  `~/.cache/jax_metal/` under `"metal"`; `scripts/device_lock.py` (every
  version since 34b8cd2) also takes the old `~/.cache/jax_metal` lock, so a
  `git bisect` to older commits still excludes a current checkout. That
  lock goes at the next pin bump, not before 2026-10-31.
- Decisions:
  - The platform is opt-in: CPU stays JAX's default backend (3c3e3e7).
  - The plugin never sets `jax_compilation_cache_dir` or the cache
    thresholds (bbb3c65): the setting is process-wide and `initialize()`
    runs for every installed plugin, so a default directory turned the
    cache on for CPU-only users too.
  - Land the minimum viable product before numerical side quests; known
    accuracy gaps are listed in `docs/accuracy.md`, not chased.
  - The size-class buffer cache is the only device allocator; the BFC
    option is gone (b0019d2). f16/bf16 GEMMs run only on steel, with its
    out-of-range shapes refused at compile time (cc33e52). MLX is the one
    benchmark reference; the jax-mps arm is gone (23bc8fd).
  - The `metal$softmax` rewriter was removed after an end-to-end A/B
    (fcbf5ce; `docs/performance.md`, "Measured and rejected").
- A scripted wheel with the dylib inside (2d777cf).

### 2026-09-23 to 2026-09-26

The original milestones: XLA's GPU compiler built on macOS without CUDA, a
StreamExecutor platform over metal-cpp (9d3388b), MLIR -> EmitC -> MSL
codegen, GEMM via MPS and steel, JAX-side lowerings (77d0d33), XLA FFI and
Python callbacks on Metal (dd160e8, e13204b), JAX's persistent compilation
cache without an XLA patch (e21d29d), and XLA command buffers, built and
then removed to keep the platform small (ab0dfcc, bdb9c4c).
`docs/archive/` has the dated record.
