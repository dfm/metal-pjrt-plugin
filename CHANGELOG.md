# Changelog

Nothing is released yet (version 0.0.1, no PyPI release). This file keeps
the renames and dated decisions, so the reference docs in `docs/` can
describe the present only. Commit hashes point at the change.

## Unreleased

### 2026-09-28

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
