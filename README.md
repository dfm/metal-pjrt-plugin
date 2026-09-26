# metal-pjrt-plugin

A PJRT plugin that runs JAX on Apple Silicon GPUs by treating Metal as a
fourth XLA:GPU platform, alongside CUDA, ROCm and SYCL, rather than by
re-interpreting StableHLO op by op.

Status: works end to end on an M3 for f32/f16/bf16 programs (JAX's
`lax_test.py`: 946 pass, 63 known failures in complex types, int4 and dot
precision algorithms); not packaged. See `docs/design.md` for the design,
`docs/integration-notes.md` for the source-verified contract with XLA,
`docs/op-coverage.md` for what runs, `docs/performance.md` for measurements,
`docs/roadmap.md` for what is next, and `docs/mlx-comparison.md` for how this
differs from the MLX-based approaches.

## Layout

- `MODULE.bazel`, `.bazelrc`: Bazel 8 / bzlmod setup mirroring jax-ml/jax at
  `jax-v0.11.2`, pinned to the same XLA commit jaxlib 0.11.2 was built from.
- `third_party/`: root-module patches (abseil, protobuf, grpc) copied from JAX.
- `metal_pjrt_plugin/runtime/`: XLA-free C++ layer over Metal (metal-cpp):
  device, streams, events, kernels, allocations. Has a standalone device test.
- `metal_pjrt_plugin/stream_executor/`: the StreamExecutor platform.
- `metal_pjrt_plugin/compiler/`: `MetalCompiler : GpuCompiler` and the
  registrations (compiler, transfer manager, collectives stub, PJRT compiler).
- `metal_pjrt_plugin/codegen/`: MLIR -> EmitC -> MSL kernel emitter and
  `MetalKernelCompiler`, the `KernelCompiler` that `MetalCompiler` hands to
  XLA's emitters.
- `metal_pjrt_plugin/blas/`: GEMM via Metal Performance Shaders (f32) and
  MSL "steel" kernels (f16/bf16), with BlasLt epilogues.
- `metal_pjrt_plugin/linalg/`: Cholesky, triangular solve and LAPACK-backed
  decompositions as FFI custom calls (Accelerate, or GPU kernels for small
  matrices).
- `metal_pjrt_plugin/ffi/`: FFI helpers and the softmax, scan and Python
  callback handlers.
- `metal_pjrt_plugin/pjrt/`: the plugin dylib target.
- `jax_plugins/metal/`: Python registration package modeled on
  `jax_plugins/cuda`, plus lowerings and host callbacks.
- `bench/`: benchmarks (Python, and the C++ dispatch microbenchmark).
- `scripts/build_spike.sh`: staged overnight build that measures whether the
  XLA GPU stack compiles here without CUDA, and what it costs.

## Building

```
brew install bazelisk
scripts/install_dev.sh                       # builds the dylib, links it into jax_plugins/metal, pip install -e .
JAX_PLATFORMS=metal python scripts/smoke_test.py
bazel test //metal_pjrt_plugin/...                             # host-only C++ tests
scripts/device_lock.py -- bazel test //metal_pjrt_plugin:device_tests   # C++ device tests, one at a time
scripts/run_jax_tests.sh tests/lax_test.py                     # JAX's own tests, serialized (slow tests get a stack dump, never a kill)
bench/run_all.sh                                               # benchmarks vs cpu, jax-mps, MLX
```

Run one GPU-heavy job at a time (the scripts take a device lock); see
`docs/performance.md` for the memory policy and why.

`scripts/build_spike.sh` is the staged overnight build of XLA's GPU stack used
to establish that the dependency graph compiles here at all.

The `.bazelrc` is tuned for an 8 GB machine (3 jobs, 4.5 GB action budget,
2.5 GB JVM). Persistent disk and repository caches live under
`~/.cache/metal-pjrt-plugin/`. `--config=public_cache` reads JAX's public
Bazel cache; hits are opportunistic and depend on matching action keys.
