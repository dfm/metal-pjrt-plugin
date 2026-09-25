# metal-pjrt-plugin

A PJRT plugin that runs JAX on Apple Silicon GPUs by treating Metal as a
fourth XLA:GPU platform, alongside CUDA, ROCm and SYCL, rather than by
re-interpreting StableHLO op by op.

Status: first implementation pass, not yet compiled against XLA. See
`docs/design.md` for the plan, `docs/integration-notes.md` for the
source-verified contract with XLA, and `docs/mlx-comparison.md` for how this
differs from the MLX-based approaches.

## Layout

- `MODULE.bazel`, `.bazelrc`: Bazel 8 / bzlmod setup mirroring jax-ml/jax at
  `jax-v0.11.2`, pinned to the same XLA commit jaxlib 0.11.2 was built from.
- `third_party/`: root-module patches (abseil, protobuf, grpc) copied from JAX.
- `metal_pjrt_plugin/runtime/`: XLA-free C++ layer over Metal (metal-cpp):
  device, streams, events, kernels, allocations. Has a standalone smoke test.
- `metal_pjrt_plugin/stream_executor/`: the StreamExecutor platform.
- `metal_pjrt_plugin/compiler/`: `MetalCompiler : GpuCompiler` and the
  registrations (compiler, transfer manager, collectives stub, PJRT compiler).
- `metal_pjrt_plugin/codegen/`: MLIR -> EmitC -> MSL kernel emitter and
  `MetalKernelCompiler`, the `KernelCompiler` that `MetalCompiler` hands to
  XLA's emitters.
- `metal_pjrt_plugin/blas/`: GEMM via Metal Performance Shaders.
- `metal_pjrt_plugin/pjrt/`: the plugin dylib target.
- `jax_plugins/metal/`: Python registration package modeled on
  `jax_plugins/cuda`.
- `scripts/build_spike.sh`: staged overnight build that measures whether the
  XLA GPU stack compiles here without CUDA, and what it costs.

## Building

```
brew install bazelisk
scripts/install_dev.sh                       # builds the dylib, links it into jax_plugins/metal, pip install -e .
JAX_PLATFORMS=metal python scripts/smoke_test.py
bazel test --test_tag_filters=local //metal_pjrt_plugin/...   # device tests
```

`scripts/build_spike.sh` is the staged overnight build of XLA's GPU stack used
to establish that the dependency graph compiles here at all.

The `.bazelrc` is tuned for an 8 GB machine (3 jobs, 4.5 GB action budget,
2.5 GB JVM). Persistent disk and repository caches live under
`~/.cache/metal-pjrt-plugin/`. `--config=public_cache` reads JAX's public
Bazel cache; hits are opportunistic and depend on matching action keys.
