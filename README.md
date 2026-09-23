# metal-pjrt-plugin

A PJRT plugin that runs JAX on Apple Silicon GPUs by treating Metal as a
fourth XLA:GPU platform, alongside CUDA, ROCm and SYCL, rather than by
re-interpreting StableHLO op by op.

Status: build spike. Nothing runs yet. See `docs/design.md` for the plan and
`docs/mlx-comparison.md` for how this differs from the MLX-based approaches.

## Layout

- `MODULE.bazel`, `.bazelrc`: Bazel 8 / bzlmod setup mirroring jax-ml/jax at
  `jax-v0.11.2`, pinned to the same XLA commit jaxlib 0.11.2 was built from.
- `third_party/`: root-module patches (abseil, protobuf, grpc) copied from JAX.
- `metal_pjrt_plugin/`: the plugin sources (StreamExecutor Metal platform,
  `MetalCompiler`, PJRT C API entry point). Stubs for now.
- `jax_plugins/metal/`: Python registration package modeled on
  `jax_plugins/cuda`.
- `scripts/build_spike.sh`: staged overnight build that measures whether the
  XLA GPU stack compiles here without CUDA, and what it costs.

## Building

```
brew install bazelisk
caffeinate -is nohup scripts/build_spike.sh > /dev/null 2>&1 &
tail -f build-logs/spike-*.log
```

The `.bazelrc` is tuned for an 8 GB machine (3 jobs, 4.5 GB action budget,
2.5 GB JVM). Persistent disk and repository caches live under
`~/.cache/metal-pjrt-plugin/`. `--config=public_cache` reads JAX's public
Bazel cache; hits are opportunistic and depend on matching action keys.
