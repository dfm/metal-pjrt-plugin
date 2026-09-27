# Development

How the repository is laid out, how to build and test it, and the knobs and
state the scripts use. The user-facing README covers installing and using
the plugin.

## Layout

- `MODULE.bazel`, `.bazelrc`: Bazel 8 / bzlmod setup mirroring jax-ml/jax at
  `jax-v0.11.2`, pinned to the same XLA commit jaxlib 0.11.2 was built from.
- `third_party/`: root-module patches (abseil, protobuf, grpc) copied from
  JAX, and the plugin's XLA patches (`third_party/xla/patches`).
- `metal_pjrt_plugin/runtime/`: XLA-free C++ layer over Metal (metal-cpp):
  device, streams, events, kernels, allocations. Has a standalone device test.
- `metal_pjrt_plugin/stream_executor/`: the StreamExecutor platform.
- `metal_pjrt_plugin/compiler/`: `MetalCompiler : GpuCompiler` and the
  registrations (compiler, transfer manager, collectives stub, PJRT compiler).
- `metal_pjrt_plugin/codegen/`: MLIR -> EmitC -> MSL kernel emitter and
  `MetalKernelCompiler`, the `KernelCompiler` that `MetalCompiler` hands to
  XLA's emitters.
- `metal_pjrt_plugin/blas/`: GEMM via Metal Performance Shaders (f32) and
  MSL "steel" kernels (f16/bf16, applying BlasLt epilogues in their store).
- `metal_pjrt_plugin/linalg/`: Cholesky, triangular solve and LAPACK-backed
  decompositions as FFI custom calls (Accelerate, or GPU kernels for small
  matrices).
- `metal_pjrt_plugin/ffi/`: FFI helpers and the scan, radix sort and Python
  callback handlers.
- `metal_pjrt_plugin/pjrt/`: the plugin dylib target and its `GetPjrtApi`.
- `metal_pjrt_plugin/xla_tripwire/`: snapshots of the XLA code the plugin
  relies on (`xla_tripwire_test`, host-only) and a list of every OneAPI branch
  in XLA (`oneapi_callsites.py`); run both after moving the XLA pin.
- `jax_plugins/openmetal/`: Python registration package (dist
  `openmetal_pjrt_plugin`) modeled on `jax_plugins/cuda`, plus lowerings and host
  callbacks.
- `tests/`: pytest suite (below); `scripts/jax_known_failures/`: the expected
  failures of JAX's own tests.
- `bench/`: benchmarks (Python, and the C++ dispatch microbenchmark).

## Building and testing

```
brew install bazelisk
scripts/install_dev.sh                       # builds the dylib, links it into jax_plugins/openmetal, pip install -e .
scripts/device_lock.py -- .venv/bin/python -m pytest tests/test_smoke.py
scripts/device_lock.py -- .venv/bin/python -m pytest tests        # Python test suite (~1 min)
bazel test //metal_pjrt_plugin/...                             # host-only C++ tests
scripts/device_lock.py -- bazel test //metal_pjrt_plugin:device_tests   # C++ device tests, one at a time
scripts/run_jax_tests.sh tests/lax_test.py                     # JAX's own tests, serialized; fails on failures not in scripts/jax_known_failures/
bench/run_all.sh                                               # benchmarks vs cpu, jax-mps, MLX
scripts/build_wheel.sh                                         # dist/openmetal_pjrt_plugin-0.0.1-py3-none-macosx_26_0_arm64.whl, dylib inside
```

The wheel is pure Python plus the dylib (a copy, not the link), so it is
tagged `py3-none-macosx_26_0_arm64` and depends on `jax==0.11.2` and
`jaxlib==0.11.2`; `jax_plugins/openmetal/__init__.py` warns at import when
either version differs (`JAX_VERSION`, checked against `pyproject.toml` by
`tests/test_packaging.py`). Checked by installing it with `uv` into a fresh
venv and running `tests/test_smoke.py`, `test_sort.py` and
`test_callbacks.py` from outside the checkout.

Fresh-clone check (2026-09-27, at 2d777cf): `git clone` into a scratch
directory, a new `.venv`, `METAL_PJRT_STATE_DIR` pointing at an empty
scratch directory, then `scripts/install_dev.sh`: 73 s of Bazel (13790
actions from the shared disk cache, 5 run locally; a machine without the
cache builds XLA for ~2 hours), and `tests/test_smoke.py`,
`test_packaging.py` and `test_memory.py` passed under the device lock. Two
manual steps came up and are now in `install_dev.sh`: creating `.venv`
(without one it installed into whatever `python3` was on the path) and
installing pytest (the `test` extra). Clean up a scratch clone with
`bazel clean --expunge` in it before deleting it (its output base is
separate, ~8 GB). Only the build, not the uv venv creation, was re-run
after that change.

`scripts/install_dev.sh` installs the `openmetal_pjrt_plugin` package (editable,
with the `test` extra) into `.venv`, with
`jax_plugins/openmetal/pjrt_c_api_openmetal_plugin.dylib` a link into
`bazel-bin`. The tests and scripts select the platform
themselves (`JAX_PLATFORMS=openmetal,cpu`), since openmetal is not JAX's
default backend.

`tests/` is a pytest suite; tests that need the GPU are marked `metal` and
refuse to run outside `scripts/device_lock.py` (`-m "not metal"` runs the
rest anywhere). Numerics are compared with a float64 CPU reference in ulps
of the output dtype (`tests/metal_testing.py`), with tolerances about twice
the measured error; `METAL_TEST_REPORT_ULPS=1 ... pytest -s` prints the
measured errors (and CPU float32's, for comparison). No test timeouts:
killing a process with GPU work in flight can wedge the driver (for the
same reason child processes go through `metal_testing.run_python` and
`scripts/device_lock.py` waits for its command, never killing it), so slow
tests get a faulthandler stack dump and GPU hangs end through the runtime's
bounded waits.

Run one GPU-heavy job at a time (the scripts take a device lock); see
`docs/performance.md` for the memory policy and why. The lock is re-entrant
for descendants of its holder, so wrapping `scripts/run_jax_tests.sh` (which
locks itself) in `scripts/device_lock.py` is fine.

The `.bazelrc` is tuned for an 8 GB machine (3 jobs, 4.5 GB action budget,
2.5 GB JVM). Persistent disk and repository caches live under
`~/.cache/metal-pjrt-plugin/`, shared by every checkout of the repository,
so a second clone rebuilds from the cache in minutes instead of hours.
`--config=public_cache` reads JAX's public Bazel cache; hits are
opportunistic and depend on matching action keys. Shut the Bazel server
down (`bazel shutdown`) before measuring or running large workloads: its JVM
holds memory the allocation guard then refuses to hand out.

## Environment and state

`JAX_OPENMETAL_ALLOCATOR` (`platform`: the runtime's caching allocator, which
returns memory ~2 s after it is freed or on system memory pressure; `bfc`:
XLA's pool, which keeps it), `JAX_OPENMETAL_MEMORY_FRACTION` (scales the
memory budget, half of RAM; beyond it allocations fail with
RESOURCE_EXHAUSTED); the runtime's knobs are `METAL_PJRT_*`
(`docs/performance.md`).

State (GPU reset log, device lock, JAX test checkout) lives in `~/.cache/openmetal/` (`METAL_PJRT_STATE_DIR` moves
the reset log). Before the rename it was `~/.cache/jax_metal/`;
`scripts/device_lock.py` also takes the old lock there (creating it), so
older checkouts still exclude this one.
