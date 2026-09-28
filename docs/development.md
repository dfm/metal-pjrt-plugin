# Development

How the repository is laid out, how to build and test it, and the knobs and
state the scripts use. The user-facing README covers installing and using
the plugin.

## Layout

- `MODULE.bazel`, `.bazelrc`: Bazel 8 / bzlmod setup mirroring jax-ml/jax at
  `jax-v0.11.2`, pinned to the same XLA commit jaxlib 0.11.2 was built from.
- `third_party/`: root-module patches (abseil, protobuf, grpc) copied from
  JAX, and the plugin's XLA patches (`third_party/xla/patches`).
- `metal_pjrt/runtime/`: XLA-free C++ layer over Metal (metal-cpp):
  device, streams, events, kernels, allocations. Has a standalone device test.
- `metal_pjrt/stream_executor/`: the StreamExecutor platform.
- `metal_pjrt/compiler/`: `MetalCompiler : GpuCompiler` and the
  registrations (compiler, transfer manager, collectives stub, PJRT compiler).
- `metal_pjrt/codegen/`: MLIR -> EmitC -> MSL kernel emitter and
  `MetalKernelCompiler`, the `KernelCompiler` that `MetalCompiler` hands to
  XLA's emitters.
- `metal_pjrt/blas/`: GEMM via Metal Performance Shaders (f32) and
  MSL "steel" kernels (f16/bf16, applying BlasLt epilogues in their store).
- `metal_pjrt/linalg/`: Cholesky, triangular solve and LAPACK-backed
  decompositions as FFI custom calls (Accelerate, or GPU kernels for small
  matrices).
- `metal_pjrt/ffi/`: FFI helpers and the scan, radix sort and Python
  callback handlers.
- `metal_pjrt/kernels/`: the hand-written MSL as `.metal` files (steel GEMM,
  radix sort, scan, small linear algebra, MPS staging, the runtime's
  fill/copy kernels, the emitter's prelude). A genrule (`embed_msl.bzl`)
  embeds each as a char array in the dylib; they are compiled at run time
  (`newLibraryWithSource`: the command-line tools have no offline `metal`
  compiler), so nothing ships besides the dylib, and the dylib's LC_UUID,
  which keys the compilation cache, covers them.
- `metal_pjrt/pjrt/`: the plugin dylib target and its `GetPjrtApi`.
- `metal_pjrt/xla_tripwire/`: snapshots of the XLA code the plugin
  relies on (`xla_tripwire_test`, host-only) and a list of every OneAPI branch
  in XLA (`oneapi_callsites.py`); run both after moving the XLA pin.
  `tests/test_jax_private_api.py` (host-only) is the same for the private
  JAX APIs the Python package calls or replaces.
- `metal_pjrt_plugin/`: the Python package (dist `metal-pjrt-plugin`, found
  by JAX through its `jax_plugins` entry point `mtl`), modeled on
  `jax_plugins/cuda`, plus lowerings and host callbacks.
- `tests/`: pytest suite (below); `scripts/jax_known_failures/`: the expected
  failures of JAX's own tests.
- `bench/`: benchmarks (Python, and the C++ dispatch microbenchmark).

## Building and testing

```
brew install bazelisk
scripts/install_dev.sh                       # builds the dylib, links it into metal_pjrt_plugin/, pip install -e .
scripts/device_lock.py -- .venv/bin/python -m pytest tests/test_smoke.py
scripts/device_lock.py -- .venv/bin/python -m pytest tests        # Python test suite (~1 min)
bazel test //metal_pjrt/...                                    # host-only C++ tests
scripts/device_lock.py -- bazel test //metal_pjrt:device_tests # C++ device tests, one at a time
scripts/run_jax_tests.sh tests/lax_test.py                     # JAX's own tests, serialized; fails on failures not in scripts/jax_known_failures/
bench/run_all.sh                                               # benchmarks vs cpu and MLX
scripts/build_wheel.sh                                         # dist/metal_pjrt_plugin-0.0.1-py3-none-macosx_26_0_arm64.whl, dylib inside
```

`install_dev.sh` creates `.venv` with uv, or without it with `python3.12`
(JAX 0.11.2 needs Python 3.12+). `run_jax_tests.sh` needs a checkout of
JAX's tests at the pinned version and the two packages they import, once:

```
git clone --depth 1 --branch jax-v0.11.2 https://github.com/jax-ml/jax ~/.cache/metal-pjrt/jax-tests
uv pip install --python .venv/bin/python absl-py hypothesis   # or .venv/bin/python -m pip install
```

(`JAX_TESTS_DIR` points it at another checkout.)

The wheel is pure Python plus the dylib (a copy, not the link), so it is
tagged `py3-none-macosx_26_0_arm64` and depends on `jax==0.11.2` and
`jaxlib==0.11.2`; `metal_pjrt_plugin/__init__.py` warns at import when
either version differs (`JAX_VERSION`, checked against `pyproject.toml` by
`tests/test_packaging.py`). Checked by installing it with `uv` into a fresh
venv and running `tests/test_smoke.py`, `test_sort.py` and
`test_callbacks.py` from outside the checkout.

A fresh clone builds with `scripts/install_dev.sh` alone (checked at
2d777cf: 73 s from the shared disk cache; ~2 hours without it). Before
deleting a scratch clone, run `bazel clean --expunge` in it: its output base
is separate (~8 GB).

`scripts/install_dev.sh` installs the `metal-pjrt-plugin` dist (editable,
with the `test` extra) into `.venv`, with
`metal_pjrt_plugin/pjrt_c_api_mtl_plugin.dylib` a link into
`bazel-bin`. The tests and scripts select the platform
themselves (`JAX_PLATFORMS=mtl,cpu`), since mtl is not JAX's
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

Run one GPU-heavy job at a time (the scripts take a device lock): two
processes can together over-commit memory, and a GPU stalled on swapped-out
pages trips the watchdog (`docs/design.md`, "Runtime"). The lock is re-entrant
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

## Environment variables

Every variable the plugin, tests and scripts read. "Compile-time" ones
change the compiled program and are part of the persistent-cache key
(`PluginVersion` in `stream_executor/metal_executor.cc`); the others are
read at run time.

| variable | read by | effect |
|---|---|---|
| `JAX_PLATFORMS` | JAX | `mtl,cpu` selects the plugin (it is not JAX's default backend) |
| `JAX_MTL_MEMORY_FRACTION` | runtime, at device creation | scales the memory budget (half of RAM, capped by the GPU's recommended working set); beyond it allocations fail with RESOURCE_EXHAUSTED |
| `METAL_PJRT_DISABLE_REWRITES` | compiler; compile-time, read once | comma list: `scan` (the `metal$scan` rewriter), `cubsort` (XLA's SortRewriter and the radix sort; every sort then takes the bitonic network), `all` |
| `METAL_PJRT_DISABLE_LAPACK` | compiler and Python lowerings; compile-time, read once | any value but `0`: no LAPACK / small-matrix GPU linear algebra; XLA's expanders and JAX's generic lowerings instead |
| `METAL_PJRT_TRACE` | runtime | `1` logs one line per committed command buffer (op count, GPU time) |
| `METAL_PJRT_STATE_DIR` | runtime, `gpu_health.py` | directory of the GPU reset log (default `~/.cache/metal-pjrt`) |
| `METAL_PJRT_QUARANTINE_STRIKES` | runtime, `gpu_health.py` | resets since boot that quarantine a kernel (default 2; 0 disables) |
| `METAL_PJRT_SYSTEM_MEMORY_RESERVE_MB` | runtime | memory the system guard keeps free (default 512; for tests) |
| `METAL_PJRT_SNAPSHOT_MAX_MB` | `pjrt/metal_pjrt_api.cc` | largest `device_put` snapshotted instead of waited for, before the reclaimable/8 cap (default 256; for tests) |
| `METAL_PJRT_FAIL_COMMAND_BUFFER` | runtime | `n` fails the n-th committed command buffer (tests of the error path) |
| `METAL_TEST_REPORT_ULPS` | `tests/metal_testing.py` | print every measured error (with `pytest -s`) |
| `JAX_MTL_DEVICE_LOCK_HELD` | `scripts/device_lock.py`, `tests/conftest.py` | set by the lock for its command: the holder's pid, which makes the lock re-entrant for descendants |
| `JAX_TESTS_DIR` | `scripts/run_jax_tests.sh` | JAX checkout with the tests (default `~/.cache/metal-pjrt/jax-tests`) |
| `JAX_NUM_GENERATED_CASES`, `PYTEST_TIMEOUT` | `scripts/run_jax_tests.sh` | JAX's generated cases per test (default 3); seconds before a slow test's stack dump (default 180; the test is not stopped) |
| `BENCH_BACKENDS`, `BENCH_ROUNDS`, `BENCH_ONLY`, `BENCH_ALLOW_DEGRADED`, `BENCH_BAZEL_SHUTDOWN` | `bench/run_all.sh` | arms (default `metal metal-gpu cpu mlx`), interleaved rounds (3), case substrings, run despite a GPU reset since boot, `bazel shutdown` first (off by default) |
| `BENCH_OUT`, `BENCH_LABEL` | `bench/*.py` | set by `run_all.sh`: the JSONL file and the backend label of the rows |

## State

State (GPU reset log, device lock, JAX test checkout) lives in
`~/.cache/metal-pjrt/`. Before the platform renames it was
`~/.cache/jax_metal/` (platform "metal"), then briefly `~/.cache/openmetal/`
(platform "openmetal", never released). `scripts/device_lock.py` (every
version since 34b8cd2) also takes the old `~/.cache/jax_metal` lock first
(creating it), and keeps doing so until the next pin bump, so a
`git bisect` to older commits still excludes this checkout.
