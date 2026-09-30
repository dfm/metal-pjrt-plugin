# Development

How the repository is laid out, how to build and test it, and the knobs and
state the scripts use. The user-facing README covers installing and using
the plugin.

## Layout

- `MODULE.bazel`, `.bazelrc`: Bazel 8 / bzlmod setup mirroring jax-ml/jax at
  `jax-v0.11.2`, pinned to the same XLA commit jaxlib 0.11.2 was built from
  (`third_party/PINS.md` lists the pins and how to move them).
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
  MSL "steel" kernels (f16/bf16, applying BlasLt epilogues in their store),
  with MLX's wide gemv (`blas:gemv`, no XLA) for 2..8 rows of x W^T.
- `metal_pjrt/linalg/`: Cholesky, triangular solve and LAPACK-backed
  decompositions as FFI custom calls (Accelerate, or GPU kernels for small
  matrices).
- `metal_pjrt/ffi/`: FFI helpers and the scan, radix sort and Python
  callback handlers.
- The handlers are thin adapters (XLA types, attributes, the stream's
  device) over libraries with no XLA: `ffi:scan`, `ffi:radix_sort`,
  `linalg:small_linalg` (GPU, on `rt::Device`/`rt::Stream`) and
  `linalg:lapack_host` (Accelerate on host pointers). `metal_pjrt/conv/`
  (`conv:conv`, MLX's steel convolutions: path choice, implicit-GEMM
  kernels, unfold + GEMM, the weight gradient as patches x dY with split-K
  parts, launches bounded in flops) is one too, behind `metal$conv`, and
  so is `metal_pjrt/fft/` (`fft:fft_plan`, MLX's FFT plan and its Rader and
  Bluestein constants in double, host only; `fft:fft`, the Stockham, Rader,
  Bluestein and four-step kernels over contiguous complex64/float32 rows,
  in row chunks; behind `metal$fft`). Their tests, `conv_test`,
  `fft_test`, `scan_test`, `radix_sort_test`, `small_linalg_test` (device
  tests), `lapack_host_test` and `fft_plan_test` (host), link no XLA, so
  they build quickly and test a kernel in isolation;
  `//metal_pjrt:xla_free_test` fails if an XLA dependency creeps back in
  (or a Metal one into the host tests).
- `metal_pjrt/kernels/`: the hand-written MSL as `.metal` files (steel GEMM,
  steel convolutions, FFTs, radix sort, scan, small linear algebra, MPS staging, the runtime's
  fill/copy kernels, the emitter's prelude). A genrule (`embed_msl.bzl`)
  embeds each as a char array in the dylib; they are compiled at run time
  (`newLibraryWithSource`: the command-line tools have no offline `metal`
  compiler), so nothing ships besides the dylib, and the dylib's LC_UUID,
  which keys the compilation cache, covers them. `kernels_test` (a device
  test) compiles every source and creates a pipeline for every kernel the
  plugin can ask for.
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
scripts/run_jax_tests.sh tests/lax_test.py                     # JAX's own tests (a path in JAX's checkout, below), serialized; fails on failures not in scripts/jax_known_failures/
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
`jaxlib==0.11.2`; `metal_pjrt_plugin/__init__.py` warns at plugin discovery when
either version differs (`_JAX_VERSION`, checked against `pyproject.toml` by
`tests/test_packaging.py`). Checked by installing it with `uv` into a fresh
venv and running `tests/test_smoke.py`, `test_sort.py` and
`test_callbacks.py` from outside the checkout.

A fresh clone builds with `scripts/install_dev.sh` alone (73 s from the
shared disk cache; ~2 hours without it). The action environment is strict
(`--incompatible_strict_action_env`), so the cache survives shell and PATH
changes. Run `bazel shutdown` after a
build: the Bazel server holds several GB that programs then have to swap
for. Before deleting a scratch clone, run `bazel clean --expunge`
in it: its output base is separate (~8.5 GB).

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
holds memory that workloads then swap for.

## Environment variables

Every variable the plugin, tests and scripts read. Scope: "compile" ones
change the compiled program, are read once per process and are part of
the persistent-cache key (`PluginVersion` in
`stream_executor/metal_executor.cc`); "run" ones are read by the runtime;
"script" ones only by the scripts, tests or benchmarks. Audience: "user"
knobs are for anyone running the plugin, "dev" for working on it, "test"
for tests of the plugin itself, "internal" are set by the scripts.

Booleans are off when unset, empty, `0`, `false`, `no` or `off` (any case)
and on for anything else, in C++ (`EnvFlag`, `metal_pjrt/runtime/env.h`),
Python (`metal_pjrt_plugin._env_flag`) and the bench script alike. A number
that does not parse (or a size in MB too large to count in bytes) is
ignored with a warning and the default kept.

| variable | scope | audience | values and default | read in |
|---|---|---|---|---|
| `METAL_PJRT_MEMORY_FRACTION` | run | user | number > 0, default 1: scales the memory budget (half of RAM, capped by the GPU's recommended working set); beyond it allocations fail with RESOURCE_EXHAUSTED. Above 1 is allowed (with a warning), up to the working set | runtime, at device creation |
| `METAL_PJRT_DISABLE_REWRITES` | compile | dev | comma list, default empty: `scan` (the `metal$scan` rewriter), `cubsort` (XLA's SortRewriter and the radix sort; every sort then takes the bitonic network), `conv` (the `metal$conv` rewriter; every convolution then takes the loop emitter), `all`; other names are ignored with a warning | compiler |
| `METAL_PJRT_DISABLE_LAPACK` | compile | dev | boolean, default off; on: no LAPACK / small-matrix GPU linear algebra; XLA's expanders and JAX's generic lowerings instead | compiler and `_linalg_lowerings.py` |
| `METAL_PJRT_DISABLE_FFT` | compile | dev | boolean, default off; on: every FFT axis lowers to the dense DFT instead of `metal$fft` | `_lowerings.py` (the compiler reads it only for the cache key) |
| `METAL_PJRT_TRACE` | run | dev | boolean, default off; on logs one line per committed command buffer (op count, GPU time) | runtime |
| `METAL_PJRT_STATE_DIR` | run | dev | directory of the GPU reset log, default `~/.cache/metal-pjrt` (not the device lock, below) | runtime, `scripts/gpu_health.py` |
| `METAL_PJRT_QUARANTINE_STRIKES` | run | dev | resets since boot that quarantine a kernel, default 2; 0 disables | runtime, `scripts/gpu_health.py` |
| `METAL_PJRT_SNAPSHOT_MAX_MB` | run | test | largest `device_put` snapshotted instead of waited for, before the reclaimable/8 cap, default 256 | `pjrt/metal_pjrt_api.cc` |
| `METAL_PJRT_FAIL_COMMAND_BUFFER` | run | test | `n` fails the n-th committed command buffer, default 0 (never) | runtime |
| `METAL_TEST_REPORT_ULPS` | script | test | boolean, default off; print every measured error (with `pytest -s`) | `tests/metal_testing.py` |
| `JAX_TESTS_DIR` | script | dev | JAX checkout with the tests, default `~/.cache/metal-pjrt/jax-tests` | `scripts/run_jax_tests.sh` |
| `BENCH_BACKENDS`, `BENCH_ROUNDS`, `BENCH_ONLY`, `BENCH_ALLOW_DEGRADED`, `BENCH_BAZEL_SHUTDOWN` | script | dev | arms (default `metal metal-gpu cpu mlx`), interleaved rounds (3), case substrings, run despite a GPU reset since boot (boolean), `bazel shutdown` first (boolean, off) | `bench/run_all.sh` |
| `METAL_PJRT_DEVICE_LOCK_HELD` | script | internal | set by `scripts/device_lock.py` for its command: the holder's pid, which makes the lock re-entrant for descendants | `scripts/device_lock.py`, `tests/conftest.py` |
| `BENCH_OUT`, `BENCH_LABEL` | script | internal | set by `bench/run_all.sh`: the JSONL file and the backend label of the rows | `bench/*.py` |

Other tools' variables the scripts set or check:

- `JAX_PLATFORMS` (JAX): `mtl,cpu` selects the plugin (it is not JAX's
  default backend). `tests/conftest.py` and `scripts/run_jax_tests.sh`
  set it.
- `JAX_NUM_GENERATED_CASES` (JAX's tests): generated cases per test in
  `scripts/run_jax_tests.sh`, default 3 (the known-failures list assumes
  3).
- `PYTEST_TIMEOUT` (`scripts/run_jax_tests.sh`): seconds before a slow
  test's stack dump, default 180; the test is not stopped.
- `XLA_FLAGS` (XLA): `tests/conftest.py` refuses to run with it set, as it
  does with any `METAL_PJRT_*` setting other than `METAL_PJRT_STATE_DIR`,
  `METAL_PJRT_TRACE` and `METAL_PJRT_DEVICE_LOCK_HELD`. The plugin forces
  `xla_gpu_enable_triton_gemm=false` and `xla_gpu_dot_merger_threshold_mb=0`
  whatever `XLA_FLAGS` says (DotMerger would copy every weight matrix that
  shares an input on each call; `docs/integration-notes.md`).

## State

Two directories:

- `~/.cache/metal-pjrt/`: the plugin's and scripts' state (GPU reset log,
  device lock, JAX test checkout). `METAL_PJRT_STATE_DIR` moves the reset
  log only; `scripts/device_lock.py` always uses
  `~/.cache/metal-pjrt/device.lock`, since the GPU is one per machine. The
  lock script also takes the pre-rename `~/.cache/jax_metal` lock until the
  next pin bump (`CHANGELOG.md`).
- `~/.cache/metal-pjrt-plugin/`: Bazel's disk and repository caches, shared
  by every checkout (above).
