# Development

Layout, building, testing, and the knobs and state the scripts use. For
installing and using the plugin, see the README.

## Layout

- `MODULE.bazel`, `.bazelrc`: Bazel 8 / bzlmod, mirroring jax-ml/jax at
  `jax-v0.11.2` and the same XLA commit (`third_party/PINS.md` lists the
  pins and how to move them).
- `third_party/`: patches copied from JAX (abseil, protobuf, grpc) and
  the plugin's XLA patches (`third_party/xla/patches`).
- `metal_pjrt/runtime/`: an XLA-free C++ layer over Metal (metal-cpp):
  device, streams, events, kernels, allocations.
- `metal_pjrt/stream_executor/`: the StreamExecutor platform.
- `metal_pjrt/compiler/`: `MetalCompiler : GpuCompiler` and its
  registrations.
- `metal_pjrt/codegen/`: the MLIR -> EmitC -> MSL kernel emitter
  (`MetalKernelCompiler`).
- `metal_pjrt/blas/`: GEMMs on Metal Performance Shaders (f32) and MSL
  "steel" kernels (f16/bf16, with fused epilogues), plus MLX's wide gemv
  for 2..8 rows.
- `metal_pjrt/linalg/`: Cholesky, triangular solve and LAPACK-backed
  decompositions (Accelerate, or GPU kernels for small matrices).
- `metal_pjrt/ffi/`: FFI helpers and the scan, radix sort and Python
  callback handlers.
- `metal_pjrt/conv/`, `metal_pjrt/fft/`: MLX's convolutions and FFTs,
  behind `metal$conv` and `metal$fft`.
- `metal_pjrt/kernels/`: the hand-written MSL, embedded in the dylib and
  compiled at run time (the command-line tools have no offline `metal`
  compiler). The dylib's LC_UUID, which keys the compilation cache, covers
  them. `kernels_test` compiles every one.
- `metal_pjrt/pjrt/`: the plugin dylib and its `GetPjrtApi`.
- `metal_pjrt/xla_tripwire/`: snapshots of the XLA code the plugin relies
  on (`xla_tripwire_test`) and a list of XLA's OneAPI branches
  (`oneapi_callsites.py`); run both after moving the XLA pin.
  `tests/test_jax_private_api.py` does the same for private JAX APIs.
- `metal_pjrt_plugin/`: the Python package (entry point `mtl`), modeled on
  `jax_plugins/cuda`, plus lowerings and host callbacks.
- `core/`: the `metal-pjrt-core` package, which holds only the dylib
  ([Packages and releases](#packages-and-releases)).
- `tests/`: the pytest suite; `scripts/jax_known_failures/`: expected
  failures of JAX's own tests.
- `bench/`: benchmarks.

The FFI handlers are thin adapters over libraries with no XLA dependency
(`ffi:scan`, `ffi:radix_sort`, `linalg:small_linalg`,
`linalg:lapack_host`, `conv:conv`, `fft:fft_plan`, `fft:fft`), whose
tests build quickly and test one kernel in isolation.
`//metal_pjrt:xla_free_test` keeps XLA out of them.

## Building and testing

```
brew install bazelisk
scripts/install_dev.sh                       # builds the dylib, links it into core/metal_pjrt_core/, pip install -e core -e .
scripts/device_lock.py -- .venv/bin/python -m pytest tests/test_smoke.py
scripts/device_lock.py -- .venv/bin/python -m pytest tests        # Python test suite (~1 min)
bazel test //metal_pjrt/...                                    # host-only C++ tests
scripts/device_lock.py -- bazel test //metal_pjrt:device_tests # C++ device tests, one at a time
scripts/run_jax_tests.sh tests/lax_test.py                     # JAX's own tests (a path in JAX's checkout, below), serialized; fails on failures not in scripts/jax_known_failures/
bench/run_all.sh                                               # benchmarks vs cpu and MLX
scripts/build_wheel.sh                                         # dist/: metal_pjrt_core-*-py3-none-macosx_26_0_arm64.whl (the dylib), metal_pjrt_plugin-*-py3-none-any.whl
```

`install_dev.sh` creates `.venv` (with uv, or `python3.12`), installs
both packages editable (the frontend with its `test` extra), and links the
dylib from `bazel-bin` into `core/metal_pjrt_core/`. Tests and scripts set `JAX_PLATFORMS=mtl,cpu` themselves.

`run_jax_tests.sh` needs JAX's tests at the pinned version, once:

```
git clone --depth 1 --branch jax-v0.11.2 https://github.com/jax-ml/jax ~/.cache/metal-pjrt/jax-tests
uv pip install --python .venv/bin/python absl-py hypothesis   # or .venv/bin/python -m pip install
```

### Build times and caches

A cold build took 95 minutes on an 8 GB M3 (2026-10-01). The disk and
repository caches in `~/.cache/metal-pjrt-plugin/` are shared by every
checkout, so later clones build in about a minute, and the strict action
environment (`--incompatible_strict_action_env`) keeps them valid across
shell and PATH changes. `--config=public_cache` exists but has no entries
for this build (see CI below).

Run `bazel shutdown` after building and before measuring: the server
holds several GB. Run `bazel clean --expunge` before deleting a scratch
clone; its ~8.5 GB output base lives elsewhere.

`.bazelrc` is tuned for 8 GB (3 jobs, 4.5 GB action budget, 2.5 GB JVM).
With more memory, override it in `user.bazelrc`, e.g. `common --jobs=8`
and `common --local_resources=memory=20000` on 32 GB.

### Packages and releases

There are two packages, released separately:

- **`metal-pjrt-plugin`** (the root `pyproject.toml`): the Python frontend
  users install, a pure-Python `py3-none-any` wheel. It holds the
  `jax_plugins` entry point, the lowerings and the host callbacks. It
  requires `jax`/`jaxlib` at or above the lowest version tested, with no
  upper bound, and exactly one `metal-pjrt-core`, the build it was tested
  with. It uses private `jax._src` APIs, so a new JAX can break it:
  testing against JAX nightly says when a new release is needed. An older
  JAX loads with a warning.
- **`metal-pjrt-core`** (`core/`): only the dylib, tagged
  `py3-none-macosx_26_0_arm64`, with no Python dependencies (it does not
  link Python; callbacks go through a ctypes trampoline). It reaches
  jaxlib only through the PJRT C API, StableHLO version negotiation and
  XLA FFI, so one build serves many JAX releases. Release it when the
  C++ changes, or for an XLA pin bump, and always with a frontend release
  that pins it: a frontend never picks up a core it wasn't tested with.
  Publish only its wheel, never an sdist: building one would make a wheel
  without the dylib.

The private contract between the two (platform name, client options, FFI
targets and their attributes, the callback trampoline, environment
variables read on both sides) has one version number,
`metal_pjrt_frontend_abi_version()` in `metal_pjrt/pjrt/metal_pjrt_api.cc`,
matched by `_CORE_ABI_VERSION` in `metal_pjrt_plugin/__init__.py`. Bump
both on any incompatible change. The exact pin already keeps released
pairs together; the check catches the rest (an editable install against a
stale build, hand-installed wheels): with a mismatch the frontend logs why
and does not register the platform.

The lowest JAX is `_JAX_MIN` in `metal_pjrt_plugin/__init__.py`, the same
as `pyproject.toml` (`tests/test_packaging.py` checks). To test another
JAX, install it with the current `metal-pjrt-core` in a scratch venv and
run `tests/test_jax_private_api.py` (no GPU) and the whole suite. On
2026-10-02, 0.10.0, 0.10.1, 0.10.2, 0.11.0, 0.11.1, 0.11.2 and
0.12.0.dev20261001 passed it, apart from a test of the old exact pin;
after the split, the installed wheels passed it under 0.10.0 and the
nightly, and the editable install under 0.11.2.
Before 0.10, `jax._src` APIs the frontend uses change; a few tests
compare bit for bit with XLA:CPU, which differs before 0.11.2, and skip
there (`OLD_CPU_REFERENCE` in `tests/metal_testing.py`).

### Releasing

Releases go to PyPI from `.github/workflows/release.yml` with trusted
publishing: no tokens are stored anywhere. Both PyPI projects trust that
workflow in the `pypi` environment, which only `v*` tags can deploy to and
which waits for a maintainer's approval.

1. Bump `version` in `pyproject.toml`. If anything the library is built
   from changed since the last core release (`metal_pjrt/`, `core/`,
   `third_party/`, `MODULE.bazel*`, `.bazelrc`, `.bazelversion`), also
   bump `core/pyproject.toml` and the frontend's `metal-pjrt-core==` pin;
   the workflow refuses to reuse a published core whose sources changed.
2. Run the GPU suite locally as a pre-check, and merge to `main`.
3. Wait for CI on that commit to finish (it saves the Bazel cache the
   release's core build reads; tagging earlier can mean a cold, five-hour
   build), then tag that commit of `main` and push the tag:
   `git tag v<version> && git push origin v<version>`.
4. The workflow checks the tag (it matches the frontend's version and is
   on `main`), that the version is new on PyPI and that the core pin
   matches (`scripts/release_plan.py`, which also runs locally:
   `REF=refs/tags/v<version> scripts/release_plan.py`); builds
   `metal-pjrt-core` only if its version is new; installs
   the wheels in a fresh venv and runs `pytest -m "not metal"` against
   them; then waits for approval.
5. Before approving, test what will be published: download the run's
   `wheel-*` artifacts (`gh run download <run-id> -p 'wheel-*'`), install
   them in a fresh venv, and run the GPU suite under
   `scripts/device_lock.py` from outside the checkout. Check the wheels'
   SHA-256 against the test job's log (the publish job prints them again
   as the record of what was uploaded). If core was not rebuilt, the
   pinned one comes from PyPI.
6. Approve. Core is uploaded first, then the frontend (never an sdist).
   If the frontend's upload fails, "Re-run failed jobs" retries it.

Run the workflow by hand from the Actions tab for a dry run: everything
but publishing. Do one before the first release.

### Tests

GPU tests are marked `metal` and refuse to run outside
`scripts/device_lock.py`; `-m "not metal"` runs the rest anywhere.
Numerics are compared against float64 CPU in ulps, with tolerances about
twice the measured error (`METAL_TEST_REPORT_ULPS=1 ... pytest -s` prints
them).

There are no test timeouts, because killing a process with GPU work in
flight can wedge the driver. Slow tests get a stack dump instead, and the
runtime's waits are bounded.

Run one GPU-heavy job at a time: two processes can over-commit memory,
and a GPU stalled on swapped pages trips the watchdog
([`design.md`](design.md#runtime)). The device lock is re-entrant for the
holder's descendants.

## Continuous integration

`.github/workflows/ci.yml` builds the plugin and runs the host-only tests
on GitHub's `macos-26` arm64 runners (3 cores, 7 GB RAM; about 95 GB of free disk, so no cleanup is needed):
`bazel build //metal_pjrt/...`, `bazel test //...` (`device_tests` is
`manual`) and `pytest tests -m "not metal"`. Anything that needs the GPU
stays local, under `scripts/device_lock.py`.

- **Cost guard.** Every job has
  `if: ${{ !github.event.repository.private }}`, since GitHub bills macOS
  minutes on private repositories. `tests/test_packaging.py` fails if a
  job lacks it.
- **Triggers.** Pushes to `main`, pull requests and manual runs, except
  changes only to Markdown, `docs/` or `examples/`. A newer push cancels a
  pull request's run; `main` runs are never cancelled.
- **Caching.** The disk and repository caches go in the Actions cache,
  keyed `bazel-macos26-xcode<X>-<hash of MODULE.bazel.lock, .bazelversion,
  third_party/PINS.md and .bazelrc>-<run id>` and restored by longest
  prefix. Only `main` and manual runs save; releases only restore. The
  setup shared by `ci.yml` and `release.yml` (pinning Xcode,
  the cache key and restore) is one composite action,
  `.github/actions/setup-bazel`. Xcode is pinned there, and the strict
  action environment fixes PATH, so
  action keys don't depend on the runner. Before saving,
  `scripts/prune_disk_cache.py` trims the disk cache to 7 GB, least
  recently used first, to fit the free 10 GB quota (Bazel's own GC only
  runs when the server is idle). Older entries are deleted after a
  successful save.
- **Memory.** The job builds with `--config=ci`: two jobs within 3.5 GB,
  and without the bytes (`--remote_download_outputs=toplevel`: cache hits
  stay in the disk cache instead of being copied into the output base). A
  warm build plus tests used about 10 GB of disk (2026-10-02).
- **Cold cache.** A cold build exceeds one job, so the build step stops
  at 300 minutes, saves the cache, and the next run (a push to `main` or a
  manual run) continues.
- **No remote cache.** JAX's public cache had no entries for this build
  (2026-09-30: 0 hits of 189 LLVM and 325 XLA actions): it's written from
  Linux only, and these macOS actions use Xcode's clang.
- **Not in CI yet:** testing the frontend against JAX nightly. The wheels
  are built and tested by the release workflow
  ([Releasing](#releasing)).

## Environment variables

"compile" variables change the compiled program and are part of the
persistent-cache key; "run" ones are read by the runtime; "script" ones
only by scripts and tests. Booleans are off when unset, empty, `0`,
`false`, `no` or `off` (any case). A number that doesn't parse is ignored
with a warning.

| variable | scope | audience | values and default | read in |
|---|---|---|---|---|
| `METAL_PJRT_MEMORY_FRACTION` | run | user | number > 0, default 1: scales the memory budget (half of RAM, capped by the GPU's recommended working set); beyond it allocations fail with RESOURCE_EXHAUSTED. Values above 1 are allowed (with a warning), up to the working set | runtime, at device creation |
| `METAL_PJRT_DISABLE_REWRITES` | compile | dev | comma list, default empty: `scan` (the `metal$scan` rewriter), `cubsort` (XLA's SortRewriter and the radix sort; every sort then takes the bitonic network), `conv` (the `metal$conv` rewriter; every convolution then takes the loop emitter), `pool` (the `metal$pool_max_bwd` rewriter; max-pool gradients then take XLA's select-and-scatter expansion), `all`. Other names are ignored with a warning | compiler |
| `METAL_PJRT_DISABLE_LAPACK` | compile | dev | boolean, default off. On: no LAPACK or small-matrix GPU linear algebra; XLA's expanders and JAX's generic lowerings instead | compiler and `_linalg_lowerings.py` |
| `METAL_PJRT_DISABLE_FFT` | compile | dev | boolean, default off. On: every FFT axis lowers to the dense DFT instead of `metal$fft` | `_lowerings.py` (the compiler reads it only for the cache key) |
| `METAL_PJRT_TRACE` | run | dev | boolean, default off. On: one log line per committed command buffer (op count, GPU time) | runtime |
| `METAL_PJRT_DEBUG_FREE_QUARANTINE` | run | dev | boolean, default off. On: freed device buffers aren't reused until 64 later frees and 100 ms have passed, and a pointer into one is reported with the backtrace of its free (for hunting use-after-free bugs) | runtime, at device creation |
| `METAL_PJRT_STATE_DIR` | run | dev | directory of the GPU reset log, default `~/.cache/metal-pjrt` (not of the device lock, below) | runtime, `scripts/gpu_health.py` |
| `METAL_PJRT_QUARANTINE_STRIKES` | run | dev | resets since boot that quarantine a kernel (refused until a reboot or `scripts/gpu_health.py --clear`), default 0 (off) | runtime, `scripts/gpu_health.py` |
| `METAL_PJRT_SNAPSHOT_MAX_MB` | run | test | largest `device_put` that is snapshotted instead of waited for, before the reclaimable/8 cap, default 256 | `pjrt/metal_pjrt_api.cc` |
| `METAL_PJRT_FAIL_COMMAND_BUFFER` | run | test | `n` fails the n-th committed command buffer, default 0 (never) | runtime |
| `METAL_TEST_REPORT_ULPS` | script | test | boolean, default off: print every measured error (with `pytest -s`) | `tests/metal_testing.py` |
| `METAL_TEST_REPORT_ERRORS` | script | test | on when set to anything (even `0`), default off: print the measured FFT errors | `metal_pjrt/fft/fft_test.cc` |
| `JAX_TESTS_DIR` | script | dev | JAX checkout with the tests, default `~/.cache/metal-pjrt/jax-tests` | `scripts/run_jax_tests.sh` |
| `BENCH_BACKENDS`, `BENCH_ROUNDS`, `BENCH_ONLY`, `BENCH_ALLOW_DEGRADED`, `BENCH_BAZEL_SHUTDOWN` | script | dev | arms (default `metal metal-gpu cpu mlx`), interleaved rounds (3), case substrings, run despite a GPU reset since boot (boolean), `bazel shutdown` first (boolean, off) | `bench/run_all.sh` |
| `METAL_PJRT_DEVICE_LOCK_HELD` | script | internal | set by `scripts/device_lock.py` for its command: the holder's pid, which makes the lock re-entrant for descendants | `scripts/device_lock.py`, `tests/conftest.py` |
| `BENCH_OUT`, `BENCH_LABEL` | script | internal | set by `bench/run_all.sh`: the JSONL file and the backend label of the rows | `bench/*.py` |

Variables of other tools:

- `JAX_PLATFORMS`: `mtl,cpu` selects the plugin. `tests/conftest.py` and
  `scripts/run_jax_tests.sh` set it.
- `JAX_NUM_GENERATED_CASES`: cases per test in `scripts/run_jax_tests.sh`,
  default 3 (the known-failures list assumes 3).
- `PYTEST_TIMEOUT` (`scripts/run_jax_tests.sh`): seconds before a slow
  test's stack dump, default 180. The test isn't stopped.
- `XLA_FLAGS`: `tests/conftest.py` refuses to run with it set, or with
  any `METAL_PJRT_*` setting other than `METAL_PJRT_STATE_DIR`,
  `METAL_PJRT_TRACE` and `METAL_PJRT_DEVICE_LOCK_HELD`. The plugin always
  forces `xla_gpu_enable_triton_gemm=false` and
  `xla_gpu_dot_merger_threshold_mb=0`
  ([`integration-notes.md`](integration-notes.md)).

## State

- `~/.cache/metal-pjrt/`: GPU reset log, device lock, JAX test checkout.
  `METAL_PJRT_STATE_DIR` moves only the reset log. Until the next pin bump
  the lock script also takes the pre-rename `~/.cache/jax_metal` lock
  ([`CHANGELOG.md`](../CHANGELOG.md)).
- `~/.cache/metal-pjrt-plugin/`: Bazel's shared disk and repository
  caches.
