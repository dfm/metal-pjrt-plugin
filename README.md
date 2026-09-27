# jax-openmetal

An open-source PJRT plugin that runs JAX on Apple Silicon GPUs. It treats
Metal as a fourth XLA:GPU platform, next to CUDA, ROCm and SYCL: XLA's own
GPU compiler fuses the program and its emitters generate the kernels, which
the plugin translates to Metal Shading Language. It does not re-interpret
StableHLO op by op. Its JAX platform is `"openmetal"`, so it can be
installed next to Apple's closed-source `jax-metal` (platform `"metal"`).

Status: a working minimum. f32, f16 and bf16 programs run end to end,
including training loops, linear algebra, sorting and host callbacks. JAX's
`lax_test.py` passes 947 cases; the 62 known failures are complex types,
int4 and dot precision algorithms. Tested on one machine (M3, 8 GB, macOS
26.2).

## Requirements

- An Apple Silicon Mac (arm64). Intel Macs are not supported.
- macOS 26 or later. The plugin is built with the 26.5 SDK and tested on
  26.2; older versions are untested, and the wheel is tagged
  `macosx_26_0_arm64`.
- Python 3.12 or later, with exactly `jax==0.11.2` and `jaxlib==0.11.2`. The
  plugin is built against that jaxlib's XLA commit and uses private
  `jax._src` APIs, so it warns at import on any other version and may not
  work there.
- To build from source: the Xcode command-line tools
  (`xcode-select --install`; full Xcode is not needed), `bazelisk`
  (`brew install bazelisk`), `uv`, and the time and disk for a first build of
  XLA: about 2 hours on an 8 GB M3, ~10 GB of Bazel output plus a disk cache
  that grows with rebuilds. Running the plugin needs no developer tools:
  the Metal framework compiles its kernels at run time.

## Install

From a wheel (the plugin library is inside it):

```
pip install jax==0.11.2 jaxlib==0.11.2 jax_openmetal-0.0.1-py3-none-macosx_26_0_arm64.whl
```

From source, as an editable install into `.venv` (see `docs/development.md`):

```
git clone <this repository> jax-openmetal && cd jax-openmetal
uv venv --python 3.12 .venv
scripts/install_dev.sh
```

## Use

openmetal is opt-in: installing it does not change JAX's default backend
(CPU). Select it for a whole program:

```
JAX_PLATFORMS=openmetal,cpu python my_script.py
```

or in Python before JAX initializes its backends:

```python
import jax
jax.config.update("jax_platforms", "openmetal,cpu")
```

Or leave the default alone and place work explicitly:

```python
dev = jax.devices("openmetal")[0]
x = jax.device_put(x, dev)   # jitted functions run where their inputs are
```

**Next to Apple's jax-metal.** Both register a JAX plugin. With
`JAX_PLATFORMS` set, JAX initializes only the platforms listed, so
`JAX_PLATFORMS=openmetal,cpu` ignores jax-metal (and `metal,cpu` ignores
this plugin). jax-metal pins its own jax version, so separate virtual
environments are simpler.

## What works

- Elementwise math, reductions, broadcasting, gather and scatter, control
  flow (`scan`, `while_loop`, `cond`), convolutions (correct but slow),
  random numbers, and autodiff through all of these, in f32, f16 and bf16
  (and integer and bool types).
- Matrix multiplication: f32 on Metal Performance Shaders, f16/bf16 on
  native kernels, with bias and activation epilogues fused.
- Sorting (`sort`, `argsort`, `top_k`, ...): a GPU radix sort for large
  arrays, bit-identical to CPU.
- Linear algebra (`cholesky`, `solve`, `triangular_solve`, `lu`, `qr`,
  `eigh`, `svd`) in f32 through Accelerate's LAPACK on the shared memory;
  `fft` as a dense DFT (correct, O(n^2) per axis).
- `pure_callback`, `io_callback`, `jax.debug.print` and
  `jax.debug.callback`.
- JAX's persistent compilation cache (in `~/.cache/openmetal/` unless you
  configure a directory; JAX's defaults decide what gets cached).

`docs/op-coverage.md` has the full table.

## What is refused (with an error, never a wrong answer)

- float64 arithmetic: Metal GPUs have no double type. With `jax_enable_x64`
  on, keep f64 work on CPU. Moving f64 data is fine; arithmetic is refused at
  compile time, naming the operation.
- Complex numbers in device buffers. Complex values that stay inside one
  fused kernel, such as `abs(fft(x))`, work.
- Integer matrix multiplications that XLA turns into a GEMM (int8 x int8 ->
  int32), refused at compile time.
- 64-bit scatter-add/min/max without `unique_indices=True` (Metal has only
  32-bit atomics), refused at compile time.
- int4/uint4 types and fp8 GEMMs.

## Known limitations

- **Accuracy.** Results agree with CPU float32 to a few ulps, with some
  known gaps: Metal's `exp`, `sin` and `cos` are biased on [0.125, 1), as are
  `tanh`, `expm1`, `erf` and `log`, and subnormal outputs flush to zero. See
  `docs/accuracy.md`.
- **A GPU error ends GPU work for the process.** After the first failed GPU
  command (a fault or a watchdog timeout), every later GPU operation in that
  process fails with the original error plus "no further GPU work is
  accepted in this process, restart it". Restart Python.
- **Memory.** GPU memory is system RAM. The plugin keeps a process under
  half of RAM, and it refuses any allocation that would push the machine
  into swap, with RESOURCE_EXHAUSTED and the reason and numbers. When you
  hit that, close memory-hungry programs (or run `bazel shutdown` right after
  a build). `jax.device_put` of a NumPy array briefly needs twice its size
  in host memory: the plugin snapshots the data before returning, so
  changing the array afterwards cannot change what the device gets. On an
  8 GB Mac that matters for multi-GB transfers.

## GPU safety

A kernel that runs too long trips macOS's GPU watchdog, which resets the GPU
for every process, and after a few resets the GPU can stay slow until a
reboot. The plugin splits work into small command buffers and logs every
reset in `~/.cache/openmetal/gpu_resets.jsonl`. A kernel involved in two
resets since boot is refused until a reboot or
`scripts/gpu_health.py --clear`. To stay clear of this:

- Run one GPU-heavy job at a time. `scripts/device_lock.py -- <cmd>`
  serializes jobs.
- Never kill a process (`kill -9`, closing its terminal) while it has GPU
  work in flight. Let it finish or fail; the plugin's waits are bounded.
- Keep problems well inside memory. Swapping is what stalls a GPU long
  enough to trip the watchdog.

## More

`docs/design.md` (design), `docs/integration-notes.md` (the source-verified
contract with XLA), `docs/op-coverage.md`, `docs/performance.md`
(measurements and the memory policy), `docs/accuracy.md`,
`docs/roadmap.md`, `docs/mlx-comparison.md` (how this differs from
MLX-based approaches) and `docs/development.md` (layout, building, tests,
benchmarks, environment variables).
