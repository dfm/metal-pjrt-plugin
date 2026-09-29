# metal-pjrt-plugin: run JAX on your Mac's GPU

An open-source JAX plugin that runs JAX programs on Apple Silicon GPUs
through Metal.

Its JAX platform is `"mtl"` (`jax.devices("mtl")`,
`JAX_PLATFORMS=mtl,cpu`), not `"metal"`: that name belongs to Apple's
closed-source `jax-metal` plugin, and the two can be installed side by
side. MTL is Metal's own prefix (`MTLDevice`, `MTLBuffer`).

Status: a working minimum on one GPU. f32, f16 and bf16 programs run end to
end, including training loops, sorting, host callbacks and linear algebra
in f32; the [support table](#what-works) has the details. Tested on one
machine (M3, 8 GB, macOS 26.2); every tolerance and performance number
comes from it.

## Requirements

- An Apple Silicon Mac (arm64). Intel Macs are not supported.
- macOS 26 or later. The plugin is built with the 26.5 SDK and tested on
  26.2; older versions are untested, and the wheel is tagged
  `macosx_26_0_arm64`.
- Python 3.12 or later, with exactly `jax==0.11.2` and `jaxlib==0.11.2`. The
  plugin is built against that jaxlib's XLA commit
  ([`third_party/PINS.md`](third_party/PINS.md)) and uses private
  `jax._src` APIs, so it warns at plugin discovery on any other version and
  may not work there.
- To build it (there is no PyPI release yet): the Xcode command-line tools
  (`xcode-select --install`; full Xcode is not needed), Homebrew, `bazelisk`
  (`brew install bazelisk`; it fetches the Bazel version in `.bazelversion`,
  8.7.0), `uv` (`brew install uv`), and the time and disk for a first build
  of XLA: about 2 hours on an 8 GB M3, ~8.5 GB of Bazel output plus a disk
  cache (`~/.cache/metal-pjrt-plugin/`) that grows with rebuilds. The cache
  is shared by every checkout, so a second clone builds in minutes.
  Running the plugin needs no developer tools: the Metal framework compiles
  its kernels at run time.

## Install

From source, as an editable install into `.venv`:

```
git clone https://github.com/dfm/metal-pjrt-plugin.git
cd metal-pjrt-plugin
scripts/install_dev.sh     # creates .venv with uv, builds, installs (editable, with pytest)
bazel shutdown             # frees the Bazel server's memory, or the plugin's memory guard may refuse allocations
```

To use the plugin in another environment, build a wheel from that
checkout (`scripts/build_wheel.sh`, which needs `uv`) and install it; it
pins `jax==0.11.2` and `jaxlib==0.11.2`:

```
scripts/build_wheel.sh     # dist/metal_pjrt_plugin-0.0.1-py3-none-macosx_26_0_arm64.whl, the plugin library inside
pip install dist/metal_pjrt_plugin-0.0.1-py3-none-macosx_26_0_arm64.whl
```

## Quick check

```
JAX_PLATFORMS=mtl,cpu .venv/bin/python -c "import jax; print(jax.devices())"
```

prints `[MtlDevice(id=0)]`, after JAX's warning "Platform 'mtl' is
experimental and not all JAX functionality may be correctly supported!".
Or in Python:

```python
import jax
jax.config.update("jax_platforms", "mtl,cpu")  # before the first device or array use
import jax.numpy as jnp

x = jax.device_put(jnp.arange(4.0))
y = jax.jit(lambda v: v * 2)(x)
print(y, y.devices())  # [0. 2. 4. 6.] {MtlDevice(id=0)}
```

The smoke tests, from the checkout:

```
# device_lock.py runs one GPU job at a time (see "GPU safety")
scripts/device_lock.py -- .venv/bin/python -m pytest tests/test_smoke.py
```

## Use

The plugin is opt-in: installing it does not change JAX's default backend
(CPU). Select it for a whole program:

```
JAX_PLATFORMS=mtl,cpu python my_script.py
```

or in Python, before the first use of a device or array:

```python
import jax
jax.config.update("jax_platforms", "mtl,cpu")
```

Or leave the default alone and place work explicitly:

```python
dev = jax.devices("mtl")[0]
x = jax.device_put(x, dev)   # jitted functions run where their inputs are
```

With `JAX_PLATFORMS` unset, JAX still initializes every installed backend,
this one included (it prints the "experimental" warning), but if the Metal
device cannot be set up it fails quietly and CPU programs carry on.
`JAX_PLATFORMS=cpu` skips the plugin entirely.

Next to Apple's jax-metal, use separate virtual environments: jax-metal
pins its own jax version.

## GPU safety

A GPU kernel that runs too long trips macOS's GPU watchdog, which resets the
GPU for every process; after several resets the GPU can stay slow until a
reboot. To stay clear of it:

- Run one GPU-heavy job at a time (`scripts/device_lock.py -- <cmd>`
  serializes jobs), and keep problems well inside memory: swapping stalls
  the GPU long enough to trip the watchdog.
- Never kill a process (`kill -9`, closing its terminal) while it has GPU
  work in flight; let it finish or fail, the plugin's waits are bounded.
  Whether Ctrl-C is safe mid-computation is untested, so avoid that too.
- Split very large single operations (a matmul over ~1e12 flops, a large
  convolution): one kernel can outlast the watchdog.

The plugin logs every reset in `~/.cache/metal-pjrt/gpu_resets.jsonl` and
refuses a kernel involved in two resets since boot until a reboot or until
that file is deleted (`scripts/gpu_health.py --clear` does it in a
checkout). [`docs/design.md`](docs/design.md#runtime) has the details.

## What works

Every "yes" has a test in `tests/`; every "no" is a compile-time error or
a test expected to fail, never a wrong answer. "No" errors either name the
operation and its source line ("Metal: ...") or come from the kernel
translator ("MSL emitter: unsupported ..."); see
[`docs/troubleshooting.md`](docs/troubleshooting.md).

| Feature | Works | Notes |
|---|---|---|
| `jit`, `grad`, `vmap`, `checkpoint`, control flow (`scan`, `while_loop`, `cond`, `switch`) | yes | `test_lax.py`, `test_smoke.py` |
| Elementwise math, reductions, broadcasting, gather, scatter, cumulative ops | yes | `test_lax.py`, `test_scan.py`. A scatter-add/min/max on 64-bit elements, and any scatter on complex64, without `unique_indices=True` is refused (Metal has 32-bit atomics only) |
| `jax.random` | yes | `test_lax.py` |
| Matmul, f32 / f16 / bf16 | yes | f32 on Metal Performance Shaders, f16/bf16 on native kernels with bias/activation fused, including few-row (small-batch decode) shapes at or above MLX's speed (`test_steel_gemm.py`, `test_epilogue.py`) |
| Matmul, integer GEMMs (int8 x int8 -> int32) and mixed types (e.g. f16 x f16 -> bf16) | no | refused at compile time. Small integer dots that XLA keeps as loops run (`test_lax.py`, "dot int32") |
| Matmul, dot precision algorithms (`TF32_TF32_F32`, `F16_F16_F16`, `BF16_BF16_BF16`) | no | XLA refuses them ("Unsupported algorithm on the current device(s)"). The `BF16_BF16_F32` family works |
| Matmul, fp8 | untested | fp8 conversions work (`test_lax.py`) |
| Matmul and sort of complex64 values | yes | a complex matmul runs as four real f32 ones on the f32 GEMM paths; sorts bit-identical to CPU (`test_lax.py`) |
| Convolutions | yes | 1-D and 2-D f32/f16/bf16 (forward and gradients) on MLX's steel convolution kernels, near MLX's speed (`test_conv.py`); grouped, 3-D, other types and tiny ones on XLA's slow loop emitter. Complex64 convolutions work (`test_lax.py`) |
| Sorting (`sort`, `argsort`, `top_k`, `searchsorted`) | yes | a GPU radix sort for large arrays, bit-identical to CPU (`test_sort.py`) |
| Linear algebra in f32 (`cholesky`, `solve`, `triangular_solve`, `lu`, `qr`, `eigh`, `svd`, `inv`, `det`), with gradients | yes | Accelerate's LAPACK on the shared memory (`test_linalg.py`). f16/bf16 linear algebra is untested |
| `eig`, `schur`, `hessenberg`, `tridiagonal` | no | no lowering on mtl (JAX: "MLIR translation rule for primitive 'eig' not found for platform mtl") |
| FFT | yes | `jnp.fft` (fft, rfft, irfft, fftn, ...) on MLX's FFT kernels, complex64 / float32, any length up to 2^24 (powers of two) or 2^23 - 1, with gradients and vmap; 0.6-1.3x MLX's time (`test_fft.py`, `bench/fft_bench.py`). Longer lengths raise NotImplementedError; complex128 is refused like float64 |
| `pure_callback`, `io_callback`, `jax.debug.print`, `jax.debug.callback` | yes | synchronous (below); sub-byte dtypes such as int4 are refused (`test_callbacks.py`) |
| `checkify` | partly, untested | functionalized checks (`checkify.checkify`) use JAX's generic path; `debug=True` checks are dropped, as on TPU |
| Several devices (`pmap`, sharding) | no | the plugin exposes one device |
| f32, f16, bf16, integer and bool types | yes | `test_lax.py` |
| float64 | no | Apple GPUs have no double type. Transfers of f64 arrays work, and inside `jit` so do copies (reshapes, contiguous slices); f64 arithmetic and other f64 data movement are refused at compile time (a strided f64 slice fails in the kernel translator instead). With `jax_enable_x64` on, keep f64 work on CPU |
| complex64 | yes | arithmetic, math, data movement, reductions, matmul, sort, cholesky / triangular_solve / qr and transfers (`test_lax.py`), except scatter without unique indices (above), and so LU (`solve`, `inv`, `det`), whose pivoting is such a scatter. complex128 is refused like float64 |
| int4 / uint4 | no | fail in the kernel translator (expected failure in `test_lax.py`) |
| Buffer donation (`donate_argnums`) | yes | the donated input's memory becomes the output, as on CUDA (`test_donation.py`; JAX's own donation tests in `api_test.py` pass) |
| JAX's persistent compilation cache | yes, opt-in | below (`test_callbacks.py`, `test_compilation_cache.py`) |

`docs/op-coverage.md` maps every XLA operation to the path it takes;
`docs/callbacks.md` covers host callbacks.

## Compilation cache

JAX's persistent compilation cache works for mtl but, as in JAX, is off
until you give it a directory; the plugin never sets one, since the
setting is process-wide and would also cache your CPU compiles:

```
export JAX_COMPILATION_CACHE_DIR=~/.cache/jax   # or jax.config.update("jax_compilation_cache_dir", ...)
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
```

The second line matters: JAX only caches compiles that took over 1 s, and
most mtl compiles are faster ([`docs/performance.md`](docs/performance.md)
has timings).

## Known limitations

- **Accuracy.** Most functions agree with CPU float32 to a few ulps; the
  tests' tolerances go up to 38 ulps for `betainc`, 93 for `reduce_prod`,
  580 for `lgamma` and 1600 for `digamma` (CPU float32 is also far off
  for these two). Metal's `exp`, `sin` and `cos` are biased for |x| >= 0.125 (below
  that the plugin uses its own polynomials), `log` is biased everywhere
  (about +-0.5 ulp), `tanh`, `sinh`, `cosh`, `expm1` and `erf` inherit
  `exp`'s bias, and subnormal inputs and outputs flush, as on XLA:CPU
  (`log(1e-40)` is `-inf`).
  [`docs/accuracy.md`](docs/accuracy.md) has the numbers.
- **A GPU error ends GPU work for the process.** After the first failed GPU
  command (a fault or a watchdog timeout), every later GPU operation in that
  process fails with "Metal device 0 accepts no further GPU work in this
  process; restart the Python process. Earlier GPU failure: ...". Restart
  Python; the machine does not need a reboot.
- **Memory.** GPU memory is system RAM. A process may hold up to half of
  RAM (capped by the GPU's recommended working set;
  `METAL_PJRT_MEMORY_FRACTION` scales it), and the plugin refuses any
  allocation that would push the machine into swap. Both fail with
  RESOURCE_EXHAUSTED and the numbers: "refused: this process already
  holds ... of its ... memory budget" is this process; "refused by the
  system memory guard: only ... of system memory is free or reclaimable"
  is the machine. On an 8 GB Mac a nanoGPT-sized training step (a 1.2 GB
  allocation) can hit the guard with a browser open: close other programs
  or use a smaller batch.
  `device_put` copies the array or waits until it is copied, so changing
  a NumPy array afterwards does not change what the device gets.
- **Small dense linear algebra is slow.** Above 32x32 it runs in Accelerate
  on the host after a full GPU synchronization: for n up to a few hundred
  that is ~10x slower than JAX on CPU (cholesky 128: ~0.3 ms vs 0.03 ms).
  Keep small factorizations on CPU if they dominate.
- **Python callbacks are synchronous.** Each `pure_callback`, `io_callback`
  or `jax.debug.*` call waits for all GPU work before it and runs on an XLA
  execution thread while the GPU stream is drained; callbacks in hot loops
  are slow.
- **Unverified.** The GPU-error path is tested only with injected failures:
  no real GPU fault or watchdog timeout was provoked, and the watchdog's
  timeout was never measured. Coexistence with jax-metal is argued from
  code, never tried. Tolerances and performance come from one M3 with
  macOS 26.2; other Apple GPUs or macOS versions (a different Metal
  compiler and math library) may need retuned tolerances.

## Troubleshooting

[`docs/troubleshooting.md`](docs/troubleshooting.md) maps the plugin's
error messages and warnings to what to do.

## More

How it works: the plugin treats Metal as a fourth XLA:GPU platform, next to
CUDA, ROCm and SYCL. XLA's own GPU compiler fuses the program and its
emitters generate the kernels, which the plugin translates to Metal Shading
Language; it does not re-interpret StableHLO op by op.

- [`docs/design.md`](docs/design.md): design, runtime policies, how this
  differs from MLX.
- [`docs/op-coverage.md`](docs/op-coverage.md),
  [`docs/accuracy.md`](docs/accuracy.md),
  [`docs/callbacks.md`](docs/callbacks.md): what runs, how accurately,
  and host callbacks.
- [`docs/performance.md`](docs/performance.md): numbers, methodology, what
  was measured and dropped.
- [`docs/integration-notes.md`](docs/integration-notes.md): the
  source-verified contract with XLA.
- [`docs/roadmap.md`](docs/roadmap.md): decisions and what is next;
  [`CHANGELOG.md`](CHANGELOG.md): renames and dated decisions.
- [`examples/llm`](examples/llm): Qwen3 inference in pure JAX (bf16,
  int8, int4), benchmarked against mlx-lm; a case study in fast decoding.
  [`examples/lora`](examples/lora): LoRA fine-tuning of Qwen3 against
  `mlx_lm.lora`; [`examples/cifar`](examples/cifar): CIFAR-10 to 94%
  (airbench94) against PyTorch on MPS.
- [`docs/development.md`](docs/development.md): layout, building, tests,
  benchmarks, environment variables; `docs/archive/` keeps the dated
  performance log and the review roadmap as history.

## Contributing

Issues and pull requests are welcome.
[`docs/development.md`](docs/development.md) covers building and testing;
please run the GPU tests under `scripts/device_lock.py`.

## License

Apache-2.0 (`LICENSE`). The plugin also contains third-party code: the
f16/bf16 matmul ("steel" and wide gemv) kernels are ported from
[MLX](https://github.com/ml-explore/mlx) (MIT), and Apple's metal-cpp headers
(Apache-2.0) are compiled in. Their notices are in `THIRD_PARTY_NOTICES`,
which the wheel ships next to `LICENSE`.
