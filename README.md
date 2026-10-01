# metal-pjrt-plugin: run JAX on your Mac's GPU

An open-source JAX plugin that runs JAX programs on Apple Silicon GPUs
through Metal. It's for people who write JAX and want their Mac's GPU to
do the work: training small models, running LLM inference, or scientific
code that's happy in float32.

Under the hood it treats Metal as one more XLA:GPU platform, next to CUDA
and ROCm. XLA's own GPU compiler fuses your program and generates the
kernels, which the plugin translates to Metal Shading Language.

The JAX platform name is `"mtl"` (Metal's own prefix, as in `MTLDevice`),
so it doesn't collide with Apple's closed-source `jax-metal` plugin, which
uses `"metal"`. This project isn't affiliated with or endorsed by Apple or
Google.

## Status

Early, but it works. Float32, float16 and bfloat16 programs run end to
end on one GPU: training loops, sorting, FFTs, host callbacks and float32
linear algebra. Performance is broadly comparable to MLX and PyTorch's MPS
backend on the workloads we've tried, with some slow spots (see
[`docs/performance.md`](docs/performance.md)). Everything so far has been tested on
a single machine (an M3 with 8 GB, macOS 26.2), so expect rough edges
elsewhere, and please [report them](CONTRIBUTING.md).

## Requirements

- An Apple Silicon Mac. Intel Macs aren't supported.
- macOS 26 or later.
- Python 3.12+ with exactly `jax==0.11.2` and `jaxlib==0.11.2`. The plugin
  is built against that release's XLA and warns on any other version.

There's no PyPI release yet, so you build from source. That needs the
Xcode command-line tools (`xcode-select --install`; full Xcode isn't
needed), plus `bazelisk` and `uv` (`brew install bazelisk uv`).

## Install

```
git clone https://github.com/dfm/metal-pjrt-plugin.git
cd metal-pjrt-plugin
scripts/install_dev.sh     # creates .venv, builds the plugin, installs it
bazel shutdown             # frees the build server's memory
```

The first build compiles XLA and takes a while: about 1.5-2 hours on an
8 GB M3 (95 minutes from a clean clone), with ~8 GB of build output plus
a disk cache in `~/.cache/metal-pjrt-plugin/` (~4.5 GB after the first
build, growing with rebuilds). Every checkout shares that cache, so later
builds and fresh clones take minutes. Running the plugin needs no
developer tools.

To check that it works, run the smoke tests (`device_lock.py` runs one GPU
job at a time; see [GPU safety](#gpu-safety)):

```
scripts/device_lock.py -- .venv/bin/python -m pytest tests/test_smoke.py
```

To install into another environment, build a wheel with
`scripts/build_wheel.sh` and `pip install` the file it puts in `dist/`.

## Quick start

```python
import jax
jax.config.update("jax_platforms", "mtl,cpu")  # before using any device
import jax.numpy as jnp

x = jnp.arange(4.0)
y = jax.jit(lambda v: v * 2)(x)
print(y, y.devices())  # [0. 2. 4. 6.] {MtlDevice(id=0)}
```

JAX prints a warning that the "mtl" platform is experimental; that's
expected. You can also pick the platform from the shell:

```
JAX_PLATFORMS=mtl,cpu python my_script.py
```

The plugin is opt-in: installing it doesn't change JAX's default (CPU).
To keep CPU as the default and send only some work to the GPU, place it
explicitly with `jax.device_put(x, jax.devices("mtl")[0])`.

## What works

A "no" below should always be an error at compile time, never a wrong
answer. If you get a wrong answer, that's a bug, so please report it.

| Feature | Works? | Notes |
|---|---|---|
| `jit`, `grad`, `vmap`, `checkpoint`, `scan`, `while_loop`, `cond` | yes | |
| Elementwise math, reductions, gather, scatter, cumulative ops | yes | some 64-bit and complex scatters need `unique_indices=True` |
| `jax.random` | yes | |
| Matmul in float32, float16, bfloat16 | yes | |
| Matmul on int8 or with mixed types (e.g. f16 x f16 -> bf16) | no | |
| Convolutions (1-D, 2-D, 3-D, grouped), with gradients | yes | 3-D and grouped ones are slow |
| Sorting (`sort`, `argsort`, `top_k`, `searchsorted`) | yes | |
| FFT (`jnp.fft`) | yes | complex64 / float32; lengths above 2^24 (powers of two) or 2^23 - 1 (others) aren't supported |
| Linear algebra in float32 (`cholesky`, `solve`, `lu`, `qr`, `eigh`, `svd`, ...) | yes | small Cholesky, triangular solve and LU (up to 32x32) run on the GPU; larger factorizations run on the CPU (Accelerate) |
| `eig`, `schur`, `hessenberg`, `tridiagonal` | no | |
| `pure_callback`, `io_callback`, `jax.debug.print` | yes | synchronous, so slow in hot loops |
| Buffer donation (`donate_argnums`) | yes | |
| JAX's persistent compilation cache | yes | opt-in, see below |
| float32, float16, bfloat16, integers, bool, complex64 | yes | complex LU, and so complex `solve`, `inv` and `det`, isn't supported |
| float64, complex128 | no | Apple GPUs have no double type; keep float64 work on the CPU |
| int4 / uint4 | no | |
| Several devices (`pmap`, sharding) | no | one GPU only |

[`docs/op-coverage.md`](docs/op-coverage.md) has the full picture, op by
op, with the tests behind each row.

## GPU safety

If one GPU kernel runs too long, macOS's watchdog resets the GPU for every
program on the machine, and after several resets it can stay slow until
you reboot. A few habits keep you clear of it:

- Run one GPU-heavy job at a time (`scripts/device_lock.py -- <command>`
  queues them), and keep your problems well inside memory: swapping can
  stall the GPU long enough to trip the watchdog.
- If you need to stop a GPU job, use Ctrl-C (`device_lock.py` passes it
  to the job); never `kill -9`. Interrupting mid-computation hasn't been
  tested thoroughly, so let jobs finish when you can.
- Split very large single operations, like a matmul with ~10^12 flops.

If something does go wrong, the first GPU error ends GPU work for that
process: every later GPU call fails and says to restart Python. The
machine doesn't need a reboot. The plugin logs every reset it sees to
`~/.cache/metal-pjrt/gpu_resets.jsonl`, and `scripts/gpu_health.py` (in a
source checkout) summarizes the log. [`docs/design.md`](docs/design.md#runtime) has the
details, including an opt-in quarantine for kernels that cause resets.

## Good to know

- **Accuracy.** Most functions agree with CPU float32 to within a few
  ulps. A few special functions are looser (`lgamma`, `digamma`,
  `betainc`), and Metal's `exp`, `log`, `sin` and `cos` are slightly
  biased. Some functions flush subnormal inputs and outputs to zero, as
  XLA's CPU backend does; a few differ.
  [`docs/accuracy.md`](docs/accuracy.md) has the numbers.
- **Memory is your RAM.** By default a process may use up to half of it
  (`METAL_PJRT_MEMORY_FRACTION` scales that). Out-of-memory errors say
  whether your process hit its budget or the whole system is short.
- **Medium-sized linear algebra is often faster on the CPU.**
  Factorizations above 32x32 run on the CPU (Accelerate) after the GPU
  finishes its queued work, so lots of them are often faster kept on the
  CPU.
- **Compilation cache.** JAX's persistent cache is off until you give it
  a directory. Since most compiles here take under a second, also lower
  JAX's threshold for caching them:

  ```
  export JAX_COMPILATION_CACHE_DIR=~/.cache/jax
  export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
  ```

- **jax-metal.** Use a separate virtual environment if you also use
  Apple's `jax-metal`; the two pin different JAX versions.

When something fails, [`docs/troubleshooting.md`](docs/troubleshooting.md)
maps the plugin's error messages to what to do.

## Examples

- [`examples/llm`](examples/llm): Qwen3 inference in pure JAX, in bf16,
  int8 and int4.
- [`examples/lora`](examples/lora): LoRA fine-tuning of Qwen3.
- [`examples/cifar`](examples/cifar): training a CIFAR-10 classifier to
  94% accuracy (airbench94).

## Learn more

- [`docs/design.md`](docs/design.md): how the plugin works, and how it
  differs from MLX.
- [`docs/op-coverage.md`](docs/op-coverage.md),
  [`docs/accuracy.md`](docs/accuracy.md),
  [`docs/callbacks.md`](docs/callbacks.md): what runs, how accurately, and
  host callbacks.
- [`docs/performance.md`](docs/performance.md): measurements and how they
  were taken.
- [`docs/integration-notes.md`](docs/integration-notes.md): the details
  of how the plugin plugs into XLA.
- [`docs/development.md`](docs/development.md): building, testing and
  environment variables.
- [`docs/roadmap.md`](docs/roadmap.md): what's next, and
  [`CHANGELOG.md`](CHANGELOG.md): what changed.

## Contributing

Issues and pull requests are welcome. [`CONTRIBUTING.md`](CONTRIBUTING.md)
says what makes a useful report, and
[`docs/development.md`](docs/development.md) covers building and testing.
Please run GPU tests under `scripts/device_lock.py`. For security issues,
see [`SECURITY.md`](SECURITY.md).

## License

Apache-2.0 ([`LICENSE`](LICENSE)). The plugin includes third-party code:
matmul, convolution and FFT kernels ported from
[MLX](https://github.com/ml-explore/mlx) (MIT), Apple's metal-cpp headers
(Apache-2.0), and a statically linked XLA. Their notices are in
[`THIRD_PARTY_NOTICES`](THIRD_PARTY_NOTICES), which ships in the wheel.
