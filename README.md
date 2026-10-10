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
- Python 3.12+ with `jax` and `jaxlib` 0.10.0 or later (tested: 0.10.0
  through 0.11.2 and a 0.12 nightly). Older versions load with a warning.

## Install

```
pip install metal-pjrt-plugin
```

## Quick start

```python
import jax
import jax.numpy as jnp

x = jnp.arange(4.0)
y = jax.jit(lambda v: v * 2)(x)
print(y, y.devices())  # [0. 2. 4. 6.] {MtlDevice(id=0)}
```

JAX prints a warning that the "mtl" platform is experimental; that's
expected. Like JAX's other GPU plugins, once installed the plugin is JAX's
default backend on a Mac with a Metal GPU, and programs it can't run
(float64, for example) raise an error rather than falling back to the CPU.
To keep CPU as the default, run with `JAX_PLATFORMS=cpu`; to keep it
available too, use `JAX_PLATFORMS=cpu,mtl` and place work on the GPU
explicitly with `jax.device_put(x, jax.devices("mtl")[0])`.

Before running anything heavy, read [Is it safe for my
GPU?](docs/faq.md#is-it-safe-for-my-gpu): a kernel that runs too long
makes macOS reset the GPU for the whole machine. The
[FAQ](docs/faq.md) also covers what works, accuracy, memory and the
compilation cache, and [`docs/troubleshooting.md`](docs/troubleshooting.md)
maps the plugin's error messages to what to do.

## Examples

Each example has a README with results and what it took to make it fast.
Run their commands from the root of this repository.

- [`examples/llm`](examples/llm): Qwen3 inference in pure JAX, in bf16,
  int8 and int4.
- [`examples/lora`](examples/lora): LoRA fine-tuning of Qwen3.
- [`examples/cifar`](examples/cifar): training a CIFAR-10 classifier to
  94% accuracy (airbench94).

## Learn more

- [`docs/faq.md`](docs/faq.md): what works, GPU safety, accuracy, memory
  and the compilation cache.
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
[`THIRD_PARTY_NOTICES`](THIRD_PARTY_NOTICES), which ships in both wheels.
