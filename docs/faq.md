# FAQ

## What works?

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
| JAX's persistent compilation cache | yes | opt-in, see [below](#how-do-i-turn-on-the-compilation-cache) |
| float32, float16, bfloat16, integers, bool, complex64 | yes | complex LU, and so complex `solve`, `inv` and `det`, isn't supported |
| float64, complex128 | no | Apple GPUs have no double type; keep float64 work on the CPU |
| int4 / uint4 | no | |
| Several devices (`pmap`, sharding) | no | one GPU only |

[`op-coverage.md`](op-coverage.md) has the full picture, op by op, with
the tests behind each row.

## Is it safe for my GPU?

Yes, with one difference from a datacenter NVIDIA card. A Mac's GPU
also drives the display, so macOS runs a watchdog: if one GPU kernel runs
too long, it resets the GPU for every program on the machine, and after
several resets the GPU can stay slow until you reboot. The plugin keeps
ordinary programs clear of it: it splits queued work into short command
buffers, caps each process's memory (see
[below](#how-much-memory-can-it-use)), and refuses allocations under
critical memory pressure. A few things are still up to you:

- Split very large single operations, like a matmul with ~10^12 flops.
- Keep your problems well inside memory. The GPU shares RAM with
  everything else, and a GPU stalled on swapped-out pages can trip the
  watchdog.
- Stop GPU jobs with Ctrl-C, never `kill -9`. Interrupting
  mid-computation hasn't been tested thoroughly, so let jobs finish when
  you can.

If you run several GPU-heavy jobs at once, `scripts/device_lock.py --
<command>` (in a source checkout) queues them one at a time and passes
Ctrl-C through.

If something does go wrong, the first GPU error ends GPU work for that
process: every later GPU call fails and says to restart Python. The
machine doesn't need a reboot. The plugin logs every reset it sees to
`~/.cache/metal-pjrt/gpu_resets.jsonl`, and `scripts/gpu_health.py` (in a
source checkout) summarizes the log. [`design.md`](design.md#runtime) has
the details, including an opt-in quarantine for kernels that cause resets.

## How accurate is it?

Most functions agree with CPU float32 to within a few ulps. A few special
functions are looser (`lgamma`, `digamma`, `betainc`), and Metal's `exp`,
`log`, `sin` and `cos` are slightly biased. Some functions flush subnormal
inputs and outputs to zero, as XLA's CPU backend does; a few differ.
[`accuracy.md`](accuracy.md) has the numbers.

## How much memory can it use?

The GPU shares your RAM. By default a process may use up to half of it,
capped by the GPU's recommended working set. `METAL_PJRT_MEMORY_FRACTION`
scales that budget (`2` allows all of RAM, still under that cap).
Separately, any allocation is refused while macOS reports critical memory
pressure. Out-of-memory errors say which of the two happened.

## Why is my linear algebra slower than on the CPU?

Factorizations above 32x32 run on the CPU (Accelerate) after the GPU
finishes its queued work, so lots of medium-sized ones are often faster
kept on the CPU.

## How do I turn on the compilation cache?

JAX's persistent cache is off until you give it a directory. Since most
compiles here take under a second, also lower JAX's threshold for caching
them:

```
export JAX_COMPILATION_CACHE_DIR=~/.cache/jax
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
```

## Can I use it alongside Apple's jax-metal?

Use a separate virtual environment if you also use `jax-metal`; the two
pin different JAX versions.

## Something failed. What now?

[`troubleshooting.md`](troubleshooting.md) maps the plugin's error
messages to what to do.
