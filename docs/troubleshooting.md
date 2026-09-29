# Troubleshooting

What the plugin's errors and warnings mean and what to do about them. The
messages are quoted as the plugin prints them; `...` stands for numbers,
names and source lines. JAX raises runtime failures as
`jax.errors.JaxRuntimeError` with the status code (`RESOURCE_EXHAUSTED`,
`UNIMPLEMENTED`, ...) in front of the message.

## My program still runs on the CPU

`jax.default_backend()` is `'cpu'` and `jax.devices()` lists only
`CpuDevice`s. The plugin is opt-in: with `JAX_PLATFORMS` unset, JAX keeps
CPU as its default backend. Select the plugin with
`JAX_PLATFORMS=mtl,cpu`, with `jax.config.update("jax_platforms",
"mtl,cpu")` before the first use of a device or array, or place arrays on
`jax.devices("mtl")[0]` explicitly (README, "Use").

If `jax.devices("mtl")` raises, the backend failed to start. With
`JAX_PLATFORMS` unset that failure is quiet (so CPU programs keep
working); run with `JAX_PLATFORMS=mtl,cpu` to see JAX's "Unable to
initialize backend 'mtl'" error with the reason, and check the warnings
below.

## Warnings when JAX starts

- `metal-pjrt-plugin is built for jax and jaxlib 0.11.2, found jax ...;
  it may fail or compute wrong results. Install jax==0.11.2
  jaxlib==0.11.2.` The plugin is built against one jaxlib's XLA and uses
  private JAX APIs; install exactly that version.
- `metal-pjrt-plugin: the PJRT plugin library is missing, so the 'mtl'
  platform is unavailable: ...` The dylib next to the Python package is
  gone. In a source checkout the message says whether the link into
  `bazel-bin` is dangling; rebuild and relink with `scripts/install_dev.sh`.
  Otherwise reinstall the wheel.
- `metal-pjrt-plugin: host callbacks unavailable: ...` Callbacks will fail
  (below); the rest of the plugin works. Please report it.
- `Platform 'mtl' is experimental and not all JAX functionality may be
  correctly supported!` JAX prints this for every plugin platform it does
  not know. It is expected.

## Out of memory (RESOURCE_EXHAUSTED)

GPU memory is system RAM. JAX's message starts with XLA's "Out of memory
while trying to allocate ..." and the plugin appends the reason. Two
different limits:

- `Metal: allocating ... refused: this process already holds ... of its
  ... memory budget (device 0; half of RAM, capped by the GPU's
  recommended working set). Use smaller arrays or batches, or raise the
  budget: METAL_PJRT_MEMORY_FRACTION=... gives ... (less memory for the
  rest of the system).` This process hit its own budget. Use smaller arrays or
  batches, free arrays you no longer need, or raise the budget with the
  value the message names (the last clause is left out when the budget
  already equals the GPU's working set).
- `Metal: allocating ... refused by the system memory guard: only ... of
  system memory is free or reclaimable and ... is kept for the OS (device
  0; this process holds ..., budget ...). Close other memory-heavy
  applications or use smaller arrays or batches` The machine is short of
  memory, not this process: the plugin refuses allocations that would push
  the system into swap, because a GPU waiting on swapped-out memory trips
  the watchdog. Close other programs (a browser, an idle Bazel server:
  `bazel shutdown`) or use smaller batches. On an 8 GB Mac a nanoGPT-sized
  training step can hit this with a browser open.

## "accepts no further GPU work in this process"

`Metal device 0 accepts no further GPU work in this process; restart the
Python process. Earlier GPU failure: ...` An earlier GPU command failed
(a fault, a watchdog timeout, the GPU running out of memory), and, as a
CUDA context error does, that ends GPU work for the process: nothing after
it can be trusted. Restart Python; the machine does not need a reboot. The
earlier failure is quoted after "Earlier GPU failure:", for example
`Metal: GPU command buffer failed on device 0: ... The GPU watchdog stopped
a computation that ran too long: split it into smaller jit calls or shrink
the inputs`.

## A kernel is quarantined (FAILED_PRECONDITION)

`Metal: kernel ... is quarantined: it was in the command buffer that timed
out in ... GPU watchdog resets since boot. A reboot lifts the quarantine,
as does deleting .../.cache/metal-pjrt/gpu_resets.jsonl
(scripts/gpu_health.py --clear in a source checkout); do that only once
the cause is fixed ...` A kernel was involved in two GPU resets since boot,
so the plugin refuses to run it again: each reset affects every process
and can leave the GPU slow until a reboot. Change the program (smaller
inputs, fewer operations per `jit`) or reboot; delete the file (or run the
script) only once the cause is fixed. The file is under
`METAL_PJRT_STATE_DIR` when that is set. `METAL_PJRT_QUARANTINE_STRIKES=0`
turns the quarantine off. A reset is also logged when it happens:
`Recorded the GPU reset in ... with ... suspect kernel(s); ...`.

## Unsupported operations (UNIMPLEMENTED)

These are compile-time errors: the program never ran, and nothing is
wrong with the device. Messages that start with `Metal:` name the JAX
operation and its source line:

- `Metal: f64 / complex128 arithmetic is not supported (Apple GPUs have
  no double precision; use float32 / complex64, or run it on the CPU
  backend): ...` and
  `Metal: f64 transpose inside jit is not supported (...; only transfers
  of f64 arrays work). Use float32, or run it on the CPU backend: ...`
  (also broadcast, concatenate, gather, iota, pad, reverse, select). Keep
  f64 work on CPU, or turn `jax_enable_x64` off.
- `Metal: matmul s8 x s8 -> s32 is not supported: both operands must be
  f32, f16 or bf16 of one type, the result that type or f32. Cast the
  operands` (also for mixed types such as `f16 x f16 -> bf16`). Cast to
  a supported type.
- `Metal: scatter with a combiner on 64-bit elements needs 64-bit atomics,
  which Metal does not have: ...` Pass `unique_indices=True` if the
  indices are unique, or use 32-bit elements. A scatter on complex64
  needs them even to overwrite (XLA writes complex elements with a
  compare-and-swap loop).
- `Metal: matmul (dot) of complex values is not supported; ...` and
  `Metal: sort of complex values is not supported; ...`: split into real
  and imaginary parts (a complex matmul is four real ones), or run it on
  the CPU backend.
- `Metal: bf16 matmul too large: ... Split the matmul or use float32` and
  `Metal: matmul operand ... has a batch, row or column group of ...
  elements; XLA's matmul config counts them in 32 bits. Split the batch
  (lax.map, or chunks)`.

Messages that start with `MSL emitter: unsupported` come from the kernel
translator, which has no code for that type, and name an internal
operation instead of yours:

- `MSL emitter: unsupported non-trivial unrealized_conversion_cast in op
  'builtin.unrealized_conversion_cast' at loc("loop_convert_fusion")`:
  an int4/uint4 conversion.
- `MSL emitter: unsupported sub-byte / odd-width integer type in op
  'arith.trunci' at ...`: int4/uint4.
- `MSL emitter: unsupported f64 type (Metal has no double precision) in op
  'llvm.load' at ...`: f64 data movement the f64 check above does not
  catch (a strided slice); keep it on CPU.

Other errors from JAX or XLA:

- `MLIR translation rule for primitive 'eig' not found for platform mtl`
  (also `schur`, `hessenberg`, `tridiagonal`): no implementation on mtl;
  compute these on CPU.
- `Unsupported algorithm on the current device(s): ALG_DOT_TF32_TF32_F32`
  (also `F16_F16_F16`, `BF16_BF16_BF16`): these dot precision algorithms
  are not available; the `BF16_BF16_F32` family is.

An error ending in `(metal-pjrt-plugin bug; please report it with the HLO
from XLA_FLAGS=--xla_dump_to=<dir>)` is a bug in the plugin: please report
it with that dump.

## Host callbacks

- `host callback raised: ...` Your callback raised a Python exception; its
  message follows.
- `metal-pjrt-plugin: unknown host callback id ...: the callable is gone.
  ...` An executable with a callback came from a persistent compilation
  cache written by another process. Clear the cache directory, or don't
  cache functions with callbacks.
- `Metal: host callbacks are unavailable: metal_pjrt_plugin did not install
  them (see the "host callbacks unavailable" warning logged when JAX
  initialized the plugin)` See that warning.
- `metal-pjrt-plugin: host callbacks do not support dtype int4 on platform
  mtl`: sub-byte types cannot be passed to callbacks.
