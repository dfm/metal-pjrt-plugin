# Troubleshooting

Messages are quoted as the plugin prints them; `...` stands for numbers,
names and source lines. Runtime failures arrive as
`jax.errors.JaxRuntimeError` with the status code in front.

## My program still runs on the CPU

The plugin is opt-in. Select it with `JAX_PLATFORMS=mtl,cpu`, with
`jax.config.update("jax_platforms", "mtl,cpu")` before using any device,
or by placing arrays on `jax.devices("mtl")[0]`.

If `jax.devices("mtl")` raises, the backend failed to start, silently
when `JAX_PLATFORMS` is unset. Run with `JAX_PLATFORMS=mtl,cpu` to see
"Unable to initialize backend 'mtl'" and its reason.

## Warnings when JAX starts

- `metal-pjrt-plugin is built for jax and jaxlib 0.11.2, found jax ...;
  it may fail or compute wrong results. Install jax==0.11.2
  jaxlib==0.11.2.` Install exactly that version.
- `metal-pjrt-plugin: the PJRT plugin library is missing, so the 'mtl'
  platform is unavailable: ...` In a source checkout, rerun
  `scripts/install_dev.sh` (the message says if the link into `bazel-bin`
  is dangling). Otherwise reinstall the wheel.
- `metal-pjrt-plugin: host callbacks unavailable: ...` Callbacks will
  fail; everything else works. Please report it.
- `metal-pjrt-plugin: buffer donation unavailable: ...` or
  `metal-pjrt-plugin: persistent compilation cache unavailable: ...` A
  private JAX hook moved (a different JAX version). Programs still run,
  but donation copies and mtl compiles aren't cached. Install the pinned
  JAX, and please report it.
- `Ignoring METAL_PJRT_MEMORY_FRACTION=... (not a number > 0); using 1, a
  budget of ...` The default budget is in use.
- `Platform 'mtl' is experimental and not all JAX functionality may be
  correctly supported!` Expected; JAX prints it for every plugin.

## Out of memory (RESOURCE_EXHAUSTED)

XLA's "Out of memory while trying to allocate ..." is followed by the
plugin's reason, naming the computation whose allocation was refused:
`in jit_f: ...`. Because JAX runs asynchronously, the error may surface in
a later computation; the plugin then says `in jit_f (an earlier
asynchronous computation; the error surfaced in jit__reduce_sum)`.

**Your process hit its budget:**

    Metal: allocating ... refused: this process already holds ... of its
    ... memory budget (device 0; half of RAM, capped by the GPU's
    recommended working set). Use smaller arrays or batches, or raise the
    budget: METAL_PJRT_MEMORY_FRACTION=... gives ... (less memory for the
    rest of the system).

Use smaller arrays or batches, drop arrays you no longer need, or raise
the budget as the message suggests.

**The whole machine is short of memory:**

    Metal: allocating ... refused: the system is under critical memory
    pressure (system-wide, not this process's budget: it holds ... of its
    ... budget; device 0). Close other memory-heavy applications or use
    smaller arrays or batches

macOS is at critical pressure, where it starts killing processes. Close
other programs (an idle Bazel server: `bazel shutdown`) or use smaller
batches. Below critical nothing is refused; at the warning level the
plugin logs once that GPU work may slow down.

Two rarer ones:

- `Metal: the driver could not allocate ... on device 0 (...): ...
  already allocated, recommended working set ..., maxBufferLength ...`
  Metal itself returned no buffer: the machine is out of memory.
- `Metal: allocating ... exceeds the largest buffer the device can create
  (maxBufferLength ...; device 0, ...)` Split the array.

## "accepts no further GPU work in this process"

    Metal device 0 accepts no further GPU work in this process; restart the
    Python process. Earlier GPU failure: ...

An earlier GPU command failed (a fault, a watchdog timeout, out of
memory), and that ends GPU work for the process, as a CUDA context error
does. Restart Python; no reboot needed. The quoted failure says what
happened, for example:

    Metal: GPU command buffer failed on device 0: ... The GPU watchdog
    stopped a computation that ran too long: split it into smaller jit
    calls or shrink the inputs

## A kernel is quarantined (FAILED_PRECONDITION)

    Metal: kernel ... is quarantined: it was in the command buffer that
    timed out in ... GPU watchdog resets since boot. A reboot lifts the
    quarantine, as does deleting .../.cache/metal-pjrt/gpu_resets.jsonl
    (scripts/gpu_health.py --clear in a source checkout); do that only once
    the cause is fixed ...

Only with the opt-in quarantine on (`METAL_PJRT_QUARANTINE_STRIKES=n`): a
kernel involved in n resets since boot is refused, since every reset hits
every process. Change the program (smaller inputs, fewer operations per
`jit`) or reboot, and clear the log only once the cause is fixed. Each
reset is logged as `Recorded the GPU reset in ... with ... suspect
kernel(s)`.

## Unsupported operations (UNIMPLEMENTED)

Compile-time errors: the program never ran and the device is fine.
Messages starting with `Metal:` name your operation and source line:

- `Metal: f64 / complex128 arithmetic is not supported (Apple GPUs have
  no double precision; use float32 / complex64, or run it on the CPU
  backend): ...` and `Metal: f64 transpose inside jit is not supported
  (...; only transfers of f64 / complex128 arrays work). Use float32 /
  complex64, or run it on the CPU backend: ...` (also broadcast,
  concatenate, gather, iota, pad, reverse, select). Keep f64 work on the
  CPU, or turn `jax_enable_x64` off.
- `Metal: matmul s8 x s8 -> s32 is not supported: both operands must be
  f32, f16 or bf16 of one type, the result that type or f32. Cast the
  operands` (also mixed types such as `f16 x f16 -> bf16`).
- `Metal: scatter with a combiner on 64-bit elements needs 64-bit atomics,
  which Metal does not have: ...` Pass `unique_indices=True` if the
  indices are unique, or use 32-bit elements.
- `Metal: scatter of complex values needs 64-bit atomics (XLA
  compare-and-swaps complex elements, even to overwrite), ...` Pass
  `unique_indices=True`, or scatter the real and imaginary parts
  separately.
- `Metal: the HLO fft op is not supported (XLA's FFT runs on cuFFT only);
  ...` The program was lowered for another platform. Lower it for mtl.
- `Metal: XLA fused a bias of ... elements into a matmul with ... output
  columns, through a slice or reshape of the matmul's result. ...` Write
  `x @ w[:, :k] + b` instead of `(x @ w)[:, :k] + b`, or add a bias as
  wide as the matmul before slicing or reshaping.
- `Metal: the HLO rng op (jax.lax.rng_uniform) is not supported ...; use
  jax.random: ...`
- `Metal: jax.lax.mulhi on 64-bit integers is not supported ...` Use
  32-bit operands, or the CPU backend.
- `Metal: host offloading (jax.experimental.compute_on("device_host")) is
  not supported; run that part on the CPU backend instead: ...` Run that
  part outside `jit` under `jax.default_device(jax.devices("cpu")[0])`.
- `Metal: bf16 matmul too large: ... Split the matmul or use float32` and
  `Metal: matmul operand ... has a batch, row or column group of ...
  elements; XLA's matmul config counts them in 32 bits. Split the batch
  (lax.map, or chunks)`.

Messages starting with `MSL emitter: unsupported` come from the kernel
translator and name an internal operation:

- `MSL emitter: unsupported non-trivial unrealized_conversion_cast in op
  'builtin.unrealized_conversion_cast' at loc("loop_convert_fusion")`
  and `MSL emitter: unsupported sub-byte / odd-width integer type in op
  'arith.trunci' at ...`: int4/uint4.
- `MSL emitter: unsupported f64 type (Metal has no double precision) in op
  'llvm.load' at ...`: f64 data movement such as a strided slice. Keep it
  on the CPU.

From JAX or XLA:

- `MLIR translation rule for primitive 'eig' not found for platform mtl`
  (also `schur`, `hessenberg`, `tridiagonal`): compute these on the CPU.
- `Unsupported algorithm on the current device(s): ALG_DOT_TF32_TF32_F32`
  (also `F16_F16_F16`, `BF16_BF16_BF16`): use the `BF16_BF16_F32` family.

An error ending in `(metal-pjrt-plugin bug; please report it with the HLO
from XLA_FLAGS=--xla_dump_to=<dir>)` is our bug; please report it with
that dump.

## Memory kinds inside jit

The device has `device` and `pinned_host` memory (no `unpinned_host`).
`device_put` to either, and `pinned_host` in `out_shardings`, work inside
and outside `jit`. Computing on a `pinned_host` array inside `jit` works
but is slow (XLA logs "does device compute in host memory space.
Converting into host compute"); `device_put` it to `device` first.

## Host callbacks

- `host callback raised: ...` Your callback raised; its message follows.
- `metal-pjrt-plugin: unknown host callback id ...: the callable is gone.
  ...` A cached executable from another process. Clear the cache
  directory, or don't cache functions with callbacks.
- `Metal: host callbacks are unavailable: metal_pjrt_plugin did not install
  them (see the "host callbacks unavailable" warning logged when JAX
  initialized the plugin)` See that warning.
- `metal-pjrt-plugin: host callbacks do not support dtype int4 on platform
  mtl` Sub-byte types can't be passed to callbacks.
