# Host callbacks

`jax.pure_callback` (with `vmap_method` and inside custom_jvp),
`jax.experimental.io_callback` (ordered or not), `jax.debug.callback` and
`jax.debug.print` (inside `scan` and `grad` too) all work under `jit`. An
exception in your callback surfaces as a `JaxRuntimeError` (UNKNOWN):
"host callback raised: <message>". checkify's runtime checks use the TPU
rule (`debug_check` is a no-op). Tests: `tests/test_callbacks.py` and
three cases in `tests/test_lax.py`.

## Limitations

- One device only.
- Sub-byte dtypes are refused ("host callbacks do not support dtype int4
  on platform mtl"). complex64 works.
- Only modules lowered for mtl alone get mtl callbacks. A module lowered
  for several platforms (e.g. `jax.export` with `platforms=("mtl",
  "cpu")`) gets upstream's refusal ("multi-platform lowering for
  python_callback").
- Callbacks are synchronous: each waits for all earlier GPU work and runs
  on the XLA execution thread while the stream drains. Avoid them in hot
  loops. Launching mtl work from inside a callback and waiting on it is
  untested and may deadlock.
- Executables with callbacks skip the persistent compilation cache and
  are recompiled in every process.

Running callbacks on the stream's host-task worker instead was rejected:
a stream commit could then wait on a task that needs the GIL
([`performance.md`](performance.md#measured-and-dropped)).

## Why not JAX's GPU path

Upstream, `emit_python_callback` (`jax/_src/callback.py`) emits
`xla_ffi_python_{cpu,gpu}_callback` with an index into the module's host
callbacks, and jaxlib's IFRT (`xla/python/pjrt_ifrt/pjrt_executable.cc`)
passes those callables to the FFI only when the client's `platform_id` is
CPU, CUDA, ROCm or OneAPI. A C API plugin's id is
`Fingerprint64(<platform name>)`, so for mtl they never arrive, and that
check is compiled into jaxlib. The plugin keeps its own table instead.

## How it works

- **Lowering** (`metal_pjrt_plugin/_callbacks.py`): the plugin wraps
  `emit_python_callback` (and its alias in `jax.interpreters.mlir`). For
  mtl-only modules it wraps the callable with upstream's shape and dtype
  checks, registers it under a fresh 64-bit `callback_id` (salted per
  process), and emits a typed-FFI custom call
  `xla_ffi_python_metal_callback` with that id, upstream's
  `has_side_effect`, and a leading token for ordered effects.
- **Lifetime**: the callable is also added to
  `module_context.host_callbacks`, so the executable owns it; the table
  holds only a weak reference.
- **Handler** (`metal_pjrt/ffi/python_callback_ffi.cc`): synchronizes the
  stream, then calls a C trampoline with pointers, types and shapes.
  Buffers are shared memory, so there are no device-host copies.
- **Trampoline**: a `ctypes` function installed through
  `metal_pjrt_register_python_callback_trampoline`. It copies operands
  into NumPy arrays, calls the callable, writes results in place and turns
  exceptions into the error message.
- **Compilation cache**: ids are per process, so the plugin wraps
  `compiler.compile_or_get_cached` to bypass the cache for executables
  with callbacks. A stale cached one fails with "metal-pjrt-plugin:
  unknown host callback id ...: the callable is gone" rather than calling
  the wrong function.

## An upstream fix

Caching would work if jaxlib checked for the PJRT FFI extension instead
of platform ids, JAX let a plugin declare its callback target, and the
plugin registered `xla_ffi_python_gpu_callback` for "METAL".
`jax.experimental.buffer_callback` has the same check and would also need
a DLPack device type for Metal; that's out of scope.
