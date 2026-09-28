# Host callbacks on mtl (`io_callback`, `pure_callback`, `jax.debug.*`)

Status: **supported** under `jit` on a single mtl device:
`jax.pure_callback` (including `vmap_method=...` and custom_jvp wrappers),
`jax.experimental.io_callback` (ordered and unordered), `jax.debug.callback`
and `jax.debug.print` (ordered or not, inside `scan`/`grad`). A Python
exception in the callback surfaces as a `JaxRuntimeError` carrying its
message. Tests: `tests/test_callbacks.py` (against CPU) and three cases in
`tests/test_lax.py`. checkify's runtime-error path uses the TPU rule
(`debug_check` is a no-op; `metal_pjrt_plugin/lowerings.py`).

## Why not the CUDA path

Upstream, `emit_python_callback` (`jax/_src/callback.py`) accepts only
cpu/cuda/rocm/tpu/oneapi and emits `xla_ffi_python_{cpu,gpu}_callback` with
an `index` into the module's `host_callbacks`. At execution, jaxlib's IFRT
(`PjRtLoadedExecutable::Execute`, `xla/python/pjrt_ifrt/pjrt_executable.cc`)
attaches those callables as FFI user data only when the client's
`platform_id` is the CPU, CUDA, ROCm or OneAPI id. A C API plugin's id is
`Fingerprint64(<platform name>)`, so for mtl the user data never
arrives, and that gate is compiled into jaxlib. The plugin therefore keeps
its own table.

## How the mtl path works

- **Lowering** (`metal_pjrt_plugin/callbacks.py`, installed by
  `initialize()`): `jax._src.callback.emit_python_callback` and its public
  alias `jax.interpreters.mlir.emit_python_callback` are wrapped. For
  modules lowered only for mtl it wraps the callable with upstream's
  output shape/dtype checks, registers it in a process-global table under a
  fresh 64-bit `callback_id` (random per-process salt in the high bits), and
  emits a typed-FFI (api_version 4) custom call
  `xla_ffi_python_metal_callback` with attribute `callback_id: u64`,
  `has_side_effect` as upstream, and a leading `!stablehlo.token`
  operand/result for ordered effects (under Shardy with its own sharding
  annotation, as upstream). The lowering rules in callback.py,
  debugging.py and checkify.py look the function up on the module at call
  time, so nothing else is patched.
- **Lifetime**: the wrapped callable is also added to
  `module_context.host_callbacks`, so the executable owns it (and JAX runs
  it with runtime tokens, as on CPU); the table holds a weak reference that
  goes with the executable.
- **Handler** (`metal_pjrt/ffi/python_callback_ffi.cc`, registered
  statically for "METAL"): calls `rt::Stream::Synchronize()` so all earlier
  GPU work is done, then a C trampoline with (pointer, PrimitiveType, dims)
  for the non-token operands and results. Buffers are shared-storage
  `MTLBuffer`s whose device pointers are host addresses, so there are no
  device-host copies. A nonzero return plus message becomes an
  `InternalError`.
- **Trampoline**: the dylib exports
  `metal_pjrt_register_python_callback_trampoline(fn)`; Python opens the
  already-loaded dylib with `ctypes.CDLL` and installs a `CFUNCTYPE`
  function (ctypes takes the GIL). It copies the operands into numpy arrays
  (XLA reuses the buffers after the call), calls the callable, writes the
  results in place (row-major, the custom call's default layout) and turns
  any exception into the error message.
- **Persistent compilation cache**: ids are per-process, so
  `compiler.compile_or_get_cached` is wrapped to compile executables with an
  mtl callback without the cache (tested in `tests/test_callbacks.py`
  with the cache on). A stale executable loaded anyway
  fails with "unknown metal host callback id" instead of calling the wrong
  function.

## Limitations

- One device only; sub-byte and complex dtypes are refused.
- Each callback costs a full stream synchronization and runs on the XLA
  execution thread while the stream is drained, so callbacks in hot loops
  are slow. Dispatching new mtl work from inside a callback and
  waiting on it is untested and may deadlock.
- The synchronization is deliberate: callbacks stay off the stream's
  host-task worker, so no stream commit can wait for a task that needs the
  GIL (`docs/performance.md`, "Measured and rejected", host LAPACK as a
  stream host task).
- Executables with callbacks are recompiled in every process.

The upstream fix that would restore caching: jaxlib gates on "the client
exposes the PJRT FFI extension" rather than platform ids, jax allows a
plugin to declare its callback target, and the plugin registers
`xla_ffi_python_gpu_callback` with the upstream signature for "METAL".
`jax.experimental.buffer_callback` (DLPack-based) has the same gate and
would also need a DLPack device type for Metal; out of scope.
