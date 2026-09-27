# Host callbacks on metal (`io_callback`, `pure_callback`, `jax.debug.*`)

Status: **supported** under `jit` on a single metal device:
`jax.pure_callback` (including `vmap_method=...` and custom_jvp wrappers),
`jax.experimental.io_callback` (ordered and unordered), `jax.debug.callback`
and `jax.debug.print` (ordered or not, inside `scan`/`grad`). A Python
exception in the callback surfaces as a `JaxRuntimeError` carrying its
message. Tests: `tests/test_callbacks.py` (compares against cpu),
`tests/test_lax.py` (three callback cases), and tinygp's quasiseparable
solver (`jax.debug.callback(_check_sorted, ...)`) in `tests/test_tinygp.py`.
checkify's runtime-error path still uses the TPU rule (`debug_check` is a
no-op; see `jax_plugins/openmetal/lowerings.py`).

## How the metal path works

jaxlib forwards an executable's host callbacks to the FFI handler only for the
cpu/cuda/rocm/oneapi platform ids (details below), and `emit_python_callback`
rejects "openmetal", so the plugin uses a self-managed table ("option A" below):

* **Lowering** (`jax_plugins/openmetal/callbacks.py`, installed by
  `jax_plugins.openmetal.initialize`): `jax._src.callback.emit_python_callback`
  is wrapped; for modules lowered only for "openmetal" it wraps the callable with
  upstream's output shape/dtype checks, registers it in a process-global
  table under a fresh 64-bit `callback_id` (random per-process salt in the
  high bits), and emits a typed-FFI (api_version 4) custom call
  `xla_ffi_python_metal_callback` with attribute `callback_id: u64`,
  `has_side_effect` as upstream, and a leading `!stablehlo.token`
  operand/result for ordered effects (same threading as the cpu/gpu path).
  All lowering rules (callback.py, debugging.py, checkify.py) look the name
  up on the module at call time, so nothing else is patched for lowering.
* **Lifetime**: the wrapped callable is also added to
  `module_context.host_callbacks`, so the compiled executable owns it (jaxlib
  wraps it in a `PyFfiLoadedHostCallback` that is never invoked on metal;
  it also makes JAX run the executable with runtime tokens, like cpu). The
  table holds a weak reference, removed when the executable dies.
* **Handler** (`metal_pjrt_plugin/ffi/python_callback_ffi.cc`, registered
  statically for "METAL"): binds `Ctx<Stream>`, `Attr<u64>("callback_id")`,
  remaining args/rets; calls `rt::Stream::Synchronize()` so all earlier GPU
  work is done, then calls a C trampoline with (pointer, PrimitiveType,
  dims) descriptors for the non-token args and results. Buffers are
  shared-storage MTLBuffers whose device pointers are host addresses, so no
  staging copies are needed. A nonzero return plus message becomes an
  `InternalError`.
* **Trampoline**: the dylib exports
  `metal_pjrt_register_python_callback_trampoline(fn)`; Python loads the
  already-loaded dylib with `ctypes.CDLL` (same handle) and installs a
  `CFUNCTYPE` function. ctypes takes the GIL on the XLA thread. It copies the
  args into numpy arrays (callees may keep them; XLA reuses the buffers),
  calls the callable, writes the results in place (row-major, the custom
  call's default layout), and turns any exception into the error message.
* **Persistent compilation cache**: callback ids are per-process, so
  `compiler.compile_or_get_cached` is wrapped to compile executables whose
  `host_callbacks` contain a metal callback without the persistent cache
  (upstream cpu/gpu can cache because ids are positions re-bound on load).
  If a stale executable were loaded anyway, the handler fails with "unknown
  metal host callback id" rather than calling the wrong function.

Limitations: single device only (no per-shard/partitioned semantics beyond
one device); sub-byte dtypes (int4 etc.) and complex types the backend lacks
are rejected; the callback runs on an XLA execution thread while the stream
is drained -- dispatching new metal work from inside a callback and waiting
on it is untested and may deadlock; each callback costs a full stream
synchronization (a GPU pipeline bubble), so callbacks in hot loops are slow;
executables with callbacks are recompiled in every process.

## How it works on cpu / cuda today

1. **Lowering** (`jax/_src/callback.py: emit_python_callback`):
   * hard-coded allowlist: `platform not in {"cpu","cuda","rocm","tpu","oneapi"}`
     raises the error above;
   * appends the Python callable to `ctx.module_context.host_callbacks` and
     emits an FFI custom call (`api_version = 4`, typed FFI) to
     `xla_ffi_python_{cpu,gpu}_callback` (or `xla_ffi_partitioned_...`) with a
     single attribute `index: u64` = position in `host_callbacks`, the
     operands (plus a leading `!stablehlo.token` for ordered effects), and
     `has_side_effect`.
2. **Compile** (`compiler.py` -> `PyClient::CompileAndLoad`, jaxlib
   `py_client.cc`): each callable is wrapped in a `PyFfiLoadedHostCallback`
   (subclass of `ifrt::PjRtFfiLoadedHostCallback`) and attached to the
   executable. This is platform-independent. The persistent compilation
   cache re-attaches callbacks by position on deserialize, which is why the
   custom call carries an index and not a pointer.
3. **Execute** (`xla/python/pjrt_ifrt/pjrt_executable.cc`,
   `PjRtLoadedExecutable::Execute`): the callables are collected into an
   `xla::FfiLoadedHostCallbacks {void** callbacks; uint32_t num_callbacks;}`
   and inserted into the `xla::ExecuteContext`'s FFI user data -- **only if
   `platform_id` is `CpuId()`, `CudaId()`, `RocmId()` or `OneapiId()`**. For a
   C-API plugin the IFRT client's `platform_id` is
   `Fingerprint64(PJRT_Client_PlatformName)`, i.e. `Fingerprint64("openmetal")`
   for us, so today **the user data is never attached**. When it is attached,
   `PjRtCApiLoadedExecutable::Execute` forwards it through
   `PJRT_ExecuteContext_Create` + `PJRT_FFI_Extension::user_data_add`
   (`ForwardExecuteContext` in `pjrt_c_api_client.cc`); our plugin (built on
   `pjrt_c_api_gpu_internal`) does expose the FFI extension, so that half
   would work.
4. **Handler.** GPU (`jaxlib/gpu/py_client_gpu.cc`):

   ```c++
   XLA_FFI_DEFINE_HANDLER_SYMBOL(kXlaFfiPythonGpuCallback, XlaFfiPythonGpuCallback,
       ffi::Ffi::Bind()
           .Ctx<ffi::PlatformStream<gpuStream_t>>()
           .Ctx<ffi::UserData<xla::FfiLoadedHostCallbacks>>()
           .Ctx<ffi::State<GpuTransposePlanCache>>()   // instantiate handler
           .Attr<uint64_t>("index")
           .RemainingArgs()
           .RemainingRets());
   ```

   It D2H-copies each arg on the stream, synchronizes, takes the GIL, wraps
   the host copies as read-only numpy arrays, calls
   `callbacks->callbacks[index]` (a borrowed `PyObject*`), checks/transposes
   the returned arrays to the result layout and H2D-copies them. Token
   operands/results are passed as `None` / skipped. CPU
   (`jaxlib/py_client_cpu.cc`) is the same minus the copies, registered for
   platform `"HOST"`.
5. **Registration of the handler.** The cuda plugin's nanobind extension
   (`jaxlib/cuda/cuda_plugin_extension.cc` + `gpu_plugin_extension.cc`)
   exposes `ffi_registrations()` (a dict of name -> handler-bundle capsule,
   including `xla_ffi_python_gpu_callback`,
   `xla_ffi_partitioned_python_gpu_callback`,
   `xla_buffer_python_gpu_callback`) and
   `register_custom_call_target(c_api, name, fn, xla_platform_name,
   api_version, traits)`, which calls the PJRT extension
   `PJRT_Gpu_Custom_Call::custom_call` (`PJRT_Gpu_Register_Custom_Call`).
   `jax_plugins/cuda/__init__.py` wires it up:

   ```python
   xla_client.register_custom_call_handler(
       "CUDA", functools.partial(ext.register_custom_call_target, c_api))
   for name, value in ext.ffi_registrations().items():
     xla_client.register_custom_call_target(name, value, platform="CUDA",
                                            api_version=1)
   ```

   `xla_client.register_custom_call_target` queues registrations per XLA
   platform name until a handler for that platform is registered, then
   replays them (this is also how `jax.ffi.register_ffi_target(...,
   platform="METAL")` from user code would reach our plugin). Inside the
   plugin, `PJRT_Gpu_Register_Custom_Call` calls
   `ffi::Ffi::RegisterStaticHandler(api, name, PJRT_GPU_PLUGIN_PLATFORM_NAME,
   bundle, traits)`; with our identity patch that platform is `"METAL"`, and
   the GPU backend's `CustomCallThunk` looks handlers up by the stream
   executor platform name (`kMetalPlatformId->ToName()`), so these must
   match.

## Design options that were considered

Because of the `platform_id` gate in step 3 (compiled into jaxlib; can't be
changed from the plugin), we cannot rely on `FfiLoadedHostCallbacks` user
data. Two options:

### Option A (implemented, works with stock jaxlib): self-managed callback table

(As built, see above: the handler is registered statically in the plugin
dylib and Python reaches it through an exported C function and a ctypes
trampoline, so no nanobind extension or `register_custom_call_handler` is
needed; the persistent cache is bypassed rather than salted.)

* **C++ (new nanobind extension shipped in `jax_plugins/openmetal`, e.g.
  `metal_plugin_extension.so`)**:
  * a process-global, mutex-protected table `uint64 id -> PyObject*`
    with `register(callable) -> id` / `unregister(id)`;
  * an FFI handler, e.g. target `metal_python_callback`:

    ```c++
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<void*>>()   // metal_pjrt::rt::Stream*
        .Attr<uint64_t>("callback_id")
        .RemainingArgs()
        .RemainingRets()
    ```

    body: commit + wait the stream (`rt::Stream::Synchronize()`); because
    buffers are shared-storage `MTLBuffer`s, the arg "device pointers" are
    host-readable `contents()` addresses, so D2H/H2D can be plain `memcpy`
    (or zero-copy numpy views for inputs, copied before returning); take the
    GIL; call; validate shapes/dtypes; memcpy results into the ret buffers.
    Must be registered with `XLA_FFI_Handler_Traits` side-effect semantics
    (custom call is emitted with `has_side_effect=True`) and must not be
    captured into command buffers (it isn't command-buffer compatible).
  * `ffi_registrations()` / `register_custom_call_target(c_api, ...)`
    exactly like the cuda extension (the latter calls
    `PJRT_Gpu_Register_Custom_Call` with platform `"METAL"`).
  * Requirements on the plugin runtime: the GPU `CustomCallThunk` path must
    work on the metal stream executor (the thunk hands the handler
    `stream->platform_specific_handle().stream`, which for us is the
    `rt::Stream*`), and a host-blocking handler in the middle of a stream
    must flush pending encoders first. Note that the handler runs on the
    thread executing the thunk; JAX sets `ExecutionMode::kSynchronous` only
    for CPU, so reentrancy into JAX from the callback (e.g. calling a jitted
    function on metal) could deadlock and must be tested.
* **Python glue (`jax_plugins/openmetal`)**:
  * in `initialize()`, after `register_plugin`, register the custom-call
    handler for `"METAL"` and the handler bundle (same code as the cuda
    plugin above);
  * a metal copy of `emit_python_callback` (the upstream one hard-codes the
    platform allowlist and target names): wrap the callback with the same
    output checks, `table.register` it, keep it alive for the executable's
    lifetime (`ctx.module_context.add_keepalive(handle)`, with a finalizer
    that unregisters), and emit
    `ffi.build_ffi_lowering_function("metal_python_callback",
    has_side_effect=True)(ctx, *operands, callback_id=np.uint64(id))`,
    threading the token as upstream does;
  * register metal lowerings for `io_callback_p`, `pure_callback_p`,
    `debug_callback_p`, `debug_print_p` (and `check_p` if runtime errors are
    wanted) that are copies of the upstream rules calling the metal emitter.
    The upstream `pure_callback`/`io_callback` rules call
    `callback.emit_python_callback` by module-global name, so either the
    rules are copied or `callback.emit_python_callback` is monkeypatched to
    dispatch `platform == "openmetal"` to our version (simpler, but patches JAX
    internals).
  * Caching caveat: since the callback is *not* added to
    `module_context.host_callbacks`, JAX will use the persistent
    compilation cache for these modules. The `callback_id` is baked into
    the HLO (so it is part of the cache key) but ids are per-process: a
    later process could hit a cached executable whose id now names a
    different callback. Either also call
    `ctx.module_context.add_host_callback(cb)` (harmless: IFRT creates a
    `PyFfiLoadedHostCallback` it never forwards for metal; it also marks the
    executable as having host callbacks) *and* key the table by a
    per-module index looked up via a per-executable table, or simply disable
    the persistent cache for modules containing `metal_python_callback`
    (e.g. salt the module with a random attribute). The simplest correct
    choice is the latter.

### Option B (upstream change, would restore persistent caching): make metal a first-class callback platform

* jaxlib: add `MetalId()` to the `platform_id` gate in
  `PjRtLoadedExecutable::Execute` (or better, gate on "client exposes the
  PJRT FFI extension"), so `FfiLoadedHostCallbacks` is forwarded through
  `PJRT_FFI_UserData_Add` to our plugin.
* jax: add `"openmetal"` to the allowlist in `emit_python_callback` and map it
  to `device = "gpu"` target names (or allow a plugin to declare its
  callback target), and register the `debug_*` rules for it.
* plugin: ship `xla_ffi_python_gpu_callback` (+ partitioned / buffer
  variants) with the upstream signature
  (`PlatformStream`, `UserData<FfiLoadedHostCallbacks>`,
  `State<...>`, `Attr<u64>("index")`, remaining args/rets), registered for
  `"METAL"` via the extension + `register_custom_call_handler("METAL", ...)`
  as above. This keeps the index-based design, which is compatible with the
  persistent compilation cache.

`jax.experimental.buffer_callback` (`xla_buffer_python_gpu_callback`,
DLPack-based) is a separate path with the same gating and would need a
DLPack device type for Metal; out of scope.
