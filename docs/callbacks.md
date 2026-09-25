# Host callbacks on metal (`io_callback`, `pure_callback`, `jax.debug.*`)

Status: **not supported.** `io_callback`, `pure_callback`, `jax.debug.print`,
`jax.debug.callback`, and checkify's runtime-error path all fail at lowering
with

    ValueError: `EmitPythonCallback` not supported on metal backend.

(`debug_print`/`debug_callback` lowering rules are registered for metal in
`jax_plugins/metal/lowerings.py` so they produce this same error rather than
"MLIR translation rule ... not found". `checkify.check` uses the TPU rule:
`debug_check` is a no-op, functionalized `checkify.checkify` works, and
unfunctionalized checks raise the usual functionalization error.)

There is no honest pure-Python workaround: every path needs the compiled
program to stop, hand buffers to Python, and resume. Treating `debug.print` as
a no-op would silently drop output, so we don't. The rest of this note is what
real support requires. References are to JAX 0.11.2 (`.venv/.../jax/_src`), the
XLA checkout the plugin builds against (`external/xla+`), and a jaxlib source
checkout (`~/src/jax-ml/jax`, Oct 2025 -- structure unchanged, re-verify
details against the 0.11.2 tag).

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
   `Fingerprint64(PJRT_Client_PlatformName)`, i.e. `Fingerprint64("metal")`
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

## What the metal plugin would need

Because of the `platform_id` gate in step 3 (compiled into jaxlib; can't be
changed from the plugin), we cannot rely on `FfiLoadedHostCallbacks` user
data. Two options:

### Option A (works with stock jaxlib): self-managed callback table

* **C++ (new nanobind extension shipped in `jax_plugins/metal`, e.g.
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
* **Python glue (`jax_plugins/metal`)**:
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
    dispatch `platform == "metal"` to our version (simpler, but patches JAX
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

### Option B (upstream change): make metal a first-class callback platform

* jaxlib: add `MetalId()` to the `platform_id` gate in
  `PjRtLoadedExecutable::Execute` (or better, gate on "client exposes the
  PJRT FFI extension"), so `FfiLoadedHostCallbacks` is forwarded through
  `PJRT_FFI_UserData_Add` to our plugin.
* jax: add `"metal"` to the allowlist in `emit_python_callback` and map it
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
