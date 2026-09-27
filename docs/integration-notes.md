# Integration notes (XLA @ 91888df6, jaxlib 0.11.2)

Source-verified facts that drive the implementation. Paths are relative to the
XLA tree (`external/xla+` in the Bazel output base).

The load-bearing sites (OneAPI branches, the copied `AddLoweringPasses`
prefix, assumptions of the plugin's HLO passes) are snapshotted by
`//metal_pjrt_plugin/xla_tripwire:xla_tripwire_test`, which fails with a diff
when a pin bump changes one; `metal_pjrt_plugin/xla_tripwire/oneapi_callsites.py`
lists every `IsOneAPI()`/`IsIntelGpu()`/`is_sycl` line XLA-wide against a
golden list. Both have an `--update` mode.

## Identity

- StreamExecutor platform: `PLATFORM_DEFINE_ID(kMetalPlatformId, METAL)`;
  `Platform::Name()` is "METAL", canonical name "metal"
  (`xla/service/platform_util.cc` lowercases unknown names).
- PJRT platform (the JAX platform): name "openmetal" (`MetalName()`), id
  `tsl::Fingerprint64("openmetal")`. Not "metal": that is Apple's
  closed-source jax-metal plugin, and both may be installed together.
  The StreamExecutor, FFI and XLA-internal names stay "METAL"/"metal": they
  live in registries private to our dylib (it exports only `GetPjrtApi`,
  the callback trampoline and two testing hooks, `metal_pjrt_memory_stats`
  and `metal_pjrt_memory_pressure`, used by `tests/test_memory.py` via
  ctypes; they are not an API), so they cannot collide with another plugin.
  JAX looks lowerings up by `backend.platform`, i.e. `MetalName()`, so
  `register_plugin`'s name and every `PLATFORM` in `jax_plugins/openmetal`
  must be exactly "openmetal".
- Things keyed on identity that need registration from our code:
  `Compiler::RegisterCompilerFactory(kMetalPlatformId, ...)`,
  `TransferManager` for `kMetalPlatformId`,
  `PjRtRegisterDefaultCompiler(MetalName(), StreamExecutorGpuCompiler(MetalId, kMetalPlatformId))`,
  `StreamExecutorPlatformIdMapping::Global().AddMapping(kMetalPlatformId, MetalId)`,
  `XLA_COLLECTIVES_REGISTER(MetalName() and "METAL", "stub", 1, GpuCollectivesStub)`
  (the client resolves collectives by the PJRT name, collective thunks by the
  SE name).
- Things keyed on identity that need XLA patches (kept in `third_party/xla/patches/`):
  1. `xla/pjrt/gpu/se_gpu_pjrt_client.cc:1874-1880` picks the PJRT platform
     name by macro; add `TENSORFLOW_USE_METAL` -> `MetalName()`.
  2. `xla/pjrt/pjrt_compiler.h:493` `IsGpuId` must include the metal id.
  3. `xla/service/gpu/gpu_executable.cc:534-551` platform-id switch must
     accept `kMetalPlatformId`.
  4. `xla/service/gpu/BUILD` `ptx_custom_kernel_emitter` has no branch when no
     GPU is configured: provide a stub `EmitPtxCustomKernelThunk`.
  5. `xla/service/gpu/gpu_compiler.{h,cc}`: `CompileToBackendResult` (private,
     non-virtual) stack-allocates a `CubinCustomKernelCompiler` (`final`) and
     passes it to `CompileModuleToLlvmIr` -> `IrEmitterContext`. Patch 0002
     moves that construction into a protected virtual
     `GpuCompiler::CreateKernelCompiler(LlvmIrCompiler, DeviceDescription,
     DebugOptions, ThreadPool*)` whose default is unchanged, so `MetalCompiler`
     can return a `MetalKernelCompiler` (see Codegen).
- Compute capability: `GpuComputeCapability` is a closed variant of
  Cuda/Rocm/OneAPI (`xla/stream_executor/device_description.h:98-178`).
  Default-constructed reads as CUDA 0.0. v1 reports **OneAPI** so that the
  SPIR-V/Intel branches are taken (scalar-only transpose, explicit NaN
  propagation, command buffers off, `ExecutableAbiVersion` accepted, atomics
  via the SPIRV path). Adding an Apple alternative is a later patch.

## StreamExecutor contract (what MetalExecutor must implement)

- Base: `gpu::GpuExecutor(Platform*, int ordinal)` (adds `device_ordinal()`).
- Pure: `Init`, `CreateStream(priority)`, `CreateEvent`, `Allocate(size,
  memory_space)`, `Deallocate`, `HostMemoryAllocate`, `SynchronizeAllActivity`,
  `SynchronousMemcpy` x2, `DeallocateStream`, `EnablePeerAccessTo`,
  `CanEnablePeerAccessTo`, `CreateDeviceDescription`.
- Required by the PJRT client: `DeviceMemoryUsage` returns true;
  `CreateMemoryAllocator(kCollective|kHost)`; `LoadKernel`, `UnloadKernel`,
  `LoadModule`, `GetSymbol` (constants), `CreateOrShareConstant`.
- Memory spaces (`memory_space.h`): kDevice=0, kUnified=1, kCollective=2,
  kHost=5. XLA colors: default 0, collective 1, temp 2. With unified memory all
  of these are the same shared MTLBuffer allocation.
- Streams: `StreamCommon` subclass implementing `WaitFor(Stream*)`,
  `WaitFor(Event*)`, `RecordEvent`, `Memcpy` x3, `MemZero`, `Memset32`,
  `BlockHostUntilDone`, `DoHostCallbackWithStatus`, `platform_specific_handle`.
  `LocalDeviceState` creates ~18 streams per device and asks for
  `StreamPriority::Highest` on some (accept and ignore).
- Events: `Event` subclass overriding `PollForStatus`.
- Kernels: `Kernel` subclass with `Arity`, `GetMaxOccupiedBlocksPerCore`,
  `Launch(thread, block, cluster, stream, KernelArgs)`. Args arrive as
  `KernelArgsPackedArrayBase` with one 8-byte device pointer per argument in
  HLO buffer order, no scalars, shared bytes usually 0.
- Binary format: no generic loader spec exists; MSL source travels in
  `KernelLoaderSpec::CreateOwningCudaCubinInMemorySpec(bytes, name, arity)`
  exactly as SYCL smuggles SPIR-V. `LoadKernel` compiles the MSL and looks up
  the function by `spec.kernel_name()`.
- Metal limit: at most 31 buffer arguments per kernel. Above that the emitter
  writes `// xla_metal_argbuffer` before the kernel and takes
  `constant ulong* xla_args [[buffer(0)]]` (one GPU address per argument,
  cast to `device char*`); LoadKernel records this from the MSL text and
  Stream::Launch packs `gpuAddress + offset` into a setBytes payload (<= 4 KB,
  512 args) and calls `useResources` on the distinct MTLBuffers.
- Types renamed: `DeviceAddressBase` (alias `DeviceMemoryBase`),
  `MemorySpace` (alias `MemoryType`).

## Compiler contract (MetalCompiler : GpuCompiler)

Template: `xla/service/gpu/intel_gpu_compiler.{h,cc}`.
- Ctor: `GpuCompiler(kMetalPlatformId, "spirv64-unknown-unknown", "")` so the
  emitters take the SPIR branches (`target_util.cc`, `fusion_emitter.cc`).
- Pure virtuals: `GetLLVMCommandLineOptions`, `AddPaddingForGpublasGemms`
  (no-op), `OptimizeHloConvolutionCanonicalization` (no-op in v1: no conv),
  `CompileTargetBinary(module_config, llvm::Module*, device_description,
  relocatable, debug_module, shard)`.
- `AddConfigAssignerPass`: no-op. `OptimizeHloPostLayoutAssignment`:
  the base pipeline (GemmRewriter), then
  `MetalDotOperandUpcaster` on the dots left for the loop emitter and
  `CheckPostGemmRewriter` (`compiler/passes/hlo_checks.h`): no narrow-operand
  kDot, no TopK custom call, every `__cublas$lt$matmul` has types and an
  epilogue `MetalBlasLt` supports (`blas/blas_lt_support.h`, shared with
  `MetalBlasLt`). A violation is a compile error naming the JAX op and line.
- Kernels are compiled one module at a time via
  `CompileSingleModule -> CompileTargetBinary`; result bytes become a
  `CustomKernelThunk` with a cubin spec. `GpuExecutable::binary()` holds only
  the constants module (`LoadModule` + `GetSymbol` per constant).
- Registration: static initializer calling `Compiler::RegisterCompilerFactory`
  in an `alwayslink` target.
- Base `OptimizeHloPostLayoutAssignment` unconditionally rewrites dots into
  library custom calls; on OneAPI every one is `__cublas$lt$matmul`
  (`GetNonFp8GemmCustomCallTarget`; the legacy `__cublas$gemm` fallback is
  gone), so matmul runs through `MetalBlasLt` (MPS or steel kernels).
- Triton is gated off for non-CUDA/ROCm; cuDNN passes default to no-op.

## Codegen: MLIR -> EmitC -> MSL

- Emitter pipeline (`mlir_kernel_emitter.cc`): `AddLoopTransformationPasses`
  then `AddLoweringPasses`: LowerTensors, SimplifyArith, SimplifyAffine,
  ConvertIndexType, float conversions, ExpandFloatOps, SCFToControlFlow,
  `createLowerToLLVMGPUPass(device)` (NVVM default, ROCDL, LLVM-SPV).
- Launch size: XLA's `AnnotateKernelLaunchDimensions` does nothing for SPIR,
  so `MetalKernelCompiler` reads the threadgroup size from the MLIR before
  any pass runs (ranged `gpu.thread_id` x/y/z, or the outermost `scf.forall`
  of the `xla/codegen/emitters` kernels) and emits
  `[[max_total_threads_per_threadgroup(N)]]`. The runtime parses N back from
  the source and refuses (InvalidArgument, before encoding) any launch with
  more than min(N, pipeline limit) threads per threadgroup: release Metal
  does not check, and a larger threadgroup is undefined behaviour on the GPU.
- Seam: the MLIR emitters call only
  `ir_emitter_context.kernel_compiler()->CompileMlirToLlvm(...)`
  (`mlir_kernel_emitter.cc`), and `KernelCompiler::CompileMlirToLlvm` is
  virtual (`xla/backends/gpu/codegen/kernel_compiler.h`). `MetalKernelCompiler`
  (`metal_pjrt_plugin/codegen/metal_kernel_compiler.{h,cc}`) overrides it and
  delegates `Compile`/`CompileToTargetBinary`/`CompileTritonToLlvm` to an inner
  `CubinCustomKernelCompiler`; `GpuCompiler` sets the pre-optimization hook on
  the outer object, so the inner one's `LlvmIrCompiler` calls it.
- MSL has no `goto` and no labeled statements (verified), so we do not
  translate the final CFG. Instead `MetalKernelCompiler` runs the pipeline up to but not
  including SCFToControlFlow, then converts func/scf/arith/math/vector/gpu and
  the LLVM-dialect memory ops that LowerTensors introduced into EmitC, and
  prints MSL. The kernel wrapper with `[[buffer(i)]]` and thread-id attributes
  is generated as text around an EmitC-printed body function.
- Thread ids are `gpu.thread_id/block_id/block_dim/grid_dim` with `xla.range`
  attrs; barriers `gpu.barrier`; shuffles `gpu.shuffle`.
- The MSL text is carried to `CompileTargetBinary` inside a stub `llvm::Module`
  (a global string plus the declared kernel function), so no other XLA code
  changes. `CompileTargetBinary` returns the MSL bytes as the "cubin".
- The constants module is real LLVM IR with global initializers; the Metal
  `CompileTargetBinary` serializes those into a private container that
  `LoadModule`/`GetSymbol` understand.

## FFI custom calls

- Custom calls with `API_VERSION_TYPED_FFI` become `CustomCallThunk`s
  (`xla/service/gpu/thunk_emitter.cc`), which look the handler up with
  `ffi::FindHandler(target, platform_name)`; `platform_name` is the SE
  platform name "METAL". Both registration and lookup go through
  `PlatformUtil::CanonicalPlatformName` (`xla/ffi/ffi_registry.cc`), so
  "METAL" and "metal" are the same key ("metal").
- The registry is a static inside the plugin dylib's copy of XLA. Handlers in
  `metal_pjrt_plugin/ffi` register with
  `XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), name, "METAL", h)` in
  an `alwayslink` library; nothing is needed from Python
  (`jax.ffi.ffi_call(name, ...)` just emits the custom call). Handlers
  compiled into another library (e.g. jaxlib's CUDA ones) are not visible.
- Stream: bind `.Ctx<xla::ffi::Stream>()` (`xla/backends/gpu/ffi.h`) to get
  the `se::Stream*`. `MetalStream::platform_specific_handle().stream` is the
  `metal_pjrt::rt::Stream*` (so `PlatformStream<rt::Stream*>` also works) and
  `stream->parent()` is the `MetalExecutor` owning the `rt::Device`.
  `metal_pjrt::ffi::GetMetalContext` / `LaunchMsl` wrap this, with a kernel
  cache keyed by (device, MSL source, function). `metal$scan` builds its MSL
  and compiles its pipeline once per call site in an FFI instantiate handler
  (state held by the thunk; no stream there, so it uses executor 0's device,
  `DefaultMetalDevice`) and only encodes the dispatch per execution
  (`LaunchKernel`). The state is not serializable, so a deserialized
  executable re-runs instantiate (checked: persistent-cache hit, same result).
- The backend config must be an MLIR dictionary (JAX writes it raw; our
  rewriters put it in `GpuBackendConfig.custom_call_backend_config.attributes`).
- Handlers: `metal$scan` (target of MetalScanRewriter; `tests/test_scan.py`
  also calls it through `jax.ffi.ffi_call`); `//metal_pjrt_plugin/ffi:cub_sort_test`
  looks handlers up in the static registry and invokes them as XLA does.
  `METAL_PJRT_DISABLE_REWRITES=scan|all` turns the scan rewriter off.
- XLA's SortRewriter targets `xla.gpu.ext.cub_sort_keys` / `cub_sort_pairs`
  (`ffi/cub_sort_ffi.cc`, mirroring `cub_sort_kernel_cuda.cc`): the
  instantiate stage returns the scratch size as `int64_t` state, which
  `EstimateCubSortScratchSize` reads at compile time (it calls the handler
  with null buffers and a zero-sized scratch); execute recomputes the layout
  and refuses a smaller scratch before encoding. Default layouts are forced
  for these calls, so rows are contiguous. SortRewriter's non-CUDA rule is
  total elements > 16384 regardless of row length; `RunHloPasses` expands
  rows <= 64 of such sorts with `MetalSortExpander` first.
  `METAL_PJRT_DISABLE_REWRITES=cubsort` turns SortRewriter off.
  (A `metal$softmax` rewriter existed until 2026-09-27; removed after an
  end-to-end A/B, docs/performance.md.)
- Dense linear algebra (`metal_pjrt_plugin/linalg/`): handlers
  `metal$cholesky`, `metal$triangular_solve` (targets of
  `MetalLinalgRewriter`, run at the start of `MetalCompiler::RunHloPasses`,
  row-major operands) and `metal$lapack_{getrf,geqrf,orgqr,syevd,gesdd,
  gesdd_novec}` (targets of `jax_plugins/openmetal/linalg_lowerings.py`,
  column-major operands via layout constraints). Matrices up to 32x32 in
  `metal$cholesky`, `metal$triangular_solve` and `metal$lapack_getrf` run as
  GPU kernels on the stream, with no synchronization. Above that, each
  handler calls
  `rt::Stream::Synchronize()` and then runs Accelerate LAPACK/BLAS directly on
  the shared-storage buffers (zero copy), synchronously on the thunk thread
  (a stream host task was measured and not faster: docs/performance.md).
  f32 only. `METAL_PJRT_DISABLE_LAPACK=1` is the one switch for both
  (checked per compile by the pass and per lowering by the Python rules) and
  falls back to XLA's expanders / JAX's generic lowerings. Ownership table:
  the `linalg_lowerings.py` docstring.

## PJRT client

- `GetStreamExecutorGpuClient` builds `LocalDeviceState`s, allocators
  (`kPlatform` is the simplest that works; `kBFC` needs the two
  `CreateMemoryAllocator` kinds), and calls `GpuCollectives::Resolve(name)`
  which CHECK-fails without a registration (hence the stub under `MetalName()`).
- `//xla/service:gpu_plugin` is empty on macOS (all deps behind
  `if_gpu_is_configured`), so our plugin target lists `gpu_compiler`,
  `gpu_executable`, the transfer manager and thunk runtime deps explicitly.
- Plugin dylib: copy the CPU plugin's macOS linkopts
  (`-Wl,-exported_symbol,_GetPjrtApi`, `-install_name @rpath/...`).
- The client option `platform_name` selects the StreamExecutor platform
  (`PlatformUtil::GetPlatform`), so the plugin passes "METAL", not the PJRT
  name; unset, it is "gpu", which patch 0001 canonicalizes to "metal" on
  macOS (the topology path relies on that).
- Platform version: `GpuPlatformVersionFromDevices`
  (`se_gpu_pjrt_client.cc:262-277`) reports `"oneapi " +
  runtime_version.ToString()`. JAX hashes it into its compilation cache key,
  so `MetalExecutor` sets `runtime_version` to `{1, fingerprint(LC_UUID of
  the plugin image), fingerprint(METAL_PJRT_DISABLE_REWRITES,
  METAL_PJRT_DISABLE_LAPACK)}`: a rebuilt plugin or a different compile-time
  setting gets a different key.
- Persistent compilation cache. Two things kept JAX 0.11.2 from using it
  for "openmetal", both fixed without an XLA patch:
  (1) serialization failed with "Unsupported platform ID for
  XlaExecutableAbiVersion": XLA's GPU C API shim adds a `PJRT_AbiVersion`
  extension, so `PjRtCApiExecutable::GetAbiVersion` reports the OneAPI ABI,
  and jaxlib's IFRT (`GetXlaExecutableVersion`,
  `pjrt_ifrt/pjrt_executable.cc`) accepts only TPU/CUDA/ROCm ids. It falls
  back to a version-less `XlaExecutableVersion` (as for CPU) when
  `GetAbiVersion` is Unimplemented, which the C API client returns when the
  extension is absent. So the plugin exports its own `GetPjrtApi`
  (`pjrt/metal_pjrt_api.cc`, depending on `pjrt_c_api_gpu_internal`
  instead of `pjrt_c_api_gpu`): XLA's `GetGpuPjrtApi()` with the
  AbiVersion node dropped from the extension list. Nothing inside the
  plugin reads that extension (its only readers are the C API client, the
  IFRT version query and the `abi_helpers` AOT tools). Deserialization
  (`StreamExecutorExecutable::Deserialize`) checks only the PJRT client
  name and then loads the GPU AOT result, so with no ABI version the cache
  key's platform version is the only guard against a foreign build, which
  is what it is for. Executables round-trip: fusions, GEMM, sort,
  scan/while, `metal$scan`/LAPACK FFI calls and
  constant-heavy programs deserialize in a fresh process and give
  bitwise-identical results (MSL rides in the GPU executable's asm/binary,
  the constants container in the constants module's binary).
  (2) `compilation_cache.is_cache_used` accepts only the platforms in a
  local list (tpu/gpu/cpu/neuron). `jax_plugins/openmetal` wraps it and, for
  openmetal backends only, calls the original with a proxy whose `platform` is
  "gpu" (all else forwarded), so upstream's one-shot bookkeeping still
  runs. `tests/test_compilation_cache.py` asserts the list is still there.
  The plugin never sets `jax_compilation_cache_dir` (dfm, 2026-09-27): the
  setting is process-wide and `initialize()` runs for every installed
  plugin whatever `JAX_PLATFORMS` says, so a default directory turned the
  cache on for CPU-only users too. Users configure it
  (`JAX_COMPILATION_CACHE_DIR`, `jax.config`; README, "Compilation cache");
  `tests/test_compilation_cache.py` checks the plugin leaves it unset.
  `~/.cache/openmetal/compilation_cache`, the default the plugin used to
  set, is orphaned and can be deleted. JAX's own thresholds apply
  (only compiles over `jax_persistent_cache_min_compile_time_secs`, 1 s by
  default, are written). Executables with metal host callbacks bypass it
  (`docs/callbacks.md`). The per-setting `~/.cache/jax_metal/variants/`
  directories an older plugin created, and `~/.cache/jax_metal/compilation_cache`
  from before the "openmetal" rename, are orphaned and can be deleted.

## Host transfers (2026-09-27, roadmap 3.2 step 0)

- The client option `should_stage_host_to_device_transfers` is False. It was
  already moot: `ShouldStageHostToDeviceTransfers` also requires
  `!IsHostMemoryPinned(ptr)`, and the default `IsHostMemoryPinned` asks
  `GetPointerMemorySpace`, which reports every pointer outside our buffers as
  `kHost`, so numpy memory counted as pinned and nothing was staged
  (measured: the host pool never allocated; 100 MB device_put 9.4 vs 8.7 ms,
  staging option on vs off).
- JAX 0.11.2 hands numpy arrays to `BufferFromHostBuffer` with semantics that
  let XLA read them after `device_put` returns (the H2D is dispatched on a
  worker thread), so refilling the array right after `device_put` (a data
  loader) changed what the device got; this predates the staging change.
  `metal_pjrt_api.cc` now wraps `PJRT_Client_BufferFromHostBuffer`: dense
  data is copied into a malloc'd buffer before returning (as CUDA does for
  pageable memory) and freed on `done_with_host_buffer`; strided and
  sub-byte inputs take XLA's synchronous `kImmutableOnlyDuringCall` path.
  Cost (5 interleaved rounds, medians): device_put 100 MB 2.56 -> 5.14 ms,
  1 MB 84 -> 104 us, 8 floats 72 -> 73 us; D2H unchanged.
- Roadmap 3.2 step 1 (part): `Stream::MemcpyHostToDevice/DeviceToHost`
  memcpy on the calling thread when the stream is idle (no open command
  buffer, deferred waits all satisfied, fence caught up); otherwise a host
  task as before.
- Only `pinned_host` memory kinds still use XLA's host BFC pool (never
  shrinks).

## Owning the PJRT entry point: decided against (roadmap 3.4, 2026-09-27)

Not doing a `MetalPjRtClient` behind our own `GetPjrtApi`: owning the entry
point would remove only ~15 of patch 0001's ~200 lines; XLA has no
`StreamExecutorGpuClient` class to subclass at this pin (the client is built
by `GetStreamExecutorGpuClient`); and zero-copy host import
(`BufferFromHostBufferSupportsZeroCopy` / `ImportForeignMemory`) would need a
~400-line client fork. The small `GetPjrtApi` wrapper in
`pjrt/metal_pjrt_api.cc` stays (drops the ABI-version extension, copies
device_put's host data). (For roadmap.md's "Decided against".)
