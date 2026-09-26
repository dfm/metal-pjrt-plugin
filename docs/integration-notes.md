# Integration notes (XLA @ 91888df6, jaxlib 0.11.2)

Source-verified facts that drive the implementation. Paths are relative to the
XLA tree (`external/xla+` in the Bazel output base).

## Identity

- StreamExecutor platform: `PLATFORM_DEFINE_ID(kMetalPlatformId, METAL)`;
  `Platform::Name()` is "METAL", canonical name "metal"
  (`xla/service/platform_util.cc` lowercases unknown names).
- PJRT platform: name "metal", id `tsl::Fingerprint64("metal")`.
- Things keyed on identity that need registration from our code:
  `Compiler::RegisterCompilerFactory(kMetalPlatformId, ...)`,
  `TransferManager` for `kMetalPlatformId`,
  `PjRtRegisterDefaultCompiler("metal", StreamExecutorGpuCompiler(MetalId, kMetalPlatformId))`,
  `StreamExecutorPlatformIdMapping::Global().AddMapping(kMetalPlatformId, MetalId)`,
  `XLA_COLLECTIVES_REGISTER("metal", "stub", 1, GpuCollectivesStub)`.
- Things keyed on identity that need XLA patches (kept in `third_party/xla/patches/`):
  1. `xla/pjrt/gpu/se_gpu_pjrt_client.cc:1874-1880` picks the PJRT platform
     name by macro; add `TENSORFLOW_USE_METAL` -> "metal".
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
- `AddConfigAssignerPass`: no-op. `OptimizeHloPostLayoutAssignment`: base.
- Kernels are compiled one module at a time via
  `CompileSingleModule -> CompileTargetBinary`; result bytes become a
  `CustomKernelThunk` with a cubin spec. `GpuExecutable::binary()` holds only
  the constants module (`LoadModule` + `GetSymbol` per constant).
- Registration: static initializer calling `Compiler::RegisterCompilerFactory`
  in an `alwayslink` target.
- Base `OptimizeHloPostLayoutAssignment` unconditionally rewrites dots into
  `__cublas$gemm` custom calls (GemmThunk -> `StreamExecutor::AsBlas()`), so
  matmul needs a `blas::BlasSupport` for Metal (MPS, Objective-C++).
- Triton is gated off for non-CUDA/ROCm; cuDNN passes default to no-op.

## Codegen: MLIR -> EmitC -> MSL

- Emitter pipeline (`mlir_kernel_emitter.cc`): `AddLoopTransformationPasses`
  then `AddLoweringPasses`: LowerTensors, SimplifyArith, SimplifyAffine,
  ConvertIndexType, float conversions, ExpandFloatOps, SCFToControlFlow,
  `createLowerToLLVMGPUPass(device)` (NVVM default, ROCDL, LLVM-SPV).
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
  cache keyed by (device, MSL source, function).
- The backend config must be an MLIR dictionary (JAX writes it raw; our
  rewriters put it in `GpuBackendConfig.custom_call_backend_config.attributes`).
- Handlers: `metal$softmax`, `metal$scan` (targets of MetalSoftmaxRewriter /
  MetalScanRewriter), `metal$test_scale` (plumbing test,
  `scripts/ffi_check.py`). `METAL_PJRT_DISABLE_REWRITES=softmax,scan|all`
  turns the rewriters off.
- Dense linear algebra (`metal_pjrt_plugin/linalg/`): handlers
  `metal$cholesky`, `metal$triangular_solve` (targets of
  `MetalLinalgRewriter`, run at the start of `MetalCompiler::RunHloPasses`,
  row-major operands) and `metal$lapack_{getrf,geqrf,orgqr,syevd,gesdd,
  gesdd_novec}` (targets of `jax_plugins/metal/linalg_lowerings.py`,
  column-major operands via layout constraints). Each handler calls
  `rt::Stream::Synchronize()` and then runs Accelerate LAPACK/BLAS directly on
  the shared-storage buffers (zero copy), synchronously on the thunk thread.
  f32 only. `METAL_PJRT_DISABLE_LAPACK=1` (or `METAL_PJRT_DISABLE_REWRITES=
  lapack` for the HLO pass alone) falls back to XLA's expanders.

## PJRT client

- `GetStreamExecutorGpuClient` builds `LocalDeviceState`s, allocators
  (`kPlatform` is the simplest that works; `kBFC` needs the two
  `CreateMemoryAllocator` kinds), and calls `GpuCollectives::Resolve(name)`
  which CHECK-fails without a registration (hence the stub under "metal").
- `//xla/service:gpu_plugin` is empty on macOS (all deps behind
  `if_gpu_is_configured`), so our plugin target lists `gpu_compiler`,
  `gpu_executable`, the transfer manager and thunk runtime deps explicitly.
- Plugin dylib: copy the CPU plugin's macOS linkopts
  (`-Wl,-exported_symbol,_GetPjrtApi`, `-install_name @rpath/...`).
- Client must always pass `platform_name="metal"`; the topology path defaults
  to "gpu" which canonicalizes to "cuda".
- `MakeComputeCapabilityAttributeString` and `GpuPlatformVersionFromDevices`
  return "unknown" for our device; harmless.
