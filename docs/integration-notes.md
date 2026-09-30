# Integration notes (XLA @ 91888df6, jaxlib 0.11.2)

For anyone moving the XLA pin (`third_party/PINS.md`) or touching the
seams between the plugin and XLA or JAX.

Source-verified facts that drive the implementation. Paths are relative to the
XLA tree (`external/xla+` in the Bazel output base).

The load-bearing sites (OneAPI branches, the copied `AddLoweringPasses`
prefix, assumptions of the plugin's HLO passes) are snapshotted by
`//metal_pjrt/xla_tripwire:xla_tripwire_test`, which fails with a diff
when a pin bump changes one; `metal_pjrt/xla_tripwire/oneapi_callsites.py`
lists every line XLA-wide that branches on the OneAPI capability, the SYCL
platform id or a SPIR target against a golden list. Both have an `--update` mode.

## Identity

- StreamExecutor platform: `PLATFORM_DEFINE_ID(kMetalPlatformId, METAL)`;
  `Platform::Name()` is "METAL", canonical name "metal"
  (`xla/service/platform_util.cc` lowercases unknown names).
- PJRT platform (the JAX platform): name "mtl" (`MetalName()`), id
  `tsl::Fingerprint64("mtl")`. Not "metal": that is Apple's
  closed-source jax-metal plugin; the names do not collide (the two have
  not been tried together).
  The StreamExecutor, FFI and XLA-internal names stay "METAL"/"metal": they
  live in registries private to our dylib (it exports only `GetPjrtApi`,
  the callback trampoline and three testing hooks, `metal_pjrt_memory_stats`,
  `metal_pjrt_sync_stats` and `metal_pjrt_testing_memory_pressure`, used by
  `tests/test_memory.py` and `tests/test_transfers.py` via ctypes; the hooks are unstable, test-only and not an API), so they cannot
  collide with another plugin.
  JAX looks lowerings up by `backend.platform`, i.e. `MetalName()`, so
  `register_plugin`'s name and every `PLATFORM` in `metal_pjrt_plugin`
  must be exactly "mtl".
- Things keyed on identity that need registration from our code:
  `Compiler::RegisterCompilerFactory(kMetalPlatformId, ...)`,
  `TransferManager` for `kMetalPlatformId`,
  `PjRtRegisterDefaultCompiler(MetalName(), StreamExecutorGpuCompiler(MetalId, kMetalPlatformId))`,
  `StreamExecutorPlatformIdMapping::Global().AddMapping(kMetalPlatformId, MetalId)`,
  `XLA_COLLECTIVES_REGISTER(MetalName() and "METAL", "stub", 1, GpuCollectivesStub)`
  (the client resolves collectives by the PJRT name, collective thunks by the
  SE name).
- Things keyed on identity that need changes to XLA. Items 1-3 are patch
  `0001-metal-pjrt-identity.patch` in `third_party/xla/patches/` (which
  also names the platform in the GPU C API shim, returns instead of
  CHECK-failing in `gpu_module_globals.cc`, and
  canonicalizes "gpu" to "metal" in `platform_util.cc`), item 4 is a stub in
  our tree, item 5 is patch 0002. Patch 0003 is unrelated: a macOS build
  fix in `record_ffi.cc` (`size_t` vs `uint64_t`).
  1. `xla/pjrt/gpu/se_gpu_pjrt_client.cc:1874-1880` picks the PJRT platform
     name by macro; add `TENSORFLOW_USE_METAL` -> `MetalName()`.
  2. `xla/pjrt/pjrt_compiler.h:493` `IsGpuId` must include the metal id.
  3. `xla/service/gpu/gpu_executable.cc:534-551` platform-id switch must
     accept `kMetalPlatformId`.
  4. `xla/service/gpu/BUILD` `ptx_custom_kernel_emitter` has no branch when no
     GPU is configured: `metal_pjrt/compiler/ptx_custom_kernel_emitter_stub.cc`
     provides a stub `EmitPtxCustomKernelThunk` (no patch).
  5. `xla/service/gpu/gpu_compiler.{h,cc}`: `CompileToBackendResult` (private,
     non-virtual) stack-allocates a `CubinCustomKernelCompiler` (`final`) and
     passes it to `CompileModuleToLlvmIr` -> `IrEmitterContext`. Patch 0002
     moves that construction into a protected virtual
     `GpuCompiler::CreateKernelCompiler(LlvmIrCompiler, DeviceDescription,
     DebugOptions, ThreadPool*)` whose default is unchanged, so `MetalCompiler`
     can return a `MetalKernelCompiler` (see Codegen).
- Compute capability: `GpuComputeCapability` is a closed variant of
  Cuda/Rocm/OneAPI (`xla/stream_executor/device_description.h:98-178`).
  Default-constructed reads as CUDA 0.0. The plugin reports **OneAPI** so that the
  SPIR-V/Intel branches are taken (scalar-only transpose, explicit NaN
  propagation, command buffers off, atomics via the SPIRV path; the
  executable ABI-version extension is dropped instead, see "PJRT client"). No Apple alternative is planned: the tripwire test
  checks the OneAPI branches instead.

## StreamExecutor contract (what MetalExecutor must implement)

- Base: `gpu::GpuExecutor(Platform*, int ordinal)` (adds `device_ordinal()`).
- Pure: `Init`, `CreateStream(priority)`, `CreateEvent`, `Allocate(size,
  memory_space)`, `Deallocate`, `HostMemoryAllocate`, `SynchronizeAllActivity`,
  `SynchronousMemcpy` x2 (Unimplemented: only multi-device thunks call
  it), `DeallocateStream`, `EnablePeerAccessTo`, `CanEnablePeerAccessTo`,
  `CreateDeviceDescription`.
- Required by the PJRT client: `DeviceMemoryUsage` returns true;
  `CreateMemoryAllocator(kCollective|kHost)`; `LoadKernel`, `LoadModule`,
  `GetSymbol` (constants). `UnloadKernel` and `CreateOrShareConstant` keep
  the base defaults (a no-op; Unimplemented, which only the CUDA/ROCm/SYCL
  executors override and nothing on our path calls).
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
  (no-op), `OptimizeHloConvolutionCanonicalization` (MetalConvRewriter:
  convolutions to `metal$conv`, the rest stay loop-emitted; then the
  expanders below),
  `CompileTargetBinary(module_config, llvm::Module*, device_description,
  relocatable, debug_module, shard)`.
- `AddConfigAssignerPass`: no-op. `OptimizeHloPostLayoutAssignment`:
  the base pipeline (GemmRewriter), then
  `MetalDotOperandUpcaster` on the dots left for the loop emitter and
  `CheckPostGemmRewriter` (`compiler/passes/hlo_checks.h`): no narrow-operand
  kDot, every `__cublas$lt$matmul` passes `ValidateMatmul`
  (`blas/blas_lt_support.h`: types, epilogue, layouts and the index limits
  of the kernel that will run it), which `MetalBlasLt::GetMatmulPlan` runs
  again as a backstop. A violation is a compile error naming the JAX op and line.
  (A TopK custom call is not checked since 8976842: TopK is decomposed to
  a sort for OneAPI, and one that survived fails at thunk emission, having
  no Metal handler.)
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
- `ApplyMetalDefaults` forces `xla_gpu_dot_merger_threshold_mb=0` (as it
  forces `xla_gpu_enable_triton_gemm=false`; `XLA_FLAGS` cannot override
  either): DotMerger turns dots sharing an input into one dot against a
  concatenation of the others, which copies every weight matrix on each
  call (336 MB per call in the case study's 28-GEMM bench).

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
  (`metal_pjrt/codegen/metal_kernel_compiler.{h,cc}`) overrides it and
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
  `metal_pjrt/ffi` register with
  `XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), name, "METAL", h)` in
  an `alwayslink` library; nothing is needed from Python
  (`jax.ffi.ffi_call(name, ...)` just emits the custom call). Handlers
  compiled into another library (e.g. jaxlib's CUDA ones) are not visible.
- Stream: bind `.Ctx<xla::ffi::Stream>()` (`xla/backends/gpu/ffi.h`) to get
  the `se::Stream*`. `MetalStream::platform_specific_handle().stream` is the
  `metal_pjrt::rt::Stream*` (so `PlatformStream<rt::Stream*>` also works) and
  `stream->parent()` is the `MetalExecutor` owning the `rt::Device`.
  `metal_pjrt::ffi::GetMetalContext` wraps this; the handlers pass its
  device and stream to XLA-free dispatch code (`ffi/scan.h`,
  `ffi/radix_sort.h`, `linalg/small_linalg.h`, which launch through
  `rt::LaunchKernel`, `runtime/kernel_launch.h`), with kernels from the
  device's cache keyed by (MSL source, function, constants). `metal$scan`
  gets its pipeline once per call site in an FFI instantiate handler
  (state held by the thunk; no stream there, so it uses executor 0's device,
  `DefaultMetalDevice`) and only encodes the dispatch per execution; a
  stream of another device looks the kernel up again. The state is not
  serializable, so a deserialized executable re-runs instantiate (checked:
  persistent-cache hit, same result).
- The backend config must be an MLIR dictionary (JAX writes it raw; our
  rewriters put it in `GpuBackendConfig.custom_call_backend_config.attributes`).
- Handlers: `metal$scan` (target of MetalScanRewriter; `tests/test_scan.py`
  also calls it through `jax.ffi.ffi_call`). The kernels behind the
  handlers are tested without XLA (`ffi:scan_test`, `ffi:radix_sort_test`,
  `linalg:small_linalg_test`, `linalg:lapack_host_test`); the handlers
  themselves end to end through JAX (`tests/test_scan.py`, `test_sort.py`,
  `test_linalg.py`).
  `METAL_PJRT_DISABLE_REWRITES=scan|all` turns the scan rewriter off.
- XLA's SortRewriter targets `xla.gpu.ext.cub_sort_keys` / `cub_sort_pairs`
  (`ffi/cub_sort_ffi.cc` over `ffi/radix_sort.h`, mirroring
  `cub_sort_kernel_cuda.cc`): the
  instantiate stage returns the scratch size as `int64_t` state, which
  `EstimateCubSortScratchSize` reads at compile time (it calls the handler
  with null buffers and a zero-sized scratch); execute recomputes the layout
  and refuses a smaller scratch before encoding. Default layouts are forced
  for these calls, so rows are contiguous. SortRewriter's non-CUDA rule is
  total elements > 16384 regardless of row length; `RunHloPasses` expands
  rows <= 64 of such sorts with `MetalSortExpander` first.
  `METAL_PJRT_DISABLE_REWRITES=cubsort` turns SortRewriter off.
- `metal$conv` (`ffi/conv_ffi.cc` over `conv/conv.h`, MLX's steel
  convolutions): the target of MetalConvRewriter
  (`compiler/passes/conv_rewriter.h`), which runs in
  `OptimizeHloConvolutionCanonicalization`, before layout assignment, where
  CUDA canonicalizes for cuDNN. XLA's own ConvRewriter is not usable: it
  rewrites every convolution, and thunk_emitter sends cuDNN targets to a
  ConvolutionThunk (a DnnSupport) before any FFI lookup. Each convolution
  becomes transposes to NHWC / OHWI (none for NHWC activations; weights are
  small), the custom call with a u8 workspace tuple element sized by
  PlanConv at rewrite time (re-planned and checked by the handler), and a
  transpose back. Two kinds: "fwd" (the forward convolution and JAX's input
  gradient, whose kReverse of the kernel folds into `flip` when the
  convolution is its only user) and "wgrad" (JAX's weight gradient: lhs
  feature dimension before its batch dimension, rhs batch before feature;
  computed as patches x dY). A window reversed on every spatial dim (XLA's
  algebraic simplifier swaps the operands of a convolution whose kernel is
  larger than its input) toggles `flip` or reverses the rhs. Left to the loop
  emitter: grouped, 3-D, negative low padding, types other than
  f32/f16/bf16, and convolutions under 4 Mflop (the loop emitter's single
  fused kernel is faster there, docs/performance.md).
  `METAL_PJRT_DISABLE_REWRITES=conv` turns the rewriter off.
- `metal$fft` (`ffi/fft_ffi.cc` over `fft/fft.h`, MLX's FFT kernels): the
  target of the `fft` lowering in `metal_pjrt_plugin/_lowerings.py`, one
  1-D transform per axis on complex64 rows (float32 on the real side of
  rfft / irfft; other axes moved last by a transpose), with a u8 workspace
  result that the Python rule sizes as `FftWorkspaceBytes` does (the
  handler refuses any other size, so the two cannot drift silently).
  XLA's FftThunk is cuFFT-only, so `CheckPostGemmRewriter` refuses the
  HLO `fft` op instead of letting it reach the thunk emitter (after
  optimization, so that the dead cpu branch of a module exported for
  ("cpu", "mtl") does not count). The plan and
  the Rader / Bluestein constants are made at instantiation and shared by
  every call site of a length while one uses them (up to ~0.6 GB of
  transient host work and 160 MB on the device near n = 2^23). Lengths the
  kernels do not cover (powers of two above 2^24, other n above 2^23 - 1)
  raise NotImplementedError; complex128 and a symbolic batch (whose
  workspace size is unknown at lowering) take the dense DFT. `METAL_PJRT_DISABLE_FFT=1` sends
  every axis to the DFT.
  (A `metal$softmax` rewriter was removed after an end-to-end A/B,
  docs/performance.md, "Measured and rejected".)
- Dense linear algebra (`metal_pjrt/linalg/`): handlers
  `metal$cholesky`, `metal$triangular_solve` (targets of
  `MetalLinalgRewriter`, run at the start of `MetalCompiler::RunHloPasses`,
  row-major operands) and `metal$lapack_{getrf,geqrf,orgqr,syevd,gesdd,
  gesdd_novec}` (targets of `metal_pjrt_plugin/_linalg_lowerings.py`,
  column-major operands via layout constraints). Matrices up to 32x32 in
  `metal$cholesky`, `metal$triangular_solve` and `metal$lapack_getrf` run as
  GPU kernels on the stream, with no synchronization. Above that, each
  handler calls
  `rt::Stream::Synchronize()` and then runs Accelerate LAPACK/BLAS directly on
  the shared-storage buffers (zero copy), synchronously on the thunk thread
  (a stream host task was measured and not faster: docs/performance.md,
  "Measured and rejected").
  f32 only. `METAL_PJRT_DISABLE_LAPACK=1` is the one switch for both
  (read once per process by the pass and by the Python rules) and
  falls back to XLA's expanders / JAX's generic lowerings. Ownership table:
  the `_linalg_lowerings.py` docstring.

## PJRT client

- `GetStreamExecutorGpuClient` builds `LocalDeviceState`s, allocators
  (the plugin always asks for `kPlatform`, a pass-through to
  `MetalExecutor::Allocate` and the runtime's buffer cache; `kBFC` would
  need the two `CreateMemoryAllocator` kinds), and calls `GpuCollectives::Resolve(name)`
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
  the plugin image), fingerprint(compile settings)}`: a rebuilt plugin or a
  different compile-time setting gets a different key. The settings
  (`compiler/compile_settings.h`: METAL_PJRT_DISABLE_LAPACK,
  METAL_PJRT_DISABLE_REWRITES and METAL_PJRT_DISABLE_FFT) are read once and the key hashes their parsed
  values, the ones the passes use: unset and `DISABLE_LAPACK=0` share a
  key, and changing the environment after the first compile changes
  neither.
- Persistent compilation cache. Two things kept JAX 0.11.2 from using it
  for "mtl", both fixed without an XLA patch:
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
  local list (tpu/gpu/cpu/neuron). `metal_pjrt_plugin` wraps it and, for
  mtl backends only, calls the original with a proxy whose `platform` is
  "gpu" (all else forwarded), so upstream's one-shot bookkeeping still
  runs. `tests/test_compilation_cache.py` asserts the list is still there.
  The plugin never sets `jax_compilation_cache_dir` (a maintainer
  decision, `docs/roadmap.md`): the setting is process-wide and `initialize()` runs for every installed
  plugin whatever `JAX_PLATFORMS` says, so a default directory turned the
  cache on for CPU-only users too. Users configure it
  (`JAX_COMPILATION_CACHE_DIR`, `jax.config`; README, "Compilation cache");
  `tests/test_compilation_cache.py` checks the plugin leaves it unset.
  JAX's own thresholds apply (only compiles over
  `jax_persistent_cache_min_compile_time_secs`, 1 s by default, are
  written). Executables with metal host callbacks bypass it
  (`docs/callbacks.md`).

## Host transfers

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
  `metal_pjrt_api.cc` wraps `PJRT_Client_BufferFromHostBuffer`. jaxlib
  passes `kImmutableZeroCopy` with explicit byte strides even for a
  C-contiguous array; strides equal to the row-major ones count as dense.
  Dense data is copied into a malloc'd buffer before returning (as CUDA
  does for pageable memory) and freed on `done_with_host_buffer`. From
  min(256 MB, max(16 MB, reclaimable / 8)) up (the system is asked only
  from 16 MB up; `METAL_PJRT_SNAPSHOT_MAX_MB` replaces
  the 256 MB, for tests) the caller's pointer goes to XLA as
  `kImmutableUntilTransferCompletes` and the wrapper awaits
  `done_with_host_buffer` (without the GIL: jaxlib releases it around the
  call) before returning; the event is still handed back. Strided and
  sub-byte inputs take XLA's synchronous `kImmutableOnlyDuringCall` path,
  which linearizes into a full-size host staging buffer. Until the stride
  fix every numpy array took that path (the malloc branch was dead code),
  which is where the old 2x host memory came from.
  A first version waited from 16 MB up; with ~100 ms of GPU work queued a
  32 MB put then returned after 101 ms instead of 1.8 ms (the H2D stream
  waits for XLA's allocation event on the compute stream), hence the
  adaptive threshold. Measured, staged path / waiting from 16 MB / now (6
  interleaved rounds, p10/median/p90): 1 MB 106/110/118, 104/110/120,
  106/111/119 us; 16 MB 812/825/895, 405/423/475, 811/828/881 us; 100 MB
  4974/5058/5277, 2481/2556/2746, 4994/5114/5501 us; 512 MB
  25.7/27.8/129.6, 12.9/13.3/26.0, 12.9/13.3/17.0 ms. Peak phys_footprint
  of a 512 MB put +1041 / +512 / +512 MB. Behind ~100 ms of queued GPU
  work, a put returns (median) at 32 MB after 1.8 / 101 / 1.9 ms and at
  512 MB after 153 / 132 / 134 ms.
- `Stream::MemcpyHostToDevice/DeviceToHost` `memcpy` on the calling thread
  when the stream is idle (no open command buffer, deferred waits all
  satisfied, fence caught up); otherwise they enqueue a host task.
- Only `pinned_host` memory kinds use XLA's host BFC pool (never
  shrinks).

