# Integration notes (XLA @ 91888df6, jaxlib 0.11.2)

For anyone moving the XLA pin (`third_party/PINS.md`) or touching the
seams between the plugin and XLA or JAX. Everything here was checked
against the source; paths are relative to the XLA tree (`external/xla+`
in the Bazel output base).

The load-bearing sites (OneAPI branches, the copied `AddLoweringPasses`
prefix, assumptions of the plugin's HLO passes) are snapshotted by
`//metal_pjrt/xla_tripwire:xla_tripwire_test`, which fails with a diff
when a pin bump changes one. `metal_pjrt/xla_tripwire/oneapi_callsites.py`
checks every line in XLA that branches on OneAPI, the SYCL platform id or
a SPIR target against a golden list. Both have an `--update` mode.

## Identity

- **StreamExecutor platform:** `kMetalPlatformId`, named "METAL"
  (canonically "metal").
- **PJRT platform** (what JAX sees): "mtl" (`MetalName()`), id
  `tsl::Fingerprint64("mtl")`. "metal" belongs to Apple's jax-metal.
  JAX looks lowerings up by `backend.platform`, so `register_plugin` and
  every `PLATFORM` in `metal_pjrt_plugin` must say exactly "mtl". The
  StreamExecutor, FFI and XLA-internal names stay "METAL" in registries
  private to our dylib, which exports only `GetPjrtApi`, the callback
  trampoline and two unstable test hooks (`metal_pjrt_memory_stats`,
  `metal_pjrt_testing_memory_pressure`).
- **Registrations in our code:** the compiler factory and
  `TransferManager` for `kMetalPlatformId`, `PjRtRegisterDefaultCompiler`
  under `MetalName()`, the platform-id mapping, and a stub collectives
  registration under both "mtl" and "METAL" (the client resolves
  collectives by the PJRT name, collective thunks by the SE name).
- **Patches to XLA** (`third_party/xla/patches/`):
  - **0001** (identity): `se_gpu_pjrt_client.cc` picks the PJRT platform
    name by macro (adds `TENSORFLOW_USE_METAL`); `IsGpuId`
    (`pjrt_compiler.h`) and the platform-id switch in
    `gpu_executable.cc` accept the metal id; the GPU C API shim names the
    platform; `gpu_module_globals.cc` returns instead of CHECK-failing;
    `platform_util.cc` canonicalizes "gpu" to "metal".
  - **0002** (codegen): `CompileToBackendResult` stack-allocated a
    `final` `CubinCustomKernelCompiler`; the patch moves that into a
    protected virtual `GpuCompiler::CreateKernelCompiler` with an
    unchanged default, so `MetalCompiler` can return a
    `MetalKernelCompiler`.
  - **0003**: a macOS build fix in `record_ffi.cc`.
  - **0004** (codegen): `GetFusionEmitter` is a closed switch over fusion
    kinds, and its `kCustomFusion` case only returns Unimplemented since
    the custom-kernel-fusion registry was removed. The patch adds
    `SetCustomFusionEmitterFactory`, asked first for those fusions, so
    `MetalCompiler` can emit its row-normalization (`metal_rownorm`)
    fusions.
  - No patch: `ptx_custom_kernel_emitter` has no branch without a
    configured GPU, so our tree provides a stub
    (`metal_pjrt/compiler/ptx_custom_kernel_emitter_stub.cc`).
- **Compute capability.** `GpuComputeCapability` is a closed
  Cuda/Rocm/OneAPI variant, and default-constructed it reads as CUDA 0.0.
  The plugin reports **OneAPI**, so the SPIR-V/Intel branches are taken
  (scalar-only transpose, explicit NaN propagation, command buffers off,
  atomics via the SPIR-V path). The tripwire test watches those branches.

## StreamExecutor contract

- Base: `gpu::GpuExecutor(Platform*, int ordinal)`.
- Pure virtuals: `Init`, `CreateStream`, `CreateEvent`, `Allocate`,
  `Deallocate`, `HostMemoryAllocate`, `SynchronizeAllActivity`,
  `SynchronousMemcpy` (Unimplemented; only multi-device thunks call it),
  `DeallocateStream`, peer access, `CreateDeviceDescription`.
- The PJRT client also needs `DeviceMemoryUsage`,
  `CreateMemoryAllocator(kCollective|kHost)`, `LoadKernel`, `LoadModule`
  and `GetSymbol` (constants). `UnloadKernel` and `CreateOrShareConstant`
  keep the base defaults.
- Memory spaces and XLA colors all map to the same shared `MTLBuffer`
  allocation (unified memory).
- Streams implement waits, `RecordEvent`, copies, memsets,
  `BlockHostUntilDone` and `DoHostCallbackWithStatus`. `LocalDeviceState`
  creates ~18 streams per device; requested priorities are ignored.
- Kernels get one 8-byte device pointer per argument in HLO buffer order
  (`KernelArgsPackedArrayBase`), no scalars.
- **Binary format:** MSL source travels in
  `KernelLoaderSpec::CreateOwningCudaCubinInMemorySpec`, as SYCL smuggles
  SPIR-V. `LoadKernel` compiles it and looks up `spec.kernel_name()`.
- **31-argument limit.** Above 31 buffers the emitter marks the kernel
  `// xla_metal_argbuffer` and takes one `constant ulong*` table of GPU
  addresses; `Stream::Launch` packs it with `setBytes` (up to 512
  arguments) and calls `useResources` on the buffers.

## Compiler contract (MetalCompiler : GpuCompiler)

Modeled on `xla/service/gpu/intel_gpu_compiler.{h,cc}`.

- Constructed with target `"spirv64-unknown-unknown"`, so emitters take
  the SPIR branches.
- `OptimizeHloConvolutionCanonicalization` runs MetalConvRewriter (see
  [`metal$conv`](#ffi-custom-calls)), then the expanders.
  `AddPaddingForGpublasGemms` and `AddConfigAssignerPass` are no-ops.
- `OptimizeHloPostLayoutAssignment`: the base pipeline always rewrites
  dots into `__cublas$lt$matmul` on OneAPI, so every matmul reaches
  `MetalBlasLt` (MPS or steel). Then `MetalDotOperandUpcaster` handles
  dots left for the loop emitter, and `CheckPostGemmRewriter`
  (`compiler/passes/hlo_checks.h`) requires every matmul to pass
  `ValidateMatmul` (types, epilogue, layouts, kernel index limits).
  `MetalBlasLt::GetMatmulPlan` checks again as a backstop. A violation is
  a compile error naming the JAX op and source line.
- Kernels compile one module at a time (`CompileSingleModule ->
  CompileTargetBinary`) into `CustomKernelThunk`s. `GpuExecutable::binary()`
  holds only the constants module.
- Registration is a static initializer in an `alwayslink` target.
- `ApplyMetalDefaults` forces Triton GEMMs off and
  `xla_gpu_dot_merger_threshold_mb=0`; `XLA_FLAGS` can't override either.
  DotMerger concatenates weight matrices that share an input, copying
  every weight on each call.

## Codegen: MLIR -> EmitC -> MSL

- **The seam.** MLIR emitters only call
  `kernel_compiler()->CompileMlirToLlvm(...)`, which is virtual.
  `MetalKernelCompiler` (`metal_pjrt/codegen/`) overrides it and delegates
  everything else to an inner `CubinCustomKernelCompiler`.
- **Pipeline.** It runs `AddLoopTransformationPasses` and
  `AddLoweringPasses` up to but not including SCFToControlFlow, converts
  func/scf/arith/math/vector/gpu and LowerTensors' LLVM-dialect memory ops
  to EmitC, and prints MSL. The kernel wrapper (buffer and thread-id
  attributes) is generated as text around the printed body.
- **Launch size.** XLA doesn't annotate launch dimensions for SPIR, so
  `MetalKernelCompiler` reads the threadgroup size from the MLIR and emits
  `[[max_total_threads_per_threadgroup(N)]]`. The runtime parses N back
  and refuses any larger launch before encoding: release Metal doesn't
  check, and a larger threadgroup is undefined behavior.
- **Transport.** The MSL rides to `CompileTargetBinary` in a stub
  `llvm::Module`, which returns it as the "cubin". The constants module is
  real LLVM IR; its initializers are serialized into a private container
  that `LoadModule`/`GetSymbol` read.

## FFI custom calls

- Typed-FFI custom calls become `CustomCallThunk`s, which find handlers
  by target and the SE platform name; "METAL" and "metal" are the same
  key. The registry is static inside our dylib's copy of XLA: handlers in
  `metal_pjrt/ffi` register with `XLA_FFI_REGISTER_HANDLER(..., "METAL",
  ...)` in `alwayslink` libraries, and handlers from other libraries
  (e.g. jaxlib's CUDA ones) aren't visible.
- `metal_pjrt::ffi::GetMetalContext` turns the bound
  `xla::ffi::Stream` into the `rt::Stream` and `rt::Device`, and handlers
  pass those to XLA-free dispatch code.
- Backend configs must be MLIR dictionaries.
- Instantiate-time state (e.g. `metal$scan`'s pipeline) isn't
  serializable, so a deserialized executable re-runs instantiate.

The handlers:

- **`metal$scan`**: minor-dimension cumulative ops, from
  MetalScanRewriter.
- **Sort**: XLA's SortRewriter targets `xla.gpu.ext.cub_sort_keys/pairs`,
  implemented over an MSL radix sort. `EstimateCubSortScratchSize` calls
  the handler with null buffers at compile time to size scratch; execute
  refuses a smaller one. Sorts with rows <= 64 are expanded first by
  `MetalSortExpander`, since SortRewriter's non-CUDA rule ignores row
  length.
- **`metal$conv`**: MLX's steel convolutions. XLA's own ConvRewriter
  can't be used, because the thunk emitter sends cuDNN targets to a
  DnnSupport before any FFI lookup. Each convolution becomes transposes
  to NHWC/OHWI, the call with a workspace sized at rewrite time (and
  rechecked by the handler), and a transpose back. Two kinds: "fwd"
  (forward and input gradient, with a reversed kernel folded into
  `flip`) and "wgrad" (weight gradient). Grouped and 3-D convolutions,
  negative padding, other types and anything under 4 Mflop stay with the
  loop emitter.
- **`metal$pool_max_bwd`**: JAX's max-pool gradient (a select-and-scatter
  with GE select and add scatter over non-overlapping, unpadded windows
  on at most two adjacent dims). One thread per window reads x and dy once
  and writes dx once, with the expander's selection rule, instead of the
  expander's atomic scatter-add.
- **`metal$fft`**: MLX's FFT kernels, from the `fft` lowering in
  `_lowerings.py`, one 1-D complex64 transform per axis. The Python rule
  sizes the workspace exactly as `FftWorkspaceBytes` does, and the handler
  refuses any other size, so the two can't drift. XLA's FftThunk is
  cuFFT-only, so `CheckPostGemmRewriter` refuses a surviving HLO `fft`.
  Plans are shared across call sites of a length. Lengths beyond the
  kernels (powers of two above 2^24, others above 2^23 - 1) raise
  NotImplementedError; complex128 and symbolic batches take the dense
  DFT.
- **Dense linear algebra** (`metal_pjrt/linalg/`): `metal$cholesky` and
  `metal$triangular_solve` from `MetalLinalgRewriter`, and
  `metal$lapack_*` (getrf, geqrf, orgqr, syevd, gesdd) from
  `_linalg_lowerings.py`. Up to 32x32, cholesky, triangular solve and LU
  run as GPU kernels; above that the handler synchronizes the stream and
  runs Accelerate directly on the shared buffers. f32 only. The ownership
  table is in the `_linalg_lowerings.py` docstring.

Switches for debugging: `METAL_PJRT_DISABLE_REWRITES=scan|cubsort|conv|pool|all`,
`METAL_PJRT_DISABLE_FFT=1` (dense DFT for every axis) and
`METAL_PJRT_DISABLE_LAPACK=1` (XLA's expanders and JAX's generic
lowerings). The kernels are tested without XLA (`ffi:*_test`,
`linalg:*_test`) and the handlers end to end through JAX.

## PJRT client

- The plugin always asks for the `kPlatform` allocator, a pass-through to
  `MetalExecutor::Allocate` and the runtime's cache.
  `GpuCollectives::Resolve` CHECK-fails without a registration, hence the
  stub.
- `//xla/service:gpu_plugin` is empty on macOS, so our target lists the
  compiler, executable, transfer manager and thunk runtime deps
  explicitly. The dylib uses the CPU plugin's macOS linkopts.
- The client option `platform_name` selects the StreamExecutor platform,
  so the plugin passes "METAL". Unset, it's "gpu", which patch 0001
  canonicalizes to "metal"; the topology path relies on that.
- **Platform version.** The client reports `"oneapi " + runtime_version`,
  and JAX hashes that into its compilation cache key. So `MetalExecutor`
  sets `runtime_version` from the plugin image's LC_UUID and the parsed
  compile settings (`compiler/compile_settings.h`: the three `DISABLE_*`
  switches, read once per process). A rebuilt plugin or a different
  setting gets a different key.

### Persistent compilation cache

Two things kept JAX 0.11.2 from caching "mtl" executables, both fixed
without patching XLA:

1. **Serialization.** XLA's GPU C API shim adds a `PJRT_AbiVersion`
   extension reporting the OneAPI ABI, which jaxlib's IFRT rejects. Without
   the extension it falls back to a version-less executable version, as
   for CPU. So the plugin's own `GetPjrtApi` returns XLA's GPU API with
   that extension dropped; nothing in the plugin reads it.
   Deserialization checks only the client name, so the cache key's
   platform version is the guard against a foreign build. Executables
   (fusions, GEMMs, sort, loops, FFI calls, constants) round-trip
   bit-identically in a fresh process.
2. **`compilation_cache.is_cache_used`** accepts a fixed platform list.
   The plugin wraps it and, for mtl backends only, passes a proxy whose
   `platform` is "gpu". `tests/test_compilation_cache.py` asserts the
   list is still there.

The plugin never sets `jax_compilation_cache_dir`
([`roadmap.md`](roadmap.md)): the setting is process-wide and plugins
initialize whatever `JAX_PLATFORMS` says, so a default would turn the
cache on for CPU-only users. Users set it themselves (see the
[FAQ](faq.md#how-do-i-turn-on-the-compilation-cache)); JAX's 1 s minimum compile time applies.
Executables with host callbacks bypass the cache
([`callbacks.md`](callbacks.md)).

## Host transfers

- `should_stage_host_to_device_transfers` is False. It was moot anyway:
  the default `IsHostMemoryPinned` treats every pointer outside our
  buffers as pinned, so nothing was staged.
- JAX 0.11.2 lets XLA read a numpy array after `device_put` returns, so
  refilling it right away could change what the device got.
  `metal_pjrt_api.cc` wraps `PJRT_Client_BufferFromHostBuffer`:
  - Strides equal to row-major count as dense (jaxlib always passes
    explicit strides).
  - Dense data is copied into a malloc'd buffer before returning, as CUDA
    does for pageable memory.
  - From min(256 MB, max(16 MB, reclaimable memory / 8)) up, the caller's
    pointer goes to XLA as `kImmutableUntilTransferCompletes` and the
    wrapper waits (without the GIL) until XLA is done with it.
    `METAL_PJRT_SNAPSHOT_MAX_MB` replaces the 256 MB, for tests.
  - Strided and sub-byte inputs take XLA's synchronous path, which stages
    a full-size host copy.

  The threshold adapts because waiting means waiting behind queued GPU
  work: with a fixed 16 MB threshold, a 32 MB put behind ~100 ms of GPU
  work returned after 101 ms instead of 1.8 ms. Above it, waiting avoids
  a second host copy: a 512 MB put peaks at +512 MB instead of +1041 MB.
  Measured medians, XLA's staged path vs now: 16 MB 825 vs 828 us,
  100 MB 5.1 vs 5.1 ms, 512 MB 27.8 vs 13.3 ms; behind ~100 ms of GPU
  work, a 32 MB put returns after 1.8 vs 1.9 ms.
- `Stream::MemcpyHostToDevice/DeviceToHost` `memcpy` on the calling thread
  when the stream is idle, and enqueue a host task otherwise.
- Only `pinned_host` memory uses XLA's host BFC pool, which never shrinks.
