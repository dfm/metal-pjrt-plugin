# Design: Metal as a fourth XLA:GPU platform

## The production GPU plugin, and what changes for Metal

The JAX CUDA backend is four layers, and only the bottom two are CUDA-specific:

1. **C API shim** (`xla/pjrt/c/pjrt_c_api_gpu_internal.cc`): parses client
   options, picks the platform name from a compile-time macro (CUDA, ROCm or
   SYCL), and chains the PJRT extensions (FFI, custom call, profiler, stream,
   layouts, memory descriptions, cross-host transfers, shardings, ABI version).
2. **Client** (`xla/pjrt/gpu/se_gpu_pjrt_client.cc`): per-device
   `LocalDeviceState`, BFC allocator, host memory spaces, async dispatch. Only
   NCCL collectives and fabric handles are CUDA-specific.
3. **StreamExecutor platform** (`xla/stream_executor/<platform>/`): allocate,
   memcpy, streams, events, kernel launch, device description, plus the optional
   BLAS/DNN interfaces that cuBLAS and cuDNN plug into.
4. **Compiler subclass** of `GpuCompiler`: overrides post-layout-assignment
   passes, autotuning passes, target triple/data layout, and the final
   LLVM-IR-to-binary step. The base class owns the entire HLO pipeline.

Metal support is therefore:

- `stream_executor/metal/`: `MetalPlatform`, `MetalExecutor`, streams as
  `MTLCommandQueue`, events as `MTLSharedEvent`, shared-storage `MTLBuffer`s,
  kernel launch from an `MTLLibrary`.
- `MetalCompiler : GpuCompiler` with Triton, cuDNN passes, autotuning and
  collectives disabled.
- A `TENSORFLOW_USE_METAL`-style build of the existing shim.
- `jax_plugins/metal` cloned from `jax_plugins/cuda`.

Intel's extension for OpenXLA did exactly this for SYCL out of tree.

## Codegen

`GpuCompiler` bottoms out in LLVM IR. There is no LLVM Metal target. Options:

1. LLVM IR -> SPIR-V (LLVM's SPIRV backend, already enabled in XLA's LLVM
   config) -> SPIRV-Cross -> MSL source -> `newLibraryWithSource` at runtime.
   Every link is proven (Intel for the first, IREE for the second). Start here.
2. Intercept XLA's MLIR emitters before LLVM and emit MSL text directly
   (EmitC-style; MSL is C++-derived). Higher ceiling (simdgroup ops). Later.
3. Emit AIR bitcode. Undocumented. No.

Runtime shader compilation works with command-line tools only (verified on this
machine: 192 ms to compile a trivial kernel from source). Compiled libraries
need a persistent cache keyed by MSL hash, because 100 fused kernels at 200 ms
each is an unacceptable first-run cost.

## Library ops

`GpuCompiler` already rewrites dots into library custom calls. The
StreamExecutor BLAS and DNN interfaces are the plug points. First fill: Metal
Performance Shaders matrix and CNN kernels (not the graph API, which has a
reputation for bugs). Later: Metal 4's MetalPerformancePrimitives (in-shader
matmul2d / conv2d, headers present in the macOS 26 SDK) for fused epilogues,
or vendoring MLX's steel kernels (MIT) as a library.

## Deliberately off at first

Triton, cuDNN fusion passes, autotuning, collectives (stub for single device),
float64 (Metal has none; reject, consider double-float emulation later).

## Milestones

1. Build spike (`scripts/build_spike.sh`): does the GPU compiler build on macOS
   arm64 without CUDA, and at what cost. Go/no-go.
2. StreamExecutor Metal platform, validated with XLA's own stream_executor tests.
3. `MetalCompiler` with the target-binary hook doing LLVM IR -> SPIR-V -> MSL.
   Goal: a fused elementwise-plus-reduce program end to end.
4. Dot and conv through BLAS/DNN backed by MPS.
5. Python package, JAX test suite under `JAX_PLATFORMS=metal`, benchmarks vs
   jax-mps, MetalHLO and native MLX (ResNet18/CIFAR, nanoGPT).

## Risks

- Building XLA on an 8 GB laptop. Mitigated by low concurrency, disk cache,
  JAX's public remote cache, and staged overnight builds.
- CUDA coupling inside `xla/service/gpu` (e.g. `gpu_compiler` depends on
  `stream_executor/cuda:cuda_compute_capability` and the Triton emitters).
  Expect a patch set against XLA, pinned to the jaxlib release's commit.
- SPIR-V to MSL gaps: atomics, subgroup ops, 32 KB threadgroup memory, 32-wide
  SIMD, launch-dimension mapping.
