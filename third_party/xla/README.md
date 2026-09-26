# XLA patches

Applied to the pinned XLA archive by `MODULE.bazel` (`patch_strip = 1`).

| Patch | Purpose | Regenerate with |
|---|---|---|
| `0001-metal-pjrt-identity.patch` | `MetalName()`/`MetalId()`, `IsGpuId`, "metal" platform selection in the GPU PJRT client and C API shim on macOS, `gpu` -> `metal` canonical name, GpuExecutable platform check | `make_identity_patch.sh <pristine xla tree>` |
| `0002-gpu-compiler-kernel-compiler-factory.patch` | Makes the `KernelCompiler` built in `GpuCompiler::CompileToBackendResult` come from a protected virtual `GpuCompiler::CreateKernelCompiler` (default: the same `CubinCustomKernelCompiler`), so `MetalCompiler` can supply `MetalKernelCompiler`, whose `CompileMlirToLlvm` emits MSL. Upstream hard-codes a stack-allocated `final` `CubinCustomKernelCompiler` in a private non-virtual method, so there is no seam without it | `make_kernel_compiler_factory_patch.sh <pristine xla tree>` |
| `0003-macos-build-fixes.patch` | Portability fixes (`size_t` vs `uint64_t` override mismatch in `record_ffi.cc`) | `make_macos_fixes_patch.sh <pristine xla tree>` |
| `0004-bfc-garbage-collection.patch` | Enable BFC garbage collection so the pool shrinks when growth is refused (unified memory) | `make_bfc_gc_patch.sh <pristine xla tree>` |
| `0005-command-buffers-on-metal.patch` | The command buffer conversion pass clears every command type for OneAPI devices; exempt device vendor "Apple" (the Metal platform reports OneAPI) so `xla_gpu_enable_command_buffer` decides, as on CUDA | `make_command_buffer_metal_patch.sh <pristine xla tree>` |

Everything else Metal-specific is registered from this repo's own code (platform,
compiler, transfer manager, collectives stub, PJRT compiler) without patching.
