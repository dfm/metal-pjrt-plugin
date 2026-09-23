# XLA patches

Applied to the pinned XLA archive by `MODULE.bazel` (`patch_strip = 1`).

| Patch | Purpose | Regenerate with |
|---|---|---|
| `0001-metal-pjrt-identity.patch` | `MetalName()`/`MetalId()`, `IsGpuId`, "metal" platform selection in the GPU PJRT client and C API shim on macOS, `gpu` -> `metal` canonical name, GpuExecutable platform check | `make_identity_patch.sh <pristine xla tree>` |
| `0002-mlir-kernel-emitter-hook.patch` | Hook in the MLIR kernel emitter so a Metal backend can produce MSL instead of LLVM IR | generated alongside `metal_pjrt_plugin/codegen` |
| `0003-macos-build-fixes.patch` | Portability fixes (`size_t` vs `uint64_t` override mismatch in `record_ffi.cc`) | `make_macos_fixes_patch.sh <pristine xla tree>` |

Everything else Metal-specific is registered from this repo's own code (platform,
compiler, transfer manager, collectives stub, PJRT compiler) without patching.
