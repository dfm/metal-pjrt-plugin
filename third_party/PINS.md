# Version pins

| What | Value | Source |
|---|---|---|
| jaxlib | 0.11.2 | PyPI latest on 2026-09-23 |
| XLA | 91888df6ce85102c30220e41d952065925e10886 | jax-ml/jax `MODULE.bazel` at tag `jax-v0.11.2` |
| rules_ml_toolchain | c0eb2743b7b12b2bbcf0e1888e26d36ba6b093de | same |
| Bazel | 8.7.0 | jax `.bazelversion` |
| PJRT C API | 0.115 | `xla/pjrt/c/pjrt_c_api.h` at that commit |
| LLVM targets configured by XLA | AArch64, AMDGPU, ARM, NVPTX, PowerPC, RISCV, SystemZ, X86, SPIRV | `@xla//third_party/extensions:llvm.bzl` |

The plugin must be built against the exact XLA commit of the jaxlib release it
targets, as the CUDA plugin is. Re-pin by updating `MODULE.bazel` from JAX's
`MODULE.bazel` at the new release tag and refreshing the patches under
`third_party/`. Then regenerate the license notices of the statically linked
projects (`THIRD_PARTY_NOTICES`, from `third_party/notices_manifest.json`):
`scripts/gen_third_party_notices.py "$(bazel info output_base)/external" --update`
stops where a pinned project's license file moved or changed; update the
manifest's versions and paths there. It lists what the dylib contained at
the current pin; a dependency XLA adds has to be added by hand.
