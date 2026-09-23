#!/bin/bash
# Regenerates third_party/xla/patches/0001-metal-pjrt-identity.patch against a
# pristine XLA tree (argument 1). The edits teach XLA's GPU PJRT client, C API
# shim and platform utilities about a "metal" platform on macOS.
set -euo pipefail
XLA=${1:?path to pristine xla tree}
OUT=$(cd "$(dirname "$0")" && pwd)/patches/0001-metal-pjrt-identity.patch
WORK=$(mktemp -d)
FILES=(
  xla/pjrt/pjrt_compiler.h
  xla/pjrt/gpu/se_gpu_pjrt_client.cc
  xla/pjrt/gpu/BUILD
  xla/pjrt/c/pjrt_c_api_gpu_internal.cc
  xla/pjrt/c/BUILD
  xla/service/platform_util.cc
  xla/service/BUILD
  xla/service/gpu/gpu_executable.cc
)
for f in "${FILES[@]}"; do
  mkdir -p "$WORK/a/$(dirname $f)" "$WORK/b/$(dirname $f)"
  cp "$XLA/$f" "$WORK/a/$f"; cp "$XLA/$f" "$WORK/b/$f"
done
B=$WORK/b

# 1. PJRT identity helpers and IsGpuId.
perl -0pi -e 's|(inline PjRtPlatformId SyclId\(\) \{ return OneapiId\(\); \}\n)|$1\n// Apple Metal (metal-pjrt-plugin).\ninline const char* MetalName() { return "metal"; }\ninline PjRtPlatformId MetalId() {\n  static const PjRtPlatformId kMetalId = tsl::Fingerprint64(MetalName());\n  return kMetalId;\n}\n|' $B/xla/pjrt/pjrt_compiler.h
perl -0pi -e 's|(platform_id == xla::SyclId\(\));|$1 \|\|\n         platform_id == xla::MetalId();|' $B/xla/pjrt/pjrt_compiler.h

# 2. GPU client: platform name selection and GPU-runtime include guards.
perl -0pi -e 's|(#elif TENSORFLOW_USE_SYCL\n\s*auto (\w+) = ((?:xla::)?)(?:OneapiName\|SyclName)\(\);\n)|$1#elif TENSORFLOW_USE_METAL\n  auto $2 = $3MetalName();\n|g' $B/xla/pjrt/gpu/se_gpu_pjrt_client.cc
perl -0pi -e 's|defined\(TENSORFLOW_USE_SYCL\)|defined(TENSORFLOW_USE_SYCL) \|\| defined(TENSORFLOW_USE_METAL)|g' $B/xla/pjrt/gpu/se_gpu_pjrt_client.cc
perl -0pi -e 's|load\("//xla/tsl:tsl.bzl", "if_google", "internal_visibility"\)|load("//xla/tsl:tsl.bzl", "if_google", "if_macos", "internal_visibility")|' $B/xla/pjrt/gpu/BUILD
perl -0pi -e 's|(    defines = if_cuda\(\["GOOGLE_CUDA=1"\]\) \+ if_rocm\(\["TENSORFLOW_USE_ROCM=1"\]\) \+ if_sycl\(\[\n        "TENSORFLOW_USE_SYCL=1",\n    \]\))|$1 + if_macos(["TENSORFLOW_USE_METAL=1"])|' $B/xla/pjrt/gpu/BUILD
perl -0pi -e 's|(        "\@local_config_sycl//sycl:sycl_headers",\n    \]\))|$1 + if_macos([\n        # keep sorted\n        "//xla/backends/gpu/runtime:thunk_executor",\n        "//xla/service/gpu:gpu_compiler",\n        "//xla/service/gpu:gpu_executable",\n        "//xla/service/gpu:gpu_executable_buffer_allocator",\n        "//xla/service/gpu:stream_executor_util",\n    ])|' $B/xla/pjrt/gpu/BUILD

# 3. C API shim: platform macro and topology id/name.
perl -0pi -e 's|(#elif TENSORFLOW_USE_SYCL\n#define PJRT_GPU_PLUGIN_PLATFORM_NAME "ONEAPI"\n)|$1#elif TENSORFLOW_USE_METAL\n#define PJRT_GPU_PLUGIN_PLATFORM_NAME "METAL"\n|' $B/xla/pjrt/c/pjrt_c_api_gpu_internal.cc
perl -0pi -e 's|(    platform_name = xla::OneapiName\(\);\n  \})|$1 else if (plugin_platform == "METAL") {\n    platform_id = xla::MetalId();\n    platform_name = xla::MetalName();\n  }|' $B/xla/pjrt/c/pjrt_c_api_gpu_internal.cc
perl -0pi -e 's|(        \]\) \+ if_sycl_is_configured\(\["TENSORFLOW_USE_SYCL=1"\]\)\n    \))|        ]) + if_sycl_is_configured(["TENSORFLOW_USE_SYCL=1"]) +\n        if_macos(["TENSORFLOW_USE_METAL=1"])\n    )|' $B/xla/pjrt/c/BUILD
perl -0pi -e 's|(load\("//xla/tsl/platform:rules_cc.bzl", "cc_library"\)\n)|$1load("//xla/tsl:tsl.bzl", "if_macos")\n|' $B/xla/pjrt/c/BUILD

# 4. platform_util: "gpu" canonicalizes to "metal" on macOS.
perl -0pi -e 's|(#elif TENSORFLOW_USE_SYCL\n    return "sycl";\n)|$1#elif TENSORFLOW_USE_METAL\n    return "metal";\n|' $B/xla/service/platform_util.cc
perl -0pi -e 's|(    local_defines = if_rocm\(\["TENSORFLOW_USE_ROCM=1"\]\) \+ if_sycl\(\[\n        "TENSORFLOW_USE_SYCL=1",\n    \]\))|$1 + if_macos(["TENSORFLOW_USE_METAL=1"])|' $B/xla/service/BUILD
grep -q 'load("//xla/tsl:tsl.bzl", "if_macos")' $B/xla/service/BUILD || \
  perl -0pi -e 's|(load\("\@bazel_skylib//rules:build_test.bzl", "build_test"\)\n)|$1load("//xla/tsl:tsl.bzl", "if_macos")\n|' $B/xla/service/BUILD

# 5. GpuExecutable: tolerate platforms without a compute-capability check.
perl -0pi -e 's|  \} else \{\n    return Internal\("Unknown platform"\);\n  \}|  } else {\n    VLOG(2) << "No compute capability check for platform "\n            << main_stream->parent()->GetPlatform()->Name();\n  }|' $B/xla/service/gpu/gpu_executable.cc

cd $WORK && (diff -ruN a b > "$OUT" || true)
# Sanity: every file must have changed.
for f in "${FILES[@]}"; do cmp -s "a/$f" "b/$f" && echo "WARNING: no change in $f"; done
echo "wrote $OUT ($(grep -c '^@@' "$OUT") hunks)"
rm -rf "$WORK"
