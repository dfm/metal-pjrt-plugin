// Copyright 2026 The jax-graft Authors
// SPDX-License-Identifier: Apache-2.0

// Compile-time settings from the environment, read once per process. The
// passes and the persistent-cache key (MetalExecutor's PluginVersion) use
// the same parsed values, so an executable is always cached under the
// settings it was compiled with, and spellings that mean the same (unset,
// empty, "0", "false") give the same key.
#ifndef METAL_PJRT_COMPILER_COMPILE_SETTINGS_H_
#define METAL_PJRT_COMPILER_COMPILE_SETTINGS_H_

#include <string>

namespace metal_pjrt {

struct CompileSettings {
  // METAL_PJRT_DISABLE_LAPACK, a boolean (runtime/env.h), turns off the
  // Accelerate LAPACK rewriter (linalg/linalg_rewriter.cc); XLA's expanders
  // take linear algebra instead. jax_graft/_linalg_lowerings.py
  // reads it the same way (_env_flag).
  bool lapack = true;
  // METAL_PJRT_DISABLE_REWRITES, a comma-separated list: "scan" turns off
  // the metal$scan rewriter, "cubsort" XLA's SortRewriter (radix sort),
  // "conv" the metal$conv rewriter (convolutions stay on the loop emitter),
  // "pool" the metal$pool_max_bwd rewriter (max-pool gradients go to XLA's
  // SelectAndScatterExpander), and "all" all four
  // (compiler/metal_compiler.cc). Other names are ignored with a warning.
  bool scan_rewrite = true;
  bool cub_sort = true;
  bool conv_rewrite = true;
  bool pool_rewrite = true;
  // METAL_PJRT_DISABLE_FFT, a boolean: jax's fft lowers to the dense DFT
  // instead of metal$fft. The switch acts in jax_graft/_lowerings.py
  // (so it shows in the HLO); it is read here only for the cache key.
  bool fft = true;

  // The parsed values, for the cache key.
  std::string Fingerprint() const;
};

// Read from the environment on first use; changing it later has no effect.
const CompileSettings& GetCompileSettings();

}  // namespace metal_pjrt

#endif  // METAL_PJRT_COMPILER_COMPILE_SETTINGS_H_
