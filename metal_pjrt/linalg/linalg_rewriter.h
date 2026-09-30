// Copyright 2026 The metal-pjrt-plugin Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef METAL_PJRT_LINALG_LINALG_REWRITER_H_
#define METAL_PJRT_LINALG_LINALG_REWRITER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {
namespace gpu {

// Replaces f32 kCholesky and kTriangularSolve with the "metal$cholesky" /
// "metal$triangular_solve" FFI custom calls (metal_pjrt/linalg/
// lapack_ffi.cc), which run Accelerate's LAPACK/BLAS on the unified-memory
// buffers after synchronizing the stream. Must run before CholeskyExpander /
// TriangularSolveExpander (i.e. at the start of RunHloPasses); those would
// otherwise turn the ops into while loops of tiny kernels.
//
// The custom calls take and return default (row-major) layouts; the options
// become FFI attributes: lower (cholesky), left_side, lower, unit_diagonal,
// transpose_a (the TriangularSolveOptions::Transpose enum value).
//
// Disabled by METAL_PJRT_DISABLE_LAPACK=1 (see LapackDisabled()).
class MetalLinalgRewriter : public HloModulePass {
 public:
  absl::string_view name() const override { return "metal-linalg-rewriter"; }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

// True when METAL_PJRT_DISABLE_LAPACK is set to a non-empty value other than
// "0" (read once, compiler/compile_settings.h): linear algebra then goes
// through XLA's expanders (A/B comparisons).
bool LapackDisabled();

}  // namespace gpu
}  // namespace xla

#endif  // METAL_PJRT_LINALG_LINALG_REWRITER_H_
