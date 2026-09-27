// Plain C++ interface to a GEMM implemented with Metal Performance Shaders
// (MPSMatrixMultiplication). The implementation (mps_gemm_objc.cc) is
// Objective-C++; this header is free of Objective-C and metal-cpp so it can be
// included from ordinary C++ and XLA code. Metal objects cross the boundary as
// void* (they are the same pointers metal-cpp wraps).
#ifndef METAL_PJRT_PLUGIN_BLAS_MPS_GEMM_H_
#define METAL_PJRT_PLUGIN_BLAS_MPS_GEMM_H_

#include <cstdint>
#include <string>

#include "absl/status/status.h"

namespace metal_pjrt {
namespace blas {

enum class MpsDType { kF32, kF16, kBF16 };

int MpsDTypeSize(MpsDType t);
const char* MpsDTypeName(MpsDType t);

// One operand, stored ROW-MAJOR in an MTLBuffer. Strides are in elements.
struct MpsOperand {
  void* buffer = nullptr;       // id<MTLBuffer> / MTL::Buffer*
  uint64_t offset = 0;          // byte offset of batch 0, element (0, 0)
  int64_t ld = 0;               // elements between consecutive stored rows
  int64_t batch_stride = 0;     // elements between batches; 0 = broadcast
  MpsDType dtype = MpsDType::kF32;
  bool transpose = false;       // operand is used as op(X) = X^T
};

// Row-major GEMM, for each batch b in [0, batch_count):
//
//   C[b] (m x n) = alpha * op(A[b]) * op(B[b]) + beta * C[b]
//
// where op(A) is m x k and op(B) is k x n. bf16 operands are staged through
// f32 temporaries (MPSMatrixMultiplication asserts on bf16), as are operands
// whose buffer is too short for MPS's whole-matrix validation (last row
// padding missing when ld > cols); batch_stride 0 broadcasts an operand. A stored matrix is (m x k) when
// !a.transpose and (k x m) when a.transpose (likewise for B). C is never
// transposed and its dtype may differ from A/B (f16/bf16 in, f32 out).
struct GemmParams {
  int64_t m = 0, n = 0, k = 0;
  int64_t batch_count = 1;
  double alpha = 1.0, beta = 0.0;
  MpsOperand a, b, c;
};

// Encodes the GEMM into `mtl_command_buffer` (id<MTLCommandBuffer>, not
// committed, no encoder open). `mtl_device` may be null, in which case the
// command buffer's device is used. Errors: InvalidArgumentError for invalid
// shapes, strides, offsets or null buffers; UnimplementedError for unsupported
// cases (k == 0, dtype combinations, devices without MPS); InternalError when a
// Metal/MPS object cannot be created; ResourceExhaustedError when a staging
// buffer cannot be allocated. Staging work may already be encoded when an error
// is returned; everything it references is kept alive.
absl::Status RunMpsGemm(void* mtl_device, void* mtl_command_buffer,
                        const GemmParams& params);

std::string GemmParamsDebugString(const GemmParams& p);

}  // namespace blas
}  // namespace metal_pjrt

#endif  // METAL_PJRT_PLUGIN_BLAS_MPS_GEMM_H_
