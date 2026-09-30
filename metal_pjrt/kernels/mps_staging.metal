// Copyright 2026 The metal-pjrt-plugin Authors
// SPDX-License-Identifier: Apache-2.0

// Operand staging for the MPS GEMM (blas/mps_gemm_objc.cc): copies the
// matrices of an operand, element (col, row, batch) at
// off + batch*bs + row*ld + col on both sides. `Args` must match StagingArgs.
#include <metal_stdlib>
using namespace metal;
struct Args { ulong in_off, out_off, ld, bs; uint cols, rows, batches, pad; };
kernel void copy_f32(device const uint* in [[buffer(0)]],
                     device uint* out [[buffer(1)]],
                     constant Args& a [[buffer(2)]],
                     uint3 g [[thread_position_in_grid]]) {
  if (g.x >= a.cols || g.y >= a.rows || g.z >= a.batches) return;
  ulong e = ulong(g.z) * a.bs + ulong(g.y) * a.ld + ulong(g.x);
  out[a.out_off + e] = in[a.in_off + e];
}
