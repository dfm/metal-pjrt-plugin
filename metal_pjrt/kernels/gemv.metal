// MSL source for the small-M GEMM ("wide gemv", gemv.h): out = x W^T for a
// few vectors x. A port of MLX's gemv_wide (mlx/backend/metal/kernels/
// gemv.h, MLX 0.32.2), which is:
//
//   Copyright (c) 2023-2024 Apple Inc. MIT License (MLX).
//   Permission is hereby granted, free of charge, to any person obtaining a
//   copy of this software and associated documentation files (the
//   "Software"), to deal in the Software without restriction, including
//   without limitation the rights to use, copy, modify, merge, publish,
//   distribute, sublicense, and/or sell copies of the Software, and to permit
//   persons to whom the Software is furnished to do so, subject to the
//   following conditions: The above copyright notice and this permission
//   notice shall be included in all copies or substantial portions of the
//   Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
//
// Changes from MLX: self-contained (no MLX headers); an output type U (T or
// float); the output addressed by two strides (per vector, per matrix row)
// so that a GEMM whose small side is N (D = A B with B's columns the
// vectors) writes D directly; batches by plain strides on grid z; in place of
// MLX's axpby, the steel epilogue (steel_gemm.metal): alpha, beta * D, bias,
// aux and activation, in f32, rounded once; the switches are function
// constants; one instantiation per compiled source (see the end).
//
// Template parameters: T (input type), U (output type), NV (vectors per
// threadgroup pass, MLX's vecs_per_tg), KL (lanes reducing K for one matrix
// row, MLX's k_lanes: 16 or 32).
#include <metal_stdlib>
using namespace metal;

#define GEMV_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")

// Per-call switches (GemvConstants in gemv.cc).
constant bool USE_C [[function_constant(0)]];  // D = alpha x W^T + beta D
constant bool EPI_BIAS [[function_constant(1)]];
constant bool EPI_AUX [[function_constant(2)]];
// 0 none, 1 ReLU, 2 GELU (tanh), 3 SiLU.
constant int EPI_ACT [[function_constant(3)]];

// Must match GemvParams in gemv.cc (64 bytes). Strides in elements.
struct GemvParams {
  int K;               // reduction length, a multiple of 4
  int rows;            // matrix rows (outputs per vector)
  int vecs;            // vectors
  int mat_ld;          // between matrix rows, a multiple of 4
  int vec_ld;          // between vectors, a multiple of 4
  int out_vec_stride;  // D between consecutive vectors
  int out_row_stride;  // D between consecutive matrix rows
  int bias_by_row;     // bias index: 1 the matrix row, 0 the vector
  long batch_stride_mat, batch_stride_vec, batch_stride_d;
  float alpha, beta;
};

// Epilogue activation, in f32 (steel_act in steel_gemm.metal).
METAL_FUNC float gemv_act(float x) {
  if (EPI_ACT == 1) return isnan(x) ? x : (x > 0.0f ? x : 0.0f);  // keeps NaN
  if (EPI_ACT == 2) {
    const float k = 0.7978845608028654f;  // sqrt(2/pi)
    return 0.5f * x * (1.0f + precise::tanh(k * (x + 0.044715f * x * x * x)));
  }
  if (EPI_ACT == 3) return x / (1.0f + precise::exp(-x));
  return x;
}

// out[v, r] = sum_k x[v, k] mat[r, k] for NV vectors per pass: each
// threadgroup streams a block of matrix rows once and applies it to NV
// input vectors, so the matrix is read ceil(vecs / NV) times instead of once
// per padded GEMM tile. KL lanes reduce K for one row, 32 / KL rows per
// simdgroup, KL / 8 simdgroups per threadgroup (4 rows per threadgroup).
template <typename T, typename U, int NV, int KL>
[[kernel]] void gemv_wide(const device T* mat [[buffer(0)]],
                          const device T* in_vec [[buffer(1)]],
                          device U* d [[buffer(2)]],
                          const device U* bias [[buffer(3)]],  // d if unused
                          device U* aux [[buffer(4)]],         // d if unused
                          const constant GemvParams& p [[buffer(5)]],
                          uint3 tid [[threadgroup_position_in_grid]],
                          uint3 tgpg [[threadgroups_per_grid]],
                          uint simd_gid [[simdgroup_index_in_threadgroup]],
                          uint simd_lid [[thread_index_in_simdgroup]]) {
  typedef float AccT;
  constexpr int unroll = 8;
  constexpr int num_simdgroups = KL / 8;
  constexpr int rows_per_simdgroup = 32 / KL;
  constexpr int rows_per_tg = rows_per_simdgroup * num_simdgroups;

  mat += p.batch_stride_mat * long(tid.z);
  in_vec += p.batch_stride_vec * long(tid.z);
  d += p.batch_stride_d * long(tid.z);
  aux += p.batch_stride_d * long(tid.z);

  const short k_lane = simd_lid % KL;  // this lane's slot in the K reduction
  const short sg_row = simd_lid / KL;  // which output row of the simdgroup

  const int out_row =
      tid.y * rows_per_tg + rows_per_simdgroup * simd_gid + sg_row;

  // Clamped tail rows/vectors read valid memory; the guarded writes below
  // never store them.
  const int row = min(out_row, p.rows - 1);
  const device T* wrow = mat + long(row) * p.mat_ld;
  const device vec<T, 4>* w4 = (const device vec<T, 4>*)wrow;
  const int n_v4 = p.K / 4;
  const int n_main = n_v4 - n_v4 % (KL * unroll);

  // Vector chunks beyond grid.x round-robin onto the same threadgroups:
  // re-walking the row block hits cache where an extra grid column would
  // re-stream it from DRAM.
  const device vec<T, 4>* x4 = (const device vec<T, 4>*)in_vec;
  const int x_ld4 = p.vec_ld / 4;
  const int n_chunks = (p.vecs + NV - 1) / NV;
  for (int chunk = tid.x; chunk < n_chunks; chunk += tgpg.x) {
    const int vec0 = chunk * NV;

    int x_off4[NV];
    for (int v = 0; v < NV; v++) {
      x_off4[v] = min(vec0 + v, p.vecs - 1) * x_ld4;
    }

    AccT result[NV] = {0};

    // Adjacent lanes read adjacent blocks so every transaction lands on
    // whole cachelines; the unroll keeps loads in flight on short rows.
    for (int base = 0; base < n_main; base += KL * unroll) {
      vec<AccT, 4> wf[unroll];
      GEMV_PRAGMA_UNROLL
      for (int i = 0; i < unroll; i++) {
        wf[i] = vec<AccT, 4>(w4[base + i * KL + k_lane]);
      }
      GEMV_PRAGMA_UNROLL
      for (int v = 0; v < NV; v++) {
        AccT acc = 0;
        GEMV_PRAGMA_UNROLL
        for (int i = 0; i < unroll; i++) {
          acc += dot(wf[i],
                     vec<AccT, 4>(x4[x_off4[v] + base + i * KL + k_lane]));
        }
        result[v] += acc;
      }
    }
    for (int idx = n_main + k_lane; idx < n_v4; idx += KL) {
      const vec<AccT, 4> wf = vec<AccT, 4>(w4[idx]);
      GEMV_PRAGMA_UNROLL
      for (int v = 0; v < NV; v++) {
        result[v] += dot(wf, vec<AccT, 4>(x4[x_off4[v] + idx]));
      }
    }

    // The halving shuffles reduce each row's KL lanes while rows sharing the
    // simdgroup stay separate.
    GEMV_PRAGMA_UNROLL
    for (int v = 0; v < NV; v++) {
      GEMV_PRAGMA_UNROLL
      for (ushort off = KL / 2; off >= 1; off >>= 1) {
        result[v] += simd_shuffle_down(result[v], off);
      }
    }

    // v = alpha * acc (+ beta * D) + bias; aux = v; D = act(v).
    if (k_lane == 0 && out_row < p.rows) {
      for (int v = 0; v < NV; v++) {
        if (vec0 + v < p.vecs) {
          const long i = long(vec0 + v) * p.out_vec_stride +
                         long(out_row) * p.out_row_stride;
          float x = p.alpha * result[v];
          if (USE_C) x += p.beta * float(d[i]);
          if (EPI_BIAS) x += float(bias[p.bias_by_row ? out_row : vec0 + v]);
          if (EPI_AUX) aux[i] = U(x);
          d[i] = U(gemv_act(x));
        }
      }
    }
  }
}

// Each source compiles one variant: gemv.cc appends one
//   instantiate_gemv_wide("host name", T, U, NV, KL)
#define instantiate_gemv_wide(name, T, U, NV, KL)                      \
  template [[host_name(name)]] [[kernel]]                              \
  decltype(gemv_wide<T, U, NV, KL>) gemv_wide<T, U, NV, KL>;
