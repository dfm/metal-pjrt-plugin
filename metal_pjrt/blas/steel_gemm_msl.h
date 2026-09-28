// MSL source for the "steel" GEMM (steel_gemm.h). A port of the tiled
// simdgroup-matrix GEMM from MLX (mlx/backend/metal/kernels/steel/gemm/
// {loader,mma,gemm}.h), which is:
//
//   Copyright (c) 2023 ml-explore. MIT License.
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
// Changes from MLX: self-contained (no MLX headers), in-place alpha/beta
// epilogue reading C from D, the cuBLASLt bias / activation / aux epilogue
// (blas_lt_support.h) applied in the store, batch strides via grid z, and
// specializations chosen by a generated prefix (steel_gemm.cc) instead of
// instantiating every variant in one library.
//
// The prefix defines: T (input type), U (output type), BM BN BK WM WN,
// TRANS_A TRANS_B MN_ALIGNED K_ALIGNED USE_C EPI_BIAS EPI_AUX (bools),
// EPI_ACT (int: 0 none, 1 ReLU, 2 GELU (tanh), 3 SiLU), then this body.
#ifndef METAL_PJRT_BLAS_STEEL_GEMM_MSL_H_
#define METAL_PJRT_BLAS_STEEL_GEMM_MSL_H_

namespace metal_pjrt {
namespace blas {

inline constexpr char kSteelGemmKernelName[] = "steel_gemm";

inline constexpr char kSteelGemmMslBody[] = R"MSL(
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>
using namespace metal;

#define STEEL_CONST static constant constexpr const
#define STEEL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")

// Must match SteelGemmParams in steel_gemm.cc (72 bytes).
struct SteelGemmParams {
  int M, N, K;
  int lda, ldb, ldd;
  int tiles_n, tiles_m;
  long batch_stride_a, batch_stride_b, batch_stride_d;
  int swizzle_log;
  int gemm_k_iterations_aligned;
  float alpha, beta;
};

// Epilogue activation, in f32 (the formulas of metal_blas.cc's second pass).
METAL_FUNC float steel_act(float x) {
  if (EPI_ACT == 1) return isnan(x) ? x : (x > 0.0f ? x : 0.0f);  // keeps NaN
  if (EPI_ACT == 2) {
    const float k = 0.7978845608028654f;  // sqrt(2/pi)
    return 0.5f * x * (1.0f + precise::tanh(k * (x + 0.044715f * x * x * x)));
  }
  if (EPI_ACT == 3) return x / (1.0f + precise::exp(-x));
  return x;
}

// Cooperative device -> threadgroup tile copy (MLX BlockLoader).
template <typename Ty, short BROWS, short BCOLS, short dst_ld,
          short reduction_dim, short tgp_size>
struct BlockLoader {
  STEEL_CONST short n_reads = (BCOLS * BROWS) / tgp_size;
  STEEL_CONST short vec_size = n_reads;
  STEEL_CONST short TCOLS = BCOLS / n_reads;
  STEEL_CONST short TROWS = tgp_size / TCOLS;

  const int src_ld;
  const int tile_stride;
  const short thread_idx;
  const short bi;
  const short bj;
  threadgroup Ty* dst;
  const device Ty* src;

  struct alignas(sizeof(Ty)) ReadVector {
    uint8_t v[sizeof(Ty) * vec_size];
  };

  METAL_FUNC BlockLoader(const device Ty* src_, const int src_ld_,
                         threadgroup Ty* dst_, ushort simd_group_id,
                         ushort simd_lane_id) thread
      : src_ld(src_ld_),
        tile_stride(reduction_dim ? BCOLS : BROWS * src_ld),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj(vec_size * (thread_idx % TCOLS)),
        dst(dst_ + bi * dst_ld + bj),
        src(src_ + bi * src_ld + bj) {}

  METAL_FUNC void load_unsafe() const thread {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      *((threadgroup ReadVector*)(&dst[i * dst_ld])) =
          *((const device ReadVector*)(&src[i * src_ld]));
    }
  }

  // src_tile_dim = (cols, rows) still valid from this tile's origin.
  METAL_FUNC void load_safe(short2 src_tile_dim) const thread {
    src_tile_dim = src_tile_dim - short2(bj, bi);
    if (src_tile_dim.x <= 0 || src_tile_dim.y <= 0) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < BROWS; i += TROWS) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; j++) dst[i * dst_ld + j] = Ty(0);
      }
      return;
    }
    bool tmp_idx[vec_size];
    Ty tmp_val[vec_size];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++)
        tmp_idx[j] = (i < src_tile_dim.y) && (j < src_tile_dim.x);
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++)
        tmp_val[j] = src[(tmp_idx[j] ? i * src_ld + j : 0)];
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++)
        tmp_val[j] = tmp_idx[j] ? tmp_val[j] : Ty(0);
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) dst[i * dst_ld + j] = tmp_val[j];
    }
  }

  METAL_FUNC void next() thread { src += tile_stride; }
};

// Per-simdgroup MMA over a (BM, BK) x (BK, BN) threadgroup tile, f32
// accumulators in 8x8 simdgroup matrices (MLX BlockMMA). Each lane owns two
// adjacent columns of one row of every 8x8 fragment.
template <typename Ty, typename Uy, int BM_, int BN_, int BK_, int WM_,
          int WN_, bool TA, bool TB, short lda_tgp, short ldb_tgp>
struct BlockMMA {
  typedef simdgroup_matrix<float, 8, 8> mat;
  STEEL_CONST short kFrag = 8;
  STEEL_CONST short TM_stride = kFrag * WM_;
  STEEL_CONST short TN_stride = kFrag * WN_;
  STEEL_CONST short TM = BM_ / TM_stride;
  STEEL_CONST short TN = BN_ / TN_stride;
  STEEL_CONST short A_str_m = TA ? 1 : lda_tgp;
  STEEL_CONST short A_str_k = TA ? lda_tgp : 1;
  STEEL_CONST short B_str_k = TB ? 1 : ldb_tgp;
  STEEL_CONST short B_str_n = TB ? ldb_tgp : 1;
  STEEL_CONST short tile_stride_a = kFrag * A_str_k;
  STEEL_CONST short tile_stride_b = kFrag * B_str_k;

  mat Atile[TM];
  mat Btile[TN];
  mat Ctile[TM * TN];
  short sm, sn;
  short As_offset, Bs_offset;

  METAL_FUNC BlockMMA(ushort simd_group_id, ushort simd_lane_id) thread {
    short tm = kFrag * (simd_group_id / WN_);
    short tn = kFrag * (simd_group_id % WN_);
    const short qid = simd_lane_id / 4;
    sm = (qid & 4) + ((simd_lane_id / 2) % 4);
    sn = (qid & 2) * 2 + (simd_lane_id % 2) * 2;
    As_offset = (tm + sm) * A_str_m + sn * A_str_k;
    Bs_offset = sm * B_str_k + (tn + sn) * B_str_n;
    sm += tm;
    sn += tn;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < TM * TN; i++)
      Ctile[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  }

  METAL_FUNC void mma(const threadgroup Ty* As, const threadgroup Ty* Bs)
      thread {
    As += As_offset;
    Bs += Bs_offset;
    STEEL_PRAGMA_UNROLL
    for (short kk = 0; kk < BK_; kk += kFrag) {
      simdgroup_barrier(mem_flags::mem_none);
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < TM; i++) {
        const threadgroup Ty* p = As + i * TM_stride * A_str_m;
        Atile[i].thread_elements()[0] = float(p[0]);
        Atile[i].thread_elements()[1] = float(p[A_str_k]);
      }
      simdgroup_barrier(mem_flags::mem_none);
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < TN; j++) {
        const threadgroup Ty* p = Bs + j * TN_stride * B_str_n;
        Btile[j].thread_elements()[0] = float(p[0]);
        Btile[j].thread_elements()[1] = float(p[B_str_n]);
      }
      simdgroup_barrier(mem_flags::mem_none);
      STEEL_PRAGMA_UNROLL
      for (short m = 0; m < TM; m++) {
        STEEL_PRAGMA_UNROLL
        for (short n = 0; n < TN; n++) {
          short ns = (m % 2) ? (TN - 1 - n) : n;  // serpentine
          simdgroup_multiply_accumulate(Ctile[m * TN + ns], Atile[m],
                                        Btile[ns], Ctile[m * TN + ns]);
        }
      }
      As += tile_stride_a;
      Bs += tile_stride_b;
    }
  }

  // One output element: v = alpha * acc (+ beta * D when use_c) + bias of
  // its column; aux = v, D = act(v), rounded once.
  template <bool use_c>
  METAL_FUNC void store_one(device Uy* d, const device Uy* bias,
                            device Uy* aux, float acc, float alpha,
                            float beta) thread {
    float v = alpha * acc;
    if (use_c) v += beta * float(*d);
    if (EPI_BIAS) v += float(*bias);
    if (EPI_AUX) *aux = Uy(v);
    *d = Uy(steel_act(v));
  }

  // Full tile. `bias` points at the tile's first column, `aux` like D.
  template <bool use_c>
  METAL_FUNC void store_result(device Uy* D, const device Uy* bias,
                               device Uy* aux, const int ldd, float alpha,
                               float beta) thread {
    D += sm * ldd + sn;
    aux += sm * ldd + sn;
    bias += sn;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < TM; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < TN; j++) {
        thread auto& acc = Ctile[i * TN + j].thread_elements();
        const int off = (i * TM_stride) * ldd + j * TN_stride;
        STEEL_PRAGMA_UNROLL
        for (short k = 0; k < 2; k++) {
          store_one<use_c>(D + off + k, bias + j * TN_stride + k,
                           aux + off + k, acc[k], alpha, beta);
        }
      }
    }
  }

  // Edge tile: dims = (cols, rows) valid from the tile origin.
  template <bool use_c>
  METAL_FUNC void store_result_safe(device Uy* D, const device Uy* bias,
                                    device Uy* aux, const int ldd, short2 dims,
                                    float alpha, float beta) thread {
    D += sm * ldd + sn;
    aux += sm * ldd + sn;
    bias += sn;
    dims -= short2(sn, sm);
    if (dims.x <= 0 || dims.y <= 0) return;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < TM; i++) {
      if (i * TM_stride < dims.y) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < TN; j++) {
          thread auto& acc = Ctile[i * TN + j].thread_elements();
          const int off = (i * TM_stride) * ldd + j * TN_stride;
          STEEL_PRAGMA_UNROLL
          for (short k = 0; k < 2; k++) {
            if (j * TN_stride + k < dims.x) {
              store_one<use_c>(D + off + k, bias + j * TN_stride + k,
                               aux + off + k, acc[k], alpha, beta);
            }
          }
        }
      }
    }
  }
};

STEEL_CONST short kTgpPadA = 16 / sizeof(T);
STEEL_CONST short kTgpPadB = 16 / sizeof(T);
STEEL_CONST short kTgpSize = WM * WN * 32;
STEEL_CONST short kLdaTgp = TRANS_A ? BM + kTgpPadA : BK + kTgpPadA;
STEEL_CONST short kLdbTgp = TRANS_B ? BK + kTgpPadB : BN + kTgpPadB;
STEEL_CONST int kTgpMemA = TRANS_A ? BK * (BM + kTgpPadA) : BM * (BK + kTgpPadA);
STEEL_CONST int kTgpMemB = TRANS_B ? BN * (BK + kTgpPadB) : BK * (BN + kTgpPadB);

typedef BlockLoader<T, TRANS_A ? BK : BM, TRANS_A ? BM : BK, kLdaTgp,
                    !TRANS_A, kTgpSize> loader_a_t;
typedef BlockLoader<T, TRANS_B ? BN : BK, TRANS_B ? BK : BN, kLdbTgp,
                    TRANS_B, kTgpSize> loader_b_t;
typedef BlockMMA<T, U, BM, BN, BK, WM, WN, TRANS_A, TRANS_B, kLdaTgp,
                 kLdbTgp> mma_t;

template <bool M_aligned, bool N_aligned>
METAL_FUNC void gemm_loop(threadgroup T* As, threadgroup T* Bs,
                          const int gemm_k_iterations,
                          thread loader_a_t& loader_a,
                          thread loader_b_t& loader_b, thread mma_t& mma_op,
                          const short tgp_bm, const short tgp_bn,
                          const short lbk) {
  short2 tile_dims_A = TRANS_A ? short2(tgp_bm, BK) : short2(BK, tgp_bm);
  short2 tile_dims_B = TRANS_B ? short2(BK, tgp_bn) : short2(tgp_bn, BK);
  for (int k = 0; k < gemm_k_iterations; k++) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (M_aligned) loader_a.load_unsafe(); else loader_a.load_safe(tile_dims_A);
    if (N_aligned) loader_b.load_unsafe(); else loader_b.load_safe(tile_dims_B);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma_op.mma(As, Bs);
    loader_a.next();
    loader_b.next();
  }
  if (!K_ALIGNED) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    short2 da = TRANS_A ? short2(tgp_bm, lbk) : short2(lbk, tgp_bm);
    short2 db = TRANS_B ? short2(lbk, tgp_bn) : short2(tgp_bn, lbk);
    loader_a.load_safe(da);
    loader_b.load_safe(db);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma_op.mma(As, Bs);
  }
}

[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void steel_gemm(
    const device T* A [[buffer(0)]],
    const device T* B [[buffer(1)]],
    device U* D [[buffer(2)]],
    const constant SteelGemmParams& params [[buffer(3)]],
    const device U* bias [[buffer(4)]],  // N elements; D when !EPI_BIAS
    device U* aux [[buffer(5)]],         // D's layout; D when !EPI_AUX
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]]) {
  threadgroup T As[kTgpMemA];
  threadgroup T Bs[kTgpMemB];

  const int tid_y = ((tid.y) << params.swizzle_log) +
                    ((tid.x) & ((1 << params.swizzle_log) - 1));
  const int tid_x = (tid.x) >> params.swizzle_log;
  if (params.tiles_n <= tid_x || params.tiles_m <= tid_y) return;

  A += params.batch_stride_a * long(tid.z);
  B += params.batch_stride_b * long(tid.z);
  D += params.batch_stride_d * long(tid.z);
  aux += params.batch_stride_d * long(tid.z);

  const int c_row = tid_y * BM;
  const int c_col = tid_x * BN;
  const size_t c_row_long = size_t(c_row);
  const size_t c_col_long = size_t(c_col);
  A += TRANS_A ? c_row_long : c_row_long * params.lda;
  B += TRANS_B ? c_col_long * params.ldb : c_col_long;
  D += c_row_long * params.ldd + c_col_long;
  aux += c_row_long * params.ldd + c_col_long;
  bias += c_col_long;

  thread loader_a_t loader_a(A, params.lda, As, simd_group_id, simd_lane_id);
  thread loader_b_t loader_b(B, params.ldb, Bs, simd_group_id, simd_lane_id);
  thread mma_t mma_op(simd_group_id, simd_lane_id);

  const int gemm_k_iterations = params.gemm_k_iterations_aligned;
  const short lbk = params.K - gemm_k_iterations * BK;

  if (MN_ALIGNED) {
    gemm_loop<true, true>(As, Bs, gemm_k_iterations, loader_a, loader_b,
                          mma_op, BM, BN, lbk);
    mma_op.template store_result<USE_C>(D, bias, aux, params.ldd,
                                        params.alpha, params.beta);
    return;
  }
  const short tgp_bm = min(BM, params.M - c_row);
  const short tgp_bn = min(BN, params.N - c_col);
  if (tgp_bm == BM && tgp_bn == BN) {
    gemm_loop<true, true>(As, Bs, gemm_k_iterations, loader_a, loader_b,
                          mma_op, tgp_bm, tgp_bn, lbk);
    mma_op.template store_result<USE_C>(D, bias, aux, params.ldd,
                                        params.alpha, params.beta);
    return;
  } else if (tgp_bn == BN) {
    gemm_loop<false, true>(As, Bs, gemm_k_iterations, loader_a, loader_b,
                           mma_op, tgp_bm, tgp_bn, lbk);
  } else if (tgp_bm == BM) {
    gemm_loop<true, false>(As, Bs, gemm_k_iterations, loader_a, loader_b,
                           mma_op, tgp_bm, tgp_bn, lbk);
  } else {
    gemm_loop<false, false>(As, Bs, gemm_k_iterations, loader_a, loader_b,
                            mma_op, tgp_bm, tgp_bn, lbk);
  }
  mma_op.template store_result_safe<USE_C>(D, bias, aux, params.ldd,
                                           short2(tgp_bn, tgp_bm),
                                           params.alpha, params.beta);
}
)MSL";

}  // namespace blas
}  // namespace metal_pjrt

#endif  // METAL_PJRT_BLAS_STEEL_GEMM_MSL_H_
