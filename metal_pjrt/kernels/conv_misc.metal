// Helper kernels of the convolutions (conv/conv.h): the explicit path's
// unfold (im2col; MLX's naive_unfold_Nd for 2-D, from mlx/backend/metal/
// kernels/conv.metal, Copyright (c) 2023-2024 Apple Inc., MIT License, see
// the full notice in steel_conv.metal) and the zero padding of the channel
// dimension for the pad-channels path.
#include <metal_stdlib>
using namespace metal;

// Must match SteelConvParams in conv/conv_kernels.cc (and steel_conv.metal).
struct SteelConvParams {
  int N;
  int C;
  int O;
  int iS[2];
  int wS[2];
  int oS[2];
  int str[2];
  int pad[2];
  int kdil[2];
  int idil[2];
  long in_strides[4];
  long wt_strides[4];
  long out_strides[4];
  int groups;
  bool flip;
};

// Must match UnfoldRows in conv/conv_kernels.cc.
struct UnfoldRows {
  int row_offset;  // first output pixel (row of the unfolded matrix)
  int rows;        // rows in this tile
};

// Unfolds rows [row_offset, row_offset + rows) of the [N * oH * oW,
// kH * kW * C] patch matrix of an NHWC input into `out` (tile-local rows),
// zero where a tap falls on padding or between dilated input elements.
// Grid: x over C, y over kH * kW, z over rows.
template <typename T>
[[kernel]] void naive_unfold_2d(const device T* in [[buffer(0)]],
                                device T* out [[buffer(1)]],
                                const constant SteelConvParams* params
                                [[buffer(2)]],
                                const constant UnfoldRows* tile [[buffer(3)]],
                                uint3 gid [[thread_position_in_grid]]) {
  const int filter_size = params->C * params->wS[0] * params->wS[1];
  if (int(gid.x) >= params->C || int(gid.y) >= params->wS[0] * params->wS[1] ||
      int(gid.z) >= tile->rows) {
    return;
  }
  const int out_pixels = params->oS[0] * params->oS[1];
  out += size_t(gid.z) * filter_size + size_t(gid.y) * params->C;

  int is[2] = {0, 0};
  const int global_row = tile->row_offset + int(gid.z);
  const int n = global_row / out_pixels;
  int oS = global_row % out_pixels;
  int wS = gid.y;
  bool valid = n < params->N;
  for (int i = 1; i >= 0; --i) {
    const int os_ = oS % params->oS[i];
    int ws_ = wS % params->wS[i];
    ws_ = params->flip ? params->wS[i] - ws_ - 1 : ws_;
    const int is_ = os_ * params->str[i] - params->pad[i] + ws_ * params->kdil[i];
    const int is_max = 1 + params->idil[i] * (params->iS[i] - 1);
    valid &= is_ >= 0 && is_ < is_max && (is_ % params->idil[i] == 0);
    is[i] = is_ / params->idil[i];
    oS /= params->oS[i];
    wS /= params->wS[i];
  }
  if (valid) {
    const size_t in_offset = size_t(n) * params->in_strides[0] +
                             size_t(is[0]) * params->in_strides[1] +
                             size_t(is[1]) * params->in_strides[2];
    out[gid.x] = in[in_offset + gid.x];
  } else {
    out[gid.x] = T(0);
  }
}

// Must match PadRows in conv/conv_kernels.cc.
struct PadRows {
  uint rows;
  uint cols;      // columns of `in`
  uint out_cols;  // columns of `out` (>= cols); the rest are zero
};

// out[r, c] = c < cols ? in[r, c] : 0 for r < rows, c < out_cols.
// Grid: x over out_cols, y over rows.
template <typename T>
[[kernel]] void pad_cols(const device T* in [[buffer(0)]],
                         device T* out [[buffer(1)]],
                         const constant PadRows* p [[buffer(2)]],
                         uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= p->out_cols || gid.y >= p->rows) return;
  out[size_t(gid.y) * p->out_cols + gid.x] =
      gid.x < p->cols ? in[size_t(gid.y) * p->cols + gid.x] : T(0);
}

#define instantiate_conv_misc(tname, T)                                      \
  template [[host_name("naive_unfold_2d_" #tname)]] [[kernel]] decltype(     \
      naive_unfold_2d<T>) naive_unfold_2d<T>;                                \
  template [[host_name("pad_cols_" #tname)]] [[kernel]] decltype(pad_cols<T>) \
      pad_cols<T>;

instantiate_conv_misc(float, float)
instantiate_conv_misc(half, half)
instantiate_conv_misc(bfloat, bfloat)
