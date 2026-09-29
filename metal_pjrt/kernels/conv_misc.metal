// Helper kernels of the convolutions (conv/conv.h): the explicit and
// weight-gradient paths' unfold (im2col, in place of MLX's naive_unfold_Nd),
// the zero padding of the channel dimension for the pad-channels path and
// the weight gradient's sum of its split-K partial products.
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
  int rows;        // rows in this chunk
  int vecs;        // vectors (of U) per input pixel: C * sizeof(T) / sizeof(U)
};

// Unfolds rows [row_offset, row_offset + rows) of the [N * oH * oW,
// kH * kW * C] patch matrix of a dense NHWC input into `out` (chunk-local
// rows), zero where a tap falls on padding or between dilated input
// elements. A bit copy in vectors U of 2-16 bytes (whole channels' worth;
// zero is all-zero bits in every type), one thread per (row, tap, vector)
// on a 1-D grid, so the pixel and tap index math is done once per vector.
// (MLX's naive_unfold_Nd copies one element per thread, with about eight
// integer divisions each: ALU-bound at ~12 GB/s on an M3.)
template <typename U>
[[kernel]] void unfold_2d(const device U* in [[buffer(0)]],
                          device U* out [[buffer(1)]],
                          const constant SteelConvParams* params [[buffer(2)]],
                          const constant UnfoldRows* chunk [[buffer(3)]],
                          uint gid [[thread_position_in_grid]]) {
  const int cv = chunk->vecs;
  const int cols = params->wS[0] * params->wS[1] * cv;
  if (gid >= uint(cols) * uint(chunk->rows)) return;
  const int local_row = int(gid) / cols;
  const int col = int(gid) - local_row * cols;
  const int out_pixels = params->oS[0] * params->oS[1];
  const int row = chunk->row_offset + local_row;
  const int n = row / out_pixels;
  const int r = row - n * out_pixels;
  const int oh = r / params->oS[1];
  const int ow = r - oh * params->oS[1];
  const int tap = col / cv;
  const int c = col - tap * cv;
  int kh = tap / params->wS[1];
  int kw = tap - kh * params->wS[1];
  if (params->flip) {
    kh = params->wS[0] - 1 - kh;
    kw = params->wS[1] - 1 - kw;
  }
  // Positions in the dilated input, then in the input.
  const int hd = oh * params->str[0] - params->pad[0] + kh * params->kdil[0];
  const int wd = ow * params->str[1] - params->pad[1] + kw * params->kdil[1];
  const bool on_grid = hd >= 0 && wd >= 0 && hd % params->idil[0] == 0 &&
                       wd % params->idil[1] == 0;
  const int ih = hd / params->idil[0];
  const int iw = wd / params->idil[1];
  U v = U(0);
  if (on_grid && ih < params->iS[0] && iw < params->iS[1] && n < params->N) {
    v = in[((size_t(n) * params->iS[0] + ih) * params->iS[1] + iw) * cv + c];
  }
  out[size_t(local_row) * cols + col] = v;
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

// Must match SumSplits in conv/conv_kernels.cc.
struct SumSplits {
  uint count;   // elements of `out`
  uint splits;  // partial sums per element, `count` apart
};

// out[i] = sum over s of partial[s * count + i], in f32, rounded once.
// Grid: x over count.
template <typename T>
[[kernel]] void sum_splits(const device float* partial [[buffer(0)]],
                           device T* out [[buffer(1)]],
                           const constant SumSplits* p [[buffer(2)]],
                           uint gid [[thread_position_in_grid]]) {
  if (gid >= p->count) return;
  float acc = 0.0f;
  for (uint s = 0; s < p->splits; ++s) {
    acc += partial[size_t(s) * p->count + gid];
  }
  out[gid] = T(acc);
}

#define instantiate_conv_misc(tname, T)                                      \
  template [[host_name("pad_cols_" #tname)]] [[kernel]] decltype(pad_cols<T>) \
      pad_cols<T>;                                                           \
  template [[host_name("sum_splits_" #tname)]] [[kernel]] decltype(          \
      sum_splits<T>) sum_splits<T>;

#define instantiate_unfold(bytes, U)                                          \
  template [[host_name("unfold_2d_" #bytes)]] [[kernel]] decltype(            \
      unfold_2d<U>) unfold_2d<U>;

instantiate_unfold(2, ushort)
instantiate_unfold(4, uint)
instantiate_unfold(8, uint2)
instantiate_unfold(16, uint4)

instantiate_conv_misc(float, float)
instantiate_conv_misc(half, half)
instantiate_conv_misc(bfloat, bfloat)
