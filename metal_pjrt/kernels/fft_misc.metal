// Helper kernels of the FFTs (fft/fft.h): the length-1 real transforms and
// the elementwise steps of the multi-upload Bluestein FFT around its two
// four-step passes (MLX chains copies, multiplies, pads and slices for
// these, mlx/backend/metal/fft.cpp multi_upload_bluestein_fft; here they
// are fused into three kernels). complex64 is float2.
#include <metal_stdlib>
using namespace metal;

// Must match FftMiscParams in fft/fft.cc.
struct FftMiscParams {
  uint rows;     // rows in this launch
  uint n;        // transform length
  uint in_len;   // elements per input row
  uint out_len;  // elements per output row
  uint pad_len;  // bluestein_n: elements per workspace row
};

// Must match FftType in fft/fft_plan.h.
constant constexpr int kFft = 0;
constant constexpr int kIfft = 1;
constant constexpr int kRfft = 2;
constant constexpr int kIrfft = 3;

float2 cmul(float2 a, float2 b) {
  return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// n == 1: rfft widens, irfft keeps the real part. Grid: x over rows.
kernel void fft_real_to_complex(const device float* in [[buffer(0)]],
                                device float2* out [[buffer(1)]],
                                constant FftMiscParams& p [[buffer(2)]],
                                uint i [[thread_position_in_grid]]) {
  if (i < p.rows) out[i] = float2(in[i], 0.0f);
}

kernel void fft_complex_to_real(const device float2* in [[buffer(0)]],
                                device float* out [[buffer(1)]],
                                constant FftMiscParams& p [[buffer(2)]],
                                uint i [[thread_position_in_grid]]) {
  if (i < p.rows) out[i] = in[i].x;
}

// Bluestein input: ws[r, k] = z[r, k] * w_k[k] for k < n, 0 up to pad_len,
// where z is the input row as complex64 (conjugated for the inverse
// transforms; for kIrfft the conjugate of the full Hermitian spectrum,
// X[n - k] above n / 2). Grid: x over pad_len, y over rows.
template <int kType, typename InT>
[[kernel]] void bluestein_pre(const device InT* in [[buffer(0)]],
                              const device float2* w_k [[buffer(1)]],
                              device float2* ws [[buffer(2)]],
                              constant FftMiscParams& p [[buffer(3)]],
                              uint2 pos [[thread_position_in_grid]]) {
  const uint k = pos.x, r = pos.y;
  if (k >= p.pad_len || r >= p.rows) return;
  float2 z = 0.0f;
  if (k < p.n) {
    const device InT* row = in + size_t(r) * p.in_len;
    if constexpr (kType == kRfft) {
      z = float2(row[k], 0.0f);
    } else if constexpr (kType == kIrfft) {
      // bins above in_len - 1 = n / 2 mirror the lower half, conjugated;
      // conjugating everything for the inverse leaves them as they are.
      z = k < p.in_len ? row[k] * float2(1.0f, -1.0f) : row[p.n - k];
    } else if constexpr (kType == kIfft) {
      z = row[k] * float2(1.0f, -1.0f);
    } else {
      z = row[k];
    }
    z = cmul(z, w_k[k]);
  }
  ws[size_t(r) * p.pad_len + k] = z;
}

// ws[r, k] *= w_q[k]. Grid: x over pad_len, y over rows.
kernel void bluestein_mul(const device float2* w_q [[buffer(0)]],
                          device float2* ws [[buffer(1)]],
                          constant FftMiscParams& p [[buffer(2)]],
                          uint2 pos [[thread_position_in_grid]]) {
  const uint k = pos.x, r = pos.y;
  if (k >= p.pad_len || r >= p.rows) return;
  const size_t i = size_t(r) * p.pad_len + k;
  ws[i] = cmul(ws[i], w_q[k]);
}

// Bluestein output: y = ws[r, n - 1 + k] * w_k[k] for k < out_len, then
// conjugated and scaled by 1 / n for kIfft, its real part scaled by 1 / n
// for kIrfft. Grid: x over out_len, y over rows.
template <int kType, typename OutT>
[[kernel]] void bluestein_post(const device float2* ws [[buffer(0)]],
                               const device float2* w_k [[buffer(1)]],
                               device OutT* out [[buffer(2)]],
                               constant FftMiscParams& p [[buffer(3)]],
                               uint2 pos [[thread_position_in_grid]]) {
  const uint k = pos.x, r = pos.y;
  if (k >= p.out_len || r >= p.rows) return;
  const float2 y =
      cmul(ws[size_t(r) * p.pad_len + p.n - 1 + k], w_k[k]);
  device OutT* row = out + size_t(r) * p.out_len;
  if constexpr (kType == kIrfft) {
    row[k] = y.x / p.n;
  } else if constexpr (kType == kIfft) {
    row[k] = float2(y.x / p.n, -y.y / p.n);
  } else {
    row[k] = y;
  }
}

#define instantiate_bluestein(name, type, in_t, out_t)                     \
  template [[host_name("bluestein_pre_" #name)]] [[kernel]]              \
  decltype(bluestein_pre<type, in_t>) bluestein_pre<type, in_t>;          \
  template [[host_name("bluestein_post_" #name)]] [[kernel]]             \
  decltype(bluestein_post<type, out_t>) bluestein_post<type, out_t>;

instantiate_bluestein(fft, kFft, float2, float2)
instantiate_bluestein(ifft, kIfft, float2, float2)
instantiate_bluestein(rfft, kRfft, float, float2)
instantiate_bluestein(irfft, kIrfft, float2, float)
