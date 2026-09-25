// "metal$softmax": row softmax / log-softmax over the minor dimension.
//
// Operand and result: same shape [..., n], row-major, f32/f16/bf16.
// Attributes: log (bool), row_length (i64, must equal the minor dim).
//
// One threadgroup per row. Pass 1 reads the row once and keeps an online
// (max, sum of exp(x - max)) pair per thread, combined with simdgroup and
// threadgroup reductions. Pass 2 re-reads the row (from cache for rows up to
// a few tens of KB) and writes exp(x - max) / sum or (x - max) - log(sum).
// Rows longer than 4 * threadgroup size loop. All math is f32.
#include <algorithm>
#include <cstdint>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "metal_pjrt_plugin/ffi/metal_ffi.h"
#include "xla/backends/gpu/ffi.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"

namespace metal_pjrt {
namespace ffi {
namespace {

namespace xffi = ::xla::ffi;

// $T: element type. Kernel names: softmax_v (n % 4 == 0, vector loads) and
// softmax_s (scalar loads).
constexpr char kSoftmaxMsl[] = R"msl(
#include <metal_stdlib>
using namespace metal;
typedef $T T;
typedef vec<$T, 4> T4;
struct Params { uint n; uint log_softmax; };

inline void accumulate4(thread float& m, thread float& s, float4 v) {
  float lm = max(max(v.x, v.y), max(v.z, v.w));
  float nm = max(m, lm);
  if (nm == -INFINITY) return;
  float scale = (m == -INFINITY) ? 0.0f : exp(m - nm);
  s = s * scale + (exp(v.x - nm) + exp(v.y - nm)) + (exp(v.z - nm) + exp(v.w - nm));
  m = nm;
}

// Combines the per-thread (m, s) pairs of the threadgroup; every thread gets
// the row's max and sum of exp(x - max).
inline void row_stats(thread float& m, thread float& s,
                      threadgroup float* sm, threadgroup float* ss,
                      uint lane, uint sg, uint nsg) {
  float gm = simd_max(m);
  float sc = (m == -INFINITY) ? 0.0f : s * exp(m - gm);
  float gs = simd_sum(sc);
  if (lane == 0) { sm[sg] = gm; ss[sg] = gs; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    float m2 = lane < nsg ? sm[lane] : -INFINITY;
    float s2 = lane < nsg ? ss[lane] : 0.0f;
    float tm = simd_max(m2);
    float t = (m2 == -INFINITY) ? 0.0f : s2 * exp(m2 - tm);
    float ts = simd_sum(t);
    if (lane == 0) { sm[32] = tm; ss[32] = ts; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  m = sm[32];
  s = ss[32];
}

kernel void softmax_v(device const T* in [[buffer(0)]],
                      device T* out [[buffer(1)]],
                      constant Params& p [[buffer(2)]],
                      uint row [[threadgroup_position_in_grid]],
                      uint tid [[thread_position_in_threadgroup]],
                      uint tg [[threads_per_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]],
                      uint nsg [[simdgroups_per_threadgroup]]) {
  threadgroup float sm[33];
  threadgroup float ss[33];
  const uint n4 = p.n / 4;
  device const T4* x = reinterpret_cast<device const T4*>(in + (ulong)row * p.n);
  device T4* y = reinterpret_cast<device T4*>(out + (ulong)row * p.n);
  float m = -INFINITY, s = 0.0f;
  for (uint i = tid; i < n4; i += tg) accumulate4(m, s, float4(x[i]));
  row_stats(m, s, sm, ss, lane, sg, nsg);
  if (p.log_softmax != 0) {
    const float logs = log(s);
    for (uint i = tid; i < n4; i += tg) y[i] = T4((float4(x[i]) - m) - logs);
  } else {
    for (uint i = tid; i < n4; i += tg) y[i] = T4(exp(float4(x[i]) - m) / s);
  }
}

kernel void softmax_s(device const T* in [[buffer(0)]],
                      device T* out [[buffer(1)]],
                      constant Params& p [[buffer(2)]],
                      uint row [[threadgroup_position_in_grid]],
                      uint tid [[thread_position_in_threadgroup]],
                      uint tg [[threads_per_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]],
                      uint nsg [[simdgroups_per_threadgroup]]) {
  threadgroup float sm[33];
  threadgroup float ss[33];
  const uint n = p.n;
  device const T* x = in + (ulong)row * n;
  device T* y = out + (ulong)row * n;
  float m = -INFINITY, s = 0.0f;
  for (uint i = tid * 4; i < n; i += tg * 4) {
    float4 v;
    v.x = float(x[i]);
    v.y = i + 1 < n ? float(x[i + 1]) : -INFINITY;
    v.z = i + 2 < n ? float(x[i + 2]) : -INFINITY;
    v.w = i + 3 < n ? float(x[i + 3]) : -INFINITY;
    accumulate4(m, s, v);
  }
  row_stats(m, s, sm, ss, lane, sg, nsg);
  if (p.log_softmax != 0) {
    const float logs = log(s);
    for (uint i = tid; i < n; i += tg) y[i] = T((float(x[i]) - m) - logs);
  } else {
    for (uint i = tid; i < n; i += tg) y[i] = T(exp(float(x[i]) - m) / s);
  }
}
)msl";

struct Params {
  uint32_t n;
  uint32_t log_softmax;
};

absl::Status Softmax(stream_executor::Stream* stream, xffi::AnyBuffer x,
                     xffi::Result<xffi::AnyBuffer> y, bool log_softmax,
                     int64_t row_length) {
  auto dims = x.dimensions();
  if (dims.empty() || dims.back() != row_length ||
      x.element_type() != y->element_type() ||
      x.element_count() != y->element_count()) {
    return absl::InvalidArgumentError(
        "metal$softmax: operand/result shape mismatch");
  }
  xla::PrimitiveType type = x.element_type();
  if (type != xla::F32 && type != xla::F16 && type != xla::BF16) {
    return absl::UnimplementedError("metal$softmax: f32/f16/bf16 only");
  }
  absl::StatusOr<std::string> tname = MslTypeName(type);
  if (!tname.ok()) return tname.status();
  const uint64_t n = static_cast<uint64_t>(row_length);
  if (n == 0 || x.element_count() == 0) return absl::OkStatus();
  const uint64_t rows = x.element_count() / n;
  if (rows > 0xffffffffu || n > 0xffffffffu) {
    return absl::UnimplementedError("metal$softmax: too large");
  }
  std::string msl = absl::StrReplaceAll(kSoftmaxMsl, {{"$T", *tname}});
  const bool vec = n % 4 == 0;
  // Each thread handles 4 elements per iteration; round to whole simdgroups.
  uint32_t threads = static_cast<uint32_t>((n + 3) / 4);
  threads = std::clamp<uint32_t>((threads + 31) / 32 * 32, 32, 1024);
  Params p{static_cast<uint32_t>(n), log_softmax ? 1u : 0u};
  return LaunchMsl(stream, msl, vec ? "softmax_v" : "softmax_s",
                   {x.untyped_data(), y->untyped_data()}, p,
                   rt::Dim3{static_cast<uint32_t>(rows), 1, 1},
                   rt::Dim3{threads, 1, 1});
}

}  // namespace

XLA_FFI_DEFINE_HANDLER(kMetalSoftmax, Softmax,
                       xffi::Ffi::Bind()
                           .Ctx<xffi::Stream>()
                           .Arg<xffi::AnyBuffer>()
                           .Ret<xffi::AnyBuffer>()
                           .Attr<bool>("log")
                           .Attr<int64_t>("row_length"));

XLA_FFI_REGISTER_HANDLER(xffi::GetXlaFfiApi(), "metal$softmax", "METAL",
                         kMetalSoftmax);

}  // namespace ffi
}  // namespace metal_pjrt
