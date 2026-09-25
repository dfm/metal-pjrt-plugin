// "metal$scan": inclusive scan (cumsum/cumprod/cummax/cummin) over the minor
// dimension.
//
// Operand and result: same shape [..., n], row-major, f32/f16/bf16/s32.
// Attributes: op ("add" | "mul" | "max" | "min"), reverse (bool),
// row_length (i64, must equal the minor dim).
//
// One threadgroup per row. Each iteration covers a chunk of 4 * threads
// elements: every thread scans 4 consecutive elements in registers, the
// per-thread totals are scanned within the simdgroup (shuffles), simdgroup
// totals are scanned through threadgroup memory, and a running carry links
// the chunks. f16/bf16 accumulate in f32; s32 add/mul wrap.
#include <algorithm>
#include <cstdint>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "metal_pjrt_plugin/ffi/metal_ffi.h"
#include "xla/backends/gpu/ffi.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"

namespace metal_pjrt {
namespace ffi {
namespace {

namespace xffi = ::xla::ffi;

constexpr char kScanMsl[] = R"msl(
#include <metal_stdlib>
using namespace metal;
typedef $T T;
typedef $A A;
struct Params { uint n; uint reverse; };

inline A op(A a, A b) { $OP }
constant constexpr A kIdentity = $ID;

// Inclusive scan across the simdgroup.
inline A simd_inclusive(A x, uint lane) {
  for (ushort d = 1; d < 32; d <<= 1) {
    A o = simd_shuffle_up(x, d);
    if (lane >= d) x = op(o, x);
  }
  return x;
}

kernel void scan(device const T* in [[buffer(0)]],
                 device T* out [[buffer(1)]],
                 constant Params& p [[buffer(2)]],
                 uint row [[threadgroup_position_in_grid]],
                 uint tid [[thread_position_in_threadgroup]],
                 uint tg [[threads_per_threadgroup]],
                 uint lane [[thread_index_in_simdgroup]],
                 uint sg [[simdgroup_index_in_threadgroup]],
                 uint nsg [[simdgroups_per_threadgroup]]) {
  threadgroup A partial[33];
  const uint n = p.n;
  const bool rev = p.reverse != 0;
  device const T* x = in + (ulong)row * n;
  device T* y = out + (ulong)row * n;
  A carry = kIdentity;
  for (uint start = 0; start < n; start += tg * 4) {
    const uint j0 = start + tid * 4;
    A v[4];
    for (uint k = 0; k < 4; ++k) {
      uint j = j0 + k;
      v[k] = j < n ? A(x[rev ? n - 1 - j : j]) : kIdentity;
    }
    v[1] = op(v[0], v[1]);
    v[2] = op(v[1], v[2]);
    v[3] = op(v[2], v[3]);
    A inc = simd_inclusive(v[3], lane);
    A exc = simd_shuffle_up(inc, 1);
    if (lane == 0) exc = kIdentity;
    if (lane == 31) partial[sg] = inc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
      A t = lane < nsg ? partial[lane] : kIdentity;
      A ti = simd_inclusive(t, lane);
      A te = simd_shuffle_up(ti, 1);
      if (lane == 0) te = kIdentity;
      if (lane < nsg) partial[lane] = te;
      if (lane == 31) partial[32] = ti;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    A prefix = op(carry, op(partial[sg], exc));
    for (uint k = 0; k < 4; ++k) {
      uint j = j0 + k;
      if (j < n) y[rev ? n - 1 - j : j] = T(op(prefix, v[k]));
    }
    carry = op(carry, partial[32]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}
)msl";

struct Params {
  uint32_t n;
  uint32_t reverse;
};

absl::StatusOr<std::string> BuildScanMsl(xla::PrimitiveType type,
                                         absl::string_view op) {
  absl::StatusOr<std::string> tname = MslTypeName(type);
  if (!tname.ok()) return tname.status();
  const bool is_int = type == xla::S32;
  std::string acc = is_int ? "int" : "float";
  std::string body, identity;
  if (op == "add") {
    body = is_int ? "return as_type<int>(as_type<uint>(a) + as_type<uint>(b));"
                  : "return a + b;";
    identity = is_int ? "0" : "0.0f";
  } else if (op == "mul") {
    body = is_int ? "return as_type<int>(as_type<uint>(a) * as_type<uint>(b));"
                  : "return a * b;";
    identity = is_int ? "1" : "1.0f";
  } else if (op == "max") {
    // XLA semantics: NaN propagates.
    body = is_int ? "return max(a, b);"
                  : "return (isnan(a) || a > b) ? a : b;";
    identity = is_int ? "(-2147483647 - 1)" : "-INFINITY";
  } else if (op == "min") {
    body = is_int ? "return min(a, b);"
                  : "return (isnan(a) || a < b) ? a : b;";
    identity = is_int ? "2147483647" : "INFINITY";
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("metal$scan: unknown op ", op));
  }
  return absl::StrReplaceAll(kScanMsl, {{"$T", *tname},
                                        {"$A", acc},
                                        {"$OP", body},
                                        {"$ID", identity}});
}

absl::Status Scan(stream_executor::Stream* stream, xffi::AnyBuffer x,
                  xffi::Result<xffi::AnyBuffer> y, absl::string_view op,
                  bool reverse, int64_t row_length) {
  auto dims = x.dimensions();
  if (dims.empty() || dims.back() != row_length ||
      x.element_type() != y->element_type() ||
      x.element_count() != y->element_count()) {
    return absl::InvalidArgumentError(
        "metal$scan: operand/result shape mismatch");
  }
  absl::StatusOr<std::string> msl = BuildScanMsl(x.element_type(), op);
  if (!msl.ok()) return msl.status();
  const uint64_t n = static_cast<uint64_t>(row_length);
  if (n == 0 || x.element_count() == 0) return absl::OkStatus();
  const uint64_t rows = x.element_count() / n;
  if (rows > 0xffffffffu || n > 0xffffffffu) {
    return absl::UnimplementedError("metal$scan: too large");
  }
  // 4 elements per thread per chunk; 256 threads measured as fast as 512
  // or 1024 on M3 for 4096-long rows, and keeps more threadgroups resident.
  uint32_t threads = static_cast<uint32_t>((n + 3) / 4);
  threads = std::clamp<uint32_t>((threads + 31) / 32 * 32, 32, 256);
  Params p{static_cast<uint32_t>(n), reverse ? 1u : 0u};
  return LaunchMsl(stream, *msl, "scan", {x.untyped_data(), y->untyped_data()},
                   p, rt::Dim3{static_cast<uint32_t>(rows), 1, 1},
                   rt::Dim3{threads, 1, 1});
}

}  // namespace

XLA_FFI_DEFINE_HANDLER(kMetalScan, Scan,
                       xffi::Ffi::Bind()
                           .Ctx<xffi::Stream>()
                           .Arg<xffi::AnyBuffer>()
                           .Ret<xffi::AnyBuffer>()
                           .Attr<absl::string_view>("op")
                           .Attr<bool>("reverse")
                           .Attr<int64_t>("row_length"));

XLA_FFI_REGISTER_HANDLER(xffi::GetXlaFfiApi(), "metal$scan", "METAL",
                         kMetalScan);

}  // namespace ffi
}  // namespace metal_pjrt
