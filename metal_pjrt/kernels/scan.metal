// Inclusive scan over rows (metal$scan, ffi/scan_ffi.cc). One threadgroup per
// row; 4 elements per thread per chunk, simdgroup shuffles, a threadgroup
// pass over the simdgroup totals and a carry between chunks. f16/bf16
// accumulate in f32; s32 add/mul wrap. Params must match the C++ struct
// there. Instantiated at the end as scan_<op>_<type>.
#include <metal_stdlib>
using namespace metal;
struct Params { uint n; uint reverse; };

// The ops, on the accumulator type A. XLA semantics: max/min propagate NaN.
struct AddF {
  typedef float A;
  static A op(A a, A b) { return a + b; }
  static A identity() { return 0.0f; }
};
struct AddI {
  typedef int A;
  static A op(A a, A b) {
    return as_type<int>(as_type<uint>(a) + as_type<uint>(b));
  }
  static A identity() { return 0; }
};
struct MulF {
  typedef float A;
  static A op(A a, A b) { return a * b; }
  static A identity() { return 1.0f; }
};
struct MulI {
  typedef int A;
  static A op(A a, A b) {
    return as_type<int>(as_type<uint>(a) * as_type<uint>(b));
  }
  static A identity() { return 1; }
};
struct MaxF {
  typedef float A;
  static A op(A a, A b) { return (isnan(a) || a > b) ? a : b; }
  static A identity() { return -INFINITY; }
};
struct MaxI {
  typedef int A;
  static A op(A a, A b) { return max(a, b); }
  static A identity() { return (-2147483647 - 1); }
};
struct MinF {
  typedef float A;
  static A op(A a, A b) { return (isnan(a) || a < b) ? a : b; }
  static A identity() { return INFINITY; }
};
struct MinI {
  typedef int A;
  static A op(A a, A b) { return min(a, b); }
  static A identity() { return 2147483647; }
};

// Inclusive scan across the simdgroup.
template <typename Op, typename A = typename Op::A>
inline A simd_inclusive(A x, uint lane) {
  for (ushort d = 1; d < 32; d <<= 1) {
    A o = simd_shuffle_up(x, d);
    if (lane >= d) x = Op::op(o, x);
  }
  return x;
}

template <typename T, typename Op>
kernel void scan(device const T* in [[buffer(0)]],
                 device T* out [[buffer(1)]],
                 constant Params& p [[buffer(2)]],
                 uint row [[threadgroup_position_in_grid]],
                 uint tid [[thread_position_in_threadgroup]],
                 uint tg [[threads_per_threadgroup]],
                 uint lane [[thread_index_in_simdgroup]],
                 uint sg [[simdgroup_index_in_threadgroup]],
                 uint nsg [[simdgroups_per_threadgroup]]) {
  typedef typename Op::A A;
  const A kIdentity = Op::identity();
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
    v[1] = Op::op(v[0], v[1]);
    v[2] = Op::op(v[1], v[2]);
    v[3] = Op::op(v[2], v[3]);
    A inc = simd_inclusive<Op>(v[3], lane);
    A exc = simd_shuffle_up(inc, 1);
    if (lane == 0) exc = kIdentity;
    if (lane == 31) partial[sg] = inc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
      A t = lane < nsg ? partial[lane] : kIdentity;
      A ti = simd_inclusive<Op>(t, lane);
      A te = simd_shuffle_up(ti, 1);
      if (lane == 0) te = kIdentity;
      if (lane < nsg) partial[lane] = te;
      if (lane == 31) partial[32] = ti;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    A prefix = Op::op(carry, Op::op(partial[sg], exc));
    for (uint k = 0; k < 4; ++k) {
      uint j = j0 + k;
      if (j < n) y[rev ? n - 1 - j : j] = T(Op::op(prefix, v[k]));
    }
    carry = Op::op(carry, partial[32]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

#define instantiate_scan(name, T, Op)             \
  template [[host_name("scan_" name)]] [[kernel]] \
  decltype(scan<T, Op>) scan<T, Op>;

instantiate_scan("add_float", float, AddF)
instantiate_scan("add_half", half, AddF)
instantiate_scan("add_bfloat", bfloat, AddF)
instantiate_scan("add_int", int, AddI)
instantiate_scan("mul_float", float, MulF)
instantiate_scan("mul_half", half, MulF)
instantiate_scan("mul_bfloat", bfloat, MulF)
instantiate_scan("mul_int", int, MulI)
instantiate_scan("max_float", float, MaxF)
instantiate_scan("max_half", half, MaxF)
instantiate_scan("max_bfloat", bfloat, MaxF)
instantiate_scan("max_int", int, MaxI)
instantiate_scan("min_float", float, MinF)
instantiate_scan("min_half", half, MinF)
instantiate_scan("min_bfloat", bfloat, MinF)
instantiate_scan("min_int", int, MinI)
