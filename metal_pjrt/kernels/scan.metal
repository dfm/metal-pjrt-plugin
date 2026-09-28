// Inclusive scan over rows (metal$scan, ffi/scan_ffi.cc). One threadgroup per
// row; 4 elements per thread per chunk, simdgroup shuffles, a threadgroup
// pass over the simdgroup totals and a carry between chunks. The host
// substitutes the types, op and identity (the $-placeholders). Params must
// match the C++ struct there.
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
