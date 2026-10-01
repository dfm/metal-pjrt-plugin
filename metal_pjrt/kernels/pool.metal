// Copyright 2026 The metal-pjrt-plugin Authors
// SPDX-License-Identifier: Apache-2.0

// Max-pool backward for non-overlapping windows (metal$pool_max_bwd,
// ffi/pool.cc): XLA's select-and-scatter with a GE select and an add
// scatter, window == stride, no padding or dilation. x is viewed as
// [A, D1, D2, B] with the window k1 x k2 on the middle dims (the host
// collapses the unwindowed dims around them). One thread per window and V
// consecutive elements of B: it reads the window once, selects per lane,
// and writes init + dy at the selected element and init at the others.
// Threads past the pooled extent (D / k rounded down) write init over the
// partial windows VALID pooling drops. Every element of x and dy is read
// once and every element of dx written once; no atomics.
//
// The selection is XLA's (SelectAndScatterExpander's reduce-window): the
// window's elements in row-major order, the first taken, then each next one
// replaces the current unless current >= next. So the first maximum wins,
// a NaN replaces anything, and a NaN is replaced by the next element.
// Params must match the C++ struct in ffi/pool.cc. Instantiated at the end
// as pool_max_bwd_<type>_v<V>.
#include <metal_stdlib>
using namespace metal;

struct PoolParams {
  uint a, d1, d2, b;  // x as [a, d1, d2, b]
  uint k1, k2;        // the window
  uint p1, p2;        // dy's d1, d2: d / k rounded down
  uint w1;            // windows along d1, partial ones included
  float init;         // the scatter's initial value
};

template <typename T, int V>
struct Lanes {
  typedef vec<T, V> type;
};
template <typename T>
struct Lanes<T, 1> {
  typedef T type;
};

// Lane l of a scalar (V = 1) or a vector.
template <typename T>
inline T lane(T v, int) { return v; }
template <typename T>
inline T lane(vec<T, 4> v, int l) { return v[l]; }
template <typename T>
inline void set_lane(thread T& v, int, T x) { v = x; }
template <typename T>
inline void set_lane(thread vec<T, 4>& v, int l, T x) { v[l] = x; }

// Offset (in units of V elements) of x[ia, r, c, bv].
inline uint pool_at(constant PoolParams& prm, uint ia, uint r, uint c,
                    uint bv, uint v) {
  return (((ia * prm.d1 + r) * prm.d2 + c) * prm.b + bv) / v;
}

template <typename T, int V>
kernel void pool_max_bwd(const device T* x [[buffer(0)]],
                         const device T* dy [[buffer(1)]],
                         device T* dx [[buffer(2)]],
                         constant PoolParams& prm [[buffer(3)]],
                         uint3 gid [[thread_position_in_grid]]) {
  typedef typename Lanes<T, V>::type L;
  const uint bv = gid.x * V;  // first lane's offset in b
  const uint o2 = gid.y;
  if (bv >= prm.b || o2 * prm.k2 >= prm.d2 || gid.z >= prm.a * prm.w1) {
    return;
  }
  const uint ia = gid.z / prm.w1, o1 = gid.z % prm.w1;
  const uint r0 = o1 * prm.k1, c0 = o2 * prm.k2;
  const L init = L(T(prm.init));
  device L* out = reinterpret_cast<device L*>(dx);
  if (o1 >= prm.p1 || o2 >= prm.p2) {
    // A partial window VALID pooling drops: nothing selected.
    for (uint r = r0; r < min(r0 + prm.k1, prm.d1); ++r) {
      for (uint c = c0; c < min(c0 + prm.k2, prm.d2); ++c) {
        out[pool_at(prm, ia, r, c, bv, V)] = init;
      }
    }
    return;
  }
  const device L* in = reinterpret_cast<const device L*>(x);
  float cur[V];
  uint sel[V];
  {
    const L v = in[pool_at(prm, ia, r0, c0, bv, V)];
    for (int l = 0; l < V; ++l) {
      cur[l] = float(lane(v, l));
      sel[l] = 0;
    }
  }
  for (uint r = 0; r < prm.k1; ++r) {
    for (uint c = 0; c < prm.k2; ++c) {
      if (r == 0 && c == 0) continue;
      const L v = in[pool_at(prm, ia, r0 + r, c0 + c, bv, V)];
      for (int l = 0; l < V; ++l) {
        const float e = float(lane(v, l));
        if (!(cur[l] >= e)) {
          cur[l] = e;
          sel[l] = r * prm.k2 + c;
        }
      }
    }
  }
  const L g = reinterpret_cast<const device L*>(
      dy)[(((ia * prm.p1 + o1) * prm.p2 + o2) * prm.b + bv) / V];
  L hit = init;
  for (int l = 0; l < V; ++l) {
    set_lane(hit, l, T(prm.init + float(lane(g, l))));
  }
  for (uint r = 0; r < prm.k1; ++r) {
    for (uint c = 0; c < prm.k2; ++c) {
      const uint pos = r * prm.k2 + c;
      L o = init;
      for (int l = 0; l < V; ++l) {
        if (sel[l] == pos) set_lane(o, l, lane(hit, l));
      }
      out[pool_at(prm, ia, r0 + r, c0 + c, bv, V)] = o;
    }
  }
}

#define instantiate_pool_max_bwd(tname, T, V)                            \
  template [[host_name("pool_max_bwd_" tname "_v" #V)]] kernel void    \
  pool_max_bwd<T, V>(const device T* x [[buffer(0)]],                  \
                     const device T* dy [[buffer(1)]],                 \
                     device T* dx [[buffer(2)]],                       \
                     constant PoolParams& prm [[buffer(3)]],           \
                     uint3 gid [[thread_position_in_grid]]);

instantiate_pool_max_bwd("float", float, 1)
instantiate_pool_max_bwd("float", float, 4)
instantiate_pool_max_bwd("half", half, 1)
instantiate_pool_max_bwd("half", half, 4)
instantiate_pool_max_bwd("bfloat", bfloat, 1)
instantiate_pool_max_bwd("bfloat", bfloat, 4)
