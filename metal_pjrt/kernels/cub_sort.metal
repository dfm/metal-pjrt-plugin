// Copyright 2026 The jax-graft Authors
// SPDX-License-Identifier: Apache-2.0

// LSD radix sort for XLA's cub_sort_keys / cub_sort_pairs targets; see
// ffi/radix_sort.h for the algorithm, the key orders and the scratch
// layout. Params must match the C++ struct of the same name in
// ffi/radix_sort.cc.
//
// Kernels are templates over the key bits' storage type K (uchar, ushort,
// uint, ulong) and the value type V (the same four; raw bits), instantiated
// below; how the key bits order (KIND) and whether there are values are
// function constants.
#include <metal_stdlib>
using namespace metal;

// Indices match GetRadixSortKernels (ffi/radix_sort.cc).
constant int KIND [[function_constant(0)]];  // 0 unsigned, 1 signed, 2 float
constant bool HAS_VALUES [[function_constant(1)]];

constant constexpr uint kThreads = 256;
constant constexpr uint kSimds = kThreads / 32;
constant constexpr uint kItems = 8;
constant constexpr uint kTile = kThreads * kItems;
constant constexpr uint kRadix = 16;

struct Params { uint n; uint tiles; uint shift; uint descending; };

// Digit of key bits k at `shift`, in CUB's order (see the file comment).
template <typename K>
inline uint Digit(K k, uint shift, bool desc) {
  const K sign = K(K(1) << (sizeof(K) * 8 - 1));
  if (KIND == 1) {
    k = K(k ^ sign);
  } else if (KIND == 2) {
    if (k == sign) k = K(0);
    k = (k & sign) != 0 ? K(~k) : K(k ^ sign);
  }
  if (desc) k = K(~k);
  return uint(k >> shift) & (kRadix - 1);
}

// Stable ranks within the tile of the kItems digits each thread holds, the
// tile being in blocked order (thread t holds items t*kItems ...). Leaves
// counts[d * kThreads] = number of the tile's items with digit < d.
inline void Rank(thread const uint* d, thread uint* rank,
                 threadgroup ushort* counts, threadgroup uint* sums,
                 uint tid, uint lane, uint sg) {
  ulong packed = 0;  // kItems <= 15, so 4 bits per digit count
  uint before[kItems];
  for (uint i = 0; i < kItems; ++i) {
    uint s = 4 * d[i];
    before[i] = uint(packed >> s) & 15u;
    packed += 1ul << s;
  }
  for (uint r = 0; r < kRadix; ++r) {
    counts[r * kThreads + tid] = ushort((packed >> (4 * r)) & 15ul);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Exclusive scan of counts in [digit][thread] order; each thread owns
  // kRadix consecutive entries.
  threadgroup ushort* mine = counts + tid * kRadix;
  uint total = 0;
  for (uint r = 0; r < kRadix; ++r) total += mine[r];
  uint excl = simd_prefix_exclusive_sum(total);
  if (lane == 31) sums[sg] = excl + total;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    uint t = lane < kSimds ? sums[lane] : 0u;
    uint te = simd_prefix_exclusive_sum(t);
    if (lane < kSimds) sums[lane] = te;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint run = sums[sg] + excl;
  for (uint r = 0; r < kRadix; ++r) {
    uint c = mine[r];
    mine[r] = ushort(run);
    run += c;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = 0; i < kItems; ++i) {
    rank[i] = (counts[d[i] * kThreads + tid] + before[i]) & (kTile - 1);
  }
}

// One threadgroup per row of n <= kTile items: all passes in threadgroup
// memory, carrying each item's original position; values are gathered at the
// end.
template <typename K, typename V>
[[max_total_threads_per_threadgroup(256)]]
kernel void sort_small(device const K* keys_in [[buffer(0)]],
                       device K* keys_out [[buffer(1)]],
                       device const V* vals_in [[buffer(2)]],
                       device V* vals_out [[buffer(3)]],
                       constant Params& p [[buffer(4)]],
                       uint row [[threadgroup_position_in_grid]],
                       uint tid [[thread_position_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup K skeys[kTile];
  threadgroup ushort sidx[kTile];
  threadgroup ushort counts[kRadix * kThreads];
  threadgroup uint sums[kSimds];
  const uint n = p.n;
  const bool desc = p.descending != 0;
  const ulong base = ulong(row) * n;
  K k[kItems];
  uint idx[kItems];
  for (uint i = 0; i < kItems; ++i) {
    uint j = tid * kItems + i;
    k[i] = j < n ? keys_in[base + j] : K(0);
    idx[i] = j;
  }
  // Padding (original position >= n) takes the last digit, so it stays
  // behind every real item.
  for (uint shift = 0; shift < sizeof(K) * 8; shift += 4) {
    uint d[kItems], rank[kItems];
    for (uint i = 0; i < kItems; ++i) {
      d[i] = idx[i] < n ? Digit(k[i], shift, desc) : kRadix - 1;
    }
    Rank(d, rank, counts, sums, tid, lane, sg);
    for (uint i = 0; i < kItems; ++i) {
      skeys[rank[i]] = k[i];
      sidx[rank[i]] = ushort(idx[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < kItems; ++i) {
      k[i] = skeys[tid * kItems + i];
      idx[i] = sidx[tid * kItems + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint i = 0; i < kItems; ++i) {
    uint j = tid * kItems + i;
    if (j < n && idx[i] < n) {
      keys_out[base + j] = k[i];
      if (HAS_VALUES) vals_out[base + j] = vals_in[base + idx[i]];
    }
  }
}

// Per-tile digit counts: hist[(row * kRadix + digit) * tiles + tile].
template <typename K>
[[max_total_threads_per_threadgroup(256)]]
kernel void sort_hist(device const K* keys [[buffer(0)]],
                      device uint* hist [[buffer(1)]],
                      constant Params& p [[buffer(2)]],
                      uint g [[threadgroup_position_in_grid]],
                      uint tid [[thread_position_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
  threadgroup atomic_uint h[kRadix];
  if (tid < kRadix) atomic_store_explicit(&h[tid], 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint row = g / p.tiles, tile = g % p.tiles;
  const ulong base = ulong(row) * p.n;
  const bool desc = p.descending != 0;
  ulong packed = 0;
  for (uint i = 0; i < kItems; ++i) {
    uint j = tile * kTile + i * kThreads + tid;
    if (j < p.n) packed += 1ul << (4 * Digit(keys[base + j], p.shift, desc));
  }
  for (uint r = 0; r < kRadix; ++r) {
    uint c = simd_sum(uint((packed >> (4 * r)) & 15ul));
    if (lane == 0 && c != 0) {
      atomic_fetch_add_explicit(&h[r], c, memory_order_relaxed);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < kRadix) {
    hist[(ulong(row) * kRadix + tid) * p.tiles + tile] =
        atomic_load_explicit(&h[tid], memory_order_relaxed);
  }
}

// Exclusive scan of each row's kRadix * tiles counts, in place.
[[max_total_threads_per_threadgroup(256)]]
kernel void sort_scan(device uint* hist [[buffer(0)]],
                      constant Params& p [[buffer(1)]],
                      uint row [[threadgroup_position_in_grid]],
                      uint tid [[thread_position_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup uint sums[kSimds + 1];
  const uint m = kRadix * p.tiles;
  device uint* h = hist + ulong(row) * m;
  uint carry = 0;
  for (uint start = 0; start < m; start += kThreads) {
    uint j = start + tid;
    uint v = j < m ? h[j] : 0u;
    uint excl = simd_prefix_exclusive_sum(v);
    if (lane == 31) sums[sg] = excl + v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
      uint t = lane < kSimds ? sums[lane] : 0u;
      uint te = simd_prefix_exclusive_sum(t);
      if (lane < kSimds) sums[lane] = te;
      if (lane == kSimds - 1) sums[kSimds] = te + t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (j < m) h[j] = carry + sums[sg] + excl;
    carry += sums[kSimds];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// One pass: each tile's items to their place in the row, by digit.
template <typename K, typename V>
[[max_total_threads_per_threadgroup(256)]]
kernel void sort_scatter(device const K* keys_in [[buffer(0)]],
                         device K* keys_out [[buffer(1)]],
                         device const V* vals_in [[buffer(2)]],
                         device V* vals_out [[buffer(3)]],
                         device const uint* hist [[buffer(4)]],
                         constant Params& p [[buffer(5)]],
                         uint g [[threadgroup_position_in_grid]],
                         uint tid [[thread_position_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup ushort counts[kRadix * kThreads];
  threadgroup uint sums[kSimds];
  const uint n = p.n;
  const uint row = g / p.tiles, tile = g % p.tiles;
  const ulong base = ulong(row) * n;
  const bool desc = p.descending != 0;
  K k[kItems];
  V v[kItems];
  uint d[kItems], rank[kItems];
  for (uint i = 0; i < kItems; ++i) {
    uint j = tile * kTile + tid * kItems + i;
    bool valid = j < n;
    k[i] = valid ? keys_in[base + j] : K(0);
    if (HAS_VALUES) v[i] = valid ? vals_in[base + j] : V(0);
    d[i] = valid ? Digit(k[i], p.shift, desc) : kRadix - 1;
  }
  Rank(d, rank, counts, sums, tid, lane, sg);
  for (uint i = 0; i < kItems; ++i) {
    uint j = tile * kTile + tid * kItems + i;
    if (j >= n) continue;
    uint dest = hist[(ulong(row) * kRadix + d[i]) * p.tiles + tile] +
                rank[i] - counts[d[i] * kThreads];
    if (dest < n) {
      keys_out[base + dest] = k[i];
      if (HAS_VALUES) vals_out[base + dest] = v[i];
    }
  }
}

#define instantiate_sort_kv(kname, K, vname, V)                        \
  template [[host_name("sort_small_" #kname "_" #vname)]] [[kernel]]   \
  decltype(sort_small<K, V>) sort_small<K, V>;                         \
  template [[host_name("sort_scatter_" #kname "_" #vname)]] [[kernel]] \
  decltype(sort_scatter<K, V>) sort_scatter<K, V>;
#define instantiate_sort(kname, K)                                        \
  template [[host_name("sort_hist_" #kname)]] [[kernel]]                 \
  decltype(sort_hist<K>) sort_hist<K>;                                    \
  instantiate_sort_kv(kname, K, u8, uchar)                                \
  instantiate_sort_kv(kname, K, u16, ushort)                              \
  instantiate_sort_kv(kname, K, u32, uint)                                \
  instantiate_sort_kv(kname, K, u64, ulong)

instantiate_sort(u8, uchar)
instantiate_sort(u16, ushort)
instantiate_sort(u32, uint)
instantiate_sort(u64, ulong)
