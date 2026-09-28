// "xla.gpu.ext.cub_sort_keys" / "xla.gpu.ext.cub_sort_pairs": XLA's
// SortRewriter targets (CUB DeviceRadixSort on CUDA), as an MSL LSD radix
// sort. Mirrors xla/stream_executor/cuda/cub_sort_kernel_cuda.cc:
//
//   operands: keys_in[, values_in]; results: keys_out[, values_out], scratch
//   attributes: descending (bool), batch_size (i64)
//
// Keys and values are row-major [batch_size, n]; each row is sorted
// independently, stably in both directions (equal keys keep their input
// order), by key bits as CUB orders them: unsigned as is, signed with the
// sign bit flipped, floats in total order (negative: all bits flipped,
// positive: sign flipped) except that -0 sorts as +0. Values are raw bits.
// The instantiate stage returns the scratch size (int64_t state), which
// EstimateCubSortScratchSize reads at compile time.
//
// 4-bit digits (16 buckets), 256 threads x 8 items per tile. Every pass ranks
// a tile stably in threadgroup memory: per-thread digit counts laid out
// [digit][thread], one exclusive scan, rank = scanned count + position among
// the thread's own items. Two paths:
//  - rows of at most one tile (2048): one threadgroup per row runs all the
//    passes in threadgroup memory (sort_small), no scratch;
//  - longer rows: per pass, sort_hist (per-tile digit counts, laid out
//    [row][digit][tile]), sort_scan (exclusive scan per row) and sort_scatter
//    (each item to its row offset), ping-ponging between the outputs and
//    scratch copies of keys and values; the input is never written. Scratch:
//    [keys | values | counts], each 256-byte aligned.
//
// GPU safety: every kernel loop is bounded by the key width or the sizes in
// Params; scatter destinations are checked against the row length; the
// execute stage recomputes the scratch layout and refuses, before encoding
// anything, when the scratch buffer is smaller.
#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "metal_pjrt_plugin/ffi/metal_ffi.h"
#include "xla/backends/gpu/ffi.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"
#include "xla/primitive_util.h"

namespace metal_pjrt {
namespace ffi {
namespace {

namespace xffi = ::xla::ffi;

constexpr char kSortMsl[] = R"msl(
#include <metal_stdlib>
using namespace metal;
typedef @KEYT@ K;
typedef @VALT@ V;
#define KBITS @KBITS@
#define KIND @KIND@
#define HAS_VALUES @HASV@

constant constexpr uint kThreads = 256;
constant constexpr uint kSimds = kThreads / 32;
constant constexpr uint kItems = 8;
constant constexpr uint kTile = kThreads * kItems;
constant constexpr uint kRadix = 16;

struct Params { uint n; uint tiles; uint shift; uint descending; };

// Digit of key bits k at `shift`, in CUB's order (see the file comment).
inline uint Digit(K k, uint shift, bool desc) {
  const K sign = K(K(1) << (KBITS - 1));
#if KIND == 1
  k = K(k ^ sign);
#elif KIND == 2
  if (k == sign) k = K(0);
  k = (k & sign) != 0 ? K(~k) : K(k ^ sign);
#endif
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
  for (uint shift = 0; shift < KBITS; shift += 4) {
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
#if HAS_VALUES
      vals_out[base + j] = vals_in[base + idx[i]];
#endif
    }
  }
}

// Per-tile digit counts: hist[(row * kRadix + digit) * tiles + tile].
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
#if HAS_VALUES
  V v[kItems];
#endif
  uint d[kItems], rank[kItems];
  for (uint i = 0; i < kItems; ++i) {
    uint j = tile * kTile + tid * kItems + i;
    bool valid = j < n;
    k[i] = valid ? keys_in[base + j] : K(0);
#if HAS_VALUES
    v[i] = valid ? vals_in[base + j] : V(0);
#endif
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
#if HAS_VALUES
      vals_out[base + dest] = v[i];
#endif
    }
  }
}
)msl";

constexpr uint32_t kThreads = 256;
constexpr uint64_t kTile = 2048;  // kThreads * kItems in the MSL
constexpr int kDigitBits = 4;

struct Params {
  uint32_t n;
  uint32_t tiles;
  uint32_t shift;
  uint32_t descending;
};

struct KeyTraits {
  const char* msl_type;
  int bits;
  int kind;  // 0 unsigned, 1 signed, 2 float
};

absl::StatusOr<KeyTraits> GetKeyTraits(xla::PrimitiveType t) {
  switch (t) {
    case xla::U8: return KeyTraits{"uchar", 8, 0};
    case xla::S8: return KeyTraits{"uchar", 8, 1};
    case xla::U16: return KeyTraits{"ushort", 16, 0};
    case xla::S16: return KeyTraits{"ushort", 16, 1};
    case xla::F16:
    case xla::BF16: return KeyTraits{"ushort", 16, 2};
    case xla::U32: return KeyTraits{"uint", 32, 0};
    case xla::S32: return KeyTraits{"uint", 32, 1};
    case xla::F32: return KeyTraits{"uint", 32, 2};
    case xla::U64: return KeyTraits{"ulong", 64, 0};
    case xla::S64: return KeyTraits{"ulong", 64, 1};
    case xla::F64: return KeyTraits{"ulong", 64, 2};
    default:
      return absl::UnimplementedError(absl::StrCat(
          "Metal radix sort: unsupported key type ",
          xla::primitive_util::LowercasePrimitiveTypeName(t)));
  }
}

absl::StatusOr<const char*> ValueMslType(xla::PrimitiveType t) {
  if (xla::primitive_util::IsArrayType(t) && t != xla::PRED) {
    switch (xla::primitive_util::BitWidth(t)) {
      case 8: return "uchar";
      case 16: return "ushort";
      case 32: return "uint";
      case 64: return "ulong";
      default: break;
    }
  }
  return absl::UnimplementedError(absl::StrCat(
      "Metal radix sort: unsupported value type ",
      xla::primitive_util::LowercasePrimitiveTypeName(t)));
}

uint64_t Align256(uint64_t x) { return (x + 255) / 256 * 256; }

// Everything the handler derives from the buffers and attributes. Computed
// the same way at instantiate (for the scratch estimate) and at execute (to
// check the scratch buffer before encoding).
struct Plan {
  KeyTraits key;
  const char* value_type = "uint";
  bool has_values = false;
  uint64_t key_bytes = 0, value_bytes = 0;
  uint64_t total = 0, batch = 0, n = 0, tiles = 0;
  bool small = true;
  // Scratch layout (large path).
  uint64_t alt_values_offset = 0, hist_offset = 0, scratch_bytes = 0;
};

absl::StatusOr<Plan> MakePlan(const xffi::AnyBuffer& keys_in,
                              const xffi::AnyBuffer& keys_out,
                              const xffi::AnyBuffer* values_in,
                              const xffi::AnyBuffer* values_out,
                              int64_t batch_size) {
  Plan plan;
  absl::StatusOr<KeyTraits> key = GetKeyTraits(keys_in.element_type());
  if (!key.ok()) return key.status();
  plan.key = *key;
  if (keys_out.element_type() != keys_in.element_type() ||
      keys_out.element_count() != keys_in.element_count()) {
    return absl::InvalidArgumentError(
        "Metal radix sort: keys_in/keys_out mismatch");
  }
  plan.total = keys_in.element_count();
  plan.key_bytes = plan.key.bits / 8;
  if (values_in != nullptr) {
    absl::StatusOr<const char*> vt = ValueMslType(values_in->element_type());
    if (!vt.ok()) return vt.status();
    if (values_out->element_type() != values_in->element_type() ||
        values_in->element_count() != plan.total ||
        values_out->element_count() != plan.total) {
      return absl::InvalidArgumentError(
          "Metal radix sort: values do not match the keys");
    }
    plan.has_values = true;
    plan.value_type = *vt;
    plan.value_bytes =
        xla::primitive_util::ByteWidth(values_in->element_type());
  }
  if (plan.total == 0) return plan;
  if (batch_size <= 0 || plan.total % batch_size != 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Metal radix sort: batch_size ", batch_size, " does not divide ",
        plan.total, " elements"));
  }
  // 32-bit offsets and counts in the kernels.
  if (plan.total > (uint64_t{1} << 31)) {
    return absl::UnimplementedError(
        "Metal radix sort: more than 2^31 elements");
  }
  plan.batch = batch_size;
  plan.n = plan.total / plan.batch;
  plan.small = plan.n <= kTile;
  plan.tiles = (plan.n + kTile - 1) / kTile;
  if (!plan.small) {
    plan.alt_values_offset = Align256(plan.total * plan.key_bytes);
    plan.hist_offset =
        plan.alt_values_offset +
        (plan.has_values ? Align256(plan.total * plan.value_bytes) : 0);
    plan.scratch_bytes = plan.hist_offset + Align256(plan.batch * plan.tiles *
                                                     16 * sizeof(uint32_t));
  }
  return plan;
}

struct SortKernels {
  const rt::Kernel* small = nullptr;
  const rt::Kernel* hist = nullptr;
  const rt::Kernel* scan = nullptr;
  const rt::Kernel* scatter = nullptr;
};

// From the device's kernel cache (compiled on first use).
absl::StatusOr<SortKernels> GetKernels(rt::Device* device, const Plan& plan) {
  std::string msl = absl::StrReplaceAll(
      kSortMsl, {{"@KBITS@", absl::StrCat(plan.key.bits)},
                 {"@KIND@", absl::StrCat(plan.key.kind)},
                 {"@HASV@", plan.has_values ? "1" : "0"},
                 {"@KEYT@", plan.key.msl_type},
                 {"@VALT@", plan.value_type}});
  SortKernels k;
  for (auto [out, name] :
       {std::make_pair(&k.small, "sort_small"),
        std::make_pair(&k.hist, "sort_hist"),
        std::make_pair(&k.scan, "sort_scan"),
        std::make_pair(&k.scatter, "sort_scatter")}) {
    absl::StatusOr<const rt::Kernel*> kernel = device->GetKernel(msl, name);
    if (!kernel.ok()) return kernel.status();
    // The kernels assume exactly kThreads threads in 32-wide SIMD groups.
    if ((*kernel)->max_total_threads_per_threadgroup() < kThreads ||
        (*kernel)->thread_execution_width() != 32) {
      return absl::UnimplementedError(absl::StrCat(
          "Metal radix sort: ", name, " cannot run 256 threads in 32-wide "
          "SIMD groups on this device"));
    }
    *out = *kernel;
  }
  return k;
}

absl::StatusOr<std::unique_ptr<int64_t>> Instantiate(
    const xffi::AnyBuffer& keys_in, const xffi::AnyBuffer& keys_out,
    const xffi::AnyBuffer* values_in, const xffi::AnyBuffer* values_out,
    int64_t batch_size) {
  absl::StatusOr<Plan> plan =
      MakePlan(keys_in, keys_out, values_in, values_out, batch_size);
  if (!plan.ok()) return plan.status();
  // Compile now (the compiler's estimate calls this), not on first execute.
  absl::StatusOr<rt::Device*> device = DefaultMetalDevice();
  if (!device.ok()) return device.status();
  if (absl::StatusOr<SortKernels> k = GetKernels(*device, *plan); !k.ok()) {
    return k.status();
  }
  return std::make_unique<int64_t>(
      std::max<int64_t>(plan->scratch_bytes, 1));
}

absl::Status Execute(stream_executor::Stream* stream,
                     const xffi::AnyBuffer& keys_in,
                     const xffi::AnyBuffer& keys_out,
                     const xffi::AnyBuffer* values_in,
                     const xffi::AnyBuffer* values_out,
                     const xffi::Result<xffi::BufferR1<xla::U8>>& scratch,
                     bool descending, int64_t batch_size) {
  absl::StatusOr<Plan> plan =
      MakePlan(keys_in, keys_out, values_in, values_out, batch_size);
  if (!plan.ok()) return plan.status();
  // Refuse before encoding: the kernels would write past a short scratch.
  if (plan->scratch_bytes > scratch->size_bytes()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Metal radix sort: scratch buffer has ", scratch->size_bytes(),
        " bytes, needs ", plan->scratch_bytes));
  }
  if (plan->total == 0) return absl::OkStatus();
  absl::StatusOr<MetalContext> ctx = GetMetalContext(stream);
  if (!ctx.ok()) return ctx.status();
  absl::StatusOr<SortKernels> k = GetKernels(ctx->device, *plan);
  if (!k.ok()) return k.status();

  const void* kin = keys_in.untyped_data();
  void* kout = keys_out.untyped_data();
  // Keys-only kernels never touch the value buffers; bind the keys instead.
  const void* vin = values_in ? values_in->untyped_data() : kin;
  void* vout = values_out ? values_out->untyped_data() : kout;
  Params p{static_cast<uint32_t>(plan->n), static_cast<uint32_t>(plan->tiles),
           0, descending ? 1u : 0u};
  const rt::Dim3 threads{kThreads, 1, 1};
  if (plan->small) {
    return LaunchKernel(ctx->stream, *k->small, {kin, kout, vin, vout}, p,
                        rt::Dim3{static_cast<uint32_t>(plan->batch), 1, 1},
                        threads);
  }
  auto* base = static_cast<uint8_t*>(scratch->untyped_data());
  void* alt_keys = base;
  void* alt_vals = plan->has_values ? base + plan->alt_values_offset : alt_keys;
  void* hist = base + plan->hist_offset;
  const rt::Dim3 tiles{static_cast<uint32_t>(plan->batch * plan->tiles), 1, 1};
  const int passes = plan->key.bits / kDigitBits;  // even: 2, 4, 8 or 16
  const void* src_k = kin;
  const void* src_v = vin;
  for (int pass = 0; pass < passes; ++pass) {
    // The last pass lands in the outputs; so does every other one before it.
    const bool to_out = (passes - 1 - pass) % 2 == 0;
    void* dst_k = to_out ? kout : alt_keys;
    void* dst_v = to_out ? vout : alt_vals;
    p.shift = pass * kDigitBits;
    if (absl::Status s = LaunchKernel(ctx->stream, *k->hist, {src_k, hist}, p,
                                      tiles, threads);
        !s.ok()) {
      return s;
    }
    if (absl::Status s = LaunchKernel(
            ctx->stream, *k->scan, {hist}, p,
            rt::Dim3{static_cast<uint32_t>(plan->batch), 1, 1}, threads);
        !s.ok()) {
      return s;
    }
    if (absl::Status s =
            LaunchKernel(ctx->stream, *k->scatter,
                         {src_k, dst_k, src_v, dst_v, hist}, p, tiles, threads);
        !s.ok()) {
      return s;
    }
    src_k = dst_k;
    src_v = dst_v;
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<int64_t>> KeysInstantiate(
    xffi::AnyBuffer keys_in, xffi::Result<xffi::AnyBuffer> keys_out,
    xffi::Result<xffi::BufferR1<xla::U8>> scratch, bool descending,
    int64_t batch_size) {
  return Instantiate(keys_in, *keys_out, nullptr, nullptr, batch_size);
}

absl::Status KeysExecute(stream_executor::Stream* stream,
                         xffi::AnyBuffer keys_in,
                         xffi::Result<xffi::AnyBuffer> keys_out,
                         xffi::Result<xffi::BufferR1<xla::U8>> scratch,
                         bool descending, int64_t batch_size) {
  return Execute(stream, keys_in, *keys_out, nullptr, nullptr, scratch,
                 descending, batch_size);
}

absl::StatusOr<std::unique_ptr<int64_t>> PairsInstantiate(
    xffi::AnyBuffer keys_in, xffi::AnyBuffer values_in,
    xffi::Result<xffi::AnyBuffer> keys_out,
    xffi::Result<xffi::AnyBuffer> values_out,
    xffi::Result<xffi::BufferR1<xla::U8>> scratch, bool descending,
    int64_t batch_size) {
  return Instantiate(keys_in, *keys_out, &values_in, &*values_out,
                     batch_size);
}

absl::Status PairsExecute(stream_executor::Stream* stream,
                          xffi::AnyBuffer keys_in, xffi::AnyBuffer values_in,
                          xffi::Result<xffi::AnyBuffer> keys_out,
                          xffi::Result<xffi::AnyBuffer> values_out,
                          xffi::Result<xffi::BufferR1<xla::U8>> scratch,
                          bool descending, int64_t batch_size) {
  return Execute(stream, keys_in, *keys_out, &values_in, &*values_out, scratch,
                 descending, batch_size);
}

}  // namespace

XLA_FFI_DEFINE_HANDLER(kCubSortKeysInstantiate, KeysInstantiate,
                       xffi::Ffi::BindInstantiate()
                           .Arg<xffi::AnyBuffer>()           // keys_in
                           .Ret<xffi::AnyBuffer>()           // keys_out
                           .Ret<xffi::BufferR1<xla::U8>>()   // scratch
                           .Attr<bool>("descending")
                           .Attr<int64_t>("batch_size"));

XLA_FFI_DEFINE_HANDLER(kCubSortKeysExecute, KeysExecute,
                       xffi::Ffi::Bind()
                           .Ctx<xffi::Stream>()
                           .Arg<xffi::AnyBuffer>()
                           .Ret<xffi::AnyBuffer>()
                           .Ret<xffi::BufferR1<xla::U8>>()
                           .Attr<bool>("descending")
                           .Attr<int64_t>("batch_size"));

XLA_FFI_DEFINE_HANDLER(kCubSortPairsInstantiate, PairsInstantiate,
                       xffi::Ffi::BindInstantiate()
                           .Arg<xffi::AnyBuffer>()           // keys_in
                           .Arg<xffi::AnyBuffer>()           // values_in
                           .Ret<xffi::AnyBuffer>()           // keys_out
                           .Ret<xffi::AnyBuffer>()           // values_out
                           .Ret<xffi::BufferR1<xla::U8>>()   // scratch
                           .Attr<bool>("descending")
                           .Attr<int64_t>("batch_size"));

XLA_FFI_DEFINE_HANDLER(kCubSortPairsExecute, PairsExecute,
                       xffi::Ffi::Bind()
                           .Ctx<xffi::Stream>()
                           .Arg<xffi::AnyBuffer>()
                           .Arg<xffi::AnyBuffer>()
                           .Ret<xffi::AnyBuffer>()
                           .Ret<xffi::AnyBuffer>()
                           .Ret<xffi::BufferR1<xla::U8>>()
                           .Attr<bool>("descending")
                           .Attr<int64_t>("batch_size"));

XLA_FFI_REGISTER_HANDLER(xffi::GetXlaFfiApi(), kCubSortKeysTarget,
                         kMetalFfiPlatform,
                         {/*instantiate=*/kCubSortKeysInstantiate,
                          /*prepare=*/nullptr, /*initialize=*/nullptr,
                          /*execute=*/kCubSortKeysExecute});

XLA_FFI_REGISTER_HANDLER(xffi::GetXlaFfiApi(), kCubSortPairsTarget,
                         kMetalFfiPlatform,
                         {/*instantiate=*/kCubSortPairsInstantiate,
                          /*prepare=*/nullptr, /*initialize=*/nullptr,
                          /*execute=*/kCubSortPairsExecute});

}  // namespace ffi
}  // namespace metal_pjrt
