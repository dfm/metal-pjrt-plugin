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
#include <tuple>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "metal_pjrt/ffi/metal_ffi.h"
#include "metal_pjrt/kernels/cub_sort.metal.h"
#include "xla/backends/gpu/ffi.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"
#include "xla/primitive_util.h"

namespace metal_pjrt {
namespace ffi {
namespace {

namespace xffi = ::xla::ffi;

constexpr uint32_t kThreads = 256;
constexpr uint64_t kTile = 2048;  // kThreads * kItems in the MSL
constexpr int kDigitBits = 4;

struct Params {
  uint32_t n;
  uint32_t tiles;
  uint32_t shift;
  uint32_t descending;
};

// Key bits' storage type as named in the kernels' instantiations
// (kernels/cub_sort.metal): u8, u16, u32 or u64; values likewise.
struct KeyTraits {
  const char* bits_type;
  int bits;
  int kind;  // 0 unsigned, 1 signed, 2 float (the KIND function constant)
};

absl::StatusOr<KeyTraits> GetKeyTraits(xla::PrimitiveType t) {
  switch (t) {
    case xla::U8: return KeyTraits{"u8", 8, 0};
    case xla::S8: return KeyTraits{"u8", 8, 1};
    case xla::U16: return KeyTraits{"u16", 16, 0};
    case xla::S16: return KeyTraits{"u16", 16, 1};
    case xla::F16:
    case xla::BF16: return KeyTraits{"u16", 16, 2};
    case xla::U32: return KeyTraits{"u32", 32, 0};
    case xla::S32: return KeyTraits{"u32", 32, 1};
    case xla::F32: return KeyTraits{"u32", 32, 2};
    case xla::U64: return KeyTraits{"u64", 64, 0};
    case xla::S64: return KeyTraits{"u64", 64, 1};
    case xla::F64: return KeyTraits{"u64", 64, 2};
    default:
      return absl::UnimplementedError(absl::StrCat(
          "Metal radix sort: unsupported key type ",
          xla::primitive_util::LowercasePrimitiveTypeName(t)));
  }
}

absl::StatusOr<const char*> ValueBitsType(xla::PrimitiveType t) {
  if (xla::primitive_util::IsArrayType(t) && t != xla::PRED) {
    switch (xla::primitive_util::BitWidth(t)) {
      case 8: return "u8";
      case 16: return "u16";
      case 32: return "u32";
      case 64: return "u64";
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
  const char* value_type = "u32";  // any instantiation, when keys only
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
    absl::StatusOr<const char*> vt = ValueBitsType(values_in->element_type());
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
  // The function_constant indices of kernels/cub_sort.metal.
  const rt::FunctionConstant constants[] = {
      rt::FunctionConstant::Int(0, plan.key.kind),
      rt::FunctionConstant::Bool(1, plan.has_values)};
  const std::string kv =
      absl::StrCat(plan.key.bits_type, "_", plan.value_type);
  SortKernels k;
  for (auto [out, name, with_constants] :
       {std::make_tuple(&k.small, absl::StrCat("sort_small_", kv), true),
        std::make_tuple(&k.hist,
                        absl::StrCat("sort_hist_", plan.key.bits_type), true),
        std::make_tuple(&k.scan, std::string("sort_scan"), false),
        std::make_tuple(&k.scatter, absl::StrCat("sort_scatter_", kv),
                        true)}) {
    absl::StatusOr<const rt::Kernel*> kernel = device->GetKernel(
        kernels::kCubSortMsl, name,
        with_constants ? absl::Span<const rt::FunctionConstant>(constants)
                       : absl::Span<const rt::FunctionConstant>());
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
