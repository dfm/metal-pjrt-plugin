// "xla.gpu.ext.cub_sort_keys" / "xla.gpu.ext.cub_sort_pairs": XLA's
// SortRewriter targets (CUB DeviceRadixSort on CUDA), as an MSL LSD radix
// sort (radix_sort.h). Mirrors xla/stream_executor/cuda/cub_sort_kernel_cuda.cc:
//
//   operands: keys_in[, values_in]; results: keys_out[, values_out], scratch
//   attributes: descending (bool), batch_size (i64)
//
// Keys and values are row-major [batch_size, n]; each row is sorted
// independently and stably, in CUB's key order (radix_sort.h). The
// instantiate stage returns the scratch size (int64_t state), which
// EstimateCubSortScratchSize reads at compile time; the execute stage
// recomputes the plan and RunRadixSort refuses a shorter scratch buffer.
#include <algorithm>
#include <cstdint>
#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "metal_pjrt/ffi/metal_ffi.h"
#include "metal_pjrt/ffi/radix_sort.h"
#include "xla/backends/gpu/ffi.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"
#include "xla/primitive_util.h"

namespace metal_pjrt {
namespace ffi {
namespace {

namespace xffi = ::xla::ffi;

struct KeyTraits {
  int bits;
  KeyKind kind;
};

absl::StatusOr<KeyTraits> GetKeyTraits(xla::PrimitiveType t) {
  switch (t) {
    case xla::U8: return KeyTraits{8, KeyKind::kUnsigned};
    case xla::S8: return KeyTraits{8, KeyKind::kSigned};
    case xla::U16: return KeyTraits{16, KeyKind::kUnsigned};
    case xla::S16: return KeyTraits{16, KeyKind::kSigned};
    case xla::F16:
    case xla::BF16: return KeyTraits{16, KeyKind::kFloat};
    case xla::U32: return KeyTraits{32, KeyKind::kUnsigned};
    case xla::S32: return KeyTraits{32, KeyKind::kSigned};
    case xla::F32: return KeyTraits{32, KeyKind::kFloat};
    case xla::U64: return KeyTraits{64, KeyKind::kUnsigned};
    case xla::S64: return KeyTraits{64, KeyKind::kSigned};
    case xla::F64: return KeyTraits{64, KeyKind::kFloat};
    default:
      return absl::UnimplementedError(absl::StrCat(
          "Metal radix sort: unsupported key type ",
          xla::primitive_util::LowercasePrimitiveTypeName(t)));
  }
}

absl::StatusOr<int> ValueBits(xla::PrimitiveType t) {
  if (xla::primitive_util::IsArrayType(t) && t != xla::PRED) {
    switch (xla::primitive_util::BitWidth(t)) {
      case 8:
      case 16:
      case 32:
      case 64:
        return xla::primitive_util::BitWidth(t);
      default: break;
    }
  }
  return absl::UnimplementedError(absl::StrCat(
      "Metal radix sort: unsupported value type ",
      xla::primitive_util::LowercasePrimitiveTypeName(t)));
}

// The buffers' types and counts, checked, then PlanRadixSort.
absl::StatusOr<RadixSortPlan> MakePlan(const xffi::AnyBuffer& keys_in,
                                       const xffi::AnyBuffer& keys_out,
                                       const xffi::AnyBuffer* values_in,
                                       const xffi::AnyBuffer* values_out,
                                       int64_t batch_size) {
  RadixSortSpec spec;
  absl::StatusOr<KeyTraits> key = GetKeyTraits(keys_in.element_type());
  if (!key.ok()) return key.status();
  spec.key_bits = key->bits;
  spec.key_kind = key->kind;
  if (keys_out.element_type() != keys_in.element_type() ||
      keys_out.element_count() != keys_in.element_count()) {
    return absl::InvalidArgumentError(
        "Metal radix sort: keys_in/keys_out mismatch");
  }
  spec.total = keys_in.element_count();
  if (values_in != nullptr) {
    absl::StatusOr<int> vb = ValueBits(values_in->element_type());
    if (!vb.ok()) return vb.status();
    if (values_out->element_type() != values_in->element_type() ||
        values_in->element_count() != spec.total ||
        values_out->element_count() != spec.total) {
      return absl::InvalidArgumentError(
          "Metal radix sort: values do not match the keys");
    }
    spec.value_bits = *vb;
  }
  spec.batch_size = batch_size;
  return PlanRadixSort(spec);
}

absl::StatusOr<std::unique_ptr<int64_t>> Instantiate(
    const xffi::AnyBuffer& keys_in, const xffi::AnyBuffer& keys_out,
    const xffi::AnyBuffer* values_in, const xffi::AnyBuffer* values_out,
    int64_t batch_size) {
  absl::StatusOr<RadixSortPlan> plan =
      MakePlan(keys_in, keys_out, values_in, values_out, batch_size);
  if (!plan.ok()) return plan.status();
  // Compile now (the compiler's estimate calls this), not on first execute.
  absl::StatusOr<rt::Device*> device = DefaultMetalDevice();
  if (!device.ok()) return device.status();
  if (absl::StatusOr<RadixSortKernels> k = GetRadixSortKernels(*device, *plan);
      !k.ok()) {
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
  absl::StatusOr<RadixSortPlan> plan =
      MakePlan(keys_in, keys_out, values_in, values_out, batch_size);
  if (!plan.ok()) return plan.status();
  absl::StatusOr<MetalContext> ctx = GetMetalContext(stream);
  if (!ctx.ok()) return ctx.status();
  return RunRadixSort(ctx->device, ctx->stream, *plan, keys_in.untyped_data(),
                      keys_out.untyped_data(),
                      values_in ? values_in->untyped_data() : nullptr,
                      values_out ? values_out->untyped_data() : nullptr,
                      scratch->untyped_data(), scratch->size_bytes(),
                      descending);
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
