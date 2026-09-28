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
//
// The kernel (kernels/scan.metal, one instantiation per type and op) is
// looked up once per call site, at the FFI instantiate stage (ScanState);
// execution only encodes the dispatch.
#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "metal_pjrt/ffi/metal_ffi.h"
#include "metal_pjrt/kernels/scan.metal.h"
#include "xla/backends/gpu/ffi.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"

namespace metal_pjrt {
namespace ffi {
namespace {

namespace xffi = ::xla::ffi;

struct Params {
  uint32_t n;
  uint32_t reverse;
};

// The kernel of kernels/scan.metal for this element type and op.
absl::StatusOr<std::string> ScanFunction(xla::PrimitiveType type,
                                         absl::string_view op) {
  absl::StatusOr<std::string> tname = MslTypeName(type);
  if (!tname.ok()) return tname.status();
  if (op != "add" && op != "mul" && op != "max" && op != "min") {
    return absl::InvalidArgumentError(
        absl::StrCat("metal$scan: unknown op ", op));
  }
  return absl::StrCat("scan_", op, "_", *tname);
}

struct ScanState {
  std::string function;
  // Compiled at instantiation for DefaultMetalDevice().
  rt::Device* device = nullptr;
  const rt::Kernel* kernel = nullptr;
};

absl::Status CheckShapes(const xffi::AnyBuffer& x, const xffi::AnyBuffer& y,
                         int64_t row_length) {
  auto dims = x.dimensions();
  if (dims.empty() || dims.back() != row_length ||
      x.element_type() != y.element_type() ||
      x.element_count() != y.element_count()) {
    return absl::InvalidArgumentError(
        "metal$scan: operand/result shape mismatch");
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<ScanState>> Instantiate(
    xffi::AnyBuffer x, xffi::Result<xffi::AnyBuffer> y, absl::string_view op,
    bool reverse, int64_t row_length) {
  if (absl::Status s = CheckShapes(x, *y, row_length); !s.ok()) return s;
  absl::StatusOr<std::string> function = ScanFunction(x.element_type(), op);
  if (!function.ok()) return function.status();
  auto state = std::make_unique<ScanState>();
  state->function = *std::move(function);
  absl::StatusOr<rt::Device*> device = DefaultMetalDevice();
  if (!device.ok()) return device.status();
  absl::StatusOr<const rt::Kernel*> kernel =
      (*device)->GetKernel(kernels::kScanMsl, state->function);
  if (!kernel.ok()) return kernel.status();
  state->device = *device;
  state->kernel = *kernel;
  return state;
}

absl::Status Scan(stream_executor::Stream* stream, xffi::AnyBuffer x,
                  xffi::Result<xffi::AnyBuffer> y, absl::string_view op,
                  bool reverse, int64_t row_length, ScanState* state) {
  if (absl::Status s = CheckShapes(x, *y, row_length); !s.ok()) return s;
  absl::StatusOr<MetalContext> ctx = GetMetalContext(stream);
  if (!ctx.ok()) return ctx.status();
  const rt::Kernel* kernel = state->kernel;
  if (ctx->device != state->device) {
    absl::StatusOr<const rt::Kernel*> k =
        ctx->device->GetKernel(kernels::kScanMsl, state->function);
    if (!k.ok()) return k.status();
    kernel = *k;
  }
  const uint64_t n = static_cast<uint64_t>(row_length);
  if (n == 0 || x.element_count() == 0) return absl::OkStatus();
  const uint64_t rows = x.element_count() / n;
  // The kernel's uint chunk loop (start += 4 * threads) wraps, and never
  // ends, for n near 2^32; MetalScanRewriter leaves such scans to XLA.
  if (rows > 0xffffffffu || n > (uint64_t{1} << 31)) {
    return absl::UnimplementedError("metal$scan: too large");
  }
  // 4 elements per thread per chunk; 256 threads measured as fast as 512
  // or 1024 on M3 for 4096-long rows, and keeps more threadgroups resident.
  uint32_t threads = static_cast<uint32_t>((n + 3) / 4);
  threads = std::clamp<uint32_t>((threads + 31) / 32 * 32, 32, 256);
  Params p{static_cast<uint32_t>(n), reverse ? 1u : 0u};
  return LaunchKernel(ctx->stream, *kernel,
                      {x.untyped_data(), y->untyped_data()}, p,
                      rt::Dim3{static_cast<uint32_t>(rows), 1, 1},
                      rt::Dim3{threads, 1, 1});
}

}  // namespace

XLA_FFI_DEFINE_HANDLER(kMetalScanInstantiate, Instantiate,
                       xffi::Ffi::BindInstantiate()
                           .Arg<xffi::AnyBuffer>()
                           .Ret<xffi::AnyBuffer>()
                           .Attr<absl::string_view>("op")
                           .Attr<bool>("reverse")
                           .Attr<int64_t>("row_length"));

XLA_FFI_DEFINE_HANDLER(kMetalScan, Scan,
                       xffi::Ffi::Bind()
                           .Ctx<xffi::Stream>()
                           .Arg<xffi::AnyBuffer>()
                           .Ret<xffi::AnyBuffer>()
                           .Attr<absl::string_view>("op")
                           .Attr<bool>("reverse")
                           .Attr<int64_t>("row_length")
                           .Ctx<xffi::State<ScanState>>());

XLA_FFI_REGISTER_HANDLER(xffi::GetXlaFfiApi(), "metal$scan", "METAL",
                         {/*instantiate=*/kMetalScanInstantiate,
                          /*prepare=*/nullptr, /*initialize=*/nullptr,
                          /*execute=*/kMetalScan});

}  // namespace ffi
}  // namespace metal_pjrt
