#include "metal_pjrt/ffi/scan.h"

#include <algorithm>
#include <cstdint>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "metal_pjrt/compiler/report_bug.h"
#include "metal_pjrt/kernels/scan.metal.h"
#include "metal_pjrt/runtime/kernel_launch.h"
#include "metal_pjrt/runtime/metal_runtime.h"

namespace metal_pjrt {
namespace ffi {
namespace {

struct Params {
  uint32_t n;
  uint32_t reverse;
};

}  // namespace

absl::StatusOr<ScanOp> ParseScanOp(absl::string_view op) {
  if (op == "add") return ScanOp::kAdd;
  if (op == "mul") return ScanOp::kMul;
  if (op == "max") return ScanOp::kMax;
  if (op == "min") return ScanOp::kMin;
  return absl::InvalidArgumentError(
      absl::StrCat("metal$scan: unknown op ", op));
}

std::string ScanFunction(ScanType type, ScanOp op) {
  const char* ops[] = {"add", "mul", "max", "min"};
  const char* types[] = {"float", "half", "bfloat", "int"};
  return absl::StrCat("scan_", ops[static_cast<int>(op)], "_",
                      types[static_cast<int>(type)]);
}

absl::StatusOr<const rt::Kernel*> GetScanKernel(rt::Device* device,
                                                ScanType type, ScanOp op) {
  return device->GetKernel(kernels::kScanMsl, ScanFunction(type, op));
}

absl::Status RunScan(rt::Device* device, rt::Stream* stream, const void* x,
                     void* y, uint64_t rows, uint64_t n, ScanType type,
                     ScanOp op, bool reverse, const rt::Kernel* kernel) {
  if (n == 0 || rows == 0) return absl::OkStatus();
  // The kernel's uint chunk loop (start += 4 * threads) wraps, and never
  // ends, for n near 2^32; MetalScanRewriter leaves such scans to XLA.
  if (rows > 0xffffffffu || n > (uint64_t{1} << 31)) {
    return absl::UnimplementedError(absl::StrCat(
        "Metal: scan of ", rows, " rows of ", n,
        " elements is too large (at most 2^32 - 1 rows of 2^31 elements)",
        metal_pjrt::kReportBug));
  }
  if (kernel == nullptr) {
    absl::StatusOr<const rt::Kernel*> k = GetScanKernel(device, type, op);
    if (!k.ok()) return k.status();
    kernel = *k;
  }
  // 4 elements per thread per chunk; 256 threads measured as fast as 512
  // or 1024 on M3 for 4096-long rows, and keeps more threadgroups resident.
  uint32_t threads = static_cast<uint32_t>((n + 3) / 4);
  threads = std::clamp<uint32_t>((threads + 31) / 32 * 32, 32, 256);
  Params p{static_cast<uint32_t>(n), reverse ? 1u : 0u};
  return rt::LaunchKernel(stream, *kernel, {x, y}, p,
                          rt::Dim3{static_cast<uint32_t>(rows), 1, 1},
                          rt::Dim3{threads, 1, 1});
}

}  // namespace ffi
}  // namespace metal_pjrt
