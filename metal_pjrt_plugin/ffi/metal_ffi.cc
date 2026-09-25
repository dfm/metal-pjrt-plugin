#include "metal_pjrt_plugin/ffi/metal_ffi.h"

#include <memory>
#include <mutex>
#include <string>
#include <tuple>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "metal_pjrt_plugin/stream_executor/metal_executor.h"
#include "metal_pjrt_plugin/stream_executor/metal_platform_id.h"
#include "xla/primitive_util.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"

namespace metal_pjrt {
namespace ffi {

absl::StatusOr<MetalContext> GetMetalContext(stream_executor::Stream* stream) {
  if (stream == nullptr) {
    return absl::InvalidArgumentError("Metal FFI handler: null stream");
  }
  stream_executor::StreamExecutor* executor = stream->parent();
  if (executor == nullptr || executor->GetPlatform() == nullptr ||
      executor->GetPlatform()->id() !=
          stream_executor::metal::kMetalPlatformId) {
    return absl::InvalidArgumentError(
        "Metal FFI handler invoked on a stream of another platform");
  }
  auto* metal_executor =
      static_cast<stream_executor::metal::MetalExecutor*>(executor);
  MetalContext ctx;
  ctx.stream =
      static_cast<rt::Stream*>(stream->platform_specific_handle().stream);
  ctx.device = metal_executor->device();
  if (ctx.stream == nullptr || ctx.device == nullptr) {
    return absl::InternalError("Metal FFI handler: stream has no runtime");
  }
  return ctx;
}

namespace {

struct KernelCache {
  std::mutex mu;
  // Key: (device, MSL source, function). Entries are never evicted, so
  // returned pointers stay valid.
  absl::flat_hash_map<std::tuple<const rt::Device*, std::string, std::string>,
                      std::unique_ptr<rt::Kernel>>
      map;
};

KernelCache& Cache() {
  static KernelCache* cache = new KernelCache();
  return *cache;
}

}  // namespace

absl::StatusOr<const rt::Kernel*> GetOrCreateKernel(
    rt::Device* device, const std::string& msl_source,
    const std::string& function) {
  KernelCache& cache = Cache();
  auto key = std::make_tuple(static_cast<const rt::Device*>(device),
                             msl_source, function);
  {
    std::lock_guard<std::mutex> lock(cache.mu);
    auto it = cache.map.find(key);
    if (it != cache.map.end()) return it->second.get();
  }
  absl::StatusOr<MTL::Library*> library = device->CompileLibrary(msl_source);
  if (!library.ok()) return library.status();
  absl::StatusOr<std::unique_ptr<rt::Kernel>> kernel =
      device->CreateKernel(*library, function);
  if (!kernel.ok()) return kernel.status();
  std::lock_guard<std::mutex> lock(cache.mu);
  auto [it, inserted] = cache.map.try_emplace(std::move(key), nullptr);
  if (inserted) it->second = std::move(*kernel);
  return it->second.get();
}

absl::StatusOr<std::string> MslTypeName(xla::PrimitiveType type) {
  switch (type) {
    case xla::F32:
      return std::string("float");
    case xla::F16:
      return std::string("half");
    case xla::BF16:
      return std::string("bfloat");
    case xla::S32:
      return std::string("int");
    default:
      return absl::UnimplementedError(
          absl::StrCat("Metal FFI: unsupported element type ",
                       xla::primitive_util::LowercasePrimitiveTypeName(type)));
  }
}

}  // namespace ffi
}  // namespace metal_pjrt
