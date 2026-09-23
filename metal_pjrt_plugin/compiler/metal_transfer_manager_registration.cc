#include <memory>

#include "metal_pjrt_plugin/stream_executor/metal_platform_id.h"
#include "xla/service/gpu/gpu_transfer_manager.h"
#include "xla/service/transfer_manager.h"

namespace {

std::unique_ptr<xla::TransferManager> CreateMetalTransferManager() {
  return std::make_unique<xla::gpu::GpuTransferManager>(
      stream_executor::metal::kMetalPlatformId, /*pointer_size=*/8);
}

bool InitModule() {
  xla::TransferManager::RegisterTransferManager(
      stream_executor::metal::kMetalPlatformId, &CreateMetalTransferManager);
  return true;
}
bool module_initialized = InitModule();

}  // namespace
