#include "metal_pjrt_plugin/stream_executor/metal_stream.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <variant>

#include "absl/functional/any_invocable.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "metal_pjrt_plugin/runtime/metal_runtime.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "metal_pjrt_plugin/stream_executor/metal_event.h"
#include "metal_pjrt_plugin/stream_executor/metal_executor.h"

namespace stream_executor {
namespace metal {

namespace rt = metal_pjrt::rt;

namespace {
absl::Status ToAbsl(const rt::Status& s) {
  if (s.ok()) return absl::OkStatus();
  return absl::InternalError(s.message());
}
}  // namespace

absl::StatusOr<std::unique_ptr<MetalStream>> MetalStream::Create(
    MetalExecutor* executor,
    std::optional<std::variant<StreamPriority, int>> priority) {
  // Metal command queues have no priorities; accept and ignore.
  std::unique_ptr<rt::Stream> rt_stream;
  TF_RETURN_IF_ERROR(ToAbsl(executor->device()->CreateStream(&rt_stream)));
  return std::unique_ptr<MetalStream>(
      new MetalStream(executor, priority, std::move(rt_stream)));
}

MetalStream::MetalStream(
    MetalExecutor* executor,
    std::optional<std::variant<StreamPriority, int>> priority,
    std::unique_ptr<rt::Stream> rt_stream)
    : StreamCommon(executor, priority),
      executor_(executor),
      rt_stream_(std::move(rt_stream)) {}

MetalStream::~MetalStream() {
  rt_stream_->Synchronize();
  parent()->DeallocateStream(this);
}

absl::Status MetalStream::WaitFor(Stream* other) {
  auto* o = static_cast<MetalStream*>(other);
  return ToAbsl(rt_stream_->WaitForStream(o->rt_stream()));
}

absl::Status MetalStream::WaitFor(Event* event) {
  auto* e = static_cast<MetalEvent*>(event);
  return ToAbsl(rt_stream_->WaitForEvent(e->rt_event()));
}

absl::Status MetalStream::RecordEvent(Event* event) {
  auto* e = static_cast<MetalEvent*>(event);
  return ToAbsl(rt_stream_->RecordEvent(e->rt_event()));
}

absl::Status MetalStream::Memcpy(void* host_dst, const DeviceAddressBase& gpu_src,
                                 uint64_t size) {
  return ToAbsl(rt_stream_->MemcpyDeviceToHost(host_dst, gpu_src.opaque(), size));
}

absl::Status MetalStream::Memcpy(DeviceAddressBase* gpu_dst,
                                 const void* host_src, uint64_t size) {
  return ToAbsl(
      rt_stream_->MemcpyHostToDevice(gpu_dst->opaque(), host_src, size));
}

absl::Status MetalStream::Memcpy(DeviceAddressBase* gpu_dst,
                                 const DeviceAddressBase& gpu_src,
                                 uint64_t size) {
  return ToAbsl(rt_stream_->MemcpyDeviceToDevice(gpu_dst->opaque(),
                                                 gpu_src.opaque(), size));
}

absl::Status MetalStream::MemZero(DeviceAddressBase* location, uint64_t size) {
  return ToAbsl(rt_stream_->Memset8(location->opaque(), 0, size));
}

absl::Status MetalStream::Memset32(DeviceAddressBase* location,
                                   uint32_t pattern, uint64_t size) {
  if (size % 4 != 0) {
    return absl::InvalidArgumentError("Memset32 size must be a multiple of 4");
  }
  return ToAbsl(rt_stream_->Memset32(location->opaque(), pattern, size));
}

absl::Status MetalStream::BlockHostUntilDone() {
  return ToAbsl(rt_stream_->Synchronize());
}

absl::Status MetalStream::DoHostCallbackWithStatus(
    absl::AnyInvocable<absl::Status() &&> callback) {
  auto shared = std::make_shared<absl::AnyInvocable<absl::Status() &&>>(
      std::move(callback));
  return ToAbsl(rt_stream_->HostCallback([shared]() {
    absl::Status s = std::move(*shared)();
    if (!s.ok()) LOG(WARNING) << "Metal host callback failed: " << s;
  }));
}

}  // namespace metal
}  // namespace stream_executor
