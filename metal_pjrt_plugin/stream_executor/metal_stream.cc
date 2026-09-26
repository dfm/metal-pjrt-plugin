#include "metal_pjrt_plugin/stream_executor/metal_stream.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <variant>

#include "absl/functional/any_invocable.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "metal_pjrt_plugin/runtime/metal_runtime.h"
#include "metal_pjrt_plugin/stream_executor/metal_event.h"
#include "metal_pjrt_plugin/stream_executor/metal_executor.h"

namespace stream_executor {
namespace metal {

namespace rt = metal_pjrt::rt;

absl::StatusOr<std::unique_ptr<MetalStream>> MetalStream::Create(
    MetalExecutor* executor,
    std::optional<std::variant<StreamPriority, int>> priority) {
  // Metal command queues have no priorities; accept and ignore.
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<rt::Stream> rt_stream,
                        executor->device()->CreateStream());
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
  absl::Status s = rt_stream_->Synchronize();
  if (!s.ok()) {
    LOG(ERROR) << "Metal stream on device " << executor_->device_ordinal()
               << " destroyed with a pending error: " << s;
  }
  // Joins the runtime's host-callback worker while this object (whose
  // callbacks may call SetError) is still alive.
  rt_stream_.reset();
  parent()->DeallocateStream(this);
}

void MetalStream::SetError(absl::Status status) {
  {
    absl::MutexLock lock(&error_mu_);
    if (!error_.ok()) return;
    error_ = status;
  }
  CheckStatus(std::move(status));  // logs and sets StreamCommon's error state
}

absl::Status MetalStream::WaitFor(Stream* other) {
  auto* o = static_cast<MetalStream*>(other);
  return rt_stream_->WaitForStream(o->rt_stream());
}

absl::Status MetalStream::WaitFor(Event* event) {
  auto* e = static_cast<MetalEvent*>(event);
  return rt_stream_->WaitForEvent(e->rt_event());
}

absl::Status MetalStream::RecordEvent(Event* event) {
  auto* e = static_cast<MetalEvent*>(event);
  return rt_stream_->RecordEvent(e->rt_event());
}

absl::Status MetalStream::Memcpy(void* host_dst,
                                 const DeviceAddressBase& gpu_src,
                                 uint64_t size) {
  return rt_stream_->MemcpyDeviceToHost(host_dst, gpu_src.opaque(), size);
}

absl::Status MetalStream::Memcpy(DeviceAddressBase* gpu_dst,
                                 const void* host_src, uint64_t size) {
  return rt_stream_->MemcpyHostToDevice(gpu_dst->opaque(), host_src, size);
}

absl::Status MetalStream::Memcpy(DeviceAddressBase* gpu_dst,
                                 const DeviceAddressBase& gpu_src,
                                 uint64_t size) {
  return rt_stream_->MemcpyDeviceToDevice(gpu_dst->opaque(), gpu_src.opaque(),
                                          size);
}

absl::Status MetalStream::MemZero(DeviceAddressBase* location, uint64_t size) {
  return rt_stream_->Memset8(location->opaque(), 0, size);
}

absl::Status MetalStream::Memset32(DeviceAddressBase* location,
                                   uint32_t pattern, uint64_t size) {
  return rt_stream_->Memset32(location->opaque(), pattern, size);
}

absl::Status MetalStream::BlockHostUntilDone() {
  // As on CUDA, a GPU failure is returned, not recorded as the stream's
  // error state: the runtime stream recovers after reporting it (XLA CHECKs
  // ok() on pooled streams).
  ABSL_RETURN_IF_ERROR(rt_stream_->Synchronize());
  absl::MutexLock lock(&error_mu_);
  return error_;
}

absl::Status MetalStream::DoHostCallbackWithStatus(
    absl::AnyInvocable<absl::Status() &&> callback) {
  return DoHostCallbackWithStatus(std::move(callback), nullptr);
}

absl::Status MetalStream::DoHostCallbackWithStatus(
    absl::AnyInvocable<absl::Status() &&> callback,
    absl::AnyInvocable<void(absl::Status) &&> error_cb) {
  // rt::Stream takes a copyable std::function.
  struct Callbacks {
    absl::AnyInvocable<absl::Status() &&> callback;
    absl::AnyInvocable<void(absl::Status) &&> error_cb;
  };
  auto shared = std::make_shared<Callbacks>(
      Callbacks{std::move(callback), std::move(error_cb)});
  // The runtime passes an error (the callback's, or that of the work it is
  // ordered after, in which case the callback does not run) to on_error.
  std::function<void(absl::Status)> on_error;
  if (shared->error_cb) {
    on_error = [shared](absl::Status s) {
      std::move(shared->error_cb)(std::move(s));
    };
  }
  absl::Status enqueued = rt_stream_->HostCallback(
      [this, shared]() {
        absl::Status s = std::move(shared->callback)();
        if (!s.ok() && !shared->error_cb) {
          SetError(absl::Status(
              s.code(),
              absl::StrFormat("Metal host callback failed on device %d: %s",
                              executor_->device_ordinal(), s.message())));
        }
        return s;
      },
      std::move(on_error));
  if (!enqueued.ok() && shared->error_cb) {
    std::move(shared->error_cb)(enqueued);
  }
  return enqueued;
}

}  // namespace metal
}  // namespace stream_executor
