// Copyright 2026 The metal-pjrt-plugin Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef METAL_PJRT_STREAM_EXECUTOR_METAL_STREAM_H_
#define METAL_PJRT_STREAM_EXECUTOR_METAL_STREAM_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <variant>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "metal_pjrt/runtime/metal_runtime.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_common.h"

namespace stream_executor {
namespace metal {

class MetalExecutor;

// A stream is one MTLCommandQueue (see runtime/metal_runtime.h for the
// command-buffer batching policy).
//
// Errors: any GPU failure, or failed host callback without an error_cb, is
// sticky for the device (see metal_runtime.h): BlockHostUntilDone returns it
// from then on, and a host callback with an error_cb enqueued after it does
// not run; the error_cb gets the error instead (XLA marks the buffers'
// definition events failed with it). One without an error_cb still runs.
// The StreamExecutor stream error state (ok()) is never set, as on CUDA
// (XLA CHECKs ok() on pooled streams).
class MetalStream : public StreamCommon {
 public:
  static absl::StatusOr<std::unique_ptr<MetalStream>> Create(
      MetalExecutor* executor,
      std::optional<std::variant<StreamPriority, int>> priority);
  ~MetalStream() override;

  // XLA names its streams (pjrt/se/local_device_state.cc); the transfer
  // streams ("Host-to-device", "Device-to-host #i", "Device-to-device #i")
  // wait on the host for their unsignaled waits before a command buffer is
  // committed (rt::Stream::set_waits_on_host). A transfer that waits for the
  // compute stream's backlog on the GPU counts that backlog against the
  // watchdog.
  void SetName(std::string name) override;
  absl::Status WaitFor(Stream* other) override;
  absl::Status WaitFor(Event* event) override;
  absl::Status RecordEvent(Event* event) override;

  absl::Status Memcpy(void* host_dst, const DeviceAddressBase& gpu_src,
                      uint64_t size) override;
  absl::Status Memcpy(DeviceAddressBase* gpu_dst, const void* host_src,
                      uint64_t size) override;
  absl::Status Memcpy(DeviceAddressBase* gpu_dst,
                      const DeviceAddressBase& gpu_src, uint64_t size) override;
  absl::Status MemZero(DeviceAddressBase* location, uint64_t size) override;
  absl::Status Memset32(DeviceAddressBase* location, uint32_t pattern,
                        uint64_t size) override;

  absl::Status BlockHostUntilDone() override;
  absl::Status DoHostCallbackWithStatus(
      absl::AnyInvocable<absl::Status() &&> callback) override;
  // With a non-null `error_cb`, a failing callback's status (or the
  // device's sticky error) goes to error_cb instead of becoming the device's
  // sticky error.
  absl::Status DoHostCallbackWithStatus(
      absl::AnyInvocable<absl::Status() &&> callback,
      absl::AnyInvocable<void(absl::Status) &&> error_cb) override;

  Stream::PlatformSpecificHandle platform_specific_handle() const override {
    return {rt_stream_.get()};
  }

  metal_pjrt::rt::Stream* rt_stream() const { return rt_stream_.get(); }

 private:
  MetalStream(MetalExecutor* executor,
              std::optional<std::variant<StreamPriority, int>> priority,
              std::unique_ptr<metal_pjrt::rt::Stream> rt_stream);

  MetalExecutor* executor_;
  std::unique_ptr<metal_pjrt::rt::Stream> rt_stream_;
};

}  // namespace metal
}  // namespace stream_executor

#endif  // METAL_PJRT_STREAM_EXECUTOR_METAL_STREAM_H_
