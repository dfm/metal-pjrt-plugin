// Copyright 2026 The jax-graft Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef METAL_PJRT_STREAM_EXECUTOR_METAL_EVENT_H_
#define METAL_PJRT_STREAM_EXECUTOR_METAL_EVENT_H_

#include <memory>

#include "absl/status/status.h"
#include "metal_pjrt/runtime/metal_runtime.h"
#include "xla/stream_executor/event.h"

namespace stream_executor {
namespace metal {

class MetalEvent : public Event {
 public:
  explicit MetalEvent(std::unique_ptr<metal_pjrt::rt::Event> event)
      : event_(std::move(event)) {}
  ~MetalEvent() override = default;

  // kError once the device has failed (the error is sticky, see
  // metal_runtime.h); Synchronize returns that error.
  Event::Status PollForStatus() override {
    absl::StatusOr<bool> done = event_->Poll();
    if (!done.ok()) return Event::Status::kError;
    return *done ? Event::Status::kComplete : Event::Status::kPending;
  }

  absl::Status Synchronize() override { return event_->WaitOnHost(); }

  metal_pjrt::rt::Event* rt_event() const { return event_.get(); }

 private:
  std::unique_ptr<metal_pjrt::rt::Event> event_;
};

}  // namespace metal
}  // namespace stream_executor

#endif  // METAL_PJRT_STREAM_EXECUTOR_METAL_EVENT_H_
