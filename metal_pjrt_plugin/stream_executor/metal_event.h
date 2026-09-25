#ifndef METAL_PJRT_PLUGIN_STREAM_EXECUTOR_METAL_EVENT_H_
#define METAL_PJRT_PLUGIN_STREAM_EXECUTOR_METAL_EVENT_H_

#include <memory>

#include "absl/status/status.h"
#include "metal_pjrt_plugin/runtime/metal_runtime.h"
#include "xla/stream_executor/event.h"

namespace stream_executor {
namespace metal {

class MetalEvent : public Event {
 public:
  explicit MetalEvent(std::unique_ptr<metal_pjrt::rt::Event> event)
      : event_(std::move(event)) {}
  ~MetalEvent() override = default;

  Event::Status PollForStatus() override {
    return event_->IsComplete() ? Event::Status::kComplete
                                : Event::Status::kPending;
  }

  absl::Status Synchronize() override { return event_->WaitOnHost(); }

  metal_pjrt::rt::Event* rt_event() const { return event_.get(); }

 private:
  std::unique_ptr<metal_pjrt::rt::Event> event_;
};

}  // namespace metal
}  // namespace stream_executor

#endif  // METAL_PJRT_PLUGIN_STREAM_EXECUTOR_METAL_EVENT_H_
