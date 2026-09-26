#include "metal_pjrt_plugin/stream_executor/metal_command_buffer.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "metal_pjrt_plugin/runtime/metal_runtime.h"
#include "metal_pjrt_plugin/stream_executor/metal_executor.h"
#include "metal_pjrt_plugin/stream_executor/metal_kernel.h"
#include "metal_pjrt_plugin/stream_executor/metal_stream.h"
#include "xla/stream_executor/bit_pattern.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream.h"

namespace stream_executor {
namespace metal {

namespace rt = metal_pjrt::rt;

MetalCommandBuffer::MetalCommandBuffer(MetalExecutor* executor, Mode mode)
    : executor_(executor), mode_(mode), list_(executor->device()) {}

absl::Status MetalCommandBuffer::CheckState(State expected,
                                            absl::string_view op) const {
  if (state_ != expected) {
    return absl::FailedPreconditionError(absl::StrFormat(
        "Metal command buffer: %s requires state %v but the buffer is in "
        "state %v",
        op, expected, state_));
  }
  return absl::OkStatus();
}

absl::StatusOr<const MetalCommandBuffer::MetalCommand*>
MetalCommandBuffer::Cast(const Command* command, absl::string_view op) const {
  const auto* c = dynamic_cast<const MetalCommand*>(command);
  if (c == nullptr || c->index >= list_.size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Metal command buffer: %s got a command that was not recorded into "
        "this buffer",
        op));
  }
  return c;
}

const CommandBuffer::Command* MetalCommandBuffer::Track(size_t index) {
  commands_.push_back(std::make_unique<MetalCommand>(index));
  return commands_.back().get();
}

absl::Status MetalCommandBuffer::Unimplemented(absl::string_view what) const {
  return absl::UnimplementedError(absl::StrCat(
      "Metal command buffers do not support ", what,
      "; only kernel launches, device copies, memsets and empty commands are "
      "recorded (enable only the FUSION command type)"));
}

absl::Status MetalCommandBuffer::ToLaunch(
    const ThreadDim& threads, const BlockDim& blocks,
    const std::optional<ClusterDim>& cluster_dims, const Kernel& kernel,
    const KernelArgs& args, LaunchParams& out) {
  const auto* metal_kernel = dynamic_cast<const MetalKernel*>(&kernel);
  if (metal_kernel == nullptr) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Metal command buffer: kernel ", kernel.name(),
        " was not loaded by the Metal executor"));
  }
  ABSL_RETURN_IF_ERROR(
      MetalKernel::CheckClusterDims(kernel.name(), cluster_dims));
  ABSL_RETURN_IF_ERROR(metal_kernel->PackArgs(args, out.args, out.shared_bytes));
  out.kernel = metal_kernel->rt_kernel();
  out.threadgroups = rt::Dim3{static_cast<uint32_t>(blocks.x),
                              static_cast<uint32_t>(blocks.y),
                              static_cast<uint32_t>(blocks.z)};
  out.threads = rt::Dim3{static_cast<uint32_t>(threads.x),
                         static_cast<uint32_t>(threads.y),
                         static_cast<uint32_t>(threads.z)};
  return absl::OkStatus();
}

// --- Commands ---------------------------------------------------------------

absl::StatusOr<const CommandBuffer::Command*>
MetalCommandBuffer::CreateEmptyCmd(absl::Span<const Command* const>,
                                   StreamPriority) {
  ABSL_RETURN_IF_ERROR(CheckState(State::kCreate, "CreateEmptyCmd"));
  return Track(list_.AddEmpty());
}

absl::StatusOr<const CommandBuffer::Command*> MetalCommandBuffer::CreateLaunch(
    const ThreadDim& threads, const BlockDim& blocks,
    const std::optional<ClusterDim>& cluster_dims, const Kernel& kernel,
    const KernelArgs& args, absl::Span<const Command* const>, StreamPriority) {
  ABSL_RETURN_IF_ERROR(CheckState(State::kCreate, "CreateLaunch"));
  LaunchParams p;
  ABSL_RETURN_IF_ERROR(ToLaunch(threads, blocks, cluster_dims, kernel, args, p));
  ABSL_ASSIGN_OR_RETURN(size_t index,
                        list_.AddLaunch(p.kernel, p.threadgroups, p.threads,
                                        p.args, p.shared_bytes));
  return Track(index);
}

absl::Status MetalCommandBuffer::UpdateLaunch(
    const Command* command, const ThreadDim& threads, const BlockDim& blocks,
    const std::optional<ClusterDim>& cluster_dims, const Kernel& kernel,
    const KernelArgs& args) {
  ABSL_RETURN_IF_ERROR(CheckState(State::kUpdate, "UpdateLaunch"));
  ABSL_ASSIGN_OR_RETURN(const MetalCommand* c, Cast(command, "UpdateLaunch"));
  LaunchParams p;
  ABSL_RETURN_IF_ERROR(ToLaunch(threads, blocks, cluster_dims, kernel, args, p));
  return list_.UpdateLaunch(c->index, p.kernel, p.threadgroups, p.threads,
                            p.args, p.shared_bytes);
}

absl::StatusOr<const CommandBuffer::Command*>
MetalCommandBuffer::CreateChildCommand(const CommandBuffer&,
                                       absl::Span<const Command* const>) {
  return Unimplemented("nested command buffers");
}

absl::Status MetalCommandBuffer::UpdateChildCommand(const Command*,
                                                    const CommandBuffer&) {
  return Unimplemented("nested command buffers");
}

absl::StatusOr<const CommandBuffer::Command*>
MetalCommandBuffer::CreateChildCommand(
    absl::AnyInvocable<absl::Status(CommandBuffer*)>,
    absl::Span<const Command* const>) {
  return Unimplemented("nested command buffers");
}

absl::Status MetalCommandBuffer::UpdateChildCommand(
    const Command*, absl::AnyInvocable<absl::Status(CommandBuffer*)>) {
  return Unimplemented("nested command buffers");
}

absl::StatusOr<const CommandBuffer::Command*>
MetalCommandBuffer::CreateMemcpyD2D(DeviceAddressBase* dst,
                                    const DeviceAddressBase& src,
                                    uint64_t size,
                                    absl::Span<const Command* const>) {
  ABSL_RETURN_IF_ERROR(CheckState(State::kCreate, "CreateMemcpyD2D"));
  ABSL_ASSIGN_OR_RETURN(size_t index,
                        list_.AddCopy(dst->opaque(), src.opaque(), size));
  return Track(index);
}

absl::Status MetalCommandBuffer::UpdateMemcpyD2D(const Command* command,
                                                 DeviceAddressBase* dst,
                                                 const DeviceAddressBase& src,
                                                 uint64_t size) {
  ABSL_RETURN_IF_ERROR(CheckState(State::kUpdate, "UpdateMemcpyD2D"));
  ABSL_ASSIGN_OR_RETURN(const MetalCommand* c, Cast(command, "UpdateMemcpyD2D"));
  return list_.UpdateCopy(c->index, dst->opaque(), src.opaque(), size);
}

absl::StatusOr<const CommandBuffer::Command*>
MetalCommandBuffer::CreateMemcpyD2H(void*, const DeviceAddressBase&, uint64_t,
                                    absl::Span<const Command* const>) {
  return Unimplemented("device-to-host copies");
}

absl::Status MetalCommandBuffer::UpdateMemcpyD2H(const Command*, void*,
                                                 const DeviceAddressBase&,
                                                 uint64_t) {
  return Unimplemented("device-to-host copies");
}

absl::StatusOr<const CommandBuffer::Command*>
MetalCommandBuffer::CreateMemcpyH2D(DeviceAddressBase*, const void*, uint64_t,
                                    absl::Span<const Command* const>) {
  return Unimplemented("host-to-device copies");
}

absl::Status MetalCommandBuffer::UpdateMemcpyH2D(const Command*,
                                                 DeviceAddressBase*,
                                                 const void*, uint64_t) {
  return Unimplemented("host-to-device copies");
}

absl::StatusOr<const CommandBuffer::Command*> MetalCommandBuffer::CreateMemset(
    DeviceAddressBase* dst, BitPattern bit_pattern, size_t num_elements,
    absl::Span<const Command* const>) {
  ABSL_RETURN_IF_ERROR(CheckState(State::kCreate, "CreateMemset"));
  const size_t width = bit_pattern.GetElementSize();
  ABSL_ASSIGN_OR_RETURN(
      size_t index,
      list_.AddFill(dst->opaque(), bit_pattern.GetPatternBroadcastedToUint32(),
                    static_cast<int>(width), num_elements * width));
  return Track(index);
}

absl::Status MetalCommandBuffer::UpdateMemset(const Command* command,
                                              DeviceAddressBase* dst,
                                              const BitPattern& bit_pattern,
                                              size_t num_elements) {
  ABSL_RETURN_IF_ERROR(CheckState(State::kUpdate, "UpdateMemset"));
  ABSL_ASSIGN_OR_RETURN(const MetalCommand* c, Cast(command, "UpdateMemset"));
  const size_t width = bit_pattern.GetElementSize();
  return list_.UpdateFill(c->index, dst->opaque(),
                          bit_pattern.GetPatternBroadcastedToUint32(),
                          static_cast<int>(width), num_elements * width);
}

absl::StatusOr<const CommandBuffer::Command*>
MetalCommandBuffer::CreateDnnGraphCommand(dnn::DnnGraph&, Stream&,
                                          absl::Span<DeviceAddressBase>,
                                          absl::Span<const Command* const>) {
  return Unimplemented("DNN graph commands");
}

absl::Status MetalCommandBuffer::UpdateDnnGraphCommand(
    const Command*, dnn::DnnGraph&, Stream&, absl::Span<DeviceAddressBase>) {
  return Unimplemented("DNN graph commands");
}

absl::StatusOr<const CommandBuffer::Command*> MetalCommandBuffer::CreateCase(
    DeviceAddress<int32_t>, std::vector<CreateCommands>,
    absl::Span<const Command* const>) {
  return Unimplemented("conditional (case) commands");
}

absl::StatusOr<const CommandBuffer::Command*> MetalCommandBuffer::CreateCase(
    DeviceAddress<bool>, std::vector<CreateCommands>,
    absl::Span<const Command* const>) {
  return Unimplemented("conditional (case) commands");
}

absl::Status MetalCommandBuffer::UpdateCase(const Command*,
                                            DeviceAddress<int32_t>,
                                            std::vector<UpdateCommands>) {
  return Unimplemented("conditional (case) commands");
}

absl::Status MetalCommandBuffer::UpdateCase(const Command*, DeviceAddress<bool>,
                                            std::vector<UpdateCommands>) {
  return Unimplemented("conditional (case) commands");
}

absl::StatusOr<const CommandBuffer::Command*> MetalCommandBuffer::CreateWhile(
    DeviceAddress<bool>, CreateCommands, CreateCommands,
    absl::Span<const Command* const>) {
  return Unimplemented("while commands");
}

absl::Status MetalCommandBuffer::UpdateWhile(const Command*, DeviceAddress<bool>,
                                             UpdateCommands, UpdateCommands) {
  return Unimplemented("while commands");
}

absl::StatusOr<const CommandBuffer::Command*> MetalCommandBuffer::CreateHost(
    absl::AnyInvocable<void()>, absl::Span<const Command* const>) {
  return Unimplemented("host callback commands");
}

absl::Status MetalCommandBuffer::Trace(
    Stream*, absl::AnyInvocable<absl::Status(Stream* stream)>) {
  return Unimplemented("stream tracing");
}

// --- State ------------------------------------------------------------------

absl::Status MetalCommandBuffer::SetPriority(StreamPriority) {
  return absl::OkStatus();  // one queue per stream; nothing to prioritize
}

absl::Status MetalCommandBuffer::Submit(Stream* stream) {
  ABSL_RETURN_IF_ERROR(CheckState(State::kFinalized, "Submit"));
  if (mode_ != Mode::kPrimary) {
    return absl::FailedPreconditionError(
        "Metal command buffer: only primary command buffers can be submitted");
  }
  auto* metal_stream = dynamic_cast<MetalStream*>(stream);
  if (metal_stream == nullptr) {
    return absl::InvalidArgumentError(
        "Metal command buffer: Submit needs a Metal stream");
  }
  return metal_stream->rt_stream()->Replay(list_);
}

absl::Status MetalCommandBuffer::Finalize() {
  if (state_ == State::kFinalized) {
    return CheckState(State::kCreate, "Finalize");
  }
  state_ = State::kFinalized;
  return absl::OkStatus();
}

absl::Status MetalCommandBuffer::Update() {
  ABSL_RETURN_IF_ERROR(CheckState(State::kFinalized, "Update"));
  state_ = State::kUpdate;
  return absl::OkStatus();
}

std::string MetalCommandBuffer::ToString() const {
  return absl::StrFormat("MetalCommandBuffer(%s, state %v): %s",
                         mode_ == Mode::kPrimary ? "primary" : "nested",
                         state_, list_.ToString());
}

}  // namespace metal
}  // namespace stream_executor
