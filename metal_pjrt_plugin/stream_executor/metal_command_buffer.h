#ifndef METAL_PJRT_PLUGIN_STREAM_EXECUTOR_METAL_COMMAND_BUFFER_H_
#define METAL_PJRT_PLUGIN_STREAM_EXECUTOR_METAL_COMMAND_BUFFER_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "metal_pjrt_plugin/runtime/metal_runtime.h"
#include "xla/stream_executor/bit_pattern.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"

namespace stream_executor {
namespace metal {

class MetalExecutor;

// StreamExecutor command buffer on Metal, as software replay: commands are
// recorded into a runtime CommandList with their buffer arguments resolved,
// and Submit encodes them into the stream in recording order. XLA records
// commands in thunk order, which is a topological order of the dependency
// graph, so serial replay honors every dependency; the dependency lists are
// therefore accepted and ignored. Supported: kernel launches, device-to-device
// copies, memsets and empty commands (what XLA's FUSION command type needs).
// Nested command buffers, conditionals, host callbacks, host transfers and
// DNN graphs report Unimplemented; XLA does not fall back when recording
// fails, so only command types this class supports may be enabled (see
// ApplyMetalDefaults in compiler/metal_compiler.cc).
class MetalCommandBuffer : public CommandBuffer {
 public:
  MetalCommandBuffer(MetalExecutor* executor, Mode mode);

  absl::StatusOr<const Command*> CreateEmptyCmd(
      absl::Span<const Command* const> dependencies,
      StreamPriority priority) override;

  absl::StatusOr<const Command*> CreateLaunch(
      const ThreadDim& threads, const BlockDim& blocks,
      const std::optional<ClusterDim>& cluster_dims, const Kernel& kernel,
      const KernelArgs& args, absl::Span<const Command* const> dependencies,
      StreamPriority priority) override;
  absl::Status UpdateLaunch(const Command* command, const ThreadDim& threads,
                            const BlockDim& blocks,
                            const std::optional<ClusterDim>& cluster_dims,
                            const Kernel& kernel,
                            const KernelArgs& args) override;

  absl::StatusOr<const Command*> CreateChildCommand(
      const CommandBuffer& nested,
      absl::Span<const Command* const> dependencies) override;
  absl::Status UpdateChildCommand(const Command* command,
                                  const CommandBuffer& nested) override;
  absl::StatusOr<const Command*> CreateChildCommand(
      absl::AnyInvocable<absl::Status(CommandBuffer*)> record_fn,
      absl::Span<const Command* const> dependencies) override;
  absl::Status UpdateChildCommand(
      const Command* command,
      absl::AnyInvocable<absl::Status(CommandBuffer*)> update_fn) override;

  absl::StatusOr<const Command*> CreateMemcpyD2D(
      DeviceAddressBase* dst, const DeviceAddressBase& src, uint64_t size,
      absl::Span<const Command* const> dependencies) override;
  absl::Status UpdateMemcpyD2D(const Command* command, DeviceAddressBase* dst,
                               const DeviceAddressBase& src,
                               uint64_t size) override;
  absl::StatusOr<const Command*> CreateMemcpyD2H(
      void* dst, const DeviceAddressBase& src, uint64_t size,
      absl::Span<const Command* const> dependencies) override;
  absl::Status UpdateMemcpyD2H(const Command* command, void* dst,
                               const DeviceAddressBase& src,
                               uint64_t size) override;
  absl::StatusOr<const Command*> CreateMemcpyH2D(
      DeviceAddressBase* dst, const void* src, uint64_t size,
      absl::Span<const Command* const> dependencies) override;
  absl::Status UpdateMemcpyH2D(const Command* command, DeviceAddressBase* dst,
                               const void* src, uint64_t size) override;

  absl::StatusOr<const Command*> CreateMemset(
      DeviceAddressBase* dst, BitPattern bit_pattern, size_t num_elements,
      absl::Span<const Command* const> dependencies) override;
  absl::Status UpdateMemset(const Command* command, DeviceAddressBase* dst,
                            const BitPattern& bit_pattern,
                            size_t num_elements) override;

  absl::StatusOr<const Command*> CreateDnnGraphCommand(
      dnn::DnnGraph&, Stream&, absl::Span<DeviceAddressBase> operands,
      absl::Span<const Command* const> dependencies) override;
  absl::Status UpdateDnnGraphCommand(
      const Command*, dnn::DnnGraph&, Stream&,
      absl::Span<DeviceAddressBase> operands) override;

  absl::StatusOr<const Command*> CreateCase(
      DeviceAddress<int32_t> index, std::vector<CreateCommands> create_branches,
      absl::Span<const Command* const> dependencies) override;
  absl::StatusOr<const Command*> CreateCase(
      DeviceAddress<bool> index, std::vector<CreateCommands> create_branches,
      absl::Span<const Command* const> dependencies) override;
  absl::Status UpdateCase(const Command* command, DeviceAddress<int32_t> index,
                          std::vector<UpdateCommands> update_branches) override;
  absl::Status UpdateCase(const Command* command, DeviceAddress<bool> index,
                          std::vector<UpdateCommands> update_branches) override;
  absl::StatusOr<const Command*> CreateWhile(
      DeviceAddress<bool> pred, CreateCommands create_cond,
      CreateCommands create_body,
      absl::Span<const Command* const> dependencies) override;
  absl::Status UpdateWhile(const Command* command, DeviceAddress<bool> pred,
                           UpdateCommands update_cond,
                           UpdateCommands update_body) override;

  absl::StatusOr<const Command*> CreateHost(
      absl::AnyInvocable<void()> callback,
      absl::Span<const Command* const> dependencies) override;

  absl::Status SetPriority(StreamPriority priority) override;
  absl::Status Submit(Stream* stream) override;
  absl::Status Finalize() override;
  absl::Status Update() override;
  Mode mode() const override { return mode_; }
  State state() const override { return state_; }
  std::string ToString() const override;

  // Number of recorded commands (for tests).
  size_t size() const { return list_.size(); }

 private:
  class MetalCommand : public Command {
   public:
    explicit MetalCommand(size_t index) : index(index) {}
    size_t index;
  };
  struct LaunchParams {
    const metal_pjrt::rt::Kernel* kernel = nullptr;
    metal_pjrt::rt::Dim3 threadgroups{1, 1, 1};
    metal_pjrt::rt::Dim3 threads{1, 1, 1};
    absl::InlinedVector<metal_pjrt::rt::KernelArg, 16> args;
    uint32_t shared_bytes = 0;
  };

  absl::Status Trace(
      Stream* stream,
      absl::AnyInvocable<absl::Status(Stream* stream)> function) override;

  absl::Status CheckState(State expected, absl::string_view op) const;
  absl::StatusOr<const MetalCommand*> Cast(const Command* command,
                                           absl::string_view op) const;
  const Command* Track(size_t index);
  static absl::Status ToLaunch(const ThreadDim& threads, const BlockDim& blocks,
                               const std::optional<ClusterDim>& cluster_dims,
                               const Kernel& kernel, const KernelArgs& args,
                               LaunchParams& out);
  absl::Status Unimplemented(absl::string_view what) const;

  MetalExecutor* executor_;
  Mode mode_;
  State state_ = State::kCreate;
  metal_pjrt::rt::CommandList list_;
  std::vector<std::unique_ptr<MetalCommand>> commands_;
};

}  // namespace metal
}  // namespace stream_executor

#endif  // METAL_PJRT_PLUGIN_STREAM_EXECUTOR_METAL_COMMAND_BUFFER_H_
