#ifndef METAL_PJRT_PLUGIN_STREAM_EXECUTOR_METAL_KERNEL_H_
#define METAL_PJRT_PLUGIN_STREAM_EXECUTOR_METAL_KERNEL_H_

#include <cstdint>
#include <memory>
#include <optional>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "metal_pjrt_plugin/runtime/metal_runtime.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream.h"

namespace stream_executor {
namespace metal {

class MetalExecutor;

// A compiled Metal compute pipeline. Arguments are device pointers in HLO
// buffer order (one [[buffer(i)]] each), optionally followed by threadgroup
// memory.
class MetalKernel : public Kernel {
 public:
  MetalKernel(MetalExecutor* executor,
              std::unique_ptr<metal_pjrt::rt::Kernel> kernel, unsigned arity);
  ~MetalKernel() override;

  unsigned Arity() const override { return arity_; }

  absl::StatusOr<int32_t> GetMaxOccupiedBlocksPerCore(
      ThreadDim threads, size_t dynamic_shared_memory_bytes) const override;

  metal_pjrt::rt::Kernel* rt_kernel() const { return kernel_.get(); }

 private:
  absl::Status Launch(const ThreadDim& thread_dims, const BlockDim& block_dims,
                      const std::optional<ClusterDim>& cluster_dims,
                      Stream* stream, const KernelArgs& args) override;

  MetalExecutor* executor_;
  std::unique_ptr<metal_pjrt::rt::Kernel> kernel_;
  unsigned arity_;
};

}  // namespace metal
}  // namespace stream_executor

#endif  // METAL_PJRT_PLUGIN_STREAM_EXECUTOR_METAL_KERNEL_H_
