#include "metal_pjrt_plugin/compiler/passes/hlo_checks.h"

#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "metal_pjrt_plugin/blas/blas_lt_support.h"
#include "metal_pjrt_plugin/compiler/passes/dot_upcast.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_module_metadata.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/shape.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

std::string DescribeOp(const HloInstruction& instr) {
  const OpMetadata& md = instr.metadata();
  std::string s(instr.name());
  if (!md.op_name().empty()) absl::StrAppend(&s, " (", md.op_name(), ")");
  const HloModule* module = instr.GetModule();
  if (md.stack_frame_id() != 0 && module != nullptr) {
    HloStackFrame f =
        module->get_stack_frame(StackFrameId{md.stack_frame_id()});
    if (!f.empty()) {
      absl::StrAppend(&s, " at ", f.file_name, ":", f.line);
      return s;
    }
  }
  if (!md.source_file().empty()) {
    absl::StrAppend(&s, " at ", md.source_file(), ":", md.source_line());
  }
  return s;
}

namespace {

absl::Status CheckGemm(const HloInstruction& instr) {
  if (!IsCublasLtMatmul(instr)) {
    return absl::UnimplementedError(
        absl::StrCat("Metal: no GEMM implementation for ",
                     instr.custom_call_target(), " (", DescribeOp(instr), ")"));
  }
  TF_ASSIGN_OR_RETURN(auto config, instr.backend_config<GpuBackendConfig>());
  const Shape& out =
      instr.shape().IsTuple() ? instr.shape().tuple_shapes(0) : instr.shape();
  absl::Status s = stream_executor::metal::CheckBlasLtTypes(
      instr.operand(0)->shape().element_type(),
      instr.operand(1)->shape().element_type(), out.element_type());
  if (s.ok()) {
    absl::StatusOr<stream_executor::gpu::BlasLt::Epilogue> e =
        gpublas_lt::AsBlasLtEpilogue(config.gemm_backend_config().epilogue());
    s = e.ok() ? stream_executor::metal::DecodeEpilogue(*e).status()
               : e.status();
  }
  if (s.ok()) return s;
  return absl::Status(s.code(),
                      absl::StrCat(s.message(), " (", DescribeOp(instr), ")"));
}

}  // namespace

absl::Status CheckPostGemmRewriter(const HloModule& module) {
  for (const HloComputation* comp : module.computations()) {
    for (const HloInstruction* instr : comp->instructions()) {
      if (IsNarrowOperandDot(instr)) {
        return absl::InternalError(absl::StrCat(
            "Metal: dot with operands narrower than its result after "
            "MetalDotOperandUpcaster: ",
            DescribeOp(*instr)));
      }
      if (instr->opcode() != HloOpcode::kCustomCall) continue;
      if (instr->custom_call_target() == "TopK" ||
          instr->custom_call_target() == kTopKCustomCallTarget) {
        return absl::InternalError(
            absl::StrCat("Metal: TopK custom call left after TopkDecomposer: ",
                         DescribeOp(*instr)));
      }
      if (IsCublasGemm(*instr)) TF_RETURN_IF_ERROR(CheckGemm(*instr));
    }
  }
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla
