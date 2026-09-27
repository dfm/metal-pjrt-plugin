#include "metal_pjrt_plugin/compiler/passes/hlo_checks.h"

#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "metal_pjrt_plugin/blas/blas_lt_support.h"
#include "metal_pjrt_plugin/compiler/passes/dot_upcast.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_module_metadata.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "absl/types/span.h"
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

// Ops that compute on element values (as opposed to moving them, calling
// something, or running on the host).
bool Computes(const HloInstruction& instr) {
  if (instr.opcode() == HloOpcode::kSelect ||
      instr.opcode() == HloOpcode::kConstant) {
    return false;
  }
  if (instr.IsElementwise()) return true;
  switch (instr.opcode()) {
    case HloOpcode::kReduce:
    case HloOpcode::kReduceWindow:
    case HloOpcode::kDot:
    case HloOpcode::kRaggedDot:
    case HloOpcode::kConvolution:
    case HloOpcode::kFft:
    case HloOpcode::kCholesky:
    case HloOpcode::kTriangularSolve:
    case HloOpcode::kSort:
    case HloOpcode::kScatter:
    case HloOpcode::kSelectAndScatter:
    case HloOpcode::kRng:
    case HloOpcode::kRngBitGenerator:
    case HloOpcode::kMap:
    case HloOpcode::kTopK:
      return true;
    default:
      return false;
  }
}

bool TouchesType(const HloInstruction& instr,
                 absl::Span<const PrimitiveType> types) {
  auto has = [&](const Shape& shape) {
    for (PrimitiveType t : types) {
      if (ShapeUtil::HasPrimitiveType(shape, t)) return true;
    }
    return false;
  };
  if (has(instr.shape())) return true;
  for (const HloInstruction* op : instr.operands()) {
    if (has(op->shape())) return true;
  }
  return false;
}

// A scatter whose combiner does more than overwrite needs atomics on the
// element type (read-modify-write when indices collide).
bool NeedsWideAtomics(const HloInstruction& instr) {
  if (instr.opcode() != HloOpcode::kScatter ||
      Cast<HloScatterInstruction>(&instr)->unique_indices()) {
    return false;
  }
  const HloInstruction* root = instr.to_apply()->root_instruction();
  const int n = instr.operand_count() / 2;  // operands, indices, updates
  bool overwrite = root->opcode() == HloOpcode::kParameter &&
                   root->parameter_number() >= n;
  if (root->opcode() == HloOpcode::kTuple) {
    overwrite = true;
    for (const HloInstruction* e : root->operands()) {
      overwrite &= e->opcode() == HloOpcode::kParameter &&
                   e->parameter_number() >= n;
    }
  }
  if (overwrite) return false;
  bool wide = false;
  ShapeUtil::ForEachSubshape(instr.shape(), [&](const Shape& s,
                                                const ShapeIndex&) {
    if (s.IsArray() && primitive_util::BitWidth(s.element_type()) > 32) {
      wide = true;
    }
  });
  return wide;
}

}  // namespace

absl::Status CheckPostGemmRewriter(const HloModule& module) {
  for (const HloComputation* comp : module.computations()) {
    for (const HloInstruction* instr : comp->instructions()) {
      if (Computes(*instr) && TouchesType(*instr, {F64})) {
        return absl::UnimplementedError(absl::StrCat(
            "Metal: f64 arithmetic is not supported (Apple GPUs have no "
            "double precision; use float32, or run it on the CPU backend): ",
            DescribeOp(*instr)));
      }
      if (NeedsWideAtomics(*instr)) {
        return absl::UnimplementedError(absl::StrCat(
            "Metal: scatter with a combiner on 64-bit elements needs 64-bit "
            "atomics, which Metal does not have: ",
            DescribeOp(*instr)));
      }
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
