#include "metal_pjrt/linalg/linalg_rewriter.h"

#include <cstdlib>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "metal_pjrt/compiler/compile_settings.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/shape.h"
#include "xla/tsl/platform/errors.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

Shape WithDefaultLayout(Shape s) {
  LayoutUtil::SetToDefaultLayout(&s);
  return s;
}

const char* Bool(bool b) { return b ? "true" : "false"; }

}  // namespace

bool LapackDisabled() { return !metal_pjrt::GetCompileSettings().lapack; }

absl::StatusOr<bool> MetalLinalgRewriter::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* comp :
       module->MakeNonfusionComputations(execution_threads)) {
    std::vector<HloInstruction*> candidates;
    for (HloInstruction* instr : comp->instructions()) {
      if ((instr->opcode() == HloOpcode::kCholesky ||
           instr->opcode() == HloOpcode::kTriangularSolve) &&
          instr->shape().IsArray() && instr->shape().element_type() == F32) {
        candidates.push_back(instr);
      }
    }
    for (HloInstruction* instr : candidates) {
      std::vector<HloInstruction*> operands(instr->operands().begin(),
                                            instr->operands().end());
      std::vector<Shape> operand_shapes;
      for (HloInstruction* op : operands) {
        operand_shapes.push_back(WithDefaultLayout(op->shape()));
      }
      std::string target;
      std::string attrs;
      if (instr->opcode() == HloOpcode::kCholesky) {
        target = "metal$cholesky";
        attrs = absl::StrCat("{lower = ",
                             Bool(instr->cholesky_options().lower()), "}");
      } else {
        const TriangularSolveOptions& o = instr->triangular_solve_options();
        target = "metal$triangular_solve";
        attrs = absl::StrCat(
            "{left_side = ", Bool(o.left_side()), ", lower = ",
            Bool(o.lower()), ", unit_diagonal = ", Bool(o.unit_diagonal()),
            ", transpose_a = ", static_cast<int>(o.transpose_a()), " : i32}");
      }
      HloInstruction* call = comp->AddInstruction(
          HloInstruction::CreateCustomCall(
              WithDefaultLayout(instr->shape()), operands, target,
              operand_shapes, /*opaque=*/"",
              CustomCallApiVersion::API_VERSION_TYPED_FFI));
      GpuBackendConfig config;
      config.mutable_custom_call_backend_config()->set_attributes(attrs);
      TF_RETURN_IF_ERROR(call->set_backend_config(config));
      call->set_metadata(instr->metadata());
      TF_RETURN_IF_ERROR(comp->ReplaceInstruction(instr, call));
      changed = true;
    }
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
