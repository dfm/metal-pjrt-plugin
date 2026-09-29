#include "metal_pjrt/compiler/passes/hlo_checks.h"

#include <cstdint>
#include <limits>
#include <string>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "metal_pjrt/blas/blas_lt_support.h"
#include "metal_pjrt/compiler/passes/dot_upcast.h"
#include "metal_pjrt/compiler/report_bug.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_module_metadata.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
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

// XLA's GetBatchRowColumnShape (matmul_utils.cc, under GemmConfig::For)
// multiplies each group of dimensions (batch, rows, columns) with an int
// accumulator (absl::c_accumulate(dims, 1, ...)), so a group of 2^31 or more
// elements wraps and the thunk would run the GEMM on the wrapped size. Checks
// the HLO's int64 dimensions before that.
absl::Status CheckGemmGroupsFitInt32(const HloInstruction& instr,
                                     const DotDimensionNumbers& dnums) {
  auto check = [](const Shape& shape, absl::Span<const int64_t> batch,
                  absl::Span<const int64_t> contracting) -> absl::Status {
    int64_t b = 1, c = 1, rest = 1;
    for (int64_t d = 0; d < shape.dimensions().size(); ++d) {
      const int64_t size = shape.dimensions(d);
      if (absl::c_linear_search(batch, d)) {
        b *= size;
      } else if (absl::c_linear_search(contracting, d)) {
        c *= size;
      } else {
        rest *= size;
      }
    }
    for (int64_t group : {b, c, rest}) {
      if (group > std::numeric_limits<int32_t>::max()) {
        return absl::UnimplementedError(absl::StrCat(
            "Metal: matmul operand ", ShapeUtil::HumanString(shape),
            " has a batch, row or column group of ", group,
            " elements; XLA's matmul config counts them in 32 bits. Split "
            "the batch (lax.map, or chunks)"));
      }
    }
    return absl::OkStatus();
  };
  TF_RETURN_IF_ERROR(check(instr.operand(0)->shape(),
                           dnums.lhs_batch_dimensions(),
                           dnums.lhs_contracting_dimensions()));
  return check(instr.operand(1)->shape(), dnums.rhs_batch_dimensions(),
               dnums.rhs_contracting_dimensions());
}

absl::Status CheckGemm(const HloInstruction& instr) {
  if (!IsCublasLtMatmul(instr)) {
    return absl::UnimplementedError(
        absl::StrCat("Metal: no matmul implementation for ",
                     instr.custom_call_target(), " (", DescribeOp(instr), ")"));
  }
  TF_ASSIGN_OR_RETURN(auto config, instr.backend_config<GpuBackendConfig>());
  // What GetMatmulPlan checks again at run time, on the config the thunk
  // builds.
  absl::Status s = [&]() -> absl::Status {
    TF_RETURN_IF_ERROR(CheckGemmGroupsFitInt32(
        instr, config.gemm_backend_config().dot_dimension_numbers()));
    TF_ASSIGN_OR_RETURN(
        stream_executor::gpu::BlasLt::Epilogue e,
        gpublas_lt::AsBlasLtEpilogue(config.gemm_backend_config().epilogue()));
    TF_ASSIGN_OR_RETURN(
        GemmConfig gemm,
        GemmConfig::For(&instr,
                        se::GpuComputeCapability(se::OneAPIComputeCapability())));
    return stream_executor::metal::ValidateMatmul(gemm, e).status();
  }();
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

// Ops that move f64 values through an emitted kernel, which cannot load f64
// (the MSL emitter would refuse it without naming the op). Contiguous
// slices, dynamic slices and updates, reshapes and bitcasts are copies or
// no-ops and work.
bool MovesF64InAKernel(const HloInstruction& instr) {
  switch (instr.opcode()) {
    case HloOpcode::kTranspose:
      if (ShapeUtil::TransposeIsBitcast(instr.operand(0)->shape(),
                                        instr.shape(), instr.dimensions())) {
        return false;
      }
      [[fallthrough]];
    case HloOpcode::kBroadcast:
    case HloOpcode::kConcatenate:
    case HloOpcode::kGather:
    case HloOpcode::kIota:
    case HloOpcode::kPad:
    case HloOpcode::kReverse:
    case HloOpcode::kSelect:
      return TouchesType(instr, {F64, C128});
    default:
      return false;
  }
}

// A scatter whose combiner does more than overwrite needs atomics on the
// element type (read-modify-write when indices collide). XLA overwrites
// complex elements with a compare-and-swap loop too (no atomic store of a
// complex, atomic_rmw_utils.cc), a 64-bit one for complex64.
bool NeedsWideAtomics(const HloInstruction& instr) {
  if (instr.opcode() != HloOpcode::kScatter ||
      Cast<HloScatterInstruction>(&instr)->unique_indices()) {
    return false;
  }
  if (TouchesType(instr, {C64, C128})) return true;
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
      if (instr->opcode() == HloOpcode::kFft) {
        return absl::UnimplementedError(absl::StrCat(
            "Metal: the HLO fft op is not supported (XLA's FFT runs on cuFFT "
            "only); jax.numpy.fft / jax.lax.fft lower to the plugin's "
            "metal$fft instead, so this program was lowered for another "
            "platform: ",
            DescribeOp(*instr)));
      }
      if (Computes(*instr) && TouchesType(*instr, {F64, C128})) {
        return absl::UnimplementedError(absl::StrCat(
            "Metal: f64 / complex128 arithmetic is not supported (Apple GPUs "
            "have no double precision; use float32 / complex64, or run it on "
            "the CPU backend): ",
            DescribeOp(*instr)));
      }
      if (MovesF64InAKernel(*instr)) {
        const bool c128 = TouchesType(*instr, {C128});
        return absl::UnimplementedError(absl::StrCat(
            "Metal: ", c128 ? "complex128 " : "f64 ",
            HloOpcodeString(instr->opcode()),
            " inside jit is not supported (Apple GPUs have no double "
            "precision, so GPU kernels cannot load f64; only transfers of "
            "f64 / complex128 arrays work). Use float32 / complex64, or run "
            "it on the CPU backend: ",
            DescribeOp(*instr)));
      }
      if (NeedsWideAtomics(*instr)) {
        if (TouchesType(*instr, {C64, C128})) {
          return absl::UnimplementedError(absl::StrCat(
              "Metal: scatter of complex values needs 64-bit atomics (XLA "
              "compare-and-swaps complex elements, even to overwrite), which "
              "Metal does not have; pass unique_indices=True if the indices "
              "do not repeat: ",
              DescribeOp(*instr)));
        }
        return absl::UnimplementedError(absl::StrCat(
            "Metal: scatter with a combiner on 64-bit elements needs 64-bit "
            "atomics, which Metal does not have: ",
            DescribeOp(*instr)));
      }
      if (instr->opcode() == HloOpcode::kDot &&
          instr->shape().element_type() == C64) {
        return absl::InternalError(absl::StrCat(
            "Metal: complex64 dot after MetalComplexDotExpander: ",
            DescribeOp(*instr), metal_pjrt::kReportBug));
      }
      if (IsNarrowOperandDot(instr)) {
        return absl::InternalError(absl::StrCat(
            "Metal: dot with operands narrower than its result after "
            "MetalDotOperandUpcaster: ",
            DescribeOp(*instr), metal_pjrt::kReportBug));
      }
      if (instr->opcode() != HloOpcode::kCustomCall) continue;
      if (IsCublasGemm(*instr)) TF_RETURN_IF_ERROR(CheckGemm(*instr));
    }
  }
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla
