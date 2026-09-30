// Copyright 2026 The metal-pjrt-plugin Authors
// SPDX-License-Identifier: Apache-2.0

#include "metal_pjrt/compiler/passes/conv_rewriter.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "metal_pjrt/conv/conv.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout_util.h"
#include "xla/permutation_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

namespace mconv = ::metal_pjrt::conv;

// A convolution the library runs: its parameters, the operands in the
// order of the call ({input, weight} or {input, dY}), the permutation that
// brings each to its canonical dimension order (before a 1-D one gains its
// unit H dimension), and the permutation from the canonical result order
// back to the convolution's.
struct ConvMatch {
  mconv::ConvParams params;
  HloInstruction* operands[2];
  std::vector<int64_t> perm[2];
  std::vector<int64_t> result_perm;  // canonical dim i = result dim perm[i]
  // The rhs spatial dims to reverse first (a reversed window of the
  // weight-gradient kind; the forward kind folds it into flip).
  std::vector<int64_t> reverse_rhs;
};

std::optional<mconv::ConvType> ConvTypeOf(PrimitiveType t) {
  switch (t) {
    case F32:
      return mconv::ConvType::kF32;
    case F16:
      return mconv::ConvType::kF16;
    case BF16:
      return mconv::ConvType::kBF16;
    default:
      return std::nullopt;
  }
}

std::optional<ConvMatch> MatchConv(HloInstruction* conv) {
  const ConvolutionDimensionNumbers& dn =
      conv->convolution_dimension_numbers();
  const int64_t ns = dn.input_spatial_dimensions_size();
  if (ns < 1 || ns > 2 || conv->feature_group_count() != 1 ||
      conv->batch_group_count() != 1) {
    return std::nullopt;
  }
  HloInstruction* lhs = conv->mutable_operand(0);
  HloInstruction* rhs = conv->mutable_operand(1);
  const PrimitiveType type = conv->shape().element_type();
  std::optional<mconv::ConvType> ctype = ConvTypeOf(type);
  if (!ctype.has_value() || lhs->shape().element_type() != type ||
      rhs->shape().element_type() != type) {
    return std::nullopt;
  }
  // Window reversal (XLA's algebraic simplifier swaps the operands of a
  // convolution whose kernel is larger than its input, reversing the new
  // kernel): all spatial dims or none.
  const Window& window = conv->window();
  int reversed = 0;
  for (const WindowDimension& w : window.dimensions()) {
    reversed += w.window_reversal() ? 1 : 0;
  }
  if (reversed != 0 && reversed != ns) return std::nullopt;

  // JAX's weight gradient swaps the lhs batch and feature dimensions; its
  // rhs (the output gradient) keeps batch (here: input feature) first.
  const bool lhs_swapped =
      dn.input_feature_dimension() < dn.input_batch_dimension();
  const bool rhs_batch_first = dn.kernel_input_feature_dimension() <
                               dn.kernel_output_feature_dimension();
  if (lhs_swapped && !rhs_batch_first) return std::nullopt;  // ambiguous
  const bool wgrad = lhs_swapped;

  ConvMatch m;
  mconv::ConvParams& p = m.params;
  p.kind = wgrad ? mconv::ConvKind::kWeightGrad : mconv::ConvKind::kForward;
  p.type = *ctype;
  const Shape& ls = lhs->shape();
  const Shape& rs = rhs->shape();
  const Shape& os = conv->shape();
  // Sizes of the canonical dims, with H = 1 for 1-D.
  auto spatial = [&](const Shape& s, auto dims, int i) -> int64_t {
    return ns == 1 ? (i == 0 ? 1 : s.dimensions(dims.Get(0)))
                   : s.dimensions(dims.Get(i));
  };
  // Window parameters on canonical dim i (H is dim 0; 1-D has only W).
  auto win = [&](int i) -> const WindowDimension* {
    if (ns == 1) return i == 0 ? nullptr : &window.dimensions(0);
    return &window.dimensions(i);
  };
  std::vector<int64_t> lhs_sp(dn.input_spatial_dimensions().begin(),
                              dn.input_spatial_dimensions().end());
  std::vector<int64_t> rhs_sp(dn.kernel_spatial_dimensions().begin(),
                              dn.kernel_spatial_dimensions().end());
  std::vector<int64_t> out_sp(dn.output_spatial_dimensions().begin(),
                              dn.output_spatial_dimensions().end());
  auto perm = [](int64_t first, const std::vector<int64_t>& sp,
                 int64_t last) {
    std::vector<int64_t> v = {first};
    v.insert(v.end(), sp.begin(), sp.end());
    v.push_back(last);
    return v;
  };

  if (!wgrad) {
    // Fold a reverse of exactly the kernel's spatial dims into flip.
    if (rhs->opcode() == HloOpcode::kReverse && rhs->user_count() == 1) {
      std::vector<int64_t> rev(rhs->dimensions().begin(),
                               rhs->dimensions().end());
      std::vector<int64_t> want = rhs_sp;
      absl::c_sort(rev);
      absl::c_sort(want);
      if (rev == want) {
        p.flip = true;
        rhs = rhs->mutable_operand(0);
      }
    }
    if (reversed != 0) p.flip = !p.flip;
    p.n = ls.dimensions(dn.input_batch_dimension());
    p.c = ls.dimensions(dn.input_feature_dimension());
    p.o = rs.dimensions(dn.kernel_output_feature_dimension());
    p.h = spatial(ls, dn.input_spatial_dimensions(), 0);
    p.w = spatial(ls, dn.input_spatial_dimensions(), 1);
    p.kh = spatial(rs, dn.kernel_spatial_dimensions(), 0);
    p.kw = spatial(rs, dn.kernel_spatial_dimensions(), 1);
    p.out_h = spatial(os, dn.output_spatial_dimensions(), 0);
    p.out_w = spatial(os, dn.output_spatial_dimensions(), 1);
    for (int i = 0; i < 2; ++i) {
      if (const WindowDimension* w = win(i)) {
        p.stride[i] = w->stride();
        p.pad_lo[i] = w->padding_low();
        p.kdil[i] = w->window_dilation();
        p.idil[i] = w->base_dilation();
      }
    }
    m.operands[0] = lhs;
    m.operands[1] = rhs;
    m.perm[0] = perm(dn.input_batch_dimension(), lhs_sp,
                     dn.input_feature_dimension());
    m.perm[1] = perm(dn.kernel_output_feature_dimension(), rhs_sp,
                     dn.kernel_input_feature_dimension());
    m.result_perm = perm(dn.output_batch_dimension(), out_sp,
                         dn.output_feature_dimension());
  } else {
    // lhs = x with (batch, feature) = (C, N); rhs = dY with (input feature,
    // output feature) = (N, O); result = dW with (batch, feature) = (C, O)
    // and the kernel's spatial dims.
    p.n = ls.dimensions(dn.input_feature_dimension());
    p.c = ls.dimensions(dn.input_batch_dimension());
    p.o = rs.dimensions(dn.kernel_output_feature_dimension());
    if (rs.dimensions(dn.kernel_input_feature_dimension()) != p.n) {
      return std::nullopt;
    }
    p.h = spatial(ls, dn.input_spatial_dimensions(), 0);
    p.w = spatial(ls, dn.input_spatial_dimensions(), 1);
    p.out_h = spatial(rs, dn.kernel_spatial_dimensions(), 0);
    p.out_w = spatial(rs, dn.kernel_spatial_dimensions(), 1);
    p.kh = spatial(os, dn.output_spatial_dimensions(), 0);
    p.kw = spatial(os, dn.output_spatial_dimensions(), 1);
    for (int i = 0; i < 2; ++i) {
      if (const WindowDimension* w = win(i)) {
        p.stride[i] = w->window_dilation();
        p.pad_lo[i] = w->padding_low();
        p.kdil[i] = w->stride();
        p.idil[i] = w->base_dilation();
      }
    }
    m.operands[0] = lhs;
    m.operands[1] = rhs;
    m.perm[0] = perm(dn.input_feature_dimension(), lhs_sp,
                     dn.input_batch_dimension());
    m.perm[1] = perm(dn.kernel_input_feature_dimension(), rhs_sp,
                     dn.kernel_output_feature_dimension());
    m.result_perm = perm(dn.output_feature_dimension(), out_sp,
                         dn.output_batch_dimension());
    if (reversed != 0) m.reverse_rhs = rhs_sp;
  }
  return m;
}

bool IsIdentity(const std::vector<int64_t>& perm) {
  for (int64_t i = 0; i < static_cast<int64_t>(perm.size()); ++i) {
    if (perm[i] != i) return false;
  }
  return true;
}

// `x` transposed by `perm` (none for the identity), with a unit H dimension
// inserted after the first for 1-D (rank 3).
absl::StatusOr<HloInstruction*> Canonical(HloInstruction* x,
                                          const std::vector<int64_t>& perm) {
  if (!IsIdentity(perm)) {
    std::vector<int64_t> dims;
    for (int64_t d : perm) dims.push_back(x->shape().dimensions(d));
    x = x->AddInstruction(HloInstruction::CreateTranspose(
        ShapeUtil::MakeShape(x->shape().element_type(), dims), x, perm));
  }
  if (perm.size() == 3) {
    const Shape& s = x->shape();
    x = x->AddInstruction(HloInstruction::CreateReshape(
        ShapeUtil::MakeShape(s.element_type(),
                             {s.dimensions(0), 1, s.dimensions(1),
                              s.dimensions(2)}),
        x));
  }
  return x;
}

std::string I64Array(const int64_t v[2]) {
  return absl::StrCat("array<i64: ", v[0], ", ", v[1], ">");
}

}  // namespace

absl::StatusOr<bool> MetalConvRewriter::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* comp :
       module->MakeNonfusionComputations(execution_threads)) {
    std::vector<HloInstruction*> convs;
    for (HloInstruction* instr : comp->instructions()) {
      if (instr->opcode() == HloOpcode::kConvolution) convs.push_back(instr);
    }
    for (HloInstruction* conv : convs) {
      std::optional<ConvMatch> m = MatchConv(conv);
      if (!m.has_value()) continue;
      const mconv::ConvParams& p = m->params;
      if (mconv::ConvFlops(p) < static_cast<uint64_t>(min_flops_)) continue;
      absl::StatusOr<mconv::ConvPlan> plan = mconv::PlanConv(p);
      if (!plan.ok()) continue;
      if (!m->reverse_rhs.empty()) {
        HloInstruction* rhs = m->operands[1];
        m->operands[1] = comp->AddInstruction(HloInstruction::CreateReverse(
            rhs->shape(), rhs, m->reverse_rhs));
      }
      HloInstruction* args[2];
      Shape arg_shapes[2];
      for (int i = 0; i < 2; ++i) {
        TF_ASSIGN_OR_RETURN(args[i], Canonical(m->operands[i], m->perm[i]));
        arg_shapes[i] = args[i]->shape();
        LayoutUtil::SetToDefaultLayout(&arg_shapes[i]);
      }
      const bool wgrad = p.kind == mconv::ConvKind::kWeightGrad;
      const Shape out_shape =
          wgrad ? ShapeUtil::MakeShapeWithDescendingLayout(
                      conv->shape().element_type(), {p.o, p.kh, p.kw, p.c})
                : ShapeUtil::MakeShapeWithDescendingLayout(
                      conv->shape().element_type(),
                      {p.n, p.out_h, p.out_w, p.o});
      const Shape ws_shape = ShapeUtil::MakeShapeWithDescendingLayout(
          U8, {static_cast<int64_t>(plan->workspace_bytes)});
      HloInstruction* call = comp->AddInstruction(
          HloInstruction::CreateCustomCall(
              ShapeUtil::MakeTupleShape({out_shape, ws_shape}),
              {args[0], args[1]}, "metal$conv",
              {arg_shapes[0], arg_shapes[1]},
              /*opaque=*/"", CustomCallApiVersion::API_VERSION_TYPED_FFI));
      GpuBackendConfig config;
      config.mutable_custom_call_backend_config()->set_attributes(
          absl::StrCat("{kind = \"", wgrad ? "wgrad" : "fwd",
                       "\", stride = ", I64Array(p.stride),
                       ", pad_lo = ", I64Array(p.pad_lo),
                       ", kdil = ", I64Array(p.kdil),
                       ", idil = ", I64Array(p.idil),
                       ", flip = ", p.flip ? "true" : "false", "}"));
      TF_RETURN_IF_ERROR(call->set_backend_config(config));
      call->set_metadata(conv->metadata());
      HloInstruction* out = comp->AddInstruction(
          HloInstruction::CreateGetTupleElement(out_shape, call, 0));
      // Back to the convolution's dimension order: drop the unit H of a
      // 1-D one, then undo the canonical permutation.
      std::vector<int64_t> result_perm = m->result_perm;
      if (result_perm.size() == 3) {
        out = comp->AddInstruction(HloInstruction::CreateReshape(
            ShapeUtil::MakeShape(out_shape.element_type(),
                                 {out_shape.dimensions(0),
                                  out_shape.dimensions(2),
                                  out_shape.dimensions(3)}),
            out));
      }
      if (!IsIdentity(result_perm)) {
        // canonical dim i holds result dim result_perm[i].
        out = comp->AddInstruction(HloInstruction::CreateTranspose(
            ShapeUtil::MakeShape(conv->shape().element_type(),
                                 conv->shape().dimensions()),
            out, InversePermutation(result_perm)));
      }
      // (Also removes a folded reverse, now unused.)
      TF_RETURN_IF_ERROR(comp->ReplaceInstruction(conv, out));
      changed = true;
    }
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
