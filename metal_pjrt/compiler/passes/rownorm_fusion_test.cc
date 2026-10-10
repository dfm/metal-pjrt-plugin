// Copyright 2026 The metal-pjrt-plugin Authors
// SPDX-License-Identifier: Apache-2.0

#include "metal_pjrt/compiler/passes/rownorm_fusion.h"

#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/gpu/transforms/fusion_wrapper.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/error_spec.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/gpu/alias_info.h"
#include "xla/service/gpu/fusion_pipeline.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tests/literal_test_util.h"

namespace xla {
namespace gpu {
namespace {

class MetalRowNormFusionTest : public HloHardwareIndependentTestBase {
 protected:
  // Runs the pass on `hlo`; returns the module (null on a parse failure).
  std::unique_ptr<HloModule> Run(absl::string_view hlo, bool expect_change,
                                 int64_t max_row_length =
                                     MetalRowNormFusion::kMaxRowLength) {
    auto module = ParseAndReturnVerifiedModule(hlo);
    EXPECT_TRUE(module.ok()) << module.status();
    if (!module.ok()) return nullptr;
    MetalRowNormFusion pass(max_row_length);
    auto changed = RunHloPass(&pass, module->get());
    EXPECT_TRUE(changed.ok()) << changed.status();
    EXPECT_EQ(changed.value_or(!expect_change), expect_change)
        << (*module)->ToString();
    return std::move(*module);
  }

  std::vector<const HloInstruction*> RowNormFusions(const HloModule& module) {
    std::vector<const HloInstruction*> fusions;
    for (const HloInstruction* instr :
         module.entry_computation()->instructions()) {
      if (IsMetalRowNormFusion(*instr)) fusions.push_back(instr);
    }
    return fusions;
  }

  static int Count(const HloComputation* computation, HloOpcode opcode) {
    int n = 0;
    for (const HloInstruction* instr : computation->instructions()) {
      n += instr->opcode() == opcode;
    }
    return n;
  }

  // The module's result on deterministic inputs, before and after the pass
  // (HloEvaluator runs fusions through their fused computations).
  void ExpectSameResult(absl::string_view hlo) {
    auto before = ParseAndReturnVerifiedModule(hlo);
    ASSERT_TRUE(before.ok()) << before.status();
    std::unique_ptr<HloModule> after = Run(hlo, /*expect_change=*/true);
    ASSERT_NE(after, nullptr);
    std::vector<Literal> args;
    const HloComputation* entry = (*before)->entry_computation();
    for (int i = 0; i < entry->num_parameters(); ++i) {
      const Shape& shape = entry->parameter_instruction(i)->shape();
      Literal arg(shape);
      int k = 0;
      ASSERT_TRUE(arg.Populate<float>([&](absl::Span<const int64_t>) {
                       ++k;
                       return 0.25f * ((k * 7 + i * 3) % 23) - 2.5f;
                     })
                      .ok());
      args.push_back(std::move(arg));
    }
    HloEvaluator before_evaluator, after_evaluator;
    auto expected = before_evaluator.Evaluate(**before, args);
    ASSERT_TRUE(expected.ok()) << expected.status();
    auto actual = after_evaluator.Evaluate(*after, args);
    ASSERT_TRUE(actual.ok()) << actual.status();
    EXPECT_TRUE(LiteralTestUtil::Near(*expected, *actual, ErrorSpec(1e-5)));
  }
};

constexpr absl::string_view kSoftmax = R"(
HloModule m
max_f32 {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT m = f32[] maximum(a, b)
}
add_f32 {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT s = f32[] add(a, b)
}
ENTRY e {
  x = f32[4,6,40]{2,1,0} parameter(0)
  ninf = f32[] constant(-inf)
  mx = f32[4,6]{1,0} reduce(x, ninf), dimensions={2}, to_apply=max_f32
  mxb = f32[4,6,40]{2,1,0} broadcast(mx), dimensions={0,1}
  sub = f32[4,6,40]{2,1,0} subtract(x, mxb)
  ex = f32[4,6,40]{2,1,0} exponential(sub)
  zero = f32[] constant(0)
  sm = f32[4,6]{1,0} reduce(ex, zero), dimensions={2}, to_apply=add_f32
  smb = f32[4,6,40]{2,1,0} broadcast(sm), dimensions={0,1}
  ROOT y = f32[4,6,40]{2,1,0} divide(ex, smb)
})";

TEST_F(MetalRowNormFusionTest, FusesSoftmax) {
  std::unique_ptr<HloModule> module = Run(kSoftmax, /*expect_change=*/true);
  ASSERT_NE(module, nullptr);
  const HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_TRUE(IsMetalRowNormFusion(*root)) << module->ToString();
  EXPECT_EQ(root->operand_count(), 1);
  EXPECT_EQ(Count(root->fused_instructions_computation(), HloOpcode::kReduce),
            2);
  EXPECT_EQ(Count(module->entry_computation(), HloOpcode::kReduce), 0);
  ExpectSameResult(kSoftmax);
}

// Attention's scale and causal mask are recomputed in the fusion, so the
// scaled, masked scores are never written.
TEST_F(MetalRowNormFusionTest, FusesMaskedScaledSoftmax) {
  constexpr absl::string_view kHlo = R"(
HloModule m
max_f32 {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT m = f32[] maximum(a, b)
}
add_f32 {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT s = f32[] add(a, b)
}
ENTRY e {
  s = f32[2,16,16]{2,1,0} parameter(0)
  c = f32[] constant(0.125)
  cb = f32[2,16,16]{2,1,0} broadcast(c), dimensions={}
  scaled = f32[2,16,16]{2,1,0} multiply(s, cb)
  i0 = s32[2,16,16]{2,1,0} iota(), iota_dimension=1
  i1 = s32[2,16,16]{2,1,0} iota(), iota_dimension=2
  mask = pred[2,16,16]{2,1,0} compare(i1, i0), direction=LE
  ninf = f32[] constant(-inf)
  ninfb = f32[2,16,16]{2,1,0} broadcast(ninf), dimensions={}
  x = f32[2,16,16]{2,1,0} select(mask, scaled, ninfb)
  mx = f32[2,16]{1,0} reduce(x, ninf), dimensions={2}, to_apply=max_f32
  mxb = f32[2,16,16]{2,1,0} broadcast(mx), dimensions={0,1}
  sub = f32[2,16,16]{2,1,0} subtract(x, mxb)
  ex = f32[2,16,16]{2,1,0} exponential(sub)
  zero = f32[] constant(0)
  sm = f32[2,16]{1,0} reduce(ex, zero), dimensions={2}, to_apply=add_f32
  smb = f32[2,16,16]{2,1,0} broadcast(sm), dimensions={0,1}
  ROOT y = f32[2,16,16]{2,1,0} divide(ex, smb)
})";
  std::unique_ptr<HloModule> module = Run(kHlo, /*expect_change=*/true);
  ASSERT_NE(module, nullptr);
  const HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_TRUE(IsMetalRowNormFusion(*root)) << module->ToString();
  EXPECT_EQ(root->operand_count(), 1);
  EXPECT_EQ(root->operand(0)->opcode(), HloOpcode::kParameter);
  ExpectSameResult(kHlo);
}

// Layer norm over the minor dimension, with weights and bias along it.
TEST_F(MetalRowNormFusionTest, FusesLayerNorm) {
  constexpr absl::string_view kHlo = R"(
HloModule m
add_f32 {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT s = f32[] add(a, b)
}
ENTRY e {
  x = f32[8,64]{1,0} parameter(0)
  w = f32[64]{0} parameter(1)
  b = f32[64]{0} parameter(2)
  zero = f32[] constant(0)
  inv_n = f32[] constant(0.015625)
  inv_nb = f32[8]{0} broadcast(inv_n), dimensions={}
  s = f32[8]{0} reduce(x, zero), dimensions={1}, to_apply=add_f32
  mean = f32[8]{0} multiply(s, inv_nb)
  meanb = f32[8,64]{1,0} broadcast(mean), dimensions={0}
  d = f32[8,64]{1,0} subtract(x, meanb)
  d2 = f32[8,64]{1,0} multiply(d, d)
  v = f32[8]{0} reduce(d2, zero), dimensions={1}, to_apply=add_f32
  var = f32[8]{0} multiply(v, inv_nb)
  eps = f32[] constant(1e-5)
  epsb = f32[8]{0} broadcast(eps), dimensions={}
  ve = f32[8]{0} add(var, epsb)
  r = f32[8]{0} rsqrt(ve)
  rb = f32[8,64]{1,0} broadcast(r), dimensions={0}
  n = f32[8,64]{1,0} multiply(d, rb)
  wb = f32[8,64]{1,0} broadcast(w), dimensions={1}
  bb = f32[8,64]{1,0} broadcast(b), dimensions={1}
  nw = f32[8,64]{1,0} multiply(n, wb)
  ROOT y = f32[8,64]{1,0} add(nw, bb)
})";
  std::unique_ptr<HloModule> module = Run(kHlo, /*expect_change=*/true);
  ASSERT_NE(module, nullptr);
  const HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_TRUE(IsMetalRowNormFusion(*root)) << module->ToString();
  EXPECT_EQ(root->operand_count(), 3);
  EXPECT_EQ(Count(root->fused_instructions_computation(), HloOpcode::kReduce),
            2);
  ExpectSameResult(kHlo);
}

// Mean and mean of squares of the same input (RMS norm's form of the
// variance) land in one fusion.
TEST_F(MetalRowNormFusionTest, FusesSiblingReductions) {
  constexpr absl::string_view kHlo = R"(
HloModule m
add_f32 {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT s = f32[] add(a, b)
}
ENTRY e {
  x = f32[8,64]{1,0} parameter(0)
  zero = f32[] constant(0)
  s = f32[8]{0} reduce(x, zero), dimensions={1}, to_apply=add_f32
  x2 = f32[8,64]{1,0} multiply(x, x)
  sq = f32[8]{0} reduce(x2, zero), dimensions={1}, to_apply=add_f32
  sb = f32[8,64]{1,0} broadcast(s), dimensions={0}
  sqb = f32[8,64]{1,0} broadcast(sq), dimensions={0}
  d = f32[8,64]{1,0} subtract(x, sb)
  ROOT y = f32[8,64]{1,0} multiply(d, sqb)
})";
  std::unique_ptr<HloModule> module = Run(kHlo, /*expect_change=*/true);
  ASSERT_NE(module, nullptr);
  std::vector<const HloInstruction*> fusions = RowNormFusions(*module);
  ASSERT_EQ(fusions.size(), 1) << module->ToString();
  EXPECT_EQ(
      Count(fusions[0]->fused_instructions_computation(), HloOpcode::kReduce),
      2);
  ExpectSameResult(kHlo);
}

// JAX's autodiff of softmax reuses the row sum and exp(x - max) in the
// backward: they become outputs of the fusion.
TEST_F(MetalRowNormFusionTest, OutputsIntermediatesUsedOutside) {
  constexpr absl::string_view kHlo = R"(
HloModule m
max_f32 {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT m = f32[] maximum(a, b)
}
add_f32 {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT s = f32[] add(a, b)
}
ENTRY e {
  x = f32[4,40]{1,0} parameter(0)
  g = f32[4,40]{1,0} parameter(1)
  ninf = f32[] constant(-inf)
  mx = f32[4]{0} reduce(x, ninf), dimensions={1}, to_apply=max_f32
  mxb = f32[4,40]{1,0} broadcast(mx), dimensions={0}
  sub = f32[4,40]{1,0} subtract(x, mxb)
  ex = f32[4,40]{1,0} exponential(sub)
  zero = f32[] constant(0)
  sm = f32[4]{0} reduce(ex, zero), dimensions={1}, to_apply=add_f32
  smb = f32[4,40]{1,0} broadcast(sm), dimensions={0}
  y = f32[4,40]{1,0} divide(ex, smb)
  t = f32[4,40]{1,0} transpose(ex), dimensions={0,1}
  tg = f32[4,40]{1,0} multiply(t, g)
  ROOT r = (f32[4,40]{1,0}, f32[4]{0}, f32[4,40]{1,0}) tuple(y, sm, tg)
})";
  std::unique_ptr<HloModule> module = Run(kHlo, /*expect_change=*/true);
  ASSERT_NE(module, nullptr);
  std::vector<const HloInstruction*> fusions = RowNormFusions(*module);
  ASSERT_EQ(fusions.size(), 1) << module->ToString();
  // y, sm and ex (read by the transpose, which is not elementwise).
  ASSERT_TRUE(fusions[0]->shape().IsTuple());
  EXPECT_EQ(fusions[0]->shape().tuple_shapes().size(), 3)
      << fusions[0]->ToString();
  ExpectSameResult(kHlo);
}

// A consumer whose other operand depends on the fusion would form a cycle;
// it stays outside, and the softmax is still fused.
TEST_F(MetalRowNormFusionTest, LeavesOutConsumersThatWouldFormCycles) {
  constexpr absl::string_view kHlo = R"(
HloModule m
max_f32 {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT m = f32[] maximum(a, b)
}
add_f32 {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT s = f32[] add(a, b)
}
ENTRY e {
  x = f32[4,40]{1,0} parameter(0)
  ninf = f32[] constant(-inf)
  mx = f32[4]{0} reduce(x, ninf), dimensions={1}, to_apply=max_f32
  mxb = f32[4,40]{1,0} broadcast(mx), dimensions={0}
  sub = f32[4,40]{1,0} subtract(x, mxb)
  ex = f32[4,40]{1,0} exponential(sub)
  zero = f32[] constant(0)
  sm = f32[4]{0} reduce(ex, zero), dimensions={1}, to_apply=add_f32
  smb = f32[4,40]{1,0} broadcast(sm), dimensions={0}
  y = f32[4,40]{1,0} divide(ex, smb)
  total = f32[] reduce(y, zero), dimensions={0,1}, to_apply=add_f32
  tb = f32[4,40]{1,0} broadcast(total), dimensions={}
  ROOT z = f32[4,40]{1,0} multiply(y, tb)
})";
  std::unique_ptr<HloModule> module = Run(kHlo, /*expect_change=*/true);
  ASSERT_NE(module, nullptr);
  std::vector<const HloInstruction*> fusions = RowNormFusions(*module);
  ASSERT_EQ(fusions.size(), 1) << module->ToString();
  EXPECT_EQ(module->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kMultiply);
  ExpectSameResult(kHlo);
}

TEST_F(MetalRowNormFusionTest, LeavesPlainReductionsAlone) {
  Run(R"(
HloModule m
add_f32 {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT s = f32[] add(a, b)
}
ENTRY e {
  x = f32[8,64]{1,0} parameter(0)
  zero = f32[] constant(0)
  s = f32[8]{0} reduce(x, zero), dimensions={1}, to_apply=add_f32
  c = f32[] constant(2)
  cb = f32[8]{0} broadcast(c), dimensions={}
  ROOT y = f32[8]{0} multiply(s, cb)
})",
      /*expect_change=*/false);
}

TEST_F(MetalRowNormFusionTest, LeavesMajorDimensionReductionsAlone) {
  Run(R"(
HloModule m
add_f32 {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT s = f32[] add(a, b)
}
ENTRY e {
  x = f32[8,64]{1,0} parameter(0)
  zero = f32[] constant(0)
  s = f32[64]{0} reduce(x, zero), dimensions={0}, to_apply=add_f32
  sb = f32[8,64]{1,0} broadcast(s), dimensions={1}
  ROOT y = f32[8,64]{1,0} divide(x, sb)
})",
      /*expect_change=*/false);
}

TEST_F(MetalRowNormFusionTest, LeavesLongRowsAlone) {
  Run(kSoftmax, /*expect_change=*/false, /*max_row_length=*/32);
}

// Priority fusion, multi-output fusion and FusionWrapper (XLA's fusion
// stage, which runs after this pass) leave the fusion exactly as built.
TEST_F(MetalRowNormFusionTest, FusionStageLeavesFusionAlone) {
  constexpr absl::string_view kHlo = R"(
HloModule m
max_f32 {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT m = f32[] maximum(a, b)
}
add_f32 {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT s = f32[] add(a, b)
}
ENTRY e {
  p = f32[4,40]{1,0} parameter(0)
  q = f32[4,40]{1,0} parameter(1)
  x = f32[4,40]{1,0} add(p, q)
  ninf = f32[] constant(-inf)
  mx = f32[4]{0} reduce(x, ninf), dimensions={1}, to_apply=max_f32
  mxb = f32[4,40]{1,0} broadcast(mx), dimensions={0}
  sub = f32[4,40]{1,0} subtract(x, mxb)
  ex = f32[4,40]{1,0} exponential(sub)
  zero = f32[] constant(0)
  sm = f32[4]{0} reduce(ex, zero), dimensions={1}, to_apply=add_f32
  smb = f32[4,40]{1,0} broadcast(sm), dimensions={0}
  y = f32[4,40]{1,0} divide(ex, smb)
  t = f32[40,4]{1,0} transpose(y), dimensions={1,0}
  xs = f32[40] reduce(x, zero), dimensions={0}, to_apply=add_f32
  ROOT r = (f32[40,4]{1,0}, f32[40]{0}) tuple(t, xs)
})";
  std::unique_ptr<HloModule> module = Run(kHlo, /*expect_change=*/true);
  ASSERT_NE(module, nullptr);
  std::vector<const HloInstruction*> fusions = RowNormFusions(*module);
  ASSERT_EQ(fusions.size(), 1) << module->ToString();
  const std::string before =
      fusions[0]->fused_instructions_computation()->ToString();
  const int64_t operand_count = fusions[0]->operand_count();
  const std::string backend_config = fusions[0]->raw_backend_config_string();

  se::DeviceDescription device = TestGpuDeviceInfo::RTXA6000DeviceInfo();
  device.set_oneapi_compute_capability(0);
  GpuAliasInfo alias_info(device);
  mlir::MLIRContext mlir_context;
  HloPassPipeline pipeline = FusionPipeline(
      module->config().debug_options(),
      [](const Shape& shape) { return ShapeUtil::ByteSizeOf(shape, 8); },
      &alias_info, /*thread_pool=*/nullptr, device, &mlir_context);
  pipeline.AddPass<FusionWrapper>(device);
  auto changed = pipeline.Run(module.get());
  ASSERT_TRUE(changed.ok()) << changed.status();

  fusions = RowNormFusions(*module);
  ASSERT_EQ(fusions.size(), 1) << module->ToString();
  EXPECT_EQ(fusions[0]->fused_instructions_computation()->ToString(), before);
  EXPECT_EQ(fusions[0]->operand_count(), operand_count);
  EXPECT_EQ(fusions[0]->raw_backend_config_string(), backend_config);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
