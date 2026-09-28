#include <memory>

#include <gtest/gtest.h>
#include "metal_pjrt/compiler/passes/dot_upcast.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"

namespace xla {
namespace gpu {
namespace {

using MetalDotOperandUpcasterTest = HloHardwareIndependentTestBase;

TEST_F(MetalDotOperandUpcasterTest, UpcastsNarrowOperands) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule m
ENTRY e {
  a = bf16[4,3]{1,0} parameter(0)
  b = bf16[3,6]{1,0} parameter(1)
  ROOT d = f32[4,6]{1,0} dot(a, b), lhs_contracting_dims={1}, rhs_contracting_dims={0}
})");
  ASSERT_TRUE(module.ok()) << module.status();
  MetalDotOperandUpcaster pass;
  auto changed = RunHloPass(&pass, module->get());
  ASSERT_TRUE(changed.ok()) << changed.status();
  EXPECT_TRUE(*changed);
  const HloInstruction* fusion =
      (*module)->entry_computation()->root_instruction();
  ASSERT_EQ(fusion->opcode(), HloOpcode::kFusion);
  EXPECT_EQ(fusion->fusion_kind(), HloInstruction::FusionKind::kLoop);
  EXPECT_EQ(fusion->operand(0)->shape().element_type(), BF16);
  const HloInstruction* dot = fusion->fused_expression_root();
  ASSERT_EQ(dot->opcode(), HloOpcode::kDot);
  for (const HloInstruction* op : dot->operands()) {
    EXPECT_EQ(op->opcode(), HloOpcode::kConvert);
    EXPECT_EQ(op->shape().element_type(), F32);
    EXPECT_EQ(op->shape().layout(), op->operand(0)->shape().layout());
  }
  EXPECT_FALSE(IsNarrowOperandDot(dot));
}

// DotAlgorithmRewriter's output for ALG_DOT_BF16_BF16_F32(_X3/_X6/_X9).
TEST_F(MetalDotOperandUpcasterTest, UpcastsBf16AlgorithmDots) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule m
ENTRY e {
  a = bf16[4,3]{1,0} parameter(0)
  b = bf16[3,6]{1,0} parameter(1)
  ROOT d = f32[4,6]{1,0} dot(a, b), lhs_contracting_dims={1}, rhs_contracting_dims={0}, algorithm=dot_bf16_bf16_f32
})");
  ASSERT_TRUE(module.ok()) << module.status();
  MetalDotOperandUpcaster pass;
  auto changed = RunHloPass(&pass, module->get());
  ASSERT_TRUE(changed.ok()) << changed.status();
  EXPECT_TRUE(*changed);
  const HloInstruction* dot =
      (*module)->entry_computation()->root_instruction()
          ->fused_expression_root();
  ASSERT_EQ(dot->opcode(), HloOpcode::kDot);
  EXPECT_EQ(dot->operand(0)->opcode(), HloOpcode::kConvert);
  EXPECT_EQ(dot->operand(1)->opcode(), HloOpcode::kConvert);
  EXPECT_EQ(dot->operand(0)->shape().element_type(), F32);
}

TEST_F(MetalDotOperandUpcasterTest, UpcastsIntegerAndOnlyNarrowSide) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule m
ENTRY e {
  a = s8[4,3]{1,0} parameter(0)
  b = s32[3,6]{1,0} parameter(1)
  ROOT d = s32[4,6]{1,0} dot(a, b), lhs_contracting_dims={1}, rhs_contracting_dims={0}
})");
  ASSERT_TRUE(module.ok()) << module.status();
  MetalDotOperandUpcaster pass;
  auto changed = RunHloPass(&pass, module->get());
  ASSERT_TRUE(changed.ok()) << changed.status();
  EXPECT_TRUE(*changed);
  const HloInstruction* dot =
      (*module)->entry_computation()->root_instruction()
          ->fused_expression_root();
  EXPECT_EQ(dot->operand(0)->opcode(), HloOpcode::kConvert);
  EXPECT_EQ(dot->operand(1)->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(dot->operand(1)->shape().element_type(), S32);
}

TEST_F(MetalDotOperandUpcasterTest, LeavesSameTypeAndGemmCallsAlone) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule m
ENTRY e {
  a = bf16[4,3]{1,0} parameter(0)
  b = bf16[3,6]{1,0} parameter(1)
  d = bf16[4,6]{1,0} dot(a, b), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  g = (f32[4,6]{1,0}, s8[0]{0}) custom-call(a, b), custom_call_target="__cublas$lt$matmul"
  g0 = f32[4,6]{1,0} get-tuple-element(g), index=0
  ROOT t = (bf16[4,6]{1,0}, f32[4,6]{1,0}) tuple(d, g0)
})");
  ASSERT_TRUE(module.ok()) << module.status();
  MetalDotOperandUpcaster pass;
  auto changed = RunHloPass(&pass, module->get());
  ASSERT_TRUE(changed.ok()) << changed.status();
  EXPECT_FALSE(*changed);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
