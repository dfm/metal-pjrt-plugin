#include "metal_pjrt/compiler/passes/complex_dot.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/literal.h"
#include "xla/literal_util.h"

namespace xla {
namespace gpu {
namespace {

class MetalComplexDotExpanderTest : public HloHardwareIndependentTestBase {
 protected:
  // Small integers: every product and sum is exact in f32, so the expanded
  // module must match the evaluator's complex dot bit for bit.
  static Literal Values(const Shape& shape, int seed) {
    Literal lit(shape);
    int64_t i = seed;
    if (shape.element_type() == C64) {
      lit.EachCell<complex64>([&](absl::Span<const int64_t> idx, complex64) {
        lit.Set<complex64>(idx, complex64((i * 7) % 11 - 5, (i * 3) % 7 - 3));
        ++i;
      });
    } else {
      lit.EachCell<float>([&](absl::Span<const int64_t> idx, float) {
        lit.Set<float>(idx, (i++ * 5) % 9 - 4);
      });
    }
    return lit;
  }

  // Runs the pass on `hlo` and checks the expanded module computes the same
  // values; returns the dots left in the entry computation.
  std::vector<const HloInstruction*> ExpandAndCompare(const std::string& hlo) {
    auto module_or = ParseAndReturnVerifiedModule(hlo);
    EXPECT_TRUE(module_or.ok()) << module_or.status();
    std::unique_ptr<HloModule> module = std::move(module_or).value();
    const HloComputation* entry = module->entry_computation();
    std::vector<Literal> args;
    std::vector<const Literal*> arg_ptrs;
    for (int i = 0; i < entry->num_parameters(); ++i) {
      args.push_back(Values(entry->parameter_instruction(i)->shape(), i * 13));
    }
    for (const Literal& a : args) arg_ptrs.push_back(&a);
    HloEvaluator before;
    auto expected = before.Evaluate(*module, arg_ptrs);
    EXPECT_TRUE(expected.ok()) << expected.status();

    MetalComplexDotExpander pass;
    auto changed = RunHloPass(&pass, module.get());
    EXPECT_TRUE(changed.ok()) << changed.status();
    EXPECT_TRUE(*changed);
    EXPECT_TRUE(verifier().Run(module.get()).ok());
    HloEvaluator after;
    auto actual = after.Evaluate(*module, arg_ptrs);
    EXPECT_TRUE(actual.ok()) << actual.status();
    EXPECT_EQ(*expected, *actual) << "expected " << expected->ToString()
                                  << "\nactual " << actual->ToString();

    std::vector<const HloInstruction*> dots;
    for (const HloInstruction* instr : entry->instructions()) {
      if (instr->opcode() == HloOpcode::kDot) {
        EXPECT_EQ(instr->shape().element_type(), F32) << instr->ToString();
        dots.push_back(instr);
      }
    }
    kept_ = std::move(module);
    return dots;
  }

  std::unique_ptr<HloModule> kept_;
};

TEST_F(MetalComplexDotExpanderTest, BatchedDotKeepsDimensionsAndPrecision) {
  auto dots = ExpandAndCompare(R"(
HloModule m
ENTRY e {
  a = c64[3,4,5] parameter(0)
  b = c64[3,6,5] parameter(1)
  ROOT d = c64[3,4,6] dot(a, b), lhs_batch_dims={0}, rhs_batch_dims={0}, lhs_contracting_dims={2}, rhs_contracting_dims={2}, operand_precision={highest,highest}, metadata={op_name="jit(f)/dot_general"}
})");
  ASSERT_EQ(dots.size(), 4);
  for (const HloInstruction* dot : dots) {
    EXPECT_EQ(dot->dot_dimension_numbers().lhs_batch_dimensions(0), 0);
    EXPECT_EQ(dot->dot_dimension_numbers().rhs_contracting_dimensions(0), 2);
    EXPECT_EQ(dot->precision_config().operand_precision(0),
              PrecisionConfig::HIGHEST);
    EXPECT_EQ(dot->metadata().op_name(), "jit(f)/dot_general");
  }
}

TEST_F(MetalComplexDotExpanderTest, ContractingDimFirst) {
  auto dots = ExpandAndCompare(R"(
HloModule m
ENTRY e {
  a = c64[5,4] parameter(0)
  b = c64[5,3] parameter(1)
  ROOT d = c64[4,3] dot(a, b), lhs_contracting_dims={0}, rhs_contracting_dims={0}
})");
  EXPECT_EQ(dots.size(), 4);
}

// A real operand has no imaginary products: two dots.
TEST_F(MetalComplexDotExpanderTest, MixedRealOperand) {
  auto dots = ExpandAndCompare(R"(
HloModule m
ENTRY e {
  a = c64[4,5] parameter(0)
  b = f32[5,3] parameter(1)
  ROOT d = c64[4,3] dot(a, b), lhs_contracting_dims={1}, rhs_contracting_dims={0}
})");
  EXPECT_EQ(dots.size(), 2);
}

// Two real operands with a complex64 result
// (jnp.matmul(..., preferred_element_type=complex64)): one f32 dot and a
// zero imaginary part.
TEST_F(MetalComplexDotExpanderTest, RealOperandsComplexResult) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule m
ENTRY e {
  a = f32[4,5] parameter(0)
  b = f32[5,3] parameter(1)
  ROOT d = c64[4,3] dot(a, b), lhs_contracting_dims={1}, rhs_contracting_dims={0}
})");
  ASSERT_TRUE(module.ok()) << module.status();
  MetalComplexDotExpander pass;
  auto changed = RunHloPass(&pass, module->get());
  ASSERT_TRUE(changed.ok()) << changed.status();
  EXPECT_TRUE(*changed);
  EXPECT_TRUE(verifier().Run(module->get()).ok());
  const HloComputation* entry = (*module)->entry_computation();
  EXPECT_EQ(entry->root_instruction()->opcode(), HloOpcode::kComplex);
  int dots = 0;
  for (const HloInstruction* instr : entry->instructions()) {
    if (instr->opcode() != HloOpcode::kDot) continue;
    EXPECT_EQ(instr->shape().element_type(), F32) << instr->ToString();
    ++dots;
  }
  EXPECT_EQ(dots, 1);
}

// complex128 stays (CheckPostGemmRewriter refuses it with the f64 ops).
TEST_F(MetalComplexDotExpanderTest, LeavesComplex128Alone) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule m
ENTRY e {
  a = c128[4,5] parameter(0)
  b = c128[5,3] parameter(1)
  ROOT d = c128[4,3] dot(a, b), lhs_contracting_dims={1}, rhs_contracting_dims={0}
})");
  ASSERT_TRUE(module.ok()) << module.status();
  MetalComplexDotExpander pass;
  auto changed = RunHloPass(&pass, module->get());
  ASSERT_TRUE(changed.ok()) << changed.status();
  EXPECT_FALSE(*changed);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
