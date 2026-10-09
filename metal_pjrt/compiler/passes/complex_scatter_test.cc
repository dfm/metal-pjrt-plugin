// Copyright 2026 The metal-pjrt-plugin Authors
// SPDX-License-Identifier: Apache-2.0

#include "metal_pjrt/compiler/passes/complex_scatter.h"

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

namespace xla {
namespace gpu {
namespace {

// A complex64 scatter-add of 6 rows of updates into 4 rows, with repeated
// indices. COMBINER, the combiner's root (or nothing: an overwrite), is
// substituted per test.
constexpr char kScatterHlo[] = R"(
HloModule m

combine {
  x = c64[] parameter(0)
  ROOT y = c64[] parameter(1)
  COMBINER
}

ENTRY e {
  z = c64[4,3] parameter(0)
  i = s32[6,1] parameter(1)
  u = c64[6,3] parameter(2)
  ROOT s = c64[4,3] scatter(z, i, u), update_window_dims={1},
      inserted_window_dims={0}, scatter_dims_to_operand_dims={0},
      index_vector_dim=1, to_apply=combine UNIQUE
}
)";

std::string Hlo(const std::string& combiner, bool unique = false) {
  std::string hlo = kScatterHlo;
  if (combiner.empty()) {
    hlo.replace(hlo.find("COMBINER"), 8, "");
  } else {
    hlo.replace(hlo.find("ROOT y"), 6, "y");
    hlo.replace(hlo.find("COMBINER"), 8, "ROOT r = " + combiner);
  }
  hlo.replace(hlo.find("UNIQUE"), 6, unique ? ", unique_indices=true" : "");
  return hlo;
}

class MetalComplexScatterSplitterTest : public HloHardwareIndependentTestBase {
 protected:
  // Small integers, so every sum is exact in f32 and the split module must
  // match the evaluator's complex scatter bit for bit. Indices repeat
  // (rows 0, 1 and 3 get two updates each, row 2 none).
  static Literal Values(const Shape& shape, int seed) {
    Literal lit(shape);
    int64_t i = seed;
    if (shape.element_type() == C64) {
      lit.EachCell<complex64>([&](absl::Span<const int64_t> idx, complex64) {
        lit.Set<complex64>(idx, complex64((i * 7) % 11 - 5, (i * 3) % 7 - 3));
        ++i;
      });
    } else {
      const int32_t rows[] = {0, 3, 1, 0, 3, 1};
      lit.EachCell<int32_t>([&](absl::Span<const int64_t> idx, int32_t) {
        lit.Set<int32_t>(idx, rows[idx[0]]);
      });
    }
    return lit;
  }
};

TEST_F(MetalComplexScatterSplitterTest, SplitsAddWithRepeatedIndices) {
  for (const char* combiner : {"c64[] add(x, y)", "c64[] add(y, x)"}) {
    auto module_or = ParseAndReturnVerifiedModule(Hlo(combiner));
    ASSERT_TRUE(module_or.ok()) << module_or.status();
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
    ASSERT_TRUE(expected.ok()) << expected.status();

    MetalComplexScatterSplitter pass;
    auto changed = RunHloPass(&pass, module.get());
    ASSERT_TRUE(changed.ok()) << changed.status();
    EXPECT_TRUE(*changed) << combiner;
    EXPECT_TRUE(verifier().Run(module.get()).ok());
    HloEvaluator after;
    auto actual = after.Evaluate(*module, arg_ptrs);
    ASSERT_TRUE(actual.ok()) << actual.status();
    EXPECT_EQ(*expected, *actual) << combiner << "\nexpected "
                                  << expected->ToString() << "\nactual "
                                  << actual->ToString();

    int scatters = 0;
    for (const HloInstruction* instr : entry->instructions()) {
      if (instr->opcode() == HloOpcode::kScatter) {
        EXPECT_EQ(instr->shape().element_type(), F32) << instr->ToString();
        ++scatters;
      }
    }
    EXPECT_EQ(scatters, 2) << combiner;
    EXPECT_EQ(entry->root_instruction()->opcode(), HloOpcode::kComplex);
  }
}

TEST_F(MetalComplexScatterSplitterTest, LeavesOtherScattersAlone) {
  // Overwrite (lowered as one 8-byte store instead), multiply (not
  // componentwise), and unique indices (no atomics needed).
  for (const auto& [combiner, unique] :
       std::vector<std::pair<std::string, bool>>{
           {"", false},
           {"c64[] multiply(x, y)", false},
           {"c64[] add(x, y)", true}}) {
    auto module_or = ParseAndReturnVerifiedModule(Hlo(combiner, unique));
    ASSERT_TRUE(module_or.ok()) << module_or.status();
    MetalComplexScatterSplitter pass;
    auto changed = RunHloPass(&pass, module_or->get());
    ASSERT_TRUE(changed.ok()) << changed.status();
    EXPECT_FALSE(*changed) << combiner << " unique=" << unique;
  }
}

}  // namespace
}  // namespace gpu
}  // namespace xla
