#include "metal_pjrt_plugin/compiler/passes/sort_expander.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/shape_util.h"

namespace xla {
namespace gpu {
namespace {

class MetalSortExpanderTest : public HloHardwareIndependentTestBase {
 protected:
  // Literal of `shape` filled with values from `gen(i)`, i = flat index.
  template <typename T, typename Gen>
  static Literal MakeLiteral(const Shape& shape, Gen gen) {
    Literal lit(shape);
    int64_t i = 0;
    lit.EachCell<T>([&](absl::Span<const int64_t> idx, T) {
      lit.Set<T>(idx, gen(i++));
    });
    return lit;
  }

  void RunAndCompare(const std::string& hlo,
                     const std::vector<Literal>& args) {
    auto module_or = ParseAndReturnVerifiedModule(hlo);
    ASSERT_TRUE(module_or.ok()) << module_or.status();
    std::unique_ptr<HloModule> module = std::move(module_or).value();
    std::vector<const Literal*> arg_ptrs;
    for (const Literal& a : args) arg_ptrs.push_back(&a);

    HloEvaluator before;
    auto expected = before.Evaluate(*module, arg_ptrs);
    ASSERT_TRUE(expected.ok()) << expected.status();

    MetalSortExpander pass;
    auto changed = RunHloPass(&pass, module.get());
    ASSERT_TRUE(changed.ok()) << changed.status();
    EXPECT_TRUE(*changed);
    for (const HloComputation* c : module->computations()) {
      for (const HloInstruction* i : c->instructions()) {
        EXPECT_NE(i->opcode(), HloOpcode::kSort) << i->ToString();
      }
    }
    ASSERT_TRUE(verifier().Run(module.get()).ok());

    HloEvaluator after;
    auto actual = after.Evaluate(*module, arg_ptrs);
    ASSERT_TRUE(actual.ok()) << actual.status();
    EXPECT_EQ(*expected, *actual) << "expected " << expected->ToString()
                                  << "\nactual " << actual->ToString();
  }

  // Distinct float values in shuffled order.
  static std::vector<float> Shuffled(int64_t n, uint32_t seed) {
    std::vector<float> v(n);
    for (int64_t i = 0; i < n; ++i) v[i] = static_cast<float>(i) * 0.5f - 7.f;
    std::mt19937 rng(seed);
    std::shuffle(v.begin(), v.end(), rng);
    return v;
  }
};

TEST_F(MetalSortExpanderTest, OneDimNonPowerOfTwo) {
  for (int64_t n : {2, 3, 16, 37, 64, 100}) {
    std::string hlo = absl::StrReplaceAll(R"(
HloModule m
lt {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT c = pred[] compare(a, b), direction=LT
}
ENTRY e {
  p = f32[$N] parameter(0)
  ROOT s = f32[$N] sort(p), dimensions={0}, to_apply=lt
})", {{"$N", absl::StrCat(n)}});
    auto v = Shuffled(n, n);
    std::vector<Literal> args;
    args.push_back(MakeLiteral<float>(ShapeUtil::MakeShape(F32, {n}),
                                      [&](int64_t i) { return v[i]; }));
    RunAndCompare(hlo, args);
  }
}

// Sort dimensions of up to 64 elements are emitted straight-line (no while
// loop); larger ones run the loop.
TEST_F(MetalSortExpanderTest, SmallSortsAreUnrolled) {
  for (int64_t n : {4, 64, 65}) {
    std::string hlo = absl::StrReplaceAll(R"(
HloModule m
lt {
  a = s32[] parameter(0)
  b = s32[] parameter(1)
  c = s32[] parameter(2)
  d = s32[] parameter(3)
  ROOT r = pred[] compare(a, b), direction=LT
}
ENTRY e {
  k = s32[7,$N] parameter(0)
  v = s32[7,$N] parameter(1)
  ROOT s = (s32[7,$N], s32[7,$N]) sort(k, v), dimensions={1}, is_stable=true,
      to_apply=lt
})", {{"$N", absl::StrCat(n)}});
    auto module_or = ParseAndReturnVerifiedModule(hlo);
    ASSERT_TRUE(module_or.ok()) << module_or.status();
    std::unique_ptr<HloModule> module = std::move(module_or).value();
    MetalSortExpander pass;
    ASSERT_TRUE(RunHloPass(&pass, module.get()).ok());
    int whiles = 0;
    for (const HloComputation* c : module->computations()) {
      for (const HloInstruction* i : c->instructions()) {
        whiles += i->opcode() == HloOpcode::kWhile;
      }
    }
    EXPECT_EQ(whiles, n <= 64 ? 0 : 1) << "n=" << n;
    // Still sorts correctly.
    std::mt19937 rng(n);
    std::vector<Literal> args;
    args.push_back(MakeLiteral<int32_t>(ShapeUtil::MakeShape(S32, {7, n}),
                                        [&](int64_t) { return rng() % 5; }));
    args.push_back(MakeLiteral<int32_t>(ShapeUtil::MakeShape(S32, {7, n}),
                                        [&](int64_t i) { return i; }));
    RunAndCompare(hlo, args);
  }
}

TEST_F(MetalSortExpanderTest, KeyValueStableWithDuplicates) {
  const char* hlo = R"(
HloModule m
lt {
  a = s32[] parameter(0)
  b = s32[] parameter(1)
  c = s32[] parameter(2)
  d = s32[] parameter(3)
  ROOT r = pred[] compare(a, b), direction=LT
}
ENTRY e {
  k = s32[5,19] parameter(0)
  v = s32[5,19] parameter(1)
  ROOT s = (s32[5,19], s32[5,19]) sort(k, v), dimensions={1}, is_stable=true,
      to_apply=lt
})";
  std::mt19937 rng(3);
  std::vector<Literal> args;
  args.push_back(MakeLiteral<int32_t>(ShapeUtil::MakeShape(S32, {5, 19}),
                                      [&](int64_t) { return rng() % 4; }));
  args.push_back(MakeLiteral<int32_t>(ShapeUtil::MakeShape(S32, {5, 19}),
                                      [&](int64_t i) { return i; }));
  RunAndCompare(hlo, args);
}

// A non-strict comparator (LE) says both a <= b and b <= a for equal keys.
// The network must still move a permutation of the input: keys sorted as
// the evaluator's, values the same multiset per row (their order among
// equal keys is unspecified for such a comparator).
TEST_F(MetalSortExpanderTest, NonStrictComparatorKeepsAPermutation) {
  const char* hlo = R"(
HloModule m
le {
  a = s32[] parameter(0)
  b = s32[] parameter(1)
  c = s32[] parameter(2)
  d = s32[] parameter(3)
  ROOT r = pred[] compare(a, b), direction=LE
}
ENTRY e {
  k = s32[5,19] parameter(0)
  v = s32[5,19] parameter(1)
  ROOT s = (s32[5,19], s32[5,19]) sort(k, v), dimensions={1}, to_apply=le
})";
  std::mt19937 rng(5);
  std::vector<Literal> args;
  args.push_back(MakeLiteral<int32_t>(ShapeUtil::MakeShape(S32, {5, 19}),
                                      [&](int64_t) { return rng() % 4; }));
  args.push_back(MakeLiteral<int32_t>(ShapeUtil::MakeShape(S32, {5, 19}),
                                      [&](int64_t i) { return i; }));
  auto module_or = ParseAndReturnVerifiedModule(hlo);
  ASSERT_TRUE(module_or.ok()) << module_or.status();
  std::unique_ptr<HloModule> module = std::move(module_or).value();
  std::vector<const Literal*> arg_ptrs = {&args[0], &args[1]};
  HloEvaluator before;
  auto expected = before.Evaluate(*module, arg_ptrs);
  ASSERT_TRUE(expected.ok()) << expected.status();
  MetalSortExpander pass;
  ASSERT_TRUE(RunHloPass(&pass, module.get()).ok());
  HloEvaluator after;
  auto actual = after.Evaluate(*module, arg_ptrs);
  ASSERT_TRUE(actual.ok()) << actual.status();
  std::vector<Literal> want = expected->DecomposeTuple();
  std::vector<Literal> got = actual->DecomposeTuple();
  EXPECT_EQ(want[0], got[0]) << "expected " << want[0].ToString()
                             << "\nactual " << got[0].ToString();
  const Literal& values = got[1];
  for (int64_t r = 0; r < 5; ++r) {
    std::vector<int32_t> row;
    for (int64_t c = 0; c < 19; ++c) row.push_back(values.Get<int32_t>({r, c}));
    std::sort(row.begin(), row.end());
    for (int64_t c = 0; c < 19; ++c) EXPECT_EQ(row[c], r * 19 + c) << "row " << r;
  }
}

TEST_F(MetalSortExpanderTest, MajorSortDimDescending) {
  const char* hlo = R"(
HloModule m
gt {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT c = pred[] compare(a, b), direction=GT
}
ENTRY e {
  p = f32[3,10,4] parameter(0)
  ROOT s = f32[3,10,4] sort(p), dimensions={1}, to_apply=gt
})";
  auto v = Shuffled(120, 7);
  std::vector<Literal> args;
  args.push_back(MakeLiteral<float>(ShapeUtil::MakeShape(F32, {3, 10, 4}),
                                    [&](int64_t i) { return v[i]; }));
  RunAndCompare(hlo, args);
}

// JAX-style total-order comparator after ComparisonExpander, with an
// argsort-style iota operand.
TEST_F(MetalSortExpanderTest, TotalOrderArgsort) {
  const char* hlo = R"(
HloModule m
cmp {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  i = s32[] parameter(2)
  j = s32[] parameter(3)
  ai = s32[] bitcast-convert(a)
  bi = s32[] bitcast-convert(b)
  z = s32[] constant(0)
  mx = s32[] constant(2147483647)
  an = s32[] xor(ai, mx)
  bn = s32[] xor(bi, mx)
  alt = pred[] compare(ai, z), direction=LT
  blt = pred[] compare(bi, z), direction=LT
  ak = s32[] select(alt, an, ai)
  bk = s32[] select(blt, bn, bi)
  ROOT r = pred[] compare(ak, bk), direction=LT
}
ENTRY e {
  p = f32[2,33] parameter(0)
  io = s32[2,33] iota(), iota_dimension=1
  s = (f32[2,33], s32[2,33]) sort(p, io), dimensions={1}, is_stable=true,
      to_apply=cmp
  ROOT g = s32[2,33] get-tuple-element(s), index=1
})";
  std::mt19937 rng(11);
  std::uniform_int_distribution<int> dist(-5, 5);
  std::vector<Literal> args;
  args.push_back(MakeLiteral<float>(ShapeUtil::MakeShape(F32, {2, 33}),
                                    [&](int64_t) { return dist(rng) * 0.25f; }));
  RunAndCompare(hlo, args);
}

// A comparator with a non-elementwise op (call) goes through kMap.
TEST_F(MetalSortExpanderTest, ComparatorWithCall) {
  const char* hlo = R"(
HloModule m
inner {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  ROOT c = pred[] compare(x, y), direction=LT
}
lt {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT c = pred[] call(a, b), to_apply=inner
}
ENTRY e {
  p = f32[4,9] parameter(0)
  ROOT s = f32[4,9] sort(p), dimensions={0}, to_apply=lt
})";
  auto v = Shuffled(36, 5);
  std::vector<Literal> args;
  args.push_back(MakeLiteral<float>(ShapeUtil::MakeShape(F32, {4, 9}),
                                    [&](int64_t i) { return v[i]; }));
  RunAndCompare(hlo, args);
}

TEST_F(MetalSortExpanderTest, TrivialDimension) {
  const char* hlo = R"(
HloModule m
lt {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT c = pred[] compare(a, b), direction=LT
}
ENTRY e {
  p = f32[6,1] parameter(0)
  ROOT s = f32[6,1] sort(p), dimensions={1}, to_apply=lt
})";
  std::vector<Literal> args;
  args.push_back(MakeLiteral<float>(ShapeUtil::MakeShape(F32, {6, 1}),
                                    [&](int64_t i) { return 6.f - i; }));
  RunAndCompare(hlo, args);
}

// The RunHloPasses filter: only rows of <= 64 in sorts of more than 16384
// elements (the ones SortRewriter would otherwise take).
TEST_F(MetalSortExpanderTest, FilterKeepsLongRowsAndSmallSorts) {
  const char* hlo = R"(
HloModule m
lt {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT c = pred[] compare(a, b), direction=LT
}
ENTRY e {
  tiny_rows = f32[1000,32] parameter(0)
  long_rows = f32[200,100] parameter(1)
  small = f32[100,32] parameter(2)
  sort_a = f32[1000,32] sort(tiny_rows), dimensions={1}, to_apply=lt
  sort_b = f32[200,100] sort(long_rows), dimensions={1}, to_apply=lt
  sort_c = f32[100,32] sort(small), dimensions={1}, to_apply=lt
  ROOT t = (f32[1000,32], f32[200,100], f32[100,32]) tuple(sort_a, sort_b, sort_c)
})";
  auto module = ParseAndReturnVerifiedModule(hlo);
  ASSERT_TRUE(module.ok()) << module.status();
  MetalSortExpander pass(/*max_sort_dim=*/64, /*min_elements=*/16384);
  auto changed = RunHloPass(&pass, module->get());
  ASSERT_TRUE(changed.ok()) << changed.status();
  EXPECT_TRUE(*changed);
  std::vector<std::string> left;
  for (const HloInstruction* i :
       (*module)->entry_computation()->instructions()) {
    if (i->opcode() == HloOpcode::kSort) left.push_back(std::string(i->name()));
  }
  std::sort(left.begin(), left.end());
  EXPECT_EQ(left, (std::vector<std::string>{"sort_b", "sort_c"}));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
