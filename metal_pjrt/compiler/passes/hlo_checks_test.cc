#include <cstdint>
#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "metal_pjrt/blas/blas_lt_support.h"
#include "metal_pjrt/compiler/passes/dot_upcast.h"
#include "metal_pjrt/compiler/passes/hlo_checks.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/stream_executor/gpu/gpu_blas_lt.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

class MetalHloChecksTest : public HloHardwareIndependentTestBase {
 protected:
  absl::Status Check(const std::string& hlo) {
    auto module = ParseAndReturnUnverifiedModule(hlo);
    EXPECT_TRUE(module.ok()) << module.status();
    if (!module.ok()) return module.status();
    return CheckPostGemmRewriter(**module);
  }
};

// A GEMM as GemmRewriter emits it; $A/$B/$O element types, $E epilogue.
constexpr char kGemm[] = R"(
HloModule m
ENTRY e {
  a = $A[64,32]{1,0} parameter(0)
  b = $B[32,16]{1,0} parameter(1)
  g = ($O[64,16]{1,0}, s8[0]{0}) custom-call(a, b), custom_call_target="__cublas$$lt$$matmul", metadata={op_name="jit(f)/dot_general" source_file="model.py" source_line=12}, backend_config={"gemm_backend_config":{"alpha_real":1,"beta":0,"dot_dimension_numbers":{"lhs_contracting_dimensions":["1"],"rhs_contracting_dimensions":["0"],"lhs_batch_dimensions":[],"rhs_batch_dimensions":[]},"epilogue":"$E"}}
  ROOT r = $O[64,16]{1,0} get-tuple-element(g), index=0
})";

std::string Gemm(const std::string& a, const std::string& b,
                 const std::string& o, const std::string& epilogue) {
  return absl::StrReplaceAll(
      kGemm, {{"$A", a}, {"$B", b}, {"$O", o}, {"$E", epilogue},
              {"$$", "$"}});
}

TEST_F(MetalHloChecksTest, SupportedGemms) {
  EXPECT_TRUE(Check(Gemm("f32", "f32", "f32", "DEFAULT")).ok());
  EXPECT_TRUE(Check(Gemm("bf16", "bf16", "f32", "DEFAULT")).ok());
  EXPECT_TRUE(Check(Gemm("f16", "f16", "f16", "RELU")).ok());
}

TEST_F(MetalHloChecksTest, UnsupportedGemmTypeNamesTheOp) {
  absl::Status s = Check(Gemm("s8", "s8", "s32", "DEFAULT"));
  EXPECT_EQ(s.code(), absl::StatusCode::kUnimplemented);
  EXPECT_NE(s.message().find("S8 x S8 -> S32"), std::string::npos) << s;
  EXPECT_NE(s.message().find("jit(f)/dot_general"), std::string::npos) << s;
  EXPECT_NE(s.message().find("model.py:12"), std::string::npos) << s;
  EXPECT_FALSE(Check(Gemm("f16", "bf16", "f32", "DEFAULT")).ok());
  EXPECT_FALSE(Check(Gemm("f32", "f32", "bf16", "DEFAULT")).ok());
}

// The index limits of the kernel that would run the GEMM (ValidateMatmul),
// checked on the HLO shapes without allocating anything.
TEST_F(MetalHloChecksTest, GemmIndexLimits) {
  // f16/bf16: steel's 32-bit index math; K = 2^23 + 16 puts A's leading
  // dimension past kSteelMaxLd. MPS (f32) has no such limit.
  auto big_k = [](const std::string& hlo) {
    return absl::StrReplaceAll(
        hlo, {{"[64,32]", "[64,8388624]"}, {"[32,16]", "[8388624,16]"}});
  };
  absl::Status s = Check(big_k(Gemm("f16", "f16", "f16", "DEFAULT")));
  EXPECT_EQ(s.code(), absl::StatusCode::kUnimplemented);
  EXPECT_NE(s.message().find("leading dimension"), std::string::npos) << s;
  EXPECT_NE(s.message().find("jit(f)/dot_general"), std::string::npos) << s;
  EXPECT_TRUE(Check(big_k(Gemm("f32", "f32", "f32", "DEFAULT"))).ok());
}

// XLA's GemmConfig::For multiplies each dimension group in an int
// (GetBatchRowColumnShape), so 2^32 rows would become a 0-row GEMM and
// 2^31 a negative one: refused on the HLO's own dimensions.
TEST_F(MetalHloChecksTest, GemmGroupsPast32BitsAreRefused) {
  for (const char* rows : {"2147483648", "4294967296"}) {
    std::string hlo = absl::StrReplaceAll(
        Gemm("f32", "f32", "f32", "DEFAULT"),
        {{"[64,32]", absl::StrCat("[", rows, ",32]")},
         {"[64,16]", absl::StrCat("[", rows, ",16]")}});
    absl::Status s = Check(hlo);
    EXPECT_EQ(s.code(), absl::StatusCode::kUnimplemented) << rows;
    EXPECT_NE(s.message().find(absl::StrCat("group of ", rows)),
              std::string::npos)
        << s;
    EXPECT_NE(s.message().find("jit(f)/dot_general"), std::string::npos) << s;
  }
  EXPECT_TRUE(Check(absl::StrReplaceAll(Gemm("f32", "f32", "f32", "DEFAULT"),
                                        {{"[64,32]", "[2147483647,32]"},
                                         {"[64,16]", "[2147483647,16]"}}))
                  .ok());
}

// f32 with an epilogue: the epilogue kernel's grid is (n, m, batch), each
// bounded by uint32. Not reachable from HLO (the check above refuses rows
// past int32 first), so ValidateMatmul is called directly: a backstop for
// the kernel's uint32 casts.
TEST(ValidateMatmulTest, F32EpilogueGridLimit) {
  namespace se_gpu = stream_executor::gpu;
  using Order = se_gpu::MatrixLayout::Order;
  const int64_t m = int64_t{1} << 32;
  const se_gpu::MatrixLayout out(F32, m, 16, Order::kRowMajor);
  se_gpu::GemmConfig cfg{se_gpu::MatrixLayout(F32, m, 32, Order::kRowMajor),
                         se_gpu::MatrixLayout(F32, 32, 16, Order::kRowMajor),
                         out, out};
  cfg.alpha = 1.0;
  cfg.beta = 0.0;
  auto v = stream_executor::metal::ValidateMatmul(
      cfg, se_gpu::BlasLt::Epilogue::kReLU);
  EXPECT_EQ(v.status().code(), absl::StatusCode::kUnimplemented);
  EXPECT_NE(v.status().message().find("epilogue on a 1x4294967296x16"),
            std::string::npos)
      << v.status();
  EXPECT_TRUE(stream_executor::metal::ValidateMatmul(
                  cfg, se_gpu::BlasLt::Epilogue::kDefault)
                  .ok());
}

TEST_F(MetalHloChecksTest, NarrowOperandDot) {
  constexpr char kDot[] = R"(
HloModule m
ENTRY e {
  a = bf16[4,3]{1,0} parameter(0)
  b = bf16[3,6]{1,0} parameter(1)
  ROOT d = f32[4,6]{1,0} dot(a, b), lhs_contracting_dims={1}, rhs_contracting_dims={0}
})";
  EXPECT_EQ(Check(kDot).code(), absl::StatusCode::kInternal);

  // After MetalDotOperandUpcaster the dot is fine.
  auto module = ParseAndReturnVerifiedModule(kDot);
  ASSERT_TRUE(module.ok());
  MetalDotOperandUpcaster pass;
  ASSERT_TRUE(RunHloPass(&pass, module->get()).ok());
  EXPECT_TRUE(CheckPostGemmRewriter(**module).ok());
}

// The upcaster's fusion after priority fusion merged a producer and a
// consumer into it.
TEST_F(MetalHloChecksTest, UpcastDotInsideLargerFusionIsFine) {
  EXPECT_TRUE(Check(R"(
HloModule m
fused {
  p0 = bf16[4,3]{1,0} parameter(0)
  p1 = bf16[3,6]{1,0} parameter(1)
  s = bf16[4,3]{1,0} sine(p0)
  c0 = f32[4,3]{1,0} convert(s)
  c1 = f32[3,6]{1,0} convert(p1)
  d = f32[4,6]{1,0} dot(c0, c1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  one = f32[] constant(1)
  ob = f32[4,6]{1,0} broadcast(one), dimensions={}
  ROOT t = f32[4,6]{1,0} add(d, ob)
}
ENTRY e {
  a = bf16[4,3]{1,0} parameter(0)
  b = bf16[3,6]{1,0} parameter(1)
  ROOT f = f32[4,6]{1,0} fusion(a, b), kind=kLoop, calls=fused
})").ok());
}

TEST_F(MetalHloChecksTest, LegalityF64) {
  auto legal = [&](const std::string& hlo) {
    auto module = ParseAndReturnUnverifiedModule(hlo);
    EXPECT_TRUE(module.ok()) << module.status();
    return CheckPostGemmRewriter(**module);
  };
  // Data movement on f64 is legal.
  EXPECT_TRUE(legal(R"(
HloModule m
ENTRY e {
  x = f64[3,4] parameter(0)
  t = f64[4,3] transpose(x), dimensions={1,0}
  r = f64[12] reshape(t)
  s = f64[5] slice(r), slice={[2:7]}
  p = pred[5] parameter(1)
  ROOT sel = f64[5] select(p, s, s)
})").ok());
  absl::Status s = legal(R"(
HloModule m
ENTRY e {
  x = f64[3] parameter(0)
  ROOT y = f64[3] add(x, x), metadata={op_name="jit(f)/add" source_file="f.py" source_line=3}
})");
  EXPECT_EQ(s.code(), absl::StatusCode::kUnimplemented);
  EXPECT_NE(s.message().find("f64"), std::string::npos) << s;
  EXPECT_NE(s.message().find("jit(f)/add"), std::string::npos) << s;
  EXPECT_NE(s.message().find("f.py:3"), std::string::npos) << s;
  // Converting to f64 computes too.
  EXPECT_FALSE(legal(R"(
HloModule m
ENTRY e {
  x = f32[3] parameter(0)
  ROOT y = f64[3] convert(x)
})").ok());
}

TEST_F(MetalHloChecksTest, LegalityWideScatter) {
  constexpr char kScatter[] = R"(
HloModule m
comb {
  a = $T[] parameter(0)
  $COMB
}
ENTRY e {
  x = $T[4] parameter(0)
  i = s32[3,1] parameter(1)
  u = $T[3] parameter(2)
  ROOT s = $T[4] scatter(x, i, u), update_window_dims={}, inserted_window_dims={0}, scatter_dims_to_operand_dims={0}, index_vector_dim=1, to_apply=comb
})";
  auto legal = [&](const std::string& t, const std::string& comb) {
    auto module = ParseAndReturnUnverifiedModule(absl::StrReplaceAll(
        kScatter, {{"$T", t}, {"$COMB", comb}}));
    EXPECT_TRUE(module.ok()) << module.status();
    return CheckPostGemmRewriter(**module);
  };
  auto op = [](const std::string& t, const std::string& o) {
    return absl::StrCat("b = ", t, "[] parameter(1)\n  ROOT r = ", t, "[] ",
                        o, "(a, b)");
  };
  EXPECT_FALSE(legal("s64", op("s64", "add")).ok());
  EXPECT_FALSE(legal("u64", op("u64", "maximum")).ok());
  EXPECT_TRUE(legal("s64", "ROOT b = s64[] parameter(1)").ok());  // overwrite
  EXPECT_TRUE(legal("s32", op("s32", "add")).ok());
  // unique_indices: no collisions, no atomics.
  {
    auto module = ParseAndReturnUnverifiedModule(absl::StrReplaceAll(
        kScatter, {{"$T", "s64"},
                   {"$COMB", op("s64", "add")},
                   {"to_apply=comb", "unique_indices=true, to_apply=comb"}}));
    ASSERT_TRUE(module.ok()) << module.status();
    EXPECT_TRUE(CheckPostGemmRewriter(**module).ok());
  }
  EXPECT_TRUE(legal("f32", op("f32", "add")).ok());
}

TEST_F(MetalHloChecksTest, ComplexArithmeticIsNotChecked) {
  EXPECT_TRUE(Check(R"(
HloModule m
ENTRY e {
  a = f32[4] parameter(0)
  c = c64[4] complex(a, a)
  ROOT m = f32[4] abs(c)
})").ok());
}

}  // namespace
}  // namespace gpu
}  // namespace xla
