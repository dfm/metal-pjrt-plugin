// Copyright 2026 The metal-pjrt-plugin Authors
// SPDX-License-Identifier: Apache-2.0

// Pattern tests for MetalScanRewriter and MetalConvRewriter.
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "metal_pjrt/compiler/passes/conv_rewriter.h"
#include "metal_pjrt/compiler/passes/scan_rewriter.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/shape_util.h"

namespace xla {
namespace gpu {
namespace {

class MetalFfiRewritersTest : public HloHardwareIndependentTestBase {
 protected:
  // Runs `pass`; returns the root's backend config if it became a custom
  // call to `target`, or "" if nothing changed.
  std::string Rewrite(HloModulePass& pass, const std::string& hlo,
                      const std::string& target) {
    auto module = ParseAndReturnVerifiedModule(hlo);
    EXPECT_TRUE(module.ok()) << module.status();
    if (!module.ok()) return "";
    auto changed = RunHloPass(&pass, module->get());
    EXPECT_TRUE(changed.ok()) << changed.status();
    if (!changed.ok() || !*changed) return "";
    const HloInstruction* root =
        (*module)->entry_computation()->root_instruction();
    EXPECT_EQ(root->opcode(), HloOpcode::kCustomCall);
    EXPECT_EQ(root->custom_call_target(), target);
    EXPECT_EQ(root->operand(0)->opcode(), HloOpcode::kParameter);
    return root->raw_backend_config_string();
  }
};

constexpr char kScan[] = R"(
HloModule m
red { a = $T[] parameter(0)  b = $T[] parameter(1)  ROOT r = $T[] $OP(a, b) }
ENTRY e {
  x = $T[64,300]{1,0} parameter(0)
  init = $T[] constant($INIT)
  ROOT y = $T[64,300]{1,0} reduce-window(x, init), window={size=1x300 pad=0_0x$PAD}, to_apply=red
})";

std::string ScanHlo(const std::string& t, const std::string& op,
                    const std::string& init, const std::string& pad) {
  return absl::StrReplaceAll(
      kScan, {{"$T", t}, {"$OP", op}, {"$INIT", init}, {"$PAD", pad}});
}

TEST_F(MetalFfiRewritersTest, Cumsum) {
  MetalScanRewriter pass;
  std::string cfg = Rewrite(pass, ScanHlo("f32", "add", "0", "299_0"),
                            "metal$scan");
  EXPECT_NE(cfg.find("op = \\\"add\\\""), std::string::npos) << cfg;
  EXPECT_NE(cfg.find("reverse = false"), std::string::npos) << cfg;
}

TEST_F(MetalFfiRewritersTest, ReverseCummaxS32) {
  MetalScanRewriter pass;
  std::string cfg = Rewrite(
      pass, ScanHlo("s32", "maximum", "-2147483648", "0_299"), "metal$scan");
  EXPECT_NE(cfg.find("reverse = true"), std::string::npos) << cfg;
}

TEST_F(MetalFfiRewritersTest, ScanNotRewritten) {
  MetalScanRewriter pass;
  // Wrong identity, a non-scan window, and an unsupported type.
  EXPECT_EQ(Rewrite(pass, ScanHlo("f32", "add", "1", "299_0"), "metal$scan"),
            "");
  EXPECT_EQ(Rewrite(pass, ScanHlo("f32", "add", "0", "150_149"), "metal$scan"),
            "");
  EXPECT_EQ(Rewrite(pass, ScanHlo("u8", "add", "0", "299_0"), "metal$scan"),
            "");
}

// Rows longer than 2^31 would make the kernel's 32-bit chunk loop wrap (an
// endless GPU loop): left to XLA. Parsed only, never allocated.
TEST_F(MetalFfiRewritersTest, ScanRowTooLongNotRewritten) {
  MetalScanRewriter pass;
  auto hlo = [](const std::string& n) {
    return absl::StrReplaceAll(R"(
HloModule m
red { a = f32[] parameter(0)  b = f32[] parameter(1)  ROOT r = f32[] add(a, b) }
ENTRY e {
  x = f32[32,$N]{1,0} parameter(0)
  init = f32[] constant(0)
  ROOT y = f32[32,$N]{1,0} reduce-window(x, init), window={size=1x$N pad=0_0x$P_0}, to_apply=red
})", {{"$N", n}, {"$P", std::to_string(std::stoll(n) - 1)}});
  };
  EXPECT_NE(Rewrite(pass, hlo("2147483648"), "metal$scan"), "");
  EXPECT_EQ(Rewrite(pass, hlo("2147483649"), "metal$scan"), "");
}

// One metal$conv call after MetalConvRewriter.
struct ConvCall {
  std::string config;
  std::string operand0, operand1, result;  // shapes, without layouts
  bool reverse_left = false;               // a kReverse still in the module
};

class MetalConvRewriterTest : public HloHardwareIndependentTestBase {
 protected:
  // The metal$conv calls of `hlo` after the pass (none if unchanged); the
  // module still verifies, and every convolution is gone when one was
  // rewritten.
  std::vector<ConvCall> Rewrite(const std::string& hlo,
                                int64_t min_flops = 0) {
    auto module = ParseAndReturnVerifiedModule(hlo);
    EXPECT_TRUE(module.ok()) << module.status();
    if (!module.ok()) return {};
    MetalConvRewriter pass(min_flops);
    auto changed = RunHloPass(&pass, module->get());
    EXPECT_TRUE(changed.ok()) << changed.status();
    if (!changed.ok() || !*changed) return {};
    EXPECT_TRUE(verifier().Run(module->get()).ok());
    std::vector<ConvCall> calls;
    bool reverse = false;
    for (const HloInstruction* instr :
         (*module)->entry_computation()->instructions()) {
      if (instr->opcode() == HloOpcode::kReverse) reverse = true;
      EXPECT_NE(instr->opcode(), HloOpcode::kConvolution);
      if (instr->opcode() != HloOpcode::kCustomCall) continue;
      EXPECT_EQ(instr->custom_call_target(), "metal$conv");
      auto str = [](const Shape& s) {
        return ShapeUtil::HumanString(s);
      };
      auto config = instr->backend_config<GpuBackendConfig>();
      EXPECT_TRUE(config.ok());
      if (!config.ok()) continue;
      calls.push_back({config->custom_call_backend_config().attributes(),
                       str(instr->operand(0)->shape()),
                       str(instr->operand(1)->shape()),
                       str(instr->shape().tuple_shapes(0))});
    }
    for (ConvCall& c : calls) c.reverse_left = reverse;
    return calls;
  }
};

bool Has(const std::string& config, const std::string& what) {
  return config.find(what) != std::string::npos;
}

// Real HLO: jax.jit(...).lower(...).as_text(dialect="hlo") on CPU (jax
// 0.11.2) for lax.conv_general_dilated, jax.vjp w.r.t. the lhs and w.r.t.
// the rhs; metadata removed.
constexpr char kJaxFwdNhwc[] = R"(
HloModule jit__lambda, entry_computation_layout={(f32[2,8,8,16]{3,2,1,0}, f32[3,3,16,32]{3,2,1,0})->f32[2,4,4,32]{3,2,1,0}}
ENTRY main.1 {
  x.1 = f32[2,8,8,16]{3,2,1,0} parameter(0)
  w.1 = f32[3,3,16,32]{3,2,1,0} parameter(1)
  ROOT conv_general_dilated.1 = f32[2,4,4,32]{3,2,1,0} convolution(x.1, w.1), window={size=3x3 stride=2x2 pad=0_1x0_1}, dim_labels=b01f_01io->b01f
})";

constexpr char kJaxVjpLhsNhwc[] = R"(
HloModule jit__lambda, entry_computation_layout={(f32[3,3,16,32]{3,2,1,0}, f32[2,4,4,32]{3,2,1,0})->f32[2,8,8,16]{3,2,1,0}}
ENTRY main.1 {
  g.1 = f32[2,4,4,32]{3,2,1,0} parameter(1)
  w.1 = f32[3,3,16,32]{3,2,1,0} parameter(0)
  rev.1 = f32[3,3,16,32]{3,2,1,0} reverse(w.1), dimensions={0,1}
  ROOT conv_general_dilated.1 = f32[2,8,8,16]{3,2,1,0} convolution(g.1, rev.1), window={size=3x3 pad=2_1x2_1 lhs_dilate=2x2}, dim_labels=b01f_01oi->b01f
})";

constexpr char kJaxVjpRhsNhwc[] = R"(
HloModule jit__lambda, entry_computation_layout={(f32[2,8,8,16]{3,2,1,0}, f32[2,4,4,32]{3,2,1,0})->f32[3,3,16,32]{3,2,1,0}}
ENTRY main.1 {
  x.1 = f32[2,8,8,16]{3,2,1,0} parameter(0)
  g.1 = f32[2,4,4,32]{3,2,1,0} parameter(1)
  ROOT conv_general_dilated.1 = f32[3,3,16,32]{3,2,1,0} convolution(x.1, g.1), window={size=4x4 pad=0_1x0_1 rhs_dilate=2x2}, dim_labels=f01b_i01o->01bf
})";

// NCHW / OIHW, window strides (1, 2), padding ((1, 0), (2, 1)), rhs
// dilation (2, 1).
constexpr char kJaxFwdNchw[] = R"(
HloModule jit__lambda, entry_computation_layout={(f32[2,5,9,10]{3,2,1,0}, f32[8,5,3,3]{3,2,1,0})->f32[2,8,6,6]{3,2,1,0}}
ENTRY main.1 {
  x.1 = f32[2,5,9,10]{3,2,1,0} parameter(0)
  w.1 = f32[8,5,3,3]{3,2,1,0} parameter(1)
  ROOT conv_general_dilated.1 = f32[2,8,6,6]{3,2,1,0} convolution(x.1, w.1), window={size=3x3 stride=1x2 pad=1_0x2_1 rhs_dilate=2x1}, dim_labels=bf01_oi01->bf01
})";

constexpr char kJaxVjpLhsNchw[] = R"(
HloModule jit__lambda, entry_computation_layout={(f32[8,5,3,3]{3,2,1,0}, f32[2,8,6,6]{3,2,1,0})->f32[2,5,9,10]{3,2,1,0}}
ENTRY main.1 {
  g.1 = f32[2,8,6,6]{3,2,1,0} parameter(1)
  w.1 = f32[8,5,3,3]{3,2,1,0} parameter(0)
  rev.1 = f32[8,5,3,3]{3,2,1,0} reverse(w.1), dimensions={2,3}
  ROOT conv_general_dilated.1 = f32[2,5,9,10]{3,2,1,0} convolution(g.1, rev.1), window={size=3x3 pad=3_4x0_1 lhs_dilate=1x2 rhs_dilate=2x1}, dim_labels=bf01_io01->bf01
})";

constexpr char kJaxVjpRhsNchw[] = R"(
HloModule jit__lambda, entry_computation_layout={(f32[2,5,9,10]{3,2,1,0}, f32[2,8,6,6]{3,2,1,0})->f32[8,5,3,3]{3,2,1,0}}
ENTRY main.1 {
  x.1 = f32[2,5,9,10]{3,2,1,0} parameter(0)
  g.1 = f32[2,8,6,6]{3,2,1,0} parameter(1)
  ROOT conv_general_dilated.1 = f32[8,5,3,3]{3,2,1,0} convolution(x.1, g.1), window={size=6x6 stride=2x1 pad=1_0x2_1 rhs_dilate=1x2}, dim_labels=fb01_io01->fb01
})";

constexpr char kJaxFwd1d[] = R"(
HloModule jit__lambda, entry_computation_layout={(f32[2,20,4]{2,1,0}, f32[5,4,8]{2,1,0})->f32[2,20,8]{2,1,0}}
ENTRY main.1 {
  x.1 = f32[2,20,4]{2,1,0} parameter(0)
  w.1 = f32[5,4,8]{2,1,0} parameter(1)
  ROOT conv_general_dilated.1 = f32[2,20,8]{2,1,0} convolution(x.1, w.1), window={size=5 pad=2_2}, dim_labels=b0f_0io->b0f
})";

constexpr char kJaxVjpRhs1d[] = R"(
HloModule jit__lambda, entry_computation_layout={(f32[2,20,4]{2,1,0}, f32[2,20,8]{2,1,0})->f32[5,4,8]{2,1,0}}
ENTRY main.1 {
  x.1 = f32[2,20,4]{2,1,0} parameter(0)
  g.1 = f32[2,20,8]{2,1,0} parameter(1)
  ROOT conv_general_dilated.1 = f32[5,4,8]{2,1,0} convolution(x.1, g.1), window={size=20 pad=2_2}, dim_labels=f0b_i0o->0bf
})";

constexpr char kJaxGrouped[] = R"(
HloModule jit__lambda, entry_computation_layout={(f32[2,8,8,16]{3,2,1,0}, f32[3,3,8,32]{3,2,1,0})->f32[2,8,8,32]{3,2,1,0}}
ENTRY main.1 {
  x.1 = f32[2,8,8,16]{3,2,1,0} parameter(0)
  w.1 = f32[3,3,8,32]{3,2,1,0} parameter(1)
  ROOT conv_general_dilated.1 = f32[2,8,8,32]{3,2,1,0} convolution(x.1, w.1), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f, feature_group_count=2
})";

TEST_F(MetalConvRewriterTest, JaxForwardNhwc) {
  std::vector<ConvCall> calls = Rewrite(kJaxFwdNhwc);
  ASSERT_EQ(calls.size(), 1);
  const std::string& cfg = calls[0].config;
  EXPECT_TRUE(Has(cfg, "kind = \"fwd\"")) << cfg;
  EXPECT_TRUE(Has(cfg, "stride = array<i64: 2, 2>")) << cfg;
  EXPECT_TRUE(Has(cfg, "pad_lo = array<i64: 0, 0>")) << cfg;
  EXPECT_TRUE(Has(cfg, "flip = false")) << cfg;
  EXPECT_EQ(calls[0].operand0, "f32[2,8,8,16]");  // NHWC as is
  EXPECT_EQ(calls[0].operand1, "f32[32,3,3,16]");  // HWIO -> OHWI
  EXPECT_EQ(calls[0].result, "f32[2,4,4,32]");
}

// The input gradient: the kReverse of the kernel's spatial dims becomes
// flip = true (and is gone); the kernel's "o" is the forward input feature.
TEST_F(MetalConvRewriterTest, JaxInputGradNhwc) {
  std::vector<ConvCall> calls = Rewrite(kJaxVjpLhsNhwc);
  ASSERT_EQ(calls.size(), 1);
  const std::string& cfg = calls[0].config;
  EXPECT_TRUE(Has(cfg, "kind = \"fwd\"")) << cfg;
  EXPECT_TRUE(Has(cfg, "idil = array<i64: 2, 2>")) << cfg;
  EXPECT_TRUE(Has(cfg, "pad_lo = array<i64: 2, 2>")) << cfg;
  EXPECT_TRUE(Has(cfg, "flip = true")) << cfg;
  EXPECT_FALSE(calls[0].reverse_left);
  EXPECT_EQ(calls[0].operand0, "f32[2,4,4,32]");
  EXPECT_EQ(calls[0].operand1, "f32[16,3,3,32]");
  EXPECT_EQ(calls[0].result, "f32[2,8,8,16]");
}

// The weight gradient: the forward convolution's stride comes from the
// rhs dilation; dW is computed as [O, kH, kW, C] and transposed to HWIO.
TEST_F(MetalConvRewriterTest, JaxWeightGradNhwc) {
  std::vector<ConvCall> calls = Rewrite(kJaxVjpRhsNhwc);
  ASSERT_EQ(calls.size(), 1);
  const std::string& cfg = calls[0].config;
  EXPECT_TRUE(Has(cfg, "kind = \"wgrad\"")) << cfg;
  EXPECT_TRUE(Has(cfg, "stride = array<i64: 2, 2>")) << cfg;
  EXPECT_TRUE(Has(cfg, "kdil = array<i64: 1, 1>")) << cfg;
  EXPECT_TRUE(Has(cfg, "pad_lo = array<i64: 0, 0>")) << cfg;
  EXPECT_EQ(calls[0].operand0, "f32[2,8,8,16]");   // x, NHWC
  EXPECT_EQ(calls[0].operand1, "f32[2,4,4,32]");   // dY, NHWC
  EXPECT_EQ(calls[0].result, "f32[32,3,3,16]");    // dW, OHWI
}

TEST_F(MetalConvRewriterTest, JaxNchw) {
  std::vector<ConvCall> fwd = Rewrite(kJaxFwdNchw);
  ASSERT_EQ(fwd.size(), 1);
  EXPECT_TRUE(Has(fwd[0].config, "kind = \"fwd\""));
  EXPECT_TRUE(Has(fwd[0].config, "stride = array<i64: 1, 2>")) << fwd[0].config;
  EXPECT_TRUE(Has(fwd[0].config, "pad_lo = array<i64: 1, 2>")) << fwd[0].config;
  EXPECT_TRUE(Has(fwd[0].config, "kdil = array<i64: 2, 1>")) << fwd[0].config;
  EXPECT_EQ(fwd[0].operand0, "f32[2,9,10,5]");
  EXPECT_EQ(fwd[0].operand1, "f32[8,3,3,5]");
  EXPECT_EQ(fwd[0].result, "f32[2,6,6,8]");

  std::vector<ConvCall> lhs = Rewrite(kJaxVjpLhsNchw);
  ASSERT_EQ(lhs.size(), 1);
  EXPECT_TRUE(Has(lhs[0].config, "flip = true")) << lhs[0].config;
  EXPECT_TRUE(Has(lhs[0].config, "idil = array<i64: 1, 2>")) << lhs[0].config;
  EXPECT_TRUE(Has(lhs[0].config, "kdil = array<i64: 2, 1>")) << lhs[0].config;
  EXPECT_FALSE(lhs[0].reverse_left);
  EXPECT_EQ(lhs[0].operand0, "f32[2,6,6,8]");
  EXPECT_EQ(lhs[0].operand1, "f32[5,3,3,8]");
  EXPECT_EQ(lhs[0].result, "f32[2,9,10,5]");

  std::vector<ConvCall> rhs = Rewrite(kJaxVjpRhsNchw);
  ASSERT_EQ(rhs.size(), 1);
  const std::string& cfg = rhs[0].config;
  EXPECT_TRUE(Has(cfg, "kind = \"wgrad\"")) << cfg;
  EXPECT_TRUE(Has(cfg, "stride = array<i64: 1, 2>")) << cfg;
  EXPECT_TRUE(Has(cfg, "kdil = array<i64: 2, 1>")) << cfg;
  EXPECT_TRUE(Has(cfg, "pad_lo = array<i64: 1, 2>")) << cfg;
  EXPECT_EQ(rhs[0].operand0, "f32[2,9,10,5]");
  EXPECT_EQ(rhs[0].operand1, "f32[2,6,6,8]");
  EXPECT_EQ(rhs[0].result, "f32[8,3,3,5]");
}

// 1-D convolutions as H = 1.
TEST_F(MetalConvRewriterTest, Jax1d) {
  std::vector<ConvCall> fwd = Rewrite(kJaxFwd1d);
  ASSERT_EQ(fwd.size(), 1);
  EXPECT_TRUE(Has(fwd[0].config, "pad_lo = array<i64: 0, 2>")) << fwd[0].config;
  EXPECT_EQ(fwd[0].operand0, "f32[2,1,20,4]");
  EXPECT_EQ(fwd[0].operand1, "f32[8,1,5,4]");
  EXPECT_EQ(fwd[0].result, "f32[2,1,20,8]");
  std::vector<ConvCall> rhs = Rewrite(kJaxVjpRhs1d);
  ASSERT_EQ(rhs.size(), 1);
  EXPECT_TRUE(Has(rhs[0].config, "kind = \"wgrad\"")) << rhs[0].config;
  EXPECT_EQ(rhs[0].operand0, "f32[2,1,20,4]");
  EXPECT_EQ(rhs[0].operand1, "f32[2,1,20,8]");
  EXPECT_EQ(rhs[0].result, "f32[8,1,5,4]");
}

// Hand-written: a reverse with another user stays (no flip); one over other
// dims too.
TEST_F(MetalConvRewriterTest, ReverseNotFolded) {
  constexpr char kShared[] = R"(
HloModule m
ENTRY e {
  g = f32[2,4,4,32] parameter(1)
  w = f32[3,3,16,32] parameter(0)
  rev = f32[3,3,16,32] reverse(w), dimensions={0,1}
  conv = f32[2,8,8,16] convolution(g, rev), window={size=3x3 pad=2_1x2_1 lhs_dilate=2x2}, dim_labels=b01f_01oi->b01f
  ROOT t = (f32[2,8,8,16], f32[3,3,16,32]) tuple(conv, rev)
})";
  std::vector<ConvCall> shared = Rewrite(kShared);
  ASSERT_EQ(shared.size(), 1);
  EXPECT_TRUE(Has(shared[0].config, "flip = false")) << shared[0].config;
  EXPECT_TRUE(shared[0].reverse_left);
  constexpr char kPartial[] = R"(
HloModule m
ENTRY e {
  g = f32[2,4,4,32] parameter(1)
  w = f32[3,3,16,32] parameter(0)
  rev = f32[3,3,16,32] reverse(w), dimensions={0}
  ROOT conv = f32[2,8,8,16] convolution(g, rev), window={size=3x3 pad=2_1x2_1 lhs_dilate=2x2}, dim_labels=b01f_01oi->b01f
})";
  std::vector<ConvCall> partial = Rewrite(kPartial);
  ASSERT_EQ(partial.size(), 1);
  EXPECT_TRUE(Has(partial[0].config, "flip = false")) << partial[0].config;
  EXPECT_TRUE(partial[0].reverse_left);
}

// Left to the loop emitter.
TEST_F(MetalConvRewriterTest, NotRewritten) {
  EXPECT_TRUE(Rewrite(kJaxGrouped).empty());
  auto one = [](const std::string& body) {
    return absl::StrCat("HloModule m\nENTRY e {\n", body, "\n}");
  };
  // Ambiguous: lhs batch/feature swapped, rhs output feature first.
  EXPECT_TRUE(Rewrite(one(R"(
  x = f32[2,8,8,16] parameter(0)
  g = f32[32,4,4,2] parameter(1)
  ROOT c = f32[3,3,16,32] convolution(x, g), window={size=4x4 pad=0_1x0_1 rhs_dilate=2x2}, dim_labels=f01b_o01i->01bf)"))
                  .empty());
  // 3-D.
  EXPECT_TRUE(Rewrite(one(R"(
  x = f32[2,4,4,4,8] parameter(0)
  w = f32[2,2,2,8,8] parameter(1)
  ROOT c = f32[2,3,3,3,8] convolution(x, w), window={size=2x2x2}, dim_labels=b012f_012io->b012f)"))
                  .empty());
  // Negative low padding, window reversal, s32, mixed types.
  EXPECT_TRUE(Rewrite(one(R"(
  x = f32[2,8,8,16] parameter(0)
  w = f32[3,3,16,32] parameter(1)
  ROOT c = f32[2,6,6,32] convolution(x, w), window={size=3x3 pad=-1_1x0_0}, dim_labels=b01f_01io->b01f)"))
                  .empty());
  EXPECT_TRUE(Rewrite(one(R"(
  x = f32[2,8,8,16] parameter(0)
  w = f32[3,3,16,32] parameter(1)
  ROOT c = f32[2,6,6,32] convolution(x, w), window={size=3x3 rhs_reversal=1x0}, dim_labels=b01f_01io->b01f)"))
                  .empty());
  EXPECT_TRUE(Rewrite(one(R"(
  x = s32[2,8,8,16] parameter(0)
  w = s32[3,3,16,32] parameter(1)
  ROOT c = s32[2,6,6,32] convolution(x, w), window={size=3x3}, dim_labels=b01f_01io->b01f)"))
                  .empty());
  EXPECT_TRUE(Rewrite(one(R"(
  x = bf16[2,8,8,16] parameter(0)
  w = bf16[3,3,16,32] parameter(1)
  ROOT c = f32[2,6,6,32] convolution(x, w), window={size=3x3}, dim_labels=b01f_01io->b01f)"))
                  .empty());
}

// XLA's form of the weight gradient of an input-dilated convolution (the
// algebraic simplifier swapped its operands: the kernel was larger than the
// input), as dumped before the rewriter for JAX's vjp of
// conv_general_dilated(x [2,7,6,16], w [3,3,16,8], strides (1, 2),
// padding ((2, 1), (1, 2)), lhs_dilation (2, 3)): a "wgrad" whose rhs is
// reversed first. A reversed forward window toggles flip.
TEST_F(MetalConvRewriterTest, WindowReversal) {
  std::vector<ConvCall> wgrad = Rewrite(R"(
HloModule m
ENTRY e {
  g = f32[2,14,9,8] parameter(0)
  x = f32[2,7,6,16] parameter(1)
  reverse = f32[2,14,9,8] reverse(g), dimensions={1,2}
  ROOT c = f32[3,3,16,8] convolution(reverse, x), window={size=7x6 pad=1_0x0_1 lhs_dilate=1x2 rhs_dilate=2x3 rhs_reversal=1x1}, dim_labels=f01b_i01o->01fb
})");
  ASSERT_EQ(wgrad.size(), 1);
  EXPECT_TRUE(Has(wgrad[0].config, "kind = \"wgrad\"")) << wgrad[0].config;
  EXPECT_TRUE(Has(wgrad[0].config, "flip = false")) << wgrad[0].config;
  EXPECT_EQ(wgrad[0].operand0, "f32[2,14,9,8]");
  EXPECT_EQ(wgrad[0].operand1, "f32[2,7,6,16]");
  EXPECT_EQ(wgrad[0].result, "f32[16,3,3,8]");
  std::vector<ConvCall> fwd = Rewrite(R"(
HloModule m
ENTRY e {
  x = f32[2,8,8,16] parameter(0)
  w = f32[3,3,16,32] parameter(1)
  ROOT c = f32[2,6,6,32] convolution(x, w), window={size=3x3 rhs_reversal=1x1}, dim_labels=b01f_01io->b01f
})");
  ASSERT_EQ(fwd.size(), 1);
  EXPECT_TRUE(Has(fwd[0].config, "flip = true")) << fwd[0].config;
}

// Below min_flops: left alone (kJaxFwdNhwc: 2 * 32 * 32 * 9 * 16 = 294912).
TEST_F(MetalConvRewriterTest, MinFlops) {
  EXPECT_EQ(Rewrite(kJaxFwdNhwc, 294912).size(), 1);
  EXPECT_TRUE(Rewrite(kJaxFwdNhwc, 294913).empty());
}

}  // namespace
}  // namespace gpu
}  // namespace xla
