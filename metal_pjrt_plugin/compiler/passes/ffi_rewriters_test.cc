// Pattern tests for MetalSoftmaxRewriter and MetalScanRewriter.
#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "absl/strings/str_replace.h"
#include "metal_pjrt_plugin/compiler/passes/scan_rewriter.h"
#include "metal_pjrt_plugin/compiler/passes/softmax_rewriter.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"

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

constexpr char kSoftmax[] = R"(
HloModule m
max_c { a = $T[] parameter(0)  b = $T[] parameter(1)  ROOT m = $T[] maximum(a, b) }
add_c { a = f32[] parameter(0)  b = f32[] parameter(1)  ROOT s = f32[] add(a, b) }
ENTRY e {
  x = $T[64,1024]{1,0} parameter(0)
  ninf = $T[] constant(-inf)
  mx = $T[64]{0} reduce(x, ninf), dimensions={1}, to_apply=max_c
  mb = $T[64,1024]{1,0} broadcast(mx), dimensions={0}
  d = $T[64,1024]{1,0} subtract(x, mb)
  ex = $T[64,1024]{1,0} exponential(d)
  exf = f32[64,1024]{1,0} convert(ex)
  zero = f32[] constant(0)
  s = f32[64]{0} reduce(exf, zero), dimensions={1}, to_apply=add_c
  sc = $T[64]{0} convert(s)
  sb = $T[64,1024]{1,0} broadcast(sc), dimensions={0}
  ROOT y = $T[64,1024]{1,0} divide(ex, sb)
})";

TEST_F(MetalFfiRewritersTest, SoftmaxBf16) {
  MetalSoftmaxRewriter pass;
  std::string cfg = Rewrite(
      pass, absl::StrReplaceAll(kSoftmax, {{"$T", "bf16"}}), "metal$softmax");
  EXPECT_NE(cfg.find("log = false"), std::string::npos) << cfg;
  EXPECT_NE(cfg.find("row_length = 1024"), std::string::npos) << cfg;
}

TEST_F(MetalFfiRewritersTest, LogSoftmaxF32) {
  constexpr char kHlo[] = R"(
HloModule m
max_c { a = f32[] parameter(0)  b = f32[] parameter(1)  ROOT m = f32[] maximum(a, b) }
add_c { a = f32[] parameter(0)  b = f32[] parameter(1)  ROOT s = f32[] add(a, b) }
ENTRY e {
  x = f32[8,3,100]{2,1,0} parameter(0)
  ninf = f32[] constant(-inf)
  mx = f32[8,3]{1,0} reduce(x, ninf), dimensions={2}, to_apply=max_c
  mb = f32[8,3,100]{2,1,0} broadcast(mx), dimensions={0,1}
  d = f32[8,3,100]{2,1,0} subtract(x, mb)
  ex = f32[8,3,100]{2,1,0} exponential(d)
  zero = f32[] constant(0)
  s = f32[8,3]{1,0} reduce(ex, zero), dimensions={2}, to_apply=add_c
  l = f32[8,3]{1,0} log(s)
  lb = f32[8,3,100]{2,1,0} broadcast(l), dimensions={0,1}
  ROOT y = f32[8,3,100]{2,1,0} subtract(d, lb)
})";
  MetalSoftmaxRewriter pass;
  std::string cfg = Rewrite(pass, kHlo, "metal$softmax");
  EXPECT_NE(cfg.find("log = true"), std::string::npos) << cfg;
}

TEST_F(MetalFfiRewritersTest, SoftmaxNotRewrittenWhenIntermediateEscapes) {
  // `ex` is also returned, so the pattern's kernels would still run.
  std::string hlo = absl::StrReplaceAll(
      absl::StrReplaceAll(kSoftmax, {{"$T", "f32"}}),
                {{"ROOT y = f32[64,1024]{1,0} divide(ex, sb)",
                  "y = f32[64,1024]{1,0} divide(ex, sb)\n"
                  "  ROOT t = (f32[64,1024]{1,0}, f32[64,1024]{1,0}) "
                  "tuple(y, ex)"}});
  MetalSoftmaxRewriter pass;
  EXPECT_EQ(Rewrite(pass, hlo, "metal$softmax"), "");
}

TEST_F(MetalFfiRewritersTest, SoftmaxNotRewrittenOverMajorDim) {
  constexpr char kHlo[] = R"(
HloModule m
max_c { a = f32[] parameter(0)  b = f32[] parameter(1)  ROOT m = f32[] maximum(a, b) }
add_c { a = f32[] parameter(0)  b = f32[] parameter(1)  ROOT s = f32[] add(a, b) }
ENTRY e {
  x = f32[64,32]{1,0} parameter(0)
  ninf = f32[] constant(-inf)
  mx = f32[32]{0} reduce(x, ninf), dimensions={0}, to_apply=max_c
  mb = f32[64,32]{1,0} broadcast(mx), dimensions={1}
  d = f32[64,32]{1,0} subtract(x, mb)
  ex = f32[64,32]{1,0} exponential(d)
  zero = f32[] constant(0)
  s = f32[32]{0} reduce(ex, zero), dimensions={0}, to_apply=add_c
  sb = f32[64,32]{1,0} broadcast(s), dimensions={1}
  ROOT y = f32[64,32]{1,0} divide(ex, sb)
})";
  MetalSoftmaxRewriter pass;
  EXPECT_EQ(Rewrite(pass, kHlo, "metal$softmax"), "");
}

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

}  // namespace
}  // namespace gpu
}  // namespace xla
