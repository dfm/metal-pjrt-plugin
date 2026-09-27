// Pattern tests for MetalScanRewriter.
#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "absl/strings/str_replace.h"
#include "metal_pjrt_plugin/compiler/passes/scan_rewriter.h"
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
