#include "metal_pjrt/runtime/env.h"

#include <cstdlib>

#include "gtest/gtest.h"

namespace metal_pjrt {
namespace {

constexpr char kVar[] = "METAL_PJRT_ENV_TEST";

TEST(EnvTest, Flag) {
  unsetenv(kVar);
  EXPECT_FALSE(EnvFlag(kVar));
  for (const char* off : {"", "0", "false", "False", "NO", "off", " off "}) {
    setenv(kVar, off, 1);
    EXPECT_FALSE(EnvFlag(kVar)) << off;
  }
  for (const char* on : {"1", "true", "TRUE", "yes", "on", "2"}) {
    setenv(kVar, on, 1);
    EXPECT_TRUE(EnvFlag(kVar)) << on;
  }
  unsetenv(kVar);
}

TEST(EnvTest, Uint) {
  unsetenv(kVar);
  EXPECT_EQ(EnvUint(kVar, 7), 7u);
  setenv(kVar, "", 1);
  EXPECT_EQ(EnvUint(kVar, 7), 7u);
  setenv(kVar, "0", 1);
  EXPECT_EQ(EnvUint(kVar, 7), 0u);
  setenv(kVar, "1024", 1);
  EXPECT_EQ(EnvUint(kVar, 7), 1024u);
  // Typos keep the default instead of reading as 0 (strtoull did).
  for (const char* bad : {"abc", "512MB", "-1", "1.5"}) {
    setenv(kVar, bad, 1);
    EXPECT_EQ(EnvUint(kVar, 7), 7u) << bad;
  }
  unsetenv(kVar);
}

}  // namespace
}  // namespace metal_pjrt
