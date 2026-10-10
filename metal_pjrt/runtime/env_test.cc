// Copyright 2026 The jax-graft Authors
// SPDX-License-Identifier: Apache-2.0

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

TEST(EnvTest, Megabytes) {
  unsetenv(kVar);
  EXPECT_EQ(EnvMegabytes(kVar, 512), uint64_t{512} << 20);
  setenv(kVar, "3", 1);
  EXPECT_EQ(EnvMegabytes(kVar, 512), uint64_t{3} << 20);
  setenv(kVar, "17592186044415", 1);  // UINT64_MAX >> 20: still fits
  EXPECT_EQ(EnvMegabytes(kVar, 512), (UINT64_MAX >> 20) << 20);
  // Would wrap around in bytes (17592186044416 << 20 == 0).
  for (const char* huge : {"17592186044416", "18446744073709551615"}) {
    setenv(kVar, huge, 1);
    EXPECT_EQ(EnvMegabytes(kVar, 512), uint64_t{512} << 20) << huge;
  }
  unsetenv(kVar);
}

}  // namespace
}  // namespace metal_pjrt
