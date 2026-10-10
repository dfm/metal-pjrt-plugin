// Copyright 2026 The jax-graft Authors
// SPDX-License-Identifier: Apache-2.0

#include "metal_pjrt/runtime/constants_container.h"

#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "gtest/gtest.h"

namespace metal_pjrt {
namespace rt {
namespace {

std::vector<uint8_t> OneBlob() {
  ConstantBlob b;
  b.name = "constant_7";
  b.data = {1, 2, 3, 4, 5};
  return SerializeConstants({b});
}

// Offsets of the first entry's lengths (see the layout in the header).
constexpr size_t kNameLenOffset = 16;
constexpr size_t kDataLenOffset = 24;

TEST(ConstantsContainerTest, RoundTrip) {
  ConstantBlob a, b;
  a.name = "a";
  a.data = {9};
  b.name = "empty";
  std::vector<uint8_t> bytes = SerializeConstants({a, b});
  std::vector<ConstantBlob> out;
  ASSERT_TRUE(ParseConstants(bytes.data(), bytes.size(), &out));
  ASSERT_EQ(out.size(), 2u);
  EXPECT_EQ(out[0].name, "a");
  EXPECT_EQ(out[0].data, std::vector<uint8_t>{9});
  EXPECT_EQ(out[1].name, "empty");
  EXPECT_TRUE(out[1].data.empty());
}

// A damaged container (an entry of JAX's persistent cache) must be refused
// before its lengths are used to allocate.
TEST(ConstantsContainerTest, CorruptLengthsAreRefused) {
  std::vector<ConstantBlob> out;
  // 8 bytes follow the name (5 of data and the padding): 9 is one too many.
  for (uint64_t data_len :
       {uint64_t{9}, uint64_t{1} << 40, uint64_t{1} << 60,
        std::numeric_limits<uint64_t>::max()}) {
    std::vector<uint8_t> bytes = OneBlob();
    std::memcpy(bytes.data() + kDataLenOffset, &data_len, 8);
    EXPECT_FALSE(ParseConstants(bytes.data(), bytes.size(), &out)) << data_len;
  }
  for (uint32_t name_len : {uint32_t{1} << 20, 0xFFFFFFFFu}) {
    std::vector<uint8_t> bytes = OneBlob();
    std::memcpy(bytes.data() + kNameLenOffset, &name_len, 4);
    EXPECT_FALSE(ParseConstants(bytes.data(), bytes.size(), &out)) << name_len;
  }
  std::vector<uint8_t> bytes = OneBlob();
  for (size_t size = 16; size < bytes.size() - 7; ++size) {
    EXPECT_FALSE(ParseConstants(bytes.data(), size, &out)) << size;
  }
}

}  // namespace
}  // namespace rt
}  // namespace metal_pjrt
