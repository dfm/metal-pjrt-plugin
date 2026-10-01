// Copyright 2026 The metal-pjrt-plugin Authors
// SPDX-License-Identifier: Apache-2.0

#include "metal_pjrt/runtime/buffer_cache.h"

#include <cstdint>
#include <list>
#include <map>

#include <gtest/gtest.h>

namespace metal_pjrt {
namespace rt {
namespace {

struct Entry {
  int id;
  uint64_t ticket;
};
using List = std::list<Entry>;
using BySize = std::multimap<uint64_t, List::iterator>;

class BufferCacheTest : public ::testing::Test {
 protected:
  void Free(int id, uint64_t size, uint64_t ticket) {
    entries_.push_back({id, ticket});
    by_size_.emplace(size, std::prev(entries_.end()));
  }
  int Find(uint64_t length, uint64_t limit, bool host_write,
           uint64_t ended_below) {
    auto it = FindCachedBuffer(by_size_, length, limit, host_write,
                               ended_below);
    return it == by_size_.end() ? -1 : it->second->id;
  }
  List entries_;
  BySize by_size_;
};

TEST_F(BufferCacheTest, GpuUseTakesTheSmallestFirstFreed) {
  Free(1, 4096, 10);
  Free(2, 8192, 3);
  Free(3, 4096, 11);
  EXPECT_EQ(Find(4096, 8192, false, 0), 1);
  EXPECT_EQ(Find(4097, 16384, false, 0), 2);
  EXPECT_EQ(Find(4097, 8192, false, 0), -1);  // limit is exclusive
}

// Fake timeline: work tickets below `ended_below` have finished.
TEST_F(BufferCacheTest, HostWriteTakesOnlyBuffersWhoseWorkEnded) {
  Free(1, 4096, 10);  // freed while ticket 10 may still run
  Free(2, 4096, 4);
  Free(3, 8192, 2);
  // Tickets < 5 ended: entry 1 is skipped, entry 2 (same size) taken.
  EXPECT_EQ(Find(4096, 16384, true, 5), 2);
  // Tickets < 3 ended: only the larger entry 3 qualifies.
  EXPECT_EQ(Find(4096, 16384, true, 3), 3);
  // Nothing ended: a fresh buffer.
  EXPECT_EQ(Find(4096, 16384, true, 1), -1);
  // Everything ended.
  EXPECT_EQ(Find(4096, 16384, true, 11), 1);
  // GPU use ignores tickets.
  EXPECT_EQ(Find(4096, 16384, false, 1), 1);
}

}  // namespace
}  // namespace rt
}  // namespace metal_pjrt
