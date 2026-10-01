// Copyright 2026 The metal-pjrt-plugin Authors
// SPDX-License-Identifier: Apache-2.0

// RunPoolMaxBwd (pool.h) against a host reference on rt::Device alone (no
// XLA), with operands 16-byte aligned (the 4-lane kernel) and 4 bytes past
// that (the scalar kernel: vector accesses need aligned addresses). Needs a
// Metal device.
#include "metal_pjrt/ffi/pool.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "metal_pjrt/runtime/metal_runtime.h"

namespace metal_pjrt {
namespace ffi {
namespace {

using ::absl_testing::IsOk;

class PoolTest : public ::testing::Test {
 protected:
  void SetUp() override {
    absl::StatusOr<std::unique_ptr<rt::Device>> dev = rt::Device::Create(0);
    ASSERT_THAT(dev, IsOk());
    dev_ = *std::move(dev);
    absl::StatusOr<std::unique_ptr<rt::Stream>> s = dev_->CreateStream();
    ASSERT_THAT(s, IsOk());
    stream_ = *std::move(s);
  }

  std::unique_ptr<rt::Device> dev_;
  std::unique_ptr<rt::Stream> stream_;
};

// f32 x [2, 6, 6, 8] in 2 x 2 windows over dims 1 and 2 (b = 8: 4 lanes
// when aligned); distinct values, so the selection has no ties.
TEST_F(PoolTest, MisalignedOperandsUseScalarKernel) {
  const std::vector<int64_t> dims = {2, 6, 6, 8}, window = {1, 2, 2, 1};
  const int64_t n = 2 * 6 * 6 * 8, ny = 2 * 3 * 3 * 8;
  std::vector<float> x(n), dy(ny), want(n, 0.0f);
  for (int64_t i = 0; i < n; ++i) x[i] = static_cast<float>((i * 37) % n);
  for (int64_t i = 0; i < ny; ++i) dy[i] = 1.0f + static_cast<float>(i);
  for (int64_t a = 0; a < 2; ++a) {
    for (int64_t r = 0; r < 3; ++r) {
      for (int64_t c = 0; c < 3; ++c) {
        for (int64_t b = 0; b < 8; ++b) {
          int64_t best = -1;
          for (int64_t i = 2 * r; i < 2 * r + 2; ++i) {
            for (int64_t j = 2 * c; j < 2 * c + 2; ++j) {
              const int64_t at = ((a * 6 + i) * 6 + j) * 8 + b;
              if (best < 0 || x[at] > x[best]) best = at;
            }
          }
          want[best] = dy[((a * 3 + r) * 3 + c) * 8 + b];
        }
      }
    }
  }
  for (int64_t skew : {0, 4}) {
    SCOPED_TRACE(skew);
    absl::StatusOr<rt::Allocation> in = dev_->Allocate(n * 4 + 16);
    absl::StatusOr<rt::Allocation> g = dev_->Allocate(ny * 4 + 16);
    absl::StatusOr<rt::Allocation> out = dev_->Allocate(n * 4 + 16);
    ASSERT_THAT(in, IsOk());
    ASSERT_THAT(g, IsOk());
    ASSERT_THAT(out, IsOk());
    char* xp = static_cast<char*>(in->ptr) + skew;
    char* dyp = static_cast<char*>(g->ptr) + skew;
    char* dxp = static_cast<char*>(out->ptr) + skew;
    std::memcpy(xp, x.data(), n * 4);
    std::memcpy(dyp, dy.data(), ny * 4);
    std::memset(dxp, 0xff, n * 4);  // missing writes show
    ASSERT_THAT(RunPoolMaxBwd(dev_.get(), stream_.get(), xp, dyp, dxp, dims,
                              window, 0.0f, PoolType::kF32),
                IsOk());
    ASSERT_THAT(stream_->Synchronize(), IsOk());
    std::vector<float> got(n);
    std::memcpy(got.data(), dxp, n * 4);
    for (int64_t i = 0; i < n; ++i) ASSERT_EQ(got[i], want[i]) << i;
    ASSERT_THAT(dev_->Deallocate(in->ptr), IsOk());
    ASSERT_THAT(dev_->Deallocate(g->ptr), IsOk());
    ASSERT_THAT(dev_->Deallocate(out->ptr), IsOk());
  }
}

}  // namespace
}  // namespace ffi
}  // namespace metal_pjrt
