// Copyright 2026 The jax-graft Authors
// SPDX-License-Identifier: Apache-2.0

// RunScan (scan.h) against a double reference on rt::Device alone (no XLA):
// every op and element type, both directions, rows of 1 to 10000 elements
// around the SIMD group (32), 4-per-thread chunk (4 * 256) and threadgroup
// boundaries. max/min and s32 add/mul (wrapping) are exact; f32/f16/bf16
// add/mul are within the f32 accumulation error (scaled with the row
// length) plus the output rounding. Needs a Metal device.
#include "metal_pjrt/ffi/scan.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "metal_pjrt/runtime/metal_runtime.h"

namespace metal_pjrt {
namespace ffi {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;

constexpr ScanOp kOps[] = {ScanOp::kAdd, ScanOp::kMul, ScanOp::kMax,
                           ScanOp::kMin};
constexpr ScanType kTypes[] = {ScanType::kF32, ScanType::kF16,
                               ScanType::kBF16, ScanType::kS32};

int Bytes(ScanType t) {
  return t == ScanType::kF32 || t == ScanType::kS32 ? 4 : 2;
}

// Host element <-> float conversions (round to nearest even).
float Bf16ToFloat(uint16_t b) {
  uint32_t u = uint32_t{b} << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
uint16_t FloatToBf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  if (std::isnan(f)) return static_cast<uint16_t>((u >> 16) | 0x40);
  u += 0x7fff + ((u >> 16) & 1);
  return static_cast<uint16_t>(u >> 16);
}

double Load(const std::vector<uint8_t>& buf, int64_t i, ScanType t) {
  switch (t) {
    case ScanType::kF32: {
      float f;
      std::memcpy(&f, buf.data() + i * 4, 4);
      return f;
    }
    case ScanType::kF16: {
      _Float16 h;
      std::memcpy(&h, buf.data() + i * 2, 2);
      return static_cast<float>(h);
    }
    case ScanType::kBF16: {
      uint16_t b;
      std::memcpy(&b, buf.data() + i * 2, 2);
      return Bf16ToFloat(b);
    }
    case ScanType::kS32: {
      int32_t v;
      std::memcpy(&v, buf.data() + i * 4, 4);
      return v;
    }
  }
  return 0;
}

void Store(std::vector<uint8_t>& buf, int64_t i, ScanType t, double v) {
  switch (t) {
    case ScanType::kF32: {
      float f = static_cast<float>(v);
      std::memcpy(buf.data() + i * 4, &f, 4);
      break;
    }
    case ScanType::kF16: {
      _Float16 h = static_cast<_Float16>(static_cast<float>(v));
      std::memcpy(buf.data() + i * 2, &h, 2);
      break;
    }
    case ScanType::kBF16: {
      uint16_t b = FloatToBf16(static_cast<float>(v));
      std::memcpy(buf.data() + i * 2, &b, 2);
      break;
    }
    case ScanType::kS32: {
      int32_t x = static_cast<int32_t>(v);
      std::memcpy(buf.data() + i * 4, &x, 4);
      break;
    }
  }
}

class ScanTest : public ::testing::Test {
 protected:
  void SetUp() override {
    absl::StatusOr<std::unique_ptr<rt::Device>> dev = rt::Device::Create(0);
    ASSERT_THAT(dev, IsOk());
    dev_ = *std::move(dev);
    absl::StatusOr<std::unique_ptr<rt::Stream>> s = dev_->CreateStream();
    ASSERT_THAT(s, IsOk());
    stream_ = *std::move(s);
  }

  // Inputs for `op`: add and max/min in +-[0.5, 1] (no element small enough
  // to hide in the tolerance), mul in 1 +- [0.005, 0.01]; s32 add and
  // max/min over the full range (add wraps), s32 mul in [-3, 3]. Float
  // rows 1 (when n >= 3) carry a NaN for max/min.
  std::vector<uint8_t> MakeInput(ScanType t, ScanOp op, uint64_t rows,
                                 uint64_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> mag(0.5, 1.0);
    std::vector<uint8_t> x(rows * n * Bytes(t));
    for (uint64_t i = 0; i < rows * n; ++i) {
      const double sign = rng() % 2 ? 1.0 : -1.0;
      if (t == ScanType::kS32) {
        int32_t v = op == ScanOp::kMul ? static_cast<int32_t>(rng() % 7) - 3
                                       : static_cast<int32_t>(rng());
        std::memcpy(x.data() + i * 4, &v, 4);
      } else if (op == ScanOp::kMul) {
        Store(x, i, t, 1.0 + sign * 0.01 * mag(rng));
      } else {
        Store(x, i, t, sign * mag(rng));
      }
    }
    if (t != ScanType::kS32 && (op == ScanOp::kMax || op == ScanOp::kMin) &&
        rows > 1 && n >= 3) {
      Store(x, n + n / 2, t, std::numeric_limits<double>::quiet_NaN());
    }
    return x;
  }

  // Runs one scan and compares with the reference.
  void Check(ScanType type, ScanOp op, bool reverse, uint64_t rows,
             uint64_t n) {
    const std::string what = absl::StrCat(ScanFunction(type, op),
                                          reverse ? " reverse " : " ", rows,
                                          "x", n);
    SCOPED_TRACE(what);
    const uint64_t bytes = rows * n * Bytes(type);
    std::vector<uint8_t> x = MakeInput(type, op, rows, n, rows * 7919 + n);
    absl::StatusOr<rt::Allocation> in = dev_->Allocate(bytes);
    absl::StatusOr<rt::Allocation> out = dev_->Allocate(bytes);
    ASSERT_THAT(in, IsOk());
    ASSERT_THAT(out, IsOk());
    std::memcpy(in->ptr, x.data(), bytes);
    std::memset(out->ptr, 0xff, bytes);  // missing writes show
    ASSERT_THAT(RunScan(dev_.get(), stream_.get(), in->ptr, out->ptr, rows, n,
                        type, op, reverse),
                IsOk());
    ASSERT_THAT(stream_->Synchronize(), IsOk());
    std::vector<uint8_t> y(bytes);
    std::memcpy(y.data(), out->ptr, bytes);
    ASSERT_THAT(dev_->Deallocate(in->ptr), IsOk());
    ASSERT_THAT(dev_->Deallocate(out->ptr), IsOk());

    // f32 accumulation error. add: each element passes through a few
    // additions per level of the SIMD and threadgroup scans plus one per
    // chunk of the carry chain (`depth`), each off by at most u times a
    // partial sum of |x|. mul: every multiplication feeding an output
    // contributes its relative rounding, so up to k + 1 of them.
    const double u32 = std::ldexp(1.0, -24);
    const double depth = 16.0 + static_cast<double>(n) / 256.0;
    const double out_eps = type == ScanType::kF16    ? std::ldexp(1.0, -11)
                            : type == ScanType::kBF16 ? std::ldexp(1.0, -8)
                                                      : 0.0;
    int failures = 0;
    for (uint64_t r = 0; r < rows && failures < 5; ++r) {
      double acc = 0, abs_acc = 0;
      uint32_t iacc = 0;
      bool nan = false;
      for (uint64_t k = 0; k < n && failures < 5; ++k) {
        const uint64_t j = reverse ? n - 1 - k : k;
        const int64_t i = r * n + j;
        const double v = Load(x, i, type);
        const double got = Load(y, i, type);
        if (type == ScanType::kS32) {
          const uint32_t u = static_cast<uint32_t>(static_cast<int32_t>(v));
          const int32_t sv = static_cast<int32_t>(v);
          const int32_t cur = static_cast<int32_t>(iacc);
          switch (op) {
            case ScanOp::kAdd: iacc = k == 0 ? u : iacc + u; break;
            case ScanOp::kMul: iacc = k == 0 ? u : iacc * u; break;
            case ScanOp::kMax:
              iacc = static_cast<uint32_t>(k == 0 ? sv : std::max(cur, sv));
              break;
            case ScanOp::kMin:
              iacc = static_cast<uint32_t>(k == 0 ? sv : std::min(cur, sv));
              break;
          }
          if (static_cast<int32_t>(iacc) != static_cast<int32_t>(got)) {
            ADD_FAILURE() << "row " << r << " index " << j << ": got " << got
                          << " want " << static_cast<int32_t>(iacc);
            ++failures;
          }
          continue;
        }
        if (op == ScanOp::kMax || op == ScanOp::kMin) {
          nan = nan || std::isnan(v);
          if (k == 0) acc = v;
          acc = op == ScanOp::kMax ? std::max(acc, v) : std::min(acc, v);
          const bool ok = nan ? std::isnan(got) : got == acc;
          if (!ok) {
            ADD_FAILURE() << "row " << r << " index " << j << ": got " << got
                          << " want " << (nan ? NAN : acc);
            ++failures;
          }
          continue;
        }
        double tol;
        if (op == ScanOp::kAdd) {
          acc = k == 0 ? v : acc + v;
          abs_acc += std::fabs(v);
          tol = depth * u32 * abs_acc;
        } else {
          acc = k == 0 ? v : acc * v;
          tol = 1.01 * static_cast<double>(k + 1) * u32 * std::fabs(acc);
        }
        tol += out_eps * (std::fabs(acc) + tol);
        if (!(std::fabs(got - acc) <= tol)) {
          ADD_FAILURE() << "row " << r << " index " << j << ": got " << got
                        << " want " << acc << " tol " << tol;
          ++failures;
        }
      }
    }
  }

  std::unique_ptr<rt::Device> dev_;
  std::unique_ptr<rt::Stream> stream_;
};

TEST_F(ScanTest, Sweep) {
  for (ScanType type : kTypes) {
    for (ScanOp op : kOps) {
      for (bool reverse : {false, true}) {
        for (uint64_t rows : {1, 3, 257}) {
          for (uint64_t n : {1, 3, 4, 5, 31, 32, 33, 127, 128, 129, 1023, 1024,
                             1025, 4096, 10000}) {
            Check(type, op, reverse, rows, n);
            if (HasFatalFailure()) return;
          }
        }
      }
    }
  }
}

// Nothing to do: no kernel lookup, no launch (null device, stream and
// buffers are never touched).
TEST_F(ScanTest, EmptyLaunchesNothing) {
  EXPECT_THAT(RunScan(nullptr, nullptr, nullptr, nullptr, 0, 5,
                      ScanType::kF32, ScanOp::kAdd, false),
              IsOk());
  EXPECT_THAT(RunScan(nullptr, nullptr, nullptr, nullptr, 5, 0,
                      ScanType::kF32, ScanOp::kAdd, false),
              IsOk());
}

// The kernel's uint loop would not terminate; refused before any lookup.
TEST_F(ScanTest, TooLargeIsUnimplemented) {
  EXPECT_THAT(RunScan(nullptr, nullptr, nullptr, nullptr, 1,
                      (uint64_t{1} << 31) + 1, ScanType::kF32, ScanOp::kAdd,
                      false),
              StatusIs(absl::StatusCode::kUnimplemented));
  EXPECT_THAT(RunScan(nullptr, nullptr, nullptr, nullptr, uint64_t{1} << 32,
                      1, ScanType::kF32, ScanOp::kAdd, false),
              StatusIs(absl::StatusCode::kUnimplemented));
}

TEST_F(ScanTest, ParseScanOp) {
  EXPECT_EQ(*ParseScanOp("add"), ScanOp::kAdd);
  EXPECT_EQ(*ParseScanOp("mul"), ScanOp::kMul);
  EXPECT_EQ(*ParseScanOp("max"), ScanOp::kMax);
  EXPECT_EQ(*ParseScanOp("min"), ScanOp::kMin);
  EXPECT_THAT(ParseScanOp("sub"),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace ffi
}  // namespace metal_pjrt
