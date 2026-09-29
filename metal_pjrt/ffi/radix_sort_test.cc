// RunRadixSort (radix_sort.h) against a CPU stable sort, bit for bit, on
// rt::Device alone (no XLA): every key type the handlers map (8/16/32/64-bit
// unsigned, signed and float, f16 and bf16), keys only and with 8/16/32/64-bit
// values, both directions, one and several rows on both paths (rows of at
// most 2048 in one threadgroup, longer ones through scratch), with
// duplicates, +-0, +-inf, NaNs of both signs with different payloads,
// denormals and the integer extremes. The inputs must stay untouched. Needs
// a Metal device.
#include "metal_pjrt/ffi/radix_sort.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/cleanup/cleanup.h"
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

struct KeyType {
  const char* name;
  int bits;
  KeyKind kind;
  int exp_bits = 0;  // floats
};

const KeyType kKeyTypes[] = {
    {"u8", 8, KeyKind::kUnsigned},      {"s8", 8, KeyKind::kSigned},
    {"u16", 16, KeyKind::kUnsigned},    {"s16", 16, KeyKind::kSigned},
    {"f16", 16, KeyKind::kFloat, 5},    {"bf16", 16, KeyKind::kFloat, 8},
    {"u32", 32, KeyKind::kUnsigned},    {"s32", 32, KeyKind::kSigned},
    {"f32", 32, KeyKind::kFloat, 8},    {"u64", 64, KeyKind::kUnsigned},
    {"s64", 64, KeyKind::kSigned},      {"f64", 64, KeyKind::kFloat, 11},
};
const KeyType& Key(const char* name) {
  for (const KeyType& k : kKeyTypes) {
    if (std::string(k.name) == name) return k;
  }
  return kKeyTypes[0];
}
// Value widths in bits; 0 sorts keys only.
const int kValueBits[] = {0, 8, 16, 32, 64};

// CUB's radix order of key bits (see radix_sort.h), as a uint64.
uint64_t OrderBits(uint64_t k, const KeyType& t) {
  const uint64_t mask =
      t.bits == 64 ? ~uint64_t{0} : (uint64_t{1} << t.bits) - 1;
  const uint64_t sign = uint64_t{1} << (t.bits - 1);
  k &= mask;
  if (t.kind == KeyKind::kSigned) return k ^ sign;
  if (t.kind == KeyKind::kFloat) {
    if (k == sign) k = 0;  // -0 as +0
    return (k & sign) ? (~k & mask) : (k ^ sign);
  }
  return k;
}

uint64_t Load(const std::vector<uint8_t>& buf, int64_t i, int bytes) {
  uint64_t v = 0;
  std::memcpy(&v, buf.data() + i * bytes, bytes);
  return v;
}

// Random key bits with many duplicates, 0, all ones and the signed
// extremes; for floats also +-0, +-inf, NaNs of both signs with different
// payloads and denormals.
std::vector<uint8_t> MakeKeys(const KeyType& t, int64_t total,
                              uint32_t seed) {
  const int bytes = t.bits / 8;
  std::mt19937_64 rng(seed);
  const uint64_t sign = uint64_t{1} << (t.bits - 1);
  const uint64_t mask =
      t.bits == 64 ? ~uint64_t{0} : (uint64_t{1} << t.bits) - 1;
  std::vector<uint64_t> pool;
  for (int i = 0; i < 8; ++i) pool.push_back(rng());
  pool.insert(pool.end(), {0, mask, sign, sign - 1});  // INT_MIN, INT_MAX
  if (t.kind == KeyKind::kFloat) {
    const uint64_t exp = ((uint64_t{1} << t.exp_bits) - 1)
                         << (t.bits - 1 - t.exp_bits);
    const uint64_t mantissa = (uint64_t{1} << (t.bits - 1 - t.exp_bits)) - 1;
    pool.insert(pool.end(),
                {exp, exp | sign,                          // +-inf
                 exp | 1, exp | 3, exp | sign | 1,         // NaN payloads
                 exp | sign | 2, 1, sign | 1, mantissa,    // denormals
                 sign | mantissa});
  }
  std::vector<uint8_t> out(total * bytes);
  for (int64_t i = 0; i < total; ++i) {
    uint64_t v = rng() % 3 == 0 ? pool[rng() % pool.size()] : rng();
    std::memcpy(out.data() + i * bytes, &v, bytes);
  }
  return out;
}

class RadixSortTest : public ::testing::Test {
 protected:
  void SetUp() override {
    absl::StatusOr<std::unique_ptr<rt::Device>> dev = rt::Device::Create(0);
    ASSERT_THAT(dev, IsOk());
    dev_ = *std::move(dev);
    absl::StatusOr<std::unique_ptr<rt::Stream>> s = dev_->CreateStream();
    ASSERT_THAT(s, IsOk());
    stream_ = *std::move(s);
  }

  // Sorts and compares with the CPU reference. scratch_delta shrinks the
  // scratch buffer below the plan's (for the refusal test).
  absl::Status Run(const KeyType& kt, int value_bits, bool descending,
                   int64_t batch, int64_t n, int64_t scratch_delta = 0) {
    const bool pairs = value_bits != 0;
    const int kb = kt.bits / 8;
    const int vb = value_bits / 8;
    const int64_t total = batch * n;
    const std::string what =
        absl::StrCat(kt.name, pairs ? absl::StrCat("/", value_bits) : "",
                     descending ? " desc " : " asc ", batch, "x", n);
    absl::StatusOr<RadixSortPlan> plan = PlanRadixSort(
        {kt.bits, kt.kind, value_bits, static_cast<uint64_t>(total), batch});
    if (!plan.ok()) return plan.status();

    std::vector<uint8_t> keys = MakeKeys(kt, total, 1234 + total);
    std::vector<uint8_t> vals(total * vb);
    for (int64_t i = 0; i < total; ++i) {
      uint64_t v = static_cast<uint64_t>(i) * 0x9E3779B97F4A7C15ull + i;
      if (pairs) std::memcpy(vals.data() + i * vb, &v, vb);
    }
    // Freed on return.
    std::vector<void*> allocations;
    auto alloc = [&](uint64_t bytes) -> void* {
      absl::StatusOr<rt::Allocation> a =
          dev_->Allocate(std::max<uint64_t>(bytes, 1));
      if (!a.ok()) return nullptr;
      allocations.push_back(a->ptr);
      return a->ptr;
    };
    absl::Cleanup free_all = [&] {
      for (void* p : allocations) (void)dev_->Deallocate(p);
    };
    void* kin = alloc(total * kb);
    void* kout = alloc(total * kb);
    void* vin = pairs ? alloc(total * vb) : nullptr;
    void* vout = pairs ? alloc(total * vb) : nullptr;
    const uint64_t scratch_bytes = plan->scratch_bytes - scratch_delta;
    void* scratch = alloc(scratch_bytes);
    if (kin == nullptr || kout == nullptr || scratch == nullptr ||
        (pairs && (vin == nullptr || vout == nullptr))) {
      return absl::ResourceExhaustedError(what + ": allocation failed");
    }
    std::memcpy(kin, keys.data(), total * kb);
    if (pairs) std::memcpy(vin, vals.data(), total * vb);
    // Poison the outputs so missing writes show.
    std::memset(kout, 0, total * kb);
    if (pairs) std::memset(vout, 0, total * vb);

    absl::Status run =
        RunRadixSort(dev_.get(), stream_.get(), *plan, kin, kout, vin, vout,
                     scratch, scratch_bytes, descending);
    if (absl::Status s = stream_->Synchronize(); !s.ok()) return s;
    if (!run.ok()) return run;

    std::vector<uint8_t> got_k(total * kb), got_v(total * vb);
    std::memcpy(got_k.data(), kout, total * kb);
    if (pairs) std::memcpy(got_v.data(), vout, total * vb);
    // The inputs must be untouched.
    if (std::memcmp(kin, keys.data(), total * kb) != 0 ||
        (pairs && std::memcmp(vin, vals.data(), total * vb) != 0)) {
      return absl::InternalError(what + ": input written");
    }
    std::vector<int64_t> perm(n);
    for (int64_t r = 0; r < batch; ++r) {
      std::iota(perm.begin(), perm.end(), r * n);
      std::stable_sort(perm.begin(), perm.end(), [&](int64_t a, int64_t b) {
        uint64_t x = OrderBits(Load(keys, a, kb), kt);
        uint64_t y = OrderBits(Load(keys, b, kb), kt);
        return descending ? x > y : x < y;
      });
      for (int64_t i = 0; i < n; ++i) {
        int64_t o = r * n + i;
        if (Load(got_k, o, kb) != Load(keys, perm[i], kb) ||
            (pairs && Load(got_v, o, vb) != Load(vals, perm[i], vb))) {
          return absl::InternalError(absl::StrCat(
              what, ": mismatch at row ", r, " index ", i, " key ",
              Load(got_k, o, kb), " want ", Load(keys, perm[i], kb)));
        }
      }
    }
    return absl::OkStatus();
  }

  std::unique_ptr<rt::Device> dev_;
  std::unique_ptr<rt::Stream> stream_;
};

// Rows around the one-threadgroup limit (2048) and the tile boundaries, one
// and several rows on both paths (cub_sort_test's shapes, plus 0, 20000 and
// 3 rows of each length).
TEST_F(RadixSortTest, Sweep) {
  std::vector<std::pair<int64_t, int64_t>> shapes = {
      {1, 17}, {5, 100}, {2, 2049}, {1, 4095}, {2, 4096}};
  for (int64_t n : {0, 1, 17, 2047, 2048, 2049, 4096, 4097, 20000}) {
    shapes.push_back({1, n});
    shapes.push_back({3, n});
  }
  for (const KeyType& kt : kKeyTypes) {
    for (int vbits : kValueBits) {
      for (bool desc : {false, true}) {
        for (auto [batch, n] : shapes) {
          ASSERT_THAT(Run(kt, vbits, desc, batch, n), IsOk());
        }
      }
    }
  }
}

TEST_F(RadixSortTest, Medium) {
  for (const char* k : {"u32", "f32", "s64", "f16"}) {
    ASSERT_THAT(Run(Key(k), 64, false, 1, 65536), IsOk());
    ASSERT_THAT(Run(Key(k), 0, true, 4, 16411), IsOk());
  }
}

// The shapes SortRewriter produces for jax: jnp.sort(f32) -> (U32 synthetic
// key, F32), argsort -> (U32, U64 packed); plus plain keys.
TEST_F(RadixSortTest, OneMillion) {
  const int64_t n = 1 << 20;
  ASSERT_THAT(Run(Key("u32"), 32, false, 1, n), IsOk());
  ASSERT_THAT(Run(Key("u32"), 64, true, 1, n), IsOk());
  ASSERT_THAT(Run(Key("f32"), 0, false, 1, n), IsOk());
  ASSERT_THAT(Run(Key("s64"), 32, false, 4, n / 4), IsOk());
  ASSERT_THAT(Run(Key("s32"), 32, false, n / 4, 4), IsOk());
}

// A scratch buffer smaller than the plan's is refused before anything is
// encoded (the kernels would otherwise write past it).
TEST_F(RadixSortTest, ShortScratchIsRefused) {
  absl::Status s = Run(Key("u32"), 32, false, 1, 10000, /*scratch_delta=*/1);
  EXPECT_EQ(s.code(), absl::StatusCode::kInvalidArgument) << s;
  EXPECT_NE(s.message().find("scratch"), std::string::npos) << s;
  // The stream is still healthy.
  ASSERT_THAT(Run(Key("u32"), 32, false, 1, 10000), IsOk());
}

TEST_F(RadixSortTest, Plan) {
  // Rows of at most one tile need no scratch.
  absl::StatusOr<RadixSortPlan> small =
      PlanRadixSort({32, KeyKind::kFloat, 32, 3 * 2048, 3});
  ASSERT_THAT(small, IsOk());
  EXPECT_TRUE(small->small);
  EXPECT_EQ(small->scratch_bytes, 0u);
  // [keys | values | counts], each 256-byte aligned.
  absl::StatusOr<RadixSortPlan> large =
      PlanRadixSort({32, KeyKind::kFloat, 16, 2 * 2049, 2});
  ASSERT_THAT(large, IsOk());
  EXPECT_FALSE(large->small);
  EXPECT_EQ(large->tiles, 2u);
  EXPECT_EQ(large->alt_values_offset, 16640u);  // align256(4098 * 4)
  EXPECT_EQ(large->hist_offset, 16640u + 8448);  // + align256(4098 * 2)
  EXPECT_EQ(large->scratch_bytes, 16640u + 8448 + 256);  // 2 * 2 * 16 * 4
  // An empty sort plans nothing, whatever the batch size.
  absl::StatusOr<RadixSortPlan> empty =
      PlanRadixSort({32, KeyKind::kUnsigned, 0, 0, 0});
  ASSERT_THAT(empty, IsOk());
  EXPECT_EQ(empty->scratch_bytes, 0u);
  EXPECT_THAT(RunRadixSort(nullptr, nullptr, *empty, nullptr, nullptr,
                           nullptr, nullptr, nullptr, 0, false),
              IsOk());
  EXPECT_THAT(PlanRadixSort({32, KeyKind::kUnsigned, 0, 10, 3}),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(PlanRadixSort({32, KeyKind::kUnsigned, 0, 10, 0}),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(PlanRadixSort({12, KeyKind::kUnsigned, 0, 10, 1}),
              StatusIs(absl::StatusCode::kInvalidArgument));
  // 32-bit offsets in the kernels.
  EXPECT_THAT(PlanRadixSort({8, KeyKind::kUnsigned, 0, uint64_t{1} << 31, 1}),
              IsOk());
  EXPECT_THAT(
      PlanRadixSort({8, KeyKind::kUnsigned, 0, (uint64_t{1} << 31) + 1, 1}),
      StatusIs(absl::StatusCode::kUnimplemented));
}

}  // namespace
}  // namespace ffi
}  // namespace metal_pjrt
