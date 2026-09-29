// RunFft (fft.h) against a double reference on rt::Device alone (no XLA):
// every plan path (Stockham, Rader, fused Bluestein, four-step, multi-upload
// Bluestein, length 1) in all four transform types (fft, ifft, rfft, irfft
// on an arbitrary, non-Hermitian half-spectrum), with odd and even row
// counts (the real transforms pack two rows), partial threadgroups, forced
// row chunks, MLX's hand-tuned sizes and the largest lengths. The error is
// the largest deviation from the reference relative to the reference's RMS,
// bounded by c * 2^-24 * log2(n) with c per path (measured, then about
// doubled). The output buffer carries a guard that must stay untouched.
// Needs a Metal device.
#include "metal_pjrt/fft/fft.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "metal_pjrt/fft/fft_plan.h"
#include "metal_pjrt/runtime/metal_runtime.h"

namespace metal_pjrt {
namespace fft {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using cdouble = std::complex<double>;

constexpr FftType kTypes[] = {FftType::kFft, FftType::kIfft, FftType::kRfft,
                              FftType::kIrfft};
constexpr int kGuardBytes = 256;
constexpr uint8_t kGuard = 0xab;

enum class Path { kLength1, kStockham, kRader, kBluestein, kFourStep,
                  kMultiBluestein };

Path PathOf(const FftPlan& p) {
  if (p.n == 1) return Path::kLength1;
  if (p.four_step) {
    return p.bluestein_n > 0 ? Path::kMultiBluestein : Path::kFourStep;
  }
  if (p.bluestein_n > 0) return Path::kBluestein;
  return p.rader_n > 1 ? Path::kRader : Path::kStockham;
}

const char* PathName(Path p) {
  switch (p) {
    case Path::kLength1:
      return "length1";
    case Path::kStockham:
      return "stockham";
    case Path::kRader:
      return "rader";
    case Path::kBluestein:
      return "bluestein";
    case Path::kFourStep:
      return "four_step";
    case Path::kMultiBluestein:
      return "multi_bluestein";
  }
  return "?";
}

// The error bound's c per path: the measured maxima (printed with
// METAL_TEST_REPORT_ERRORS=1; Stockham 4.0, four-step 3.6, Rader 5.4, fused
// Bluestein 4.0, multi-upload Bluestein 2.5, all on irfft) about doubled.
double Coefficient(Path p) {
  switch (p) {
    case Path::kLength1:
    case Path::kStockham:
    case Path::kFourStep:
    case Path::kBluestein:
      return 8;
    case Path::kRader:
      return 12;
    case Path::kMultiBluestein:
      return 6;
  }
  return 0;
}

bool Report() { return std::getenv("METAL_TEST_REPORT_ERRORS") != nullptr; }

// The reference for `n` from the inputs (as complex, full length n: the
// Hermitian extension for irfft), at output indices `bins` (all when
// empty): sum_j z[j] exp(sign 2 pi i j k / n), scaled by 1/n for the
// inverses; irfft takes the real part.
class Reference {
 public:
  Reference(FftType type, int64_t n, std::vector<cdouble> z)
      : type_(type), n_(n), z_(std::move(z)) {
    const int64_t out = OutputLength(type, n);
    if ((n & (n - 1)) == 0 && n > 1) {
      full_ = z_;
      if (IsInverse(type)) {
        for (cdouble& v : full_) v = std::conj(v);
      }
      internal::Pow2FftDouble(full_);
      if (IsInverse(type)) {
        for (cdouble& v : full_) v = std::conj(v) / double(n);
      }
      full_.resize(out);
    } else if (n * out <= (int64_t{1} << 26)) {
      full_.resize(out);
      std::vector<cdouble> table(n);
      for (int64_t t = 0; t < n; ++t) table[t] = Twiddle(t);
      for (int64_t k = 0; k < out; ++k) {
        cdouble sum = 0.0;
        for (int64_t j = 0; j < n; ++j) sum += z_[j] * table[j * k % n];
        full_[k] = Scale(sum);
      }
    }
  }

  bool has_all() const { return !full_.empty(); }

  cdouble At(int64_t k) const {
    if (has_all()) return full_[k];
    cdouble sum = 0.0;
    for (int64_t j = 0; j < n_; ++j) sum += z_[j] * Twiddle(j * k);
    return Scale(sum);
  }

 private:
  cdouble Twiddle(int64_t t) const {
    const cdouble w = internal::Twiddle(t, n_);
    return IsInverse(type_) ? std::conj(w) : w;
  }
  cdouble Scale(cdouble v) const {
    return IsInverse(type_) ? v / double(n_) : v;
  }

  FftType type_;
  int64_t n_;
  std::vector<cdouble> z_;
  std::vector<cdouble> full_;
};

class FftTest : public ::testing::Test {
 protected:
  void SetUp() override {
    absl::StatusOr<std::unique_ptr<rt::Device>> dev = rt::Device::Create(0);
    ASSERT_THAT(dev, IsOk());
    dev_ = *std::move(dev);
    absl::StatusOr<std::unique_ptr<rt::Stream>> s = dev_->CreateStream();
    ASSERT_THAT(s, IsOk());
    stream_ = *std::move(s);
  }

  void* Alloc(uint64_t bytes) {
    absl::StatusOr<rt::Allocation> a =
        dev_->Allocate(std::max<uint64_t>(bytes, 1));
    EXPECT_THAT(a, IsOk());
    if (!a.ok()) return nullptr;
    allocations_.push_back(a->ptr);
    return a->ptr;
  }

  void TearDown() override {
    for (void* p : allocations_) EXPECT_THAT(dev_->Deallocate(p), IsOk());
  }

  // Runs `rows` rows of the `type` transform of length n and checks every
  // row against the reference; returns the worst relative error.
  double Check(int64_t n, FftType type, int64_t rows,
               int64_t max_chunk_elements = kMaxChunkElements) {
    absl::StatusOr<FftPlan> plan_or = PlanFft(n);
    EXPECT_THAT(plan_or, IsOk());
    if (!plan_or.ok()) return 0;
    const FftPlan& plan = *plan_or;
    const Path path = PathOf(plan);
    SCOPED_TRACE(absl::StrCat(PathName(path), " ", FftTypeName(type), " n=",
                              n, " rows=", rows, " chunk=",
                              max_chunk_elements));
    const int64_t in_len = InputLength(type, n);
    const int64_t out_len = OutputLength(type, n);
    const int in_item = type == FftType::kRfft ? 4 : 8;
    const int out_item = type == FftType::kIrfft ? 4 : 8;

    // Inputs uniform in [-1, 1) (both parts).
    std::mt19937 rng(static_cast<uint32_t>(n * 31 + rows));
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> x(rows * in_len * in_item / 4);
    for (float& v : x) v = u(rng);

    const FftConstants constants = MakeFftConstants(plan);
    void* consts = nullptr;
    if (!constants.bytes.empty()) {
      consts = Alloc(constants.bytes.size());
      std::memcpy(consts, constants.bytes.data(), constants.bytes.size());
    }
    const uint64_t ws_bytes =
        FftWorkspaceBytes(plan, rows, max_chunk_elements);
    void* ws = ws_bytes > 0 ? Alloc(ws_bytes) : nullptr;
    if (ws != nullptr) std::memset(ws, 0x7f, ws_bytes);  // stale values show
    void* in = Alloc(x.size() * 4);
    const uint64_t out_bytes = rows * out_len * out_item;
    auto* out = static_cast<uint8_t*>(Alloc(out_bytes + kGuardBytes));
    if (in == nullptr || out == nullptr) return 0;
    std::memcpy(in, x.data(), x.size() * 4);
    std::memset(out, 0xff, out_bytes);  // NaNs: missing writes show
    std::memset(out + out_bytes, kGuard, kGuardBytes);
    EXPECT_THAT(RunFft(dev_.get(), stream_.get(), plan, type, rows, constants,
                       consts, in, out, ws, ws_bytes, max_chunk_elements),
                IsOk());
    EXPECT_THAT(stream_->Synchronize(), IsOk());
    for (int i = 0; i < kGuardBytes; ++i) {
      if (out[out_bytes + i] != kGuard) {
        ADD_FAILURE() << "write past the output at byte " << i;
        break;
      }
    }

    // Rows to check: all, up to 8 (first, last and some between).
    std::vector<int64_t> check_rows;
    for (int64_t r = 0; r < rows; r += std::max<int64_t>(rows / 7, 1)) {
      check_rows.push_back(r);
    }
    if (check_rows.back() != rows - 1) check_rows.push_back(rows - 1);

    double worst = 0;
    for (int64_t r : check_rows) {
      std::vector<cdouble> z(n);
      const float* xr = x.data() + r * in_len * in_item / 4;
      if (type == FftType::kRfft) {
        for (int64_t j = 0; j < n; ++j) z[j] = xr[j];
      } else if (type == FftType::kIrfft) {
        // numpy's irfft: bins 0 and n/2 (n even) real, the rest mirrored.
        for (int64_t k = 0; k < in_len; ++k) {
          z[k] = {xr[2 * k], xr[2 * k + 1]};
          if (k > 0) z[n - k] = std::conj(z[k]);
        }
        z[0] = z[0].real();
        if (n % 2 == 0) z[n / 2] = z[n / 2].real();
      } else {
        for (int64_t j = 0; j < n; ++j) z[j] = {xr[2 * j], xr[2 * j + 1]};
      }
      const Reference ref(type, n, std::move(z));
      std::vector<int64_t> bins;
      if (ref.has_all()) {
        for (int64_t k = 0; k < out_len; ++k) bins.push_back(k);
      } else {
        std::mt19937 brng(static_cast<uint32_t>(r));
        std::uniform_int_distribution<int64_t> ub(0, out_len - 1);
        bins = {0, 1, out_len / 2, out_len - 1};
        for (int i = 0; i < 60; ++i) bins.push_back(ub(brng));
      }
      double err = 0, sumsq = 0;
      for (int64_t k : bins) {
        cdouble want = ref.At(k);
        cdouble got;
        const uint8_t* o = out + (r * out_len + k) * out_item;
        if (type == FftType::kIrfft) {
          float f;
          std::memcpy(&f, o, 4);
          got = f;
          want = want.real();
        } else {
          float f[2];
          std::memcpy(f, o, 8);
          got = {f[0], f[1]};
        }
        const double e = std::abs(got - want);
        err = std::isnan(e) ? INFINITY : std::max(err, e);
        sumsq += std::norm(want);
      }
      const double rms = std::sqrt(sumsq / bins.size());
      worst = std::max(worst, err / rms);
    }
    const double log2n = std::max(1.0, std::log2(double(n)));
    const double bound = Coefficient(path) * std::ldexp(1.0, -24) * log2n;
    EXPECT_LE(worst, bound);
    if (Report()) {
      std::cout << "fft_test " << PathName(path) << " " << FftTypeName(type)
                << " n=" << n << " rows=" << rows
                << " c=" << worst / (std::ldexp(1.0, -24) * log2n) << "\n";
    }
    return worst;
  }

  void CheckAllTypes(int64_t n, Path expect, std::vector<int64_t> rows) {
    absl::StatusOr<FftPlan> plan = PlanFft(n);
    ASSERT_THAT(plan, IsOk());
    EXPECT_EQ(PathName(PathOf(*plan)), std::string(PathName(expect))) << n;
    for (FftType type : kTypes) {
      for (int64_t r : rows) Check(n, type, r);
    }
  }

  std::unique_ptr<rt::Device> dev_;
  std::unique_ptr<rt::Stream> stream_;
  std::vector<void*> allocations_;
};

TEST_F(FftTest, Length1) { CheckAllTypes(1, Path::kLength1, {1, 3}); }

// n = 2..16 (radices, Rader 11/13 codelets, 7 x 2...), with batching of
// many transforms per threadgroup and a partial last threadgroup.
TEST_F(FftTest, SmallLengths) {
  for (int64_t n = 2; n <= 16; ++n) {
    CheckAllTypes(n, Path::kStockham, {1, 3, 257});
  }
}

TEST_F(FftTest, Stockham) {
  for (int64_t n = 32; n <= 4096; n *= 2) {
    CheckAllTypes(n, Path::kStockham, {1, 3});
  }
  // 13-smooth, including MLX's hand-tuned 3159, 3645, 3969 and odd sizes.
  for (int64_t n : {60, 1000, 3159, 3645, 3969, 1001, 143, 2187, 4095}) {
    CheckAllTypes(n, Path::kStockham, {1, 2});
  }
  Check(64, FftType::kFft, 1000);
  Check(64, FftType::kRfft, 1001);
}

TEST_F(FftTest, Rader) {
  // 1982 = 2 * 991 is MLX's hand-tuned size.
  for (int64_t n : {17, 34, 97, 257, 991, 1982, 2017, 2029}) {
    CheckAllTypes(n, Path::kRader, {1, 3});
  }
}

TEST_F(FftTest, FusedBluestein) {
  for (int64_t n : {47, 289, 323, 1021, 2039}) {
    CheckAllTypes(n, Path::kBluestein, {1, 3});
  }
}

TEST_F(FftTest, FourStep) {
  for (int64_t n : {8192, 65536, 131072, 1 << 20}) {
    CheckAllTypes(n, Path::kFourStep, {1, 3});
  }
  // The largest: n1 = 4096 and n2 = 2048 / 4096.
  Check(1 << 23, FftType::kFft, 1);
  Check(1 << 24, FftType::kFft, 1);
  Check(1 << 24, FftType::kIrfft, 1);
}

TEST_F(FftTest, MultiUploadBluestein) {
  // 2053: a prime above the Rader limit; 4099, 6000: above 4096.
  for (int64_t n : {2053, 4099, 6000, 10007}) {
    CheckAllTypes(n, Path::kMultiBluestein, {1, 3});
  }
  // bluestein_n = 2^24.
  Check((1 << 22) + 1, FftType::kFft, 1);
}

// Several chunks of rows, the last one partial.
TEST_F(FftTest, Chunks) {
  for (FftType type : kTypes) {
    Check(64, type, 257, 64 * 10);
    Check(1000, type, 7, 1000 * 2);
    Check(8192, type, 3, 8192 * 2);
    Check(2053, type, 3, 8192);
  }
}

TEST_F(FftTest, EmptyAndShortWorkspace) {
  const FftPlan plan = *PlanFft(8192);
  const FftConstants constants = MakeFftConstants(plan);
  void* buf = Alloc(8192 * 8 * 2);
  // No rows: nothing to do, no workspace needed.
  EXPECT_THAT(RunFft(dev_.get(), stream_.get(), plan, FftType::kFft, 0,
                     constants, nullptr, buf, buf, nullptr, 0),
              IsOk());
  const uint64_t need = FftWorkspaceBytes(plan, 2);
  EXPECT_THAT(RunFft(dev_.get(), stream_.get(), plan, FftType::kFft, 2,
                     constants, nullptr, buf, buf, buf, need - 1),
              StatusIs(absl::StatusCode::kInvalidArgument));
  // Rader constants must be on the device.
  const FftPlan rader = *PlanFft(17);
  EXPECT_THAT(RunFft(dev_.get(), stream_.get(), rader, FftType::kFft, 1,
                     MakeFftConstants(rader), nullptr, buf, buf, nullptr, 0),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(stream_->Synchronize(), IsOk());
}

// The launches are charged 5 L log2 L flops per transform.
TEST(FftFlopsTest, Flops) {
  EXPECT_EQ(FftFlops(1024, 3), 5ull * 1024 * 10 * 3);
  EXPECT_EQ(FftFlops(1000, 1), 5ull * 1000 * 10);
  EXPECT_EQ(FftFlops(1, 5), 0);
}

}  // namespace
}  // namespace fft
}  // namespace metal_pjrt
