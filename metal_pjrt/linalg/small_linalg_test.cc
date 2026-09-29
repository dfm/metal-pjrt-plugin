// The small-matrix GPU kernels (small_linalg.h) against double references
// on rt::Device alone (no XLA): n in {1, 2, 3, 4, 16, 31, 32}, batches of 1,
// 7 and 65. Cholesky lower and upper (the other triangle of the input is
// NaN, so it must not be read; the other triangle of the output exactly 0;
// a matrix that is not positive definite all NaN), in place too. Triangular
// solves for all 16 side/triangle/transpose/unit combinations, square and
// rectangular right-hand sides. LU: the pivots match first-max partial
// pivoting wherever the choice is not a near tie, the permutation matches
// the pivots exactly, |L| <= 1 and P A = L U within the rounding bound; exact
// ties pick the first row; singular matrices are not an error; m < n and
// m > n. And the predicates' edges. Needs a Metal device.
#include "metal_pjrt/linalg/small_linalg.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "metal_pjrt/runtime/metal_runtime.h"

namespace metal_pjrt {
namespace linalg {
namespace {

using ::absl_testing::IsOk;

constexpr double kU32 = 1.0 / (1 << 24);
constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
constexpr int64_t kSizes[] = {1, 2, 3, 4, 16, 31, 32};
constexpr int64_t kBatches[] = {1, 7, 65};

class SmallLinalgTest : public ::testing::Test {
 protected:
  void SetUp() override {
    absl::StatusOr<std::unique_ptr<rt::Device>> dev = rt::Device::Create(0);
    ASSERT_THAT(dev, IsOk());
    dev_ = *std::move(dev);
    absl::StatusOr<std::unique_ptr<rt::Stream>> s = dev_->CreateStream();
    ASSERT_THAT(s, IsOk());
    stream_ = *std::move(s);
  }
  void TearDown() override {
    for (void* p : allocations_) EXPECT_THAT(dev_->Deallocate(p), IsOk());
  }

  // A device buffer holding `host` (shared memory: the pointer is host
  // visible). Freed at the end of the test.
  template <typename T>
  T* Upload(const std::vector<T>& host) {
    absl::StatusOr<rt::Allocation> a =
        dev_->Allocate(std::max<uint64_t>(host.size() * sizeof(T), 1));
    EXPECT_THAT(a, IsOk());
    if (!a.ok()) return nullptr;
    allocations_.push_back(a->ptr);
    std::memcpy(a->ptr, host.data(), host.size() * sizeof(T));
    return static_cast<T*>(a->ptr);
  }
  template <typename T>
  std::vector<T> Download(const T* p, size_t count) {
    std::vector<T> host(count);
    std::memcpy(host.data(), p, count * sizeof(T));
    return host;
  }
  void Sync() { ASSERT_THAT(stream_->Synchronize(), IsOk()); }

  std::mt19937 rng_{42};
  std::uniform_real_distribution<float> uniform_{-1.0f, 1.0f};
  std::unique_ptr<rt::Device> dev_;
  std::unique_ptr<rt::Stream> stream_;
  std::vector<void*> allocations_;
};

//===--------------------------------------------------------------------===//
// Cholesky (row-major).
//===--------------------------------------------------------------------===//

// Row-major SPD n x n (M M^T / n + I); only the triangle the call reads is
// kept, the other is NaN.
std::vector<float> Spd(std::mt19937& rng, int64_t batch, int64_t n,
                       bool lower) {
  std::uniform_real_distribution<float> u(-1.0f, 1.0f);
  std::vector<float> a(batch * n * n);
  std::vector<double> m(n * n);
  for (int64_t b = 0; b < batch; ++b) {
    for (double& x : m) x = u(rng);
    for (int64_t i = 0; i < n; ++i) {
      for (int64_t j = 0; j < n; ++j) {
        double s = i == j ? 1.0 : 0.0;
        for (int64_t t = 0; t < n; ++t) s += m[i * n + t] * m[j * n + t] / n;
        const bool read = lower ? i >= j : i <= j;
        a[b * n * n + i * n + j] = read ? static_cast<float>(s) : kNaN;
      }
    }
  }
  return a;
}

// Double Cholesky of the referenced triangle: row-major L (lower) or L^T.
// Empty if not positive definite.
std::vector<double> CholeskyRef(const float* a, int64_t n, bool lower) {
  auto at = [&](int64_t i, int64_t j) -> double {  // i >= j
    return lower ? a[i * n + j] : a[j * n + i];
  };
  std::vector<double> l(n * n, 0.0);
  for (int64_t j = 0; j < n; ++j) {
    double s = at(j, j);
    for (int64_t t = 0; t < j; ++t) s -= l[j * n + t] * l[j * n + t];
    if (!(s > 0)) return {};
    l[j * n + j] = std::sqrt(s);
    for (int64_t i = j + 1; i < n; ++i) {
      double v = at(i, j);
      for (int64_t t = 0; t < j; ++t) v -= l[i * n + t] * l[j * n + t];
      l[i * n + j] = v / l[j * n + j];
    }
  }
  if (lower) return l;
  std::vector<double> u(n * n);
  for (int64_t i = 0; i < n; ++i) {
    for (int64_t j = 0; j < n; ++j) u[i * n + j] = l[j * n + i];
  }
  return u;
}

TEST_F(SmallLinalgTest, Cholesky) {
  for (int64_t n : kSizes) {
    for (int64_t batch : kBatches) {
      for (bool lower : {true, false}) {
        for (bool in_place : {false, true}) {
          SCOPED_TRACE(absl::StrCat("n ", n, " batch ", batch, " lower ",
                                    lower, " in place ", in_place));
          std::vector<float> a = Spd(rng_, batch, n, lower);
          // Batch element 0 of the larger batches is not positive definite.
          const bool bad0 = batch > 1;
          if (bad0) a[0] = -1.0f;
          float* da = Upload(a);
          float* dout = in_place ? da : Upload(std::vector<float>(a.size()));
          ASSERT_THAT(RunSmallCholesky(dev_.get(), stream_.get(), da, dout,
                                       batch, n, lower),
                      IsOk());
          Sync();
          std::vector<float> got = Download(dout, a.size());
          for (int64_t b = 0; b < batch; ++b) {
            const float* g = got.data() + b * n * n;
            std::vector<double> ref = CholeskyRef(a.data() + b * n * n, n,
                                                  lower);
            if (b == 0 && bad0) {
              ASSERT_TRUE(ref.empty());
              for (int64_t i = 0; i < n * n; ++i) {
                ASSERT_TRUE(std::isnan(g[i])) << "element " << i;
              }
              continue;
            }
            ASSERT_FALSE(ref.empty());
            double scale = 1;
            for (double x : ref) scale = std::max(scale, std::fabs(x));
            const double tol = 16 * n * kU32 * scale;
            for (int64_t i = 0; i < n; ++i) {
              for (int64_t j = 0; j < n; ++j) {
                const int64_t e = i * n + j;
                if (lower ? j > i : j < i) {
                  ASSERT_EQ(g[e], 0.0f) << "b " << b << " (" << i << ", "
                                        << j << ")";
                } else {
                  ASSERT_NEAR(g[e], ref[e], tol)
                      << "b " << b << " (" << i << ", " << j << ")";
                }
              }
            }
          }
        }
      }
    }
  }
}

//===--------------------------------------------------------------------===//
// Triangular solve (row-major).
//===--------------------------------------------------------------------===//

// Solves M y = r in place for a triangular k x k M (row-major, double).
void TriSolve(const std::vector<double>& m, int64_t k, bool m_lower,
              std::vector<double>& r) {
  if (m_lower) {
    for (int64_t i = 0; i < k; ++i) {
      for (int64_t j = 0; j < i; ++j) r[i] -= m[i * k + j] * r[j];
      r[i] /= m[i * k + i];
    }
  } else {
    for (int64_t i = k - 1; i >= 0; --i) {
      for (int64_t j = i + 1; j < k; ++j) r[i] -= m[i * k + j] * r[j];
      r[i] /= m[i * k + i];
    }
  }
}

TEST_F(SmallLinalgTest, TriangularSolve) {
  for (int64_t kk : kSizes) {
    for (int64_t batch : kBatches) {
      for (int flags = 0; flags < 16; ++flags) {
        const bool left = flags & 1, lower = flags & 2, trans = flags & 4,
                   unit = flags & 8;
        for (int64_t other : {kk, int64_t{5}}) {
          const int64_t m = left ? kk : other, n = left ? other : kk;
          const bool in_place = flags == 5 && other == 5;
          SCOPED_TRACE(absl::StrCat("k ", kk, " batch ", batch, " left ",
                                    left, " lower ", lower, " trans ", trans,
                                    " unit ", unit, " B ", m, "x", n));
          // A: the unread triangle NaN, the diagonal NaN when unit (not
          // read either), else in [2, 3]; off-diagonal in (-1, 1) / k.
          std::vector<float> a(batch * kk * kk), b(batch * m * n);
          for (int64_t e = 0; e < batch; ++e) {
            for (int64_t i = 0; i < kk; ++i) {
              for (int64_t j = 0; j < kk; ++j) {
                float v;
                if (i == j) {
                  v = unit ? kNaN : 2.5f + 0.5f * uniform_(rng_);
                } else if (lower ? i > j : i < j) {
                  v = uniform_(rng_) / kk;
                } else {
                  v = kNaN;
                }
                a[e * kk * kk + i * kk + j] = v;
              }
            }
          }
          for (float& v : b) v = uniform_(rng_);
          float* da = Upload(a);
          float* db = Upload(b);
          float* dx = in_place ? db : Upload(std::vector<float>(b.size()));
          ASSERT_THAT(RunSmallTrsm(dev_.get(), stream_.get(), da, db, dx,
                                   batch, m, n, left, lower, trans, unit),
                      IsOk());
          Sync();
          std::vector<float> got = Download(dx, b.size());
          for (int64_t e = 0; e < batch; ++e) {
            // E = op(A) (left: op(A) X = B, columns of X) or op(A)^T
            // (right: op(A)^T X^T = B^T, rows of X).
            std::vector<double> t(kk * kk, 0.0);
            for (int64_t i = 0; i < kk; ++i) {
              for (int64_t j = 0; j < kk; ++j) {
                if (lower ? i < j : i > j) continue;
                double v = i == j && unit ? 1.0 : a[e * kk * kk + i * kk + j];
                const bool transpose_e = trans != !left;
                t[transpose_e ? j * kk + i : i * kk + j] = v;
              }
            }
            const bool e_lower = lower != trans != !left;
            const int64_t lines = left ? n : m;
            std::vector<double> ref(m * n);
            for (int64_t line = 0; line < lines; ++line) {
              std::vector<double> r(kk);
              for (int64_t i = 0; i < kk; ++i) {
                r[i] = left ? b[e * m * n + i * n + line]
                            : b[e * m * n + line * n + i];
              }
              TriSolve(t, kk, e_lower, r);
              for (int64_t i = 0; i < kk; ++i) {
                (left ? ref[i * n + line] : ref[line * n + i]) = r[i];
              }
            }
            double scale = 1;
            for (double x : ref) scale = std::max(scale, std::fabs(x));
            const double tol = 16 * kk * kU32 * scale;
            for (int64_t i = 0; i < m * n; ++i) {
              ASSERT_NEAR(got[e * m * n + i], ref[i], tol)
                  << "b " << e << " element " << i;
            }
          }
        }
      }
    }
  }
}

//===--------------------------------------------------------------------===//
// LU (column-major).
//===--------------------------------------------------------------------===//

// Checks one m x n column-major factorization of `a`: pivots against a
// double first-max partial pivoting up to its first near tie (exactly, and
// all of them when `exact`), the permutation against the pivots, |L| <= 1
// and P A = L U within n u |L| |U| per element (twice, for margin).
void CheckLu(const float* a, const float* lu, const int32_t* piv,
             const int32_t* perm, int64_t m, int64_t n, bool exact) {
  const int64_t k = std::min(m, n);
  // Double reference with the kernel's algorithm.
  std::vector<double> r(a, a + m * n);
  auto R = [&](int64_t i, int64_t j) -> double& { return r[j * m + i]; };
  bool tie = false;
  for (int64_t j = 0; j < k; ++j) {
    int64_t p = j;
    double best = std::fabs(R(j, j)), second = -1;
    for (int64_t i = j + 1; i < m; ++i) {
      const double v = std::fabs(R(i, j));
      if (v > best) {
        second = best;
        best = v;
        p = i;
      } else {
        second = std::max(second, v);
      }
    }
    tie = tie || (!exact && best - second <= 1e-3 * best);
    if (!tie) ASSERT_EQ(piv[j], p) << "pivot " << j;
    for (int64_t c = 0; c < n; ++c) std::swap(R(j, c), R(p, c));
    const double d = R(j, j);
    if (d != 0) {
      for (int64_t i = j + 1; i < m; ++i) R(i, j) /= d;
    }
    for (int64_t i = j + 1; i < m; ++i) {
      for (int64_t c = j + 1; c < n; ++c) R(i, c) -= R(i, j) * R(j, c);
    }
  }
  if (exact) {
    for (int64_t e = 0; e < m * n; ++e) ASSERT_EQ(lu[e], r[e]) << e;
  }
  // The permutation the pivots produce.
  std::vector<int32_t> want(m);
  for (int64_t i = 0; i < m; ++i) want[i] = static_cast<int32_t>(i);
  for (int64_t i = 0; i < k; ++i) {
    ASSERT_GE(piv[i], i);
    ASSERT_LT(piv[i], m);
    std::swap(want[i], want[piv[i]]);
  }
  for (int64_t i = 0; i < m; ++i) ASSERT_EQ(perm[i], want[i]) << i;
  // P A = L U: row i of L U is row perm[i] of A.
  auto L = [&](int64_t i, int64_t t) -> double {
    return t == i ? 1.0 : t < i ? lu[t * m + i] : 0.0;
  };
  auto U = [&](int64_t t, int64_t j) -> double {
    return t <= j ? lu[j * m + t] : 0.0;
  };
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t t = 0; t < std::min(i, k); ++t) {
      ASSERT_LE(std::fabs(lu[t * m + i]), 1.0f) << "L(" << i << ", " << t
                                                << ")";
    }
    for (int64_t j = 0; j < n; ++j) {
      double s = 0, bound = 0;
      for (int64_t t = 0; t < k; ++t) {
        s += L(i, t) * U(t, j);
        bound += std::fabs(L(i, t) * U(t, j));
      }
      ASSERT_NEAR(s, a[j * m + perm[i]], 2 * k * kU32 * bound + 1e-30)
          << "(" << i << ", " << j << ")";
    }
  }
}

TEST_F(SmallLinalgTest, Getrf) {
  std::vector<std::pair<int64_t, int64_t>> shapes;
  for (int64_t n : kSizes) shapes.push_back({n, n});
  for (auto s : {std::pair<int64_t, int64_t>{3, 5}, {5, 3}, {32, 7}, {7, 32},
                 {1, 32}, {32, 1}}) {
    shapes.push_back(s);
  }
  for (auto [m, n] : shapes) {
    for (int64_t batch : kBatches) {
      for (bool in_place : {false, true}) {
        SCOPED_TRACE(absl::StrCat(m, "x", n, " batch ", batch, " in place ",
                                  in_place));
        const int64_t k = std::min(m, n);
        std::vector<float> a(batch * m * n);
        for (float& v : a) v = uniform_(rng_);
        float* da = Upload(a);
        float* dlu = in_place ? da : Upload(std::vector<float>(a.size()));
        int32_t* dpiv = Upload(std::vector<int32_t>(batch * k, -7));
        int32_t* dperm = Upload(std::vector<int32_t>(batch * m, -7));
        ASSERT_THAT(RunSmallGetrf(dev_.get(), stream_.get(), da, dlu, dpiv,
                                  dperm, batch, m, n),
                    IsOk());
        Sync();
        std::vector<float> lu = Download(dlu, a.size());
        std::vector<int32_t> piv = Download(dpiv, batch * k);
        std::vector<int32_t> perm = Download(dperm, batch * m);
        for (int64_t b = 0; b < batch; ++b) {
          SCOPED_TRACE(absl::StrCat("b ", b));
          CheckLu(a.data() + b * m * n, lu.data() + b * m * n,
                  piv.data() + b * k, perm.data() + b * m, m, n,
                  /*exact=*/false);
          if (HasFatalFailure()) return;
        }
      }
    }
  }
}

// Small integer matrices, where float and double agree exactly: exact ties
// pick the first row of largest magnitude; singular and zero matrices keep
// their zero pivots and are not an error.
TEST_F(SmallLinalgTest, GetrfTiesAndSingular) {
  struct Case {
    int64_t m, n;
    std::vector<float> a;  // column-major
    std::vector<int32_t> pivots;
  };
  const Case cases[] = {
      {2, 2, {1, -1, 2, 3}, {0, 1}},             // |1| == |-1|: row 0
      {3, 3, {0, 2, -2, 1, 1, 1, 0, 1, 2}, {1, 2, 2}},  // 2 then the first
      {4, 4, std::vector<float>(16, 1.0f), {0, 1, 2, 3}},  // rank 1
      {3, 3, std::vector<float>(9, 0.0f), {0, 1, 2}},      // zero
      {2, 3, {0, 0, 1, 2, 3, 4}, {0, 1}},        // zero first column
  };
  for (const Case& c : cases) {
    SCOPED_TRACE(absl::StrCat(c.m, "x", c.n, " a[0] ", c.a[0]));
    const int64_t k = std::min(c.m, c.n);
    float* da = Upload(c.a);
    float* dlu = Upload(std::vector<float>(c.a.size()));
    int32_t* dpiv = Upload(std::vector<int32_t>(k, -7));
    int32_t* dperm = Upload(std::vector<int32_t>(c.m, -7));
    ASSERT_THAT(RunSmallGetrf(dev_.get(), stream_.get(), da, dlu, dpiv, dperm,
                              1, c.m, c.n),
                IsOk());
    Sync();
    std::vector<float> lu = Download(dlu, c.a.size());
    std::vector<int32_t> piv = Download(dpiv, k);
    std::vector<int32_t> perm = Download(dperm, c.m);
    EXPECT_EQ(piv, c.pivots);
    CheckLu(c.a.data(), lu.data(), piv.data(), perm.data(), c.m, c.n,
            /*exact=*/true);
  }
}

// An empty batch launches nothing (null device, stream and buffers).
TEST_F(SmallLinalgTest, EmptyBatchLaunchesNothing) {
  EXPECT_THAT(RunSmallCholesky(nullptr, nullptr, nullptr, nullptr, 0, 4, true),
              IsOk());
  EXPECT_THAT(RunSmallTrsm(nullptr, nullptr, nullptr, nullptr, nullptr, 0, 4,
                           4, true, true, false, false),
              IsOk());
  EXPECT_THAT(RunSmallGetrf(nullptr, nullptr, nullptr, nullptr, nullptr,
                            nullptr, 0, 4, 4),
              IsOk());
}

TEST_F(SmallLinalgTest, Predicates) {
  EXPECT_TRUE(UseSmallCholesky(1, 1));
  EXPECT_TRUE(UseSmallCholesky(1, 32));
  EXPECT_FALSE(UseSmallCholesky(1, 33));
  EXPECT_FALSE(UseSmallCholesky(1, 0));
  // batch * n^2 below 2^32 (32-bit offsets in the kernels).
  EXPECT_TRUE(UseSmallCholesky((int64_t{1} << 22) - 1, 32));
  EXPECT_FALSE(UseSmallCholesky(int64_t{1} << 22, 32));

  EXPECT_TRUE(UseSmallTrsm(1, 32, 1000, /*left_side=*/true));
  EXPECT_FALSE(UseSmallTrsm(1, 33, 5, /*left_side=*/true));
  EXPECT_TRUE(UseSmallTrsm(1, 33, 5, /*left_side=*/false));
  EXPECT_FALSE(UseSmallTrsm(1, 5, 33, /*left_side=*/false));
  EXPECT_FALSE(UseSmallTrsm(1, 0, 5, /*left_side=*/true));
  EXPECT_FALSE(UseSmallTrsm(1, 5, 0, /*left_side=*/true));
  // batch * m * n (B) and batch * lines (threads) below 2^32 as well.
  EXPECT_FALSE(UseSmallTrsm(1 << 10, 32, int64_t{1} << 22, true));
  EXPECT_TRUE(UseSmallTrsm(1 << 10, 32, (int64_t{1} << 17) - 1, true));

  EXPECT_TRUE(UseSmallGetrf(1, 32, 32));
  EXPECT_TRUE(UseSmallGetrf(1, 1, 32));
  EXPECT_FALSE(UseSmallGetrf(1, 33, 5));
  EXPECT_FALSE(UseSmallGetrf(1, 5, 33));
  EXPECT_FALSE(UseSmallGetrf(1, 0, 5));
  EXPECT_FALSE(UseSmallGetrf(1, 5, 0));
  EXPECT_TRUE(UseSmallGetrf((int64_t{1} << 22) - 1, 32, 32));
  EXPECT_FALSE(UseSmallGetrf(int64_t{1} << 22, 32, 32));
}

}  // namespace
}  // namespace linalg
}  // namespace metal_pjrt
