// Copyright 2026 The jax-graft Authors
// SPDX-License-Identifier: Apache-2.0

// lapack_host (the Accelerate calls behind the LAPACK handlers) on host
// memory: reconstruction residuals of every factorization, batched, above
// the small-kernel size (32); a Cholesky that is not positive definite
// becomes NaN and the unused triangle exactly 0; getrf's 1-based pivots
// become 0-based with the matching permutation; empty matrices do nothing;
// the workspace rounding. A host test: no Metal, no XLA.
#include "metal_pjrt/linalg/lapack_host.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"

namespace metal_pjrt {
namespace linalg {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;

constexpr double kU32 = 1.0 / (1 << 24);
constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

std::vector<float> Random(std::mt19937& rng, size_t count) {
  std::uniform_real_distribution<float> u(-1.0f, 1.0f);
  std::vector<float> v(count);
  for (float& x : v) x = u(rng);
  return v;
}

double MaxAbs(const float* v, size_t count) {
  double m = 0;
  for (size_t i = 0; i < count; ++i) m = std::max(m, std::fabs(double{v[i]}));
  return m;
}

// Symmetric positive definite n x n (M M^T / n + I), with NaN in the
// triangle that is not `keep_lower` (row-major; the other triangle
// column-major).
std::vector<float> Spd(std::mt19937& rng, int64_t batch, int n,
                       bool keep_lower) {
  std::vector<float> a(batch * n * n);
  for (int64_t b = 0; b < batch; ++b) {
    std::vector<float> m = Random(rng, n * n);
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        double s = i == j ? 1.0 : 0.0;
        for (int t = 0; t < n; ++t) s += double{m[i * n + t]} * m[j * n + t] / n;
        const bool keep = keep_lower ? i >= j : i <= j;
        a[b * n * n + i * n + j] = keep ? static_cast<float>(s) : kNaN;
      }
    }
  }
  return a;
}

TEST(LapackHostTest, Cholesky) {
  std::mt19937 rng(1);
  for (int n : {1, 5, 40, 64}) {
    for (bool lower : {true, false}) {
      SCOPED_TRACE(absl::StrCat("n ", n, " lower ", lower));
      const int64_t batch = 3;
      const std::vector<float> a = Spd(rng, batch, n, lower);
      std::vector<float> x = a;
      x[1 * n * n] = -1.0f;  // batch element 1 is not positive definite
      ASSERT_THAT(HostCholesky(x.data(), batch, n, lower), IsOk());
      for (int64_t b = 0; b < batch; ++b) {
        const float* f = x.data() + b * n * n;
        const float* ab = a.data() + b * n * n;
        if (b == 1) {
          for (int i = 0; i < n * n; ++i) ASSERT_TRUE(std::isnan(f[i]));
          continue;
        }
        // L(i, j), i >= j: row-major lower, or the transpose of upper.
        auto L = [&](int i, int j) -> double {
          return i < j ? 0.0 : lower ? f[i * n + j] : f[j * n + i];
        };
        for (int i = 0; i < n; ++i) {
          for (int j = 0; j < n; ++j) {
            if (lower ? j > i : j < i) ASSERT_EQ(f[i * n + j], 0.0f);
            if (lower ? j > i : j < i) continue;
            double s = 0;
            for (int t = 0; t < n; ++t) s += L(i, t) * L(j, t);
            ASSERT_NEAR(s, ab[i * n + j], 16 * n * kU32 * 8)
                << "(" << i << ", " << j << ")";
          }
        }
      }
    }
  }
  EXPECT_THAT(HostCholesky(nullptr, 3, 0, true), IsOk());
}

TEST(LapackHostTest, TriangularSolve) {
  std::mt19937 rng(2);
  const int k = 40, other = 7, batch = 2;
  for (int flags = 0; flags < 16; ++flags) {
    const bool left = flags & 1, lower = flags & 2, trans = flags & 4,
               unit = flags & 8;
    const int m = left ? k : other, n = left ? other : k;
    SCOPED_TRACE(absl::StrCat("left ", left, " lower ", lower, " trans ",
                              trans, " unit ", unit));
    // Row-major triangular A: diagonal in [2, 3] (NaN, unread, when unit),
    // off-diagonal (-1, 1) / k, NaN in the unused triangle.
    std::vector<float> a = Random(rng, batch * k * k);
    for (int64_t e = 0; e < batch; ++e) {
      for (int i = 0; i < k; ++i) {
        for (int j = 0; j < k; ++j) {
          float& v = a[e * k * k + i * k + j];
          if (i == j) {
            v = unit ? kNaN : 2.5f + 0.5f * v;
          } else if (lower ? i > j : i < j) {
            v /= k;
          } else {
            v = kNaN;
          }
        }
      }
    }
    const std::vector<float> b = Random(rng, batch * m * n);
    std::vector<float> x = b;
    HostTriangularSolve(a.data(), x.data(), batch, m, n, left, lower, trans,
                        unit);
    for (int64_t e = 0; e < batch; ++e) {
      auto T = [&](int i, int j) -> double {  // op(A)
        if (trans) std::swap(i, j);
        if (i == j) return unit ? 1.0 : a[e * k * k + i * k + j];
        return (lower ? i > j : i < j) ? a[e * k * k + i * k + j] : 0.0;
      };
      auto X = [&](int i, int j) -> double { return x[e * m * n + i * n + j]; };
      // op(A) X (left) or X op(A) (right) == B.
      for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
          double s = 0;
          for (int t = 0; t < k; ++t) {
            s += left ? T(i, t) * X(t, j) : X(i, t) * T(t, j);
          }
          ASSERT_NEAR(s, b[e * m * n + i * n + j], 64 * k * kU32)
              << "b " << e << " (" << i << ", " << j << ")";
        }
      }
    }
  }
  HostTriangularSolve(nullptr, nullptr, 2, 0, 5, true, true, false, false);
}

// sgetrf's 1-based pivots -> 0-based, and the permutation.
TEST(LapackHostTest, GetrfPivots) {
  std::vector<float> a = {1, 3, 2, 4};  // column-major [[1, 2], [3, 4]]
  std::vector<int32_t> piv(2, -7), perm(2, -7);
  ASSERT_THAT(HostGetrf(a.data(), piv.data(), perm.data(), 1, 2, 2), IsOk());
  EXPECT_EQ(piv, (std::vector<int32_t>{1, 1}));
  EXPECT_EQ(perm, (std::vector<int32_t>{1, 0}));
  // L = [[1, 0], [1/3, 1]], U = [[3, 4], [0, 2/3]].
  EXPECT_FLOAT_EQ(a[0], 3);
  EXPECT_FLOAT_EQ(a[1], 1.0f / 3);
  EXPECT_FLOAT_EQ(a[2], 4);
  EXPECT_NEAR(a[3], 2.0f / 3, 1e-6);
  // Singular: not an error.
  std::vector<float> ones(9, 1.0f);
  std::vector<int32_t> p3(3), q3(3);
  ASSERT_THAT(HostGetrf(ones.data(), p3.data(), q3.data(), 1, 3, 3), IsOk());
  EXPECT_EQ(p3, (std::vector<int32_t>{0, 1, 2}));
  EXPECT_EQ(q3, (std::vector<int32_t>{0, 1, 2}));
  // Empty: n = 0 leaves the identity permutation, m = 0 nothing.
  std::vector<int32_t> q(3, -7);
  ASSERT_THAT(HostGetrf(nullptr, nullptr, q.data(), 1, 3, 0), IsOk());
  EXPECT_EQ(q, (std::vector<int32_t>{0, 1, 2}));
  EXPECT_THAT(HostGetrf(nullptr, nullptr, nullptr, 2, 0, 3), IsOk());
}

TEST(LapackHostTest, Getrf) {
  std::mt19937 rng(3);
  for (auto [m, n] : {std::pair{50, 50}, {45, 30}, {30, 45}}) {
    SCOPED_TRACE(absl::StrCat(m, "x", n));
    const int64_t batch = 2;
    const int k = std::min(m, n);
    const std::vector<float> a = Random(rng, batch * m * n);
    std::vector<float> lu = a;
    std::vector<int32_t> piv(batch * k), perm(batch * m);
    ASSERT_THAT(HostGetrf(lu.data(), piv.data(), perm.data(), batch, m, n),
                IsOk());
    for (int64_t e = 0; e < batch; ++e) {
      const float* f = lu.data() + e * m * n;
      std::vector<int32_t> want(m);
      for (int i = 0; i < m; ++i) want[i] = i;
      for (int i = 0; i < k; ++i) {
        ASSERT_GE(piv[e * k + i], i);
        ASSERT_LT(piv[e * k + i], m);
        std::swap(want[i], want[piv[e * k + i]]);
      }
      for (int i = 0; i < m; ++i) ASSERT_EQ(perm[e * m + i], want[i]);
      // Row i of L U is row perm[i] of A (column-major).
      for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
          double s = 0;
          for (int t = 0; t <= std::min({i, j, k - 1}); ++t) {
            s += (t == i ? 1.0 : double{f[t * m + i]}) * f[j * m + t];
          }
          ASSERT_NEAR(s, a[e * m * n + j * m + perm[e * m + i]],
                      64 * k * kU32)
              << "b " << e << " (" << i << ", " << j << ")";
        }
      }
    }
  }
}

// JAX's two pivoting cases (linalg_test: a zero or small leading pivot),
// a singular system (NaN), and random batched systems by their residual.
TEST(LapackHostTest, GtsvPivots) {
  struct Case {
    std::vector<float> dl, d, du, b, want;
  };
  const std::vector<Case> cases = {
      {{0, 2, -2, 3}, {1, 4, 1, -1}, {2, -1, 1, 0}, {1, 2, 3, 4},
       {8, -3.5f, 0, -4}},
      {{0, 1, -6, 1}, {1, -1, 2, 1}, {2, 1, -1, 0}, {1, 2, -1, -2},
       {5, -2, -5, 3}},
  };
  for (const Case& c : cases) {
    std::vector<float> x = c.b;
    ASSERT_THAT(HostGtsv(c.dl.data(), c.d.data(), c.du.data(), x.data(), 1, 4,
                         1),
                IsOk());
    for (int i = 0; i < 4; ++i) EXPECT_NEAR(x[i], c.want[i], 1e-5) << i;
  }
  const std::vector<float> zeros(3, 0.0f);
  std::vector<float> x = {1, 1, 1};
  ASSERT_THAT(HostGtsv(zeros.data(), zeros.data(), zeros.data(), x.data(), 1,
                       3, 1),
              IsOk());
  for (float v : x) EXPECT_TRUE(std::isnan(v));
  EXPECT_THAT(HostGtsv(nullptr, nullptr, nullptr, nullptr, 3, 0, 2), IsOk());
}

TEST(LapackHostTest, Gtsv) {
  std::mt19937 rng(7);
  for (auto [batch, n, nrhs] : {std::tuple{1, 1, 1}, std::tuple{3, 2, 2},
                                std::tuple{4, 50, 3}}) {
    std::vector<float> dl = Random(rng, batch * n), d = Random(rng, batch * n),
                       du = Random(rng, batch * n);
    const std::vector<float> b = Random(rng, batch * n * nrhs);
    const std::vector<float> dl0 = dl, d0 = d, du0 = du;
    std::vector<float> x = b;
    ASSERT_THAT(HostGtsv(dl.data(), d.data(), du.data(), x.data(), batch, n,
                         nrhs),
                IsOk());
    EXPECT_EQ(dl, dl0);
    EXPECT_EQ(d, d0);
    EXPECT_EQ(du, du0);
    for (int k = 0; k < batch; ++k) {
      for (int j = 0; j < nrhs; ++j) {
        const float* xs = x.data() + (int64_t{k} * nrhs + j) * n;
        const float* bs = b.data() + (int64_t{k} * nrhs + j) * n;
        double scale = 0;
        for (int i = 0; i < n; ++i) scale = std::max(scale, std::fabs(double(xs[i])));
        for (int i = 0; i < n; ++i) {
          const int64_t o = int64_t{k} * n + i;
          double r = double(d[o]) * xs[i] - bs[i];
          if (i > 0) r += double(dl[o]) * xs[i - 1];
          if (i + 1 < n) r += double(du[o]) * xs[i + 1];
          EXPECT_LT(std::fabs(r), 64 * n * kU32 * (1 + scale) * 4)
              << "batch " << k << " rhs " << j << " row " << i;
        }
      }
    }
  }
}

TEST(LapackHostTest, GeqrfOrgqr) {
  std::mt19937 rng(4);
  for (auto [m, n] : {std::pair{50, 30}, {40, 40}}) {
    SCOPED_TRACE(absl::StrCat(m, "x", n));
    const int64_t batch = 2;
    const std::vector<float> a = Random(rng, batch * m * n);
    std::vector<float> qr = a;
    std::vector<float> taus(batch * n, kNaN);
    ASSERT_THAT(HostGeqrf(qr.data(), taus.data(), batch, m, n), IsOk());
    std::vector<float> q = qr;
    ASSERT_THAT(HostOrgqr(q.data(), taus.data(), batch, m, n, n), IsOk());
    for (int64_t e = 0; e < batch; ++e) {
      const float* R = qr.data() + e * m * n;
      const float* Q = q.data() + e * m * n;
      for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
          double s = 0;  // Q R
          for (int t = 0; t <= j; ++t) s += double{Q[t * m + i]} * R[j * m + t];
          ASSERT_NEAR(s, a[e * m * n + j * m + i], 64 * m * kU32)
              << "QR (" << i << ", " << j << ")";
        }
      }
      for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
          double s = 0;  // Q^T Q
          for (int t = 0; t < m; ++t) s += double{Q[i * m + t]} * Q[j * m + t];
          ASSERT_NEAR(s, i == j ? 1.0 : 0.0, 64 * m * kU32)
              << "QtQ (" << i << ", " << j << ")";
        }
      }
    }
  }
  // k = 0: nothing to do (taus is empty).
  EXPECT_THAT(HostGeqrf(nullptr, nullptr, 2, 0, 5), IsOk());
  EXPECT_THAT(HostGeqrf(nullptr, nullptr, 2, 5, 0), IsOk());
  EXPECT_THAT(HostOrgqr(nullptr, nullptr, 2, 5, 0, 0), IsOk());
}

TEST(LapackHostTest, Syevd) {
  std::mt19937 rng(5);
  const int n = 40;
  const int64_t batch = 2;
  for (bool lower : {true, false}) {
    SCOPED_TRACE(absl::StrCat("lower ", lower));
    // Column-major; the unread triangle NaN.
    std::vector<float> a = Random(rng, batch * n * n);
    for (int64_t e = 0; e < batch; ++e) {
      for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
          float& v = a[e * n * n + j * n + i];
          if (lower ? i < j : i > j) v = kNaN;
        }
      }
    }
    auto A = [&](int64_t e, int i, int j) -> double {
      if (lower ? i < j : i > j) std::swap(i, j);
      return a[e * n * n + j * n + i];
    };
    std::vector<float> v = a;
    std::vector<float> w(batch * n);
    ASSERT_THAT(HostSyevd(v.data(), w.data(), batch, n, lower), IsOk());
    for (int64_t e = 0; e < batch; ++e) {
      const float* V = v.data() + e * n * n;
      const float* W = w.data() + e * n;
      for (int j = 1; j < n; ++j) ASSERT_LE(W[j - 1], W[j]);
      for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
          double s = 0;  // (A V)(i, j) = w_j V(i, j)
          for (int t = 0; t < n; ++t) s += A(e, i, t) * V[j * n + t];
          ASSERT_NEAR(s, double{W[j]} * V[j * n + i], 64 * n * kU32 * 4)
              << "b " << e << " (" << i << ", " << j << ")";
        }
      }
    }
  }
  EXPECT_THAT(HostSyevd(nullptr, nullptr, 2, 0, true), IsOk());
}

// A non-finite value in the triangle read gives NaN without calling LAPACK
// (whose result for it is not specified); the other batch elements are
// unaffected.
TEST(LapackHostTest, SyevdNonFiniteInput) {
  const int n = 8;
  for (bool lower : {true, false}) {
    SCOPED_TRACE(absl::StrCat("lower ", lower));
    std::vector<float> a(3 * n * n, 0.0f);
    for (int64_t e = 0; e < 3; ++e) {
      for (int i = 0; i < n; ++i) a[e * n * n + i * n + i] = 1.0f + i;
    }
    // Column-major (i, j) at j * n + i; (5, 2) is in the lower triangle.
    const int64_t off = lower ? 2 * n + 5 : 5 * n + 2;
    a[1 * n * n + off] = std::numeric_limits<float>::infinity();
    a[2 * n * n + off] = kNaN;
    std::vector<float> w(3 * n);
    ASSERT_THAT(HostSyevd(a.data(), w.data(), 3, n, lower), IsOk());
    for (int i = 0; i < n; ++i) EXPECT_EQ(w[i], 1.0f + i);
    for (int64_t e = 1; e < 3; ++e) {
      for (int i = 0; i < n; ++i) EXPECT_TRUE(std::isnan(w[e * n + i]));
      for (int i = 0; i < n * n; ++i) {
        EXPECT_TRUE(std::isnan(a[e * n * n + i]));
      }
    }
  }
}

// LAPACK 3.10+ sgesdd calls a NaN input an illegal argument; it gives NaN
// (as on JAX's CPU backend), not an error.
TEST(LapackHostTest, GesddNonFiniteInput) {
  const int m = 6, n = 4, k = 4;
  for (float bad : {kNaN, std::numeric_limits<float>::infinity()}) {
    std::vector<float> a(2 * m * n, 0.0f);
    for (int64_t e = 0; e < 2; ++e) {
      for (int i = 0; i < k; ++i) a[e * m * n + i * m + i] = 1.0f + i;
    }
    a[m * n + 3] = bad;
    std::vector<float> s(2 * k), u(2 * m * k), vt(2 * k * n);
    ASSERT_THAT(HostGesdd(a.data(), s.data(), u.data(), vt.data(), 2, m, n,
                          /*full_matrices=*/false),
                IsOk());
    for (int i = 0; i < k; ++i) EXPECT_EQ(s[i], 4.0f - i);
    for (int i = 0; i < k; ++i) EXPECT_TRUE(std::isnan(s[k + i]));
    for (int i = 0; i < m * k; ++i) EXPECT_TRUE(std::isnan(u[m * k + i]));
    for (int i = 0; i < k * n; ++i) EXPECT_TRUE(std::isnan(vt[k * n + i]));
    std::vector<float> s2(2 * k);
    a[m * n + 3] = bad;
    ASSERT_THAT(HostGesdd(a.data(), s2.data(), nullptr, nullptr, 2, m, n,
                          false),
                IsOk());
    EXPECT_TRUE(std::isnan(s2[k]));
  }
}

TEST(LapackHostTest, Gesdd) {
  std::mt19937 rng(6);
  for (auto [m, n] : {std::pair{40, 25}, {25, 40}}) {
    const int k = std::min(m, n);
    const int64_t batch = 2;
    const std::vector<float> a = Random(rng, batch * m * n);
    std::vector<float> s_novec(batch * k);
    {
      std::vector<float> x = a;
      ASSERT_THAT(HostGesdd(x.data(), s_novec.data(), nullptr, nullptr, batch,
                            m, n, false),
                  IsOk());
    }
    for (bool full : {false, true}) {
      SCOPED_TRACE(absl::StrCat(m, "x", n, " full ", full));
      const int ucols = full ? m : k, vtrows = full ? n : k;
      std::vector<float> x = a;
      std::vector<float> s(batch * k), u(batch * m * ucols),
          vt(batch * vtrows * n);
      ASSERT_THAT(HostGesdd(x.data(), s.data(), u.data(), vt.data(), batch, m,
                            n, full),
                  IsOk());
      for (int64_t e = 0; e < batch; ++e) {
        const float* S = s.data() + e * k;
        const float* U = u.data() + e * m * ucols;
        const float* Vt = vt.data() + e * vtrows * n;
        for (int j = 0; j < k; ++j) {
          ASSERT_GE(S[j], 0.0f);
          if (j > 0) ASSERT_LE(S[j], S[j - 1]);
          ASSERT_NEAR(S[j], s_novec[e * k + j], 64 * k * kU32 * S[0]);
        }
        // U[:, :k] diag(S) Vt[:k, :] == A (column-major).
        for (int i = 0; i < m; ++i) {
          for (int j = 0; j < n; ++j) {
            double r = 0;
            for (int t = 0; t < k; ++t) {
              r += double{U[t * m + i]} * S[t] * Vt[j * vtrows + t];
            }
            ASSERT_NEAR(r, a[e * m * n + j * m + i], 64 * k * kU32 * S[0])
                << "b " << e << " (" << i << ", " << j << ")";
          }
        }
      }
    }
  }
  EXPECT_THAT(HostGesdd(nullptr, nullptr, nullptr, nullptr, 2, 0, 4, true),
              IsOk());
}

TEST(LapackHostTest, Workspace) {
  // One float step up, then the ceiling: exact integer queries gain one.
  EXPECT_THAT(LapackWorkspace("w", 100.0f, 10), IsOkAndHolds(101));
  // 2^24 + 1 is not a float: a query that rounded down to 2^24 still gets
  // at least 2^24 + 1 (the next float is 2^24 + 2).
  EXPECT_THAT(LapackWorkspace("w", 16777216.0f, 1), IsOkAndHolds(16777218));
  // The documented minimum wins, and at least 1.
  EXPECT_THAT(LapackWorkspace("w", 5.0f, 1000), IsOkAndHolds(1000));
  EXPECT_THAT(LapackWorkspace("w", 0.0f, 0), IsOkAndHolds(1));
  EXPECT_THAT(LapackWorkspace("w", 0.0f, -5), IsOkAndHolds(1));
  // Beyond 32-bit LAPACK.
  EXPECT_THAT(LapackWorkspace("w", 3e9f, 1),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(LapackWorkspace("w", 1.0f, int64_t{1} << 40),
              StatusIs(absl::StatusCode::kInvalidArgument));
  // The largest floats below 2^31: the step up reaches 2^31 for the last.
  EXPECT_THAT(LapackWorkspace("w", 2147483392.0f, 1),
              IsOkAndHolds(2147483520));
  EXPECT_THAT(LapackWorkspace("w", 2147483520.0f, 1),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace linalg
}  // namespace metal_pjrt
