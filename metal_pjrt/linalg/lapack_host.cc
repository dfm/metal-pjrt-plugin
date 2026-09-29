#include "metal_pjrt/linalg/lapack_host.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "metal_pjrt/compiler/report_bug.h"

// Accelerate's classic LAPACK (LP64, 32-bit integers) and CBLAS interfaces.
// Declared here rather than through <Accelerate/Accelerate.h> to keep the
// framework headers out of the build.
extern "C" {
int spotrf_(const char* uplo, const int* n, float* a, const int* lda,
            int* info);
int sgetrf_(const int* m, const int* n, float* a, const int* lda, int* ipiv,
            int* info);
int sgeqrf_(const int* m, const int* n, float* a, const int* lda, float* tau,
            float* work, const int* lwork, int* info);
int sorgqr_(const int* m, const int* n, const int* k, float* a, const int* lda,
            const float* tau, float* work, const int* lwork, int* info);
int ssyevd_(const char* jobz, const char* uplo, const int* n, float* a,
            const int* lda, float* w, float* work, const int* lwork, int* iwork,
            const int* liwork, int* info);
int sgesdd_(const char* jobz, const int* m, const int* n, float* a,
            const int* lda, float* s, float* u, const int* ldu, float* vt,
            const int* ldvt, float* work, const int* lwork, int* iwork,
            int* info);
void cblas_strsm(int order, int side, int uplo, int transa, int diag, int m,
                 int n, float alpha, const float* a, int lda, float* b,
                 int ldb);
}

namespace metal_pjrt {
namespace linalg {
namespace {

// CBLAS enum values (cblas.h).
constexpr int kCblasRowMajor = 101;
constexpr int kCblasNoTrans = 111;
constexpr int kCblasTrans = 112;
constexpr int kCblasUpper = 121;
constexpr int kCblasLower = 122;
constexpr int kCblasNonUnit = 131;
constexpr int kCblasUnit = 132;
constexpr int kCblasLeft = 141;
constexpr int kCblasRight = 142;

constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

absl::Status LapackError(const char* name, int info) {
  return absl::InternalError(
      absl::StrCat(name, ": LAPACK reported illegal argument ", -info,
                   metal_pjrt::kReportBug));
}

}  // namespace

absl::StatusOr<int> LapackWorkspace(const char* name, float query,
                                    int64_t minimum) {
  const double w = std::max<double>(
      std::ceil(std::nextafter(query, std::numeric_limits<float>::infinity())),
      static_cast<double>(std::max<int64_t>(minimum, 1)));
  if (w > std::numeric_limits<int>::max()) {
    return absl::InvalidArgumentError(
        absl::StrCat(name, ": workspace too large for 32-bit LAPACK"));
  }
  return static_cast<int>(w);
}

//===--------------------------------------------------------------------===//
// HLO-level ops (row-major).
//===--------------------------------------------------------------------===//

absl::Status HostCholesky(float* x, int64_t batch, int n, bool lower) {
  constexpr char kName[] = "metal$cholesky";
  if (n == 0) return absl::OkStatus();
  // Row-major lower L (A = L L^T) is, read column-major, the upper factor U =
  // L^T with A^T = A = U^T U; likewise row-major upper is column-major lower.
  const char uplo = lower ? 'U' : 'L';
  for (int64_t b = 0; b < batch; ++b) {
    float* m = x + b * int64_t{n} * n;
    int info = 0;
    spotrf_(&uplo, &n, m, &n, &info);
    if (info < 0) return LapackError(kName, info);
    if (info > 0) {
      std::fill(m, m + int64_t{n} * n, kNaN);
      continue;
    }
    // Zero the (row-major) triangle LAPACK did not write.
    for (int i = 0; i < n; ++i) {
      float* row = m + int64_t{i} * n;
      if (lower) {
        std::fill(row + i + 1, row + n, 0.0f);
      } else {
        std::fill(row, row + i, 0.0f);
      }
    }
  }
  return absl::OkStatus();
}

void HostTriangularSolve(const float* a, float* x, int64_t batch, int m,
                         int n, bool left_side, bool lower, bool transpose,
                         bool unit_diagonal) {
  if (m == 0 || n == 0) return;
  const int64_t k = left_side ? m : n;
  for (int64_t i = 0; i < batch; ++i) {
    cblas_strsm(kCblasRowMajor, left_side ? kCblasLeft : kCblasRight,
                lower ? kCblasLower : kCblasUpper,
                transpose ? kCblasTrans : kCblasNoTrans,
                unit_diagonal ? kCblasUnit : kCblasNonUnit, m, n, 1.0f,
                a + i * k * k, static_cast<int>(k), x + i * int64_t{m} * n, n);
  }
}

//===--------------------------------------------------------------------===//
// JAX-level ops (column-major matrices).
//===--------------------------------------------------------------------===//

absl::Status HostGetrf(float* x, int32_t* pivots, int32_t* permutation,
                       int64_t batch, int m, int n) {
  constexpr char kName[] = "metal$lapack_getrf";
  const int k = std::min(m, n);
  for (int64_t b = 0; b < batch; ++b) {
    float* mb = x + b * int64_t{m} * n;
    int32_t* pb = pivots + b * k;
    int32_t* qb = permutation + b * m;
    int info = 0;
    if (m > 0 && n > 0) {
      const int lda = std::max(m, 1);
      sgetrf_(&m, &n, mb, &lda, pb, &info);
    }
    if (info < 0) return LapackError(kName, info);
    for (int i = 0; i < m; ++i) qb[i] = i;
    for (int i = 0; i < k; ++i) {
      pb[i] -= 1;  // 1-based to 0-based
      std::swap(qb[i], qb[pb[i]]);
    }
  }
  return absl::OkStatus();
}

absl::Status HostGeqrf(float* x, float* taus, int64_t batch, int m, int n) {
  constexpr char kName[] = "metal$lapack_geqrf";
  const int k = std::min(m, n);
  if (k == 0) return absl::OkStatus();
  const int lda = std::max(m, 1);
  int info = 0;
  int lwork = -1;
  float wq = 0;
  sgeqrf_(&m, &n, x, &lda, taus, &wq, &lwork, &info);
  if (info < 0) return LapackError(kName, info);
  absl::StatusOr<int> ws = LapackWorkspace(kName, wq, n);
  if (!ws.ok()) return ws.status();
  lwork = *ws;
  std::vector<float> work(lwork);
  for (int64_t b = 0; b < batch; ++b) {
    sgeqrf_(&m, &n, x + b * int64_t{m} * n, &lda, taus + b * k, work.data(),
            &lwork, &info);
    if (info < 0) return LapackError(kName, info);
  }
  return absl::OkStatus();
}

absl::Status HostOrgqr(float* x, const float* taus, int64_t batch, int m,
                       int n, int k) {
  constexpr char kName[] = "metal$lapack_orgqr";
  if (n == 0) return absl::OkStatus();
  const int lda = std::max(m, 1);
  int info = 0;
  int lwork = -1;
  float wq = 0;
  sorgqr_(&m, &n, &k, x, &lda, taus, &wq, &lwork, &info);
  if (info < 0) return LapackError(kName, info);
  absl::StatusOr<int> ws = LapackWorkspace(kName, wq, n);
  if (!ws.ok()) return ws.status();
  lwork = *ws;
  std::vector<float> work(lwork);
  for (int64_t b = 0; b < batch; ++b) {
    sorgqr_(&m, &n, &k, x + b * int64_t{m} * n, &lda, taus + b * k,
            work.data(), &lwork, &info);
    if (info < 0) return LapackError(kName, info);
  }
  return absl::OkStatus();
}

absl::Status HostSyevd(float* x, float* w, int64_t batch, int n, bool lower) {
  constexpr char kName[] = "metal$lapack_syevd";
  if (n == 0) return absl::OkStatus();
  const char jobz = 'V';
  const char uplo = lower ? 'L' : 'U';
  int info = 0;
  int lwork = -1;
  int liwork = -1;
  float wq = 0;
  int iwq = 0;
  ssyevd_(&jobz, &uplo, &n, x, &n, w, &wq, &lwork, &iwq, &liwork, &info);
  if (info < 0) return LapackError(kName, info);
  // Documented minimums for jobz = 'V'.
  absl::StatusOr<int> ws =
      LapackWorkspace(kName, wq, 1 + 6 * int64_t{n} + 2 * int64_t{n} * n);
  if (!ws.ok()) return ws.status();
  lwork = *ws;
  liwork = std::max(3 + 5 * n, iwq);
  std::vector<float> work(lwork);
  std::vector<int> iwork(liwork);
  for (int64_t b = 0; b < batch; ++b) {
    float* mb = x + b * int64_t{n} * n;
    float* wb = w + b * n;
    ssyevd_(&jobz, &uplo, &n, mb, &n, wb, work.data(), &lwork, iwork.data(),
            &liwork, &info);
    if (info < 0) return LapackError(kName, info);
    if (info > 0) {
      std::fill(mb, mb + int64_t{n} * n, kNaN);
      std::fill(wb, wb + n, kNaN);
    }
  }
  return absl::OkStatus();
}

absl::Status HostGesdd(float* x, float* s, float* u, float* vt,
                       int64_t batch, int m, int n, bool full_matrices) {
  constexpr char kName[] = "metal$lapack_gesdd";
  const int k = std::min(m, n);
  if (k == 0) return absl::OkStatus();
  const bool uv = u != nullptr;
  const char jobz = !uv ? 'N' : (full_matrices ? 'A' : 'S');
  const int ucols = full_matrices ? m : k;
  const int vtrows = full_matrices ? n : k;
  const int ldu = std::max(m, 1);
  const int ldvt = uv ? std::max(vtrows, 1) : 1;
  float udummy = 0, vtdummy = 0;
  int info = 0;
  int lwork = -1;
  float wq = 0;
  std::vector<int> iwork(8 * static_cast<size_t>(k));
  sgesdd_(&jobz, &m, &n, x, &m, s, uv ? u : &udummy, &ldu,
          uv ? vt : &vtdummy, &ldvt, &wq, &lwork, iwork.data(), &info);
  if (info < 0) return LapackError(kName, info);
  // Documented minimums: jobz = 'N' needs 3mn + max(mx, 7mn); 'S' and 'A'
  // need 4mn^2 + 7mn and 4mn^2 + 6mn + mx (LAPACK 3.7+), older releases
  // 3mn + max(mx, 4mn^2 + 4mn). Take the largest that applies.
  const int64_t mn = k;
  const int64_t mx = std::max(m, n);
  const int64_t min_work =
      !uv ? 3 * mn + std::max(mx, 7 * mn)
          : std::max({4 * mn * mn + 7 * mn, 4 * mn * mn + 6 * mn + mx,
                      3 * mn + std::max(mx, 4 * mn * mn + 4 * mn)});
  absl::StatusOr<int> ws = LapackWorkspace(kName, wq, min_work);
  if (!ws.ok()) return ws.status();
  lwork = *ws;
  std::vector<float> work(lwork);
  for (int64_t b = 0; b < batch; ++b) {
    float* xb = x + b * int64_t{m} * n;
    float* sb = s + b * k;
    float* ub = uv ? u + b * int64_t{m} * ucols : &udummy;
    float* vb = uv ? vt + b * int64_t{vtrows} * n : &vtdummy;
    sgesdd_(&jobz, &m, &n, xb, &m, sb, ub, &ldu, vb, &ldvt, work.data(),
            &lwork, iwork.data(), &info);
    if (info < 0) return LapackError(kName, info);
    if (info > 0) {
      std::fill(sb, sb + k, kNaN);
      if (uv) {
        std::fill(ub, ub + int64_t{m} * ucols, kNaN);
        std::fill(vb, vb + int64_t{vtrows} * n, kNaN);
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace linalg
}  // namespace metal_pjrt
