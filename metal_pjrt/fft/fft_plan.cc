#include "metal_pjrt/fft/fft_plan.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <set>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"

namespace metal_pjrt {
namespace fft {
namespace {

using cdouble = std::complex<double>;

bool IsPowerOf2(int64_t n) { return n > 0 && (n & (n - 1)) == 0; }

int64_t NextPowerOf2(int64_t n) {
  int64_t p = 1;
  while (p < n) p *= 2;
  return p;
}

// prime_factors (fft.cpp 40-55): with multiplicity, ascending.
std::vector<int> PrimeFactors(int n) {
  int z = 2;
  std::vector<int> factors;
  while (z * z <= n) {
    if (n % z == 0) {
      factors.push_back(z);
      n /= z;
    } else {
      z++;
    }
  }
  if (n > 1) {
    factors.push_back(n);
  }
  return factors;
}

bool IsRadix(int f) {
  return std::find(std::begin(kRadices), std::end(kRadices), f) !=
         std::end(kRadices);
}

// plan_stockham_fft (fft.cpp 94-116); false where MLX throws
// "Unplannable" (a prime factor above 13).
bool PlanStockham(int n, std::array<int, kNumRadices>& plan) {
  plan.fill(0);
  int orig_n = n;
  if (n == 1) {
    return true;
  }
  for (int i = 0; i < kNumRadices; i++) {
    int radix = kRadices[i];
    // Manually tuned radices for powers of 2
    if (IsPowerOf2(orig_n) && orig_n < 512 && radix > 4) {
      continue;
    }
    while (n % radix == 0) {
      plan[i] += 1;
      n /= radix;
      if (n == 1) {
        return true;
      }
    }
  }
  return false;
}

// mod_exp (fft.cpp 251-261).
int ModExp(int x, int y, int n) {
  int out = 1;
  while (y) {
    if (y & 1) {
      out = out * x % n;
    }
    y >>= 1;
    x = x * x % n;
  }
  return out;
}

uint64_t Align256(uint64_t x) { return (x + 255) / 256 * 256; }

void PutComplex(std::vector<uint8_t>& bytes, uint64_t offset,
                const std::vector<cdouble>& v) {
  for (size_t i = 0; i < v.size(); ++i) {
    const float f[2] = {static_cast<float>(v[i].real()),
                        static_cast<float>(v[i].imag())};
    std::memcpy(bytes.data() + offset + 8 * i, f, 8);
  }
}

void PutShort(std::vector<uint8_t>& bytes, uint64_t offset,
              const std::vector<int16_t>& v) {
  std::memcpy(bytes.data() + offset, v.data(), 2 * v.size());
}

}  // namespace

namespace internal {

cdouble Twiddle(int64_t k, int64_t n) {
  k %= n;
  if (k < 0) k += n;
  const double theta = -2.0 * M_PI * static_cast<double>(k) / n;
  return {std::cos(theta), std::sin(theta)};
}

void Pow2FftDouble(std::vector<cdouble>& x) {
  const int64_t n = static_cast<int64_t>(x.size());
  if (n <= 1) return;
  // Bit reversal.
  for (int64_t i = 1, j = 0; i < n; ++i) {
    int64_t bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) std::swap(x[i], x[j]);
  }
  // Twiddle(t, n) for t < n / 2 as lo[t % kL] * hi[t / kL]: two small
  // tables, each entry accurate to an ulp.
  constexpr int64_t kL = 4096;
  std::vector<cdouble> lo(std::min(kL, n)), hi(n / 2 / kL + 1);
  for (int64_t t = 0; t < static_cast<int64_t>(lo.size()); ++t) {
    lo[t] = Twiddle(t, n);
  }
  for (int64_t t = 0; t < static_cast<int64_t>(hi.size()); ++t) {
    hi[t] = Twiddle(t * kL, n);
  }
  for (int64_t len = 2; len <= n; len *= 2) {
    const int64_t half = len / 2, step = n / len;
    for (int64_t j = 0; j < half; ++j) {
      const int64_t t = j * step;
      const cdouble w = lo[t % kL] * hi[t / kL];
      for (int64_t i = j; i < n; i += len) {
        const cdouble u = x[i], v = x[i + half] * w;
        x[i] = u + v;
        x[i + half] = u - v;
      }
    }
  }
}

// primitive_root (fft.cpp 263-279).
int PrimitiveRoot(int n) {
  auto factors = PrimeFactors(n - 1);

  for (int r = 2; r < n - 1; r++) {
    bool found = true;
    for (int factor : factors) {
      if (ModExp(r, (n - 1) / factor, n) == 1) {
        found = false;
        break;
      }
    }
    if (found) {
      return r;
    }
  }
  return -1;
}

}  // namespace internal

const char* FftTypeName(FftType type) {
  switch (type) {
    case FftType::kFft:
      return "fft";
    case FftType::kIfft:
      return "ifft";
    case FftType::kRfft:
      return "rfft";
    case FftType::kIrfft:
      return "irfft";
  }
  return "?";
}

bool IsInverse(FftType type) {
  return type == FftType::kIfft || type == FftType::kIrfft;
}

bool IsReal(FftType type) {
  return type == FftType::kRfft || type == FftType::kIrfft;
}

int64_t InputLength(FftType type, int64_t n) {
  return type == FftType::kIrfft ? n / 2 + 1 : n;
}

int64_t OutputLength(FftType type, int64_t n) {
  return type == FftType::kRfft ? n / 2 + 1 : n;
}

absl::StatusOr<FftPlan> PlanFft(int64_t n64) {
  if (n64 < 1 || n64 > kMaxFftSize) {
    return absl::UnimplementedError(
        absl::StrCat("FFT length ", n64, " not supported on Metal (1 to ",
                     kMaxFftSize, ")"));
  }
  const int n = static_cast<int>(n64);
  FftPlan plan;
  plan.n = n;
  auto factors = PrimeFactors(n);
  int remaining_n = n;

  // Four Step FFT when N is too large for shared mem.
  if (n > kMaxStockhamSize && IsPowerOf2(n)) {
    // For power's of two we have a fast, no transpose four step
    // implementation.
    plan.four_step = true;
    // Rough heuristic for choosing faster powers of two when we can
    plan.n2 = n > 65536 ? 1024 : 64;
    plan.n1 = n / plan.n2;
    // Each step is a single threadgroup FFT, so neither factor can be larger
    // than the largest Stockham size. The no transpose kernels cannot
    // decompose a step any further, so grow n2 instead of nesting four step.
    if (plan.n1 > kMaxStockhamSize) {
      plan.n1 = kMaxStockhamSize;
      plan.n2 = n / plan.n1;
    }
    // (MLX throws for n2 > 4096 here; n <= kMaxFftSize rules it out.)
    return plan;
  } else if (n > kMaxStockhamSize) {
    // Otherwise we use a multi-upload Bluestein's
    plan.four_step = true;
    plan.bluestein_n = static_cast<int>(NextPowerOf2(2 * int64_t{n} - 1));
    if (plan.bluestein_n > kMaxFftSize) {
      return absl::UnimplementedError(absl::StrCat(
          "FFT length ", n, " not supported on Metal (Bluestein length ",
          plan.bluestein_n, " > ", kMaxFftSize, ")"));
    }
    return plan;
  }

  // Bluestein's algorithm for n (fft.cpp 165-169, 177-181).
  auto bluestein = [&]() -> absl::StatusOr<FftPlan> {
    plan.four_step = n > kMaxBluesteinSize;
    plan.bluestein_n = static_cast<int>(NextPowerOf2(2 * int64_t{n} - 1));
    if (!PlanStockham(plan.bluestein_n, plan.stockham)) {
      return absl::InternalError("unplannable Bluestein length");
    }
    plan.rader.fill(0);
    return plan;
  };
  for (int factor : factors) {
    // Make sure the factor is a supported radix
    if (!IsRadix(factor)) {
      // We only support a single Rader factor currently
      if (plan.rader_n > 1 || n > kMaxRaderSize) {
        return bluestein();
      }
      // See if we can use Rader's algorithm to Stockham decompose n - 1
      for (int rf : PrimeFactors(factor - 1)) {
        // We don't nest Rader's algorithm so if `factor - 1`
        // isn't Stockham decomposable we give up and do Bluestein's.
        if (!IsRadix(rf)) {
          return bluestein();
        }
      }
      if (!PlanStockham(factor - 1, plan.rader)) {
        return absl::InternalError("unplannable Rader length");
      }
      plan.rader_n = factor;
      remaining_n /= factor;
    }
  }

  if (!PlanStockham(remaining_n, plan.stockham)) {
    return absl::InternalError("unplannable Stockham length");
  }
  return plan;
}

int ElemsPerThread(const FftPlan& plan) {
  // Heuristics for selecting an efficient number
  // of threads to use for a particular mixed-radix FFT.
  auto n = plan.n;

  std::vector<int> steps;
  steps.insert(steps.end(), plan.stockham.begin(), plan.stockham.end());
  steps.insert(steps.end(), plan.rader.begin(), plan.rader.end());
  std::set<int> used_radices;
  for (int i = 0; i < static_cast<int>(steps.size()); i++) {
    int radix = kRadices[i % kNumRadices];
    if (steps[i] > 0) {
      used_radices.insert(radix);
    }
  }

  // Manual tuning for 7/11/13
  if (used_radices.find(7) != used_radices.end() &&
      (used_radices.find(11) != used_radices.end() ||
       used_radices.find(13) != used_radices.end())) {
    return 7;
  } else if (used_radices.find(11) != used_radices.end() &&
             used_radices.find(13) != used_radices.end()) {
    return 11;
  }

  // TODO(alexbarron) Some really weird stuff is going on
  // for certain `elems_per_thread` on large composite n.
  // Possibly a compiler issue?
  if (n == 3159) return 13;
  if (n == 3645) return 5;
  if (n == 3969) return 7;
  if (n == 1982) return 5;

  if (used_radices.size() == 1) {
    return *(used_radices.begin());
  }
  if (used_radices.size() == 2) {
    if (used_radices.find(11) != used_radices.end() ||
        used_radices.find(13) != used_radices.end()) {
      return std::accumulate(used_radices.begin(), used_radices.end(), 0) / 2;
    }
    std::vector<int> radix_vec(used_radices.begin(), used_radices.end());
    return radix_vec[1];
  }
  // In all other cases use the second smallest radix.
  std::vector<int> radix_vec(used_radices.begin(), used_radices.end());
  return radix_vec[1];
}

FftLaunchGeometry LaunchGeometry(const FftPlan& plan, bool real,
                                 bool four_step, int64_t sequences) {
  // MLX's MIN_THREADGROUP_MEM_SIZE and MIN_COALESCE_WIDTH (fft.cpp 27-29).
  constexpr int kMinThreadgroupMemSize = 256;
  constexpr int kMinCoalesceWidth = 4;
  FftLaunchGeometry g;
  g.fft_size = plan.bluestein_n > 0 ? plan.bluestein_n : plan.n;
  g.elems_per_thread = ElemsPerThread(plan);
  g.threads_per_fft =
      (g.fft_size + g.elems_per_thread - 1) / g.elems_per_thread;
  // We batch among threadgroups for improved efficiency when n is small
  g.batch_per_group = std::max(kMinThreadgroupMemSize / g.fft_size, 1);
  if (four_step) {
    // Batch the four step FFT so we can coalesce the memory accesses, but
    // never past what fits in threadgroup memory.
    g.batch_per_group =
        std::max(g.batch_per_group,
                 std::min(kMinCoalesceWidth, kMaxStockhamSize / g.fft_size));
  }
  g.tg_mem_size =
      static_cast<int>(NextPowerOf2(int64_t{g.batch_per_group} * g.fft_size));
  g.groups = (sequences + g.batch_per_group - 1) / g.batch_per_group;
  if (real && !four_step) {
    // We can perform 2 RFFTs at once so the batch size is halved.
    g.groups = (g.groups + 2 - 1) / 2;
  }
  return g;
}

FftConstants MakeFftConstants(const FftPlan& plan) {
  FftConstants c;
  uint64_t size = 0;
  if (plan.bluestein_n > 0) {
    // compute_bluestein_constants (fft.cpp 323-370). In numpy:
    //   w_k = np.exp(-1j * np.pi / N * (np.arange(-N + 1, N) ** 2))
    //   w_q = np.fft.fft(1/w_k)
    // with pi i^2 / n reduced exactly, as 2 pi (i^2 mod 2n) / 2n.
    const int64_t n = plan.n, bn = plan.bluestein_n;
    std::vector<cdouble> w_k(n), w_q(bn, 0.0);
    for (int64_t i = -n + 1; i < n; i++) {
      const cdouble w = internal::Twiddle(i * i % (2 * n), 2 * n);
      w_q[i + n - 1] = std::conj(w);
      if (i >= 0) {
        w_k[i] = w;
      }
    }
    internal::Pow2FftDouble(w_q);
    c.w_q = size;
    size = Align256(size + 8 * bn);
    c.w_k = size;
    size = Align256(size + 8 * n);
    c.bytes.resize(size);
    PutComplex(c.bytes, c.w_q, w_q);
    PutComplex(c.bytes, c.w_k, w_k);
  } else if (plan.rader_n > 1) {
    // compute_raders_constants (fft.cpp 281-320), with the DFT of b_q as a
    // plain sum over an exact twiddle table.
    const int rader_n = plan.rader_n, m = rader_n - 1;
    int proot = internal::PrimitiveRoot(rader_n);
    // Fermat's little theorem
    int inv = ModExp(proot, rader_n - 2, rader_n);
    std::vector<int16_t> g_q(m), g_minus_q(m);
    for (int i = 0; i < m; i++) {
      g_q[i] = ModExp(proot, i, rader_n);
      g_minus_q[i] = ModExp(inv, i, rader_n);
    }
    std::vector<cdouble> b_q(m), table(m), b_q_fft(m);
    for (int i = 0; i < m; i++) {
      b_q[i] = internal::Twiddle(g_minus_q[i], rader_n);
      table[i] = internal::Twiddle(i, m);
    }
    for (int k = 0; k < m; ++k) {
      cdouble sum = 0.0;
      for (int j = 0; j < m; ++j) sum += b_q[j] * table[int64_t{j} * k % m];
      b_q_fft[k] = sum;
    }
    c.b_q = size;
    size = Align256(size + 8 * m);
    c.g_q = size;
    size = Align256(size + 2 * m);
    c.g_minus_q = size;
    size = Align256(size + 2 * m);
    c.bytes.resize(size);
    PutComplex(c.bytes, c.b_q, b_q_fft);
    PutShort(c.bytes, c.g_q, g_q);
    PutShort(c.bytes, c.g_minus_q, g_minus_q);
  }
  return c;
}

int64_t ChunkRows(const FftPlan& plan, int64_t rows,
                  int64_t max_chunk_elements) {
  const int64_t widest = std::max<int64_t>(plan.n, plan.bluestein_n);
  return std::clamp<int64_t>(max_chunk_elements / widest, 1,
                             std::max<int64_t>(rows, 1));
}

uint64_t FftWorkspaceBytes(const FftPlan& plan, int64_t rows,
                           int64_t max_chunk_elements) {
  if (!plan.four_step || rows == 0) return 0;
  const uint64_t chunk = ChunkRows(plan, rows, max_chunk_elements);
  return plan.bluestein_n > 0 ? 2 * chunk * plan.bluestein_n * 8
                              : chunk * plan.n * 8;
}

}  // namespace fft
}  // namespace metal_pjrt
