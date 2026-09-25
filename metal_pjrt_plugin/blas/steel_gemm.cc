#include "metal_pjrt_plugin/blas/steel_gemm.h"

#include <Metal/Metal.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/str_cat.h"
#include "metal_pjrt_plugin/blas/steel_gemm_msl.h"

namespace metal_pjrt {
namespace blas {
namespace {

// Must match SteelGemmParams in steel_gemm_msl.h.
struct SteelGemmParams {
  int32_t M, N, K;
  int32_t lda, ldb, ldd;
  int32_t tiles_n, tiles_m;
  int64_t batch_stride_a, batch_stride_b, batch_stride_d;
  int32_t swizzle_log;
  int32_t gemm_k_iterations_aligned;
  float alpha, beta;
};
static_assert(sizeof(SteelGemmParams) == 72, "layout must match the MSL");

enum class Forced { kNone, kMps, kSteel };

Forced ForcedBackend() {
  static const Forced f = [] {
    const char* e = std::getenv("METAL_PJRT_GEMM");
    if (e == nullptr) return Forced::kNone;
    if (std::strcmp(e, "mps") == 0) return Forced::kMps;
    if (std::strcmp(e, "steel") == 0) return Forced::kSteel;
    return Forced::kNone;
  }();
  return f;
}

bool EnvTile(SteelTile* t) {
  static const std::pair<bool, SteelTile> env = [] {
    SteelTile t;
    const char* e = std::getenv("METAL_PJRT_STEEL_TILE");
    if (e == nullptr) return std::make_pair(false, t);
    bool ok = std::sscanf(e, "%d,%d,%d,%d,%d", &t.bm, &t.bn, &t.bk, &t.wm,
                          &t.wn) == 5;
    return std::make_pair(ok, t);
  }();
  if (env.first) *t = env.second;
  return env.first;
}

const char* MslType(MpsDType t) {
  switch (t) {
    case MpsDType::kF32:
      return "float";
    case MpsDType::kF16:
      return "half";
    case MpsDType::kBF16:
      return "bfloat";
  }
  return "?";
}

bool Pow2(int x) { return x > 0 && (x & (x - 1)) == 0; }

// Loader geometry (BlockLoader in the MSL): a BROWS x BCOLS tile read by
// `threads` threads, n_reads contiguous elements each, must split evenly.
bool LoaderOk(int rows, int cols, int threads) {
  if ((rows * cols) % threads != 0) return false;
  const int n_reads = rows * cols / threads;
  return n_reads >= 1 && cols % n_reads == 0;
}

bool ValidTileFor(const SteelTile& t, const GemmParams& p) {
  if (!Pow2(t.bm) || !Pow2(t.bn) || !Pow2(t.bk) || !Pow2(t.wm) ||
      !Pow2(t.wn) || t.bk < 8 || t.bm % (8 * t.wm) != 0 ||
      t.bn % (8 * t.wn) != 0) {
    return false;
  }
  const int threads = 32 * t.wm * t.wn;
  if (threads > 1024) return false;
  const bool loader_a = p.a.transpose ? LoaderOk(t.bk, t.bm, threads)
                                      : LoaderOk(t.bm, t.bk, threads);
  const bool loader_b = p.b.transpose ? LoaderOk(t.bn, t.bk, threads)
                                      : LoaderOk(t.bk, t.bn, threads);
  return loader_a && loader_b;
}

std::string VariantSource(MpsDType in, MpsDType out, const SteelTile& t,
                          bool ta, bool tb, bool mn_aligned, bool k_aligned,
                          bool use_c) {
  auto b = [](bool v) { return v ? "true" : "false"; };
  return absl::StrCat(
      "#include <metal_stdlib>\nusing namespace metal;\n",
      "typedef ", MslType(in), " T;\ntypedef ", MslType(out), " U;\n",
      "constant constexpr int BM = ", t.bm, ";\n",
      "constant constexpr int BN = ", t.bn, ";\n",
      "constant constexpr int BK = ", t.bk, ";\n",
      "constant constexpr int WM = ", t.wm, ";\n",
      "constant constexpr int WN = ", t.wn, ";\n",
      "constant constexpr bool TRANS_A = ", b(ta), ";\n",
      "constant constexpr bool TRANS_B = ", b(tb), ";\n",
      "constant constexpr bool MN_ALIGNED = ", b(mn_aligned), ";\n",
      "constant constexpr bool K_ALIGNED = ", b(k_aligned), ";\n",
      "constant constexpr bool USE_C = ", b(use_c), ";\n", kSteelGemmMslBody);
}

absl::StatusOr<std::shared_ptr<rt::Kernel>> GetKernel(
    rt::Device* device, const std::string& source) {
  static std::mutex mu;
  static auto* cache =
      new std::unordered_map<std::string, std::shared_ptr<rt::Kernel>>();
  // Key on the variant prefix (everything before the shared body) + device.
  const std::string key =
      absl::StrCat(reinterpret_cast<uintptr_t>(device), "|",
                   source.substr(0, source.size() -
                                        (sizeof(kSteelGemmMslBody) - 1)));
  {
    std::lock_guard<std::mutex> lock(mu);
    auto it = cache->find(key);
    if (it != cache->end()) return it->second;
  }
  ABSL_ASSIGN_OR_RETURN(MTL::Library * lib, device->CompileLibrary(source));
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<rt::Kernel> k,
                        device->CreateKernel(lib, kSteelGemmKernelName));
  std::shared_ptr<rt::Kernel> shared(std::move(k));
  std::lock_guard<std::mutex> lock(mu);
  return cache->emplace(key, shared).first->second;
}

const void* DevicePtr(const MpsOperand& x) {
  auto* buf = static_cast<MTL::Buffer*>(x.buffer);
  return static_cast<const char*>(buf->contents()) + x.offset;
}

bool FitsInt(int64_t v) {
  return v >= 0 && v <= std::numeric_limits<int32_t>::max();
}

}  // namespace

SteelTile ChooseSteelTile(const GemmParams& p) {
  SteelTile t;
  if (EnvTile(&t) && ValidTileFor(t, p)) return t;
  const bool nt = !p.a.transpose && p.b.transpose;
  const int64_t tiles64 = ((p.m + 63) / 64) * ((p.n + 63) / 64) *
                          std::max<int64_t>(p.batch_count, 1);
  if (tiles64 < 32) {
    // Too few 64x64 tiles to fill the GPU: smaller tiles, more threadgroups.
    t = {32, 32, 16, 2, 2};
  } else if (nt) {
    t = {64, 32, 32, 2, 2};
  } else if (p.a.dtype == MpsDType::kF32) {
    t = {64, 64, 16, 2, 2};
  } else {
    t = {64, 64, 16, 1, 2};  // MLX's choice for half/bf16 on base M-series.
  }
  return t;
}

bool SteelGemmSupports(const GemmParams& p, std::string* why) {
  auto no = [&](const char* w) {
    if (why != nullptr) *why = w;
    return false;
  };
  if (p.a.dtype != p.b.dtype) return no("A and B dtypes differ");
  if (p.c.dtype != p.a.dtype && p.c.dtype != MpsDType::kF32)
    return no("unsupported output dtype");
  if (p.c.transpose) return no("transposed C");
  if (!FitsInt(p.m) || !FitsInt(p.n) || !FitsInt(p.k))
    return no("dimension exceeds int32");
  // Loader/tile index math is 32-bit within a matrix (MLX convention); keep
  // ld * 128 (largest tile extent) comfortably in range.
  constexpr int64_t kMaxLd = std::numeric_limits<int32_t>::max() / 256;
  for (const MpsOperand* x : {&p.a, &p.b, &p.c}) {
    if (x->ld < 0 || x->ld > kMaxLd) return no("leading dimension too large");
    if (x->batch_stride < 0) return no("negative batch stride");
  }
  if (p.batch_count > std::numeric_limits<int32_t>::max())
    return no("batch too large");
  return true;
}

bool UseSteelGemm(const GemmParams& p) {
  const Forced f = ForcedBackend();
  if (f == Forced::kMps) return false;
  if (f != Forced::kSteel && p.a.dtype == MpsDType::kF32) return false;
  return SteelGemmSupports(p);
}

absl::Status RunSteelGemm(rt::Device* device, rt::Stream* stream,
                          const GemmParams& p, const SteelTile* tile) {
  std::string why;
  if (!SteelGemmSupports(p, &why)) {
    return absl::InvalidArgumentError(
        absl::StrCat("RunSteelGemm: ", why, ": ", GemmParamsDebugString(p)));
  }
  if (p.m < 0 || p.n < 0 || p.k < 0 || p.batch_count < 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "RunSteelGemm: negative size: ", GemmParamsDebugString(p)));
  }
  const int64_t a_cols = p.a.transpose ? p.m : p.k;
  const int64_t b_cols = p.b.transpose ? p.k : p.n;
  if ((p.k > 0 && ((p.m > 0 && p.a.ld < a_cols) ||
                   (p.n > 0 && p.b.ld < b_cols))) ||
      (p.m > 0 && p.c.ld < p.n)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "RunSteelGemm: leading dimension smaller than row length: ",
        GemmParamsDebugString(p)));
  }
  if (p.m == 0 || p.n == 0 || p.batch_count == 0) return absl::OkStatus();
  for (const MpsOperand* x : {&p.a, &p.b, &p.c}) {
    if (x->buffer == nullptr && (x == &p.c || p.k > 0)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "RunSteelGemm: null buffer: ", GemmParamsDebugString(p)));
    }
  }

  SteelTile t = tile != nullptr ? *tile : ChooseSteelTile(p);
  if (!ValidTileFor(t, p)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "RunSteelGemm: invalid tile ", t.bm, "x", t.bn, "x", t.bk, " wm=",
        t.wm, " wn=", t.wn));
  }
  const bool use_c = p.beta != 0.0;
  const bool mn_aligned = p.m % t.bm == 0 && p.n % t.bn == 0;
  const bool k_aligned = p.k % t.bk == 0;
  ABSL_ASSIGN_OR_RETURN(
      std::shared_ptr<rt::Kernel> kernel,
      GetKernel(device, VariantSource(p.a.dtype, p.c.dtype, t, p.a.transpose,
                                      p.b.transpose, mn_aligned, k_aligned,
                                      use_c)));

  const int tn = static_cast<int>((p.n + t.bn - 1) / t.bn);
  const int tm = static_cast<int>((p.m + t.bm - 1) / t.bm);
  const int swizzle_log = tm <= 3 ? 0 : 1;
  const bool batched = p.batch_count > 1;
  SteelGemmParams sp;
  sp.M = static_cast<int32_t>(p.m);
  sp.N = static_cast<int32_t>(p.n);
  sp.K = static_cast<int32_t>(p.k);
  sp.lda = static_cast<int32_t>(p.a.ld);
  sp.ldb = static_cast<int32_t>(p.b.ld);
  sp.ldd = static_cast<int32_t>(p.c.ld);
  sp.tiles_n = tn;
  sp.tiles_m = tm;
  sp.batch_stride_a = batched ? p.a.batch_stride : 0;
  sp.batch_stride_b = batched ? p.b.batch_stride : 0;
  sp.batch_stride_d = batched ? p.c.batch_stride : 0;
  sp.swizzle_log = swizzle_log;
  sp.gemm_k_iterations_aligned = static_cast<int32_t>(p.k / t.bk);
  sp.alpha = static_cast<float>(p.alpha);
  sp.beta = static_cast<float>(p.beta);

  const int sw = 1 << swizzle_log;
  rt::Dim3 groups{static_cast<uint32_t>(tn * sw),
                  static_cast<uint32_t>((tm + sw - 1) / sw),
                  static_cast<uint32_t>(p.batch_count)};
  rt::Dim3 threads{static_cast<uint32_t>(32 * t.wm * t.wn), 1, 1};
  // With k == 0, A and B are never read; bind D in their place if missing.
  const void* d = DevicePtr(p.c);
  const void* a = p.a.buffer != nullptr ? DevicePtr(p.a) : d;
  const void* b = p.b.buffer != nullptr ? DevicePtr(p.b) : d;
  return stream->Launch(*kernel, groups, threads,
                        {rt::KernelArg::Buffer(a), rt::KernelArg::Buffer(b),
                         rt::KernelArg::Buffer(d),
                         rt::KernelArg::Bytes(&sp, sizeof(sp))});
}

}  // namespace blas
}  // namespace metal_pjrt
