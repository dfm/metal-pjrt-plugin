// Compiles every embedded MSL source (kernels/*.metal) and creates a pipeline
// for every kernel the plugin can ask for, through rt::Device::GetKernel as
// the plugin does, so that a kernel that stops compiling fails this test
// rather than a user's first call. Needs a Metal device.
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "metal_pjrt/blas/mps_gemm.h"
#include "metal_pjrt/blas/gemv.h"
#include "metal_pjrt/blas/steel_gemm.h"
#include "metal_pjrt/conv/conv_kernels.h"
#include "metal_pjrt/fft/fft.h"
#include "metal_pjrt/kernels/conv_misc.metal.h"
#include "metal_pjrt/kernels/cub_sort.metal.h"
#include "metal_pjrt/kernels/fft.metal.h"
#include "metal_pjrt/kernels/fft_misc.metal.h"
#include "metal_pjrt/kernels/mps_staging.metal.h"
#include "metal_pjrt/kernels/msl_prelude.metal.h"
#include "metal_pjrt/kernels/runtime_builtins.metal.h"
#include "metal_pjrt/kernels/scan.metal.h"
#include "metal_pjrt/kernels/small_linalg.metal.h"
#include "metal_pjrt/kernels/gemv.metal.h"
#include "metal_pjrt/kernels/steel_conv.metal.h"
#include "metal_pjrt/kernels/steel_gemm.metal.h"
#include "metal_pjrt/runtime/metal_runtime.h"

namespace metal_pjrt {
namespace kernels {
namespace {

using ::absl_testing::IsOk;
using rt::FunctionConstant;

class KernelsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    absl::StatusOr<std::unique_ptr<rt::Device>> dev = rt::Device::Create(0);
    ASSERT_THAT(dev, IsOk());
    dev_ = *std::move(dev);
  }

  // The kernel functions in `source` (compiled directly, as GetKernel does:
  // fast math off). Empty, with a test failure, if it does not compile.
  std::vector<std::string> FunctionNames(const char* source) {
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    std::vector<std::string> names;
    MTL::CompileOptions* opts = MTL::CompileOptions::alloc()->init();
    opts->setFastMathEnabled(false);
    NS::Error* err = nullptr;
    MTL::Library* lib = dev_->mtl()->newLibrary(
        NS::String::string(source, NS::UTF8StringEncoding), opts, &err);
    opts->release();
    if (lib == nullptr) {
      ADD_FAILURE() << "does not compile: "
                    << (err ? err->localizedDescription()->utf8String() : "");
    } else {
      NS::Array* fns = lib->functionNames();
      for (NS::UInteger i = 0; i < fns->count(); ++i) {
        names.push_back(fns->object<NS::String>(i)->utf8String());
      }
      lib->release();
    }
    pool->release();
    return names;
  }

  void ExpectKernel(const char* source, const std::string& name,
                    absl::Span<const FunctionConstant> constants = {}) {
    absl::StatusOr<const rt::Kernel*> k =
        dev_->GetKernel(source, name, constants);
    EXPECT_THAT(k, IsOk()) << name;
  }

  std::unique_ptr<rt::Device> dev_;
};

// Sources that only define helpers (the emitter's prelude) or get their
// instantiation appended by the host (steel) still compile on their own.
TEST_F(KernelsTest, EverySourceCompiles) {
  for (const char* source :
       {kConvMiscMsl, kCubSortMsl, kFftMsl, kFftMiscMsl, kMpsStagingMsl,
        kMslPrelude, kRuntimeBuiltinsMsl, kScanMsl, kSmallLinalgMsl,
        kSteelConvMsl, kSteelGemmMsl, kGemvMsl}) {
    FunctionNames(source);
  }
}

// complex64 as the MSL emitter emits it (msl_emitter_test Complex64AsFloat2):
// a float2 aggregate constant, a constant table of 64-bit words loaded as
// float2, float2 through threadgroup memory and device buffers.
TEST_F(KernelsTest, EmitterComplex64Constructs) {
  const std::string source = absl::StrCat(kMslPrelude, R"(
constant uint64_t cplx_table[2] = {4611686019492741120ul, 4647714818650800128ul};
kernel void cplx(device char* in [[buffer(0)]], device char* out [[buffer(1)]],
                 uint3 tid [[thread_position_in_threadgroup]]) {
  threadgroup uint4 xla_shared_0[16];
  threadgroup char* tile = (threadgroup char*)xla_shared_0;
  float2 zero = {0.0e+00f, 0.0e+00f};
  float2 v = xla_load<float2>(xla_gep(in, (long)tid.x * 8));
  float2 t = xla_load<float2>(xla_gep(((constant char*)cplx_table), 8));
  float r = xla_vext<float>(v, 0) > 0.0f ? xla_vext<float>(t, 1) : 0.0f;
  float2 w = xla_vins(zero, r, 0);
  xla_store(xla_gep(tile, (long)tid.x * 8), w);
  xla_barrier();
  xla_store(xla_gep(out, (long)tid.x * 8),
            xla_load<float2>(xla_gep(tile, (long)tid.x * 8)));
}
)");
  ExpectKernel(source.c_str(), "cplx");
}

TEST_F(KernelsTest, RuntimeBuiltins) {
  using B = rt::Device::Builtin;
  for (B b : {B::kFill32, B::kFill8, B::kCopy16, B::kCopy8}) {
    EXPECT_THAT(dev_->BuiltinKernel(b), IsOk());
  }
  EXPECT_EQ(FunctionNames(kRuntimeBuiltinsMsl).size(), 4);
}

TEST_F(KernelsTest, SmallLinalgAndMpsStaging) {
  const std::vector<std::string> small = FunctionNames(kSmallLinalgMsl);
  EXPECT_EQ(std::set<std::string>(small.begin(), small.end()),
            (std::set<std::string>{"small_cholesky", "small_getrf",
                                   "small_trsm"}));
  for (const std::string& name : small) ExpectKernel(kSmallLinalgMsl, name);
  ExpectKernel(kMpsStagingMsl, "copy_f32");
}

// scan_<op>_<type>: 4 ops x (f32, f16, bf16, s32).
TEST_F(KernelsTest, Scan) {
  const std::vector<std::string> names = FunctionNames(kScanMsl);
  EXPECT_EQ(names.size(), 16);
  for (const char* op : {"add", "mul", "max", "min"}) {
    for (const char* t : {"float", "half", "bfloat", "int"}) {
      ExpectKernel(kScanMsl, absl::StrCat("scan_", op, "_", t));
    }
  }
}

// Every instantiation under every KIND / HAS_VALUES the handler passes; the
// kernels must run exactly 256 threads in 32-wide SIMD groups.
TEST_F(KernelsTest, CubSort) {
  const std::vector<std::string> names = FunctionNames(kCubSortMsl);
  // sort_small and sort_scatter per key x value width, sort_hist per key
  // width, one sort_scan.
  EXPECT_EQ(names.size(), 16 + 16 + 4 + 1);
  for (const std::string& name : names) {
    std::vector<std::vector<FunctionConstant>> variants;
    if (name == "sort_scan") {
      variants.push_back({});
    } else {
      for (int kind : {0, 1, 2}) {
        for (bool has_values : {false, true}) {
          variants.push_back({FunctionConstant::Int(0, kind),
                              FunctionConstant::Bool(1, has_values)});
        }
      }
    }
    for (const auto& constants : variants) {
      absl::StatusOr<const rt::Kernel*> k =
          dev_->GetKernel(kCubSortMsl, name, constants);
      ASSERT_THAT(k, IsOk()) << name;
      EXPECT_EQ((*k)->max_total_threads_per_threadgroup(), 256u) << name;
      EXPECT_EQ((*k)->thread_execution_width(), 32u) << name;
    }
  }
}

// Every steel variant RunSteelGemm can pick (ChooseSteelTile's tiles for
// every supported type pair and transpose), each with the switches all off
// and all on under every activation.
TEST_F(KernelsTest, SteelGemm) {
  using blas::MpsDType;
  const std::pair<MpsDType, MpsDType> types[] = {
      {MpsDType::kF32, MpsDType::kF32},   {MpsDType::kF16, MpsDType::kF16},
      {MpsDType::kF16, MpsDType::kF32},   {MpsDType::kBF16, MpsDType::kBF16},
      {MpsDType::kBF16, MpsDType::kF32}};
  std::set<std::string> seen;
  for (auto [in, out] : types) {
    for (bool ta : {false, true}) {
      for (bool tb : {false, true}) {
        for (int64_t size : {16, 64, 4096}) {  // few rows, few tiles, many
          blas::GemmParams p;
          p.m = p.n = p.k = size;
          p.batch_count = 1;
          p.a.dtype = p.b.dtype = in;
          p.c.dtype = out;
          p.a.transpose = ta;
          p.b.transpose = tb;
          const blas::SteelTile tile = blas::ChooseSteelTile(p);
          const blas::SteelKernelSource source =
              blas::SteelGemmKernel(in, out, tile, ta, tb);
          if (!seen.insert(source.function).second) continue;
          blas::SteelEpilogue off;
          ExpectKernel(source.msl.c_str(), source.function,
                       blas::SteelGemmConstants(false, false, false, off));
          int dummy;
          for (int act : {1, 2, 3}) {
            blas::SteelEpilogue on{act, &dummy, &dummy};
            ExpectKernel(source.msl.c_str(), source.function,
                         blas::SteelGemmConstants(true, true, true, on));
          }
        }
      }
    }
  }
  EXPECT_EQ(seen.size(), 5 * 4 * 2 + 4 * 4);  // f32 has no few-rows tile
}

// Every wide gemv variant (gemv.h: 2..kGemvMaxVectors vectors per pass,
// both K-lane counts, f16/bf16 to themselves and f32), switches off and on.
TEST_F(KernelsTest, Gemv) {
  using blas::MpsDType;
  const std::pair<MpsDType, MpsDType> types[] = {
      {MpsDType::kF16, MpsDType::kF16}, {MpsDType::kF16, MpsDType::kF32},
      {MpsDType::kBF16, MpsDType::kBF16}, {MpsDType::kBF16, MpsDType::kF32}};
  std::set<std::string> seen;
  for (auto [in, out] : types) {
    for (int vecs = 2; vecs <= blas::kGemvMaxVectors; ++vecs) {
      for (int64_t rows : {64, 4096}) {  // 32 and 16 K lanes past one pass
        blas::GemmParams p;
        p.m = vecs;
        p.n = rows;
        p.k = 64;
        p.a.dtype = p.b.dtype = in;
        p.c.dtype = out;
        p.a.ld = p.b.ld = 64;
        p.c.ld = rows;
        p.b.transpose = true;
        std::optional<blas::GemvPlan> plan = blas::ChooseGemv(p, 9);
        ASSERT_TRUE(plan.has_value());
        const blas::GemvKernelSource source =
            blas::GemvKernel(in, out, plan->vecs_per_tg, plan->k_lanes);
        if (!seen.insert(source.function).second) continue;
        ExpectKernel(source.msl.c_str(), source.function,
                     blas::GemvConstants(false, blas::SteelEpilogue{}));
        int dummy;
        for (int act : {1, 2, 3}) {
          blas::SteelEpilogue on{act, &dummy, &dummy};
          ExpectKernel(source.msl.c_str(), source.function,
                       blas::GemvConstants(true, on));
        }
      }
    }
  }
  // (vectors per pass, K lanes): (2..5, 32) and (3, 16), (4, 16).
  EXPECT_EQ(seen.size(), 4 * 6);
}

// Every convolution variant the dispatch can select (conv::AllConvKernels:
// the tile rules across their branch points, both ALIGN_C values of the
// general kernel), plus conv_misc's three kernels per type. The weight
// gradient's GEMMs are steel variants (SteelGemm above: T x T -> f32, A
// transposed).
TEST_F(KernelsTest, SteelConv) {
  EXPECT_EQ(FunctionNames(kConvMiscMsl).size(), 3 * 3);
  const std::vector<conv::ConvKernelSource> all = conv::AllConvKernels();
  // Per type: implicit c1..c4 on the 3 bm-32 tiles, small/large filter on
  // all 5 implicit tiles; general 3 tiles x ALIGN_C; unfold, pad, sum.
  EXPECT_EQ(all.size(), 3 * (3 * 4 + 5 * 2 + 3 * 2 + 3));
  for (const conv::ConvKernelSource& k : all) {
    ExpectKernel(k.msl.c_str(), k.function, k.constants);
  }
}

// Every FFT variant the dispatch can select (fft::AllFftKernels: the
// single-kernel plans of every length up to 4096 in the three I/O type
// pairs, the four-step passes of every power of two up to 2^24), each
// specialized for the first length that selects it, and fft_misc's kernels.
TEST_F(KernelsTest, Fft) {
  EXPECT_EQ(FunctionNames(kFftMiscMsl).size(), 11);
  const std::vector<fft::FftKernelSource> all = fft::AllFftKernels();
  // Stockham on 5 threadgroup memory sizes (256-4096), Rader on 4 (n <=
  // 2048), fused Bluestein on 5, each x 3 I/O pairs; four-step pass 0 on 4
  // sizes and pass 1 on 2 (256, 4096), each for c2c, rfft and irfft.
  EXPECT_EQ(all.size(), (5 + 4 + 5) * 3 + (4 + 2) * 3 + 11);
  for (const fft::FftKernelSource& k : all) {
    ExpectKernel(k.msl.c_str(), k.function, k.constants);
  }
}

}  // namespace
}  // namespace kernels
}  // namespace metal_pjrt
