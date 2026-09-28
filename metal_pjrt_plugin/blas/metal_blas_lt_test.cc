// BlasLt epilogues (bias / ReLU / GELU / SiLU, with and without aux output)
// through the Metal StreamExecutor, against a host reference. Needs a Metal
// device. f16/bf16 epilogues run fused in steel, f32 ones on MPS + the
// second-pass epilogue kernel.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "metal_pjrt_plugin/stream_executor/metal_platform.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/gpu_blas_lt.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace stream_executor {
namespace metal {
namespace {

using Epilogue = gpu::BlasLt::Epilogue;
using Order = gpu::MatrixLayout::Order;

// Host storage conversions.
uint16_t ToBf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  u += 0x7fffu + ((u >> 16) & 1u);
  return static_cast<uint16_t>(u >> 16);
}
float FromBf16(uint16_t h) {
  uint32_t u = static_cast<uint32_t>(h) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

int Size(xla::PrimitiveType t) { return t == xla::F32 ? 4 : 2; }

std::vector<uint8_t> Encode(const std::vector<float>& v, xla::PrimitiveType t) {
  std::vector<uint8_t> out(v.size() * Size(t));
  for (size_t i = 0; i < v.size(); ++i) {
    if (t == xla::F32) {
      std::memcpy(&out[4 * i], &v[i], 4);
    } else if (t == xla::F16) {
      _Float16 h = static_cast<_Float16>(v[i]);
      std::memcpy(&out[2 * i], &h, 2);
    } else {
      uint16_t h = ToBf16(v[i]);
      std::memcpy(&out[2 * i], &h, 2);
    }
  }
  return out;
}

std::vector<float> Decode(const std::vector<uint8_t>& b, xla::PrimitiveType t) {
  std::vector<float> out(b.size() / Size(t));
  for (size_t i = 0; i < out.size(); ++i) {
    if (t == xla::F32) {
      std::memcpy(&out[i], &b[4 * i], 4);
    } else if (t == xla::F16) {
      _Float16 h;
      std::memcpy(&h, &b[2 * i], 2);
      out[i] = static_cast<float>(h);
    } else {
      uint16_t h;
      std::memcpy(&h, &b[2 * i], 2);
      out[i] = FromBf16(h);
    }
  }
  return out;
}

float Round(float v, xla::PrimitiveType t) {
  return Decode(Encode({v}, t), t)[0];
}

struct Case {
  Epilogue epilogue;
  bool bias, aux;
  int act;  // 0 none, 1 relu, 2 gelu, 3 silu
};

float Act(int act, float x) {
  switch (act) {
    case 1:
      return x > 0 ? x : 0;
    case 2:
      return 0.5f * x *
             (1.0f + std::tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));
    case 3:
      return x / (1.0f + std::exp(-x));
    default:
      return x;
  }
}

struct Shape {
  int64_t m, n, k;
};

class MetalBlasLtEpilogueTest
    : public ::testing::TestWithParam<
          std::tuple<Case, xla::PrimitiveType, Order, Shape>> {};

TEST_P(MetalBlasLtEpilogueTest, MatchesReference) {
  const auto& [c, dtype, out_order, shape] = GetParam();
  TF_ASSERT_OK_AND_ASSIGN(Platform * platform,
                          PlatformManager::PlatformWithName("METAL"));
  TF_ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                          platform->ExecutorForDevice(0));
  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());
  ASSERT_NE(executor->AsBlas(), nullptr);
  gpu::BlasLt* lt = executor->AsBlas()->GetBlasLt();
  ASSERT_NE(lt, nullptr);

  const int64_t batch = 2, m = shape.m, n = shape.n, k = shape.k;
  // Row-major A (m x k), B (k x n) per batch; small values, exact products.
  std::vector<float> a(batch * m * k), b(batch * k * n);
  for (size_t i = 0; i < a.size(); ++i) a[i] = ((i * 7) % 9) * 0.25f - 1.0f;
  for (size_t i = 0; i < b.size(); ++i) b[i] = ((i * 5) % 11) * 0.25f - 1.25f;
  // Bias runs along the output's minor dimension: n entries for a row-major
  // output, m for a column-major one.
  const int64_t bias_len = out_order == Order::kRowMajor ? n : m;
  std::vector<float> bias(bias_len);
  for (int64_t i = 0; i < bias_len; ++i) bias[i] = 0.5f * i - 1.0f;

  const gpu::MatrixLayout out_layout(dtype, m, n, out_order, batch);
  gpu::GemmConfig cfg{
      gpu::MatrixLayout(dtype, m, k, Order::kRowMajor, batch),
      gpu::MatrixLayout(dtype, k, n, Order::kRowMajor, batch), out_layout,
      out_layout};
  cfg.alpha = 1.0;
  cfg.beta = 0.0;
  cfg.compute_precision = 0;
  cfg.precision_algorithm = xla::PrecisionConfig::ALG_UNSET;
  cfg.grad_x = cfg.grad_y = false;
  TF_ASSERT_OK_AND_ASSIGN(auto plan, lt->GetMatmulPlan(cfg, c.epilogue));
  TF_ASSERT_OK_AND_ASSIGN(auto algos, plan->GetAlgorithms(1, 0));
  TF_ASSERT_OK(plan->SetAlgorithm(algos[0]));

  auto upload = [&](const std::vector<float>& v) {
    std::vector<uint8_t> bytes = Encode(v, dtype);
    DeviceAddressBase mem = executor->Allocate(bytes.size());
    EXPECT_FALSE(mem.is_null());
    EXPECT_TRUE(stream->Memcpy(&mem, bytes.data(), bytes.size()).ok());
    // Host transfers are enqueued; keep `bytes` alive until done.
    EXPECT_TRUE(stream->BlockHostUntilDone().ok());
    return mem;
  };
  const uint64_t out_bytes = batch * m * n * Size(dtype);
  DeviceAddressBase da = upload(a), db = upload(b), dbias = upload(bias);
  DeviceAddressBase dd = executor->Allocate(out_bytes);
  DeviceAddressBase daux = executor->Allocate(out_bytes);
  const DeviceAddressBase none;
  gpu::BlasLt::MemoryArgs args{da,    db,   dd,   dd,   c.bias ? dbias : none,
                               c.aux ? daux : none,   none, none, none, none,
                               {none}, none, nullptr};
  TF_ASSERT_OK(plan->ExecuteOnStream(stream.get(), args, nullptr));
  std::vector<uint8_t> hd(out_bytes), haux(out_bytes);
  TF_ASSERT_OK(stream->Memcpy(hd.data(), dd, out_bytes));
  TF_ASSERT_OK(stream->Memcpy(haux.data(), daux, out_bytes));
  TF_ASSERT_OK(stream->BlockHostUntilDone());
  std::vector<float> got = Decode(hd, dtype), got_aux = Decode(haux, dtype);

  const float tol = dtype == xla::F32 ? 1e-5f : (dtype == xla::F16 ? 4e-3f : 2e-2f);
  for (int64_t bt = 0; bt < batch; ++bt) {
    for (int64_t i = 0; i < m; ++i) {
      for (int64_t j = 0; j < n; ++j) {
        float acc = 0;
        for (int64_t l = 0; l < k; ++l) {
          acc += Round(a[bt * m * k + i * k + l], dtype) *
                 Round(b[bt * k * n + l * n + j], dtype);
        }
        float pre = Round(acc, dtype);
        if (c.bias) {
          pre += Round(bias[out_order == Order::kRowMajor ? j : i], dtype);
        }
        const float want = Act(c.act, pre);
        const int64_t idx = bt * m * n + (out_order == Order::kRowMajor
                                              ? i * n + j
                                              : j * m + i);
        const std::string where = absl::StrCat("batch ", bt, " (", i, ",", j, ")");
        EXPECT_NEAR(got[idx], want, tol * (1 + std::fabs(want))) << where;
        if (c.aux) {
          EXPECT_NEAR(got_aux[idx], pre, tol * (1 + std::fabs(pre))) << where;
        }
      }
    }
  }
  for (DeviceAddressBase* mem : {&da, &db, &dbias, &dd, &daux}) {
    executor->Deallocate(mem);
  }
}

// f16/bf16 GEMMs run only on steel: a shape past its 32-bit index limits
// is refused when the plan is created (before any buffer or GPU work; the
// compiler refuses it earlier still, with the same ValidateMatmul), and f32
// is not limited.
TEST(MetalBlasLtTest, SteelOnlyShapeLimit) {
  TF_ASSERT_OK_AND_ASSIGN(Platform * platform,
                          PlatformManager::PlatformWithName("METAL"));
  TF_ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                          platform->ExecutorForDevice(0));
  gpu::BlasLt* lt = executor->AsBlas()->GetBlasLt();
  ASSERT_NE(lt, nullptr);
  const int64_t k = (int64_t{1} << 23) + 16;  // lda > INT32_MAX / 256
  for (xla::PrimitiveType t : {xla::BF16, xla::F16, xla::F32}) {
    const gpu::MatrixLayout out(t, 2, 2, Order::kRowMajor);
    gpu::GemmConfig cfg{gpu::MatrixLayout(t, 2, k, Order::kRowMajor),
                        gpu::MatrixLayout(t, k, 2, Order::kRowMajor), out,
                        out};
    cfg.alpha = 1.0;
    cfg.beta = 0.0;
    cfg.compute_precision = 0;
    cfg.precision_algorithm = xla::PrecisionConfig::ALG_UNSET;
    cfg.grad_x = cfg.grad_y = false;
    auto plan = lt->GetMatmulPlan(cfg, Epilogue::kDefault);
    if (t == xla::F32) {
      EXPECT_TRUE(plan.ok()) << plan.status();
    } else {
      EXPECT_EQ(plan.status().code(), absl::StatusCode::kUnimplemented);
      EXPECT_NE(plan.status().message().find("leading dimension"),
                std::string::npos)
          << plan.status();
    }
  }
}

const Case kCases[] = {
    {Epilogue::kDefault, false, false, 0},
    {Epilogue::kReLU, false, false, 1},
    {Epilogue::kBias, true, false, 0},
    {Epilogue::kBiasThenReLU, true, false, 1},
    {Epilogue::kGELU, false, false, 2},
    {Epilogue::kGELUWithAux, false, true, 2},
    {Epilogue::kBiasThenGELU, true, false, 2},
    {Epilogue::kBiasThenGELUWithAux, true, true, 2},
    {Epilogue::kSILU, false, false, 3},
    {Epilogue::kSILUWithAux, false, true, 3},
    {Epilogue::kBiasThenSILU, true, false, 3},
    {Epilogue::kBiasThenSILUWithAux, true, true, 3},
};

INSTANTIATE_TEST_SUITE_P(
    All, MetalBlasLtEpilogueTest,
    ::testing::Combine(::testing::ValuesIn(kCases),
                       ::testing::Values(xla::F32, xla::F16, xla::BF16),
                       ::testing::Values(Order::kRowMajor,
                                         Order::kColumnMajor),
                       // Edge tiles only (steel's store_result_safe), and
                       // whole tiles (store_result) plus a K remainder.
                       ::testing::Values(Shape{5, 7, 3}, Shape{64, 96, 36})),
    [](const auto& info) {
      const Case& c = std::get<0>(info.param);
      const Shape& s = std::get<3>(info.param);
      return absl::StrCat(
          "epilogue", static_cast<int>(c.epilogue), "_",
          xla::PrimitiveType_Name(std::get<1>(info.param)), "_",
          std::get<2>(info.param) == Order::kRowMajor ? "row" : "col", "_",
          s.m, "x", s.n, "x", s.k);
    });

}  // namespace
}  // namespace metal
}  // namespace stream_executor
