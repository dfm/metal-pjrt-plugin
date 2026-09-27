// The radix sort handlers (xla.gpu.ext.cub_sort_keys/pairs, cub_sort_ffi.cc)
// against a CPU stable sort, bit for bit: every key type, keys and pairs
// (8/16/32/64-bit values), both directions, batched, with duplicates, +-0,
// NaNs and infinities. Invoked like XLA does: the instantiate stage for the
// scratch size (as EstimateCubSortScratchSize), then execute on a stream.
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "xla/ffi/attribute_map.h"
#include "xla/ffi/call_frame.h"
#include "xla/ffi/execution_state.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"
#include "xla/ffi/ffi_registry.h"
#include "xla/ffi/invoke.h"
#include "xla/primitive_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla_data.pb.h"

namespace metal_pjrt {
namespace ffi {
namespace {

namespace se = ::stream_executor;
namespace xffi = ::xla::ffi;
using xla::PrimitiveType;

constexpr char kKeys[] = "xla.gpu.ext.cub_sort_keys";
constexpr char kPairs[] = "xla.gpu.ext.cub_sort_pairs";

// CUB's radix order of key bits (see cub_sort_ffi.cc), as a uint64.
uint64_t OrderBits(uint64_t k, PrimitiveType t) {
  const int bits = xla::primitive_util::BitWidth(t);
  const uint64_t mask = bits == 64 ? ~uint64_t{0} : (uint64_t{1} << bits) - 1;
  const uint64_t sign = uint64_t{1} << (bits - 1);
  k &= mask;
  if (xla::primitive_util::IsSignedIntegralType(t)) return k ^ sign;
  if (xla::primitive_util::IsFloatingPointType(t)) {
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

// Random key bits with many duplicates; for floats also +-0, +-inf and NaNs
// of both signs with different payloads.
std::vector<uint8_t> MakeKeys(PrimitiveType t, int64_t total, uint32_t seed) {
  const int bytes = xla::primitive_util::ByteWidth(t);
  const int bits = bytes * 8;
  std::mt19937_64 rng(seed);
  std::vector<uint64_t> pool;
  for (int i = 0; i < 8; ++i) pool.push_back(rng());
  if (xla::primitive_util::IsFloatingPointType(t)) {
    int exp_bits = t == xla::F64 ? 11 : t == xla::F32 ? 8 : t == xla::F16 ? 5
                                                                         : 8;
    uint64_t sign = uint64_t{1} << (bits - 1);
    uint64_t exp = ((uint64_t{1} << exp_bits) - 1) << (bits - 1 - exp_bits);
    pool.insert(pool.end(), {0, sign, exp, exp | sign, exp | 1, exp | 3,
                             exp | sign | 1, exp | sign | 2});
  }
  std::vector<uint8_t> out(total * bytes);
  for (int64_t i = 0; i < total; ++i) {
    uint64_t v = rng() % 3 == 0 ? pool[rng() % pool.size()] : rng();
    std::memcpy(out.data() + i * bytes, &v, bytes);
  }
  return out;
}

struct Buffers {
  se::StreamExecutor* executor;
  std::vector<se::DeviceAddressBase> all;
  se::DeviceAddressBase Alloc(uint64_t bytes) {
    se::DeviceAddressBase b = executor->Allocate(std::max<uint64_t>(bytes, 1));
    all.push_back(b);
    return b;
  }
  ~Buffers() {
    for (auto& b : all) executor->Deallocate(&b);
  }
};

class CubSortTest : public ::testing::Test {
 protected:
  void SetUp() override {
    TF_ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                            se::PlatformManager::PlatformWithName("METAL"));
    TF_ASSERT_OK_AND_ASSIGN(executor_, platform->ExecutorForDevice(0));
    TF_ASSERT_OK_AND_ASSIGN(stream_, executor_->CreateStream());
  }

  xffi::CallFrame Frame(PrimitiveType kt, PrimitiveType vt, bool pairs,
                        int64_t batch, int64_t n, se::DeviceAddressBase kin,
                        se::DeviceAddressBase vin, se::DeviceAddressBase kout,
                        se::DeviceAddressBase vout,
                        se::DeviceAddressBase scratch, bool descending) {
    xffi::CallFrameBuilder b(pairs ? 2 : 1, pairs ? 3 : 2);
    b.AddBufferArg(kin, kt, {batch, n});
    if (pairs) b.AddBufferArg(vin, vt, {batch, n});
    b.AddBufferRet(kout, kt, {batch, n});
    if (pairs) b.AddBufferRet(vout, vt, {batch, n});
    b.AddBufferRet(scratch, xla::U8,
                   {static_cast<int64_t>(scratch.size())});
    xffi::CallFrameBuilder::AttributesBuilder attrs;
    attrs.Insert("descending", descending);
    attrs.Insert("batch_size", batch);
    b.AddAttributes(attrs.Build());
    return b.Build();
  }

  // Sorts and compares with the CPU reference. scratch_delta shrinks the
  // scratch buffer below the estimate (for the refusal test).
  absl::Status Run(PrimitiveType kt, PrimitiveType vt, bool descending,
                   int64_t batch, int64_t n, int64_t scratch_delta = 0) {
    const bool pairs = vt != xla::PRIMITIVE_TYPE_INVALID;
    const int kb = xla::primitive_util::ByteWidth(kt);
    const int vb = pairs ? xla::primitive_util::ByteWidth(vt) : 0;
    const int64_t total = batch * n;
    std::string what = absl::StrCat(
        xla::primitive_util::LowercasePrimitiveTypeName(kt), pairs ? "/" : "",
        pairs ? xla::primitive_util::LowercasePrimitiveTypeName(vt) : "",
        descending ? " desc " : " asc ", batch, "x", n);
    TF_ASSIGN_OR_RETURN(xffi::HandlerRegistration handler,
                        xffi::FindHandler(pairs ? kPairs : kKeys, "METAL"));

    std::vector<uint8_t> keys = MakeKeys(kt, total, 1234 + total);
    std::vector<uint8_t> vals(total * vb);
    for (int64_t i = 0; i < total; ++i) {
      uint64_t v = static_cast<uint64_t>(i) * 0x9E3779B97F4A7C15ull + i;
      if (pairs) std::memcpy(vals.data() + i * vb, &v, vb);
    }

    Buffers bufs{executor_};
    se::DeviceAddressBase kin = bufs.Alloc(total * kb);
    se::DeviceAddressBase kout = bufs.Alloc(total * kb);
    se::DeviceAddressBase vin = pairs ? bufs.Alloc(total * vb) : kin;
    se::DeviceAddressBase vout = pairs ? bufs.Alloc(total * vb) : kout;

    // Instantiate with a zero-sized scratch placeholder, as the compiler does.
    xffi::ExecutionState state;
    {
      se::DeviceAddressBase k(nullptr, total * kb), v(nullptr, total * vb);
      xffi::CallFrame frame =
          Frame(kt, vt, pairs, batch, n, k, v, k, v,
                se::DeviceAddressBase(nullptr, 0), descending);
      xffi::InvokeContext ctx{};
      ctx.state_context = {&state, nullptr, nullptr};
      TF_RETURN_IF_ERROR(xffi::Invoke(xffi::GetXlaFfiApi(),
                                      handler.bundle.instantiate, frame, ctx,
                                      XLA_FFI_ExecutionStage_INSTANTIATE));
    }
    TF_ASSIGN_OR_RETURN(int64_t* scratch_size, state.Get<int64_t>());
    const int64_t scratch_bytes = *scratch_size - scratch_delta;
    se::DeviceAddressBase scratch = bufs.Alloc(scratch_bytes);
    scratch = se::DeviceAddressBase(scratch.opaque(), scratch_bytes);

    if (total > 0) {
      TF_RETURN_IF_ERROR(stream_->Memcpy(&kin, keys.data(), total * kb));
      if (pairs) {
        TF_RETURN_IF_ERROR(stream_->Memcpy(&vin, vals.data(), total * vb));
      }
      // Poison the outputs so missing writes show.
      TF_RETURN_IF_ERROR(stream_->MemZero(&kout, total * kb));
      if (pairs) TF_RETURN_IF_ERROR(stream_->MemZero(&vout, total * vb));
    }
    xffi::CallFrame frame = Frame(kt, vt, pairs, batch, n, kin, vin, kout,
                                  vout, scratch, descending);
    xffi::InvokeContext ctx{};
    ctx.backend_context = xffi::InvokeContext::GpuContext{stream_.get()};
    absl::Status run = xffi::Invoke(xffi::GetXlaFfiApi(),
                                    handler.bundle.execute, frame, ctx);
    TF_RETURN_IF_ERROR(stream_->BlockHostUntilDone());
    TF_RETURN_IF_ERROR(run);

    std::vector<uint8_t> got_k(total * kb), got_v(total * vb);
    if (total > 0) {
      TF_RETURN_IF_ERROR(stream_->Memcpy(got_k.data(), kout, total * kb));
      if (pairs) {
        TF_RETURN_IF_ERROR(stream_->Memcpy(got_v.data(), vout, total * vb));
      }
      TF_RETURN_IF_ERROR(stream_->BlockHostUntilDone());
    }
    // The input must be untouched.
    std::vector<uint8_t> in_after(total * kb);
    if (total > 0) {
      TF_RETURN_IF_ERROR(stream_->Memcpy(in_after.data(), kin, total * kb));
      TF_RETURN_IF_ERROR(stream_->BlockHostUntilDone());
    }
    if (in_after != keys) return absl::InternalError(what + ": input written");

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

  se::StreamExecutor* executor_ = nullptr;
  std::unique_ptr<se::Stream> stream_;
};

const PrimitiveType kKeyTypes[] = {xla::U8,  xla::S8,  xla::U16, xla::S16,
                                   xla::F16, xla::BF16, xla::U32, xla::S32,
                                   xla::F32, xla::U64, xla::S64, xla::F64};
const PrimitiveType kValueTypes[] = {xla::PRIMITIVE_TYPE_INVALID, xla::U8,
                                     xla::S16, xla::U32, xla::U64};

// Rows around the one-threadgroup limit (2048) and the tile boundaries.
TEST_F(CubSortTest, Sweep) {
  const std::pair<int64_t, int64_t> shapes[] = {
      {1, 1}, {1, 17}, {3, 17}, {1, 2048}, {2, 2049},
      {1, 4095}, {3, 4097}, {5, 100}};
  for (PrimitiveType kt : kKeyTypes) {
    for (PrimitiveType vt : kValueTypes) {
      for (bool desc : {false, true}) {
        for (auto [batch, n] : shapes) {
          TF_ASSERT_OK(Run(kt, vt, desc, batch, n));
        }
      }
    }
  }
}

TEST_F(CubSortTest, Empty) {
  TF_ASSERT_OK(Run(xla::F32, xla::U32, false, 1, 0));
}

TEST_F(CubSortTest, Medium) {
  for (PrimitiveType kt : {xla::U32, xla::F32, xla::S64, xla::F16}) {
    TF_ASSERT_OK(Run(kt, xla::U64, false, 1, 65536));
    TF_ASSERT_OK(Run(kt, xla::PRIMITIVE_TYPE_INVALID, true, 4, 16411));
  }
}

// The shapes SortRewriter produces for jax: jnp.sort(f32) -> (U32 synthetic
// key, F32), argsort -> (U32, U64 packed); plus plain keys.
TEST_F(CubSortTest, OneMillion) {
  const int64_t n = 1 << 20;
  TF_ASSERT_OK(Run(xla::U32, xla::F32, false, 1, n));
  TF_ASSERT_OK(Run(xla::U32, xla::U64, true, 1, n));
  TF_ASSERT_OK(Run(xla::F32, xla::PRIMITIVE_TYPE_INVALID, false, 1, n));
  TF_ASSERT_OK(Run(xla::S64, xla::S32, false, 4, n / 4));
  TF_ASSERT_OK(Run(xla::S32, xla::S32, false, n / 4, 4));
}

// A scratch buffer smaller than the layout needs is refused before anything
// is encoded (the kernels would otherwise write past it).
TEST_F(CubSortTest, ShortScratchIsRefused) {
  absl::Status s = Run(xla::U32, xla::U32, false, 1, 10000,
                       /*scratch_delta=*/1);
  EXPECT_EQ(s.code(), absl::StatusCode::kInvalidArgument) << s;
  EXPECT_NE(s.message().find("scratch"), std::string::npos) << s;
  // The stream is still healthy.
  TF_ASSERT_OK(Run(xla::U32, xla::U32, false, 1, 10000));
}

}  // namespace
}  // namespace ffi
}  // namespace metal_pjrt
