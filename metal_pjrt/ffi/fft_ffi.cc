// Copyright 2026 The metal-pjrt-plugin Authors
// SPDX-License-Identifier: Apache-2.0

// "metal$fft": 1-D FFTs over the minor dimension on MLX's kernels
// (metal_pjrt/fft/fft.h), the target of the fft lowering in
// metal_pjrt_plugin/_lowerings.py (XLA's FftThunk is cuFFT-only, and
// hlo_checks refuses the HLO fft op).
//
// Operand [..., InputLength(type, n)] and result [..., OutputLength(type,
// n)], row-major: complex64, except the operand of "rfft" and the result of
// "irfft" (float32). Second result: a u8 workspace of exactly
// FftWorkspaceBytes(plan, rows) bytes, which the Python rule computes the
// same way (so a drift fails here rather than going unnoticed).
// Attributes: fft_type ("fft" | "ifft" | "rfft" | "irfft"), n (i64, the
// transform length).
//
// The plan and the Rader / Bluestein constants (on the device) of a length
// are made at the FFI instantiate stage and shared by every call site of
// that length (FftTables, while one uses them): MakeFftConstants works in
// double and needs ~0.6 GB of transient host memory near n = 2^23, and the
// device copy is up to 160 MB. Execution only encodes the launches.
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "metal_pjrt/ffi/metal_ffi.h"
#include "metal_pjrt/fft/fft.h"
#include "metal_pjrt/fft/fft_plan.h"
#include "xla/backends/gpu/ffi.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"
#include "xla/primitive_util.h"

namespace metal_pjrt {
namespace ffi {
namespace {

namespace xffi = ::xla::ffi;

absl::StatusOr<fft::FftType> ParseFftType(absl::string_view s) {
  if (s == "fft") return fft::FftType::kFft;
  if (s == "ifft") return fft::FftType::kIfft;
  if (s == "rfft") return fft::FftType::kRfft;
  if (s == "irfft") return fft::FftType::kIrfft;
  return absl::InvalidArgumentError(
      absl::StrCat("metal$fft: unknown fft_type ", s));
}

// The plan and constants of one length on one device.
struct FftTables {
  ~FftTables() {
    if (constants_device != nullptr) {
      absl::Status s = device->Deallocate(constants_device);
      if (!s.ok()) LOG(ERROR) << "metal$fft: " << s;
    }
  }
  fft::FftPlan plan;
  fft::FftConstants constants;  // host bytes released after the upload
  rt::Device* device = nullptr;
  void* constants_device = nullptr;
};

absl::StatusOr<std::shared_ptr<const FftTables>> MakeTables(rt::Device* device,
                                                            int64_t n) {
  auto t = std::make_shared<FftTables>();
  ABSL_ASSIGN_OR_RETURN(t->plan, fft::PlanFft(n));
  t->device = device;
  t->constants = fft::MakeFftConstants(t->plan);
  if (t->constants.size > 0) {
    // Written on the host, off any stream (see Device::Use::kHostWrite).
    ABSL_ASSIGN_OR_RETURN(
        rt::Allocation a,
        device->Allocate(t->constants.size, rt::Device::Use::kHostWrite));
    t->constants_device = a.ptr;
    std::memcpy(a.ptr, t->constants.bytes.data(), t->constants.size);
    std::vector<uint8_t>().swap(t->constants.bytes);
  }
  return t;
}

// The tables of length n, made on first use and kept while a call site
// holds them.
absl::StatusOr<std::shared_ptr<const FftTables>> GetTables(rt::Device* device,
                                                           int64_t n) {
  static absl::Mutex mu(absl::kConstInit);
  static auto* cache =  // guarded by mu
      new absl::flat_hash_map<std::pair<rt::Device*, int64_t>,
                              std::weak_ptr<const FftTables>>();
  absl::MutexLock lock(&mu);
  std::weak_ptr<const FftTables>& entry = (*cache)[{device, n}];
  if (std::shared_ptr<const FftTables> t = entry.lock()) return t;
  ABSL_ASSIGN_OR_RETURN(std::shared_ptr<const FftTables> t,
                        MakeTables(device, n));
  entry = t;
  return t;
}

struct FftState {
  fft::FftType type;
  std::shared_ptr<const FftTables> tables;
};

// The rows of the call, after checking the buffers against the type and n.
absl::StatusOr<int64_t> Rows(const xffi::AnyBuffer& x,
                             const xffi::AnyBuffer& y, fft::FftType type,
                             int64_t n) {
  const bool real_in = type == fft::FftType::kRfft;
  const bool real_out = type == fft::FftType::kIrfft;
  const int64_t in_len = fft::InputLength(type, n);
  const int64_t out_len = fft::OutputLength(type, n);
  auto xd = x.dimensions(), yd = y.dimensions();
  const bool ok =
      !xd.empty() && xd.size() == yd.size() && xd.back() == in_len &&
      yd.back() == out_len &&
      std::equal(xd.begin(), xd.end() - 1, yd.begin()) &&
      x.element_type() == (real_in ? xla::F32 : xla::C64) &&
      y.element_type() == (real_out ? xla::F32 : xla::C64);
  if (!ok) {
    return absl::InvalidArgumentError(absl::StrCat(
        "metal$fft: operand/result shapes or types do not match ",
        fft::FftTypeName(type), " of length ", n));
  }
  return in_len == 0 ? 0 : x.element_count() / in_len;
}

absl::StatusOr<std::unique_ptr<FftState>> Instantiate(
    xffi::AnyBuffer x, xffi::Result<xffi::AnyBuffer> y,
    xffi::Result<xffi::BufferR1<xla::U8>> workspace,
    absl::string_view fft_type, int64_t n) {
  auto state = std::make_unique<FftState>();
  ABSL_ASSIGN_OR_RETURN(state->type, ParseFftType(fft_type));
  ABSL_RETURN_IF_ERROR(Rows(x, *y, state->type, n).status());
  ABSL_ASSIGN_OR_RETURN(rt::Device * device, DefaultMetalDevice());
  ABSL_ASSIGN_OR_RETURN(state->tables, GetTables(device, n));
  return state;
}

absl::Status Fft(stream_executor::Stream* stream, xffi::AnyBuffer x,
                 xffi::Result<xffi::AnyBuffer> y,
                 xffi::Result<xffi::BufferR1<xla::U8>> workspace,
                 absl::string_view fft_type, int64_t n, FftState* state) {
  ABSL_ASSIGN_OR_RETURN(const int64_t rows, Rows(x, *y, state->type, n));
  const FftTables& t = *state->tables;
  const uint64_t need = fft::FftWorkspaceBytes(t.plan, rows);
  if (workspace->size_bytes() != need) {
    return absl::InternalError(absl::StrCat(
        "metal$fft: workspace of ", workspace->size_bytes(),
        " bytes, the plan needs ", need,
        " (metal_pjrt_plugin/_lowerings.py and FftWorkspaceBytes disagree)"));
  }
  ABSL_ASSIGN_OR_RETURN(MetalContext ctx, GetMetalContext(stream));
  // The tables (and the constants' device allocation) belong to the device
  // they were made on at instantiation. The plugin has one device, and XLA
  // instantiates per executable per device, so a mismatch is a bug; a
  // re-lookup here would allocate and upload constants on the hot path.
  if (ctx.device != t.device) {
    return absl::InternalError("metal$fft: instantiated for another device");
  }
  return fft::RunFft(ctx.device, ctx.stream, t.plan, state->type, rows,
                     t.constants, t.constants_device, x.untyped_data(),
                     y->untyped_data(),
                     need > 0 ? workspace->untyped_data() : nullptr, need);
}

}  // namespace

XLA_FFI_DEFINE_HANDLER(kMetalFftInstantiate, Instantiate,
                       xffi::Ffi::BindInstantiate()
                           .Arg<xffi::AnyBuffer>()
                           .Ret<xffi::AnyBuffer>()
                           .Ret<xffi::BufferR1<xla::U8>>()  // workspace
                           .Attr<absl::string_view>("fft_type")
                           .Attr<int64_t>("n"));

XLA_FFI_DEFINE_HANDLER(kMetalFft, Fft,
                       xffi::Ffi::Bind()
                           .Ctx<xffi::Stream>()
                           .Arg<xffi::AnyBuffer>()
                           .Ret<xffi::AnyBuffer>()
                           .Ret<xffi::BufferR1<xla::U8>>()  // workspace
                           .Attr<absl::string_view>("fft_type")
                           .Attr<int64_t>("n")
                           .Ctx<xffi::State<FftState>>());

XLA_FFI_REGISTER_HANDLER(xffi::GetXlaFfiApi(), "metal$fft", "METAL",
                         {/*instantiate=*/kMetalFftInstantiate,
                          /*prepare=*/nullptr, /*initialize=*/nullptr,
                          /*execute=*/kMetalFft});

}  // namespace ffi
}  // namespace metal_pjrt
