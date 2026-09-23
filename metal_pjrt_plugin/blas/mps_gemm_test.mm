// Standalone test for RunMpsGemm through a metal_pjrt::rt::Stream (no XLA, no
// absl, no gtest). Objective-C++ only because mps_gemm.mm is; this file does
// not itself use Objective-C.
//
// Build and run (from the repository root):
//   clang++ -std=c++17 -O1 -I . -I <metal-cpp> \
//     -framework Metal -framework Foundation -framework QuartzCore \
//     -framework MetalPerformanceShaders \
//     metal_pjrt_plugin/runtime/metal_cpp_impl.cc \
//     metal_pjrt_plugin/runtime/metal_runtime.cc \
//     -x objective-c++ -fobjc-arc metal_pjrt_plugin/blas/mps_gemm.mm \
//     metal_pjrt_plugin/blas/mps_gemm_test.mm -o /tmp/mps_gemm_test
#include "metal_pjrt_plugin/blas/mps_gemm.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "metal_pjrt_plugin/runtime/metal_runtime.h"

using metal_pjrt::rt::Allocation;
using metal_pjrt::rt::BufferRef;
using metal_pjrt::rt::Device;
using metal_pjrt::rt::Status;
using metal_pjrt::rt::Stream;
using namespace metal_pjrt::blas;

namespace {

uint16_t F32ToBf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  u += 0x7fff + ((u >> 16) & 1);  // round to nearest even
  return static_cast<uint16_t>(u >> 16);
}
float Bf16ToF32(uint16_t h) {
  uint32_t u = static_cast<uint32_t>(h) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
uint16_t F32ToF16(float f) {
  _Float16 h = static_cast<_Float16>(f);
  uint16_t u;
  std::memcpy(&u, &h, 2);
  return u;
}
float F16ToF32(uint16_t u) {
  _Float16 h;
  std::memcpy(&h, &u, 2);
  return static_cast<float>(h);
}

void Store(void* base, int64_t idx, MpsDType t, float v) {
  switch (t) {
    case MpsDType::kF32:
      static_cast<float*>(base)[idx] = v;
      break;
    case MpsDType::kF16:
      static_cast<uint16_t*>(base)[idx] = F32ToF16(v);
      break;
    case MpsDType::kBF16:
      static_cast<uint16_t*>(base)[idx] = F32ToBf16(v);
      break;
  }
}
float Load(const void* base, int64_t idx, MpsDType t) {
  switch (t) {
    case MpsDType::kF32:
      return static_cast<const float*>(base)[idx];
    case MpsDType::kF16:
      return F16ToF32(static_cast<const uint16_t*>(base)[idx]);
    case MpsDType::kBF16:
      return Bf16ToF32(static_cast<const uint16_t*>(base)[idx]);
  }
  return 0;
}

struct Case {
  std::string name;
  int64_t m, n, k, batch = 1;
  MpsDType in = MpsDType::kF32, out = MpsDType::kF32;
  bool ta = false, tb = false;
  double alpha = 1.0, beta = 0.0;
  int64_t pad = 0;             // extra elements per row (ld = cols + pad)
  bool broadcast_b = false;    // B batch stride 0
  bool column_major = false;   // describe in BLAS column-major convention
  uint64_t base_offset_elems = 0;  // operands live at an offset in a buffer
  bool tight = true;  // buffers end at the last element (no last-row padding),
                      // like XLA's exactly sized buffers
};

// Operand storage: `rows x cols` stored matrices (row-major, or column-major
// when `col` is set, in which case the stored matrix is cols x rows as seen
// row-major) with leading dim ld and batch stride.
struct HostOp {
  int64_t rows, cols, ld, stride, batches;
  MpsDType t;
  Allocation alloc;
  uint64_t off_bytes;
  void* data() const {
    return static_cast<char*>(alloc.ptr) + off_bytes;
  }
};

Status MakeOp(Device* dev, int64_t rows, int64_t cols, int64_t pad,
              int64_t batches, bool broadcast, MpsDType t, uint64_t off_elems,
              bool tight, HostOp* op) {
  op->rows = rows;
  op->cols = cols;
  op->ld = cols + pad;
  op->stride = broadcast ? 0 : rows * op->ld;
  op->batches = batches;
  op->t = t;
  const int64_t es = MpsDTypeSize(t);
  op->off_bytes = off_elems * es;
  const int64_t nb = broadcast ? 1 : batches;
  uint64_t elems = tight ? (nb - 1) * op->stride + (rows - 1) * op->ld + cols
                         : nb * rows * op->ld;
  uint64_t bytes = op->off_bytes + elems * es;
  return dev->Allocate(bytes, &op->alloc);
}

bool RunCase(Device* dev, Stream* stream, const Case& c) {
  // Logical row-major problem: C[b] (m x n) = alpha op(A) op(B) + beta C.
  // Stored shapes (row-major):
  const int64_t ar = c.ta ? c.k : c.m, ac = c.ta ? c.m : c.k;
  const int64_t br = c.tb ? c.n : c.k, bc = c.tb ? c.k : c.n;
  HostOp A, B, C;
  Status s = MakeOp(dev, ar, ac, c.pad, c.batch, false, c.in,
                    c.base_offset_elems, c.tight, &A);
  if (s.ok())
    s = MakeOp(dev, br, bc, c.pad, c.batch, c.broadcast_b, c.in,
               c.base_offset_elems, c.tight, &B);
  if (s.ok())
    s = MakeOp(dev, c.m, c.n, c.pad, c.batch, false, c.out,
               c.base_offset_elems, c.tight, &C);
  if (!s.ok()) {
    printf("FAIL %s: alloc: %s\n", c.name.c_str(), s.message().c_str());
    return false;
  }
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  auto fill = [&](HostOp& op) {
    int64_t nb = op.stride == 0 ? 1 : op.batches;
    int64_t n = (op.alloc.size - op.off_bytes) / MpsDTypeSize(op.t);
    for (int64_t i = 0; i < std::min(n, nb * op.rows * op.ld); ++i)
      Store(op.data(), i, op.t, dist(rng));
  };
  fill(A);
  fill(B);
  fill(C);
  // Reference, computed from the rounded input values.
  std::vector<double> ref(c.batch * c.m * c.n);
  for (int64_t b = 0; b < c.batch; ++b)
    for (int64_t i = 0; i < c.m; ++i)
      for (int64_t j = 0; j < c.n; ++j) {
        double acc = 0;
        for (int64_t l = 0; l < c.k; ++l) {
          int64_t ia = b * A.stride + (c.ta ? l * A.ld + i : i * A.ld + l);
          int64_t ib = b * B.stride + (c.tb ? j * B.ld + l : l * B.ld + j);
          acc += double(Load(A.data(), ia, A.t)) * Load(B.data(), ib, B.t);
        }
        double old = Load(C.data(), b * C.stride + i * C.ld + j, C.t);
        ref[(b * c.m + i) * c.n + j] = c.alpha * acc + c.beta * old;
      }

  // Snapshot of C to check that row padding is left untouched.
  std::vector<char> c_before(static_cast<char*>(C.alloc.ptr),
                             static_cast<char*>(C.alloc.ptr) + C.alloc.size);
  auto operand = [&](const HostOp& op, bool trans) {
    BufferRef r;
    Status st = dev->Resolve(op.data(), &r);
    if (!st.ok()) printf("resolve failed: %s\n", st.message().c_str());
    MpsOperand o;
    o.buffer = r.buffer;
    o.offset = r.offset;
    o.ld = op.ld;
    o.batch_stride = op.stride;
    o.dtype = op.t;
    o.transpose = trans;
    return o;
  };
  GemmParams p;
  p.m = c.m;
  p.n = c.n;
  p.k = c.k;
  p.batch_count = c.batch;
  p.alpha = c.alpha;
  p.beta = c.beta;
  p.a = operand(A, c.ta);
  p.b = operand(B, c.tb);
  p.c = operand(C, false);
  if (c.column_major) {
    // The same memory viewed column-major: C^T (n x m) = op(B)^T op(A)^T,
    // i.e. a BLAS call with (A:=B, B:=A, m:=n, n:=m) and unchanged flags.
    GemmParams q = p;
    std::swap(q.a, q.b);
    std::swap(q.m, q.n);
    ColumnMajorToRowMajor(&q);  // must undo the swap
    p = q;
  }

  s = stream->EncodeExternal([&](void* cmd) {
    return RunMpsGemm(nullptr, cmd, p);
  });
  if (s.ok()) s = stream->Synchronize();
  if (!s.ok()) {
    printf("FAIL %s: %s\n", c.name.c_str(), s.message().c_str());
    return false;
  }
  double tol = (c.in == MpsDType::kF32 && c.out == MpsDType::kF32) ? 1e-4
               : (c.out == MpsDType::kBF16 || c.in == MpsDType::kBF16)
                   ? 3e-2 * std::sqrt(double(c.k))
                   : 5e-3 * std::sqrt(double(c.k));
  double max_err = 0;
  for (int64_t b = 0; b < c.batch; ++b)
    for (int64_t i = 0; i < c.m; ++i)
      for (int64_t j = 0; j < c.n; ++j) {
        double got = Load(C.data(), b * C.stride + i * C.ld + j, C.t);
        max_err = std::max(max_err, std::fabs(got - ref[(b * c.m + i) * c.n + j]));
      }
  bool padding_ok = true;
  {
    const int64_t es = MpsDTypeSize(C.t);
    const char* now = static_cast<const char*>(C.data());
    const char* was = c_before.data() + C.off_bytes;
    for (int64_t b = 0; b < c.batch; ++b)
      for (int64_t i = 0; i < c.m; ++i)
        for (int64_t j = c.n; j < C.ld; ++j) {
          int64_t e = b * C.stride + i * C.ld + j;
          if ((e + 1) * es > int64_t(C.alloc.size - C.off_bytes)) continue;
          if (std::memcmp(now + e * es, was + e * es, es) != 0)
            padding_ok = false;
        }
    if (std::memcmp(C.alloc.ptr, c_before.data(), C.off_bytes) != 0)
      padding_ok = false;
  }
  dev->Deallocate(A.alloc.ptr);
  dev->Deallocate(B.alloc.ptr);
  dev->Deallocate(C.alloc.ptr);
  bool ok = max_err <= tol && padding_ok;
  printf("%s %-40s max_err=%.3g tol=%.3g%s\n", ok ? "ok  " : "FAIL",
         c.name.c_str(), max_err, tol, padding_ok ? "" : " (padding clobbered)");
  return ok;
}

}  // namespace

int main() {
  std::unique_ptr<Device> dev;
  Status s = Device::Create(0, &dev);
  if (!s.ok()) {
    printf("no device: %s\n", s.message().c_str());
    return 1;
  }
  std::unique_ptr<Stream> stream;
  s = dev->CreateStream(&stream);
  if (!s.ok()) {
    printf("no stream: %s\n", s.message().c_str());
    return 1;
  }
  std::vector<Case> cases;
  auto add = [&](Case c) { cases.push_back(c); };
  for (bool ta : {false, true})
    for (bool tb : {false, true}) {
      Case c{"f32 " + std::string(ta ? "T" : "N") + (tb ? "T" : "N"), 37, 23,
             19};
      c.ta = ta;
      c.tb = tb;
      add(c);
    }
  { Case c{"f32 square 128", 128, 128, 128}; add(c); }
  { Case c{"f32 alpha/beta", 17, 9, 33}; c.alpha = 0.5; c.beta = -1.25; add(c); }
  { Case c{"f32 padded ld TN", 12, 10, 7}; c.pad = 3; c.ta = true; add(c); }
  { Case c{"f32 batched", 8, 6, 5}; c.batch = 4; c.tb = true; add(c); }
  { Case c{"f32 batched padded", 8, 6, 5}; c.batch = 3; c.pad = 2; add(c); }
  { Case c{"f32 batched padded (roomy buffers)", 8, 6, 5}; c.batch = 3; c.pad = 2; c.tight = false; add(c); }
  { Case c{"f32 padded offset TT beta", 9, 7, 4}; c.pad = 1; c.ta = c.tb = true; c.beta = 2.0; c.base_offset_elems = 5; add(c); }
  { Case c{"f32 broadcast B", 8, 6, 5}; c.batch = 3; c.broadcast_b = true; add(c); }
  { Case c{"f32 column-major NT", 9, 14, 11}; c.column_major = true; c.tb = true; add(c); }
  { Case c{"f32 buffer offset", 5, 7, 3}; c.base_offset_elems = 3; add(c); }
  { Case c{"f32 k=1", 6, 4, 1}; add(c); }
  { Case c{"f32 m=n=1", 1, 1, 64}; add(c); }
  for (MpsDType in : {MpsDType::kF16, MpsDType::kBF16}) {
    std::string nm = in == MpsDType::kF16 ? "f16" : "bf16";
    for (bool ta : {false, true}) {
      Case c{nm + (ta ? " TN" : " NN"), 33, 17, 9};
      c.in = c.out = in;
      c.ta = ta;
      add(c);
    }
    { Case c{nm + " odd k (rowBytes%4!=0)", 7, 5, 3}; c.in = c.out = in; add(c); }
    { Case c{nm + " ->f32", 20, 12, 16}; c.in = in; c.out = MpsDType::kF32; add(c); }
    { Case c{nm + " batched beta", 10, 8, 6}; c.in = c.out = in; c.batch = 2; c.beta = 1.0; add(c); }
    { Case c{nm + " odd offset", 5, 3, 3}; c.in = c.out = in; c.base_offset_elems = 1; add(c); }
    { Case c{nm + " padded odd offset", 5, 3, 3}; c.in = c.out = in; c.pad = 1; c.base_offset_elems = 1; c.beta = 1.0; add(c); }
    { Case c{nm + " padded broadcast TT", 6, 5, 4}; c.in = c.out = in; c.batch = 3; c.pad = 1; c.broadcast_b = true; c.ta = c.tb = true; c.alpha = 2.0; c.beta = 0.5; add(c); }
    { Case c{nm + " column-major 64", 64, 48, 80}; c.in = c.out = in; c.column_major = true; add(c); }
  }
  int failures = 0;
  for (const Case& c : cases) failures += RunCase(dev.get(), stream.get(), c) ? 0 : 1;

  // Ordering with other stream work and command-buffer rollover: 150 GEMMs
  // chained C <- A*C (identity A) plus 1 each time via beta and memset.
  {
    const int64_t n = 16;
    Allocation a, cbuf;
    dev->Allocate(n * n * 4, &a);
    dev->Allocate(n * n * 4, &cbuf);
    float* ap = static_cast<float*>(a.ptr);
    for (int64_t i = 0; i < n * n; ++i) ap[i] = (i / n == i % n) ? 1.f : 0.f;
    std::memset(cbuf.ptr, 0, n * n * 4);
    BufferRef ra, rc;
    dev->Resolve(a.ptr, &ra);
    dev->Resolve(cbuf.ptr, &rc);
    GemmParams p;
    p.m = p.n = p.k = n;
    p.alpha = 1.0;
    p.beta = 1.0;  // C = I*C + C = 2C
    p.a = {ra.buffer, ra.offset, n, 0, MpsDType::kF32, false};
    p.c = {rc.buffer, rc.offset, n, 0, MpsDType::kF32, false};
    p.b = p.c;
    Status st;
    // Fill with 1.0f via host-ordered path, then 10 doublings, repeatedly.
    float expect = 0;
    for (int rep = 0; rep < 15 && st.ok(); ++rep) {
      st = stream->Memset32(cbuf.ptr, 0x3f800000u, n * n * 4);  // 1.0f
      for (int i = 0; i < 10 && st.ok(); ++i)
        st = stream->EncodeExternal(
            [&](void* cmd) { return RunMpsGemm(nullptr, cmd, p); });
      expect = 1024.f;
    }
    if (st.ok()) st = stream->Synchronize();
    bool ok = st.ok();
    for (int64_t i = 0; ok && i < n * n; ++i)
      ok = static_cast<float*>(cbuf.ptr)[i] == expect;
    printf("%s %-40s %s\n", ok ? "ok  " : "FAIL", "ordering + 150 ops rollover",
           st.message().c_str());
    failures += ok ? 0 : 1;
    dev->Deallocate(a.ptr);
    dev->Deallocate(cbuf.ptr);
  }
  printf("%d failures\n", failures);
  return failures == 0 ? 0 : 1;
}
