// MPSMatrixMultiplication-backed GEMM. Objective-C++ with ARC; must not
// include metal-cpp headers (they clash with <Metal/Metal.h>), so Metal objects
// arrive as void* and are bridged here. absl is plain C++ and fine to use here.
#include "metal_pjrt_plugin/blas/mps_gemm.h"

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <sstream>
#include <string>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "metal_pjrt_plugin/runtime/system_memory.h"

#if !__has_feature(objc_arc)
#error "mps_gemm_objc.cc must be compiled with -fobjc-arc"
#endif

namespace metal_pjrt {
namespace blas {

int MpsDTypeSize(MpsDType t) {
  switch (t) {
    case MpsDType::kF32:
      return 4;
    case MpsDType::kF16:
    case MpsDType::kBF16:
      return 2;
  }
  return 0;
}

const char* MpsDTypeName(MpsDType t) {
  switch (t) {
    case MpsDType::kF32:
      return "f32";
    case MpsDType::kF16:
      return "f16";
    case MpsDType::kBF16:
      return "bf16";
  }
  return "?";
}

uint64_t GemmWork(const GemmParams& p) {
  const double b = static_cast<double>(p.batch_count);
  const double m = static_cast<double>(p.m), n = static_cast<double>(p.n),
               k = static_cast<double>(p.k);
  const double flops = 2.0 * b * m * n * k;
  const double bytes = b * (m * k * MpsDTypeSize(p.a.dtype) +
                            k * n * MpsDTypeSize(p.b.dtype) +
                            m * n * MpsDTypeSize(p.c.dtype));
  const double work = std::max(flops, 64.0 * bytes);
  return work >= 1.8e19 ? UINT64_MAX : static_cast<uint64_t>(work);
}

std::string GemmParamsDebugString(const GemmParams& p) {
  std::ostringstream os;
  auto op = [&os](const char* name, const MpsOperand& x) {
    os << " " << name << "{" << MpsDTypeName(x.dtype) << " off=" << x.offset
       << " ld=" << x.ld << " bs=" << x.batch_stride
       << (x.transpose ? " T" : " N") << "}";
  };
  os << "GemmParams{m=" << p.m << " n=" << p.n << " k=" << p.k
     << " batch=" << p.batch_count << " alpha=" << p.alpha
     << " beta=" << p.beta;
  op("a", p.a);
  op("b", p.b);
  op("c", p.c);
  os << "}";
  return os.str();
}

namespace {

// Stored (row-major) shape of an operand.
struct Stored {
  int64_t rows, cols;
};

std::string Describe(absl::string_view what, const GemmParams& p) {
  return absl::StrCat("RunMpsGemm: ", what, ": ", GemmParamsDebugString(p));
}
absl::Status Invalid(absl::string_view what, const GemmParams& p) {
  return absl::InvalidArgumentError(Describe(what, p));
}

// "<description> (domain=<domain>, code=<code>)".
std::string NSErrorToString(NSError* err) {
  if (err == nil) return "unknown Metal error (no NSError)";
  return absl::StrCat(err.localizedDescription.UTF8String, " (domain=",
                      err.domain.UTF8String, ", code=",
                      static_cast<int64_t>(err.code), ")");
}

Stored StoredA(const GemmParams& p) {
  return p.a.transpose ? Stored{p.k, p.m} : Stored{p.m, p.k};
}
Stored StoredB(const GemmParams& p) {
  return p.b.transpose ? Stored{p.n, p.k} : Stored{p.k, p.n};
}


// One batched MPS call when every operand's batches are laid out as whole,
// non-overlapping matrices; otherwise (broadcast operand with stride 0, or
// overlapping batches) one MPS call per batch.
bool UseBatched(const GemmParams& p) {
  if (p.batch_count <= 1) return false;
  auto fits = [](const MpsOperand& x, const Stored& s) {
    return x.batch_stride >= s.rows * x.ld;
  };
  return fits(p.a, StoredA(p)) && fits(p.b, StoredB(p)) &&
         fits(p.c, Stored{p.m, p.n});
}

// Elements MPS requires the buffer to hold past the operand's offset: MPS
// validates whole matrices (rows * rowBytes, and matrices * matrixBytes when
// batched), including the padding after the last row, which XLA's exactly
// sized buffers may not have.
int64_t RequiredElems(const MpsOperand& x, const Stored& s, int64_t batches,
                      bool batched) {
  if (batched) return batches * x.batch_stride;
  return (batches - 1) * x.batch_stride + s.rows * x.ld;
}

// ---------------------------------------------------------------------------
// Staging kernel: a copy for operands whose buffer is too short for MPS's
// whole-matrix validation. Each thread copies element (col, row, batch) at
// off + batch*bs + row*ld + col on both sides, so only the elements of the
// matrices are touched, never row padding. Buffers are bound at offset 0 and
// indexed with element offsets, so no binding-offset alignment applies.

struct StagingArgs {  // must match `Args` in kStagingMsl
  uint64_t in_off, out_off, ld, bs;
  uint32_t cols, rows, batches, pad;
};

constexpr const char* kStagingMsl = R"MSL(
#include <metal_stdlib>
using namespace metal;
struct Args { ulong in_off, out_off, ld, bs; uint cols, rows, batches, pad; };
kernel void copy_f32(device const uint* in [[buffer(0)]],
                     device uint* out [[buffer(1)]],
                     constant Args& a [[buffer(2)]],
                     uint3 g [[thread_position_in_grid]]) {
  if (g.x >= a.cols || g.y >= a.rows || g.z >= a.batches) return;
  ulong e = ulong(g.z) * a.bs + ulong(g.y) * a.ld + ulong(g.x);
  out[a.out_off + e] = in[a.in_off + e];
}
)MSL";

absl::StatusOr<id<MTLComputePipelineState>> GetStagingPipeline(
    id<MTLDevice> device) {
  static std::mutex mu;
  static NSMutableDictionary* cache = nil;  // registryID -> PSO
  std::lock_guard<std::mutex> lock(mu);
  if (cache == nil) cache = [NSMutableDictionary dictionary];
  NSNumber* key = @(device.registryID);
  id<MTLComputePipelineState> hit = cache[key];
  if (hit == nil) {
    NSError* err = nil;
    id<MTLLibrary> lib =
        [device newLibraryWithSource:@(kStagingMsl) options:nil error:&err];
    if (lib == nil) {
      return absl::InternalError(absl::StrCat(
          "RunMpsGemm: compiling the staging kernel failed: ",
          NSErrorToString(err)));
    }
    id<MTLFunction> fn = [lib newFunctionWithName:@"copy_f32"];
    if (fn == nil) {
      return absl::InternalError("RunMpsGemm: staging kernel not found");
    }
    hit = [device newComputePipelineStateWithFunction:fn error:&err];
    if (hit == nil) {
      return absl::InternalError(absl::StrCat(
          "RunMpsGemm: creating the staging pipeline failed: ",
          NSErrorToString(err)));
    }
    cache[key] = hit;
  }
  return hit;
}

// Copies the matrices of an operand between `in` and `out`, which share
// ld/bs (all quantities in elements).
absl::Status EncodeStaging(id<MTLCommandBuffer> cmd, id<MTLComputePipelineState> pso,
                   id<MTLBuffer> in, id<MTLBuffer> out, const StagingArgs& a) {
  id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
  if (enc == nil) {
    return absl::InternalError(
        "RunMpsGemm: computeCommandEncoder creation failed for staging");
  }
  [enc setComputePipelineState:pso];
  [enc setBuffer:in offset:0 atIndex:0];
  [enc setBuffer:out offset:0 atIndex:1];
  [enc setBytes:&a length:sizeof(a) atIndex:2];
  [enc dispatchThreads:MTLSizeMake(a.cols, a.rows, a.batches)
      threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
  [enc endEncoding];
  return absl::OkStatus();
}

absl::Status CheckParams(const GemmParams& p) {
  if (p.m < 0 || p.n < 0 || p.k < 0 || p.batch_count < 0) {
    return Invalid("negative dimension", p);
  }
  if (p.a.buffer == nullptr || p.b.buffer == nullptr ||
      p.c.buffer == nullptr) {
    return Invalid("null buffer", p);
  }
  if (p.c.transpose) return Invalid("transposed output", p);
  if (p.a.dtype != MpsDType::kF32 || p.b.dtype != MpsDType::kF32 ||
      p.c.dtype != MpsDType::kF32) {
    return absl::UnimplementedError(
        Describe("f32 only (f16/bf16 GEMMs run on steel)", p));
  }
  const Stored sa = StoredA(p), sb = StoredB(p);
  if (p.a.ld < sa.cols || p.b.ld < sb.cols || p.c.ld < p.n) {
    return Invalid("leading dimension smaller than stored columns", p);
  }
  if (p.a.batch_stride < 0 || p.b.batch_stride < 0 || p.c.batch_stride < 0) {
    return Invalid("negative batch stride", p);
  }
  for (const MpsOperand* x : {&p.a, &p.b, &p.c}) {
    if (x->offset % 4 != 0) return Invalid("misaligned operand offset", p);
  }
  return absl::OkStatus();
}

absl::Status EncodeMpsGemm(id<MTLDevice> device, id<MTLCommandBuffer> cmd,
                         const GemmParams& p, NSMutableArray* keep_alive) {
  const MPSDataType ta = MPSDataTypeFloat32, tc = MPSDataTypeFloat32;
  const Stored sa = StoredA(p), sb = StoredB(p), sc{p.m, p.n};
  const int64_t ea = 4, ec = 4;
  id<MTLBuffer> buf_a = (__bridge id<MTLBuffer>)p.a.buffer;
  id<MTLBuffer> buf_b = (__bridge id<MTLBuffer>)p.b.buffer;
  id<MTLBuffer> buf_c = (__bridge id<MTLBuffer>)p.c.buffer;

  const bool batched = UseBatched(p);
  const int64_t mats = batched ? p.batch_count : 1;

  auto desc = [&](const MpsOperand& x, const Stored& s, int64_t es,
                  MPSDataType t) {
    const NSUInteger row_bytes = static_cast<NSUInteger>(x.ld * es);
    const NSUInteger matrix_bytes =
        mats > 1 ? static_cast<NSUInteger>(x.batch_stride * es)
                 : static_cast<NSUInteger>(s.rows) * row_bytes;
    return [MPSMatrixDescriptor
        matrixDescriptorWithRows:static_cast<NSUInteger>(s.rows)
                         columns:static_cast<NSUInteger>(s.cols)
                        matrices:static_cast<NSUInteger>(mats)
                        rowBytes:row_bytes
                     matrixBytes:matrix_bytes
                        dataType:t];
  };
  MPSMatrixDescriptor* da = desc(p.a, sa, ea, ta);
  MPSMatrixDescriptor* db = desc(p.b, sb, ea, ta);
  MPSMatrixDescriptor* dc = desc(p.c, sc, ec, tc);

  MPSMatrixMultiplication* kernel = [[MPSMatrixMultiplication alloc]
      initWithDevice:device
       transposeLeft:p.a.transpose
      transposeRight:p.b.transpose
          resultRows:static_cast<NSUInteger>(p.m)
       resultColumns:static_cast<NSUInteger>(p.n)
     interiorColumns:static_cast<NSUInteger>(p.k)
               alpha:p.alpha
                beta:p.beta];
  if (kernel == nil) {
    return absl::InternalError(
        Describe("MPSMatrixMultiplication init failed", p));
  }

  const int64_t calls = batched ? 1 : p.batch_count;
  for (int64_t i = 0; i < calls; ++i) {
    auto at = [&](const MpsOperand& x, id<MTLBuffer> buf, int64_t es,
                  MPSMatrixDescriptor* d) {
      const NSUInteger off =
          static_cast<NSUInteger>(x.offset + i * x.batch_stride * es);
      return [[MPSMatrix alloc] initWithBuffer:buf offset:off descriptor:d];
    };
    MPSMatrix* ma = at(p.a, buf_a, ea, da);
    MPSMatrix* mb = at(p.b, buf_b, ea, db);
    MPSMatrix* mc = at(p.c, buf_c, ec, dc);
    if (ma == nil || mb == nil || mc == nil) {
      return absl::InternalError(
          Describe(absl::StrCat("MPSMatrix init failed for batch ", i), p));
    }
    if (batched) {
      kernel.batchStart = 0;
      kernel.batchSize = static_cast<NSUInteger>(p.batch_count);
    }
    [kernel encodeToCommandBuffer:cmd
                       leftMatrix:ma
                      rightMatrix:mb
                     resultMatrix:mc];
    [keep_alive addObject:ma];
    [keep_alive addObject:mb];
    [keep_alive addObject:mc];
  }
  [keep_alive addObject:kernel];
  return absl::OkStatus();
}

absl::Status RunMpsGemmImpl(void* mtl_device, void* mtl_command_buffer,
                            const GemmParams& p) {
  if (mtl_command_buffer == nullptr) {
    return Invalid("null command buffer", p);
  }
  ABSL_RETURN_IF_ERROR(CheckParams(p));
  if (p.m == 0 || p.n == 0 || p.batch_count == 0) return absl::OkStatus();
  if (p.k == 0) {
    // C = beta * C; MPS rejects interiorColumns == 0. XLA does not emit these.
    return absl::UnimplementedError(Describe("k == 0 is not supported", p));
  }

  id<MTLCommandBuffer> cmd =
      (__bridge id<MTLCommandBuffer>)mtl_command_buffer;
  id<MTLDevice> device = mtl_device != nullptr
                             ? (__bridge id<MTLDevice>)mtl_device
                             : cmd.device;
  if (!MPSSupportsMTLDevice(device)) {
    return absl::UnimplementedError(Describe(
        absl::StrCat("MPS does not support device ", device.name.UTF8String),
        p));
  }
  // Everything created here is kept alive until the GPU is done with it (the
  // command buffer retains the buffers it binds, not the MPS objects).
  // Registered up front so that objects referenced by commands encoded before
  // an error return are covered too; the array is filled in below.
  NSMutableArray* keep_alive = [NSMutableArray array];
  [cmd addCompletedHandler:^(id<MTLCommandBuffer>) {
    (void)keep_alive;
  }];

  // Stage an operand into a private temporary with the same ld/batch stride
  // when its buffer is too short for MPS's whole-matrix validation. Only
  // matrix elements are copied, in and (for C) back out, so row padding in
  // the caller's buffer is never written.
  const bool batched = UseBatched(p);
  const int64_t batches = p.batch_count;
  id<MTLComputePipelineState> pipe = nil;
  GemmParams q = p;
  struct Staged {
    bool active = false;
    id<MTLBuffer> tmp = nil;
  };
  auto stage = [&](MpsOperand& x, const Stored& st, bool load,
                   Staged* out) -> absl::Status {
    constexpr int64_t es = 4;
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)x.buffer;
    const int64_t required = RequiredElems(x, st, batches, batched);
    if (buf.length >= x.offset + static_cast<uint64_t>(required * es)) {
      return absl::OkStatus();
    }
    if (pipe == nil) {
      ABSL_ASSIGN_OR_RETURN(pipe, GetStagingPipeline(device));
    }
    const int64_t tmp_bytes = required * es;
    // Staging buffers come straight from the device, not the pool; they
    // still obey the allocation guard.
    uint64_t reclaimable = 0;
    if (!rt::FitsInSystemMemory(tmp_bytes, &reclaimable)) {
      return absl::ResourceExhaustedError(Describe(
          absl::StrFormat("a %d-byte staging buffer was refused: only %d "
                          "bytes of system memory are reclaimable",
                          tmp_bytes, reclaimable),
          p));
    }
    id<MTLBuffer> tmp =
        [device newBufferWithLength:static_cast<NSUInteger>(tmp_bytes)
                            options:MTLResourceStorageModePrivate];
    if (tmp == nil) {
      return absl::ResourceExhaustedError(Describe(
          absl::StrFormat("allocating a %d-byte staging buffer failed "
                          "(maxBufferLength %d, current allocated size %d)",
                          tmp_bytes, device.maxBufferLength,
                          device.currentAllocatedSize),
          p));
    }
    [keep_alive addObject:tmp];
    StagingArgs args{x.offset / es, 0, static_cast<uint64_t>(x.ld),
                     static_cast<uint64_t>(x.batch_stride),
                     static_cast<uint32_t>(st.cols),
                     static_cast<uint32_t>(st.rows),
                     static_cast<uint32_t>(batches), 0};
    if (load) {
      ABSL_RETURN_IF_ERROR(EncodeStaging(cmd, pipe, buf, tmp, args));
    }
    out->active = true;
    out->tmp = tmp;
    x.buffer = (__bridge void*)tmp;
    x.offset = 0;
    return absl::OkStatus();
  };
  Staged ga, gb, gc;
  const Stored sc{p.m, p.n};
  ABSL_RETURN_IF_ERROR(stage(q.a, StoredA(p), /*load=*/true, &ga));
  ABSL_RETURN_IF_ERROR(stage(q.b, StoredB(p), /*load=*/true, &gb));
  // A staged C is only read for beta * C; the write-back below touches just
  // the matrix elements, so skipping the load cannot clobber row padding.
  ABSL_RETURN_IF_ERROR(stage(q.c, sc, /*load=*/p.beta != 0.0, &gc));
  ABSL_RETURN_IF_ERROR(EncodeMpsGemm(device, cmd, q, keep_alive));
  if (gc.active) {
    StagingArgs args{0, p.c.offset / 4, static_cast<uint64_t>(p.c.ld),
                     static_cast<uint64_t>(p.c.batch_stride),
                     static_cast<uint32_t>(p.n), static_cast<uint32_t>(p.m),
                     static_cast<uint32_t>(batches), 0};
    ABSL_RETURN_IF_ERROR(EncodeStaging(
        cmd, pipe, gc.tmp, (__bridge id<MTLBuffer>)p.c.buffer, args));
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status RunMpsGemm(void* mtl_device, void* mtl_command_buffer,
                        const GemmParams& p) {
  // The descriptors, MPSMatrix objects and keep_alive array are autoreleased,
  // and XLA's execute threads never drain a pool: without this one they, and
  // the MTLBuffers the MPSMatrix objects retain, leaked for the life of the
  // thread (208 MB per call of a 100 MB matmul with the platform allocator).
  @autoreleasepool {
    return RunMpsGemmImpl(mtl_device, mtl_command_buffer, p);
  }
}

}  // namespace blas
}  // namespace metal_pjrt
