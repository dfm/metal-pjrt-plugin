#include "metal_pjrt_plugin/codegen/msl_emitter.h"

#include <memory>
#include <optional>
#include <regex>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "metal_pjrt_plugin/codegen/msl_llvm_bridge.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "xla/stream_executor/device_description.h"

namespace metal_pjrt::codegen {
namespace {

using ::testing::HasSubstr;
using ::testing::Not;

class MslEmitterTest : public ::testing::Test {
 protected:
  MslEmitterTest() {
    mlir::DialectRegistry registry;
    registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                    mlir::gpu::GPUDialect, mlir::LLVM::LLVMDialect,
                    mlir::math::MathDialect, mlir::scf::SCFDialect,
                    mlir::ub::UBDialect, mlir::vector::VectorDialect>();
    context_.appendDialectRegistry(registry);
    context_.loadAllAvailableDialects();
  }

  absl::StatusOr<MslKernel> Emit(const char* ir, const char* entry) {
    mlir::OwningOpRef<mlir::ModuleOp> module =
        mlir::parseSourceString<mlir::ModuleOp>(ir, &context_);
    if (!module) return absl::InvalidArgumentError("failed to parse test IR");
    stream_executor::DeviceDescription device;
    return EmitMslKernel(*module, entry, device);
  }

  mlir::MLIRContext context_;
};

// Elementwise loop fusion: scf.for + llvm.load/store + arith/math.
constexpr char kElementwise[] = R"mlir(
module {
  func.func @fusion(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c128 = arith.constant 128 : index
    %c1024 = arith.constant 1024 : index
    %tid = gpu.thread_id x
    %bid = gpu.block_id x
    %base = arith.muli %bid, %c128 : index
    %i0 = arith.addi %base, %tid : index
    scf.for %j = %c0 to %c4 step %c1 {
      %off = arith.muli %j, %c1024 : index
      %i = arith.addi %i0, %off : index
      %ii = arith.index_castui %i : index to i32
      %p0 = llvm.getelementptr inbounds %arg0[0, %ii] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.array<4096 x f32>
      %p1 = llvm.getelementptr inbounds %arg1[0, %ii] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.array<4096 x f32>
      %a = llvm.load %p0 : !llvm.ptr -> f32
      %b = llvm.load %p1 : !llvm.ptr -> f32
      %e = math.exp %a : f32
      %m = arith.mulf %e, %b : f32
      %t = math.tanh %m : f32
      %l = math.log1p %t : f32
      %mx = arith.maximumf %l, %a : f32
      %p2 = llvm.getelementptr inbounds %arg2[0, %ii] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.array<4096 x f32>
      llvm.store %mx, %p2 : f32, !llvm.ptr
    }
    return
  }
})mlir";

TEST_F(MslEmitterTest, ElementwiseLoop) {
  absl::StatusOr<MslKernel> kernel = Emit(kElementwise, "fusion");
  ASSERT_TRUE(kernel.ok()) << kernel.status();
  EXPECT_EQ(kernel->kernel_name, "fusion");
  EXPECT_EQ(kernel->num_buffer_args, 3);
  const std::string& msl = kernel->msl_source;
  EXPECT_THAT(msl, HasSubstr("#include <metal_stdlib>"));
  EXPECT_THAT(msl, HasSubstr("kernel void fusion("));
  EXPECT_THAT(msl, HasSubstr("device char* xla_arg0 [[buffer(0)]]"));
  EXPECT_THAT(msl, HasSubstr("device char* xla_arg2 [[buffer(2)]]"));
  EXPECT_THAT(msl, Not(HasSubstr("[[buffer(3)]]")));
  EXPECT_THAT(msl, HasSubstr("[[thread_position_in_threadgroup]]"));
  EXPECT_THAT(msl, HasSubstr("[[threadgroup_position_in_grid]]"));
  EXPECT_THAT(msl, HasSubstr("fusion_impl("));
  EXPECT_THAT(msl, HasSubstr("for ("));
  EXPECT_THAT(msl, HasSubstr("xla_gep("));
  EXPECT_THAT(msl, HasSubstr("xla_load<float>("));
  EXPECT_THAT(msl, HasSubstr("xla_store("));
  EXPECT_THAT(msl, HasSubstr("exp("));
  EXPECT_THAT(msl, HasSubstr("tanh("));
  EXPECT_THAT(msl, HasSubstr("xla_log1p("));
}

constexpr char kRangedIds[] = R"mlir(
module {
  func.func @ranged(%arg0: !llvm.ptr) {
    %tx = gpu.thread_id x {xla.range = [0 : index, 127 : index]}
    %ty = gpu.thread_id y {xla.range = [0 : index, 1 : index]}
    %tz = gpu.thread_id z {xla.range = [0 : index, 0 : index]}
    %s = arith.addi %tx, %ty : index
    %t = arith.addi %s, %tz : index
    %i = arith.index_cast %t : index to i32
    %p = llvm.getelementptr %arg0[%i] : (!llvm.ptr, i32) -> !llvm.ptr, i32
    llvm.store %i, %p : i32, !llvm.ptr
    return
  }
})mlir";

TEST_F(MslEmitterTest, MaxTotalThreadsFromThreadIdRanges) {
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kRangedIds, &context_);
  ASSERT_TRUE(module);
  EXPECT_EQ(ThreadsPerThreadgroupFromRanges(*module, "ranged"), 256);
  EXPECT_EQ(ThreadsPerThreadgroupFromRanges(*module, "other"), 0);
  stream_executor::DeviceDescription device;
  absl::StatusOr<MslKernel> kernel =
      EmitMslKernel(*module, "ranged", device, 256);
  ASSERT_TRUE(kernel.ok()) << kernel.status();
  EXPECT_THAT(kernel->msl_source,
              HasSubstr("[[max_total_threads_per_threadgroup(256)]]\n"
                        "kernel void ranged("));

  // A dimension without a ranged id (kElementwise has only an unranged x):
  // no attribute.
  mlir::OwningOpRef<mlir::ModuleOp> plain =
      mlir::parseSourceString<mlir::ModuleOp>(kElementwise, &context_);
  ASSERT_TRUE(plain);
  EXPECT_EQ(ThreadsPerThreadgroupFromRanges(*plain, "fusion"), 0);
  absl::StatusOr<MslKernel> k2 = Emit(kElementwise, "fusion");
  ASSERT_TRUE(k2.ok());
  EXPECT_THAT(k2->msl_source,
              Not(HasSubstr("max_total_threads_per_threadgroup")));
}

// Reduction: shuffles, shared memory, barrier, atomics.
constexpr char kReduction[] = R"mlir(
module {
  llvm.mlir.global private @shared_0() {addr_space = 3 : i32} : !llvm.array<32 x f32>
  func.func @reduce(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
    %tid = gpu.thread_id x
    %tid_i = arith.index_castui %tid : index to i32
    %p = llvm.getelementptr inbounds %arg0[0, %tid_i] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.array<1024 x f32>
    %v = llvm.load %p : !llvm.ptr -> f32
    %c1 = arith.constant 1 : i32
    %c16 = arith.constant 16 : i32
    %c31 = arith.constant 31 : i32
    %c32 = arith.constant 32 : i32
    %s, %valid = gpu.shuffle down %v, %c16, %c32 : f32
    %sum = arith.addf %v, %s : f32
    %x, %valid2 = gpu.shuffle xor %sum, %c1, %c32 : f32
    %sum2 = arith.addf %sum, %x : f32
    %sh = llvm.mlir.addressof @shared_0 : !llvm.ptr<3>
    %shg = llvm.addrspacecast %sh : !llvm.ptr<3> to !llvm.ptr
    %lane = arith.andi %tid_i, %c31 : i32
    %sp = llvm.getelementptr inbounds %shg[0, %lane] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.array<32 x f32>
    llvm.store %sum2, %sp : f32, !llvm.ptr
    gpu.barrier
    %r = llvm.load %sp : !llvm.ptr -> f32
    %op = llvm.getelementptr inbounds %arg1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x f32>
    %old = llvm.atomicrmw fadd %op, %r monotonic : !llvm.ptr, f32
    %iv = llvm.bitcast %r : f32 to i32
    %cas = llvm.cmpxchg %op, %iv, %iv monotonic monotonic : !llvm.ptr, i32
    %cv = llvm.extractvalue %cas[0] : !llvm.struct<(i32, i1)>
    %ok = llvm.extractvalue %cas[1] : !llvm.struct<(i32, i1)>
    %sel = arith.select %ok, %cv, %iv : i32
    llvm.store %sel, %op : i32, !llvm.ptr
    return
  }
})mlir";

TEST_F(MslEmitterTest, ReductionWithShuffleAndSharedMemory) {
  absl::StatusOr<MslKernel> kernel = Emit(kReduction, "reduce");
  ASSERT_TRUE(kernel.ok()) << kernel.status();
  EXPECT_EQ(kernel->num_buffer_args, 2);
  const std::string& msl = kernel->msl_source;
  EXPECT_THAT(msl, HasSubstr("threadgroup uint4 xla_shared_0[8];"));
  EXPECT_THAT(msl, HasSubstr("(threadgroup char*)xla_shared_0"));
  EXPECT_THAT(msl, HasSubstr("threadgroup char*"));
  EXPECT_THAT(msl, HasSubstr("xla_shfl_down("));
  EXPECT_THAT(msl, HasSubstr("xla_shfl_xor("));
  EXPECT_THAT(msl, HasSubstr("xla_barrier();"));
  EXPECT_THAT(msl, HasSubstr("xla_atomic_fadd("));
  EXPECT_THAT(msl, HasSubstr("as_type<int32_t>("));
  EXPECT_THAT(msl, HasSubstr("xla_cmpxchg("));
  EXPECT_THAT(msl, HasSubstr("xla_cas_value("));
  EXPECT_THAT(msl, HasSubstr("xla_cas_ok("));
}

// Bounds check with scf.if, f16 data and a private helper function.
constexpr char kBoundsCheck[] = R"mlir(
module {
  func.func private @add(%a: f16, %b: f16) -> f16 {
    %r = arith.addf %a, %b : f16
    return %r : f16
  }
  func.func @bounded(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
    %tid = gpu.thread_id x
    %bid = gpu.block_id x
    %bdim = gpu.block_dim x
    %m = arith.muli %bid, %bdim : index
    %i = arith.addi %m, %tid : index
    %n = arith.constant 1000 : index
    %inb = arith.cmpi ult, %i, %n : index
    scf.if %inb {
      %ii = arith.index_castui %i : index to i32
      %p = llvm.getelementptr inbounds %arg0[0, %ii] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.array<1000 x f16>
      %v = llvm.load %p : !llvm.ptr -> f16
      %one = arith.constant 1.5 : f16
      %s = func.call @add(%v, %one) : (f16, f16) -> f16
      %q = llvm.getelementptr inbounds %arg1[0, %ii] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.array<1000 x f16>
      llvm.store %s, %q : f16, !llvm.ptr
    }
    return
  }
})mlir";

TEST_F(MslEmitterTest, BoundsCheckWithHelperFunction) {
  absl::StatusOr<MslKernel> kernel = Emit(kBoundsCheck, "bounded");
  ASSERT_TRUE(kernel.ok()) << kernel.status();
  const std::string& msl = kernel->msl_source;
  EXPECT_THAT(msl, HasSubstr("if ("));
  EXPECT_THAT(msl, HasSubstr("inline _Float16 bounded_fn_add("));
  EXPECT_THAT(msl, HasSubstr("bounded_fn_add("));
  EXPECT_THAT(msl, HasSubstr("xla_load<_Float16>("));
  EXPECT_THAT(msl, HasSubstr("(_Float16)"));
  EXPECT_THAT(msl, Not(HasSubstr("f16;")));  // no C++23 literal suffixes
  EXPECT_THAT(msl, Not(HasSubstr("static ")));
  // The helper must be defined before the kernel body that calls it.
  EXPECT_LT(msl.find("bounded_fn_add("), msl.find("bounded_impl("));
}

// Compare-and-swap loop as XLA emits it for float scatter min/max: an
// scf.while whose after region is a bare forwarding yield. The loop-carried
// "expected" value must be refreshed from the CAS result on every retry.
constexpr char kCasLoop[] = R"mlir(
module {
  func.func @casmin(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
    %tid = gpu.thread_id x
    %ii = arith.index_castui %tid : index to i32
    %p = llvm.getelementptr inbounds %arg0[0, %ii] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.array<1024 x i32>
    %v = llvm.load %arg1 : !llvm.ptr -> i32
    %old = llvm.load %p : !llvm.ptr -> i32
    %r = scf.while (%cur = %old) : (i32) -> i32 {
      %new = arith.minsi %cur, %v : i32
      %pair = llvm.cmpxchg %p, %cur, %new acq_rel monotonic : !llvm.ptr, i32
      %val = llvm.extractvalue %pair[0] : !llvm.struct<(i32, i1)>
      %ok = llvm.extractvalue %pair[1] : !llvm.struct<(i32, i1)>
      %true = arith.constant true
      %retry = arith.xori %ok, %true : i1
      scf.condition(%retry) %val : i32
    } do {
    ^bb0(%x: i32):
      scf.yield %x : i32
    }
    return
  }
})mlir";

TEST_F(MslEmitterTest, CasLoopRefreshesExpectedValue) {
  absl::StatusOr<MslKernel> kernel = Emit(kCasLoop, "casmin");
  ASSERT_TRUE(kernel.ok()) << kernel.status();
  const std::string& msl = kernel->msl_source;
  size_t do_pos = msl.find("do {");
  ASSERT_NE(do_pos, std::string::npos) << msl;
  size_t end_pos = msl.find("} while (", do_pos);
  ASSERT_NE(end_pos, std::string::npos) << msl;
  std::string body = msl.substr(do_pos, end_pos - do_pos);
  // xla_cmpxchg(ptr, expected, desired): find the expected value's name and
  // the loop variable it was loaded from.
  std::smatch m;
  ASSERT_TRUE(std::regex_search(body, m,
                                std::regex(R"(xla_cmpxchg\(v\d+, (v\d+), )")))
      << body;
  std::string expected = m[1];
  ASSERT_TRUE(std::regex_search(
      body, m, std::regex("int32_t " + expected + R"( = (v\d+);)")))
      << body;
  std::string loop_var = m[1];
  // The loop variable must be written back inside the loop body.
  EXPECT_TRUE(std::regex_search(body, std::regex(loop_var + R"( = v\d+;)")))
      << "loop variable " << loop_var << " never updated:\n"
      << body;
}

// Vectorized loads/stores (VectorizeLoadsAndStores + LowerTensors output).
constexpr char kVectorized[] = R"mlir(
module {
  func.func @vec(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
    %tid = gpu.thread_id x
    %ii = arith.index_castui %tid : index to i32
    %c4 = arith.constant 4 : i32
    %base = arith.muli %ii, %c4 : i32
    %p = llvm.getelementptr inbounds %arg0[0, %base] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.array<1024 x f32>
    %v = llvm.load %p : !llvm.ptr -> vector<4xf32>
    %e0 = vector.extract %v[0] : f32 from vector<4xf32>
    %e3 = vector.extract %v[3] : f32 from vector<4xf32>
    %s = arith.addf %e0, %e3 : f32
    %w = vector.insert %s, %v [1] : f32 into vector<4xf32>
    %cst = arith.constant dense<2.0> : vector<4xf32>
    %w2 = arith.mulf %w, %cst : vector<4xf32>
    %q = llvm.getelementptr inbounds %arg1[0, %base] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.array<1024 x f32>
    llvm.store %w2, %q : vector<4xf32>, !llvm.ptr
    %h = llvm.load %p : !llvm.ptr -> vector<8xf16>
    llvm.store %h, %q : vector<8xf16>, !llvm.ptr
    return
  }
})mlir";

TEST_F(MslEmitterTest, VectorizedLoadStore) {
  absl::StatusOr<MslKernel> kernel = Emit(kVectorized, "vec");
  ASSERT_TRUE(kernel.ok()) << kernel.status();
  const std::string& msl = kernel->msl_source;
  EXPECT_THAT(msl, HasSubstr("xla_load<float4>("));
  EXPECT_THAT(msl, HasSubstr("xla_vext<float>("));
  EXPECT_THAT(msl, HasSubstr("xla_vins("));
  EXPECT_THAT(msl, HasSubstr("float4()"));
  EXPECT_THAT(msl, HasSubstr("xla_load<xla_vec<half, 8>>("));
}

// float3 occupies 16 bytes in MSL, so a 3-lane vector's storage and GEP
// stride must be 16 bytes, not 12.
constexpr char kThreeLane[] = R"mlir(
module {
  func.func @three(%arg0: !llvm.ptr) {
    %c4 = arith.constant 4 : i32
    %a = llvm.alloca %c4 x vector<3xf32> : (i32) -> !llvm.ptr
    %tid = gpu.thread_id x
    %i = arith.index_castui %tid : index to i32
    %p = llvm.getelementptr %a[%i] : (!llvm.ptr, i32) -> !llvm.ptr, vector<3xf32>
    %v = llvm.load %p : !llvm.ptr -> vector<3xf32>
    llvm.store %v, %arg0 : vector<3xf32>, !llvm.ptr
    return
  }
})mlir";

TEST_F(MslEmitterTest, ThreeLaneVectorsUseFourLanesOfStorage) {
  absl::StatusOr<MslKernel> kernel = Emit(kThreeLane, "three");
  ASSERT_TRUE(kernel.ok()) << kernel.status();
  const std::string& msl = kernel->msl_source;
  EXPECT_THAT(msl, HasSubstr("xla_alloca_0[4];"));
  EXPECT_THAT(msl, HasSubstr(" = 16;"));
  EXPECT_THAT(msl, Not(HasSubstr(" = 12;")));
}

TEST_F(MslEmitterTest, RejectsF64) {
  constexpr char kIr[] = R"mlir(
module {
  func.func @dbl(%arg0: !llvm.ptr) {
    %v = llvm.load %arg0 : !llvm.ptr -> f64
    %w = arith.addf %v, %v : f64
    llvm.store %w, %arg0 : f64, !llvm.ptr
    return
  }
})mlir";
  absl::StatusOr<MslKernel> kernel = Emit(kIr, "dbl");
  ASSERT_FALSE(kernel.ok());
  EXPECT_EQ(kernel.status().code(), absl::StatusCode::kUnimplemented);
  EXPECT_THAT(kernel.status().message(), HasSubstr("f64"));
}

TEST_F(MslEmitterTest, RejectsUnknownOps) {
  constexpr char kIr[] = R"mlir(
module {
  func.func @wide(%arg0: !llvm.ptr) {
    %v = llvm.load %arg0 : !llvm.ptr -> i32
    %lo, %hi = arith.mulsi_extended %v, %v : i32
    llvm.store %hi, %arg0 : i32, !llvm.ptr
    return
  }
})mlir";
  absl::StatusOr<MslKernel> kernel = Emit(kIr, "wide");
  ASSERT_FALSE(kernel.ok());
  EXPECT_EQ(kernel.status().code(), absl::StatusCode::kUnimplemented);
  EXPECT_THAT(kernel.status().message(), HasSubstr("arith.mulsi_extended"));
}

// A pointer made from an integer we cannot trace to a known pointer has no
// address space; the emitter refuses it instead of guessing "device".
TEST_F(MslEmitterTest, RejectsUntracedIntToPtr) {
  constexpr char kIr[] = R"mlir(
module {
  func.func @untraced(%arg0: !llvm.ptr) {
    %a = llvm.load %arg0 : !llvm.ptr -> i64
    %p = llvm.inttoptr %a : i64 to !llvm.ptr
    %v = llvm.load %p : !llvm.ptr -> i32
    llvm.store %v, %arg0 : i32, !llvm.ptr
    return
  }
})mlir";
  absl::StatusOr<MslKernel> kernel = Emit(kIr, "untraced");
  ASSERT_FALSE(kernel.ok());
  EXPECT_EQ(kernel.status().code(), absl::StatusCode::kUnimplemented);
  EXPECT_THAT(kernel.status().message(), HasSubstr("unknown address space"));
}

// Index is size_t in MSL (unsigned), but arith.divsi/remsi on index (e.g.
// from LowerAffine's mod/floordiv) are signed: -1 rem 3 must be -1 (which
// LowerAffine's mod then corrects to 2), not SIZE_MAX % 3 == 0.
constexpr char kSignedIndexDivRem[] = R"mlir(
module {
  func.func @divrem(%arg0: !llvm.ptr) {
    %cm1 = arith.constant -1 : index
    %c3 = arith.constant 3 : index
    %tid = gpu.thread_id x
    %a = arith.addi %tid, %cm1 : index
    %r = arith.remsi %a, %c3 : index
    %q = arith.divsi %a, %c3 : index
    %s = arith.addi %r, %q : index
    %v = arith.index_cast %s : index to i32
    %ti = arith.index_castui %tid : index to i32
    %p = llvm.getelementptr inbounds %arg0[0, %ti] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.array<4 x i32>
    llvm.store %v, %p : i32, !llvm.ptr
    return
  }
})mlir";

TEST_F(MslEmitterTest, SignedIndexDivRem) {
  absl::StatusOr<MslKernel> kernel = Emit(kSignedIndexDivRem, "divrem");
  ASSERT_TRUE(kernel.ok()) << kernel.status();
  const std::string& msl = kernel->msl_source;
  // Each '/' and '%' must compute in a signed 64-bit type (it was size_t).
  const std::regex divrem(R"((\w+) \w+ = \w+ ([/%]) \w+;)");
  int count = 0;
  for (auto it = std::sregex_iterator(msl.begin(), msl.end(), divrem);
       it != std::sregex_iterator(); ++it) {
    EXPECT_EQ((*it)[1].str(), "int64_t") << (*it)[0].str();
    ++count;
  }
  EXPECT_EQ(count, 2);
}

// scf.for over index must compare signed: a negative upper bound means no
// iterations, not ~2^64 of them (size_t).
constexpr char kSignedIndexLoop[] = R"mlir(
module {
  func.func @loop(%arg0: !llvm.ptr) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %tid = gpu.thread_id x
    %ub = arith.subi %c4, %tid : index
    scf.for %j = %c0 to %ub step %c1 {
      %ji = arith.index_cast %j : index to i32
      %p = llvm.getelementptr inbounds %arg0[0, %ji] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.array<4 x i32>
      llvm.store %ji, %p : i32, !llvm.ptr
    }
    return
  }
})mlir";

TEST_F(MslEmitterTest, SignedIndexLoop) {
  absl::StatusOr<MslKernel> kernel = Emit(kSignedIndexLoop, "loop");
  ASSERT_TRUE(kernel.ok()) << kernel.status();
  const std::string& msl = kernel->msl_source;
  const std::regex loop(R"(for \((\w+) \w+ = )");
  std::smatch m;
  ASSERT_TRUE(std::regex_search(msl, m, loop));
  EXPECT_EQ(m[1].str(), "int64_t");
}

TEST(MslLlvmBridgeTest, RoundTrip) {
  llvm::LLVMContext ctx;
  llvm::Module m("test", ctx);
  MslKernel kernel{"my_kernel", "kernel void my_kernel() {}\n", 3};
  ASSERT_TRUE(EmbedMslInLlvmModule(m, kernel).ok());
  EXPECT_FALSE(EmbedMslInLlvmModule(m, kernel).ok());  // duplicate

  llvm::Function* fn = m.getFunction("my_kernel");
  ASSERT_NE(fn, nullptr);
  EXPECT_EQ(fn->getCallingConv(), llvm::CallingConv::SPIR_KERNEL);
  EXPECT_EQ(fn->arg_size(), 3u);
  EXPECT_EQ(fn->getArg(0)->getType()->getPointerAddressSpace(), 1u);

  std::optional<std::string> msl = ExtractMslFromLlvmModule(m);
  ASSERT_TRUE(msl.has_value());
  EXPECT_EQ(*msl, kernel.msl_source);

  std::vector<EmbeddedMslKernel> kernels = ListMslKernels(m);
  ASSERT_EQ(kernels.size(), 1u);
  EXPECT_EQ(kernels[0].kernel_name, "my_kernel");
  EXPECT_EQ(kernels[0].num_buffer_args, 3);

  llvm::Module empty("empty", ctx);
  EXPECT_FALSE(ExtractMslFromLlvmModule(empty).has_value());
}

}  // namespace
}  // namespace metal_pjrt::codegen
