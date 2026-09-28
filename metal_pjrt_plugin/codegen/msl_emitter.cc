#include "metal_pjrt_plugin/codegen/msl_emitter.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToEmitC/ArithToEmitC.h"
#include "mlir/Conversion/ComplexToStandard/ComplexToStandard.h"
#include "mlir/Conversion/FuncToEmitC/FuncToEmitC.h"
#include "mlir/Conversion/SCFToEmitC/SCFToEmitC.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/Cpp/CppEmitter.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "xla/backends/gpu/codegen/emitters/transforms/passes.h"
#include "xla/codegen/emitters/transforms/passes.h"
#include "xla/stream_executor/device_description.h"

namespace metal_pjrt::codegen {

// ---------------------------------------------------------------------------
// Pass pipeline prefix.
// ---------------------------------------------------------------------------

void AddMslLoweringPasses(mlir::OpPassManager& pm,
                          const stream_executor::DeviceDescription& device) {
  // Mirrors xla::gpu::AddLoweringPasses (mlir_kernel_emitter.cc) for the
  // OneAPI/SPIR case, up to but excluding SCFToControlFlow. No
  // ConvertFloat{Nvidia,AMD} pass runs for OneAPI.
  pm.addNestedPass<mlir::func::FuncOp>(
      xla::emitters::createConvertPureCallOpsPass());
  pm.addPass(xla::emitters::createLowerTensorsPass(device));
  pm.addPass(xla::emitters::createLowerPdlWaitPass());
  pm.addPass(mlir::createConvertComplexToStandardPass());
  pm.addPass(xla::emitters::createMergePointersToSameSlicePass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  // OneAPI: explicit NaN propagation for min/max (MSL fmin/fmax, like SPIR-V,
  // do not propagate NaN).
  xla::emitters::SimplifyArithPassOptions simplify_arith_options;
  simplify_arith_options.fast_min_max_ = false;
  simplify_arith_options.explicit_nan_propagation_ = true;
  pm.addNestedPass<mlir::func::FuncOp>(
      xla::emitters::createSimplifyArithPass(simplify_arith_options));
  pm.addPass(xla::emitters::createSimplifyAffinePass());
  pm.addPass(xla::gpu::createConvertIndexTypePass());
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(mlir::createLoopInvariantCodeMotionPass());
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(xla::emitters::createExpandFloatOpsPass());
  pm.addPass(mlir::createLowerAffinePass());
  // XLA continues with SCFToControlFlow + LowerToLLVMGPU; we stop here.
}

namespace {

namespace ml = ::mlir::LLVM;
using ::mlir::Location;
using ::mlir::LogicalResult;
using ::mlir::ModuleOp;
using ::mlir::OpBuilder;
using ::mlir::Operation;
using ::mlir::Type;
using ::mlir::Value;
using ::mlir::ValueRange;

// Internal pointer address spaces after inference (only used on the retyped
// !llvm.ptr<N> values and in the type converter).
constexpr unsigned kMslDevice = 1;
constexpr unsigned kMslThreadgroup = 3;
constexpr unsigned kMslConstant = 4;
constexpr unsigned kMslThread = 5;

// Order of the extra i32 arguments appended to the body function.
constexpr int kNumIdArgs = 12;  // tid.xyz, bid.xyz, bdim.xyz, gdim.xyz

constexpr absl::string_view kPrelude = R"msl(#include <metal_stdlib>
using namespace metal;

// Spellings used by MLIR's C++ emitter.
#define _Float16 half
#define __bf16 bfloat
#define ssize_t long
#ifndef INFINITY
#define INFINITY as_type<float>(0x7f800000u)
#endif
#ifndef NAN
#define NAN as_type<float>(0x7fc00000u)
#endif

// Vectors that have no native MSL spelling (more than 4 lanes).
template <typename T, int N>
struct xla_vec {
  T e[N];
  thread T& operator[](int i) { return e[i]; }
};

template <typename T, typename V>
inline T xla_vext(V v, int i) { return v[i]; }
template <typename V, typename T>
inline V xla_vins(V v, T s, int i) { v[i] = s; return v; }

// Memory. All pointers are byte pointers; loads/stores reinterpret.
#define XLA_PTR_HELPERS(AS, NAME)                                         \
  template <typename T>                                                   \
  inline T xla_load(AS char* p) { return *((AS T*)p); }                   \
  inline AS char* xla_gep(AS char* p, long off) { return p + off; }       \
  template <typename I>                                                   \
  inline I xla_ptrtoint(AS char* p) { return (I)((ulong)p); }             \
  template <typename I>                                                   \
  inline AS char* xla_inttoptr_##NAME(I i) { return (AS char*)((ulong)i); }
XLA_PTR_HELPERS(device, device)
XLA_PTR_HELPERS(constant, constant)
XLA_PTR_HELPERS(threadgroup, threadgroup)
XLA_PTR_HELPERS(thread, thread)
#undef XLA_PTR_HELPERS

#define XLA_STORE_HELPERS(AS)                                             \
  template <typename T>                                                   \
  inline void xla_store(AS char* p, T v) { *((AS T*)p) = v; }
XLA_STORE_HELPERS(device)
XLA_STORE_HELPERS(threadgroup)
XLA_STORE_HELPERS(thread)
#undef XLA_STORE_HELPERS

// Atomics (relaxed, like LLVM `monotonic`).
template <typename T>
struct xla_cas_result { T value; bool ok; };
template <typename T>
inline T xla_cas_value(xla_cas_result<T> r) { return r.value; }
template <typename T>
inline bool xla_cas_ok(xla_cas_result<T> r) { return r.ok; }

#define XLA_ATOMIC_RMW(AS, NAME, FN)                                      \
  template <typename T>                                                   \
  inline T xla_atomic_##NAME(AS char* p, T v) {                           \
    return FN((AS atomic<T>*)p, v, memory_order_relaxed);                 \
  }
#define XLA_ATOMIC_URMW(AS, NAME, FN)                                     \
  template <typename T>                                                   \
  inline T xla_atomic_##NAME(AS char* p, T v) {                           \
    return as_type<T>(FN((AS atomic<uint>*)p, as_type<uint>(v),           \
                         memory_order_relaxed));                          \
  }
#define XLA_ATOMIC_HELPERS(AS)                                            \
  XLA_ATOMIC_RMW(AS, add, atomic_fetch_add_explicit)                      \
  XLA_ATOMIC_RMW(AS, fadd, atomic_fetch_add_explicit)                     \
  XLA_ATOMIC_RMW(AS, sub, atomic_fetch_sub_explicit)                      \
  XLA_ATOMIC_RMW(AS, andi, atomic_fetch_and_explicit)                      \
  XLA_ATOMIC_RMW(AS, ori, atomic_fetch_or_explicit)                       \
  XLA_ATOMIC_RMW(AS, xori, atomic_fetch_xor_explicit)                     \
  XLA_ATOMIC_RMW(AS, max, atomic_fetch_max_explicit)                      \
  XLA_ATOMIC_RMW(AS, min, atomic_fetch_min_explicit)                      \
  XLA_ATOMIC_RMW(AS, xchg, atomic_exchange_explicit)                      \
  XLA_ATOMIC_URMW(AS, umax, atomic_fetch_max_explicit)                    \
  XLA_ATOMIC_URMW(AS, umin, atomic_fetch_min_explicit)                    \
  template <typename T>                                                   \
  inline xla_cas_result<T> xla_cmpxchg(AS char* p, T cmp, T val) {        \
    T expected = cmp;                                                     \
    bool ok = atomic_compare_exchange_weak_explicit(                      \
        (AS atomic<T>*)p, &expected, val, memory_order_relaxed,           \
        memory_order_relaxed);                                            \
    xla_cas_result<T> r;                                                  \
    r.value = expected;                                                   \
    r.ok = ok;                                                            \
    return r;                                                             \
  }
XLA_ATOMIC_HELPERS(device)
XLA_ATOMIC_HELPERS(threadgroup)
#undef XLA_ATOMIC_HELPERS
#undef XLA_ATOMIC_URMW
#undef XLA_ATOMIC_RMW

// Synchronization and SIMD-group shuffles (simdgroup size is 32).
inline void xla_barrier() {
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
}
template <typename T, typename I>
inline T xla_shfl_down(T v, I d) { return simd_shuffle_down(v, (ushort)d); }
template <typename T, typename I>
inline T xla_shfl_up(T v, I d) { return simd_shuffle_up(v, (ushort)d); }
template <typename T, typename I>
inline T xla_shfl_xor(T v, I m) { return simd_shuffle_xor(v, (ushort)m); }
template <typename T, typename I>
inline T xla_shfl_idx(T v, I l) { return simd_shuffle(v, (ushort)l); }

// Math functions MSL does not provide. Computed in float, compile with
// fast math disabled.
template <typename T>
inline T xla_log1p(T x) {
  float xf = float(x);
  float u = 1.0f + xf;
  if (u == 1.0f) return x;
  if (isinf(u)) return T(log(u));
  return T(log(u) * (xf / (u - 1.0f)));
}
template <typename T>
inline T xla_expm1(T x) {
  float xf = float(x);
  float u = exp(xf);
  if (u == 1.0f) return x;
  float um1 = u - 1.0f;
  if (um1 == -1.0f) return T(-1.0f);
  if (isinf(u)) return T(u);
  return T(um1 * xf / log(u));
}
// Metal's float exp/sin/cos are biased by ~0.3 ulp for small |x|, which a
// long recursion accumulates (docs/accuracy.md). Taylor polynomials there
// (truncation < 1e-9 relative for |x| < 0.125); Metal's elsewhere.
inline float xla_exp(float x) {
  if (!(fabs(x) < 0.125f)) return exp(x);
  float p = fma(x, fma(x, fma(x, fma(x, fma(x, 1.0f / 720, 1.0f / 120),
                                     1.0f / 24), 1.0f / 6), 0.5f), 1.0f);
  return fma(x, p, 1.0f);
}
inline float xla_sin(float x) {
  if (!(fabs(x) < 0.125f)) return sin(x);
  if (fabs(x) < 1e-4f) return x;  // sin(x) rounds to x; keeps sin(-0) = -0
  float x2 = x * x;
  return fma(x * x2, fma(x2, fma(x2, -1.0f / 5040, 1.0f / 120), -1.0f / 6),
             x);
}
inline float xla_cos(float x) {
  if (!(fabs(x) < 0.125f)) return cos(x);
  float x2 = x * x;
  return fma(x2, fma(x2, fma(x2, -1.0f / 720, 1.0f / 24), -0.5f), 1.0f);
}
// Metal's float arithmetic flushes subnormals, so its log/log2/log10 treat
// them as 0 (log(1e-40) = -inf, not -92.1). A subnormal is m * 2^-149 with
// the integer m = its bits, so log(x) = log(m) - 149 ln 2; a negative one
// has a negative int(bits), so log gives NaN. Selects: a branch (either on
// the input or on a -inf result) cost twice as much in an ALU-bound loop.
inline bool xla_subnormal(float x) {
  return (as_type<uint>(x) & 0x7fffffffu) - 1u < 0x7fffffu;
}
inline float xla_log(float x) {
  bool s = xla_subnormal(x);
  float y = log(s ? float(as_type<int>(x)) : x);
  return s ? y - 103.278930f : y;  // 149 ln 2
}
inline float xla_log2(float x) {
  bool s = xla_subnormal(x);
  float y = log2(s ? float(as_type<int>(x)) : x);
  return s ? y - 149.0f : y;
}
inline float xla_log10(float x) {
  bool s = xla_subnormal(x);
  float y = log10(s ? float(as_type<int>(x)) : x);
  return s ? y - 44.8534694f : y;  // 149 log10(2)
}
template <typename T>
inline T xla_log(T x) { return log(x); }
template <typename T>
inline T xla_log2(T x) { return log2(x); }
template <typename T>
inline T xla_log10(T x) { return log10(x); }
template <typename T>
inline T xla_exp(T x) { return exp(x); }
template <typename T>
inline T xla_sin(T x) { return sin(x); }
template <typename T>
inline T xla_cos(T x) { return cos(x); }
// erfc for a >= 0 (Numerical Recipes erfcc; fractional error < 1.2e-7).
inline float xla_erfc_pos(float a) {
  float t = 1.0f / (1.0f + 0.5f * a);
  float p = -a * a - 1.26551223f +
            t * (1.00002368f +
            t * (0.37409196f +
            t * (0.09678418f +
            t * (-0.18628806f +
            t * (0.27886807f +
            t * (-1.13520398f +
            t * (1.48851587f +
            t * (-0.82215223f + t * 0.17087277f))))))));
  return t * exp(p);
}
template <typename T>
inline T xla_erf(T x) {
  float xf = float(x);
  float a = fabs(xf);
  if (a < 0.25f) {
    float x2 = xf * xf;
    return T(1.1283791671f * xf *
             (1.0f + x2 * (-1.0f / 3.0f +
                           x2 * (0.1f + x2 * (-1.0f / 42.0f +
                                              x2 * (1.0f / 216.0f))))));
  }
  return T(copysign(1.0f - xla_erfc_pos(a), xf));
}
template <typename T>
inline T xla_erfc(T x) {
  float xf = float(x);
  float r = xla_erfc_pos(fabs(xf));
  return T(xf < 0.0f ? 2.0f - r : r);
}
// pow(|x|, 1/3) alone is 7-12 ulps off away from 1 (1/3 rounds up in float,
// an error that grows with |log x|) and 0 for subnormals (flushed). One
// Newton step, (2r + a / r^2) / 3, which cannot overflow near FLT_MAX,
// fixes the first; a subnormal m * 2^-149 is cbrt(2m) * 2^-50.
inline float xla_cbrt(float x) {
  bool s = xla_subnormal(x);
  float a = s ? 2.0f * float(as_type<int>(x) & 0x7fffffff) : fabs(x);
  if (a == 0.0f || !isfinite(a)) return x;
  float r = pow(a, 1.0f / 3.0f);
  r = (2.0f * r + a / (r * r)) / 3.0f;
  return copysign(s ? r * 8.8817841970012523e-16f : r, x);  // 2^-50
}
template <typename T>
inline T xla_cbrt(T x) { return T(xla_cbrt(float(x))); }
template <typename T>
inline T xla_powf(T x, T y) {
  float xf = float(x);
  float yf = float(y);
  if (xf < 0.0f) {
    if (yf != trunc(yf)) return T(NAN);
    float r = pow(-xf, yf);
    return T(fmod(fabs(yf), 2.0f) == 1.0f ? -r : r);
  }
  return T(pow(xf, yf));
}
template <typename T, typename I>
inline T xla_fpowi(T x, I n) {
  float b = float(x);
  long e = long(n);
  bool neg = e < 0;
  if (neg) e = -e;
  float r = 1.0f;
  while (e != 0) {
    if (e & 1) r *= b;
    b *= b;
    e >>= 1;
  }
  return T(neg ? 1.0f / r : r);
}
template <typename T>
inline T xla_ipowi(T b, T e) {
  if (e < 0) {
    if (b == 1) return T(1);
    if (b == -1) return (e & 1) ? T(-1) : T(1);
    return T(0);
  }
  T r = T(1);
  while (e != 0) {
    if (e & 1) r *= b;
    b *= b;
    e >>= 1;
  }
  return r;
}
)msl";

// ---------------------------------------------------------------------------
// Type helpers.
// ---------------------------------------------------------------------------

bool IsSmallFloat(Type t) {
  auto f = mlir::dyn_cast<mlir::FloatType>(t);
  return f && f.getWidth() <= 8;
}

std::optional<std::string> MslScalarName(Type t) {
  if (auto i = mlir::dyn_cast<mlir::IntegerType>(t)) {
    switch (i.getWidth()) {
      case 1:
        return "bool";
      case 8:
        return "char";
      case 16:
        return "short";
      case 32:
        return "int";
      case 64:
        return "long";
      default:
        return std::nullopt;
    }
  }
  if (t.isF16()) return "half";
  if (t.isBF16()) return "bfloat";
  if (t.isF32()) return "float";
  if (IsSmallFloat(t)) return "char";  // f8/f6/f4 are stored as raw bytes.
  if (t.isIndex()) return "size_t";
  return std::nullopt;
}

bool IsNativeVector(mlir::VectorType vt) {
  if (vt.getRank() != 1) return false;
  int64_t n = vt.getNumElements();
  return n >= 2 && n <= 4 && !vt.getElementType().isIndex();
}

std::optional<std::string> MslVectorName(mlir::VectorType vt) {
  if (vt.getRank() != 1) return std::nullopt;
  std::optional<std::string> elem = MslScalarName(vt.getElementType());
  if (!elem) return std::nullopt;
  int64_t n = vt.getNumElements();
  if (IsNativeVector(vt)) return absl::StrCat(*elem, n);
  return absl::StrCat("xla_vec<", *elem, ", ", n, ">");
}

const char* AddressSpaceKeyword(unsigned as) {
  switch (as) {
    case kMslDevice:
      return "device";
    case kMslThreadgroup:
      return "threadgroup";
    case kMslConstant:
      return "constant";
    case kMslThread:
      return "thread";
    default:
      return nullptr;
  }
}

bool IsCasStruct(Type t) {
  auto st = mlir::dyn_cast<ml::LLVMStructType>(t);
  if (!st || st.getBody().size() != 2) return false;
  return st.getBody()[0].isInteger(32) && st.getBody()[1].isInteger(1);
}

// Byte size / alignment of LLVM-dialect memory types (natural layout).
std::optional<int64_t> ByteSize(Type t);

std::optional<int64_t> ByteAlign(Type t) {
  if (auto at = mlir::dyn_cast<ml::LLVMArrayType>(t)) {
    return ByteAlign(at.getElementType());
  }
  // xla_vec<T, N> is a plain T[N]; native vectors align to their size.
  if (auto vt = mlir::dyn_cast<mlir::VectorType>(t)) {
    if (!IsNativeVector(vt)) return ByteAlign(vt.getElementType());
  }
  if (auto st = mlir::dyn_cast<ml::LLVMStructType>(t)) {
    if (st.isPacked()) return 1;
    int64_t a = 1;
    for (Type f : st.getBody()) {
      std::optional<int64_t> fa = ByteAlign(f);
      if (!fa) return std::nullopt;
      a = std::max(a, *fa);
    }
    return a;
  }
  return ByteSize(t);
}

std::optional<int64_t> StructFieldOffset(ml::LLVMStructType st, int64_t field) {
  int64_t off = 0;
  for (int64_t i = 0; i < static_cast<int64_t>(st.getBody().size()); ++i) {
    Type f = st.getBody()[i];
    std::optional<int64_t> a =
        st.isPacked() ? std::optional<int64_t>(1) : ByteAlign(f);
    std::optional<int64_t> s = ByteSize(f);
    if (!a || !s) return std::nullopt;
    off = (off + *a - 1) / *a * *a;
    if (i == field) return off;
    off += *s;
  }
  return std::nullopt;
}

std::optional<int64_t> ByteSize(Type t) {
  if (auto it = mlir::dyn_cast<mlir::IntegerType>(t)) {
    return std::max<int64_t>(1, (it.getWidth() + 7) / 8);
  }
  if (auto ft = mlir::dyn_cast<mlir::FloatType>(t)) {
    return std::max<int64_t>(1, (ft.getWidth() + 7) / 8);
  }
  if (mlir::isa<ml::LLVMPointerType>(t)) return 8;
  if (auto vt = mlir::dyn_cast<mlir::VectorType>(t)) {
    if (vt.getRank() != 1) return std::nullopt;
    std::optional<int64_t> e = ByteSize(vt.getElementType());
    if (!e) return std::nullopt;
    // MSL's 3-lane vectors (float3, half3, ...) occupy four lanes.
    int64_t n = vt.getNumElements();
    return *e * (IsNativeVector(vt) && n == 3 ? 4 : n);
  }
  if (auto at = mlir::dyn_cast<ml::LLVMArrayType>(t)) {
    std::optional<int64_t> e = ByteSize(at.getElementType());
    if (!e) return std::nullopt;
    return *e * static_cast<int64_t>(at.getNumElements());
  }
  if (auto st = mlir::dyn_cast<ml::LLVMStructType>(t)) {
    int64_t n = st.getBody().size();
    if (n == 0) return 0;
    std::optional<int64_t> last = StructFieldOffset(st, n - 1);
    std::optional<int64_t> last_size = ByteSize(st.getBody()[n - 1]);
    std::optional<int64_t> align = ByteAlign(st);
    if (!last || !last_size || !align) return std::nullopt;
    int64_t end = *last + *last_size;
    return (end + *align - 1) / *align * *align;
  }
  return std::nullopt;
}

std::string OpName(Operation* op) {
  return op->getName().getStringRef().str();
}

absl::Status Unimplemented(Operation* op, absl::string_view what) {
  std::string loc;
  llvm::raw_string_ostream os(loc);
  op->getLoc().print(os);
  return absl::UnimplementedError(absl::StrCat(
      "MSL emitter: unsupported ", what, " in op '", OpName(op), "' at ", loc));
}

// ---------------------------------------------------------------------------
// Math function table: op name -> MSL function. bf16 operands are computed in
// f32 (MSL has no bfloat math library).
// ---------------------------------------------------------------------------

const llvm::StringMap<std::string>& MathFunctions() {
  static const auto* table = new llvm::StringMap<std::string>{
      {"math.absf", "fabs"},        {"math.absi", "abs"},
      {"math.acos", "acos"},        {"math.acosh", "acosh"},
      {"math.asin", "asin"},        {"math.asinh", "asinh"},
      {"math.atan", "atan"},        {"math.atanh", "atanh"},
      {"math.atan2", "atan2"},      {"math.cbrt", "xla_cbrt"},
      {"math.ceil", "ceil"},        {"math.copysign", "copysign"},
      {"math.cos", "xla_cos"},         {"math.cosh", "cosh"},
      {"math.sin", "xla_sin"},         {"math.sinh", "sinh"},
      {"math.tan", "tan"},          {"math.tanh", "tanh"},
      {"math.ctlz", "clz"},         {"math.cttz", "ctz"},
      {"math.ctpop", "popcount"},   {"math.erf", "xla_erf"},
      {"math.erfc", "xla_erfc"},    {"math.exp", "xla_exp"},
      {"math.exp2", "exp2"},        {"math.expm1", "xla_expm1"},
      {"math.floor", "floor"},      {"math.fma", "fma"},
      {"math.ipowi", "xla_ipowi"},  {"math.fpowi", "xla_fpowi"},
      {"math.isfinite", "isfinite"}, {"math.isinf", "isinf"},
      {"math.isnan", "isnan"},      {"math.isnormal", "isnormal"},
      {"math.log", "xla_log"},      {"math.log10", "xla_log10"},
      {"math.log1p", "xla_log1p"},  {"math.log2", "xla_log2"},
      {"math.powf", "xla_powf"},    {"math.rsqrt", "rsqrt"},
      {"math.sqrt", "sqrt"},        {"math.roundeven", "rint"},
      {"math.round", "round"},      {"math.trunc", "trunc"},
      {"arith.remf", "fmod"},
      {"llvm.intr.fma", "fma"},     {"llvm.intr.fmuladd", "fma"},
      {"llvm.intr.fabs", "fabs"},   {"llvm.intr.sqrt", "sqrt"},
      {"llvm.intr.exp", "xla_exp"},    {"llvm.intr.exp2", "exp2"},
      {"llvm.intr.log", "xla_log"},  {"llvm.intr.log2", "xla_log2"},
      {"llvm.intr.log10", "xla_log10"}, {"llvm.intr.sin", "xla_sin"},
      {"llvm.intr.cos", "xla_cos"},    {"llvm.intr.tan", "tan"},
      {"llvm.intr.floor", "floor"}, {"llvm.intr.ceil", "ceil"},
      {"llvm.intr.trunc", "trunc"}, {"llvm.intr.rint", "rint"},
      {"llvm.intr.nearbyint", "rint"}, {"llvm.intr.roundeven", "rint"},
      {"llvm.intr.round", "round"}, {"llvm.intr.copysign", "copysign"},
      {"llvm.intr.maxnum", "fmax"}, {"llvm.intr.minnum", "fmin"},
      {"llvm.intr.pow", "xla_powf"}, {"llvm.intr.ctlz", "clz"},
      {"llvm.intr.cttz", "ctz"},    {"llvm.intr.ctpop", "popcount"},
      {"llvm.intr.abs", "abs"},
  };
  return *table;
}

// Ops (other than the math table) that may appear in the input.
const llvm::StringSet<>& SupportedOps() {
  static const auto* ops = new llvm::StringSet<>{
      "builtin.module", "builtin.unrealized_conversion_cast",
      // func
      "func.func", "func.call", "func.return",
      // scf
      "scf.for", "scf.if", "scf.while", "scf.condition", "scf.yield",
      "scf.index_switch",
      // arith (min/max/ceildiv/floordiv are expanded before conversion)
      "arith.constant", "arith.addi", "arith.subi", "arith.muli",
      "arith.divsi", "arith.divui", "arith.remsi", "arith.remui",
      "arith.andi", "arith.ori", "arith.xori", "arith.shli", "arith.shrui",
      "arith.shrsi", "arith.addf", "arith.subf", "arith.mulf", "arith.divf",
      "arith.negf", "arith.cmpi", "arith.cmpf", "arith.select",
      "arith.extui", "arith.extsi", "arith.trunci", "arith.sitofp",
      "arith.uitofp", "arith.fptosi", "arith.fptoui", "arith.extf",
      "arith.truncf", "arith.bitcast", "arith.index_cast",
      "arith.index_castui", "arith.minsi", "arith.maxsi", "arith.minui",
      "arith.maxui", "arith.minimumf", "arith.maximumf", "arith.minnumf",
      "arith.maxnumf", "arith.ceildivsi", "arith.ceildivui",
      "arith.floordivsi",
      // vector
      "vector.extract", "vector.insert", "vector.broadcast",
      "vector.from_elements", "vector.extract_strided_slice",
      "vector.insert_strided_slice",
      // gpu
      "gpu.thread_id", "gpu.block_id", "gpu.block_dim", "gpu.grid_dim",
      "gpu.barrier", "gpu.shuffle",
      // ub
      "ub.poison",
      // llvm
      "llvm.load", "llvm.store", "llvm.getelementptr", "llvm.inttoptr",
      "llvm.ptrtoint", "llvm.alloca", "llvm.mlir.constant",
      "llvm.mlir.undef", "llvm.mlir.poison", "llvm.mlir.zero",
      "llvm.mlir.addressof", "llvm.mlir.global", "llvm.atomicrmw",
      "llvm.cmpxchg", "llvm.extractvalue", "llvm.bitcast",
      "llvm.addrspacecast", "llvm.extractelement", "llvm.insertelement",
      "llvm.freeze",
      // llvm arithmetic (rewritten to arith before conversion)
      "llvm.and", "llvm.or", "llvm.xor", "llvm.add", "llvm.sub", "llvm.mul",
      "llvm.shl", "llvm.lshr", "llvm.ashr", "llvm.udiv", "llvm.sdiv",
      "llvm.urem", "llvm.srem", "llvm.trunc", "llvm.zext", "llvm.sext",
      "llvm.icmp", "llvm.fcmp", "llvm.select", "llvm.fadd", "llvm.fsub",
      "llvm.fmul", "llvm.fdiv", "llvm.frem", "llvm.fneg", "llvm.fpext",
      "llvm.fptrunc", "llvm.sitofp", "llvm.uitofp", "llvm.fptosi",
      "llvm.fptoui",
  };
  return *ops;
}

// ---------------------------------------------------------------------------
// Pre-conversion rewrites on the MLIR module.
// ---------------------------------------------------------------------------

// Removes unrealized_conversion_casts left behind by LowerTensors.
absl::Status CleanUpUnrealizedCasts(ModuleOp module) {
  bool changed = true;
  while (changed) {
    changed = false;
    llvm::SmallVector<mlir::UnrealizedConversionCastOp> casts;
    module.walk([&](mlir::UnrealizedConversionCastOp c) { casts.push_back(c); });
    for (mlir::UnrealizedConversionCastOp c : casts) {
      if (c->use_empty()) {
        c->erase();
        changed = true;
        continue;
      }
      if (c.getNumOperands() != 1 || c.getNumResults() != 1) continue;
      Value in = c.getOperand(0);
      Type out_ty = c.getResult(0).getType();
      if (in.getType() == out_ty) {
        c.getResult(0).replaceAllUsesWith(in);
        c->erase();
        changed = true;
        continue;
      }
      // cast(cast(x : A -> B) : B -> A) == x
      if (auto inner = in.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
        if (inner.getNumOperands() == 1 &&
            inner.getOperand(0).getType() == out_ty) {
          c.getResult(0).replaceAllUsesWith(inner.getOperand(0));
          c->erase();
          changed = true;
          continue;
        }
      }
    }
  }
  absl::Status status;
  module.walk([&](mlir::UnrealizedConversionCastOp c) {
    if (status.ok()) {
      status = Unimplemented(c, "non-trivial unrealized_conversion_cast");
    }
  });
  return status;
}

// Rewrites scalar/vector LLVM-dialect arithmetic into arith ops so that the
// upstream ArithToEmitC patterns handle them.
absl::Status RewriteLlvmArithmetic(ModuleOp module) {
  llvm::SmallVector<Operation*> ops;
  module.walk([&](Operation* op) {
    if (op->getDialect() &&
        op->getDialect()->getNamespace() == ml::LLVMDialect::getDialectNamespace()) {
      ops.push_back(op);
    }
  });
  for (Operation* op : ops) {
    OpBuilder b(op);
    Location loc = op->getLoc();
    auto operand = [&](int i) { return op->getOperand(i); };
    Type res_ty = op->getNumResults() == 1 ? op->getResult(0).getType() : Type();
    if (mlir::isa<ml::SelectOp>(op)) {
      // arith.select accepts any type, including pointers.
      Value sel = mlir::arith::SelectOp::create(b, loc, operand(0), operand(1),
                                                operand(2));
      op->getResult(0).replaceAllUsesWith(sel);
      op->erase();
      continue;
    }
    // Pointer-typed operands are left alone (and rejected by validation if
    // unsupported).
    if (llvm::any_of(op->getOperandTypes(), [](Type t) {
          return mlir::isa<ml::LLVMPointerType>(t);
        }) ||
        (res_ty && mlir::isa<ml::LLVMPointerType>(res_ty))) {
      continue;
    }
    Value repl;
    llvm::StringRef name = op->getName().getStringRef();
    if (name == "llvm.and") {
      repl = mlir::arith::AndIOp::create(b, loc, operand(0), operand(1));
    } else if (name == "llvm.or") {
      repl = mlir::arith::OrIOp::create(b, loc, operand(0), operand(1));
    } else if (name == "llvm.xor") {
      repl = mlir::arith::XOrIOp::create(b, loc, operand(0), operand(1));
    } else if (name == "llvm.add") {
      repl = mlir::arith::AddIOp::create(b, loc, operand(0), operand(1));
    } else if (name == "llvm.sub") {
      repl = mlir::arith::SubIOp::create(b, loc, operand(0), operand(1));
    } else if (name == "llvm.mul") {
      repl = mlir::arith::MulIOp::create(b, loc, operand(0), operand(1));
    } else if (name == "llvm.shl") {
      repl = mlir::arith::ShLIOp::create(b, loc, operand(0), operand(1));
    } else if (name == "llvm.lshr") {
      repl = mlir::arith::ShRUIOp::create(b, loc, operand(0), operand(1));
    } else if (name == "llvm.ashr") {
      repl = mlir::arith::ShRSIOp::create(b, loc, operand(0), operand(1));
    } else if (name == "llvm.udiv") {
      repl = mlir::arith::DivUIOp::create(b, loc, operand(0), operand(1));
    } else if (name == "llvm.sdiv") {
      repl = mlir::arith::DivSIOp::create(b, loc, operand(0), operand(1));
    } else if (name == "llvm.urem") {
      repl = mlir::arith::RemUIOp::create(b, loc, operand(0), operand(1));
    } else if (name == "llvm.srem") {
      repl = mlir::arith::RemSIOp::create(b, loc, operand(0), operand(1));
    } else if (name == "llvm.fadd") {
      repl = mlir::arith::AddFOp::create(b, loc, operand(0), operand(1));
    } else if (name == "llvm.fsub") {
      repl = mlir::arith::SubFOp::create(b, loc, operand(0), operand(1));
    } else if (name == "llvm.fmul") {
      repl = mlir::arith::MulFOp::create(b, loc, operand(0), operand(1));
    } else if (name == "llvm.fdiv") {
      repl = mlir::arith::DivFOp::create(b, loc, operand(0), operand(1));
    } else if (name == "llvm.frem") {
      repl = mlir::arith::RemFOp::create(b, loc, operand(0), operand(1));
    } else if (name == "llvm.fneg") {
      repl = mlir::arith::NegFOp::create(b, loc, operand(0));
    } else if (name == "llvm.trunc") {
      repl = mlir::arith::TruncIOp::create(b, loc, res_ty, operand(0));
    } else if (name == "llvm.zext") {
      repl = mlir::arith::ExtUIOp::create(b, loc, res_ty, operand(0));
    } else if (name == "llvm.sext") {
      repl = mlir::arith::ExtSIOp::create(b, loc, res_ty, operand(0));
    } else if (name == "llvm.fpext") {
      repl = mlir::arith::ExtFOp::create(b, loc, res_ty, operand(0));
    } else if (name == "llvm.fptrunc") {
      repl = mlir::arith::TruncFOp::create(b, loc, res_ty, operand(0));
    } else if (name == "llvm.sitofp") {
      repl = mlir::arith::SIToFPOp::create(b, loc, res_ty, operand(0));
    } else if (name == "llvm.uitofp") {
      repl = mlir::arith::UIToFPOp::create(b, loc, res_ty, operand(0));
    } else if (name == "llvm.fptosi") {
      repl = mlir::arith::FPToSIOp::create(b, loc, res_ty, operand(0));
    } else if (name == "llvm.fptoui") {
      repl = mlir::arith::FPToUIOp::create(b, loc, res_ty, operand(0));
    } else if (auto icmp = mlir::dyn_cast<ml::ICmpOp>(op)) {
      // LLVM and arith predicate enums have identical numbering.
      repl = mlir::arith::CmpIOp::create(
          b, loc,
          static_cast<mlir::arith::CmpIPredicate>(
              static_cast<uint64_t>(icmp.getPredicate())),
          operand(0), operand(1));
    } else if (auto fcmp = mlir::dyn_cast<ml::FCmpOp>(op)) {
      repl = mlir::arith::CmpFOp::create(
          b, loc,
          static_cast<mlir::arith::CmpFPredicate>(
              static_cast<uint64_t>(fcmp.getPredicate())),
          operand(0), operand(1));
    } else if (auto cst = mlir::dyn_cast<ml::ConstantOp>(op)) {
      mlir::Attribute value = cst.getValue();
      if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(value)) {
        if (!mlir::isa<mlir::IntegerType>(res_ty)) {
          return Unimplemented(op, "constant type");
        }
        repl = mlir::arith::ConstantOp::create(
            b, loc,
            mlir::cast<mlir::TypedAttr>(b.getIntegerAttr(
                res_ty,
                ia.getValue().sextOrTrunc(res_ty.getIntOrFloatBitWidth()))));
      } else if (auto fa = mlir::dyn_cast<mlir::FloatAttr>(value)) {
        repl = mlir::arith::ConstantOp::create(
            b, loc,
            mlir::cast<mlir::TypedAttr>(b.getFloatAttr(res_ty, fa.getValue())));
      } else if (auto da = mlir::dyn_cast<mlir::DenseElementsAttr>(value)) {
        if (da.getType() != res_ty) {
          return Unimplemented(op, "dense constant type");
        }
        repl = mlir::arith::ConstantOp::create(b, loc,
                                               mlir::cast<mlir::TypedAttr>(da));
      } else {
        return Unimplemented(op, "constant attribute");
      }
    } else {
      continue;
    }
    op->getResult(0).replaceAllUsesWith(repl);
    op->erase();
  }
  return absl::OkStatus();
}

// Expands arith ops that have no EmitC lowering into ones that do, makes
// signed index arithmetic and index loops signed in MSL, and 16-bit
// multiplies free of C's integer promotion.
absl::Status ExpandArith(ModuleOp module) {
  mlir::RewritePatternSet patterns(module.getContext());
  mlir::arith::populateCeilFloorDivExpandOpsPatterns(patterns);
  mlir::arith::populateExpandMinMaxPatterns(patterns);
  if (mlir::failed(mlir::applyPatternsGreedily(module, std::move(patterns)))) {
    return absl::InternalError("MSL emitter: arith expansion failed");
  }
  // Index is size_t in MSL, and ArithToEmitC emits divsi/remsi/shrsi on it
  // as unsigned C operators. They come from LowerAffine (mod/floordiv of
  // possibly negative values, after ConvertIndexType has run) and from the
  // floordivsi/ceildivsi expansion above, and must be signed: compute them
  // in i64 (XLA's LLVM path treats index as a signed i64 too).
  llvm::SmallVector<Operation*> signed_index_ops;
  module.walk([&](Operation* op) {
    if (mlir::isa<mlir::arith::DivSIOp, mlir::arith::RemSIOp,
                  mlir::arith::ShRSIOp>(op) &&
        op->getResult(0).getType().isIndex()) {
      signed_index_ops.push_back(op);
    }
  });
  for (Operation* op : signed_index_ops) {
    OpBuilder b(op);
    const Location loc = op->getLoc();
    auto to_i64 = [&](Value v) -> Value {
      return mlir::arith::IndexCastOp::create(b, loc, b.getI64Type(), v);
    };
    Value lhs = to_i64(op->getOperand(0));
    Value rhs = to_i64(op->getOperand(1));
    Value r;
    if (mlir::isa<mlir::arith::DivSIOp>(op)) {
      r = mlir::arith::DivSIOp::create(b, loc, lhs, rhs);
    } else if (mlir::isa<mlir::arith::RemSIOp>(op)) {
      r = mlir::arith::RemSIOp::create(b, loc, lhs, rhs);
    } else {
      r = mlir::arith::ShRSIOp::create(b, loc, lhs, rhs);
    }
    op->getResult(0).replaceAllUsesWith(
        mlir::arith::IndexCastOp::create(b, loc, b.getIndexType(), r));
    op->erase();
  }
  // Likewise scf.for over index would become `for (size_t i = lb; i < ub;
  // ...)`, an unsigned compare that loops ~2^64 times on a negative bound
  // (a GPU hang). Loop over i64 instead and cast the induction variable.
  llvm::SmallVector<mlir::scf::ForOp> index_loops;
  module.walk([&](mlir::scf::ForOp f) {
    if (f.getInductionVar().getType().isIndex()) index_loops.push_back(f);
  });
  for (mlir::scf::ForOp f : index_loops) {
    OpBuilder b(f);
    const Location loc = f.getLoc();
    auto to_i64 = [&](Value v) -> Value {
      return mlir::arith::IndexCastOp::create(b, loc, b.getI64Type(), v);
    };
    f.getLowerBoundMutable().assign(to_i64(f.getLowerBound()));
    f.getUpperBoundMutable().assign(to_i64(f.getUpperBound()));
    f.getStepMutable().assign(to_i64(f.getStep()));
    auto iv = mlir::cast<mlir::BlockArgument>(f.getInductionVar());
    iv.setType(b.getI64Type());
    b.setInsertionPointToStart(f.getBody());
    auto idx = mlir::arith::IndexCastOp::create(b, loc, b.getIndexType(), iv);
    iv.replaceAllUsesExcept(idx, idx);
  }
  // C promotes 16-bit operands to int, so the uint16_t multiply ArithToEmitC
  // emits can overflow int (65535 * 65535, undefined behaviour). Multiply
  // in 32 bits and truncate: the low 16 bits are the same. (8-bit products
  // fit in int; uint32_t operands are not promoted.)
  llvm::SmallVector<mlir::arith::MulIOp> mul16;
  module.walk([&](mlir::arith::MulIOp m) {
    if (mlir::getElementTypeOrSelf(m.getType()).isInteger(16)) {
      mul16.push_back(m);
    }
  });
  for (mlir::arith::MulIOp m : mul16) {
    OpBuilder b(m);
    const Location loc = m.getLoc();
    const Type narrow = m.getType();
    Type wide = b.getI32Type();
    if (auto v = mlir::dyn_cast<mlir::VectorType>(narrow)) {
      wide = mlir::VectorType::get(v.getShape(), wide);
    }
    Value lhs = mlir::arith::ExtUIOp::create(b, loc, wide, m.getLhs());
    Value rhs = mlir::arith::ExtUIOp::create(b, loc, wide, m.getRhs());
    Value prod = mlir::arith::MulIOp::create(b, loc, lhs, rhs);
    m.getResult().replaceAllUsesWith(
        mlir::arith::TruncIOp::create(b, loc, narrow, prod));
    m.erase();
  }
  return absl::OkStatus();
}

// Vectors only survive as values moved through memory and extract/insert.
// Elementwise arith/math ops (and vector constants) are scalarized here.
absl::Status ScalarizeVectorArithmetic(ModuleOp module) {
  llvm::SmallVector<Operation*> ops;
  module.walk([&](Operation* op) {
    mlir::Dialect* d = op->getDialect();
    if (!d) return;
    llvm::StringRef ns = d->getNamespace();
    if (ns != "arith" && ns != "math") return;
    if (llvm::none_of(op->getResultTypes(), [](Type t) {
          return mlir::isa<mlir::VectorType>(t);
        })) {
      return;
    }
    ops.push_back(op);
  });
  for (Operation* op : ops) {
    OpBuilder b(op);
    Location loc = op->getLoc();
    auto vt = mlir::cast<mlir::VectorType>(op->getResult(0).getType());
    if (vt.getRank() != 1) return Unimplemented(op, "multi-dimensional vector");
    int64_t n = vt.getNumElements();
    if (auto cst = mlir::dyn_cast<mlir::arith::ConstantOp>(op)) {
      auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(cst.getValue());
      if (!dense) return Unimplemented(op, "vector constant attribute");
      llvm::SmallVector<Value> lanes;
      for (mlir::Attribute a : dense.getValues<mlir::Attribute>()) {
        lanes.push_back(
            mlir::arith::ConstantOp::create(b, loc, mlir::cast<mlir::TypedAttr>(a)));
      }
      if (static_cast<int64_t>(lanes.size()) != n) {
        return Unimplemented(op, "vector constant");
      }
      Value v = mlir::vector::FromElementsOp::create(b, loc, vt, lanes);
      op->getResult(0).replaceAllUsesWith(v);
      op->erase();
      continue;
    }
    if (!mlir::OpTrait::hasElementwiseMappableTraits(op) ||
        op->getNumRegions() != 0) {
      return Unimplemented(op, "vector-typed non-elementwise op");
    }
    llvm::SmallVector<llvm::SmallVector<Value>> results(op->getNumResults());
    for (int64_t i = 0; i < n; ++i) {
      mlir::IRMapping mapping;
      for (Value v : op->getOperands()) {
        if (mlir::isa<mlir::VectorType>(v.getType())) {
          mapping.map(v, mlir::vector::ExtractOp::create(b, loc, v, i));
        }
      }
      Operation* lane = b.clone(*op, mapping);
      for (auto [r, res] : llvm::enumerate(lane->getResults())) {
        auto rvt = mlir::cast<mlir::VectorType>(res.getType());
        res.setType(rvt.getElementType());
        results[r].push_back(res);
      }
    }
    for (auto [r, res] : llvm::enumerate(op->getResults())) {
      Value v = mlir::vector::FromElementsOp::create(
          b, loc, mlir::cast<mlir::VectorType>(res.getType()), results[r]);
      res.replaceAllUsesWith(v);
    }
    op->erase();
  }
  return absl::OkStatus();
}

struct SharedArray {
  std::string name;    // wrapper-local array name
  int64_t bytes = 0;   // size in bytes
};

struct ConstantGlobal {
  std::string msl_name;
  std::string definition;  // program-scope MSL definition
};

struct KernelInfo {
  std::string kernel_name;
  int max_threads_per_threadgroup = 0;  // 0: no attribute
  std::string body_name;
  int num_buffer_args = 0;
  std::vector<SharedArray> shared;
  llvm::StringMap<ConstantGlobal> constants;  // keyed by LLVM global symbol
  llvm::StringMap<unsigned> global_address_space;
};

absl::StatusOr<std::string> DefineConstantGlobal(ml::GlobalOp g,
                                                 const std::string& name) {
  auto dense = mlir::dyn_cast_or_null<mlir::DenseElementsAttr>(g.getValueOrNull());
  if (!dense) return Unimplemented(g, "global initializer (need dense elements)");
  Type et = dense.getElementType();
  if (!et.isIntOrFloat()) return Unimplemented(g, "global element type");
  int bits = et.getIntOrFloatBitWidth();
  int storage_bits = bits <= 8 ? 8 : bits <= 16 ? 16 : bits <= 32 ? 32 : 64;
  if (bits > 64) return Unimplemented(g, "global element width");
  std::vector<std::string> elems;
  int64_t n = dense.getNumElements();
  elems.reserve(n);
  auto push = [&](const llvm::APInt& v) {
    llvm::SmallString<32> s;
    v.zextOrTrunc(storage_bits).toStringUnsigned(s, 10);
    elems.push_back(absl::StrCat(s.str().str(), storage_bits == 64 ? "ul" : "u"));
  };
  if (mlir::isa<mlir::FloatType>(et)) {
    for (const llvm::APFloat& f : dense.getValues<llvm::APFloat>()) {
      push(f.bitcastToAPInt());
    }
  } else if (bits == 1) {
    for (bool v : dense.getValues<bool>()) push(llvm::APInt(8, v ? 1 : 0));
  } else {
    for (const llvm::APInt& v : dense.getValues<llvm::APInt>()) push(v);
  }
  if (elems.empty()) elems.push_back("0u");
  return absl::StrCat("constant uint", storage_bits, "_t ", name, "[",
                      std::max<int64_t>(n, 1), "] = {",
                      absl::StrJoin(elems, ", "), "};\n");
}

// Renames functions, collects globals, turns gpu ids and shared memory into
// extra entry arguments.
absl::Status PrepareFunctions(ModuleOp module, mlir::func::FuncOp& entry,
                              KernelInfo& info) {
  mlir::MLIRContext* ctx = module.getContext();
  if (entry.getNumResults() != 0) {
    return Unimplemented(entry, "entry function with results");
  }
  for (Type t : entry.getArgumentTypes()) {
    if (!mlir::isa<ml::LLVMPointerType>(t)) {
      return Unimplemented(entry, "non-pointer entry argument");
    }
  }
  info.num_buffer_args = entry.getNumArguments();

  // Rename functions to avoid clashes with MSL builtins / other kernels.
  llvm::SmallVector<mlir::func::FuncOp> funcs;
  module.walk([&](mlir::func::FuncOp f) { funcs.push_back(f); });
  for (mlir::func::FuncOp f : funcs) {
    std::string new_name = f == entry
                               ? info.body_name
                               : absl::StrCat(info.kernel_name, "_fn_",
                                              f.getSymName().str());
    auto new_attr = mlir::StringAttr::get(ctx, new_name);
    if (mlir::failed(mlir::SymbolTable::replaceAllSymbolUses(
            f.getSymNameAttr(), new_attr, module))) {
      return absl::InternalError("MSL emitter: failed to rename function");
    }
    f.setSymName(new_name);
    f.removeArgAttrsAttr();
    f.removeResAttrsAttr();
  }

  // Globals.
  absl::Status status;
  module.walk([&](ml::GlobalOp g) {
    if (!status.ok()) return;
    unsigned as = g.getAddrSpace();
    if (as == 3) {
      info.global_address_space[g.getSymName()] = kMslThreadgroup;
    } else if (g.getConstant()) {
      info.global_address_space[g.getSymName()] = kMslConstant;
      std::string name =
          absl::StrCat(info.kernel_name, "_", g.getSymName().str());
      absl::StatusOr<std::string> def = DefineConstantGlobal(g, name);
      if (!def.ok()) {
        status = def.status();
        return;
      }
      info.constants[g.getSymName()] = ConstantGlobal{name, *def};
    } else {
      status = Unimplemented(g, "mutable global variable");
    }
  });
  if (!status.ok()) return status;

  mlir::Block& body = entry.front();
  Type i32 = mlir::IntegerType::get(ctx, 32);
  unsigned id_base = entry.getNumArguments();
  for (int i = 0; i < kNumIdArgs; ++i) body.addArgument(i32, entry.getLoc());

  // Thread/block ids.
  llvm::SmallVector<Operation*> id_ops;
  module.walk([&](Operation* op) {
    if (mlir::isa<mlir::gpu::ThreadIdOp, mlir::gpu::BlockIdOp,
                  mlir::gpu::BlockDimOp, mlir::gpu::GridDimOp>(op)) {
      id_ops.push_back(op);
    }
  });
  for (Operation* op : id_ops) {
    if (op->getParentOfType<mlir::func::FuncOp>() != entry) {
      return Unimplemented(op, "thread id outside the entry function");
    }
    int group = 0;
    int dim = 0;
    if (auto o = mlir::dyn_cast<mlir::gpu::ThreadIdOp>(op)) {
      group = 0;
      dim = static_cast<int>(o.getDimension());
    } else if (auto o = mlir::dyn_cast<mlir::gpu::BlockIdOp>(op)) {
      group = 1;
      dim = static_cast<int>(o.getDimension());
    } else if (auto o = mlir::dyn_cast<mlir::gpu::BlockDimOp>(op)) {
      group = 2;
      dim = static_cast<int>(o.getDimension());
    } else {
      group = 3;
      dim = static_cast<int>(
          mlir::cast<mlir::gpu::GridDimOp>(op).getDimension());
    }
    OpBuilder b(op);
    Value arg = body.getArgument(id_base + group * 3 + dim);
    Value idx = mlir::arith::IndexCastUIOp::create(
        b, op->getLoc(), b.getIndexType(), arg);
    op->getResult(0).replaceAllUsesWith(idx);
    op->erase();
  }

  // Shared memory: one extra `threadgroup char*` argument per global.
  llvm::StringMap<unsigned> shared_arg;
  llvm::SmallVector<ml::AddressOfOp> addr_ops;
  module.walk([&](ml::AddressOfOp a) { addr_ops.push_back(a); });
  for (ml::AddressOfOp a : addr_ops) {
    auto it = info.global_address_space.find(a.getGlobalName());
    if (it == info.global_address_space.end()) {
      return Unimplemented(a, "address of unknown global");
    }
    if (it->second != kMslThreadgroup) continue;
    if (a->getParentOfType<mlir::func::FuncOp>() != entry) {
      return Unimplemented(a, "shared memory outside the entry function");
    }
    auto [arg_it, inserted] =
        shared_arg.try_emplace(a.getGlobalName(), body.getNumArguments());
    if (inserted) {
      auto g = module.lookupSymbol<ml::GlobalOp>(a.getGlobalName());
      std::optional<int64_t> bytes = g ? ByteSize(g.getGlobalType())
                                       : std::nullopt;
      if (!bytes) return Unimplemented(a, "shared memory type");
      body.addArgument(ml::LLVMPointerType::get(ctx, kMslThreadgroup),
                       a.getLoc());
      info.shared.push_back(SharedArray{
          absl::StrCat("xla_shared_", info.shared.size()), *bytes});
    }
    a.getResult().replaceAllUsesWith(body.getArgument(arg_it->second));
    a->erase();
  }
  entry.setFunctionType(mlir::FunctionType::get(
      ctx, body.getArgumentTypes(), entry.getResultTypes()));
  return absl::OkStatus();
}

// Rejects anything we cannot translate, with a precise error.
absl::Status Validate(ModuleOp module) {
  absl::Status status;
  auto check_type = [&](Operation* op, Type t) -> bool {
    Type elem = t;
    if (auto vt = mlir::dyn_cast<mlir::VectorType>(t)) elem = vt.getElementType();
    if (auto it = mlir::dyn_cast<mlir::IntegerType>(elem)) {
      unsigned w = it.getWidth();
      if (w != 1 && w != 8 && w != 16 && w != 32 && w != 64) {
        status = Unimplemented(op, "sub-byte / odd-width integer type");
        return false;
      }
    }
    if (IsSmallFloat(elem)) {
      // f8/f6/f4 values are carried as raw bytes; only data movement is
      // allowed (ExpandFloatOps turns conversions into integer math).
      llvm::StringRef ns = op->getDialect() ? op->getDialect()->getNamespace()
                                            : llvm::StringRef();
      if ((ns == "arith" || ns == "math") &&
          !mlir::isa<mlir::arith::BitcastOp, mlir::arith::SelectOp,
                     mlir::arith::ConstantOp>(op)) {
        status = Unimplemented(op, "arithmetic on an 8-bit-or-smaller float");
        return false;
      }
    }
    if (t.isF64()) {
      status = Unimplemented(op, "f64 type (Metal has no double precision)");
      return false;
    }
    if (mlir::isa<mlir::ComplexType, mlir::TensorType, mlir::MemRefType>(t)) {
      status = Unimplemented(op, "complex/tensor/memref-typed value");
      return false;
    }
    if (auto vt = mlir::dyn_cast<mlir::VectorType>(t)) {
      if (vt.getRank() != 1 || vt.isScalable()) {
        status = Unimplemented(op, "vector shape");
        return false;
      }
      if (vt.getElementType().isF64()) {
        status = Unimplemented(op, "f64 type (Metal has no double precision)");
        return false;
      }
    }
    return true;
  };
  module.walk([&](Operation* op) -> mlir::WalkResult {
    std::string name = OpName(op);
    if (!SupportedOps().contains(name) && !MathFunctions().contains(name)) {
      status = absl::UnimplementedError(
          absl::StrCat("MSL emitter: unsupported op '", name, "'"));
      return mlir::WalkResult::interrupt();
    }
    for (Type t : op->getOperandTypes()) {
      if (!check_type(op, t)) return mlir::WalkResult::interrupt();
    }
    for (Type t : op->getResultTypes()) {
      if (!check_type(op, t)) return mlir::WalkResult::interrupt();
    }
    for (mlir::Region& r : op->getRegions()) {
      for (mlir::Block& blk : r) {
        for (Type t : blk.getArgumentTypes()) {
          if (!check_type(op, t)) return mlir::WalkResult::interrupt();
        }
      }
    }
    if (auto rmw = mlir::dyn_cast<ml::AtomicRMWOp>(op)) {
      Type t = rmw.getVal().getType();
      switch (rmw.getBinOp()) {
        case ml::AtomicBinOp::fadd:
          if (!t.isF32()) status = Unimplemented(op, "atomic fadd type");
          break;
        case ml::AtomicBinOp::xchg:
          if (!t.isInteger(32) && !t.isF32()) {
            status = Unimplemented(op, "atomic xchg type");
          }
          break;
        case ml::AtomicBinOp::add:
        case ml::AtomicBinOp::sub:
        case ml::AtomicBinOp::_and:
        case ml::AtomicBinOp::_or:
        case ml::AtomicBinOp::_xor:
        case ml::AtomicBinOp::max:
        case ml::AtomicBinOp::min:
        case ml::AtomicBinOp::umax:
        case ml::AtomicBinOp::umin:
          if (!t.isInteger(32)) {
            status = Unimplemented(op, "atomic type (only 32-bit integers)");
          }
          break;
        default:
          status = Unimplemented(op, "atomic binary operation");
      }
    } else if (auto cas = mlir::dyn_cast<ml::AtomicCmpXchgOp>(op)) {
      if (!cas.getVal().getType().isInteger(32)) {
        status = Unimplemented(op, "cmpxchg type (only i32)");
      }
    } else if (auto ev = mlir::dyn_cast<ml::ExtractValueOp>(op)) {
      if (!ev.getContainer().getDefiningOp<ml::AtomicCmpXchgOp>() ||
          ev.getPosition().size() != 1) {
        status = Unimplemented(op, "extractvalue (only of cmpxchg results)");
      }
    } else if (auto alloca = mlir::dyn_cast<ml::AllocaOp>(op)) {
      if (!mlir::getConstantIntValue(alloca.getArraySize()) ||
          !ByteSize(alloca.getElemType())) {
        status = Unimplemented(op, "dynamic or unsized alloca");
      }
    } else if (auto shfl = mlir::dyn_cast<mlir::gpu::ShuffleOp>(op)) {
      Type t = shfl.getValue().getType();
      if (!t.isIntOrFloat() || t.getIntOrFloatBitWidth() > 32) {
        status = Unimplemented(op, "shuffle type (only <= 32-bit scalars)");
      } else if (shfl.getMode() != mlir::gpu::ShuffleMode::XOR) {
        std::optional<int64_t> width =
            mlir::getConstantIntValue(shfl.getWidth());
        if (!width || *width != 32) {
          status = Unimplemented(op, "shuffle width other than 32");
        }
      }
      if (status.ok() && !shfl.getValid().use_empty()) {
        status = Unimplemented(op, "use of shuffle `valid` result");
      }
    } else if (mlir::isa<ml::BitcastOp, mlir::arith::BitcastOp>(op)) {
      Type in = op->getOperand(0).getType();
      Type out = op->getResult(0).getType();
      auto non_native = [](Type t) {
        auto vt = mlir::dyn_cast<mlir::VectorType>(t);
        return vt && !IsNativeVector(vt);
      };
      if (!mlir::isa<ml::LLVMPointerType>(in) &&
          (non_native(in) || non_native(out))) {
        status = Unimplemented(op, "bitcast of a vector wider than 4 lanes");
      }
    } else if (name.rfind("llvm.intr.", 0) == 0 || name.rfind("math.", 0) == 0) {
      if (op->getNumResults() != 1) {
        status = Unimplemented(op, "multi-result math op");
      } else if (mlir::isa<mlir::VectorType>(op->getResult(0).getType())) {
        status = Unimplemented(op, "vector math intrinsic");
      }
    } else if (auto call = mlir::dyn_cast<mlir::func::CallOp>(op)) {
      auto callee = module.lookupSymbol<mlir::func::FuncOp>(call.getCallee());
      if (!callee || callee.isExternal()) {
        status = Unimplemented(
            op, absl::StrCat("call to external function '",
                             call.getCallee().str(), "'"));
      }
    } else if (mlir::isa<ml::ICmpOp>(op)) {
      // Only pointer comparisons survive RewriteLlvmArithmetic.
      status = Unimplemented(op, "pointer comparison");
    }
    return status.ok() ? mlir::WalkResult::advance()
                       : mlir::WalkResult::interrupt();
  });
  return status;
}

// ---------------------------------------------------------------------------
// Pointer address-space inference. LLVM pointers are opaque and generic
// (address space 0); MSL needs an address space on every pointer.
// ---------------------------------------------------------------------------

std::optional<Value> FindPtrToIntSource(Value v, int depth) {
  Operation* def = v.getDefiningOp();
  if (!def) return std::nullopt;
  if (auto p2i = mlir::dyn_cast<ml::PtrToIntOp>(def)) return p2i.getArg();
  if (depth <= 0) return std::nullopt;
  for (Value operand : def->getOperands()) {
    if (std::optional<Value> src = FindPtrToIntSource(operand, depth - 1)) {
      return src;
    }
  }
  return std::nullopt;
}

absl::Status InferAndApplyAddressSpaces(ModuleOp module,
                                        mlir::func::FuncOp entry,
                                        const KernelInfo& info) {
  mlir::MLIRContext* ctx = module.getContext();
  llvm::DenseMap<Value, unsigned> as;
  bool changed = false;
  std::string conflict;
  auto is_ptr = [](Value v) {
    return mlir::isa<ml::LLVMPointerType>(v.getType());
  };
  auto set = [&](Value v, unsigned space) {
    if (!is_ptr(v)) return;
    auto [it, inserted] = as.try_emplace(v, space);
    if (inserted) {
      changed = true;
    } else if (it->second != space && conflict.empty()) {
      std::string s;
      llvm::raw_string_ostream os(s);
      v.print(os);
      conflict = absl::StrCat("pointer used in two address spaces: ", s);
    }
  };
  auto join = [&](Value dst, Value src) {
    auto it = as.find(src);
    if (it != as.end()) set(dst, it->second);
  };
  auto unify = [&](Value a, Value b) {
    join(a, b);
    join(b, a);
  };

  for (mlir::BlockArgument arg : entry.getArguments()) {
    if (!is_ptr(arg)) continue;
    unsigned space = mlir::cast<ml::LLVMPointerType>(arg.getType())
                                 .getAddressSpace() == kMslThreadgroup
                         ? kMslThreadgroup
                         : kMslDevice;
    set(arg, space);
  }

  auto propagate = [&]() {
    module.walk([&](Operation* op) {
      if (auto a = mlir::dyn_cast<ml::AddressOfOp>(op)) {
        auto it = info.global_address_space.find(a.getGlobalName());
        if (it != info.global_address_space.end()) set(a.getResult(), it->second);
      } else if (auto al = mlir::dyn_cast<ml::AllocaOp>(op)) {
        set(al.getResult(), kMslThread);
      } else if (auto gep = mlir::dyn_cast<ml::GEPOp>(op)) {
        unify(gep.getResult(), gep.getBase());
      } else if (mlir::isa<ml::AddrSpaceCastOp, ml::FreezeOp, ml::BitcastOp,
                           mlir::UnrealizedConversionCastOp>(op)) {
        if (op->getNumOperands() == 1 && op->getNumResults() == 1) {
          unify(op->getResult(0), op->getOperand(0));
        }
      } else if (auto sel = mlir::dyn_cast<mlir::arith::SelectOp>(op)) {
        unify(sel.getResult(), sel.getTrueValue());
        unify(sel.getResult(), sel.getFalseValue());
      } else if (auto i2p = mlir::dyn_cast<ml::IntToPtrOp>(op)) {
        if (std::optional<Value> src = FindPtrToIntSource(i2p.getArg(), 8)) {
          join(i2p.getResult(), *src);
        }
      } else if (auto f = mlir::dyn_cast<mlir::scf::ForOp>(op)) {
        auto yield = f.getBody()->getTerminator();
        for (auto [i, init] : llvm::enumerate(f.getInitArgs())) {
          Value iter = f.getRegionIterArgs()[i];
          unify(iter, init);
          unify(iter, yield->getOperand(i));
          unify(f.getResult(i), iter);
        }
      } else if (auto w = mlir::dyn_cast<mlir::scf::WhileOp>(op)) {
        for (auto [i, init] : llvm::enumerate(w.getInits())) {
          unify(w.getBeforeArguments()[i], init);
          unify(w.getBeforeArguments()[i], w.getYieldOp()->getOperand(i));
        }
        mlir::scf::ConditionOp cond = w.getConditionOp();
        for (auto [i, v] : llvm::enumerate(cond.getArgs())) {
          unify(w.getAfterArguments()[i], v);
          unify(w.getResult(i), v);
        }
      } else if (mlir::isa<mlir::scf::IfOp, mlir::scf::IndexSwitchOp>(op)) {
        for (mlir::Region& r : op->getRegions()) {
          if (r.empty()) continue;
          Operation* term = r.front().getTerminator();
          for (auto [i, res] : llvm::enumerate(op->getResults())) {
            unify(res, term->getOperand(i));
          }
        }
      } else if (auto call = mlir::dyn_cast<mlir::func::CallOp>(op)) {
        auto callee = module.lookupSymbol<mlir::func::FuncOp>(call.getCallee());
        if (!callee || callee.isExternal()) return;
        for (auto [i, operand] : llvm::enumerate(call.getOperands())) {
          unify(callee.getArgument(i), operand);
        }
        callee.walk([&](mlir::func::ReturnOp ret) {
          for (auto [i, res] : llvm::enumerate(call.getResults())) {
            unify(res, ret.getOperand(i));
          }
        });
      }
    });
  };

  do {
    changed = false;
    propagate();
  } while (changed);
  if (!conflict.empty()) {
    return absl::UnimplementedError(absl::StrCat("MSL emitter: ", conflict));
  }

  // Every pointer value must have an address space now; retype them.
  absl::Status status;
  auto retype = [&](Value v, Operation* owner) {
    if (!status.ok() || !is_ptr(v)) return;
    auto it = as.find(v);
    if (it == as.end()) {
      status = Unimplemented(owner, "pointer with unknown address space");
      return;
    }
    v.setType(ml::LLVMPointerType::get(ctx, it->second));
  };
  module.walk([&](Operation* op) {
    for (Value r : op->getResults()) retype(r, op);
    for (mlir::Region& region : op->getRegions()) {
      for (mlir::Block& blk : region) {
        for (mlir::BlockArgument a : blk.getArguments()) retype(a, op);
      }
    }
  });
  if (!status.ok()) return status;
  module.walk([&](mlir::func::FuncOp f) {
    if (f.isExternal()) return;
    llvm::SmallVector<Type> results(f.getResultTypes());
    f.walk([&](mlir::func::ReturnOp ret) {
      if (ret->getParentOfType<mlir::func::FuncOp>() != f) return;
      results.assign(ret.getOperandTypes().begin(), ret.getOperandTypes().end());
    });
    f.setFunctionType(mlir::FunctionType::get(
        ctx, f.front().getArgumentTypes(), results));
  });
  return absl::OkStatus();
}

// ---------------------------------------------------------------------------
// EmitC conversion.
// ---------------------------------------------------------------------------

class MslTypeConverter : public mlir::TypeConverter {
 public:
  explicit MslTypeConverter(mlir::MLIRContext* ctx) {
    addConversion([](Type t) -> std::optional<Type> {
      if (mlir::isa<mlir::emitc::OpaqueType, mlir::emitc::PointerType,
                    mlir::emitc::LValueType, mlir::emitc::ArrayType,
                    mlir::emitc::SizeTType, mlir::emitc::SignedSizeTType,
                    mlir::emitc::PtrDiffTType>(t)) {
        return t;
      }
      return std::nullopt;
    });
    addConversion([](mlir::IntegerType t) -> std::optional<Type> {
      switch (t.getWidth()) {
        case 1:
        case 8:
        case 16:
        case 32:
        case 64:
          return t;
        default:
          return std::nullopt;
      }
    });
    addConversion([ctx](mlir::FloatType t) -> std::optional<Type> {
      if (t.isF16() || t.isBF16() || t.isF32()) return t;
      if (t.getWidth() <= 8) return mlir::IntegerType::get(ctx, 8);
      return std::nullopt;
    });
    addConversion([ctx](mlir::VectorType t) -> std::optional<Type> {
      std::optional<std::string> name = MslVectorName(t);
      if (!name) return std::nullopt;
      return mlir::emitc::OpaqueType::get(ctx, *name);
    });
    addConversion([ctx](ml::LLVMPointerType t) -> std::optional<Type> {
      const char* kw = AddressSpaceKeyword(t.getAddressSpace());
      if (!kw) return std::nullopt;
      return mlir::emitc::PointerType::get(
          mlir::emitc::OpaqueType::get(ctx, absl::StrCat(kw, " char")));
    });
    addConversion([ctx](ml::LLVMStructType t) -> std::optional<Type> {
      if (!IsCasStruct(t)) return std::nullopt;
      return mlir::emitc::OpaqueType::get(ctx, "xla_cas_result<int>");
    });
  }
};

// MSL spelling of a converted type, used to value-initialize (`T()`).
std::optional<std::string> ConvertedTypeName(Type t) {
  if (auto o = mlir::dyn_cast<mlir::emitc::OpaqueType>(t)) {
    return o.getValue().str();
  }
  if (mlir::isa<mlir::emitc::SizeTType>(t)) return "size_t";
  return MslScalarName(t);
}

Value CallOpaque(OpBuilder& b, Location loc, Type result,
                 llvm::StringRef callee, ValueRange operands,
                 llvm::ArrayRef<Type> template_types = {}) {
  mlir::ArrayAttr targs;
  if (!template_types.empty()) {
    llvm::SmallVector<mlir::Attribute> attrs;
    for (Type t : template_types) attrs.push_back(mlir::TypeAttr::get(t));
    targs = b.getArrayAttr(attrs);
  }
  llvm::SmallVector<Type, 1> results;
  if (result) results.push_back(result);
  auto call = mlir::emitc::CallOpaqueOp::create(
      b, loc, mlir::TypeRange(results), callee, operands, mlir::ArrayAttr(),
      targs);
  return result ? call.getResult(0) : Value();
}

Value I32Constant(OpBuilder& b, Location loc, int64_t v) {
  return mlir::emitc::ConstantOp::create(b, loc, b.getI32Type(),
                                         b.getI32IntegerAttr(v));
}

// Value-initialized `T()`.
Value ZeroValue(OpBuilder& b, Location loc, Type converted) {
  std::optional<std::string> name = ConvertedTypeName(converted);
  if (!name) return Value();
  return CallOpaque(b, loc, converted, *name, {});
}

struct ConversionState {
  const KernelInfo* info = nullptr;
  int alloca_counter = 0;
};

template <typename OpTy>
class MslPattern : public mlir::OpConversionPattern<OpTy> {
 public:
  MslPattern(const mlir::TypeConverter& tc, mlir::MLIRContext* ctx,
             ConversionState* state, mlir::PatternBenefit benefit = 1)
      : mlir::OpConversionPattern<OpTy>(tc, ctx, benefit), state_(state) {}

 protected:
  Type Convert(Type t) const { return this->getTypeConverter()->convertType(t); }
  ConversionState* state_;
};

// --- LLVM memory ops -------------------------------------------------------

class LoadLowering : public MslPattern<ml::LoadOp> {
 public:
  using MslPattern::MslPattern;
  LogicalResult matchAndRewrite(
      ml::LoadOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    Type ty = Convert(op->getResult(0).getType());
    if (!ty) return rewriter.notifyMatchFailure(op, "type");
    rewriter.replaceOp(op, CallOpaque(rewriter, op.getLoc(), ty, "xla_load",
                                      {adaptor.getAddr()}, {ty}));
    return mlir::success();
  }
};

class StoreLowering : public MslPattern<ml::StoreOp> {
 public:
  using MslPattern::MslPattern;
  LogicalResult matchAndRewrite(
      ml::StoreOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    CallOpaque(rewriter, op.getLoc(), Type(), "xla_store",
               {adaptor.getAddr(), adaptor.getValue()});
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class GEPLowering : public MslPattern<ml::GEPOp> {
 public:
  using MslPattern::MslPattern;
  LogicalResult matchAndRewrite(
      ml::GEPOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    Type i64 = rewriter.getI64Type();
    Type cur = op.getElemType();
    llvm::ArrayRef<int32_t> raw = op.getRawConstantIndices();
    ValueRange dynamic = adaptor.getDynamicIndices();
    size_t dyn_pos = 0;
    int64_t const_offset = 0;
    Value offset;
    auto add_term = [&](Value term) {
      offset = offset ? mlir::emitc::AddOp::create(rewriter, loc, i64, offset,
                                                   term)
                            .getResult()
                      : term;
    };
    for (size_t k = 0; k < raw.size(); ++k) {
      bool is_dynamic = raw[k] == ml::GEPOp::kDynamicIndex;
      int64_t stride = 0;
      if (k > 0) {
        if (auto st = mlir::dyn_cast<ml::LLVMStructType>(cur)) {
          if (is_dynamic) {
            return rewriter.notifyMatchFailure(op, "dynamic struct index");
          }
          std::optional<int64_t> off = StructFieldOffset(st, raw[k]);
          if (!off) return rewriter.notifyMatchFailure(op, "struct layout");
          const_offset += *off;
          cur = st.getBody()[raw[k]];
          continue;
        }
        if (auto at = mlir::dyn_cast<ml::LLVMArrayType>(cur)) {
          cur = at.getElementType();
        } else if (auto vt = mlir::dyn_cast<mlir::VectorType>(cur)) {
          cur = vt.getElementType();
        } else {
          return rewriter.notifyMatchFailure(op, "cannot index into type");
        }
      }
      std::optional<int64_t> size = ByteSize(cur);
      if (!size) return rewriter.notifyMatchFailure(op, "element size");
      stride = *size;
      if (!is_dynamic) {
        const_offset += static_cast<int64_t>(raw[k]) * stride;
        continue;
      }
      Value idx = dynamic[dyn_pos++];
      if (idx.getType() != i64) {
        idx = mlir::emitc::CastOp::create(rewriter, loc, i64, idx);
      }
      if (stride != 1) {
        Value s = mlir::emitc::ConstantOp::create(
            rewriter, loc, i64, rewriter.getI64IntegerAttr(stride));
        idx = mlir::emitc::MulOp::create(rewriter, loc, i64, idx, s);
      }
      add_term(idx);
    }
    if (const_offset != 0) {
      add_term(mlir::emitc::ConstantOp::create(
          rewriter, loc, i64, rewriter.getI64IntegerAttr(const_offset)));
    }
    if (!offset) {
      rewriter.replaceOp(op, adaptor.getBase());
      return mlir::success();
    }
    Type ty = Convert(op->getResult(0).getType());
    if (!ty) return rewriter.notifyMatchFailure(op, "type");
    rewriter.replaceOp(op, CallOpaque(rewriter, loc, ty, "xla_gep",
                                      {adaptor.getBase(), offset}));
    return mlir::success();
  }
};

class PtrToIntLowering : public MslPattern<ml::PtrToIntOp> {
 public:
  using MslPattern::MslPattern;
  LogicalResult matchAndRewrite(
      ml::PtrToIntOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    Type ty = Convert(op->getResult(0).getType());
    if (!ty) return rewriter.notifyMatchFailure(op, "type");
    rewriter.replaceOp(op, CallOpaque(rewriter, op.getLoc(), ty,
                                      "xla_ptrtoint", {adaptor.getArg()}, {ty}));
    return mlir::success();
  }
};

class IntToPtrLowering : public MslPattern<ml::IntToPtrOp> {
 public:
  using MslPattern::MslPattern;
  LogicalResult matchAndRewrite(
      ml::IntToPtrOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    Type ty = Convert(op->getResult(0).getType());
    const char* kw = AddressSpaceKeyword(
        mlir::cast<ml::LLVMPointerType>(op->getResult(0).getType())
            .getAddressSpace());
    if (!ty || !kw) return rewriter.notifyMatchFailure(op, "type");
    rewriter.replaceOp(
        op, CallOpaque(rewriter, op.getLoc(), ty,
                       absl::StrCat("xla_inttoptr_", kw), {adaptor.getArg()}));
    return mlir::success();
  }
};

// addrspacecast / freeze / pointer bitcast: identity after retyping.
template <typename OpTy>
class PassThroughLowering : public MslPattern<OpTy> {
 public:
  using MslPattern<OpTy>::MslPattern;
  LogicalResult matchAndRewrite(
      OpTy op, typename OpTy::Adaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    Value in = adaptor.getOperands()[0];
    Type ty = this->Convert(op->getResult(0).getType());
    if (!ty || ty != in.getType()) {
      return rewriter.notifyMatchFailure(op, "not an identity");
    }
    rewriter.replaceOp(op, in);
    return mlir::success();
  }
};

class AddressOfLowering : public MslPattern<ml::AddressOfOp> {
 public:
  using MslPattern::MslPattern;
  LogicalResult matchAndRewrite(
      ml::AddressOfOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    auto it = state_->info->constants.find(op.getGlobalName());
    if (it == state_->info->constants.end()) {
      return rewriter.notifyMatchFailure(op, "not a constant global");
    }
    Type ty = Convert(op->getResult(0).getType());
    if (!ty) return rewriter.notifyMatchFailure(op, "type");
    rewriter.replaceOpWithNewOp<mlir::emitc::LiteralOp>(
        op, ty, absl::StrCat("((constant char*)", it->second.msl_name, ")"));
    return mlir::success();
  }
};

class AllocaLowering : public MslPattern<ml::AllocaOp> {
 public:
  using MslPattern::MslPattern;
  LogicalResult matchAndRewrite(
      ml::AllocaOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    std::optional<int64_t> count = mlir::getConstantIntValue(op.getArraySize());
    std::optional<int64_t> size = ByteSize(op.getElemType());
    Type ty = Convert(op->getResult(0).getType());
    if (!count || !size || !ty) return rewriter.notifyMatchFailure(op, "alloca");
    int64_t chunks = std::max<int64_t>(1, (*count * *size + 15) / 16);
    std::string name = absl::StrCat("xla_alloca_", state_->alloca_counter++);
    // An LLVM alloca lives until the function returns; a C array declared
    // in a loop or branch body would end with that block, and a pointer to
    // it may be yielded out. Declare it at the top of the function.
    auto fn = op->getParentOfType<mlir::FunctionOpInterface>();
    if (!fn) return rewriter.notifyMatchFailure(op, "alloca outside a function");
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&fn.getFunctionBody().front());
      mlir::emitc::VerbatimOp::create(
          rewriter, op.getLoc(),
          absl::StrCat("thread uint4 ", name, "[", chunks, "];"));
    }
    rewriter.replaceOpWithNewOp<mlir::emitc::LiteralOp>(
        op, ty, absl::StrCat("((thread char*)", name, ")"));
    return mlir::success();
  }
};

template <typename OpTy>
class UndefLowering : public MslPattern<OpTy> {
 public:
  using MslPattern<OpTy>::MslPattern;
  LogicalResult matchAndRewrite(
      OpTy op, typename OpTy::Adaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    Type ty = this->Convert(op->getResult(0).getType());
    Value zero = ty ? ZeroValue(rewriter, op->getLoc(), ty) : Value();
    if (!zero) return rewriter.notifyMatchFailure(op, "type");
    rewriter.replaceOp(op, zero);
    return mlir::success();
  }
};

class AtomicRMWLowering : public MslPattern<ml::AtomicRMWOp> {
 public:
  using MslPattern::MslPattern;
  LogicalResult matchAndRewrite(
      ml::AtomicRMWOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    const char* fn = nullptr;
    switch (op.getBinOp()) {
      case ml::AtomicBinOp::add:
        fn = "xla_atomic_add";
        break;
      case ml::AtomicBinOp::fadd:
        fn = "xla_atomic_fadd";
        break;
      case ml::AtomicBinOp::sub:
        fn = "xla_atomic_sub";
        break;
      case ml::AtomicBinOp::_and:
        fn = "xla_atomic_andi";
        break;
      case ml::AtomicBinOp::_or:
        fn = "xla_atomic_ori";
        break;
      case ml::AtomicBinOp::_xor:
        fn = "xla_atomic_xori";
        break;
      case ml::AtomicBinOp::max:
        fn = "xla_atomic_max";
        break;
      case ml::AtomicBinOp::min:
        fn = "xla_atomic_min";
        break;
      case ml::AtomicBinOp::umax:
        fn = "xla_atomic_umax";
        break;
      case ml::AtomicBinOp::umin:
        fn = "xla_atomic_umin";
        break;
      case ml::AtomicBinOp::xchg:
        fn = "xla_atomic_xchg";
        break;
      default:
        return rewriter.notifyMatchFailure(op, "unsupported atomic");
    }
    Type ty = Convert(op->getResult(0).getType());
    if (!ty) return rewriter.notifyMatchFailure(op, "type");
    rewriter.replaceOp(op, CallOpaque(rewriter, op.getLoc(), ty, fn,
                                      {adaptor.getPtr(), adaptor.getVal()}));
    return mlir::success();
  }
};

class CmpXchgLowering : public MslPattern<ml::AtomicCmpXchgOp> {
 public:
  using MslPattern::MslPattern;
  LogicalResult matchAndRewrite(
      ml::AtomicCmpXchgOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    Type ty = Convert(op->getResult(0).getType());
    if (!ty) return rewriter.notifyMatchFailure(op, "type");
    rewriter.replaceOp(
        op, CallOpaque(rewriter, op.getLoc(), ty, "xla_cmpxchg",
                       {adaptor.getPtr(), adaptor.getCmp(), adaptor.getVal()}));
    return mlir::success();
  }
};

// scf.while -> emitc.do. This replaces upstream's WhileLowering
// (SCFToEmitC.cpp), which only writes the loop-carried variables back in the
// lowering of the after-region's scf.yield and skips that step entirely when
// the after region is a bare forwarding yield. That is exactly the shape of
// XLA's compare-and-swap loops (float scatter min/max, atomic RMW emulation):
// the "expected" value was never refreshed, so a thread that lost the race
// retried with a stale value forever and tripped the GPU watchdog. This
// version always lowers the after region and its yield.
class WhileLowering : public mlir::OpConversionPattern<mlir::scf::WhileOp> {
 public:
  WhileLowering(const mlir::TypeConverter& tc, mlir::MLIRContext* ctx)
      : OpConversionPattern(tc, ctx, /*benefit=*/2) {}

  LogicalResult matchAndRewrite(
      mlir::scf::WhileOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    namespace emitc = mlir::emitc;
    mlir::Location loc = op.getLoc();
    mlir::MLIRContext* ctx = loc.getContext();
    emitc::OpaqueAttr no_init = emitc::OpaqueAttr::get(ctx, "");
    auto make_var = [&](Type t) -> Value {
      return emitc::VariableOp::create(rewriter, loc,
                                       emitc::LValueType::get(t), no_init);
    };
    auto load = [&](Value var) -> Value {
      Type t = mlir::cast<emitc::LValueType>(var.getType()).getValueType();
      return emitc::LoadOp::create(rewriter, loc, t, var).getResult();
    };

    // Result variables (one per while result) and loop-carried variables (one
    // per init), the latter assigned from the inits before the loop.
    llvm::SmallVector<Value> result_vars, loop_vars;
    for (mlir::OpResult r : op.getResults()) {
      Type t = getTypeConverter()->convertType(r.getType());
      if (!t || mlir::isa<emitc::ArrayType>(t)) {
        return rewriter.notifyMatchFailure(op, "result type");
      }
      result_vars.push_back(make_var(t));
    }
    for (Value init : adaptor.getInits()) {
      Type t = init.getType();
      if (mlir::isa<emitc::ArrayType>(t)) {
        return rewriter.notifyMatchFailure(op, "array loop variable");
      }
      Value var = make_var(t);
      emitc::AssignOp::create(rewriter, loc, var, init);
      loop_vars.push_back(var);
    }
    Type i1 = rewriter.getI1Type();
    Value cond_var = make_var(i1);

    auto do_op = emitc::DoOp::create(rewriter, loc);
    if (failed(rewriter.convertRegionTypes(&op.getBefore(),
                                           *getTypeConverter(), nullptr)) ||
        failed(rewriter.convertRegionTypes(&op.getAfter(),
                                           *getTypeConverter(), nullptr))) {
      return rewriter.notifyMatchFailure(op, "region types");
    }

    // Body: load loop vars, run the before region, store condition args into
    // the result vars, then (if continuing) run the after region and store its
    // yield back into the loop vars.
    mlir::Block* body = rewriter.createBlock(&do_op.getBodyRegion());
    rewriter.setInsertionPointToStart(body);
    llvm::SmallVector<Value> before_args;
    for (Value v : loop_vars) before_args.push_back(load(v));
    rewriter.mergeBlocks(&op.getBefore().front(), body, before_args);

    auto cond_op = mlir::cast<mlir::scf::ConditionOp>(body->getTerminator());
    rewriter.setInsertionPoint(cond_op);
    llvm::SmallVector<Value> cond_args;
    for (Value a : cond_op.getArgs()) {
      cond_args.push_back(rewriter.getRemappedValue(a));
    }
    for (auto [v, var] : llvm::zip(cond_args, result_vars)) {
      emitc::AssignOp::create(rewriter, loc, var, v);
    }
    Value cond = rewriter.getRemappedValue(cond_op.getCondition());
    emitc::AssignOp::create(rewriter, loc, cond_var, cond);

    auto if_op = emitc::IfOp::create(rewriter, loc, cond, false, false);
    mlir::Block* if_body = rewriter.createBlock(&if_op.getBodyRegion());
    rewriter.mergeBlocks(&op.getAfter().front(), if_body, cond_args);
    auto yield = mlir::cast<mlir::scf::YieldOp>(if_body->getTerminator());
    rewriter.setInsertionPoint(yield);
    llvm::SmallVector<Value> yielded;
    if (failed(rewriter.getRemappedValues(yield.getOperands(), yielded))) {
      return rewriter.notifyMatchFailure(op, "yield operands");
    }
    for (auto [v, var] : llvm::zip(yielded, loop_vars)) {
      emitc::AssignOp::create(rewriter, loc, var, v);
    }
    emitc::YieldOp::create(rewriter, loc);
    rewriter.eraseOp(yield);
    rewriter.eraseOp(cond_op);

    // Condition region: an expression that loads the flag.
    mlir::Block* cond_block = rewriter.createBlock(&do_op.getConditionRegion());
    rewriter.setInsertionPointToStart(cond_block);
    auto expr = emitc::ExpressionOp::create(rewriter, loc, i1, cond_var,
                                            /*do_not_inline=*/false);
    mlir::Block* expr_block = rewriter.createBlock(&expr.getBodyRegion());
    expr_block->addArgument(cond_var.getType(), loc);
    rewriter.setInsertionPointToStart(expr_block);
    Value flag = emitc::LoadOp::create(rewriter, loc, i1,
                                       expr_block->getArgument(0));
    emitc::YieldOp::create(rewriter, loc, flag);
    rewriter.setInsertionPointToEnd(cond_block);
    emitc::YieldOp::create(rewriter, loc, expr);

    rewriter.setInsertionPointAfter(op);
    llvm::SmallVector<Value> results;
    for (Value v : result_vars) results.push_back(load(v));
    rewriter.replaceOp(op, results);
    return mlir::success();
  }
};

class ExtractValueLowering : public MslPattern<ml::ExtractValueOp> {
 public:
  using MslPattern::MslPattern;
  LogicalResult matchAndRewrite(
      ml::ExtractValueOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    if (!IsCasStruct(op.getContainer().getType()) ||
        op.getPosition().size() != 1) {
      return rewriter.notifyMatchFailure(op, "not a cmpxchg result");
    }
    Type ty = Convert(op->getResult(0).getType());
    if (!ty) return rewriter.notifyMatchFailure(op, "type");
    const char* fn = op.getPosition()[0] == 0 ? "xla_cas_value" : "xla_cas_ok";
    rewriter.replaceOp(op, CallOpaque(rewriter, op.getLoc(), ty, fn,
                                      {adaptor.getContainer()}));
    return mlir::success();
  }
};

// llvm.bitcast / arith.bitcast (non-pointer).
template <typename OpTy>
class BitcastLowering : public MslPattern<OpTy> {
 public:
  using MslPattern<OpTy>::MslPattern;
  LogicalResult matchAndRewrite(
      OpTy op, typename OpTy::Adaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    Value in = adaptor.getOperands()[0];
    Type ty = this->Convert(op->getResult(0).getType());
    if (!ty) return rewriter.notifyMatchFailure(op, "type");
    if (ty == in.getType()) {
      rewriter.replaceOp(op, in);
      return mlir::success();
    }
    rewriter.replaceOp(
        op, CallOpaque(rewriter, op->getLoc(), ty, "as_type", {in}, {ty}));
    return mlir::success();
  }
};

// --- Vectors ---------------------------------------------------------------

Value VExtract(OpBuilder& b, Location loc, Type elem, Value vec, Value idx) {
  return CallOpaque(b, loc, elem, "xla_vext", {vec, idx}, {elem});
}
Value VInsert(OpBuilder& b, Location loc, Value vec, Value scalar, Value idx) {
  return CallOpaque(b, loc, vec.getType(), "xla_vins", {vec, scalar, idx});
}

class VectorExtractLowering : public MslPattern<mlir::vector::ExtractOp> {
 public:
  using MslPattern::MslPattern;
  LogicalResult matchAndRewrite(
      mlir::vector::ExtractOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    llvm::ArrayRef<int64_t> pos = op.getStaticPosition();
    if (pos.empty()) {
      rewriter.replaceOp(op, adaptor.getSource());
      return mlir::success();
    }
    if (pos.size() != 1) return rewriter.notifyMatchFailure(op, "rank");
    Value idx = pos[0] == mlir::ShapedType::kDynamic
                    ? adaptor.getDynamicPosition()[0]
                    : I32Constant(rewriter, op.getLoc(), pos[0]);
    Type ty = Convert(op->getResult(0).getType());
    if (!ty) return rewriter.notifyMatchFailure(op, "type");
    rewriter.replaceOp(
        op, VExtract(rewriter, op.getLoc(), ty, adaptor.getSource(), idx));
    return mlir::success();
  }
};

class VectorInsertLowering : public MslPattern<mlir::vector::InsertOp> {
 public:
  using MslPattern::MslPattern;
  LogicalResult matchAndRewrite(
      mlir::vector::InsertOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    llvm::ArrayRef<int64_t> pos = op.getStaticPosition();
    if (pos.empty()) {
      rewriter.replaceOp(op, adaptor.getValueToStore());
      return mlir::success();
    }
    if (pos.size() != 1) return rewriter.notifyMatchFailure(op, "rank");
    Value idx = pos[0] == mlir::ShapedType::kDynamic
                    ? adaptor.getDynamicPosition()[0]
                    : I32Constant(rewriter, op.getLoc(), pos[0]);
    rewriter.replaceOp(op, VInsert(rewriter, op.getLoc(), adaptor.getDest(),
                                   adaptor.getValueToStore(), idx));
    return mlir::success();
  }
};

class ExtractElementLowering : public MslPattern<ml::ExtractElementOp> {
 public:
  using MslPattern::MslPattern;
  LogicalResult matchAndRewrite(
      ml::ExtractElementOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    Type ty = Convert(op->getResult(0).getType());
    if (!ty) return rewriter.notifyMatchFailure(op, "type");
    rewriter.replaceOp(op, VExtract(rewriter, op.getLoc(), ty,
                                    adaptor.getVector(), adaptor.getPosition()));
    return mlir::success();
  }
};

class InsertElementLowering : public MslPattern<ml::InsertElementOp> {
 public:
  using MslPattern::MslPattern;
  LogicalResult matchAndRewrite(
      ml::InsertElementOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOp(op, VInsert(rewriter, op.getLoc(), adaptor.getVector(),
                                   adaptor.getValue(), adaptor.getPosition()));
    return mlir::success();
  }
};

class FromElementsLowering : public MslPattern<mlir::vector::FromElementsOp> {
 public:
  using MslPattern::MslPattern;
  LogicalResult matchAndRewrite(
      mlir::vector::FromElementsOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    Type ty = Convert(op->getResult(0).getType());
    Value v = ty ? ZeroValue(rewriter, op.getLoc(), ty) : Value();
    if (!v) return rewriter.notifyMatchFailure(op, "type");
    for (auto [i, e] : llvm::enumerate(adaptor.getElements())) {
      v = VInsert(rewriter, op.getLoc(), v, e,
                  I32Constant(rewriter, op.getLoc(), i));
    }
    rewriter.replaceOp(op, v);
    return mlir::success();
  }
};

class BroadcastLowering : public MslPattern<mlir::vector::BroadcastOp> {
 public:
  using MslPattern::MslPattern;
  LogicalResult matchAndRewrite(
      mlir::vector::BroadcastOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    auto vt = mlir::cast<mlir::VectorType>(op->getResult(0).getType());
    Type ty = Convert(vt);
    if (!ty) return rewriter.notifyMatchFailure(op, "type");
    if (mlir::isa<mlir::VectorType>(op.getSource().getType())) {
      if (op.getSource().getType() != vt) {
        return rewriter.notifyMatchFailure(op, "vector-to-vector broadcast");
      }
      rewriter.replaceOp(op, adaptor.getSource());
      return mlir::success();
    }
    Value v = ZeroValue(rewriter, op.getLoc(), ty);
    if (!v) return rewriter.notifyMatchFailure(op, "type");
    for (int64_t i = 0; i < vt.getNumElements(); ++i) {
      v = VInsert(rewriter, op.getLoc(), v, adaptor.getSource(),
                  I32Constant(rewriter, op.getLoc(), i));
    }
    rewriter.replaceOp(op, v);
    return mlir::success();
  }
};

int64_t IntAt(mlir::ArrayAttr a, int i) {
  return mlir::cast<mlir::IntegerAttr>(a[i]).getInt();
}

class ExtractStridedSliceLowering
    : public MslPattern<mlir::vector::ExtractStridedSliceOp> {
 public:
  using MslPattern::MslPattern;
  LogicalResult matchAndRewrite(
      mlir::vector::ExtractStridedSliceOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    auto vt = mlir::cast<mlir::VectorType>(op->getResult(0).getType());
    Type ty = Convert(vt);
    Type elem = Convert(vt.getElementType());
    if (!ty || !elem || IntAt(op.getStrides(), 0) != 1) {
      return rewriter.notifyMatchFailure(op, "unsupported");
    }
    Location loc = op.getLoc();
    int64_t off = IntAt(op.getOffsets(), 0);
    Value v = ZeroValue(rewriter, loc, ty);
    for (int64_t j = 0; j < IntAt(op.getSizes(), 0); ++j) {
      Value e = VExtract(rewriter, loc, elem, adaptor.getSource(),
                         I32Constant(rewriter, loc, off + j));
      v = VInsert(rewriter, loc, v, e, I32Constant(rewriter, loc, j));
    }
    rewriter.replaceOp(op, v);
    return mlir::success();
  }
};

class InsertStridedSliceLowering
    : public MslPattern<mlir::vector::InsertStridedSliceOp> {
 public:
  using MslPattern::MslPattern;
  LogicalResult matchAndRewrite(
      mlir::vector::InsertStridedSliceOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    auto src_vt = mlir::cast<mlir::VectorType>(op.getValueToStore().getType());
    Type elem = Convert(src_vt.getElementType());
    if (!elem || IntAt(op.getStrides(), 0) != 1) {
      return rewriter.notifyMatchFailure(op, "unsupported");
    }
    Location loc = op.getLoc();
    int64_t off = IntAt(op.getOffsets(), 0);
    Value v = adaptor.getDest();
    for (int64_t j = 0; j < src_vt.getNumElements(); ++j) {
      Value e = VExtract(rewriter, loc, elem, adaptor.getValueToStore(),
                         I32Constant(rewriter, loc, j));
      v = VInsert(rewriter, loc, v, e, I32Constant(rewriter, loc, off + j));
    }
    rewriter.replaceOp(op, v);
    return mlir::success();
  }
};

// --- GPU -------------------------------------------------------------------

class BarrierLowering : public MslPattern<mlir::gpu::BarrierOp> {
 public:
  using MslPattern::MslPattern;
  LogicalResult matchAndRewrite(
      mlir::gpu::BarrierOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    CallOpaque(rewriter, op.getLoc(), Type(), "xla_barrier", {});
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class ShuffleLowering : public MslPattern<mlir::gpu::ShuffleOp> {
 public:
  using MslPattern::MslPattern;
  LogicalResult matchAndRewrite(
      mlir::gpu::ShuffleOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    const char* fn = nullptr;
    switch (op.getMode()) {
      case mlir::gpu::ShuffleMode::XOR:
        fn = "xla_shfl_xor";
        break;
      case mlir::gpu::ShuffleMode::DOWN:
        fn = "xla_shfl_down";
        break;
      case mlir::gpu::ShuffleMode::UP:
        fn = "xla_shfl_up";
        break;
      case mlir::gpu::ShuffleMode::IDX:
        fn = "xla_shfl_idx";
        break;
    }
    Type ty = Convert(op.getShuffleResult().getType());
    if (!ty || !fn) return rewriter.notifyMatchFailure(op, "shuffle");
    Location loc = op.getLoc();
    Value r = CallOpaque(rewriter, loc, ty, fn,
                         {adaptor.getValue(), adaptor.getOffset()});
    Value valid = mlir::emitc::ConstantOp::create(rewriter, loc,
                                                  rewriter.getI1Type(),
                                                  rewriter.getBoolAttr(true));
    rewriter.replaceOp(op, {r, valid});
    return mlir::success();
  }
};

// --- Arith / math ----------------------------------------------------------

// f16/bf16 literals have no MSL suffix EmitC could print; go through f32.
class SmallFloatConstantLowering : public MslPattern<mlir::arith::ConstantOp> {
 public:
  using MslPattern::MslPattern;
  LogicalResult matchAndRewrite(
      mlir::arith::ConstantOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    Type t = op.getType();
    auto fa = mlir::dyn_cast<mlir::FloatAttr>(op.getValue());
    if (!fa) return rewriter.notifyMatchFailure(op, "not a float");
    Location loc = op.getLoc();
    if (IsSmallFloat(t)) {
      // f8 and friends are raw bytes.
      llvm::APInt bits = fa.getValue().bitcastToAPInt().zextOrTrunc(8);
      rewriter.replaceOpWithNewOp<mlir::emitc::ConstantOp>(
          op, rewriter.getI8Type(),
          rewriter.getIntegerAttr(rewriter.getI8Type(), bits));
      return mlir::success();
    }
    if (!t.isF16() && !t.isBF16()) {
      return rewriter.notifyMatchFailure(op, "not f16/bf16");
    }
    llvm::APFloat v = fa.getValue();
    bool loses_info = false;
    v.convert(llvm::APFloat::IEEEsingle(), llvm::APFloat::rmNearestTiesToEven,
              &loses_info);
    Value f32 = mlir::emitc::ConstantOp::create(
        rewriter, loc, rewriter.getF32Type(),
        rewriter.getFloatAttr(rewriter.getF32Type(), v));
    rewriter.replaceOpWithNewOp<mlir::emitc::CastOp>(op, t, f32);
    return mlir::success();
  }
};

class MathCallLowering : public mlir::ConversionPattern {
 public:
  MathCallLowering(const mlir::TypeConverter& tc, mlir::MLIRContext* ctx)
      : mlir::ConversionPattern(tc, mlir::Pattern::MatchAnyOpTypeTag(),
                                /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(
      Operation* op, llvm::ArrayRef<Value> operands,
      mlir::ConversionPatternRewriter& rewriter) const override {
    auto it = MathFunctions().find(op->getName().getStringRef());
    if (it == MathFunctions().end() || op->getNumResults() != 1) {
      return mlir::failure();
    }
    Location loc = op->getLoc();
    Type f32 = rewriter.getF32Type();
    llvm::SmallVector<Value> args;
    for (auto [orig, conv] : llvm::zip(op->getOperands(), operands)) {
      Value a = conv;
      if (orig.getType().isBF16()) {
        a = mlir::emitc::CastOp::create(rewriter, loc, f32, a);
      }
      args.push_back(a);
    }
    Type res_ty = getTypeConverter()->convertType(op->getResult(0).getType());
    if (!res_ty) return rewriter.notifyMatchFailure(op, "type");
    Type call_ty = res_ty.isBF16() ? f32 : res_ty;
    Value r = CallOpaque(rewriter, loc, call_ty, it->second, args);
    if (call_ty != res_ty) {
      r = mlir::emitc::CastOp::create(rewriter, loc, res_ty, r);
    }
    rewriter.replaceOp(op, r);
    return mlir::success();
  }
};

// ---------------------------------------------------------------------------
// Printing.
// ---------------------------------------------------------------------------

absl::Status TopologicalFunctionOrder(
    ModuleOp module, llvm::StringRef entry_name,
    llvm::SmallVector<mlir::emitc::FuncOp>& order) {
  llvm::StringMap<mlir::emitc::FuncOp> funcs;
  module.walk([&](mlir::emitc::FuncOp f) { funcs[f.getSymName()] = f; });
  llvm::StringSet<> done;
  llvm::StringSet<> active;
  std::function<absl::Status(mlir::emitc::FuncOp)> visit =
      [&](mlir::emitc::FuncOp f) -> absl::Status {
    if (done.contains(f.getSymName())) return absl::OkStatus();
    if (!active.insert(f.getSymName()).second) {
      return absl::UnimplementedError(
          "MSL emitter: recursive functions are not supported by Metal");
    }
    absl::Status status;
    f.walk([&](mlir::emitc::CallOp call) {
      if (!status.ok()) return;
      auto it = funcs.find(call.getCallee());
      if (it != funcs.end()) status = visit(it->second);
    });
    if (!status.ok()) return status;
    active.erase(f.getSymName());
    done.insert(f.getSymName());
    order.push_back(f);
    return absl::OkStatus();
  };
  auto entry = funcs.find(entry_name);
  if (entry == funcs.end()) {
    return absl::InternalError("MSL emitter: entry function lost in conversion");
  }
  return visit(entry->second);
}

std::string KernelWrapper(const KernelInfo& info) {
  // Metal's argument table has 31 buffer slots. Kernels with more buffer
  // arguments take them through an argument buffer instead: [[buffer(0)]] is
  // an array of 64-bit GPU addresses (MTLBuffer.gpuAddress + offset), one per
  // argument, and the runtime makes the buffers resident with useResource.
  // The marker comment tells the loader (MetalExecutor::LoadKernel) which
  // convention the kernel uses.
  const bool arg_buffer = info.num_buffer_args > kMaxDirectBufferArgs;
  std::string s;
  // Lets the Metal compiler allocate registers for the actual threadgroup
  // size instead of the device maximum.
  if (info.max_threads_per_threadgroup > 0) {
    absl::StrAppend(&s, "[[max_total_threads_per_threadgroup(",
                    info.max_threads_per_threadgroup, ")]]\n");
  }
  if (arg_buffer) absl::StrAppend(&s, kArgumentBufferMarker, "\n");
  absl::StrAppend(&s, "kernel void ", info.kernel_name, "(\n");
  if (arg_buffer) {
    absl::StrAppend(&s, "    constant ulong* xla_args [[buffer(0)]],\n");
  } else {
    for (int i = 0; i < info.num_buffer_args; ++i) {
      absl::StrAppend(&s, "    device char* xla_arg", i, " [[buffer(", i,
                      ")]],\n");
    }
  }
  absl::StrAppend(&s,
                  "    uint3 xla_tid [[thread_position_in_threadgroup]],\n"
                  "    uint3 xla_bid [[threadgroup_position_in_grid]],\n"
                  "    uint3 xla_bdim [[threads_per_threadgroup]],\n"
                  "    uint3 xla_gdim [[threadgroups_per_grid]]) {\n");
  if (arg_buffer) {
    for (int i = 0; i < info.num_buffer_args; ++i) {
      absl::StrAppend(&s, "  device char* xla_arg", i,
                      " = (device char*)xla_args[", i, "];\n");
    }
  }
  for (const SharedArray& a : info.shared) {
    absl::StrAppend(&s, "  threadgroup uint4 ", a.name, "[",
                    std::max<int64_t>(1, (a.bytes + 15) / 16), "];\n");
  }
  std::vector<std::string> args;
  for (int i = 0; i < info.num_buffer_args; ++i) {
    args.push_back(absl::StrCat("xla_arg", i));
  }
  for (const char* v : {"xla_tid", "xla_bid", "xla_bdim", "xla_gdim"}) {
    for (const char* c : {"x", "y", "z"}) {
      args.push_back(absl::StrCat("(int)", v, ".", c));
    }
  }
  for (const SharedArray& a : info.shared) {
    args.push_back(absl::StrCat("(threadgroup char*)", a.name));
  }
  absl::StrAppend(&s, "  ", info.body_name, "(", absl::StrJoin(args, ", "),
                  ");\n}\n");
  return s;
}

}  // namespace

int ThreadsPerThreadgroupFromRanges(mlir::ModuleOp module,
                                    absl::string_view entry_function) {
  // 0: no information for that dimension.
  int64_t counts[3] = {0, 0, 0};
  bool ok = true;
  // MlirKernelEmitter kernels: gpu.thread_id x, y and z with xla.range.
  module.walk([&](mlir::gpu::ThreadIdOp op) {
    auto f = op->getParentOfType<mlir::func::FuncOp>();
    if (!f || f.getSymName().str() != entry_function) return;
    auto range = op->getAttrOfType<mlir::ArrayAttr>("xla.range");
    auto hi = range && range.size() == 2
                  ? mlir::dyn_cast<mlir::IntegerAttr>(range[1])
                  : mlir::IntegerAttr();
    if (!hi) {
      ok = false;
      return;
    }
    int d = static_cast<int>(op.getDimension());
    counts[d] = std::max<int64_t>(counts[d], hi.getInt() + 1);
  });
  // xla/codegen/emitters kernels (loop, concatenate, ...): the outermost
  // scf.forall is the threadgroup; LowerXlaShared turns its dimensions into
  // ranged gpu.thread_id ops and absent dimensions are 1.
  bool forall = false;
  module.walk([&](mlir::scf::ForallOp op) {
    if (op->getParentOfType<mlir::scf::ForallOp>()) return;
    if (!op.getDynamicUpperBound().empty() || !op.isNormalized() ||
        op.getRank() > 3) {
      ok = false;
      return;
    }
    forall = true;
    for (auto [d, size] : llvm::enumerate(op.getStaticUpperBound())) {
      counts[d] = std::max<int64_t>(counts[d], size);
    }
  });
  int64_t n = 1;
  for (int64_t c : counts) n *= c == 0 && forall ? 1 : c;
  if (!ok || n <= 0 || n > 1024) return 0;
  return static_cast<int>(n);
}

absl::StatusOr<MslKernel> EmitMslKernel(
    mlir::ModuleOp module, absl::string_view entry_function,
    const stream_executor::DeviceDescription& device,
    int max_threads_per_threadgroup) {
  (void)device;
  mlir::MLIRContext* ctx = module.getContext();
  ctx->getOrLoadDialect<mlir::emitc::EmitCDialect>();
  ctx->getOrLoadDialect<mlir::arith::ArithDialect>();
  ctx->getOrLoadDialect<mlir::vector::VectorDialect>();
  ctx->getOrLoadDialect<mlir::func::FuncDialect>();
  ctx->getOrLoadDialect<mlir::scf::SCFDialect>();
  ctx->getOrLoadDialect<ml::LLVMDialect>();

  std::string diagnostics;
  mlir::ScopedDiagnosticHandler handler(ctx, [&](mlir::Diagnostic& d) {
    absl::StrAppend(&diagnostics, d.str(), "\n");
    return mlir::success();
  });

  auto entry = module.lookupSymbol<mlir::func::FuncOp>(
      llvm::StringRef(entry_function.data(), entry_function.size()));
  if (!entry || entry.isExternal()) {
    return absl::NotFoundError(absl::StrCat(
        "MSL emitter: entry function '", entry_function, "' not found"));
  }

  KernelInfo info;
  info.kernel_name = std::string(entry_function);
  info.max_threads_per_threadgroup = max_threads_per_threadgroup;
  info.body_name = absl::StrCat(info.kernel_name, "_impl");

  if (absl::Status s = CleanUpUnrealizedCasts(module); !s.ok()) return s;
  if (absl::Status s = RewriteLlvmArithmetic(module); !s.ok()) return s;
  if (absl::Status s = ExpandArith(module); !s.ok()) return s;
  if (absl::Status s = ScalarizeVectorArithmetic(module); !s.ok()) return s;
  if (absl::Status s = PrepareFunctions(module, entry, info); !s.ok()) return s;
  if (absl::Status s = Validate(module); !s.ok()) return s;
  if (absl::Status s = InferAndApplyAddressSpaces(module, entry, info);
      !s.ok()) {
    return s;
  }

  // Dialect conversion to EmitC.
  MslTypeConverter type_converter(ctx);
  ConversionState state;
  state.info = &info;
  mlir::RewritePatternSet patterns(ctx);
  mlir::populateArithToEmitCPatterns(type_converter, patterns);
  mlir::populateSCFToEmitCConversionPatterns(patterns, type_converter);
  mlir::populateFuncToEmitCPatterns(type_converter, patterns,
                                    /*lowerToCpp=*/false);
  patterns.add<LoadLowering, StoreLowering, GEPLowering, PtrToIntLowering,
               IntToPtrLowering, PassThroughLowering<ml::AddrSpaceCastOp>,
               PassThroughLowering<ml::FreezeOp>, AddressOfLowering,
               AllocaLowering, UndefLowering<ml::UndefOp>,
               UndefLowering<ml::PoisonOp>, UndefLowering<ml::ZeroOp>,
               UndefLowering<mlir::ub::PoisonOp>, AtomicRMWLowering,
               CmpXchgLowering, ExtractValueLowering,
               BitcastLowering<ml::BitcastOp>,
               BitcastLowering<mlir::arith::BitcastOp>, VectorExtractLowering,
               VectorInsertLowering, ExtractElementLowering,
               InsertElementLowering, FromElementsLowering, BroadcastLowering,
               ExtractStridedSliceLowering, InsertStridedSliceLowering,
               BarrierLowering, ShuffleLowering>(type_converter, ctx, &state);
  patterns.add<SmallFloatConstantLowering>(type_converter, ctx, &state,
                                           /*benefit=*/2);
  patterns.add<MathCallLowering>(type_converter, ctx);
  patterns.add<WhileLowering>(type_converter, ctx);

  mlir::ConversionTarget target(*ctx);
  target.addLegalDialect<mlir::emitc::EmitCDialect>();
  target.addLegalOp<mlir::ModuleOp>();
  target.addIllegalDialect<mlir::arith::ArithDialect, mlir::math::MathDialect,
                           mlir::scf::SCFDialect, mlir::func::FuncDialect,
                           mlir::vector::VectorDialect, mlir::gpu::GPUDialect,
                           mlir::ub::UBDialect, ml::LLVMDialect>();
  target.addLegalOp<ml::GlobalOp>();
  if (mlir::failed(mlir::applyPartialConversion(module, target,
                                                std::move(patterns)))) {
    return absl::InternalError(
        absl::StrCat("MSL emitter: conversion to EmitC failed:\n", diagnostics));
  }
  llvm::SmallVector<mlir::UnrealizedConversionCastOp> casts;
  module.walk([&](mlir::UnrealizedConversionCastOp c) { casts.push_back(c); });
  llvm::SmallVector<mlir::UnrealizedConversionCastOp> remaining;
  mlir::reconcileUnrealizedCasts(casts, &remaining);
  if (!remaining.empty()) {
    return absl::InternalError(
        "MSL emitter: unresolved type conversions after EmitC conversion");
  }
  absl::Status leftover;
  module.walk([&](Operation* op) {
    if (!leftover.ok() || mlir::isa<mlir::ModuleOp, ml::GlobalOp>(op)) return;
    if (op->getDialect() &&
        op->getDialect()->getNamespace() ==
            mlir::emitc::EmitCDialect::getDialectNamespace()) {
      return;
    }
    leftover = Unimplemented(op, "op left after EmitC conversion");
  });
  if (!leftover.ok()) return leftover;

  // No `static` (FuncToEmitC's choice for private functions) in MSL.
  mlir::ArrayAttr inline_spec = OpBuilder(ctx).getStrArrayAttr({"inline"});
  module.walk([&](mlir::emitc::FuncOp f) { f.setSpecifiersAttr(inline_spec); });

  llvm::SmallVector<mlir::emitc::FuncOp> order;
  if (absl::Status s = TopologicalFunctionOrder(module, info.body_name, order);
      !s.ok()) {
    return s;
  }

  std::string out(kPrelude);
  llvm::raw_string_ostream os(out);
  os << "\n";
  for (const auto& entry_it : info.constants) {
    os << entry_it.second.definition;
  }
  llvm::SmallVector<mlir::emitc::ClassOp> classes;
  module.walk([&](mlir::emitc::ClassOp c) { classes.push_back(c); });
  for (mlir::emitc::ClassOp c : classes) {
    if (mlir::failed(mlir::emitc::translateToCpp(c, os))) {
      return absl::InternalError(absl::StrCat(
          "MSL emitter: printing struct failed:\n", diagnostics));
    }
    os << "\n";
  }
  for (mlir::emitc::FuncOp f : order) {
    if (mlir::failed(mlir::emitc::translateToCpp(f, os))) {
      return absl::InternalError(absl::StrCat(
          "MSL emitter: printing function failed:\n", diagnostics));
    }
    os << "\n";
  }
  os << KernelWrapper(info);
  os.flush();

  MslKernel kernel;
  kernel.kernel_name = info.kernel_name;
  kernel.msl_source = std::move(out);
  kernel.num_buffer_args = info.num_buffer_args;
  return kernel;
}

}  // namespace metal_pjrt::codegen
