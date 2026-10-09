// Copyright 2026 The metal-pjrt-plugin Authors
// SPDX-License-Identifier: Apache-2.0

#include <metal_stdlib>
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

// IEEE 754-2019 maximum/minimum, as XLA:CPU's llvm.maximum/minimum: a NaN
// operand gives NaN and -0 < +0 (so relu(-0.0) = +0.0).
template <typename T>
inline T xla_maximum(T a, T b) {
  if (isnan(a) || isnan(b)) return a + b;
  if (a == b) return signbit(a) ? b : a;
  return a > b ? a : b;
}
template <typename T>
inline T xla_minimum(T a, T b) {
  if (isnan(a) || isnan(b)) return a + b;
  if (a == b) return signbit(a) ? a : b;
  return a < b ? a : b;
}

// Integer min/max (scalars of 8 to 64 bits, index as size_t) through
// Metal's builtins, with the op's signedness. Not a compare and select:
// Metal miscompiled that as a gather's index clamp (msl_emitter.cc,
// ExpandArith).
template <int N> struct xla_int_of;
template <> struct xla_int_of<1> { typedef char s; typedef uchar u; };
template <> struct xla_int_of<2> { typedef short s; typedef ushort u; };
template <> struct xla_int_of<4> { typedef int s; typedef uint u; };
template <> struct xla_int_of<8> { typedef long s; typedef ulong u; };
template <typename T>
inline T xla_maxsi(T a, T b) {
  typedef typename xla_int_of<sizeof(T)>::s S;
  return T(max(S(a), S(b)));
}
template <typename T>
inline T xla_minsi(T a, T b) {
  typedef typename xla_int_of<sizeof(T)>::s S;
  return T(min(S(a), S(b)));
}
template <typename T>
inline T xla_maxui(T a, T b) {
  typedef typename xla_int_of<sizeof(T)>::u U;
  return T(max(U(a), U(b)));
}
template <typename T>
inline T xla_minui(T a, T b) {
  typedef typename xla_int_of<sizeof(T)>::u U;
  return T(min(U(a), U(b)));
}

// Bit counts. Metal computes popcount/clz/ctz of an int8_t (the emitter's
// i8, signless) as of a sign-extended int: popcount(int8_t(-1)) is 32,
// clz(int8_t(5)) is 29. 8-bit values go through their zero-extended uint.
template <typename T>
inline T xla_popcount(T x) { return popcount(x); }
template <typename T>
inline T xla_clz(T x) { return clz(x); }
template <typename T>
inline T xla_ctz(T x) { return ctz(x); }
inline int8_t xla_popcount(int8_t x) { return int8_t(popcount(uint(uchar(x)))); }
inline int8_t xla_clz(int8_t x) { return int8_t(clz(uint(uchar(x))) - 24); }
inline int8_t xla_ctz(int8_t x) { return int8_t(ctz(uint(uchar(x)) | 0x100u)); }

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
  // From x ~ 17 on, u - 1 rounds to u; um1 * xf would overflow for
  // x > 84.3, where exp(x) is still finite (up to 88.72).
  if (um1 == u) return T(u);
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
template <typename T>
inline T xla_exp(T x) { return exp(x); }
template <typename T>
inline T xla_sin(T x) { return sin(x); }
template <typename T>
inline T xla_cos(T x) { return cos(x); }
// erf for the types XLA's RewriteErf32Pattern leaves alone (it expands f32
// only), with that pattern's rational approximation evaluated in float:
// coefficients and the saturation bound from XLA's
// xla/codegen/emitters/transforms/expand_float_ops.cc (Apache-2.0,
// Copyright The OpenXLA Authors).
template <typename T>
inline T xla_erf(T x) {
  float xf = float(x);
  // erfinv(1 - 2^-23): beyond it erf rounds to +-1 in float.
  if (fabs(xf) >= 3.7439211627767994f) return T(copysign(1.0f, xf));
  float x2 = xf * xf;
  float p = 0.00022905065861350646f;
  p = fma(p, x2, 0.0034082910107109506f);
  p = fma(p, x2, 0.050955695062380861f);
  p = fma(p, x2, 0.18520832239976145f);
  p = fma(p, x2, 1.128379143519084f);
  float q = -1.1791602954361697e-7f;
  q = fma(q, x2, 0.000023547966471313185f);
  q = fma(q, x2, 0.0010179625278914885f);
  q = fma(q, x2, 0.014070470171167667f);
  q = fma(q, x2, 0.11098505178285362f);
  q = fma(q, x2, 0.49746925110067538f);
  q = fma(q, x2, 1.0f);
  return T(xf * p / q);
}
// pow(|x|, 1/3) alone is 7-12 ulps off away from 1 (1/3 rounds up in float,
// an error that grows with |log x|); one Newton step, (2r + a / r^2) / 3,
// which cannot overflow near FLT_MAX, fixes it. Zeros, subnormals,
// infinities and NaN are returned as they are, as XLA:CPU does (it flushes
// subnormal inputs, so cbrt(1e-40) = 1e-40 there, not 4.6e-14).
inline float xla_cbrt(float x) {
  uint e = as_type<uint>(x) & 0x7f800000u;
  if (e == 0u || e == 0x7f800000u) return x;
  float a = fabs(x);
  float r = pow(a, 1.0f / 3.0f);
  r = (2.0f * r + a / (r * r)) / 3.0f;
  return copysign(r, x);
}
template <typename T>
inline T xla_cbrt(T x) { return T(xla_cbrt(float(x))); }
template <typename T>
inline T xla_powf(T x, T y) {
  float xf = float(x);
  float yf = float(y);
  if (xf < 0.0f) {
    float r = pow(-xf, yf);
    // pow(-inf, y) = pow(inf, y) for y not an odd integer, as on XLA:CPU.
    if (yf != trunc(yf)) return T(isinf(xf) ? r : NAN);
    return T(fmod(fabs(yf), 2.0f) == 1.0f ? -r : r);
  }
  return T(pow(xf, yf));
}
