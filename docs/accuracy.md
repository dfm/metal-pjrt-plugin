# Accuracy

Most functions agree with CPU float32 to within a few ulps. A few special
functions need wide tolerances, and Metal's `exp`, `sin`, `cos` and `log`
carry a small bias.

## Policy

mtl matches XLA:CPU. Where Metal's math library disagrees with CPU, the
plugin follows CPU, or uses its own code where that is closer to CPU
(the small-argument `exp` / `sin` / `cos` polynomials and `cbrt`, below).
Subnormals flush, as on CPU and CUDA's ftz: `log(1e-40) = -inf`,
`cbrt(1e-40) = 1e-40`, `exp(-100) = 0`.

Errors are measured against a float64 CPU reference in ulps of the output
dtype (`tests/metal_testing.py`), with tolerances about twice the measured
error; `METAL_TEST_REPORT_ULPS=1` prints them next to CPU float32's. All
numbers come from an M3 on macOS 26.2. Other GPUs or macOS versions may
need retuned tolerances.

## Known gaps

- **`exp`, `sin`, `cos` for |x| >= 0.125** keep Metal's bias. On
  [0.125, 1) the mean error is +0.11 ulps for exp(-x), -0.10 for sin,
  -0.01 for cos (max 1.4 / 2.6 / 2.2; CPU ~0, max 0.8 / 0.6 / 0.5). On
  1..10, exp is +0.37.
- **`sinh`, `cosh`, `expm1`** use Metal's biased `exp`. f32 `tanh` and
  `erf` don't (XLA expands them to rational approximations); f16 `tanh`
  is Metal's builtin.
- **`log`** is biased by about +-0.5 ulps everywhere (positive below 1,
  negative above), max ~2.5 ulps (~2.6 for `log2`).
- **Widest test tolerances** (`tests/test_lax.py`): `lgamma` 580 and
  `digamma` 1600 ulps (CPU float32 536 / 770, near digamma's root),
  `reduce_prod` 93, `betainc` 38.
- **Subnormals that flush differently** (`test_subnormals_match_cpu` pins
  the cases that agree):
  - `sin`, `tan`, `sinh`, `asinh`, `log1p` return a subnormal input
    unchanged; CPU gives +-0.
  - `pow` flushes a subnormal exponent (`pow(0, 1e-40) = 1`,
    `pow(inf, 1e-40) = 1`, `pow(-0.5, 1e-40) = 1`; CPU 0, inf, NaN) and a
    subnormal base (`pow(1e-40, 0.5) = 0`, CPU 1e-20; `pow(1e-40, 1) =
    1e-40`, CPU 0).
  - `rem(x, 1e-40)` is NaN; CPU gives +-0.
  - `atan2(1e-40, 0)` is 0; CPU gives 1e-40.
  - Converting a float32 subnormal to bfloat16 at run time gives +-0
    (Metal's conversion flushes; CPU keeps it): `x.astype(bfloat16)` for
    |x| < 1.18e-38, and `jnp.spacing` of bfloat16 subnormals and zero (0;
    CPU 9.2e-41: it computes those in float32 and converts back). Arithmetic that
    lands in that range flushes on both; float16 is unaffected.
- **`atan2(+-0, negative)`** is float32(pi) = 3.1415927, correctly rounded;
  CPU gives one ulp less.
- **tinygp's parallel solver** (quasisep-par) at n = 200000 gives a mean
  ~1% off (1.7e5-2.0e5 ulps normwise; CPU float32 gives NaN). Cause
  unknown. `bench/tinygp_bench.py` still times it.

## Metal's biased math functions

Metal's float32 `exp`, `sin`, `cos` and `log` have errors that don't
average to zero. Signed error per decade of |x| from `bench/math_bias.py`
(mean signed / mean absolute / max ulps):

| fn | \|x\| | Metal | CPU |
|---|---|---|---|
| exp(-x) | 1e-4 .. 1e-2 | -0.32 / 0.48 / 1.5 | +0.00 / 0.25 / 0.5 |
| exp(-x) | 0.01 .. 0.1 | -0.27 / 0.45 / 1.5 | -0.00 / 0.25 / 0.6 |
| exp(+x) | 1e-4 .. 0.1 | +0.01 / 0.29 / 0.9 | -0.00 / 0.25 / 0.5 |
| exp(+-x) | 1 .. 10 | +0.37 / 0.45 / 1.4 | -0.01 / 0.27 / 1.0 |
| sin | 1e-4 .. 0.1 | -0.19 .. -0.26 / 0.4-0.56 / 2.6 | +0.00 / 0.25 / 0.5 |
| cos | 1e-4 .. 0.1 | +0.21 .. +0.28 / 0.5-0.7 / 2.2 | +0.00 / 0.25 / 0.5 |
| log | 1e-4 .. 1 | +0.47 .. +0.52 / 0.55 / 1.9 | +0.00 / 0.25 / 0.6 |
| log | 1 .. 10 | -0.51 / 0.56 / 2.5 | -0.00 / 0.28 / 1.2 |

A third of an ulp is harmless in one call, but a long recursion that
multiplies by the same biased value adds it up: tinygp's sequential
gradient at n = 20000 was 896 ulps off (CPU 103). Division, `sqrt`,
`rsqrt` and multiplication are correctly rounded, nothing is contracted
into an FMA, and Metal's safe/precise math modes give identical results.

    scripts/device_lock.py -- .venv/bin/python bench/math_bias.py exp- sin cos

## Fixes in the MSL prelude

- **Small-|x| `exp` / `sin` / `cos`** (2026-09-27): for |x| < 0.125 the
  float versions use Taylor polynomials with `fma`, which removes the bias
  there (mean ~0.00, max ~0.5 ulps, as CPU) and is no slower. tinygp's
  n = 20000 gradient went from 896 to 103 ulps (CPU 103).
- **`cbrt`** (2026-09-27): a Newton step on `pow(|x|, 1/3)` brings it from
  7-12 ulps to max 1.28 (CPU 0.5).
- **`pow(-inf, y)`** for non-integer y now gives `pow(inf, y)`, as CPU,
  not NaN.

## Other paths

- **Convolutions** on `metal$conv` accumulate f16/bf16 in f32 and round
  once: <= 0.5 ulps normwise, as CPU. f32 is within CPU's own error
  (forward <= 5.7 ulps, CPU 6.0; gradients <= 10.7, CPU up to 38.7). The
  loop emitter (small, grouped, 3-D) is less accurate in f16: 2.6 ulps on
  one weight gradient, CPU 0.47.
- **FFTs** (`metal$fft`): max |error| / max |value| <= 1.2 x 2^-24 log2 n
  over every plan in `tests/test_fft.py` (n up to 2^20), <= 5.4 x 2^-24
  log2 n per plan up to 2^24 in `fft:fft_test`. The plugin computes the
  Rader and Bluestein constants in double, which fixes two inaccuracies in
  MLX itself (near n = 2^23, and non-Hermitian irfft at n = 47).
- **f16 softmax** runs as plain XLA fusions: 9.5 ulps max, as CPU; f16
  log_softmax 1.0 (CPU 0.55).
- **Few-row f16/bf16 GEMMs** and **fused GEMM epilogues** accumulate in
  f32 and round once: <= 0.5 ulps normwise. f32 epilogue GEMMs are
  bitwise identical to a fused kernel.
- **bf16 and f8 conversions** round exactly. A f32 -> bf16 -> f32 round
  trip inside one `jit` is removed by XLA's excess-precision
  simplification, as on CUDA
  ([`op-coverage.md`](op-coverage.md#element-types)).
