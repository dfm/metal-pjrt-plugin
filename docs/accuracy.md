# Accuracy notes

For users: most functions agree with CPU float32 to a few ulps, a few
(`lgamma`, `digamma`, `betainc`, `reduce_prod`) need wide tolerances, and
Metal's `exp`, `sin`, `cos` and `log` carry a small bias; the known gaps are
listed below.

How accuracy is measured: `tests/metal_testing.py` (f64 CPU reference, ulps
of the output dtype, `METAL_TEST_REPORT_ULPS=1` prints every comparison with
the CPU float32 error next to it). Tolerances are about twice the measured
error. Machine: M3 10-core GPU, macOS 26.2; another GPU or macOS version (a
different Metal compiler and math library) may need retuned tolerances.

## Known gaps

Open, not being fixed right now:

- tinygp's parallel (quasisep-par) solver mean at n = 200000 is ~1% off
  (1.7e5-2.0e5 ulps normwise against the float64 sequential solver; CPU
  float32 gives NaN there); cause unknown. The tests no longer run tinygp
  (`bench/tinygp_bench.py` still times it).
- exp / sin / cos keep Metal's bias for |x| >= 0.125 (the prelude's
  polynomials cover only smaller arguments): on [0.125, 1) mean +0.11 ulps
  for exp(-x), -0.10 for sin, -0.01 for cos (max 1.4 / 2.6 / 2.2; CPU ~0,
  max 0.8 / 0.6 / 0.5); exp on 1 .. 10 +0.37 (table below).
- tanh, sinh / cosh and the prelude's `xla_expm1` / `erf` still use Metal's
  biased `exp`.
- Metal's `log` is biased everywhere, by about +-0.5 ulps (+ below 1, -
  above), max ~2.5 ulps (~2.6 for `log2`, which is `log` times 1/ln 2).
- The widest test tolerances (`tests/test_lax.py`, measured error doubled):
  `lgamma` 580 and `digamma` 1600 ulps (CPU float32 536 / 770: the inputs
  straddle digamma's root, where the error is relative to a tiny result),
  `reduce_prod` 93, `betainc` 38.
- Subnormal outputs flush to zero, as with CUDA's ftz (`exp(-100)`,
  `1e-10 * 1e-30`); XLA:CPU flushes them too.

## Metal's biased math functions

Metal's float32 `exp`, `sin`, `cos` and `log` are *biased*: their errors do
not average to zero. `bench/math_bias.py` prints the signed error per |x|
decade (mean signed / mean absolute / max ulps, 2^18 log-uniform samples
per row); Metal's own functions, against CPU (XLA's polynomials):

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

A third of an ulp is harmless for one call, but a long recursion that
multiplies by the same biased value every step accumulates it coherently.
That is how it was found: tinygp's sequential quasisep gradient at n = 20000
was 896 ulps normwise against CPU's 103, with the error entering in the
Cholesky scan's `exp` / `cos` / `sin` of arguments ~1e-3. Ruled out by
A/B: the `metal$scan` rewriter, the GEMM backend, XLA's excess precision,
and Metal's math mode (`mathMode = Safe` + `mathFloatingPointFunctions =
Precise` is bitwise identical to the `fastMathEnabled = false` in use).
Division, sqrt, rsqrt and multiply are correctly rounded and nothing is
contracted into an FMA.

Repro:

    scripts/device_lock.py -- .venv/bin/python bench/math_bias.py exp- sin cos

## Fixed in the MSL prelude

- **Small |x| exp / sin / cos** (a3a91ba). The prelude's `xla_exp` /
  `xla_sin` / `xla_cos` (float overloads; half and vectors call Metal's)
  evaluate a degree-6 / 7 / 6 Taylor polynomial with `fma` for |x| < 0.125
  (truncation < 1e-9 relative) and call Metal's function elsewhere; `sin`
  returns x for |x| < 1e-4, which keeps `sin(-0) = -0`. On |x| < 0.125 the
  bias is gone (mean ~0.00, mean abs 0.25, max 0.50-0.56, as CPU). tinygp
  normwise ulps before -> after (CPU): quasisep n=20000 gradient 896 -> 103
  (103), value 115 -> 41 (41); quasisep-par n=20000 gradient 977 -> 114
  (97); quasisep-par n=200000 value / gradient 1200 / 2290 -> 79 / 303
  (243 / 1040). The polynomial is faster than Metal's functions on
  ALU-bound code (8 rounds of exp+sin+cos over 4M: 1.81 -> 1.22-1.25 ms)
  and costs nothing measurable elsewhere.
- **cbrt** (ce2475b). `pow(|x|, 1/3)` was 7-12 ulps off away from 1 (1/3
  rounds up in float) and 0 for subnormal inputs. The prelude's `xla_cbrt`
  adds one Newton step and handles a subnormal as cbrt(2m) * 2^-50: max
  1.38 ulps against float64 over 1e-45..3e38 (`lax.cbrt wide range` in
  `tests/test_lax.py`).
- **log of subnormal inputs** (70bf680). Metal's float32 arithmetic flushes
  subnormals even with fast math off, so `log(1e-40)` was -inf (true
  -92.10) and `log(-1e-40)` -inf (true NaN). The prelude's `xla_log` /
  `xla_log2` / `xla_log10` (float overloads) read the bits: a subnormal is
  m * 2^-149, so `log(x) = log(float(m)) - 149 ln 2`, and a negative one
  gives NaN. Error on [1.4e-45, FLT_MIN): log -0.16 / 0.29 / 0.83, log2
  +0.01 / 0.26 / 1.25, log10 -0.38 / 0.44 / 1.43. Written with selects: +19%
  on an ALU-bound chain of 16 logs (a branch was worse), no change on
  memory-bound kernels or real programs. XLA:CPU returns -inf here.

### Policy (open question)

Fix clearly wrong values on valid inputs when the fix is cheap. The
subnormal `log` fix makes mtl closer to numpy / IEEE than JAX's CPU
backend; subnormal *outputs* still flush, as with CUDA's ftz, and are
deliberately not fixed. Open question: "match numpy / IEEE" or "match
XLA:CPU" as the rule for cases like this.

## Other changes that moved errors

- softmax in f16 runs as XLA's fusions since the `metal$softmax` rewriter
  was deleted (fcbf5ce): 9.5 ulps max over `tests/test_scan.py`'s shapes,
  the same as CPU, against 5.0 with the rewriter (which accumulated in
  f32); f16 log_softmax 1.0 vs 0.55. f32 (30.6, CPU 30.3) and bf16 (0.5)
  are unchanged.
- GEMM epilogues fused into steel (7704d02) round once: f16/bf16 bias and
  activation lose the intermediate rounding of D (normwise ulps f16 bias
  0.74 -> 0.50, bf16 bias+gelu 0.71 -> 0.48, test_steel_gemm "+bias relu"
  0.61-0.93 -> 0.34-0.50). f32 epilogue GEMMs (MPS + second pass) are
  bitwise identical to a fused f32 kernel.
- bf16 and f8 conversions round exactly (0 ulps against CPU); a round trip
  f32 -> bf16 -> f32 inside one jit is removed by XLA's excess-precision
  simplification, as on CUDA (`docs/op-coverage.md`).
