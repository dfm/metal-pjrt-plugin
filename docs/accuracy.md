# Accuracy notes

How accuracy is measured: `tests/metal_testing.py` (f64 CPU reference, ulps
of the output dtype, `METAL_TEST_REPORT_ULPS=1` prints every comparison with
the CPU float32 error next to it). Machine: M3 10-core GPU, macOS 26.2.

## Known gaps

Open, not being fixed right now; details in the sections below.

- tinygp's parallel (quasisep-par) solver mean at n = 200000 is ~1% off;
  cause unknown.
- exp / sin / cos keep Metal's bias on [0.125, 1): mean +0.11 ulps for
  exp(-x), -0.10 for sin (table below).
- tanh, sinh / cosh and the prelude's `xla_expm1` / `erf` still use Metal's
  biased `exp`.
- Metal's `log` is biased by about +-0.5 ulps (+ below 1, - above), max
  ~2.5 ulps (~2.6 for `log2`).
- Subnormal outputs flush to zero, as with CUDA's ftz.

## Biased exp / sin / cos / log (2026-09-26)

The one accuracy gap the test suite found is tinygp's gradient at n = 20000
(sequential quasisep solver): 896 ulps normwise against 103 for CPU float32
(relative 6e-5 on d/dlog_sigma, CPU ~6e-6). Ruled out by A/B: the metal$scan
rewriter, GEMM backend, XLA excess precision, and the Metal math mode
(`mathMode = Safe` + `mathFloatingPointFunctions = Precise` gives outputs
bitwise identical to the `fastMathEnabled = false` we use).

Bisect (dump HLO, compare every intermediate of the quasisep program against
f64 CPU): the matrix build and the reductions are as accurate as on CPU (sum
error 0.1 ulp of the terms' own sum); the error enters in the Cholesky scan,
whose transition matrices are `exp`, `cos` and `sin` of `-dt * sqrt(3) /
scale` with |dt * sqrt(3) / scale| ~ 1e-3. Those come out 1.9 ulps normwise
against CPU's 1.16, and the scan outputs are biased: the logdet terms sum to
27 ulps of error (CPU 6), z.z to 85 (CPU 26). Division, sqrt, rsqrt and
multiply are correctly rounded and nothing is contracted into an FMA.

Cause: Metal's float32 `exp`, `sin`, `cos` (and `log`) are *biased* for
small arguments. `bench/math_bias.py` prints the signed error per |x| decade;
mean signed / mean absolute / max ulps, 2^18 log-uniform samples per row:

| fn | \|x\| | Metal | CPU (XLA's own polynomials) |
|---|---|---|---|
| exp(-x) | 1e-4 .. 1e-2 | -0.32 / 0.48 / 1.5 | +0.00 / 0.25 / 0.5 |
| exp(-x) | 0.01 .. 0.1 | -0.27 / 0.45 / 1.5 | -0.00 / 0.25 / 0.6 |
| exp(+x) | 1e-4 .. 0.1 | +0.01 / 0.29 / 0.9 | -0.00 / 0.25 / 0.5 |
| exp(+-x) | 1 .. 10 | +0.37 / 0.45 / 1.4 | -0.01 / 0.27 / 1.0 |
| sin | 1e-4 .. 0.1 | -0.19 .. -0.26 / 0.4-0.56 / 2.6 | +0.00 / 0.25 / 0.5 |
| cos | 1e-4 .. 0.1 | +0.21 .. +0.28 / 0.5-0.7 / 2.2 | +0.00 / 0.25 / 0.5 |
| log | 1e-4 .. 1 | +0.47 .. +0.52 / 0.55 / 1.9 | +0.00 / 0.25 / 0.6 |
| log | 1 .. 10 | -0.51 / 0.56 / 2.5 | -0.00 / 0.28 / 1.2 |

A per-call bias of a third of an ulp is harmless for one call, but a
20000-step recursion whose every step multiplies by the same biased
`exp(-dt / scale)` accumulates it coherently (at most 0.3 ulp x 20000 steps
~ 4e-4 relative; the gradient is off by 6e-5). CPU's errors average to zero
and cancel instead. (`tanh` and `expm1` are XLA / prelude expansions on both
backends; their CPU numbers are no better.)

Repro:

    scripts/device_lock.py -- .venv/bin/python bench/math_bias.py exp- sin cos
    scripts/device_lock.py -- env METAL_TEST_REPORT_ULPS=1 .venv/bin/python \
      -m pytest tests/test_tinygp.py -s -k 20000

### Fix: Taylor polynomials for |x| < 0.125 (2026-09-27)

The prelude's `xla_exp` / `xla_sin` / `xla_cos` (float overloads; half and
vectors call Metal's) evaluate a degree-6 / 7 / 6 Taylor polynomial with
`fma` for |x| < 0.125 (truncation < 1e-9 relative) and Metal's function
elsewhere; `sin` returns x for |x| < 1e-4, which keeps `sin(-0) = -0`.
(The alternative, `exp = 1 + expm1`, is no cheaper: the prelude's
`xla_expm1` is built on Metal's `exp` and `log`.)

| fn | \|x\| | Metal before | Metal after | CPU |
|---|---|---|---|---|
| exp(-x) | 1e-4 .. 1e-2 | -0.32 / 0.48 / 1.5 | +0.00 / 0.25 / 0.50 | +0.00 / 0.25 / 0.5 |
| exp(-x) | 0.01 .. 0.125 | -0.26 / 0.45 / 1.5 | -0.00 / 0.25 / 0.56 | -0.00 / 0.25 / 0.57 |
| exp(+x) | 1e-4 .. 0.125 | +0.00..+0.01 / 0.29 / 0.9 | -0.00 / 0.25 / 0.56 | -0.00 / 0.25 / 0.57 |
| sin | 1e-4 .. 0.125 | -0.18 .. -0.27 / 0.4-0.58 / 2.6 | -0.00 .. +0.07 / 0.19-0.25 / 0.50 | same as after |
| cos | 1e-4 .. 0.125 | +0.20 .. +0.28 / 0.5-0.7 / 2.3 | +0.00 / 0.2-0.25 / 0.50 | same as after |
| exp(-x) / sin / cos | 0.125 .. 1 | +0.11 / -0.10 / -0.01 (max 1.4 / 2.6 / 2.2) | unchanged (Metal's) | ~0 (max 0.8 / 0.6 / 0.5) |

Zeros and subnormals keep value and sign (`math_bias.py` checks them; for
`log` see the next section).

tinygp's transition-matrix arguments are all below 0.125 for n >= 20000
(max 0.06) and 70-86% of them for n = 1000. Normwise ulps before -> after
(CPU float32):

| program | before | after | CPU |
|---|---|---|---|
| quasisep n=20000 gradient | 896 | 103 | 103 |
| quasisep n=20000 value | 115 | 41 | 41 |
| quasisep-par n=20000 gradient | 977 | 114 | 97 |
| quasisep-par n=200000 value / gradient | 1200 / 2290 | 79 / 303 | 243 / 1040 |
| quasisep-par n=200000 mean | 1.7e5 | 2.0e5 | NaN |

The parallel solver's mean at n = 200000 (~1% off) is a different problem.
No other pytest comparison changed (METAL_TEST_REPORT_ULPS=1 over the whole
suite, both builds): lax_test's exp/sin/cos use |x| up to 2.

Cost (4 interleaved rounds, same machine, medians, ms): 8 rounds of
exp+sin+cos over 4M elements, ALU-bound, 1.81 -> 1.22-1.25 (in range; one
outlier round 1.88); exp over 16M and sin+cos over 16M, memory-bound, 1.8-2.0
both; gelu/tanh MLP 2048x1024x4096 12.9-13.0 both; softmax 8192x1024 1.04-1.06
vs 0.99-1.06; nanoGPT train step 181.8-185.3 vs 180.9-186.1. Taking the
polynomial is faster than Metal's `exp`/`sin`/`cos`, and the branch costs
nothing measurable outside the range.

## log of subnormal inputs (2026-09-27)

Metal's float32 arithmetic flushes subnormals, even with fast math off:
`x * 1e10` is 0 for x = 1e-40, and its `log` / `log2` / `log10` saw them as
0: `log(1e-40) = -inf` (true -92.10) and `log(-1e-40) = -inf` (true NaN).
(Loads, stores and bit moves keep subnormals, and so does `log1p`, which
returns x when 1 + x == 1.) The prelude's `xla_log` / `xla_log2` /
`xla_log10` (float overloads; half and bf16 compute in float and are
unaffected by f16 subnormals) test the bits: a subnormal is m * 2^-149 with
the integer m = its bits, so `log(x) = log(float(m)) - 149 ln 2`; a negative
subnormal's int(bits) is negative, so the result is NaN. `jnp.log2` and
`jnp.log10` are `log` times a constant in JAX, so all three go through
`xla_log`; the other two are mapped for XLA-emitted `math.log2/log10`.

`bench/math_bias.py` (row [1.4e-45, FLT_MIN), mean signed / mean abs / max
ulps): log -0.16 / 0.29 / 0.83, log2 +0.01 / 0.26 / 1.25, log10 -0.38 / 0.44 /
1.43. XLA CPU flushes subnormals too and returns -inf for all of these.
Normal inputs are unchanged, including the +-0.5 ulp bias of Metal's `log`
(table above). (`log2(1e-30)` is 2 ulps off, which the zeros/subnormals check
flags: `log2` is Metal's `log` times 1/ln 2 and its max error is ~2.6 ulps.)

Written with selects, not a branch. Cost, 3 interleaved rounds, p10 / median
/ p90 ms: an ALU-bound chain of 16 logs per element over 4M, 0.75-0.77 /
0.79-0.81 / 0.82 -> 0.92 / 0.95 / 0.96-1.24 (+19%); a branch on the input
was 1.16-1.24 median, a branch on a -inf result 1.20-1.23. Memory-bound
log over 16M (1.77-1.92 both), log_softmax 8192x1024 (1.73 both) and the
nanoGPT train step (180.1-182.4 vs 181.2-183.0) are unchanged.

### Policy (open for dfm to confirm)

Current stance: fix clearly wrong values on valid inputs when the fix is
cheap. `log(subnormal) = -inf` is fixed although XLA:CPU also returns -inf,
so openmetal is closer to numpy / IEEE than JAX's CPU backend here. The cost
is +19% on ALU-bound log chains; memory-bound kernels and real programs are
unchanged. Subnormal *outputs* still flush to zero (`exp(-100)` and
`exp(-88)` give 0 on both openmetal and XLA:CPU; `1e-10 * 1e-30` gives 0), as
with CUDA's ftz, and are deliberately not fixed. Open question for dfm:
"match numpy / IEEE" or "match XLA:CPU" as the rule for cases like this.

## softmax in f16 after the metal$softmax removal (2026-09-27)

With the softmax rewriter deleted (no end-to-end win, docs/performance.md),
f16 softmax runs as XLA's fusions: 9.5 ulps max over
`tests/test_scan.py`'s shapes, the same as CPU, against 5.0 via
`metal$softmax` (which accumulated in f32). f32 (30.6 ulps, CPU 30.3) and
bf16 (0.5) are unchanged; f16 log_softmax 1.0 vs 0.55.

## GEMM epilogue fused into steel (2026-09-27)

Bias / activation now run in steel's store with one rounding
(docs/performance.md, "GEMM epilogue inside steel"). f32 epilogue GEMMs stay
on MPS + the second pass (fused steel gave bitwise identical results);
f16/bf16 epilogue GEMMs lose the intermediate rounding of D (normwise ulps
e.g. f16 bias 0.74 -> 0.50, bf16 bias+gelu 0.71 -> 0.48, test_steel_gemm
"+bias relu" 0.61-0.93 -> 0.34-0.50).
