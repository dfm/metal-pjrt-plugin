# Accuracy notes

How accuracy is measured: `tests/metal_testing.py` (f64 CPU reference, ulps
of the output dtype, `METAL_TEST_REPORT_ULPS=1` prints every comparison with
the CPU float32 error next to it). Machine: M3 10-core GPU, macOS 26.2.

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
