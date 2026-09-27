"""tinygp on metal against a float64 CPU reference: the quasiseparable
solver (parallel associative-scan and sequential) and the dense solver;
log-likelihood, its gradient and the conditional mean. A NaN in the
reference fails the test (it used to be skipped over, which let metal pass
whenever CPU float32 produced NaNs). Timings: bench/tinygp_bench.py.
"""
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from metal_testing import REPORT, assert_close, cpu, f64_reference, metal, run_on

tinygp = pytest.importorskip("tinygp")
from tinygp import GaussianProcess, kernels  # noqa: E402
from tinygp.kernels import quasisep  # noqa: E402
from tinygp.solvers import QuasisepSolver  # noqa: E402

pytestmark = pytest.mark.metal

PARAMS = ["log_scale", "log_sigma", "log_omega", "log_q", "log_sigma2",
          "log_jitter", "mean"]
P0 = np.array([1.0, 0.0, 0.5, 1.0, -0.5, -3.0, 0.0], np.float32)


def build(kind, p, x, yerr):
    p = dict(zip(PARAMS, p))
    if kind.startswith("quasisep"):
        k = quasisep.Matern32(scale=jnp.exp(p["log_scale"]), sigma=jnp.exp(p["log_sigma"])) + \
            quasisep.SHO(omega=jnp.exp(p["log_omega"]), quality=jnp.exp(p["log_q"]), sigma=jnp.exp(p["log_sigma2"]))
        return GaussianProcess(k, x, diag=yerr**2 + jnp.exp(2 * p["log_jitter"]), solver=QuasisepSolver, mean=p["mean"],
                               assume_sorted=True,  # the sortedness check is a jax.debug.callback
                               parallel=(kind == "quasisep-par"))  # associative-scan algorithms (tinygp main)
    k = jnp.exp(2 * p["log_sigma"]) * kernels.Matern32(scale=jnp.exp(p["log_scale"])) + \
        jnp.exp(2 * p["log_sigma2"]) * kernels.ExpSquared(scale=jnp.exp(p["log_omega"]))
    return GaussianProcess(k, x, diag=yerr**2 + jnp.exp(2 * p["log_jitter"]), mean=p["mean"])


def data(n):
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(0, 100, n)).astype(np.float32)
    yerr = (0.1 + 0.1 * rng.random(n)).astype(np.float32)
    y = (np.sin(x / 3) + 0.3 * rng.standard_normal(n)).astype(np.float32)
    xt = np.linspace(0, 100, 500, dtype=np.float32)
    return x, y, yerr, xt


def program(kind):
    """(value, gradient, conditional mean) of the GP at parameters p."""
    def f(p, x, y, yerr, xt):
        loss = lambda p: -build(kind, p, x, yerr).log_probability(y)
        v, g = jax.value_and_grad(loss)(p)
        return v, g, build(kind, p, x, yerr).condition(y, xt).gp.loc
    return f


# Normwise float32 ulps, measured (METAL_TEST_REPORT_ULPS=1, M3) and doubled.
# CPU float32 for comparison (value / gradient / mean): quasisep-par 1000
# 54 / 63 / 223, 20000 43 / 97 / NaN, 200000 243 / 1040 / NaN; quasisep 1000
# 46 / 56 / 92, 20000 41 / 103 / 1300; dense 1000 29 / 42 / 707, 3000
# 82 / 284 / 2230.
# Known accuracy gap (not accepted; don't loosen further): parallel solver
# mean at n = 200000: 2e5 ulps (~1% of max|mean|); CPU float32 is NaN there,
# so no float32 baseline. Not the exp/sin/cos bias below (unchanged by it).
# Tracked by test_tinygp_parallel_mean_gap (strict xfail at a tight tolerance).
# The gradient gap at n = 20000 (896 / 977 ulps vs CPU 103 / 97) was
# Metal's float32 exp / sin / cos being biased for the small arguments of
# the transition matrices, accumulated over the 20000-step scan; fixed by
# the prelude's small-|x| polynomials (xla_exp etc.): now 103 / 114
# (docs/accuracy.md). Repro:
#   scripts/device_lock.py -- env METAL_TEST_REPORT_ULPS=1 .venv/bin/python \
#     -m pytest tests/test_tinygp.py -s -k 20000
#   scripts/device_lock.py -- .venv/bin/python bench/math_bias.py
ULPS = {  # (value, gradient, mean)
    ('quasisep-par', 1000): (110, 130, 600),
    ('quasisep-par', 20000): (90, 230, 20000),
    ('quasisep-par', 200000): (160, 610, 400000),
    ('quasisep', 1000): (110, 130, 240),
    ('quasisep', 20000): (90, 210, 2500),
    ('dense', 1000): (160, 170, 800),
    ('dense', 3000): (230, 810, 2500),
}


@functools.cache
def results(kind, n):
    """(metal, float64 reference, CPU float32 or None) for one case."""
    fn, args = program(kind), (P0, *data(n))
    # tinygp's parallel condition() is NaN on CPU even in float64 for
    # n >= 20000, so the parallel solver is checked against the sequential
    # one (same model; the two agree to 1e-15 on the log-likelihood).
    want = f64_reference(program("quasisep" if kind == "quasisep-par" else kind), *args)
    assert not any(np.isnan(w).any() for w in want), "float64 reference has NaNs"
    got = run_on(metal(), fn, *args)
    cpu32 = run_on(cpu(), fn, *args) if REPORT else None
    return got, want, cpu32


@pytest.mark.parametrize("kind,n", [
    ("quasisep-par", 1000), ("quasisep-par", 20000), ("quasisep-par", 200000),
    ("quasisep", 1000), ("quasisep", 20000), ("dense", 1000), ("dense", 3000)])
def test_tinygp(kind, n):
    got, want, cpu32 = results(kind, n)
    for i, part in enumerate(["value", "gradient", "mean"]):
        assert_close(got[i], want[i], ULPS[kind, n][i], normwise=True,
                     name=f"{kind} n={n} {part}",
                     cpu32=None if cpu32 is None else cpu32[i])


@pytest.mark.xfail(strict=True, reason="known gap: parallel solver mean at "
                   "n = 200000 is ~2e5 ulps; tracked at the sequential "
                   "solver's tolerance")
def test_tinygp_parallel_mean_gap():
    got, want, _ = results("quasisep-par", 200000)
    assert_close(got[2], want[2], ULPS["quasisep", 20000][2], normwise=True,
                 name="quasisep-par n=200000 mean (tight)")
