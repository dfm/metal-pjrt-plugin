"""tinygp on metal against a float64 CPU reference: the quasiseparable
solver (parallel associative-scan and sequential) and the dense solver;
log-likelihood, its gradient and the conditional mean. A NaN in the
reference fails the test (it used to be skipped over, which let metal pass
whenever CPU float32 produced NaNs). Timings: bench/tinygp_bench.py.
"""
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
# 82 / 284 / 2230. Metal's gradients at n = 20000 are ~10x less accurate than
# CPU's, and the parallel solver's mean at n = 200000 is off by ~1%
# (1.7e5 ulps), where CPU float32 gives NaN.
ULPS = {  # (value, gradient, mean)
    ('quasisep-par', 1000): (64, 130, 980),
    ('quasisep-par', 20000): (240, 2000, 22000),
    ('quasisep-par', 200000): (2400, 4600, 340000),
    ('quasisep', 1000): (52, 190, 220),
    ('quasisep', 20000): (230, 1800, 2700),
    ('dense', 1000): (160, 170, 750),
    ('dense', 3000): (220, 680, 2200),
}


@pytest.mark.parametrize("kind,n", [
    ("quasisep-par", 1000), ("quasisep-par", 20000), ("quasisep-par", 200000),
    ("quasisep", 1000), ("quasisep", 20000), ("dense", 1000), ("dense", 3000)])
def test_tinygp(kind, n):
    fn, args = program(kind), (P0, *data(n))
    # tinygp's parallel condition() is NaN on CPU even in float64 for
    # n >= 20000, so the parallel solver is checked against the sequential
    # one (same model; the two agree to 1e-15 on the log-likelihood).
    want = f64_reference(program("quasisep" if kind == "quasisep-par" else kind), *args)
    assert not any(np.isnan(w).any() for w in want), "float64 reference has NaNs"
    got = run_on(metal(), fn, *args)
    cpu32 = run_on(cpu(), fn, *args) if REPORT else None
    for i, part in enumerate(["value", "gradient", "mean"]):
        assert_close(got[i], want[i], ULPS[kind, n][i], normwise=True,
                     name=f"{kind} n={n} {part}",
                     cpu32=None if cpu32 is None else cpu32[i])
