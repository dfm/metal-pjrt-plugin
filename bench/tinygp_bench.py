# Copyright 2026 The metal-pjrt-plugin Authors
# SPDX-License-Identifier: Apache-2.0

"""tinygp value+grad and predict timings (ms, mean of 5) on metal and CPU:
the quasiseparable solver (parallel associative-scan and sequential) and
the dense solver. Needs tinygp installed (it is not a test dependency).

  scripts/device_lock.py -- .venv/bin/python bench/tinygp_bench.py
"""
import os, time
# mtl is opt-in (not the default backend).
os.environ.setdefault("JAX_PLATFORMS", "mtl,cpu")
import jax
import jax.numpy as jnp
import numpy as np
from tinygp import GaussianProcess, kernels
from tinygp.kernels import quasisep
from tinygp.solvers import QuasisepSolver

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


def run(kind, n, dev):
    x, y, yerr, xt = (jax.device_put(a, dev) for a in data(n))
    p = jax.device_put(P0, dev)
    vg = jax.jit(jax.value_and_grad(lambda p: -build(kind, p, x, yerr).log_probability(y)))
    pred = jax.jit(lambda p: build(kind, p, x, yerr).condition(y, xt).gp.loc)
    out = []
    for f in (vg, pred):
        jax.block_until_ready(f(p))
        t0 = time.perf_counter()
        for _ in range(5):
            jax.block_until_ready(f(p))
        out.append((time.perf_counter() - t0) / 5 * 1e3)
    return out


for kind, n in [("quasisep-par", 1000), ("quasisep-par", 20000), ("quasisep-par", 200000),
                ("quasisep", 1000), ("quasisep", 20000), ("dense", 1000), ("dense", 3000)]:
    (vm, pm), (vc, pc) = run(kind, n, jax.devices("mtl")[0]), run(kind, n, jax.devices("cpu")[0])
    print(f"{kind:12s} n={n:6d} | value+grad: metal {vm:7.1f} ms, cpu {vc:7.1f} ms"
          f" | predict: metal {pm:7.1f} ms, cpu {pc:7.1f} ms", flush=True)
