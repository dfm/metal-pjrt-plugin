"""tinygp on the Metal backend: correctness vs CPU and timing.

Exercises the quasiseparable (parallel associative-scan) solver and the
dense solver: log-likelihood, gradients, conditioning/prediction.
  scripts/device_lock.py -- env JAX_PLATFORMS=metal,cpu .venv/bin/python scripts/tinygp_check.py
"""
import sys, time
import numpy as np
import jax, jax.numpy as jnp
from tinygp import GaussianProcess, kernels
from tinygp.solvers import QuasisepSolver
from tinygp.kernels import quasisep

jax.config.update("jax_enable_x64", False)

def build(kind, params, x, yerr):
    if kind == "quasisep":
        k = quasisep.Matern32(scale=jnp.exp(params["log_scale"]), sigma=jnp.exp(params["log_sigma"])) + \
            quasisep.SHO(omega=jnp.exp(params["log_omega"]), quality=jnp.exp(params["log_q"]), sigma=jnp.exp(params["log_sigma2"]))
        return GaussianProcess(k, x, diag=yerr**2 + jnp.exp(2 * params["log_jitter"]), solver=QuasisepSolver, mean=params["mean"])
    k = jnp.exp(2 * params["log_sigma"]) * kernels.Matern32(scale=jnp.exp(params["log_scale"])) + \
        jnp.exp(2 * params["log_sigma2"]) * kernels.ExpSquared(scale=jnp.exp(params["log_omega"]))
    return GaussianProcess(k, x, diag=yerr**2 + jnp.exp(2 * params["log_jitter"]), mean=params["mean"])

def run(kind, n, dev):
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(0, 100, n)).astype(np.float32); yerr = (0.1 + 0.1 * rng.random(n)).astype(np.float32)
    y = (np.sin(x / 3) + 0.3 * rng.standard_normal(n)).astype(np.float32)
    xt = np.linspace(0, 100, 500, dtype=np.float32)
    params = {"log_scale": jnp.float32(1.0), "log_sigma": jnp.float32(0.0), "log_omega": jnp.float32(0.5),
              "log_q": jnp.float32(1.0), "log_sigma2": jnp.float32(-0.5), "log_jitter": jnp.float32(-3.0), "mean": jnp.float32(0.0)}
    with jax.default_device(dev):
        x, y, yerr, xt = (jax.device_put(a, dev) for a in (x, y, yerr, xt))
        params = jax.tree_util.tree_map(lambda v: jax.device_put(v, dev), params)
        def loss(p): return -build(kind, p, x, yerr).log_probability(y)
        vg = jax.jit(jax.value_and_grad(loss))
        pred = jax.jit(lambda p: build(kind, p, x, yerr).condition(y, xt).gp.loc)
        v, g = vg(params); jax.block_until_ready((v, g)); mu = pred(params); jax.block_until_ready(mu)
        t0 = time.perf_counter(); [jax.block_until_ready(vg(params)) for _ in range(5)]; t_vg = (time.perf_counter() - t0) / 5 * 1e3
        t0 = time.perf_counter(); [jax.block_until_ready(pred(params)) for _ in range(5)]; t_pred = (time.perf_counter() - t0) / 5 * 1e3
    return float(v), np.asarray(jax.tree_util.tree_leaves(g)), np.asarray(mu), t_vg, t_pred

def main():
    cpu = jax.devices("cpu")[0]; metal = [d for d in jax.devices() if d.platform == "metal"][0]
    ok = True
    for kind, n in [("quasisep", 1000), ("quasisep", 20000), ("quasisep", 200000), ("dense", 1000), ("dense", 3000)]:
        vc, gc, mc, tvc, tpc = run(kind, n, cpu)
        try:
            vm, gm, mm, tvm, tpm = run(kind, n, metal)
        except Exception as e:  # noqa
            ok = False; print(f"{kind:9s} n={n:6d}  metal FAILED: {type(e).__name__}: {str(e).splitlines()[0][:120]}"); continue
        rel = abs(vm - vc) / max(1.0, abs(vc)); grel = np.max(np.abs(gm - gc) / (1e-3 + np.abs(gc))); mrel = np.max(np.abs(mm - mc))
        good = rel < 1e-3 and grel < 5e-2 and mrel < 1e-2
        ok &= good
        print(f"{kind:9s} n={n:6d}  logp cpu {vc:12.3f} metal {vm:12.3f} (rel {rel:.1e})  grad maxrel {grel:.1e}  pred maxabs {mrel:.1e}  "
              f"| value+grad: cpu {tvc:7.1f} ms, metal {tvm:7.1f} ms | predict: cpu {tpc:7.1f} ms, metal {tpm:7.1f} ms  {'OK' if good else 'MISMATCH'}", flush=True)
    print("ALL OK" if ok else "SOME FAILURES"); return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
