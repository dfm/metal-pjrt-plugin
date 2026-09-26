"""Localize the NaN in tinygp's parallel quasisep predictions on Metal by
comparing intermediates against CPU."""
import sys, time
import numpy as np, jax, jax.numpy as jnp
from tinygp import GaussianProcess
from tinygp.solvers import QuasisepSolver
from tinygp.kernels import quasisep

n = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
rng = np.random.default_rng(0)
x = np.sort(rng.uniform(0, 100, n)).astype(np.float32); y = (np.sin(x / 3) + 0.3 * rng.standard_normal(n)).astype(np.float32)
xt = np.linspace(0, 100, 500, dtype=np.float32)
cpu = jax.devices("cpu")[0]; metal = [d for d in jax.devices() if d.platform == "metal"][0]

def pieces(dev):
    with jax.default_device(dev):
        xd, yd, xtd = (jax.device_put(a, dev) for a in (x, y, xt))
        k = quasisep.Matern32(scale=1.0, sigma=1.0)
        gp = GaussianProcess(k, xd, diag=0.01, solver=QuasisepSolver, assume_sorted=True, parallel=True)
        out = {}
        f = jax.jit(lambda yy: gp.solver.solve_triangular(yy)); out["solve_lower"] = f(yd)
        g = jax.jit(lambda yy: gp.solver.solve_triangular(yy, transpose=True)); out["solve_upper_of_lower"] = g(out["solve_lower"])
        out["logp"] = jax.jit(lambda yy: gp.log_probability(yy))(yd)
        cond = jax.jit(lambda yy: gp.condition(yy, xtd).gp.loc); out["pred_loc"] = cond(yd)
        h = jax.jit(lambda yy: gp.solver.normalization()); out["normalization"] = h(yd)
        m = jax.jit(lambda yy: gp.solver.matrix.matmul(yy, parallel=True) if hasattr(gp.solver.matrix, "matmul") else yy); out["K_matmul"] = m(yd)
        return {kk: np.asarray(v) for kk, v in out.items()}

t0 = time.perf_counter(); c = pieces(cpu); print("cpu pieces %.1fs" % (time.perf_counter() - t0), flush=True)
t0 = time.perf_counter(); m = pieces(metal); print("metal pieces %.1fs" % (time.perf_counter() - t0), flush=True)
for kk in c:
    a, b = c[kk], m[kk]
    nans = int(np.isnan(b).sum()); first = int(np.argmax(np.isnan(b.ravel()))) if nans else -1
    err = np.nanmax(np.abs(a - b)) if a.size else 0
    print(f"{kk:22s} shape {a.shape} metal NaNs {nans:6d} (first idx {first})  max|cpu-metal| {err:.3e}  cpu range [{np.nanmin(a):.3g}, {np.nanmax(a):.3g}]", flush=True)
