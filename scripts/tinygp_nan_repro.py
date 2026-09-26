"""Rerun the exact tinygp_check predict configuration repeatedly to see whether the NaN is deterministic."""
import sys, time
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, "scripts")
from tinygp_check import build  # noqa
n = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
rng = np.random.default_rng(0)
x = np.sort(rng.uniform(0, 100, n)).astype(np.float32); yerr = (0.1 + 0.1 * rng.random(n)).astype(np.float32)
y = (np.sin(x / 3) + 0.3 * rng.standard_normal(n)).astype(np.float32); xt = np.linspace(0, 100, 500, dtype=np.float32)
params = {"log_scale": jnp.float32(1.0), "log_sigma": jnp.float32(0.0), "log_omega": jnp.float32(0.5), "log_q": jnp.float32(1.0),
          "log_sigma2": jnp.float32(-0.5), "log_jitter": jnp.float32(-3.0), "mean": jnp.float32(0.0)}
for dev in (jax.devices("cpu")[0], [d for d in jax.devices() if d.platform == "metal"][0]):
    with jax.default_device(dev):
        xd, yd, yerrd, xtd = (jax.device_put(a, dev) for a in (x, y, yerr, xt))
        p = jax.tree_util.tree_map(lambda v: jax.device_put(v, dev), params)
        pred = jax.jit(lambda p: build("quasisep-par", p, xd, yerrd).condition(yd, xtd).gp.loc)
        for i in range(4):
            t0 = time.perf_counter(); mu = np.asarray(pred(p)); dt = time.perf_counter() - t0
            print(f"{dev.platform:6s} run {i}: NaNs {int(np.isnan(mu).sum()):4d}  mean {np.nanmean(mu):.5f}  first NaN idx {int(np.argmax(np.isnan(mu))) if np.isnan(mu).any() else -1}  ({dt*1e3:.0f} ms)", flush=True)
