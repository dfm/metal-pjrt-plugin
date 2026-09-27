"""tinygp value+grad and predict timings (ms, mean of 5) on metal and CPU,
for the configurations tests/test_tinygp.py checks.

  scripts/device_lock.py -- env JAX_PLATFORMS=openmetal,cpu .venv/bin/python bench/tinygp_bench.py
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
import jax
from test_tinygp import P0, build, data  # noqa: E402


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
    (vm, pm), (vc, pc) = run(kind, n, jax.devices("openmetal")[0]), run(kind, n, jax.devices("cpu")[0])
    print(f"{kind:12s} n={n:6d} | value+grad: metal {vm:7.1f} ms, cpu {vc:7.1f} ms"
          f" | predict: metal {pm:7.1f} ms, cpu {pc:7.1f} ms", flush=True)
