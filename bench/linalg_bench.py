"""Median wall times (ms) of linear algebra on metal and CPU.

  scripts/device_lock.py -- env JAX_PLATFORMS=openmetal,cpu .venv/bin/python bench/linalg_bench.py [n ...]

Sizes default to 256, 1000, 3000; eigh / svd are only timed for n <= 1000.
"""
import sys

import numpy as np
import jax
import jax.numpy as jnp

cpu = jax.devices("cpu")[0]
rng = np.random.default_rng(0)


def spd(n):
  a = rng.standard_normal((n, n)).astype(np.float32)
  return (a @ a.T / n + np.eye(n, dtype=np.float32))


def mat(*s):
  return rng.standard_normal(s).astype(np.float32)


def timeit(dev, fn, *args, reps=5):
  import time
  with jax.default_device(dev):
    f = jax.jit(fn)
    xs = [jax.device_put(a, dev) for a in args]
    jax.block_until_ready(f(*xs))
    ts = []
    for _ in range(reps):
      t0 = time.perf_counter()
      jax.block_until_ready(f(*xs))
      ts.append(time.perf_counter() - t0)
  return 1e3 * float(np.median(ts))


def timings(sizes):
  ops = {
      "cholesky": lambda a, b: jnp.linalg.cholesky(a),
      "cho_solve": lambda a, b: jax.scipy.linalg.cho_solve((a, True), b),
      "solve (lu)": lambda a, b: jnp.linalg.solve(a, b),
      "gp nll+grad": lambda a, b: jax.value_and_grad(
          lambda k, y: 0.5 * y @ jax.scipy.linalg.cho_solve(
              jax.scipy.linalg.cho_factor(k, lower=True), y)
          + jnp.sum(jnp.log(jnp.diag(jnp.linalg.cholesky(k)))))(a, b[:, 0]),
      "eigh": lambda a, b: jnp.linalg.eigh(a),
      "svd": lambda a, b: jnp.linalg.svd(a),
  }
  print(f"\n{'op':<14}{'n':>6}{'metal ms':>11}{'cpu ms':>9}", flush=True)
  for n in sizes:
    a = spd(n)
    b = mat(n, 4)
    for name, fn in ops.items():
      if name in ("eigh", "svd") and n > 1000:
        continue
      try:
        tm = timeit(jax.devices()[0], fn, a, b)
      except Exception as e:  # noqa: BLE001
        tm = float("nan")
        print(f"  {name} {n}: {type(e).__name__}", flush=True)
      tc = timeit(cpu, fn, a, b)
      print(f"{name:<14}{n:>6}{tm:>11.1f}{tc:>9.1f}", flush=True)


if __name__ == "__main__":
  print("backend:", jax.default_backend())
  timings([int(x) for x in sys.argv[1:]] or [256, 1000, 3000])
