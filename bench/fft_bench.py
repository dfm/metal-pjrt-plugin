# Copyright 2026 The metal-pjrt-plugin Authors
# SPDX-License-Identifier: Apache-2.0

"""FFTs on metal: metal$fft (MLX's kernels, the default lowering) vs the dense
DFT (the METAL_PJRT_DISABLE_FFT=1 lowering) vs MLX's own mx.fft, in one
process.

  scripts/device_lock.py -- .venv/bin/python bench/fft_bench.py [rounds]

Each round times a burst of BURST calls per arm and case (wall time per
call, in ms), the arms interleaved (GPU performance states make single
runs bimodal); the table reports p10 (the headline on a throttling
machine), median and p90 over the rounds. The DFT is O(n^2) per axis and
materializes n x n twiddles, so it only runs up to n = 4096.
"""
import os
import statistics
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "mtl,cpu")
import jax
import jax.numpy as jnp
import mlx.core as mx
import numpy as np

from metal_pjrt_plugin import _lowerings

BURST = 10
rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 15
rng = np.random.default_rng(0)


def cx(*s):
  return (rng.standard_normal(s) + 1j * rng.standard_normal(s)).astype(
      np.complex64)


def rl(*s):
  return rng.standard_normal(s).astype(np.float32)


# (name, jnp function, mx function, input, run the DFT)
CASES = [
    ("fft 17 x 4096 (Rader)", jnp.fft.fft, mx.fft.fft, cx(4096, 17), True),
    ("fft 1000 x 256 (Stockham)", jnp.fft.fft, mx.fft.fft, cx(256, 1000), True),
    ("fft 1024 x 256", jnp.fft.fft, mx.fft.fft, cx(256, 1024), True),
    ("fft 1021 x 256 (Bluestein)", jnp.fft.fft, mx.fft.fft, cx(256, 1021), True),
    ("rfft 4096 x 64", jnp.fft.rfft, mx.fft.rfft, rl(64, 4096), True),
    ("irfft 4096 x 64", lambda x: jnp.fft.irfft(x, 4096),
     lambda x: mx.fft.irfft(x, 4096), cx(64, 2049), True),
    ("fft 2053 x 16 (multi Bluestein)", jnp.fft.fft, mx.fft.fft, cx(16, 2053), True),
    ("fft2 512 x 512", jnp.fft.fft2, mx.fft.fft2, cx(512, 512), True),
    ("fft 2^16 x 16 (four-step)", jnp.fft.fft, mx.fft.fft, cx(16, 1 << 16), False),
    ("fft 2^20 (four-step)", jnp.fft.fft, mx.fft.fft, cx(1, 1 << 20), False),
    ("rfft 2^20 (four-step)", jnp.fft.rfft, mx.fft.rfft, rl(1, 1 << 20), False),
]


def jax_arm(fn, x, dft):
  """A compiled jit(fn): with the DFT lowering if `dft` (lowered while
  fft_disabled() says so; the flag is read at lowering time)."""
  saved = _lowerings.fft_disabled
  if dft:
    _lowerings.fft_disabled = lambda: True
  jax.clear_caches()  # lax.fft is a jit: its lowering is cached
  try:
    f = jax.jit(fn).lower(x).compile()
  finally:
    _lowerings.fft_disabled = saved
    jax.clear_caches()
  text = f.as_text()
  assert ("metal$fft" in text) != dft, "wrong lowering"
  xd = jax.device_put(x, jax.devices("mtl")[0])
  jax.block_until_ready(f(xd))

  def run():
    t0 = time.perf_counter()
    outs = [f(xd) for _ in range(BURST)]
    jax.block_until_ready(outs)
    return (time.perf_counter() - t0) * 1e3 / BURST
  return run


def mlx_arm(fn, x):
  xm = mx.array(x)
  mx.eval(fn(xm))

  def run():
    t0 = time.perf_counter()
    outs = [fn(xm) for _ in range(BURST)]
    mx.eval(outs)
    return (time.perf_counter() - t0) * 1e3 / BURST
  return run


def pct(ts, q):
  return float(np.percentile(ts, q))


def main():
  print(f"{'case':<34}{'arm':>8}{'p10 ms':>10}{'median':>10}{'p90':>10}"
        f"{'vs mlx':>9}", flush=True)
  for name, jfn, mfn, x, with_dft in CASES:
    arms = {"native": jax_arm(jfn, x, False), "mlx": mlx_arm(mfn, x)}
    if with_dft:
      arms["dft"] = jax_arm(jfn, x, True)
    times = {a: [] for a in arms}
    for _ in range(rounds):
      for a, run in arms.items():
        times[a].append(run())
    mlx_p10 = pct(times["mlx"], 10)
    for a in ("native", "dft", "mlx"):
      if a not in times:
        continue
      ts = times[a]
      print(f"{name:<34}{a:>8}{pct(ts, 10):>10.3f}"
            f"{statistics.median(ts):>10.3f}{pct(ts, 90):>10.3f}"
            f"{pct(ts, 10) / mlx_p10:>9.2f}", flush=True)
    del arms


if __name__ == "__main__":
  main()
