"""Compare linear algebra / FFT results on the metal backend against CPU.

  JAX_PLATFORMS=metal,cpu python scripts/linalg_check.py

Decompositions are compared through invariants (reconstruction, orthogonality,
sorted spectra) rather than raw factors, which are only unique up to signs /
phases. Complex values cannot currently live in metal buffers, so FFT checks
reduce to real outputs (abs / real / imag) inside the jitted function.
"""
import sys
import traceback

import numpy as np
import jax
import jax.numpy as jnp

cpu = jax.devices("cpu")[0]
rng = np.random.default_rng(0)


def on(dev, fn, *args):
  with jax.default_device(dev):
    out = jax.jit(fn)(*[jax.device_put(a, dev) for a in args])
    return jax.tree.map(np.asarray, out)


def compare(fn, *args, rtol=1e-3, atol=1e-3):
  got = on(jax.devices()[0], fn, *args)
  want = on(cpu, fn, *args)
  jax.tree.map(lambda g, w: np.testing.assert_allclose(g, w, rtol=rtol, atol=atol), got, want)


def spd(n):
  a = rng.standard_normal((n, n)).astype(np.float32)
  return (a @ a.T / n + np.eye(n, dtype=np.float32))


def mat(*s):
  return rng.standard_normal(s).astype(np.float32)


def eigh_check(a):
  w, v = jnp.linalg.eigh(a)
  resid = jnp.abs(a @ v - v * w).max() / jnp.abs(w).max()
  ortho = jnp.abs(v.T @ v - jnp.eye(a.shape[0])).max()
  return w, jnp.array([resid < 1e-3, ortho < 1e-3]).astype(jnp.float32)


def svd_check(a):
  u, s, vt = jnp.linalg.svd(a, full_matrices=False)
  recon = jnp.abs(u * s @ vt - a).max() / s.max()
  return s, (recon < 1e-3).astype(jnp.float32)


def qr_check(a):
  q, r = jnp.linalg.qr(a)
  return jnp.abs(r), (jnp.abs(q @ r - a).max() < 1e-3).astype(jnp.float32)


def lu_check(a):
  p, l, u = jax.scipy.linalg.lu(a)
  return jnp.abs(p @ l @ u - a).max() < 1e-3, l, u


CHECKS = {
    "cholesky 16": (lambda a: jnp.linalg.cholesky(a), spd(16)),
    "cholesky 300": (lambda a: jnp.linalg.cholesky(a), spd(300)),
    "eigh 8": (eigh_check, spd(8)),
    "eigh 64": (eigh_check, spd(64)),
    "eigh 300 (QDWH path)": (eigh_check, spd(300)),
    "eigvalsh batched": (lambda a: jnp.linalg.eigvalsh(a), np.stack([spd(12) for _ in range(3)])),
    "svd 12x8": (svd_check, mat(12, 8)),
    "svd 8x12": (svd_check, mat(8, 12)),
    "svd vals 40x40": (lambda a: jnp.linalg.svd(a, compute_uv=False), mat(40, 40)),
    "qr 12x8": (qr_check, mat(12, 8)),
    "qr 8x12": (qr_check, mat(8, 12)),
    "lu 16": (lu_check, mat(16, 16)),
    "solve 16": (lambda a, b: jnp.linalg.solve(a + 4 * jnp.eye(16), b), mat(16, 16), mat(16, 3)),
    "inv 16": (lambda a: jnp.linalg.inv(a + 4 * jnp.eye(16)), mat(16, 16)),
    "det/slogdet 16": (lambda a: jnp.linalg.slogdet(a + 4 * jnp.eye(16)), mat(16, 16)),
    "cho_solve 16": (lambda a, b: jax.scipy.linalg.cho_solve(jax.scipy.linalg.cho_factor(a), b), spd(16), mat(16, 2)),
    "fft abs": (lambda x: jnp.abs(jnp.fft.fft(x)), mat(64)),
    "fft re/im odd n": (lambda x: (jnp.fft.fft(x).real, jnp.fft.fft(x).imag), mat(4, 45)),
    "ifft(fft) roundtrip": (lambda x: jnp.fft.ifft(jnp.fft.fft(x)).real, mat(33)),
    "rfft": (lambda x: (jnp.fft.rfft(x).real, jnp.fft.rfft(x).imag), mat(3, 64)),
    "irfft(rfft) even/odd": (lambda x, y: (jnp.fft.irfft(jnp.fft.rfft(x), 64), jnp.fft.irfft(jnp.fft.rfft(y), 63)), mat(2, 64), mat(63)),
    "irfft of arbitrary spectrum": (lambda a, b: jnp.fft.irfft(a + 1j * b, 20), mat(3, 11), mat(3, 11)),
    "fft2 / ifftn": (lambda x: jnp.abs(jnp.fft.ifftn(jnp.fft.fft2(x))), mat(8, 12)),
    "rfft2 / irfft2": (lambda x: (jnp.abs(jnp.fft.rfft2(x)), jnp.fft.irfft2(jnp.fft.rfft2(x), (8, 16))), mat(8, 16)),
    # Known failures: these materialize complex intermediates in device
    # memory, which the MSL emitter does not support yet (not a lowering issue).
    "[complex buffers] fft2 * 2 / ifftn": (lambda x: jnp.abs(jnp.fft.ifftn(jnp.fft.fft2(x) * 2)), mat(8, 12)),
    "[complex buffers] fft grad": (lambda x: jax.grad(lambda v: jnp.sum(jnp.abs(jnp.fft.rfft(v)) ** 2))(x), mat(32)),
}


def main():
  print("backend:", jax.default_backend())
  fails = 0
  for name, (fn, *args) in CHECKS.items():
    try:
      compare(fn, *args)
      print(f"PASS {name}", flush=True)
    except Exception as e:  # noqa: BLE001
      fails += 1
      msg = (str(e).strip().splitlines() or [""])[0][:200]
      print(f"FAIL {name}: {type(e).__name__}: {msg}", flush=True)
      if "-v" in sys.argv:
        traceback.print_exc()
  print(f"\n{len(CHECKS) - fails}/{len(CHECKS)} passed")
  return 1 if fails else 0


if __name__ == "__main__":
  sys.exit(main())
