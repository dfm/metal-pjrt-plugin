"""Compare linear algebra / FFT results on the metal backend against CPU.

  JAX_PLATFORMS=metal,cpu python scripts/linalg_check.py [-v] [--time [N ...]]

Float32 cholesky / triangular_solve go through Accelerate LAPACK (HLO
rewriter), lu / qr / eigh / svd through the LAPACK JAX lowerings
(jax_plugins/metal/linalg_lowerings.py). METAL_PJRT_DISABLE_LAPACK=1 restores
XLA's expanders / the pure-JAX paths for A/B comparisons. `--time` prints
median wall times (ms) on metal and CPU at n = 256, 1000, 3000 (or the given
sizes); eigh / svd are only timed for n <= 1000.

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


def gp_nll(a, y):
  k = a @ a.T / a.shape[0] + jnp.eye(a.shape[0])
  c = jax.scipy.linalg.cho_factor(k, lower=True)
  return (0.5 * y @ jax.scipy.linalg.cho_solve(c, y)
          + jnp.sum(jnp.log(jnp.diag(c[0]))))


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
    # Batched / variant inputs (LAPACK paths).
    "cholesky batched 3x40": (lambda a: jnp.linalg.cholesky(a), np.stack([spd(40) for _ in range(3)])),
    "cholesky not PD -> nan": (lambda a: jnp.isnan(jnp.linalg.cholesky(a)).all(), -spd(8)),
    **{f"triangular_solve left={l} lower={lo} trans={t} unit={u}": (
        (lambda l, lo, t, u: lambda a, b: jax.lax.linalg.triangular_solve(
            a, b if l else jnp.swapaxes(b, -1, -2), left_side=l, lower=lo,
            transpose_a=t, unit_diagonal=u))(l, lo, t, u),
        np.stack([mat(9, 9) + 6 * np.eye(9, dtype=np.float32) for _ in range(2)]),
        mat(2, 9, 4))
       for l in (True, False) for lo in (True, False) for t in (False, True)
       for u in (False, True)},
    "eigh upper (asymmetric input)": (lambda a: jax.lax.linalg.eigh(a, lower=False, symmetrize_input=False)[1], mat(10, 10)),
    "eigh lower (asymmetric input)": (lambda a: jax.lax.linalg.eigh(a, lower=True, symmetrize_input=False)[1], mat(10, 10)),
    "eigh batched vectors": (lambda a: jax.vmap(eigh_check)(a), np.stack([spd(20) for _ in range(3)])),
    "svd full 7x5": (lambda a: (lambda u, s, vt: (s, jnp.abs(u[:, :5] * s @ vt - a).max() < 1e-3,
                                                  jnp.abs(u.T @ u - jnp.eye(7)).max() < 1e-3,
                                                  jnp.abs(vt @ vt.T - jnp.eye(5)).max() < 1e-3))(*jnp.linalg.svd(a)), mat(7, 5)),
    "svd batched 2x6x9": (lambda a: jax.vmap(svd_check)(a), mat(2, 6, 9)),
    "qr complete 9x5": (lambda a: (lambda q, r: (jnp.abs(r), jnp.abs(q @ r - a).max() < 1e-3,
                                                 jnp.abs(q.T @ q - jnp.eye(9)).max() < 1e-3))(*jnp.linalg.qr(a, mode="complete")), mat(9, 5)),
    "qr batched 3x6x4": (lambda a: jax.vmap(qr_check)(a), mat(3, 6, 4)),
    "lu 7x5 / 5x7": (lambda a, b: (lu_check(a), lu_check(b)), mat(7, 5), mat(5, 7)),
    "lu batched pivots/perm": (lambda a: jax.lax.linalg.lu(a), mat(3, 12, 12)),
    "solve batched": (lambda a, b: jnp.linalg.solve(a + 4 * jnp.eye(10), b), mat(3, 10, 10), mat(3, 10, 2)),
    # Gradients through the LAPACK paths.
    "grad cholesky": (lambda a: jax.grad(lambda x: jnp.sum(jnp.linalg.cholesky(x @ x.T + 20 * jnp.eye(20)) ** 2))(a), mat(20, 20)),
    "grad solve": (lambda a, b: jax.grad(lambda x: jnp.sum(jnp.linalg.solve(x + 5 * jnp.eye(12), b) ** 2))(a), mat(12, 12), mat(12, 3)),
    "grad cho_solve/logdet": (lambda a, y: jax.grad(gp_nll)(a, y), mat(30, 30), mat(30)),
    "grad eigh": (lambda a: jax.grad(lambda x: jnp.sum(jnp.linalg.eigh(x + x.T)[0] ** 3))(a), mat(10, 10)),
    "grad slogdet": (lambda a: jax.grad(lambda x: jnp.linalg.slogdet(x + 5 * jnp.eye(12))[1])(a), mat(12, 12)),
    # Known failures: these materialize complex intermediates in device
    # memory, which the MSL emitter does not support yet (not a lowering issue).
    "[complex buffers] fft2 * 2 / ifftn": (lambda x: jnp.abs(jnp.fft.ifftn(jnp.fft.fft2(x) * 2)), mat(8, 12)),
    "[complex buffers] fft grad": (lambda x: jax.grad(lambda v: jnp.sum(jnp.abs(jnp.fft.rfft(v)) ** 2))(x), mat(32)),
}


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


def main():
  if "--time" in sys.argv:
    sizes = [int(x) for x in sys.argv[sys.argv.index("--time") + 1:]
             if x.isdigit()] or [256, 1000, 3000]
    print("backend:", jax.default_backend())
    timings(sizes)
    return 0
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
