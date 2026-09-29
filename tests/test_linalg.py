"""Linear algebra / FFT on metal against a float64 CPU reference
(tests/metal_testing.py), normwise in ulps per output.

Float32 cholesky / triangular_solve go through Accelerate LAPACK (HLO
rewriter), lu / qr / eigh / svd through the LAPACK JAX lowerings
(metal_pjrt_plugin/_linalg_lowerings.py). METAL_PJRT_DISABLE_LAPACK=1 restores
XLA's expanders / the pure-JAX paths for A/B comparisons. Timings:
bench/linalg_bench.py.

Decompositions are compared through invariants (reconstruction, orthogonality,
sorted spectra) rather than raw factors, which are only unique up to signs /
phases. FFT (metal$fft; tests/test_fft.py covers it per path) checks
reduce to real outputs (abs / real / imag), which the
f64 reference compares in ulps.
"""
import os
import re

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from metal_testing import check, run_python

pytestmark = pytest.mark.metal

rng = np.random.default_rng(0)


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
    # Small matrices run as GPU kernels (n <= 32); larger ones on the host.
    "cholesky batched 5x4": (lambda a: jnp.linalg.cholesky(a), np.stack([spd(4) for _ in range(5)])),
    "cholesky 32 / 33": (lambda a, b: (jnp.linalg.cholesky(a), jnp.linalg.cholesky(b)), spd(32), spd(33)),
    "cholesky upper small": (lambda a: jax.lax.linalg.cholesky(a, symmetrize_input=False), spd(6)),
    "cholesky batched not PD -> nan rows": (lambda a: jnp.isnan(jnp.linalg.cholesky(a)).all(axis=(1, 2)), np.stack([spd(5), -spd(5), spd(5)])),
    "lu batched 4x4 / 2x2": (lambda a, b: (jax.lax.linalg.lu(a), jax.lax.linalg.lu(b)), mat(7, 4, 4), mat(3, 2, 2)),
    "lu small non-square 5x3 / 3x5": (lambda a, b: (lu_check(a), lu_check(b)), mat(5, 3), mat(3, 5)),
    "lu 32 / 33": (lambda a, b: (lu_check(a), lu_check(b)), mat(32, 32), mat(33, 33)),
    "solve batched 4x4": (lambda a, b: jnp.linalg.solve(a + 3 * jnp.eye(4), b), mat(50, 4, 4), mat(50, 4, 4)),
    "solve batched 4x4 grad": (lambda a, b: jax.grad(lambda x: jnp.sum(jnp.linalg.solve(x + 3 * jnp.eye(4), b) ** 2))(a), mat(6, 4, 4), mat(6, 4, 2)),
    **{f"triangular_solve 3x3 left={l} lower={lo} trans={t} unit={u}": (
        (lambda l, lo, t, u: lambda a, b: jax.lax.linalg.triangular_solve(
            a, b if l else jnp.swapaxes(b, -1, -2), left_side=l, lower=lo,
            transpose_a=t, unit_diagonal=u))(l, lo, t, u),
        np.stack([mat(3, 3) + 4 * np.eye(3, dtype=np.float32) for _ in range(4)]),
        mat(4, 3, 5))
       for l in (True, False) for lo in (True, False) for t in (False, True)
       for u in (False, True)},
    # Gradients through the LAPACK paths.
    "grad cholesky": (lambda a: jax.grad(lambda x: jnp.sum(jnp.linalg.cholesky(x @ x.T + 20 * jnp.eye(20)) ** 2))(a), mat(20, 20)),
    "grad solve": (lambda a, b: jax.grad(lambda x: jnp.sum(jnp.linalg.solve(x + 5 * jnp.eye(12), b) ** 2))(a), mat(12, 12), mat(12, 3)),
    "grad cho_solve/logdet": (lambda a, y: jax.grad(gp_nll)(a, y), mat(30, 30), mat(30)),
    "grad eigh": (lambda a: jax.grad(lambda x: jnp.sum(jnp.linalg.eigh(x + x.T)[0] ** 3))(a), mat(10, 10)),
    "grad slogdet": (lambda a: jax.grad(lambda x: jnp.linalg.slogdet(x + 5 * jnp.eye(12))[1])(a), mat(12, 12)),
    "fft2 * 2 / ifftn (complex buffers)": (lambda x: jnp.abs(jnp.fft.ifftn(jnp.fft.fft2(x) * 2)), mat(8, 12)),
    "fft grad (complex buffers)": (lambda x: jax.grad(lambda v: jnp.sum(jnp.abs(jnp.fft.rfft(v)) ** 2))(x), mat(32)),
}



# Measured (METAL_TEST_REPORT_ULPS=1, M3) and doubled; normwise ulps.
ULPS = {
    'cholesky 16': 1.4, 'cholesky 300': 4.3, 'eigh 8': 2, 'eigh 64': 6,
    'eigh 300 (QDWH path)': 5.9, 'eigvalsh batched': 4.2, 'svd 12x8': 6,
    'svd 8x12': 5.6, 'svd vals 40x40': 6.9, 'qr 12x8': 2.2, 'qr 8x12': 4.6,
    'lu 16': 7.1, 'solve 16': 6.2, 'inv 16': 12, 'det/slogdet 16': 1.7,
    'cho_solve 16': 8.1, 'fft abs': 3.4, 'fft re/im odd n': 5,
    'ifft(fft) roundtrip': 4, 'rfft': 4.3, 'irfft(rfft) even/odd': 6,
    'irfft of arbitrary spectrum': 5.4, 'fft2 / ifftn': 4,
    'rfft2 / irfft2': 4, 'cholesky batched 3x40': 1.2,
    'triangular_solve left=True lower=True trans=False unit=False': 1.6,
    'triangular_solve left=True lower=True trans=False unit=True': 2.7,
    'triangular_solve left=True lower=True trans=True unit=False': 1,
    'triangular_solve left=True lower=True trans=True unit=True': 1.8,
    'triangular_solve left=True lower=False trans=False unit=False': 2.2,
    'triangular_solve left=True lower=False trans=False unit=True': 3.4,
    'triangular_solve left=True lower=False trans=True unit=False': 1,
    'triangular_solve left=True lower=False trans=True unit=True': 1,
    'triangular_solve left=False lower=True trans=False unit=False': 3.1,
    'triangular_solve left=False lower=True trans=False unit=True': 2.5,
    'triangular_solve left=False lower=True trans=True unit=False': 1.7,
    'triangular_solve left=False lower=True trans=True unit=True': 2.2,
    'triangular_solve left=False lower=False trans=False unit=False': 5.6,
    'triangular_solve left=False lower=False trans=False unit=True': 2.3,
    'triangular_solve left=False lower=False trans=True unit=False': 2.3,
    'triangular_solve left=False lower=False trans=True unit=True': 3,
    'eigh upper (asymmetric input)': 5.2,
    'eigh lower (asymmetric input)': 6.6, 'eigh batched vectors': 3.6,
    'svd full 7x5': 8.9, 'svd batched 2x6x9': 6.8, 'qr complete 9x5': 1.8,
    'qr batched 3x6x4': 2, 'lu 7x5 / 5x7': 1.5, 'lu batched pivots/perm': 3.6,
    'solve batched': 3.5, 'cholesky batched 5x4': 1.6,
    'cholesky 32 / 33': 2.3, 'cholesky upper small': 1.4,
    'lu batched 4x4 / 2x2': 1, 'lu small non-square 5x3 / 3x5': 1,
    'lu 32 / 33': 19, 'solve batched 4x4': 1.7, 'solve batched 4x4 grad': 3.9,
    'triangular_solve 3x3 left=True lower=True trans=False unit=False': 1.3,
    'triangular_solve 3x3 left=True lower=True trans=False unit=True': 1.1,
    'triangular_solve 3x3 left=True lower=True trans=True unit=False': 1.7,
    'triangular_solve 3x3 left=True lower=True trans=True unit=True': 1.2,
    'triangular_solve 3x3 left=True lower=False trans=False unit=False': 1.2,
    'triangular_solve 3x3 left=True lower=False trans=False unit=True': 1,
    'triangular_solve 3x3 left=True lower=False trans=True unit=False': 1,
    'triangular_solve 3x3 left=True lower=False trans=True unit=True': 1,
    'triangular_solve 3x3 left=False lower=True trans=False unit=False': 1.9,
    'triangular_solve 3x3 left=False lower=True trans=False unit=True': 1.8,
    'triangular_solve 3x3 left=False lower=True trans=True unit=False': 1.4,
    'triangular_solve 3x3 left=False lower=True trans=True unit=True': 1,
    'triangular_solve 3x3 left=False lower=False trans=False unit=False': 1,
    'triangular_solve 3x3 left=False lower=False trans=False unit=True': 1,
    'triangular_solve 3x3 left=False lower=False trans=True unit=False': 1.9,
    'triangular_solve 3x3 left=False lower=False trans=True unit=True': 1.1,
    'grad cholesky': 9, 'grad solve': 7.4, 'grad cho_solve/logdet': 13,
    'grad eigh': 12, 'grad slogdet': 2.7,
    'fft2 * 2 / ifftn (complex buffers)': 4,
    'fft grad (complex buffers)': 2.6,
}


@pytest.mark.parametrize("name", list(CHECKS))
def test_linalg(name):
  fn, *args = CHECKS[name]
  check(fn, *args, ulps=ULPS.get(name, 0), normwise=True, name=name)


DISABLE_LAPACK_CHILD = r"""
import re, sys, numpy as np, jax, jax.numpy as jnp
def f(a, b):
    l = jnp.linalg.cholesky(a)
    x = jax.scipy.linalg.solve_triangular(l, b, lower=True)
    lu = jax.scipy.linalg.lu_factor(a)[0]
    q, r = jnp.linalg.qr(a)
    w = jnp.linalg.eigh(a)[0]
    u, s, vt = jnp.linalg.svd(a)
    return x, lu, q, r, w, u, s, vt, jnp.linalg.svd(a, compute_uv=False)
m = np.random.default_rng(0).standard_normal((8, 8)).astype(np.float32)
a, b = m @ m.T + 8 * np.eye(8, dtype=np.float32), m[:, :2]
text = jax.jit(f).lower(a, b).compile().as_text()
print(sorted(set(re.findall(r'custom_call_target="(metal\$[a-z_]+)"', text))))
"""


@pytest.mark.parametrize("value", [None, "0", "false", "1", "yes"])
def test_disable_lapack(value):
  # METAL_PJRT_DISABLE_LAPACK is the one switch for both owners (the C++
  # rewriter and _linalg_lowerings.py), read once per process as a boolean
  # (runtime/env.h): "1"/"yes" leave no metal$* LAPACK custom call; unset,
  # "0" or "false" keep every one.
  env = {k: v for k, v in os.environ.items()
         if k != "METAL_PJRT_DISABLE_LAPACK"}
  if value is not None:
    env["METAL_PJRT_DISABLE_LAPACK"] = value
  out = run_python(DISABLE_LAPACK_CHILD, dict(env, JAX_PLATFORMS="mtl"))
  assert out.returncode == 0, out.stderr[-3000:]
  want = [] if value in ("1", "yes") else sorted(
      ["metal$cholesky", "metal$triangular_solve", "metal$lapack_getrf",
       "metal$lapack_geqrf", "metal$lapack_orgqr", "metal$lapack_syevd",
       "metal$lapack_gesdd", "metal$lapack_gesdd_novec"])
  assert out.stdout.strip() == str(want)


# Host LAPACK (n > 32, lapack_ffi.cc) between GPU work in one program: the
# GPU work before it and after it must see it in order, at sizes from just
# above the GPU small-matrix path to large.
@pytest.mark.parametrize("n", [33, 100, 500, 2000])
def test_host_lapack_in_order(n):
  import scipy.linalg
  a = mat(n, n)
  eye = np.eye(n, dtype=np.float32)
  b = mat(n, 3)

  @jax.jit
  def f(a, b):
    k = a @ a.T / n + eye  # GPU before
    l = jnp.linalg.cholesky(k)
    x = jax.scipy.linalg.solve_triangular(l, b, lower=True)
    lu, piv = jax.scipy.linalg.lu_factor(k + a)
    return l * 2.0, x + 1.0, lu - 1.0, piv  # GPU after

  l, x, lu, piv = (np.asarray(t, np.float64) for t in f(a, b))
  a64 = a.astype(np.float64)
  k = a64 @ a64.T / n + np.eye(n)
  l_ref = np.linalg.cholesky(k)
  x_ref = scipy.linalg.solve_triangular(l_ref, b, lower=True)
  lu_ref, piv_ref = scipy.linalg.lu_factor(k + a64)
  err = lambda got, want: np.abs(got - want).max() / np.abs(want).max()
  assert err(l / 2, l_ref) < 1e-4
  assert err(x - 1, x_ref) < 1e-4
  if np.array_equal(piv, piv_ref):  # same pivots: same factors
    assert err(lu + 1, lu_ref) < 1e-3
  pa = k + a64  # row swaps applied in order (0-based pivots) give P A = L U
  for i, j in enumerate(piv.astype(int)):
    pa[[i, j]] = pa[[j, i]]
  lo = np.tril(lu + 1, -1) + np.eye(n)
  assert err(lo @ np.triu(lu + 1), pa) < 1e-4


# Data-dependent LAPACK outcomes are values (NaN, as on JAX's CPU backend),
# never errors, and leave the device usable (were LAPACK ever moved into a
# stream host task, a failing task would set the device's sticky error and
# break every later computation in the process). A malformed call fails only
# that call. Fresh process, so a regression cannot poison the rest of the
# suite.
FAILURE_CHILD = r"""
import numpy as np, jax, jax.numpy as jnp
f = jax.jit(lambda x: jnp.sin(x) * 2.0)
x = np.arange(64, dtype=np.float32)
def healthy():
  return np.allclose(np.asarray(f(x)), np.sin(x.astype(np.float64)) * 2, atol=1e-5)
def run(name, fn):
  try:
    out = [np.asarray(t) for t in jax.tree.leaves(fn())]
    res = "nan" if any(np.isnan(t).any() for t in out) else "finite"
  except Exception as e:
    res = "raised " + str(e).splitlines()[0][:100]
  print(f"{name}: {res} healthy={healthy()}")
n = 64
a = np.random.default_rng(0).standard_normal((n, n)).astype(np.float32)
sing = a.copy(); sing[:, 3] = 0
nanm = a @ a.T; nanm[5, 7] = nanm[7, 5] = np.nan
run("cholesky not PD", lambda: jnp.linalg.cholesky(-np.eye(n, dtype=np.float32)))
run("solve singular", lambda: jnp.linalg.solve(sing, a[:, :2]))
run("lu singular", lambda: jax.scipy.linalg.lu_factor(sing))
run("eigh nan", lambda: jnp.linalg.eigh(nanm))
run("svd nan", lambda: jnp.linalg.svd(nanm))
run("svd novec nan", lambda: jnp.linalg.svd(nanm, compute_uv=False))
run("bad call", lambda: jax.ffi.ffi_call(
    "metal$cholesky", jax.ShapeDtypeStruct((40, 48), jnp.float32))(
        np.ones((40, 48), np.float32), lower=True))
"""


def test_lapack_failures_are_values():
  import os
  env = dict(os.environ, JAX_PLATFORMS="mtl")
  # No timeout: never kill a process with GPU work in flight.
  out = run_python(FAILURE_CHILD, env)
  got = dict(l.split(": ", 1) for l in out.stdout.splitlines() if ": " in l)
  assert out.returncode == 0 and len(got) == 7, (out.returncode, got, out.stderr[-2000:])
  assert all(v.endswith("healthy=True") for v in got.values()), got
  expect = {"cholesky not PD": "nan", "solve singular": "nan",
            "lu singular": "finite", "eigh nan": "nan", "svd nan": "nan",
            "svd novec nan": "nan"}
  for k, v in expect.items():
    assert got[k].startswith(v + " "), (k, got[k])
  assert got["bad call"].startswith("raised INVALID_ARGUMENT"), got["bad call"]


# LAPACK returns workspace sizes as floats: above 2^24 the value can round
# below the minimum (ssyevd at n=3000 needs 1 + 6n + 2n^2 = 18,018,001 and
# reported 18,018,000), which LAPACK rejects (info=-8) and which then failed
# the call. Fresh process: such a failure used to poison the device.
BIG_EIGH_CHILD = r"""
import numpy as np, jax, jax.numpy as jnp
n = 3000
a = np.random.default_rng(0).standard_normal((n, n)).astype(np.float32)
a = (a + a.T) / 2
f = lambda a: jnp.linalg.eigh(a)[0]
got = np.asarray(jax.jit(f)(jax.device_put(a, jax.devices("mtl")[0])))
want = np.asarray(jax.jit(f)(jax.device_put(a, jax.devices("cpu")[0])))
print("err:", np.abs(got - want).max() / np.abs(want).max())
"""


def test_eigh_workspace_above_2_24():
  import os
  env = dict(os.environ, JAX_PLATFORMS="mtl,cpu")
  out = run_python(BIG_EIGH_CHILD, env)
  assert out.returncode == 0 and "err:" in out.stdout, (out.returncode, out.stdout, out.stderr[-2000:])
  assert float(out.stdout.split("err:")[1]) < 1e-5, out.stdout
