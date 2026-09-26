"""MLIR lowering rules for primitives whose upstream rules are registered only
for named platforms (cpu/cuda/rocm/tpu), so that they also lower on "metal".

Policy: reuse JAX's own platform-independent implementations wherever one
exists, i.e. do what the TPU platform does (TPU is JAX's reference "no vendor
solver library" platform). Everything here is pure Python on top of generic
HLO; nothing requires C++ support from the plugin beyond the XLA expander
passes the GPU compiler pipeline already runs (CholeskyExpander, QrExpander,
EighExpander, TriangularSolveExpander).

Registered for platform "metal":

* ``eigh``: ``jax._src.tpu.linalg.eigh._eigh_tpu_lowering``. n <= 256 emits
  the XLA ``Eigh`` custom call (Jacobi; rewritten by EighExpander), larger
  matrices use the pure-JAX QDWH spectral divide-and-conquer. This also fixes
  ``svd``, whose generic lowering (``_svd_tpu_lowering_rule``) is QDWH-based
  and calls ``eigh``.
* ``fft``: a pure-JAX dense DFT (real matmuls against in-graph twiddle
  matrices). O(n^2) per transformed axis -- correct but slow; a stopgap until
  there is a native FFT (XLA's FftThunk is cuFFT/hipFFT-only).
* ``check`` (checkify): the TPU rule, i.e. a no-op for ``debug=True`` checks
  and the usual "functionalize with checkify" error otherwise (the runtime
  error path needs host callbacks).
* ``debug_callback`` / ``debug_print``: the upstream cpu/gpu rule. Lowering
  goes through ``emit_python_callback``, which jax_plugins/metal/callbacks.py
  redirects to the metal host-callback custom call. See docs/callbacks.md.

Primitives that already lower via generic rules and need nothing here:
cholesky (CholeskyExpander), triangular_solve, lu (``_lu_python``),
lu_pivots_to_permutation, geqrf/householder_product/qr (XLA ``Qr`` /
``ProductOfElementaryHouseholderReflectors`` custom calls -> QrExpander),
svd, tridiagonal_solve, threefry2x32, rng_bit_generator, approx_top_k
(generic fallback), cholesky_update, symmetric_product.

Deliberately NOT registered (no platform-independent implementation exists
in JAX; unsupported on TPU too): eig, schur, hessenberg, tridiagonal, geqp3.
GPU-only/vendor primitives (cudnn_fusion, dot_product_attention_*,
pbroadcast) are also out of scope.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

PLATFORM = "metal"


# --------------------------------------------------------------------------
# FFT as a dense DFT
# --------------------------------------------------------------------------

def _twiddles(n_out, n_in, period, dtype):
  """cos/sin of 2*pi*(row*col mod period)/period as (n_out, n_in) arrays.

  Reducing the phase index modulo `period` in integer arithmetic keeps the
  angles in [0, 2*pi), so accuracy does not degrade with n (int32 products
  are exact for n <= 46340)."""
  from jax import lax
  import jax.numpy as jnp
  r = lax.broadcasted_iota(jnp.int32, (n_out, n_in), 0)
  c = lax.broadcasted_iota(jnp.int32, (n_out, n_in), 1)
  phase = lax.rem(r * c, jnp.int32(period)).astype(dtype)
  ang = phase * jnp.asarray(2 * np.pi / period, dtype)
  return jnp.cos(ang), jnp.sin(ang)


def _mm(x, m):
  """Contract the last axis of x with axis 1 of m: (..., in) x (out, in)."""
  from jax import lax
  return lax.dot_general(x, m, (((x.ndim - 1,), (1,)), ((), ())),
                         precision=lax.Precision.HIGHEST)


def _c2c_last(re, im, inverse):
  n = re.shape[-1]
  cos, sin = _twiddles(n, n, n, re.dtype)
  s = 1.0 if inverse else -1.0
  # (re + i im)(cos + s i sin)
  yr = _mm(re, cos) - s * _mm(im, sin)
  yi = _mm(im, cos) + s * _mm(re, sin)
  if inverse:
    yr, yi = yr / n, yi / n
  return yr, yi


def _r2c_last(x):
  n = x.shape[-1]
  cos, sin = _twiddles(n // 2 + 1, n, n, x.dtype)
  return _mm(x, cos), -_mm(x, sin)


def _c2r_last(re, im, n):
  import jax.numpy as jnp
  m = n // 2 + 1
  assert re.shape[-1] == m, (re.shape, n)
  cos, sin = _twiddles(n, m, n, re.dtype)
  # Hermitian weights: DC (and Nyquist for even n) once, the rest twice.
  w = np.full((m,), 2.0)
  w[0] = 1.0
  if n % 2 == 0:
    w[-1] = 1.0
  w = jnp.asarray(w / n, re.dtype)
  return _mm(re, cos * w) - _mm(im, sin * w)


def _on_axis(fn, axis, *xs):
  import jax.numpy as jnp
  outs = fn(*[jnp.moveaxis(x, axis, -1) for x in xs])
  if not isinstance(outs, tuple):
    return jnp.moveaxis(outs, -1, axis)
  return tuple(jnp.moveaxis(o, -1, axis) for o in outs)


def _fft_dft(x, *, fft_type, fft_lengths):
  from functools import partial
  from jax import lax
  from jax._src.lax.fft import FftType

  nd = len(fft_lengths)
  axes = list(range(x.ndim - nd, x.ndim))
  if fft_type == FftType.RFFT:
    re, im = _r2c_last(x)
    for a in axes[:-1]:
      re, im = _on_axis(partial(_c2c_last, inverse=False), a, re, im)
    return lax.complex(re, im)
  re, im = lax.real(x), lax.imag(x)
  if fft_type == FftType.IRFFT:
    for a in axes[:-1]:
      re, im = _on_axis(partial(_c2c_last, inverse=True), a, re, im)
    return _c2r_last(re, im, fft_lengths[-1])
  inverse = fft_type == FftType.IFFT
  for a in axes:
    re, im = _on_axis(partial(_c2c_last, inverse=inverse), a, re, im)
  return lax.complex(re, im)


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------

_registered = False


def register() -> None:
  """Register the metal lowerings. Must run after the "metal" plugin has been
  registered with xla_bridge (register_lowering rejects unknown platforms)."""
  global _registered
  if _registered:
    return
  from jax._src.interpreters import mlir

  def reg(what, fn):
    try:
      fn()
    except Exception as e:  # noqa: BLE001 - never break plugin init
      logger.warning("metal: could not register lowering for %s: %s", what, e)

  def _eigh():
    from jax._src.lax import linalg as lax_linalg
    from jax._src.tpu.linalg import eigh as tpu_eigh
    mlir.register_lowering(lax_linalg.eigh_p, tpu_eigh._eigh_tpu_lowering,
                           platform=PLATFORM)

  def _fft():
    from jax._src.lax import fft as lax_fft
    mlir.register_lowering(
        lax_fft.fft_p, mlir.lower_fun(_fft_dft, multiple_results=False),
        platform=PLATFORM)

  def _check():
    from jax._src import checkify
    mlir.register_lowering(checkify.check_p,
                           checkify.check_lowering_rule_unsupported,
                           platform=PLATFORM)

  def _debug():
    from jax._src import debugging
    mlir.register_lowering(debugging.debug_callback_p,
                           debugging.debug_callback_lowering,
                           platform=PLATFORM)
    mlir.register_lowering(debugging.debug_print_p,
                           debugging.debug_print_lowering_rule,
                           platform=PLATFORM)

  reg("eigh", _eigh)
  reg("fft", _fft)
  reg("check", _check)
  reg("debug_callback/debug_print", _debug)
  from jax_plugins.metal import linalg_lowerings; reg("lapack linalg", linalg_lowerings.register)  # noqa: E702
  _registered = True
