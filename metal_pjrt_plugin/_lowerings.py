"""MLIR lowering rules for primitives whose upstream rules are registered only
for named platforms (cpu/cuda/rocm/tpu), so that they also lower on "mtl".

Policy: reuse JAX's own platform-independent implementations wherever one
exists, i.e. do what the TPU platform does (TPU is JAX's reference "no vendor
solver library" platform). Everything here is pure Python on top of generic
HLO; nothing requires C++ support from the plugin beyond the XLA expander
passes the GPU compiler pipeline already runs (CholeskyExpander, QrExpander,
EighExpander, TriangularSolveExpander).

Registered for platform "mtl":

* ``fft``: one 1-D transform per axis, each a ``metal$fft`` FFI call (MLX's
  FFT kernels, metal_pjrt/fft/fft.h; XLA's FftThunk is cuFFT/hipFFT-only)
  on complex64 (float32 on the real side of rfft / irfft); lengths past the
  kernels' limits raise NotImplementedError. Other dtypes (complex128, which
  the compiler then refuses like float64), batch dims that are symbolic
  (jax.export) and ``METAL_PJRT_DISABLE_FFT=1`` (for A/B comparisons) take
  a pure-JAX dense DFT (real matmuls against in-graph twiddle matrices,
  O(n^2) per axis, n <= 46340).
* ``check`` (checkify): the TPU rule, i.e. a no-op for ``debug=True`` checks
  and the usual "functionalize with checkify" error otherwise. The cpu/gpu
  rule, which raises from a host callback, is not wired up (host callbacks
  work, see docs/callbacks.md; nobody has needed it).
* ``debug_callback`` / ``debug_print``: the upstream cpu/gpu rule. Lowering
  goes through ``emit_python_callback``, which metal_pjrt_plugin/_callbacks.py
  redirects to the metal host-callback custom call. See docs/callbacks.md.

Float32 cholesky, triangular_solve, lu, geqrf/householder_product (qr), eigh
and svd go through Accelerate LAPACK instead (the C++ MetalLinalgRewriter and
metal_pjrt_plugin/_linalg_lowerings.py, registered last; its docstring has
the ownership table). The generic rules and the TPU ``eigh`` rule
(``_eigh_tpu_lowering``: Jacobi via EighExpander for n <= 256, QDWH above;
``svd``'s generic rule calls it) are their fallbacks for other dtypes,
unsupported options and METAL_PJRT_DISABLE_LAPACK.

Primitives that already lower via generic rules and need nothing here:
lu_pivots_to_permutation, tridiagonal_solve, threefry2x32, rng_bit_generator, approx_top_k
(generic fallback), cholesky_update, symmetric_product.

Deliberately NOT registered (no platform-independent implementation exists
in JAX; unsupported on TPU too): eig, schur, hessenberg, tridiagonal, geqp3.
GPU-only/vendor primitives (cudnn_fusion, dot_product_attention_*,
pbroadcast) are also out of scope.
"""

from __future__ import annotations

import functools
from functools import partial
import logging

import numpy as np

from metal_pjrt_plugin import PLATFORM, _env_flag  # "mtl"

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# FFT as a dense DFT
# --------------------------------------------------------------------------

_MAX_DFT = 46340  # r * c < 2^31 below this; the n x n matrix is ~8 GB here


def _twiddles(n_out, n_in, period, dtype):
  """cos/sin of 2*pi*(row*col mod period)/period as (n_out, n_in) arrays.

  Reducing the phase index modulo `period` in integer arithmetic keeps the
  angles in [0, 2*pi), so accuracy does not degrade with n (int32 products
  are exact for n <= 46340; longer DFTs raise, their twiddle matrix would
  not fit in memory anyway)."""
  from jax import lax
  import jax.numpy as jnp
  if period > _MAX_DFT:
    raise NotImplementedError(
        f"Metal: an FFT of length {period} through the dense DFT fallback "
        f"(METAL_PJRT_DISABLE_FFT, complex128 or a symbolic batch) needs an "
        f"{period} x {period} matrix; the DFT is limited to n <= {_MAX_DFT}")
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


# --------------------------------------------------------------------------
# FFT: metal$fft per axis, the DFT where it does not apply
# --------------------------------------------------------------------------

_MAX_FFT = 1 << 24  # fft_plan.h kMaxFftSize (four-step of 4096 x 4096)
_MAX_CHUNK_ELEMENTS = 1 << 24  # fft_plan.h kMaxChunkElements


@functools.cache
def fft_disabled() -> bool:
  return _env_flag("METAL_PJRT_DISABLE_FFT")


def _is_pow2(n):
  return n > 0 and n & (n - 1) == 0


def _next_pow2(n):
  return 1 << (n - 1).bit_length()


def _smooth13(n):
  for p in (2, 3, 5, 7, 11, 13):
    while n % p == 0:
      n //= p
  return n == 1


def fft_supported(n: int) -> bool:
  """Whether PlanFft (metal_pjrt/fft/fft_plan.h) has a plan for length n: up
  to 2^24 for powers of two, else up to a Bluestein length of 2^24 above
  4096 (n <= 2^23 - 1)."""
  if n < 1 or n > _MAX_FFT:
    return False
  return n <= 4096 or _is_pow2(n) or _next_pow2(2 * n - 1) <= _MAX_FFT


def fft_workspace_bytes(n: int, rows: int) -> int:
  """FftWorkspaceBytes(PlanFft(n), rows) (fft_plan.h) with the default chunk:
  the multi-pass plans need a complex64 row of n per chunk row (four-step:
  powers of two above 4096) or two of the Bluestein length (every other n
  above 4096, and n above 2048 with a prime factor above 13, the Rader and
  fused Bluestein limit). metal$fft refuses any other size."""
  if rows == 0:
    return 0
  if n > 4096 and _is_pow2(n):
    width, per_row = n, 8 * n
  elif n > 4096 or (n > 2048 and not _smooth13(n)):
    width = _next_pow2(2 * n - 1)
    per_row = 2 * 8 * width
  else:
    return 0
  chunk = min(max(_MAX_CHUNK_ELEMENTS // width, 1), rows)
  return chunk * per_row


def _metal_fft_last(x, kind, n):
  """metal$fft over the last axis: kind "fft" / "ifft" / "rfft" / "irfft"."""
  import jax
  import jax.numpy as jnp
  out_len = n // 2 + 1 if kind == "rfft" else n
  out_dtype = jnp.float32 if kind == "irfft" else jnp.complex64
  rows = int(np.prod(x.shape[:-1], dtype=np.int64))  # constant (native())
  out, _ = jax.ffi.ffi_call(
      "metal$fft",
      (jax.ShapeDtypeStruct(x.shape[:-1] + (out_len,), out_dtype),
       jax.ShapeDtypeStruct((fft_workspace_bytes(n, rows),), jnp.uint8)),
  )(x, fft_type=kind, n=np.int64(n))
  return out


def _fft_per_axis(x, *, fft_type, fft_lengths):
  """lax.fft_p as one 1-D transform per axis, like numpy: rfft on the last
  axis then forward transforms of the outer axes; irfft after inverse
  transforms of the outer axes. Each axis is metal$fft for complex64 /
  float32, else the DFT (module docstring)."""
  from jax import lax
  from jax._src.lax.fft import FftType

  native_dtype = x.dtype in (np.complex64, np.float32)
  # metal$fft's workspace is sized by the row count, which must be known at
  # lowering time: a symbolic batch (jax.export) takes the DFT.
  constant_batch = all(isinstance(d, (int, np.integer)) for d in x.shape)

  def native(n):
    if not native_dtype or fft_disabled():
      return False
    if not fft_supported(n):
      # The DFT would need an n x n matrix (>= 2^46 elements here).
      raise NotImplementedError(
          f"Metal: FFT of length {n} is not supported: the kernels take "
          f"powers of two up to 2^24 and other lengths up to 2^23 - 1")
    return constant_batch

  def c2c(z, axis, inverse):
    n = z.shape[axis]
    if native(n):
      return _on_axis(
          lambda v: _metal_fft_last(v, "ifft" if inverse else "fft", n),
          axis, z)
    re, im = _on_axis(partial(_c2c_last, inverse=inverse), axis,
                      lax.real(z), lax.imag(z))
    return lax.complex(re, im)

  nd = len(fft_lengths)
  axes = list(range(x.ndim - nd, x.ndim))
  n = fft_lengths[-1]
  if 0 in fft_lengths:  # an empty sum: zeros (the DFT would divide by 0)
    import jax.numpy as jnp
    if fft_type == FftType.RFFT:
      return jnp.zeros(x.shape[:-1] + (n // 2 + 1,),
                       jnp.result_type(x.dtype, jnp.complex64))
    if fft_type == FftType.IRFFT:
      return jnp.zeros(x.shape[:-nd] + tuple(fft_lengths),
                       jnp.finfo(x.dtype).dtype)
    return jnp.zeros(x.shape, x.dtype)
  if fft_type == FftType.RFFT:
    z = (_metal_fft_last(x, "rfft", n) if native(n)
         else lax.complex(*_r2c_last(x)))
    for a in axes[:-1]:
      z = c2c(z, a, inverse=False)
    return z
  if fft_type == FftType.IRFFT:
    for a in axes[:-1]:
      x = c2c(x, a, inverse=True)
    if native(n):
      return _metal_fft_last(x, "irfft", n)
    return _c2r_last(lax.real(x), lax.imag(x), n)
  for a in axes:
    x = c2c(x, a, inverse=fft_type == FftType.IFFT)
  return x


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------

_registered = False


def register() -> None:
  """Register the lowerings. Must run after the "mtl" plugin has been
  registered with xla_bridge (register_lowering rejects unknown platforms)."""
  global _registered
  if _registered:
    return
  from jax._src.interpreters import mlir

  def reg(what, fn):
    try:
      fn()
    except Exception as e:  # noqa: BLE001 - never break plugin init
      logger.warning("metal-pjrt-plugin: could not register lowering for %s: %s", what, e)

  def _fft():
    from jax._src.lax import fft as lax_fft
    mlir.register_lowering(
        lax_fft.fft_p, mlir.lower_fun(_fft_per_axis, multiple_results=False),
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

  reg("fft", _fft)
  reg("check", _check)
  reg("debug_callback/debug_print", _debug)
  from metal_pjrt_plugin import _linalg_lowerings; reg("lapack linalg", _linalg_lowerings.register)  # noqa: E702
  _registered = True
