# Copyright 2026 The metal-pjrt-plugin Authors
# SPDX-License-Identifier: Apache-2.0

"""LAPACK-backed lowerings of JAX's linear algebra primitives on "mtl".

The plugin's C++ side (metal_pjrt/linalg/lapack_ffi.cc) registers FFI
handlers that synchronize the Metal stream and run Apple Accelerate's LAPACK
on the unified-memory buffers (zero copy). This module lowers JAX primitives
to those handlers, mirroring JAX's own CPU lowerings (``lapack_*_ffi``): the
matrix operands and results request column-major layouts (XLA inserts the
transposes on the GPU), so the handlers see LAPACK's native layout.

Who owns which primitive (float32; other dtypes take the fallback):

  primitive            owner    custom call               fallback
  cholesky             C++ [1]  metal$cholesky            CholeskyExpander
  triangular_solve     C++ [1]  metal$triangular_solve    TriangularSolveExpander
  lu                   here     metal$lapack_getrf        _lu_python
  geqrf                here     metal$lapack_geqrf        Qr -> QrExpander
  householder_product  here     metal$lapack_orgqr        [2] -> QrExpander
  eigh                 here     metal$lapack_syevd        _eigh_tpu_lowering
  svd                  here     metal$lapack_gesdd[_novec]  _svd_tpu_lowering_rule
  tridiagonal_solve    here     metal$lapack_gtsv         _tridiagonal_solve_jax [3]

  [1] MetalLinalgRewriter (metal_pjrt/linalg/linalg_rewriter.cc)
  [2] the ProductOfElementaryHouseholderReflectors custom call
  [3] JAX's generic Thomas algorithm, which does not pivot: a system that
      needs pivoting (a zero or small leading pivot) gives NaN or garbage.
      gtsv pivots, as JAX's CPU and CUDA lowerings do.

cholesky_p and triangular_solve_p need nothing here: their generic lowerings
emit the HLO ``cholesky`` / ``triangular_solve`` ops, which the compiler's
MetalLinalgRewriter turns into the custom calls at the start of
RunHloPasses. getrf returns lu, 0-based pivots and the permutation.

Unsupported options (eigh/svd subsets, Jacobi/polar/QDWH algorithms, m < n
householder products, tridiagonal_solve's perturb_singular, dynamic shapes)
take the fallback too.
``METAL_PJRT_DISABLE_LAPACK=1`` is the one switch for both owners: every rule
here and the C++ rewriter restore the fallbacks (for A/B comparisons). Both
read it once per process, as the persistent compilation cache key does
(compiler/compile_settings.h), so set it before starting JAX.
"""

from __future__ import annotations

import functools

import numpy as np

from metal_pjrt_plugin import PLATFORM, _env_flag  # "mtl"


@functools.cache
def lapack_disabled() -> bool:
  return _env_flag("METAL_PJRT_DISABLE_LAPACK")


def _is_f32(aval) -> bool:
  return np.dtype(aval.dtype) == np.float32


def _static(avals) -> bool:
  from jax._src import core
  return all(core.is_constant_shape(a.shape) for a in avals)


def register() -> None:
  """Register the LAPACK lowerings for platform "mtl" (after _lowerings.py's
  own registrations, which these override for float32)."""
  from jax._src.interpreters import mlir
  from jax._src.lax import linalg as ll
  from jax._src.tpu.linalg import eigh as tpu_eigh
  from jax._src.tpu.linalg import svd as tpu_svd

  ffi = ll._linalg_ffi_lowering

  def ok(ctx):
    return (not lapack_disabled()
            and all(_is_f32(a) for a in ctx.avals_in if a.dtype.kind == "f")
            and all(a.dtype.kind in "fi" for a in ctx.avals_in)
            and _static((*ctx.avals_in, *ctx.avals_out)))

  # lu ---------------------------------------------------------------------
  lu_fallback = mlir.lower_fun(ll._lu_python, multiple_results=True)

  def lu_rule(ctx, operand):
    if not ok(ctx):
      return lu_fallback(ctx, operand)
    return ffi("metal$lapack_getrf", operand_output_aliases={0: 0})(
        ctx, operand)

  # qr ---------------------------------------------------------------------
  def geqrf_rule(ctx, operand):
    if not ok(ctx):
      return ll._geqrf_lowering_rule(ctx, operand)
    return ffi("metal$lapack_geqrf", operand_output_aliases={0: 0})(
        ctx, operand)

  def householder_rule(ctx, a, taus):
    m, n = ctx.avals_in[0].shape[-2:]
    if not ok(ctx) or m < n:
      return ll._householder_product_lowering(ctx, a, taus)
    return ffi("metal$lapack_orgqr", operand_output_aliases={0: 0})(
        ctx, a, taus)

  # eigh -------------------------------------------------------------------
  def eigh_rule(ctx, operand, *, lower, sort_eigenvalues, subset_by_index,
                algorithm):
    n = ctx.avals_in[0].shape[-1]
    if (not ok(ctx)
        or not (subset_by_index is None or tuple(subset_by_index) == (0, n))
        or algorithm not in (None, ll.EighImplementation.QR)):
      return tpu_eigh._eigh_tpu_lowering(
          ctx, operand, lower=lower, sort_eigenvalues=sort_eigenvalues,
          subset_by_index=subset_by_index, algorithm=algorithm)
    # LAPACK always sorts eigenvalues ascending.
    return ffi("metal$lapack_syevd", operand_output_aliases={0: 0})(
        ctx, operand, lower=bool(lower))

  # svd --------------------------------------------------------------------
  def svd_rule(ctx, operand, *, full_matrices, compute_uv, subset_by_index,
               algorithm=None):
    operand_aval, = ctx.avals_in
    m, n = operand_aval.shape[-2:]
    if (not ok(ctx) or m == 0 or n == 0
        or not (subset_by_index is None
                or tuple(subset_by_index) == (0, min(m, n)))
        or algorithm not in (None, ll.SvdAlgorithm.DEFAULT,
                             ll.SvdAlgorithm.DIVIDE_AND_CONQUER)):
      return tpu_svd._svd_tpu_lowering_rule(
          ctx, operand, full_matrices=full_matrices, compute_uv=compute_uv,
          subset_by_index=subset_by_index, algorithm=algorithm)
    if compute_uv:
      s_aval, u_aval, vt_aval = ctx.avals_out
      rule = ffi("metal$lapack_gesdd",
                 avals_out=[operand_aval, s_aval, u_aval, vt_aval],
                 operand_output_aliases={0: 0})
      _, s, u, vt = rule(ctx, operand, full_matrices=bool(full_matrices))
      return [s, u, vt]
    s_aval, = ctx.avals_out
    rule = ffi("metal$lapack_gesdd_novec", avals_out=[operand_aval, s_aval],
               operand_output_aliases={0: 0})
    _, s = rule(ctx, operand)
    return [s]

  # tridiagonal_solve ------------------------------------------------------
  tridiagonal_fallback = mlir.lower_fun(ll._tridiagonal_solve_jax,
                                        multiple_results=False)

  def tridiagonal_rule(ctx, dl, d, du, b, *, perturb_singular):
    if perturb_singular or not ok(ctx):
      return tridiagonal_fallback(ctx, dl, d, du, b,
                                  perturb_singular=perturb_singular)
    return ffi("metal$lapack_gtsv", operand_output_aliases={3: 0})(
        ctx, dl, d, du, b)

  for prim, rule in ((ll.lu_p, lu_rule), (ll.geqrf_p, geqrf_rule),
                     (ll.householder_product_p, householder_rule),
                     (ll.eigh_p, eigh_rule), (ll.svd_p, svd_rule),
                     (ll.tridiagonal_solve_p, tridiagonal_rule)):
    mlir.register_lowering(prim, rule, platform=PLATFORM)
