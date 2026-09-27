"""Shared helpers for the metal tests: devices, references, ulp comparison.

Accuracy is measured in ulps of the output dtype against a float64
reference computed on CPU (inputs upcast exactly, so the reference is the
true answer for the rounded inputs up to f64 rounding):

- elementwise (default): |got - want| / ulp(|want|) per element, the right
  measure for elementwise ops;
- normwise: in ulps of max|want|, for sums, products of matrices and whole
  programs, whose small outputs come from cancellation.

Tolerances in the tests are small multiples of the measured error, so real
regressions show. They were measured on an Apple M3 (10-core GPU), macOS
26.2: on other Apple GPUs or OS versions (different Metal math library /
compiler), a slightly larger error means retune, not regression. METAL_TEST_REPORT_ULPS=1 prints the measured error of
every comparison (with pytest -s), plus the CPU float32 error for context,
which is how the tolerances were set.
"""
import functools
import os

import jax
import ml_dtypes
import numpy as np

REPORT = bool(os.environ.get("METAL_TEST_REPORT_ULPS"))


@functools.cache
def metal():
    return jax.devices("openmetal")[0]


@functools.cache
def cpu():
    return jax.devices("cpu")[0]


def run_on(dev, fn, *args):
    """jit(fn)(*args) on `dev`, outputs as numpy."""
    with jax.default_device(dev):
        out = jax.jit(fn)(*[jax.device_put(a, dev) for a in args])
        return jax.tree.map(np.asarray, out)


def _upcast(a):
    a = np.asarray(a)
    return a.astype(np.float64) if np.issubdtype(a.dtype, np.floating) or \
        a.dtype in (ml_dtypes.bfloat16, ml_dtypes.float8_e4m3fn) else a


def f64_reference(fn, *args):
    """fn on CPU with x64 on and floating inputs upcast to float64."""
    with jax.enable_x64(True):
        return run_on(cpu(), fn, *[_upcast(a) for a in args])


def ulp_error(got, want, dtype, normwise=False):
    """Max error of `got` against `want` in ulps of `dtype` (see module doc).
    Non-finite values must match exactly (else inf)."""
    got = np.asarray(got).astype(np.float64)
    want = np.asarray(want).astype(np.float64)
    assert got.shape == want.shape, (got.shape, want.shape)
    fin = np.isfinite(want)
    if not (np.array_equal(got[~fin], want[~fin], equal_nan=True)
            and np.isfinite(got[fin]).all()):
        return np.inf
    if not fin.any():
        return 0.0
    fi = ml_dtypes.finfo(dtype)
    mag = np.abs(want[fin])
    if normwise:
        mag = np.full_like(mag, mag.max())
    ulp = np.exp2(np.floor(np.log2(np.maximum(mag, fi.smallest_normal)))
                  - fi.nmant)
    return float(np.max(np.abs(got[fin] - want[fin]) / ulp))


def _is_float(dtype):
    return np.issubdtype(dtype, np.floating) or dtype in (
        ml_dtypes.bfloat16, ml_dtypes.float8_e4m3fn)


def assert_close(got, want, ulps, normwise=False, name="", cpu32=None):
    """Compare pytrees leafwise: floating leaves within `ulps` (of the
    dtype of `got`), everything else exactly. `cpu32`, if given, is the
    same computation on CPU in the original precision, reported alongside
    the metal error when METAL_TEST_REPORT_ULPS is set."""
    got_leaves, want_leaves = jax.tree.leaves(got), jax.tree.leaves(want)
    assert len(got_leaves) == len(want_leaves)
    cpu_leaves = jax.tree.leaves(cpu32) if cpu32 is not None else None
    worst = worst_cpu = 0.0
    for i, (g, w) in enumerate(zip(got_leaves, want_leaves)):
        g, w = np.asarray(g), np.asarray(w)
        if not _is_float(g.dtype):
            np.testing.assert_array_equal(g, w.astype(g.dtype), err_msg=name)
            continue
        err = ulp_error(g, w, g.dtype, normwise)
        worst = max(worst, err)
        if cpu_leaves is not None:
            worst_cpu = max(worst_cpu, ulp_error(cpu_leaves[i], w, g.dtype,
                                                 normwise))
    if REPORT:
        cpu_note = f" cpu32={worst_cpu:.3g}" if cpu_leaves is not None else ""
        print(f"\nULPS {name!r} metal={worst:.3g}{cpu_note}"
              f" tol={ulps}{' normwise' if normwise else ''}")
    assert worst <= ulps, (f"{name}: {worst:.3g} ulps > {ulps}"
                           f"{' (normwise)' if normwise else ''}")


def check(fn, *args, ulps, normwise=False, ref="f64", name=""):
    """Run fn on metal and compare with the CPU reference: "f64" (default)
    or "same" (CPU in the input precision, for bit-level ops such as
    nextafter / bitcast / RNG bits, and for host callbacks)."""
    got = run_on(metal(), fn, *args)
    cpu32 = run_on(cpu(), fn, *args) if REPORT or ref == "same" else None
    want = f64_reference(fn, *args) if ref == "f64" else cpu32
    assert_close(got, want, ulps, normwise, name,
                 cpu32 if ref == "f64" else None)
