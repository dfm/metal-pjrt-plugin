# Copyright 2026 The metal-pjrt-plugin Authors
# SPDX-License-Identifier: Apache-2.0

"""jax.numpy.fft on metal: the fft lowering (metal_pjrt_plugin/_lowerings.py)
sends each transformed axis to metal$fft (MLX's kernels,
metal_pjrt/fft/fft.h; the kernels themselves are tested per path in
//metal_pjrt/fft:fft_test) or, with METAL_PJRT_DISABLE_FFT=1 or a symbolic
batch, to the dense DFT. Against numpy in float64; the size limit;
jax.export; which calls end up in the HLO.

Errors are normwise, max|got - want| / max|want|, in units of
2^-24 log2(n) (the FFT's error growth).
"""
import os
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from metal_testing import REPORT, cpu, metal, run_on, run_python
from metal_pjrt_plugin import _lowerings

pytestmark = pytest.mark.metal

EPS = 2.0 ** -24
# Measured (METAL_TEST_REPORT_ULPS=1, M3) and doubled: at most 1.2 (fft of
# the fused Bluestein length 1021), multi-dimensional transforms <= 0.94.
COEFF = 2.5

rng = np.random.default_rng(0)


def cx(*s):
    return (rng.standard_normal(s) + 1j * rng.standard_normal(s)).astype(
        np.complex64)


def rl(*s):
    return rng.standard_normal(s).astype(np.float32)


def err(got, want):
    got, want = np.asarray(got), np.asarray(want)
    assert got.shape == want.shape, (got.shape, want.shape)
    return np.abs(got - want).max() / np.abs(want).max()


def assert_fft_close(got, want, n, name):
    got = np.asarray(got)
    want = np.asarray(want)
    assert got.dtype == (np.float32 if want.dtype == np.float64
                         else np.complex64), (got.dtype, name)
    c = err(got, want) / (EPS * max(1.0, np.log2(n)))
    if REPORT:
        print(f"\nfft {name}: {c:.2f} (x 2^-24 log2 n)")
    assert c <= COEFF, (name, c)


def hlo(fn, *args):
    return jax.jit(fn).lower(*args).compile().as_text()


CALL = re.compile(r'custom_call_target="metal\$fft"')


def n_calls(text):
    return len(CALL.findall(text))

# One length per plan (fft_plan.h): Stockham (radices, 13-smooth), Rader,
# fused Bluestein, multi-upload Bluestein (above 2048 with a prime factor
# above 13, and non-powers of two above 4096), four-step; and length 1.
LENGTHS = [1, 2, 7, 16, 60, 1000, 4096, 17, 1021, 47, 2053, 6000, 8192]


@pytest.mark.parametrize("n", LENGTHS)
def test_1d(n):
    x = cx(3, n)
    xr = x.real.copy()
    # A non-Hermitian half spectrum: numpy's irfft ignores the imaginary
    # parts of bins 0 and n/2 (n even) and mirrors the rest.
    h = x[:, :n // 2 + 1]
    f = jax.jit(lambda x, xr, h: (jnp.fft.fft(x), jnp.fft.ifft(x),
                                  jnp.fft.rfft(xr), jnp.fft.irfft(h, n)))
    got = f(x, xr, h)
    x64, h64 = x.astype(np.complex128), h.astype(np.complex128)
    want = (np.fft.fft(x64), np.fft.ifft(x64), np.fft.rfft(xr.astype(float)),
            np.fft.irfft(h64, n))
    for name, g, w in zip(["fft", "ifft", "rfft", "irfft"], got, want):
        assert_fft_close(g, w, n, f"{name} n={n}")
    assert n_calls(f.lower(x, xr, h).compile().as_text()) == 4


# Multi-dimensional transforms: one metal$fft per axis (the outer axes
# through a transpose), numpy's order (irfftn: the inverse transforms of the
# outer axes first, then irfft on the last axis).
ND = {
    "fft2": (lambda x: jnp.fft.fft2(x), lambda x: np.fft.fft2(x), "c", (4, 12, 16), 2),
    "ifft2": (lambda x: jnp.fft.ifft2(x), lambda x: np.fft.ifft2(x), "c", (4, 12, 16), 2),
    "fftn": (lambda x: jnp.fft.fftn(x), lambda x: np.fft.fftn(x), "c", (6, 17, 10), 3),
    "ifftn": (lambda x: jnp.fft.ifftn(x), lambda x: np.fft.ifftn(x), "c", (6, 17, 10), 3),
    "rfftn": (lambda x: jnp.fft.rfftn(x), lambda x: np.fft.rfftn(x), "r", (6, 10, 12), 3),
    # Non-Hermitian on every axis.
    "irfftn": (lambda x: jnp.fft.irfftn(x, (6, 10, 12)),
               lambda x: np.fft.irfftn(x, (6, 10, 12), axes=(0, 1, 2)), "c", (6, 10, 7), 3),
    "irfft2 odd": (lambda x: jnp.fft.irfft2(x, (9, 13)),
                   lambda x: np.fft.irfft2(x, (9, 13)), "c", (3, 9, 7), 2),
    "fftn axes (0, 2)": (lambda x: jnp.fft.fftn(x, axes=(0, 2)),
                         lambda x: np.fft.fftn(x, axes=(0, 2)), "c", (8, 3, 20), 2),
    "fft axis 0": (lambda x: jnp.fft.fft(x, axis=0),
                   lambda x: np.fft.fft(x, axis=0), "c", (47, 5), 1),
    "rfft axis 1 of 3": (lambda x: jnp.fft.rfft(x, axis=1),
                         lambda x: np.fft.rfft(x, axis=1), "r", (4, 30, 3), 1),
}


@pytest.mark.parametrize("name", list(ND))
def test_nd(name):
    fn, ref, kind, shape, axes = ND[name]
    x = cx(*shape) if kind == "c" else rl(*shape)
    x64 = x.astype(np.complex128 if kind == "c" else np.float64)
    assert_fft_close(jax.jit(fn)(x), ref(x64), max(shape), name)
    assert n_calls(hlo(fn, x)) == axes


def test_vmap():
    x = cx(5, 3, 64)
    got = jax.jit(jax.vmap(jnp.fft.fft))(x)
    assert_fft_close(got, np.fft.fft(x.astype(np.complex128)), 64, "vmap")
    # Mapped over an inner axis: batching moves it out, the transform stays
    # on the last axis.
    got = jax.jit(jax.vmap(lambda v: jnp.fft.rfft(v.real), in_axes=1))(x)
    want = np.moveaxis(np.fft.rfft(x.real.astype(np.float64)), 1, 0)
    assert_fft_close(got, want, 64, "vmap in_axes=1")


def test_grad():
    # Transposes of rfft, irfft and fft are themselves metal$fft calls.
    x, y = rl(4, 40), cx(4, 21)

    def f(x, y):
        a = jnp.sum(jnp.abs(jnp.fft.rfft(x)) ** 2)
        b = jnp.sum(jnp.fft.irfft(y, 40) * x)
        c = jnp.sum(jnp.abs(jnp.fft.ifft(jnp.fft.fft(y) * 2.0)))
        return a + b + c

    g = jax.grad(f, argnums=(0, 1))
    got = run_on(metal(), g, x, y)
    want = run_on(cpu(), g, x, y)
    for i, (gm, gc) in enumerate(zip(got, want)):
        assert gm.dtype == gc.dtype
        assert err(gm, gc) < 1e-5, (i, err(gm, gc))
    assert n_calls(hlo(g, x, y)) >= 4


def test_complex_through_jit():
    # Complex in, complex out (device buffers of complex64), and complex
    # intermediates between two calls and a fusion.
    x = jax.device_put(cx(8, 32), metal())
    y = jax.jit(lambda x: jnp.fft.ifft(jnp.fft.fft(x) * (1 + 2j)))(x)
    assert y.dtype == np.complex64
    np.testing.assert_allclose(np.asarray(y), np.asarray(x) * (1 + 2j),
                               rtol=0, atol=5e-6)
    z = jax.jit(jnp.fft.fft)(y)
    assert_fft_close(z, np.fft.fft(np.asarray(y).astype(np.complex128)), 32,
                     "fft of a device array")


def test_large_and_chunked():
    # A four-step length whose rows go in two chunks (the workspace holds
    # 16 rows of 2^20; the Python rule must size it as FftWorkspaceBytes
    # does, or metal$fft refuses), and a long Bluestein length.
    n = 1 << 20
    assert _lowerings.fft_workspace_bytes(n, 17) == 16 * n * 8
    x = cx(17, n)
    got = np.asarray(jax.jit(jnp.fft.fft)(x))
    want = np.fft.fft(x.astype(np.complex128))
    assert_fft_close(got, want, n, "fft 2^20 x 17")
    n = 100003
    x = rl(2, n)
    assert_fft_close(jax.jit(jnp.fft.rfft)(x), np.fft.rfft(x.astype(float)),
                     n, "rfft 100003")


def test_empty():
    x = np.zeros((0, 16), np.complex64)
    # (XLA folds the empty transform away.)
    assert jax.jit(jnp.fft.fft)(x).shape == (0, 16)
    assert jax.jit(jnp.fft.irfft)(x).shape == (0, 30)
    # Length 0 (JAX's fft_test testFftEmpty).
    assert jnp.fft.fft(jnp.zeros((0,), jnp.complex64)).shape == (0,)
    assert jnp.fft.ifft2(jnp.zeros((3, 0), jnp.complex64)).shape == (3, 0)


def test_supported_lengths():
    # PlanFft's limits (fft_plan.h): powers of two up to 2^24, anything up
    # to 4096, other lengths while the Bluestein length next_pow2(2n - 1)
    # stays within 2^24.
    s = _lowerings.fft_supported
    assert s(1) and s(4096) and s(4097) and s(1 << 24)
    assert s((1 << 23) - 1) and s(3 * (1 << 21))
    assert not s(0) and not s((1 << 24) + 1) and not s((1 << 23) + 1)
    assert not s(1 << 25)
    ws = _lowerings.fft_workspace_bytes
    assert ws(4096, 7) == 0 and ws(2029, 7) == 0 and ws(2039, 7) == 0
    assert ws(2053, 7) == 7 * 2 * 8192 * 8  # multi-upload Bluestein
    assert ws(4095, 7) == 0 and ws(4097, 7) == 7 * 2 * 16384 * 8
    assert ws(2187 * 2, 1) == 2 * 16384 * 8  # 13-smooth, above 4096
    assert ws(8192, 3) == 3 * 8192 * 8 and ws(8192, 0) == 0
    assert ws(1 << 24, 5) == (1 << 24) * 8  # one row per chunk


def test_size_limit_error(monkeypatch):
    # Lengths metal$fft does not support raise instead of taking the DFT,
    # which would need an n x n matrix (>= 2^46 elements past the real
    # limits). Lower the limit to 64 to check it: the 16-axis alone is fine.
    monkeypatch.setattr(_lowerings, "fft_supported", lambda n: n <= 64)
    jax.clear_caches()  # lax.fft is a jit: its lowering is cached
    x = cx(100, 16)
    for fn in [jnp.fft.fft2, lambda x: jnp.fft.rfft(x.real, axis=0),
               lambda x: jnp.fft.irfft(x, 100, axis=0)]:
        with pytest.raises(NotImplementedError,
                           match=r"FFT of length 100 .*2\^24.*2\^23 - 1"):
            jax.jit(fn)(x)
    assert n_calls(hlo(jnp.fft.fft, x)) == 1
    jax.clear_caches()


def test_dft_length_cap():
    # The DFT fallback's int32 phase index r * c is exact up to 46340 (its
    # matrix would not fit in memory past that anyway).
    _lowerings._twiddles(4, 4, 46340, jnp.float32)
    with pytest.raises(NotImplementedError, match="n <= 46340"):
        _lowerings._twiddles(4, 4, 46341, jnp.float32)


def test_export_multi_platform():
    # A module exported for cpu and mtl holds the stablehlo fft in its cpu
    # branch. Today the StableHLO -> HLO conversion already folds the
    # platform conditional of exp.call (so this also passed when the fft op
    # was refused before optimization); the compiler refuses the HLO fft op
    # only after optimization, so a conditional that reaches HLO is folded
    # first too (hlo_checks_test FftIsRefusedAfterOptimization).
    from jax import export
    x = cx(8, 64)
    exp = export.export(
        jax.jit(jnp.fft.fft), platforms=("cpu", "mtl"),
        disabled_checks=[export.DisabledSafetyCheck.custom_call("metal$fft")])(x)
    assert "stablehlo.fft" in exp.mlir_module()
    assert "metal$fft" in exp.mlir_module()
    assert_fft_close(exp.call(x), np.fft.fft(x.astype(np.complex128)), 64,
                     "export cpu+mtl")


def test_export_symbolic_batch():
    # metal$fft's workspace needs the row count at lowering time: a symbolic
    # batch takes the DFT (and still runs).
    from jax import export
    (b,) = export.symbolic_shape("b")
    exp = export.export(jax.jit(jnp.fft.fft), platforms=("mtl",))(
        jax.ShapeDtypeStruct((b, 64), jnp.complex64))
    assert "metal$fft" not in exp.mlir_module()
    x = cx(5, 64)
    assert_fft_close(exp.call(x), np.fft.fft(x.astype(np.complex128)), 64,
                     "export symbolic batch")


DISABLE_CHILD = r"""
import re, sys, numpy as np, jax, jax.numpy as jnp
rng = np.random.default_rng(1)
x = (rng.standard_normal((4, 60)) + 1j * rng.standard_normal((4, 60))).astype(np.complex64)
f = jax.jit(lambda x: (jnp.fft.fft(x), jnp.fft.rfft(x.real), jnp.fft.irfft(x, 60),
                       jnp.fft.fft2(x)))
out = [np.asarray(o) for o in f(x)]
text = f.lower(x).compile().as_text()
print(len(re.findall(r'custom_call_target="metal\$fft"', text)))
np.savez(OUT, *out)
"""


@pytest.mark.parametrize("value", [None, "0", "1"])
def test_disable_fft(value, tmp_path):
    # METAL_PJRT_DISABLE_FFT=1 lowers every axis to the DFT (no metal$fft
    # in the HLO); the numbers agree with the native path (and numpy).
    env = {k: v for k, v in os.environ.items() if k != "METAL_PJRT_DISABLE_FFT"}
    if value is not None:
        env["METAL_PJRT_DISABLE_FFT"] = value
    path = tmp_path / "out.npz"
    out = run_python(DISABLE_CHILD.replace("OUT", repr(str(path))),
                     env)
    assert out.returncode == 0, out.stderr[-3000:]
    assert int(out.stdout.strip()) == (0 if value == "1" else 5)
    got = np.load(path)
    rng2 = np.random.default_rng(1)
    x = (rng2.standard_normal((4, 60))
         + 1j * rng2.standard_normal((4, 60))).astype(np.complex64)
    x64 = x.astype(np.complex128)
    want = [np.fft.fft(x64), np.fft.rfft(x64.real), np.fft.irfft(x64, 60),
            np.fft.fft2(x64)]
    native = jax.jit(lambda x: (jnp.fft.fft(x), jnp.fft.rfft(x.real),
                                jnp.fft.irfft(x, 60), jnp.fft.fft2(x)))(x)
    for i, w in enumerate(want):
        g = got[f"arr_{i}"]
        assert_fft_close(g, w, 60, f"disable={value} #{i}")
        assert err(g, native[i]) <= 2 * COEFF * EPS * np.log2(60), i
