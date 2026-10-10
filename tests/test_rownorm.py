# Copyright 2026 The metal-pjrt-plugin Authors
# SPDX-License-Identifier: Apache-2.0

"""Row-normalization fusions (MetalRowNormFusion + MetalRowNormEmitter):
softmax, log-softmax, layer and RMS norm, a causal masked softmax and the
gradient of softmax run as one `metal_rownorm` kernel each, against a
float64 CPU reference. The shapes cover one simdgroup per row (n <= 256,
several rows per threadgroup) up to 32 simdgroups per row reducing through
threadgroup memory (n = 16384), and rows that don't fill a threadgroup.
"""
import os

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from metal_testing import assert_close, f64_reference, metal, run_on, cpu, run_python

pytestmark = pytest.mark.metal


def layer_norm(x):
    n = x.shape[-1]
    w = jnp.linspace(0.5, 1.5, n, dtype=x.dtype)
    b = jnp.linspace(-1, 1, n, dtype=x.dtype)
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) * jax.lax.rsqrt(v + 1e-5) * w + b


def rms_norm(x):
    w = jnp.linspace(0.5, 1.5, x.shape[-1], dtype=x.dtype)
    x32 = x.astype(jnp.float32)
    y = x32 * jax.lax.rsqrt(jnp.mean(x32 * x32, -1, keepdims=True) + 1e-6)
    return (y * w.astype(jnp.float32)).astype(x.dtype)


def causal_softmax(s):
    # Attention's scale and mask: recomputed in each of the kernel's loops.
    mask = jnp.tril(jnp.ones(s.shape[-2:], bool))
    return jax.nn.softmax(jnp.where(mask, s * 0.125, -jnp.inf), axis=-1)


CASES = {
    "softmax": lambda x: jax.nn.softmax(x, axis=-1),
    "log_softmax": lambda x: jax.nn.log_softmax(x, axis=-1),
    "layer_norm": layer_norm,
    "rms_norm": rms_norm,
    "causal_softmax": causal_softmax,
    # JAX differentiates through softmax: the row sum and exp(x - max) are
    # outputs of the forward kernel, read by the backward.
    "softmax_grad": jax.grad(
        lambda x: (jax.nn.softmax(x, axis=-1) * jnp.cos(x)).sum()),
}
NORMWISE = {"softmax": False, "log_softmax": True, "layer_norm": True,
            "rms_norm": True, "causal_softmax": False, "softmax_grad": True}
SHAPES = [(4, 32), (3, 5, 64), (16, 128), (8, 1000), (5, 1024), (64, 64),
          (2, 4096), (2, 16384), (513, 256)]
DTYPES = {"float32": jnp.float32, "float16": jnp.float16,
          "bfloat16": jnp.bfloat16}
# Ulps vs float64 (normwise where NORMWISE), measured on an M3 over SHAPES
# (max) and doubled. The unfused path (METAL_PJRT_DISABLE_REWRITES=rownorm)
# measured the same or worse in every case.
ULPS = {
    "softmax": {"float32": 38, "float16": 19, "bfloat16": 1},
    "log_softmax": {"float32": 2.1, "float16": 2.2, "bfloat16": 1},
    "layer_norm": {"float32": 6, "float16": 5.1, "bfloat16": 3.3},
    "rms_norm": {"float32": 6, "float16": 3, "bfloat16": 3.2},
    "causal_softmax": {"float32": 8.2, "float16": 7.4, "bfloat16": 1},
    "softmax_grad": {"float32": 7.5, "float16": 3.2, "bfloat16": 1},
}


def make_input(shape, dtype, seed=0):
    x = np.random.default_rng(seed).standard_normal(shape).astype(np.float32) * 3
    return x.astype(dtype)


def compiled_text(fn, *args):
    return jax.jit(fn).lower(
        *[jax.device_put(a, metal()) for a in args]).compile().as_text()


@pytest.mark.parametrize("dtype", list(DTYPES))
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize("case", list(CASES))
def test_rownorm(case, shape, dtype):
    fn = CASES[case]
    x = make_input(shape, DTYPES[dtype])
    assert "metal_rownorm" in compiled_text(fn, x), "the fusion did not fire"
    got = run_on(metal(), fn, x)
    assert_close(got, f64_reference(fn, x), ULPS[case][dtype],
                 normwise=NORMWISE[case], name=f"{case} {shape} [{dtype}]",
                 cpu32=run_on(cpu(), fn, x))


def _two_reductions_of_one_input(x):
    y = x - jnp.max(x, axis=-1, keepdims=True)
    return y * jnp.sum(y, -1, keepdims=True) - jnp.min(y, -1, keepdims=True)


# Fusions where a reduction's input is also an output (the gradient of max
# writes its one-hot mask), or feeds two reductions: each input has one
# epilogue (epilogue functions are named after their roots).
@pytest.mark.parametrize("fn", [
    jax.grad(lambda x: (jnp.max(x, axis=-1) * jnp.arange(x.shape[0])).sum()),
    _two_reductions_of_one_input,
], ids=["max_grad", "two_reductions_of_one_input"])
@pytest.mark.parametrize("shape", [(6, 64), (4, 1000), (2, 4, 2, 128)],
                         ids=lambda s: "x".join(map(str, s)))
def test_shared_reduction_inputs(fn, shape):
    x = make_input(shape, np.float32)
    assert "metal_rownorm" in compiled_text(fn, x)
    assert_close(run_on(metal(), fn, x), f64_reference(fn, x), 8,
                 normwise=True, cpu32=run_on(cpu(), fn, x))


@pytest.mark.parametrize("col", [0, 1, 31, 32, 33, 500, 999])
def test_softmax_nan_spreads_over_its_row(col):
    # Max and sum propagate NaN through every lane's partial result and
    # every shuffle, as on CPU: the row is all NaN, the others are exact.
    fn = CASES["softmax"]
    x = make_input((4, 1000), np.float32)
    x[2, col] = np.nan
    got = run_on(metal(), fn, x)
    want = run_on(cpu(), fn, x)
    assert np.isnan(got[2]).all()
    assert_close(np.delete(got, 2, 0), np.delete(want, 2, 0), 38)


@pytest.mark.parametrize("n", [64, 1000, 16384])
def test_all_negative_rows(n):
    # The max starts from -inf (the reduction's identity), not 0.
    fn = CASES["softmax"]
    x = make_input((3, n), np.float32) - 100
    assert "metal_rownorm" in compiled_text(fn, x)
    assert_close(run_on(metal(), fn, x), f64_reference(fn, x), 38)


def test_non_identity_init_is_left_to_xla():
    # jnp.max(initial=0) is a reduce with init 0 (combined once per thread
    # in the fused kernel), so no fusion takes it.
    fn = lambda x: x - jnp.max(x, axis=-1, keepdims=True, initial=0.0)
    x = make_input((8, 256), np.float32) - 10
    np.testing.assert_array_equal(run_on(metal(), fn, x), run_on(cpu(), fn, x))


def test_kill_switch():
    env = dict(os.environ, METAL_PJRT_DISABLE_REWRITES="rownorm")
    code = (
        "import jax, jax.numpy as jnp\n"
        "x = jnp.ones((8, 128))\n"
        "t = jax.jit(lambda x: jax.nn.softmax(x, -1)).lower(x).compile().as_text()\n"
        "print('fused' if 'metal_rownorm' in t else 'unfused')\n")
    out = run_python(code, env)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().endswith("unfused"), out.stdout
