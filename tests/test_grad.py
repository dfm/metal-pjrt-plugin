# Copyright 2026 The metal-pjrt-plugin Authors
# SPDX-License-Identifier: Apache-2.0

"""Automatic differentiation on metal against a float64 CPU reference
(tests/metal_testing.py): VJPs of elementwise ops, reductions, indexing and
dots; grad through scan, cond, while (forward mode) and remat;
value_and_grad with has_aux; jacfwd / jacrev / hessian; grad of grad and
vmap of grad; custom_vjp; a conv weight gradient through jax.grad; and
eager grad next to jit. Gradients of convolutions, linear algebra, FFTs,
pooling and half-precision GEMMs are also in those ops' own tests.

Tolerances (ULPS) are ~2x the measured error, in ulps of the output dtype,
elementwise for ELEMENTWISE cases and normwise otherwise. Measure with:
  scripts/device_lock.py -- env METAL_TEST_REPORT_ULPS=1 .venv/bin/python -m pytest tests/test_grad.py -s
"""
import ml_dtypes
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
import pytest

import metal_testing

pytestmark = pytest.mark.metal


def R(*s, seed=None):
    rng = np.random.default_rng(int(np.prod(s)) if seed is None else seed)
    return rng.standard_normal(s).astype(np.float32)


POS = np.linspace(0.1, 3.0, 64, dtype=np.float32)  # for log, sqrt, pow
UNIT = np.linspace(-0.9, 0.9, 64, dtype=np.float32)  # for asin, atanh
# Avoids 0, where abs / relu / sign-like ops have a kink.
X = np.linspace(-3.0, 3.0, 64, dtype=np.float32) + 0.01


def grad_sum(f):
    """d/dx sum(f(x)): f' elementwise, for unary f."""
    return jax.grad(lambda x: jnp.sum(f(x)))


CASES = {}  # name -> (fn, args)
ELEMENTWISE = set()  # compared elementwise; the rest normwise

# ---- elementwise VJPs ----
for name, f, x in [
    ("exp", jnp.exp, X), ("log", jnp.log, POS), ("log1p", jnp.log1p, POS),
    ("expm1", jnp.expm1, X), ("sin", jnp.sin, X), ("cos", jnp.cos, X),
    ("tan", jnp.tan, UNIT), ("tanh", jnp.tanh, X),
    ("sigmoid", jax.nn.sigmoid, X), ("softplus", jax.nn.softplus, X),
    ("sqrt", jnp.sqrt, POS), ("rsqrt", lax.rsqrt, POS),
    ("pow 2.5", lambda v: v ** 2.5, POS), ("integer_pow 3", lambda v: v ** 3, X),
    ("reciprocal", lambda v: 1 / v, POS), ("abs", jnp.abs, X),
    ("relu", jax.nn.relu, X), ("gelu", jax.nn.gelu, X),
    ("silu", jax.nn.silu, X), ("erf", jax.scipy.special.erf, X),
    ("asin", jnp.arcsin, UNIT), ("atanh", jnp.arctanh, UNIT),
    ("atan", jnp.arctan, X), ("sinh", jnp.sinh, X),
    ("clip", lambda v: jnp.clip(v, -1.0, 1.5), X),
    ("where", lambda v: jnp.where(v > 0, v ** 2, jnp.sin(v)), X),
]:
    CASES[f"grad {name}"] = (grad_sum(f), (x,))
    ELEMENTWISE.add(f"grad {name}")
CASES["grad binary (x * y, x / y, atan2, logaddexp)"] = (
    jax.grad(lambda x, y: jnp.sum(x * y + x / y + jnp.arctan2(x, y)
                                  + jnp.logaddexp(x, y)), argnums=(0, 1)),
    (X, POS))
# Ties: JAX splits the cotangent of max/min evenly between equal operands.
CASES["grad maximum/minimum with ties"] = (
    jax.grad(lambda x, y: jnp.sum(jnp.maximum(x, y) * 2 + jnp.minimum(x, y)),
             argnums=(0, 1)),
    (np.array([1, 2, 3, 4, 5, 6], np.float32),
     np.array([1, 0, 3, 5, 5, 7], np.float32)))
ELEMENTWISE.add("grad maximum/minimum with ties")
CASES["grad broadcasting"] = (
    jax.grad(lambda x, y: jnp.sum(jnp.tanh(x[:, None] * y[None, :] + y)),
             argnums=(0, 1)),
    (R(12), R(20)))

# ---- reductions ----
CASES["grad sum/mean axes"] = (
    jax.grad(lambda x: jnp.sum(jnp.sum(x, 0) ** 2) + jnp.sum(jnp.mean(x, (1, 2)) ** 3)),
    (R(6, 7, 8),))
# Distinct maxima (ties would follow JAX's even split, tested above for
# maximum).
CASES["grad max/min reduce"] = (
    jax.grad(lambda x: jnp.sum(jnp.max(x, -1) ** 2) - jnp.sum(jnp.min(x, 0))),
    (R(16, 24),))
CASES["grad prod"] = (jax.grad(lambda x: jnp.sum(jnp.prod(x, -1))),
                      (1 + 0.1 * R(8, 10),))
CASES["grad logsumexp"] = (
    jax.grad(lambda x: jnp.sum(jax.scipy.special.logsumexp(x, -1))), (R(16, 33),))
CASES["grad var/std"] = (
    jax.grad(lambda x: jnp.sum(jnp.var(x, 0)) + jnp.sum(jnp.std(x, 1))), (R(20, 17),))
CASES["grad cumsum/cumprod"] = (
    jax.grad(lambda x: jnp.sum(jnp.cumsum(x, -1) ** 2)
             + jnp.sum(jnp.cumprod(1 + 0.1 * x, 0))), (R(10, 50),))
CASES["grad softmax cross entropy"] = (
    jax.grad(lambda z, y: -jnp.mean(jnp.sum(jax.nn.log_softmax(z) * y, -1))),
    (R(32, 10), jax.nn.one_hot(np.arange(32) % 10, 10)))
CASES["grad sort"] = (jax.grad(lambda x: jnp.sum(jnp.sort(x, -1) * jnp.arange(30.0))),
                      (R(4, 30),))

# ---- indexing and data movement ----
IDX = np.array([3, 0, 7, 3, 3, 9, 1], np.int32)  # repeats: scatter-add
CASES["grad gather (x[idx])"] = (
    jax.grad(lambda x: jnp.sum(jnp.sin(x[IDX]) * jnp.arange(7.0)[:, None])),
    (R(10, 5),))
CASES["grad scatter add (.at[].add)"] = (
    jax.grad(lambda x, u: jnp.sum(jnp.cos(x.at[IDX].add(u))), argnums=(0, 1)),
    (R(10, 5), R(7, 5)))
CASES["grad scatter set (.at[].set)"] = (
    jax.grad(lambda x, u: jnp.sum(x.at[::2].set(u ** 2) ** 2), argnums=(0, 1)),
    (R(10, 3), R(5, 3)))
CASES["grad take_along_axis"] = (
    jax.grad(lambda x: jnp.sum(jnp.take_along_axis(
        x, jnp.argsort(x, -1)[:, :4], -1) ** 2)), (R(6, 12),))
CASES["grad dynamic_slice / dynamic_update_slice"] = (
    jax.grad(lambda x, i: jnp.sum(lax.dynamic_update_slice(
        x, lax.dynamic_slice(x, (i, i - i), (3, 4)) ** 2, (i - i, i)) ** 2)),
    (R(8, 9), np.int32(2)))
CASES["grad reshape/transpose/concat/pad/rev"] = (
    jax.grad(lambda x: jnp.sum(jnp.sin(jnp.pad(jnp.concatenate(
        [x.T, x[::-1].reshape(6, 4)], 0), ((1, 2), (0, 1))))
        * jnp.arange(15.0)[:, None])),
    (R(4, 6),))

# ---- dots ----
CASES["grad matmul"] = (
    jax.grad(lambda a, b: jnp.sum(jnp.tanh(a @ b)), argnums=(0, 1)),
    (R(33, 40), R(40, 24)))
CASES["grad matmul large (GEMM)"] = (
    jax.grad(lambda a, b: jnp.sum((a @ b) ** 2), argnums=(0, 1)),
    (R(128, 256), R(256, 96)))
CASES["grad matvec / vecmat"] = (
    jax.grad(lambda a, v, w: jnp.sum(jnp.sin(a @ v)) + jnp.sum(jnp.cos(w @ a)),
             argnums=(0, 1, 2)), (R(30, 50), R(50), R(30)))
CASES["grad batched einsum"] = (
    jax.grad(lambda a, b: jnp.sum(jnp.einsum("bij,bjk->bik", a, b) ** 2),
             argnums=(0, 1)), (R(5, 12, 20), R(5, 20, 9)))
CASES["grad batched einsum broadcast rhs"] = (
    jax.grad(lambda a, b: jnp.sum(jnp.einsum("bij,jk->bik", a, b) ** 2),
             argnums=(0, 1)), (R(5, 12, 20), R(20, 9)))
# Contracting and batch dimensions that are not the trailing ones.
CASES["grad dot_general odd dims"] = (
    jax.grad(lambda a, b: jnp.sum(jnp.tanh(lax.dot_general(
        a, b, (((0, 3), (2, 0)), ((1,), (1,)))))), argnums=(0, 1)),
    (R(6, 3, 7, 4), R(4, 3, 6, 5)))
CASES["grad attention"] = (
    jax.grad(lambda q, k, v: jnp.sum(jax.nn.softmax(
        q @ k.swapaxes(-1, -2) / 4.0, -1) @ v), argnums=(0, 1, 2)),
    (R(2, 16, 16), R(2, 16, 16, seed=1), R(2, 16, 16, seed=2)))

# ---- control flow, remat ----
def _rnn_loss(w, h0, xs):
    def step(h, x):
        h = jnp.tanh(h @ w + x)
        return h, jnp.sum(h ** 2)
    h, ys = lax.scan(step, h0, xs)
    return jnp.sum(ys) + jnp.sum(h)
CASES["grad scan (rnn)"] = (jax.grad(_rnn_loss, argnums=(0, 1, 2)),
                            (0.3 * R(8, 8), R(4, 8), R(20, 4, 8)))
CASES["grad scan reverse, unroll 2"] = (
    jax.grad(lambda c, xs: jnp.sum(lax.scan(
        lambda c, x: (c * jnp.cos(x) + x, c), c, xs, reverse=True, unroll=2)[1]),
        argnums=(0, 1)), (R(6), R(15, 6)))
CASES["grad fori_loop (static bounds)"] = (
    jax.grad(lambda x: jnp.sum(lax.fori_loop(0, 6, lambda i, a: jnp.sin(a) * 1.2 + i, x))),
    (R(16),))
CASES["grad cond"] = (
    jax.grad(lambda x, p: lax.cond(p > 0, lambda v: jnp.sum(jnp.exp(v)),
                                   lambda v: jnp.sum(v ** 3), x)
             + lax.cond(p < 0, lambda v: jnp.sum(jnp.exp(v)),
                        lambda v: jnp.sum(v ** 3), x)),
    (R(10), np.float32(1.0)))
# Reverse mode cannot go through a while_loop with a dynamic trip count;
# forward mode can.
CASES["jacfwd while_loop"] = (
    jax.jacfwd(lambda x: lax.while_loop(lambda c: c[1] < 5,
                                        lambda c: (jnp.sin(c[0]) * 1.1, c[1] + 1),
                                        (x, 0))[0]),
    (R(6),))
CASES["grad checkpoint"] = (
    jax.grad(lambda w, x: jnp.sum(jax.checkpoint(
        lambda w, x: jnp.tanh(jnp.tanh(x @ w) @ w))(w, x) ** 2), argnums=(0, 1)),
    (0.3 * R(16, 16), R(8, 16)))

# ---- transformations ----
def _loss_aux(w, x):
    y = jnp.tanh(x @ w)
    return jnp.mean(y ** 2), {"max": jnp.max(y), "y": y}
CASES["value_and_grad has_aux"] = (
    jax.value_and_grad(_loss_aux, has_aux=True), (R(8, 4), R(16, 8)))
CASES["jacfwd"] = (jax.jacfwd(lambda x: jnp.sin(x) * jnp.sum(x ** 2)), (R(7),))
CASES["jacrev"] = (jax.jacrev(lambda x: jnp.tanh(R(5, 7) @ x) * x[:5]), (R(7),))
CASES["jacrev matrix -> vector"] = (
    jax.jacrev(lambda a: jnp.sum(jnp.sin(a), 0)), (R(4, 6),))
CASES["hessian"] = (jax.hessian(lambda x: jnp.sum(jnp.exp(x) * x[::-1])
                                + jnp.sum(x) ** 3), (R(8),))
CASES["grad of grad"] = (jax.grad(lambda x: jnp.sum(grad_sum(jnp.tanh)(x))), (X,))
CASES["vmap of grad (per-example)"] = (
    jax.vmap(jax.grad(lambda w, x: jnp.sum(jnp.tanh(x @ w) ** 2)),
             in_axes=(None, 0)), (R(8, 4), R(10, 3, 8)))
CASES["jvp"] = (lambda x, t: jax.jvp(lambda v: jnp.sin(v) @ v.T, (x,), (t,)),
                (R(6, 5), R(6, 5, seed=1)))
CASES["linearize/vjp"] = (
    lambda x, g: jax.vjp(lambda v: jnp.cumsum(jnp.exp(v)), x)[1](g), (X, R(64)))


@jax.custom_vjp
def _clip_grad(x):
    return x
_clip_grad.defvjp(lambda x: (x, None), lambda _, g: (jnp.clip(g, -0.5, 0.5),))
CASES["custom_vjp"] = (
    jax.grad(lambda x: jnp.sum(_clip_grad(x) * jnp.arange(16.0) / 8)), (R(16),))

# The conv weight gradient whose f32 partials were misaligned (fixed on
# 2026-09-28): N=1, 63x63x3, 3x3 VALID -> 32, so unfold_rows * k (3721 * 27)
# is odd, through jax.grad in every precision (the partials follow f16/bf16
# patches). test_conv_wgrad_rewritten checks it runs on metal$conv.
CONV_DTYPES = {"f32": np.float32, "f16": np.float16, "bf16": ml_dtypes.bfloat16}
for d, dt in CONV_DTYPES.items():
    CASES[f"grad conv odd wgrad partials {d}"] = (
        jax.grad(lambda w, x: jnp.sum(lax.conv_general_dilated(
            x, w, (1, 1), "VALID",
            dimension_numbers=("NHWC", "HWIO", "NHWC")).astype(jnp.float32) ** 2),
            argnums=(0, 1)),
        (R(3, 3, 3, 32).astype(dt), R(1, 63, 63, 3).astype(dt)))

# ---- backward programs that take a path the forward one does not ----
# Convolutions XLA's loop emitter runs (small, grouped, 3-D): their gradients
# are convolutions the forward pass never has, with input dilation (the
# input's gradient of a strided one), kernel dilation (the kernel's) and,
# for a grouped one, batch groups (ConvolutionGroupConverter, see
# tests/test_conv.py for the stride-1 kernel gradients).
NHWC = ("NHWC", "HWIO", "NHWC")


def _conv_grads(dimension_numbers=NHWC, **kw):
    return jax.grad(lambda x, w: jnp.sum(lax.conv_general_dilated(
        x, w, dimension_numbers=dimension_numbers, **kw) ** 2), argnums=(0, 1))


CASES["grad small conv, strided"] = (
    _conv_grads(window_strides=(2, 2), padding="SAME"),
    (R(2, 9, 9, 3), R(3, 3, 3, 4)))
CASES["grad small conv, kernel dilation and uneven padding"] = (
    _conv_grads(window_strides=(1, 2), padding=((1, 2), (0, 1)),
                rhs_dilation=(2, 1)),
    (R(2, 9, 8, 3), R(3, 2, 3, 4)))
CASES["grad grouped conv, strided"] = (
    _conv_grads(window_strides=(2, 2), padding="SAME", feature_group_count=2),
    (R(2, 9, 9, 4), R(3, 3, 2, 6)))
CASES["grad depthwise conv"] = (
    _conv_grads(window_strides=(1, 1), padding="SAME", feature_group_count=4),
    (R(2, 8, 8, 4), R(3, 3, 1, 8)))
CASES["grad conv 3d, strided"] = (
    _conv_grads(dimension_numbers=("NDHWC", "DHWIO", "NDHWC"),
                window_strides=(1, 2, 1), padding="SAME"),
    (R(1, 4, 5, 6, 3), R(2, 2, 2, 3, 4)))
# Sum (average) pooling: the transpose pads the cotangent with interior
# zeros (stride - 1) and sums windows of it.
CASES["grad sum pool 3x3 stride 2 SAME"] = (
    jax.grad(lambda x: jnp.sum(lax.reduce_window(
        x, 0.0, lax.add, (1, 3, 3, 1), (1, 2, 2, 1), "SAME") ** 2)),
    (R(2, 9, 9, 3),))
# max over an axis with ties: the cotangent is split evenly among the
# maxima (an equality mask and a count; exact here).
CASES["grad max reduce with ties"] = (
    jax.grad(lambda x: jnp.sum(jnp.max(x, -1) * jnp.arange(1.0, 5.0))),
    (np.array([[1, 3, 3, 0], [2, 2, 2, 2], [0, -1, 0, -1], [5, 1, 2, 3]],
              np.float32),))
ELEMENTWISE.add("grad max reduce with ties")
# An embedding lookup in bf16: its gradient is a scatter-add on 16-bit
# elements with repeated indices (XLA's compare-and-swap on the 32-bit word).
CASES["grad embedding lookup bf16"] = (
    jax.grad(lambda e, w: jnp.sum((e[IDX] * w).astype(jnp.float32) ** 2)),
    (R(10, 8).astype(ml_dtypes.bfloat16), R(7, 8).astype(ml_dtypes.bfloat16)))

# Measured (METAL_TEST_REPORT_ULPS=1, M3) and doubled; cases missing here
# must match exactly. The large elementwise errors (tanh, sigmoid, gelu,
# silu, expm1) are relative to derivatives near zero, large on CPU float32
# as well (tanh 127, gelu 223, silu 65, expm1 8.5). grad broadcasting is
# 1.37, and 3.03 under jax < 0.11.2, whose lowering differs.
ULPS = {
    'grad exp': 2.2, 'grad log': 1, 'grad log1p': 2.4, 'grad expm1': 47,
    'grad sin': 2.7, 'grad cos': 2.9, 'grad tan': 3.5, 'grad tanh': 260,
    'grad sigmoid': 30, 'grad softplus': 3.9, 'grad sqrt': 1,
    'grad rsqrt': 1.9, 'grad pow 2.5': 3.2, 'grad integer_pow 3': 2.5,
    'grad reciprocal': 2.3, 'grad gelu': 450, 'grad silu': 340,
    'grad erf': 9.0, 'grad asin': 1.2, 'grad atanh': 2.9, 'grad atan': 2.9,
    'grad sinh': 3.6, 'grad where': 2.0,
    'grad binary (x * y, x / y, atan2, logaddexp)': 2.1,
    'grad broadcasting': 6.1 if metal_testing.OLD_JAX else 2.8,
    'grad sum/mean axes': 1.8, 'grad prod': 4.3,
    'grad logsumexp': 1, 'grad var/std': 2.4, 'grad cumsum/cumprod': 3.5,
    'grad softmax cross entropy': 2.1, 'grad gather (x[idx])': 1.1,
    'grad scatter add (.at[].add)': 8.1, 'grad scatter set (.at[].set)': 1,
    'grad dynamic_slice / dynamic_update_slice': 1,
    'grad reshape/transpose/concat/pad/rev': 2.9, 'grad matmul': 21,
    'grad matmul large (GEMM)': 14, 'grad matvec / vecmat': 11,
    'grad batched einsum': 2.3, 'grad batched einsum broadcast rhs': 1.6,
    'grad dot_general odd dims': 13, 'grad attention': 6.0,
    'grad scan (rnn)': 15, 'grad scan reverse, unroll 2': 2.2,
    'grad fori_loop (static bounds)': 11, 'grad cond': 1,
    'jacfwd while_loop': 3.9, 'grad checkpoint': 5.3,
    'value_and_grad has_aux': 17, 'jacfwd': 1.3, 'jacrev': 4.2,
    'jacrev matrix -> vector': 3.5, 'hessian': 2.2, 'grad of grad': 16,
    'vmap of grad (per-example)': 11, 'jvp': 3.1, 'linearize/vjp': 1.5,
    'grad conv odd wgrad partials f32': 11,
    'grad conv odd wgrad partials f16': 1,
    'grad conv odd wgrad partials bf16': 1.1,
    'grad small conv, strided': 4.6,
    'grad small conv, kernel dilation and uneven padding': 2.9,
    'grad grouped conv, strided': 2.8, 'grad depthwise conv': 3.5,
    'grad conv 3d, strided': 4.9, 'grad sum pool 3x3 stride 2 SAME': 1.8,
    'grad embedding lookup bf16': 1,
}


@pytest.mark.parametrize("name", list(CASES))
def test_grad(name):
    fn, args = CASES[name]
    metal_testing.check(fn, *args, ulps=ULPS.get(name, 0),
                        normwise=name not in ELEMENTWISE, name=name)


@pytest.mark.parametrize("name", ["grad matmul", "grad scan (rnn)",
                                  "value_and_grad has_aux", "hessian"])
def test_eager_matches_jit(name):
    # Unjitted: each primitive (forward and backward) is its own executable.
    fn, args = CASES[name]
    with jax.default_device(metal_testing.metal()):
        got = jax.tree.map(np.asarray, fn(*[jax.device_put(a) for a in args]))
    want = metal_testing.f64_reference(fn, *args)
    metal_testing.assert_close(got, want, ULPS.get(name, 0), normwise=True,
                               name=f"eager {name}")


@pytest.mark.parametrize("d", list(CONV_DTYPES))
def test_conv_wgrad_rewritten(d):
    fn, args = CASES[f"grad conv odd wgrad partials {d}"]
    with jax.default_device(metal_testing.metal()):
        text = jax.jit(fn).lower(*args).compile().as_text()
    assert text.count('kind = \\"wgrad\\"') == 1, text


LOOP_EMITTER_CONVS = [
    "grad small conv, strided",
    "grad small conv, kernel dilation and uneven padding",
    "grad grouped conv, strided", "grad depthwise conv",
    "grad conv 3d, strided"]


@pytest.mark.parametrize("name", LOOP_EMITTER_CONVS)
def test_small_conv_grads_stay_on_the_loop_emitter(name):
    # These cases are about XLA's loop emitter: far below MetalConvRewriter's
    # 4 Mflop, grouped or 3-D, so no convolution of the forward or backward
    # program becomes metal$conv, and the grouped kernel gradients' batch
    # groups are converted away.
    fn, args = CASES[name]
    with jax.default_device(metal_testing.metal()):
        text = jax.jit(fn).lower(*args).compile().as_text()
    assert "metal$conv" not in text, text
    assert "batch_group_count" not in text, text


def test_grad_through_complex_indexing():
    # The gradient of z[idx] is a complex64 scatter-add with repeated
    # indices (IDX), split into two f32 scatter-adds by
    # MetalComplexScatterSplitter.
    z = (R(10) + 1j * R(10, seed=1)).astype(np.complex64)
    fn = jax.jit(jax.grad(lambda z: jnp.sum(jnp.abs(z[IDX]) ** 2)))
    want = np.asarray(fn(jax.device_put(z, metal_testing.cpu())))
    got = np.asarray(fn(jax.device_put(z, metal_testing.metal())))
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)
