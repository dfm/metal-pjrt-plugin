# Copyright 2026 The metal-pjrt-plugin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convolutions on MLX's steel kernels (MetalConvRewriter -> metal$conv,
metal_pjrt/conv/conv.h): forward, input gradient and weight gradient
against a float64 CPU reference, in f32, f16 and bf16 (which accumulate in
f32); which convolutions the rewriter takes; the rest still compiles on the
loop emitter; METAL_PJRT_DISABLE_REWRITES=conv.
"""
import os

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
from jax import lax

import metal_testing
from metal_testing import run_python

pytestmark = pytest.mark.metal

DTYPES = {"f32": np.float32, "f16": np.float16, "bf16": ml_dtypes.bfloat16}
# Normwise ulps of the output type: measured (METAL_TEST_REPORT_ULPS=1, M3)
# and doubled. f16/bf16 round once, after the f32 accumulation (<= 0.5
# ulp); f32 is within the f32 accumulation error of CPU float32 (fwd <= 5.7
# ulps; the weight gradients' long contractions, N * oH * oW up to 4032
# here, <= 10.7, CPU up to 38.7).
ULPS = {"f32": 10, "f16": 1, "bf16": 1}
GRAD_ULPS = {"f32": 22, "f16": 1, "bf16": 1}

NHWC = ("NHWC", "HWIO", "NHWC")
NCHW = ("NCHW", "OIHW", "NCHW")

# (name, lhs shape, rhs shape, keyword arguments of conv_general_dilated),
# each at least MetalConvRewriter's kDefaultMinFlops (4 Mflop) so it runs
# on metal$conv.
CASES = [
    ("cnn conv1", (8, 24, 24, 3), (3, 3, 3, 32),
     dict(window_strides=(1, 1), padding="SAME", dimension_numbers=NHWC)),
    ("cnn conv2", (4, 16, 16, 32), (3, 3, 32, 64),
     dict(window_strides=(2, 2), padding="SAME", dimension_numbers=NHWC)),
    ("valid unaligned", (8, 32, 24, 5), (3, 2, 5, 17),
     dict(window_strides=(1, 1), padding="VALID", dimension_numbers=NHWC)),
    ("explicit pads + rhs dilation", (8, 24, 20, 16), (3, 3, 16, 16),
     dict(window_strides=(2, 1), padding=((2, 1), (0, 3)),
          rhs_dilation=(2, 1), dimension_numbers=NHWC)),
    ("lhs dilation", (8, 14, 12, 16), (3, 3, 16, 16),
     dict(window_strides=(1, 2), padding=((2, 1), (1, 2)),
          lhs_dilation=(2, 3), dimension_numbers=NHWC)),
    ("nchw", (16, 8, 18, 20), (16, 8, 3, 3),
     dict(window_strides=(1, 2), padding=((1, 0), (2, 1)),
          rhs_dilation=(2, 1), dimension_numbers=NCHW)),
    # airbench94's first 3x3 layer: its input gradient (64 -> 24 channels)
    # takes the implicit kernel with the weight's O padded to 32.
    ("airbench g1c1 (input grad O=24)", (8, 31, 31, 24), (3, 3, 24, 64),
     dict(window_strides=(1, 1), padding="SAME", dimension_numbers=NHWC)),
    ("1d", (16, 256, 16), (5, 16, 32),
     dict(window_strides=(2,), padding="SAME",
          dimension_numbers=("NWC", "WIO", "NWC"))),
]


def _inputs(lhs, rhs, dtype):
    rng = np.random.default_rng(0)
    return (rng.standard_normal(lhs).astype(dtype),
            rng.standard_normal(rhs).astype(dtype))


def _conv(kw):
    return lambda x, w: lax.conv_general_dilated(x, w, **kw)


@pytest.mark.parametrize("dtype", list(DTYPES))
@pytest.mark.parametrize("name,lhs,rhs,kw", CASES, ids=[c[0] for c in CASES])
def test_forward(name, lhs, rhs, kw, dtype):
    x, w = _inputs(lhs, rhs, DTYPES[dtype])
    metal_testing.check(_conv(kw), x, w, ulps=ULPS[dtype], normwise=True,
                        name=f"conv {name} {dtype}")


@pytest.mark.parametrize("dtype", list(DTYPES))
@pytest.mark.parametrize("name,lhs,rhs,kw", CASES, ids=[c[0] for c in CASES])
def test_gradients(name, lhs, rhs, kw, dtype):
    # Both operands' gradients: JAX's input gradient (reverse + lhs
    # dilation) and weight gradient (batch/feature swapped), both on
    # metal$conv.
    x, w = _inputs(lhs, rhs, DTYPES[dtype])
    conv = _conv(kw)
    out = jax.eval_shape(conv, x, w)
    g = np.random.default_rng(1).standard_normal(out.shape).astype(x.dtype)

    def grads(x, w, g):
        return jax.vjp(conv, x, w)[1](g)

    metal_testing.check(grads, x, w, g, ulps=GRAD_ULPS[dtype], normwise=True,
                        name=f"conv grads {name} {dtype}")
    text = _compiled(grads, x, w, g)
    assert text.count('custom_call_target="metal$conv"') == 2, text
    assert text.count('kind = \\"wgrad\\"') == 1, text
    assert " convolution(" not in text


def _compiled(fn, *args):
    with jax.default_device(metal_testing.metal()):
        return jax.jit(fn).lower(*args).compile().as_text()


def test_rewritten():
    # The cnn bench's step (batch 8): both forward convolutions, the input
    # gradient and both weight gradients are metal$conv calls; no
    # convolution is left.
    x = np.ones((8, 32, 32, 3), np.float32)
    w1 = np.ones((3, 3, 3, 16), np.float32)
    w2 = np.ones((3, 3, 16, 32), np.float32)

    def loss(w1, w2, x):
        h = jax.nn.relu(lax.conv_general_dilated(x, w1, (1, 1), "SAME",
                                                 dimension_numbers=NHWC))
        h = jax.nn.relu(lax.conv_general_dilated(h, w2, (2, 2), "SAME",
                                                 dimension_numbers=NHWC))
        return jnp.mean(h ** 2)

    text = _compiled(jax.grad(loss, argnums=(0, 1)), w1, w2, x)
    assert text.count('custom_call_target="metal$conv"') == 5, text
    assert text.count('kind = \\"wgrad\\"') == 2, text
    assert text.count("flip = true") == 1, text
    assert " convolution(" not in text


@pytest.mark.parametrize("name,fn,shapes", [
    ("grouped", lambda x, w: lax.conv_general_dilated(
        x, w, (1, 1), "SAME", dimension_numbers=NHWC, feature_group_count=2),
     [(2, 8, 8, 8), (3, 3, 4, 8)]),
    ("3d", lambda x, w: lax.conv_general_dilated(
        x, w, (1, 1, 1), "SAME",
        dimension_numbers=("NDHWC", "DHWIO", "NDHWC")),
     [(1, 4, 5, 6, 3), (2, 2, 2, 3, 4)]),
    ("negative padding", lambda x, w: lax.conv_general_dilated(
        x, w, (1, 1), ((-1, 2), (0, 0)), dimension_numbers=NHWC),
     [(8, 32, 32, 16), (3, 3, 16, 16)]),
    # Below 4 Mflop: the loop emitter's single fused kernel is faster.
    ("small", lambda x, w: lax.conv_general_dilated(
        x, w, (1, 1), "SAME", dimension_numbers=NHWC),
     [(2, 16, 16, 16), (3, 3, 16, 16)]),
])
def test_not_rewritten(name, fn, shapes):
    # Left to the loop emitter, still correct.
    x, w = _inputs(*shapes, np.float32)
    assert "metal$conv" not in _compiled(fn, x, w)
    metal_testing.check(fn, x, w, ulps=ULPS["f32"], normwise=True,
                        name=f"conv {name}")


def _grouped_kernel_grad(groups):
    def loss(w, x):
        y = lax.conv_general_dilated(x, w, (1, 1), "SAME",
                                     dimension_numbers=NHWC,
                                     feature_group_count=groups)
        return jnp.sum(y ** 2)
    return jax.grad(loss)


@pytest.mark.parametrize("name,fn,shapes", [
    # JAX's kernel gradient of a grouped convolution is a convolution with
    # batch_group_count = the groups (jax/_src/lax/convolution.py).
    ("depthwise kernel gradient", _grouped_kernel_grad(4),
     [(3, 3, 1, 8), (2, 8, 8, 4)]),
    ("grouped kernel gradient", _grouped_kernel_grad(2),
     [(3, 3, 4, 6), (4, 8, 8, 8)]),
    ("batch groups", lambda x, w: lax.conv_general_dilated(
        x, w, (1, 1), "VALID", dimension_numbers=NHWC, batch_group_count=2),
     [(4, 8, 8, 3), (3, 3, 3, 6)]),
    ("batch groups = batch = features", lambda x, w: lax.conv_general_dilated(
        x, w, (1, 1), "VALID", dimension_numbers=NHWC, batch_group_count=4),
     [(4, 8, 8, 3), (3, 3, 3, 4)]),
])
def test_batch_groups(name, fn, shapes):
    # batch_group_count > 1: XLA's loop emitter sums over every batch group
    # for each output feature, so MetalCompiler converts these to ordinary
    # convolutions first (ConvolutionGroupConverter, as on XLA:CPU), and none
    # is left.
    a, b = _inputs(*shapes, np.float32)
    assert "batch_group_count" not in _compiled(fn, a, b)
    metal_testing.check(fn, a, b, ulps=GRAD_ULPS["f32"], normwise=True,
                        name=f"conv {name}")


def test_batch_groups_1d_large():
    # The kernel gradient of a 1-D depthwise convolution: the converter's
    # extra dimension makes it a 2-D convolution with base dilation G and
    # stride G - 1 on that dimension, which MetalConvRewriter may take (it is
    # over 4 Mflop), so metal$conv sees a weight gradient with kernel and
    # input dilation together. The tolerance is an estimate for a 65536-term
    # contraction in f32 (not measured).
    def loss(w, x):
        y = lax.conv_general_dilated(x, w, (1,), "SAME",
                                     dimension_numbers=("NWC", "WIO", "NWC"),
                                     feature_group_count=4)
        return jnp.sum(y ** 2)
    w, x = _inputs((5, 1, 8), (32, 2048, 4), np.float32)
    fn = jax.grad(loss)
    assert "batch_group_count" not in _compiled(fn, w, x)
    metal_testing.check(fn, w, x, ulps=128, normwise=True,
                        name="conv 1-d depthwise kernel gradient")


DISABLE_CONV_CHILD = """
import numpy as np, jax
from jax import lax
x = np.random.default_rng(0).standard_normal((8, 16, 16, 16)).astype(np.float32)
w = np.random.default_rng(1).standard_normal((3, 3, 16, 32)).astype(np.float32)
f = jax.jit(lambda x, w: lax.conv_general_dilated(
    x, w, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")))
text = f.lower(x, w).compile().as_text()
got = np.asarray(f(x, w))
with jax.default_device(jax.devices("cpu")[0]):
    ref = np.asarray(f(x, w))
print("metal$conv" in text, bool(np.abs(got - ref).max() <= 1e-4 * np.abs(ref).max()))
"""


@pytest.mark.parametrize("value", [None, "conv"])
def test_disable_conv(value):
    # METAL_PJRT_DISABLE_REWRITES=conv (read once per process): the loop
    # emitter takes every convolution.
    env = {k: v for k, v in os.environ.items()
           if k != "METAL_PJRT_DISABLE_REWRITES"}
    if value is not None:
        env["METAL_PJRT_DISABLE_REWRITES"] = value
    out = run_python(DISABLE_CONV_CHILD, dict(env, JAX_PLATFORMS="mtl,cpu"))
    assert out.returncode == 0, out.stderr[-3000:]
    assert out.stdout.split() == [str(value is None), "True"]
