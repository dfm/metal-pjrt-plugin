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
    # A sum starting from 1 would add 1 once per thread in the fused kernel,
    # so no fusion takes it. (XLA's own reduction emitter combines the init
    # per thread too, so the result isn't CPU's either: HLO leaves reduce
    # with a non-identity init unspecified.)
    def fn(x):
        s = jax.lax.reduce(x, jnp.ones((), x.dtype), jax.lax.add, (1,))
        return x / s[:, None]
    x = make_input((8, 256), np.float32) + 10
    assert "metal_rownorm" not in compiled_text(fn, x)


@pytest.mark.parametrize("n", [2, 7, 31])
def test_tiny_rows(n):
    for case in ("softmax", "layer_norm"):
        fn = CASES[case]
        x = make_input((37, n), np.float32)
        assert "metal_rownorm" in compiled_text(fn, x)
        assert_close(run_on(metal(), fn, x), f64_reference(fn, x),
                     ULPS[case]["float32"], normwise=NORMWISE[case],
                     name=f"{case} n={n}")


def test_int32_row_reduction():
    # A count per row (an s32 sum of a bool) scaling the row.
    fn = lambda x: x * jnp.sum(x > 0, axis=-1, keepdims=True)
    x = make_input((300, 700), np.float32)
    assert "metal_rownorm" in compiled_text(fn, x)
    np.testing.assert_array_equal(run_on(metal(), fn, x), run_on(cpu(), fn, x))


@pytest.mark.parametrize("shape", [(513, 256), (7, 1000), (1031, 32)],
                         ids=lambda s: "x".join(map(str, s)))
def test_row_outputs_with_padding_rows(shape):
    # A [rows] output reading a [rows] parameter (computed in lane 0 of
    # real rows only), with a last threadgroup that is partly padding.
    def fn(x, p):
        m = jnp.max(x, axis=-1)
        return x - m[:, None], m * p
    x = make_input(shape, np.float32)
    p = make_input(shape[:1], np.float32, seed=2)
    assert "metal_rownorm" in compiled_text(fn, x, p)
    assert_close(run_on(metal(), fn, x, p), f64_reference(fn, x, p), 1)


def lm_loss(z, labels):
    logp = jax.nn.log_softmax(z, axis=-1)
    return -jnp.take_along_axis(logp, labels[:, None], axis=-1).mean()


# Rows past 16384 (vocabularies) are fused before TreeReductionRewriter
# would split them: the LM loss and its gradient (whose one-hot scatter and
# zero fill stay outside), and a plain log-softmax.
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("n", [16385, 50304, 151936])
@pytest.mark.parametrize("what", ["loss", "grad", "log_softmax"])
def test_long_rows(what, n, dtype):
    rows = 4 if n > 100000 else 16
    z = make_input((rows, n), DTYPES[dtype])
    labels = np.random.default_rng(1).integers(0, n, rows).astype(np.int32)
    fn = {"loss": lm_loss, "grad": jax.grad(lm_loss),
          "log_softmax": lambda z, labels: jax.nn.log_softmax(z, axis=-1)}[what]
    assert "metal_rownorm" in compiled_text(fn, z, labels)
    got = run_on(metal(), fn, z, labels)
    want = f64_reference(fn, z, labels)
    # Normwise, measured (M3) and doubled: the loss is a mean of n-term
    # logsumexps.
    ulps = {"float32": 8, "bfloat16": 2}[dtype]
    assert_close(got, want, ulps, normwise=True, name=f"{what} {n} [{dtype}]",
                 cpu32=run_on(cpu(), fn, z, labels))


def gpt_style_loss(x, w, labels):
    # Logits from a matmul over [B, T, d]: XLA reduces the GEMM's [B*T, V]
    # output and computes the rest on the [B, T, V] bitcast of it.
    logp = jax.nn.log_softmax(x @ w, axis=-1)
    return -jnp.take_along_axis(logp, labels[..., None], axis=-1).mean()


@pytest.mark.parametrize("v", [1000, 20000])
def test_loss_over_matmul_logits(v):
    rng = np.random.default_rng(3)
    x = rng.standard_normal((2, 8, 16)).astype(np.float32)
    w = (rng.standard_normal((16, v)) / 4).astype(np.float32)
    labels = rng.integers(0, v, (2, 8)).astype(np.int32)
    fn = jax.grad(gpt_style_loss, argnums=(0, 1))
    text = compiled_text(fn, x, w, labels)
    assert "metal_rownorm" in text
    # The max joins the fusion: no reduction over V is left outside it.
    assert_close(run_on(metal(), fn, x, w, labels),
                 f64_reference(fn, x, w, labels), 16, normwise=True,
                 cpu32=run_on(cpu(), fn, x, w, labels))


def test_long_row_fusion_survives_the_stock_pipeline():
    # The long-row instance runs before the stock post-layout pipeline
    # (FloatNormalization, LayoutNormalization, ...); the compiled program
    # still has the one fusion, taking the scatter and the logits and
    # writing the gradient.
    z = make_input((16, 50304), np.float32)
    labels = np.arange(16, dtype=np.int32)
    text = compiled_text(jax.grad(lm_loss), z, labels)
    fusions = [l for l in text.splitlines()
               if " fusion(" in l and "metal_rownorm" in l and "calls=" in l]
    assert len(fusions) == 1, fusions
    assert fusions[0].split("=")[1].strip().startswith("f32[16,50304]"), fusions[0]
    assert fusions[0].count("%") - 2 == 2, fusions[0]  # operands


def test_long_row_nan():
    z = make_input((4, 50304), np.float32)
    z[1, 40000] = np.nan
    got = run_on(metal(), CASES["log_softmax"], z)
    assert np.isnan(got[1]).all()
    assert_close(np.delete(got, 1, 0),
                 np.delete(run_on(cpu(), CASES["log_softmax"], z), 1, 0), 4,
                 normwise=True)


@pytest.mark.parametrize("n", [128, 50304], ids=["short", "long"])
def test_kill_switch(n):
    # One switch for both instances (short rows after the stock post-layout
    # pipeline, long rows before it).
    env = dict(os.environ, METAL_PJRT_DISABLE_REWRITES="rownorm")
    code = (
        "import jax, jax.numpy as jnp, numpy as np\n"
        f"x = jnp.asarray(np.linspace(-3, 3, 8 * {n}, dtype=np.float32).reshape(8, {n}))\n"
        "f = jax.jit(lambda x: jax.nn.softmax(x, -1))\n"
        "t = f.lower(x).compile().as_text()\n"
        "print('fused' if 'metal_rownorm' in t else 'unfused')\n"
        "print(repr(np.asarray(f(x)).tolist()))\n")
    out = run_python(code, env)
    assert out.returncode == 0, out.stderr
    lines = out.stdout.strip().splitlines()
    assert lines[-2] == "unfused", out.stdout
    unfused = np.array(eval(lines[-1]), np.float32)
    x = np.linspace(-3, 3, 8 * n, dtype=np.float32).reshape(8, n)
    fn = lambda x: jax.nn.softmax(x, -1)
    assert "metal_rownorm" in compiled_text(fn, x)
    assert_close(run_on(metal(), fn, x), unfused, 38)
