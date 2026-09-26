"""Host callbacks on metal vs cpu: pure_callback, io_callback (ordered and
unordered), debug.print, debug.callback, errors, vmap/grad. These test
callback plumbing, so values are compared with CPU in the same precision
(ulps; tests/metal_testing.py), and printed output must match exactly.
"""
import contextlib, io
import numpy as np
import jax, jax.numpy as jnp
from jax import lax
from jax.experimental import io_callback
import pytest

from metal_testing import assert_close, cpu, metal

pytestmark = pytest.mark.metal

CASES = []
def case(fn):
    CASES.append(fn); return fn

def on(dev, fn, *args):
    """Run jit(fn) on `dev`; return (outputs as numpy, captured stdout)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = jax.jit(fn)(*[jax.device_put(a, dev) for a in args])
        out = jax.tree.map(np.asarray, out)
        jax.effects_barrier()
    return out, buf.getvalue()

def both(fn, *args, ulps=0):
    got, got_out = on(metal(), fn, *args)
    exp, exp_out = on(cpu(), fn, *args)
    assert_close(got, exp, ulps, normwise=True, name=fn.__qualname__)
    return got_out, exp_out

x8 = np.arange(-4, 4, dtype=np.float32) / 4  # dyadic: sums are exact in any order

@case
def pure_callback_multi_output():
    def host(a, n):
        a = np.asarray(a)
        return np.sin(a) * 2, (a > 0), np.asarray(n) + np.arange(3, dtype=np.int32)
    def f(x, n):
        y = jnp.cos(x) * 3                              # GPU work before
        s, pos, k = jax.pure_callback(host, (jax.ShapeDtypeStruct(y.shape, y.dtype),
                                            jax.ShapeDtypeStruct(y.shape, jnp.bool_),
                                            jax.ShapeDtypeStruct((3,), jnp.int32)), y, n)
        return s + 1, jnp.where(pos, s, -s), k * 2       # GPU work after
    both(f, x8, jnp.int32(5))

@case
def pure_callback_large_and_empty():
    def f(x):
        y = x * 2 + 1
        z, e = jax.pure_callback(lambda a: (np.asarray(a)[::-1].copy(), np.zeros((0, 3), np.float32)),
                                 (jax.ShapeDtypeStruct(y.shape, y.dtype), jax.ShapeDtypeStruct((0, 3), jnp.float32)), y)
        return jnp.sum(z * y), z[:5], e
    both(f, jnp.arange(1 << 20, dtype=jnp.float32) / (1 << 20), ulps=4)

@case
def pure_callback_2d_bf16():
    def f(x):
        y = (x @ x.T).astype(jnp.bfloat16)
        return jax.pure_callback(lambda a: np.asarray(a).T.astype(a.dtype), jax.ShapeDtypeStruct(y.shape, y.dtype), y).astype(jnp.float32)
    both(f, jnp.arange(12, dtype=jnp.float32).reshape(3, 4), ulps=0)

@case
def pure_callback_vmap():
    def f(x):
        g = lambda v: jax.pure_callback(lambda a: np.asarray(a) ** 2, jax.ShapeDtypeStruct(v.shape, v.dtype), v, vmap_method="sequential")
        h = lambda v: jax.pure_callback(lambda a: np.asarray(a) + 1, jax.ShapeDtypeStruct(v.shape, v.dtype), v, vmap_method="expand_dims")
        return jax.vmap(g)(x), jax.vmap(h)(x)
    both(f, jnp.arange(12, dtype=jnp.float32).reshape(3, 4))

@case
def pure_callback_custom_jvp_grad():
    @jax.custom_jvp
    def sin_host(x):
        return jax.pure_callback(lambda a: np.sin(a), jax.ShapeDtypeStruct(x.shape, x.dtype), x)
    sin_host.defjvp(lambda p, t: (sin_host(p[0]), jnp.cos(p[0]) * t[0]))
    both(lambda x: jax.grad(lambda v: jnp.sum(sin_host(v) ** 2))(x), x8, ulps=1)

LOG = []
@case
def io_callback_unordered():
    LOG.clear()
    def f(x):
        y = io_callback(lambda a: (LOG.append(np.asarray(a).copy()), np.asarray(a) * 2)[1],
                        jax.ShapeDtypeStruct(x.shape, x.dtype), x + 1)
        return y - 1
    got, _ = on(metal(), f, x8)
    assert_close(got, (x8.astype(np.float64) + 1) * 2 - 1, ulps=0, normwise=True, name="io_callback")
    assert len(LOG) == 1 and np.allclose(LOG[0], np.asarray(x8) + 1), LOG

@case
def io_callback_ordered_in_scan():
    def run(dev):
        LOG.clear()
        def f(x):
            def body(c, i):
                io_callback(lambda i, c: LOG.append((int(i), float(c))), None, i, c, ordered=True)
                return c * 1.5 + i, c
            return lax.scan(body, x, jnp.arange(6, dtype=jnp.float32))
        out, _ = on(dev, f, jnp.float32(1.0))
        return out, list(LOG)
    (g, glog), (e, elog) = run(metal()), run(cpu())
    assert_close(g, e, ulps=0, normwise=True, name="ordered scan")
    assert glog == elog and [i for i, _ in glog] == list(range(6)), (glog, elog)

@case
def io_callback_vmap_unordered():
    def f(x):
        return jax.vmap(lambda v: io_callback(lambda a: np.asarray(a) * 3, jax.ShapeDtypeStruct(v.shape, v.dtype), v))(x)
    both(f, jnp.arange(6, dtype=jnp.float32).reshape(2, 3))

@case
def debug_print_jit_and_scan():
    def f(x):
        jax.debug.print("x0={a} sum={b}", a=x[0], b=jnp.sum(x))
        def body(c, i):
            jax.debug.print("step {} c={}", i, c)
            return c + i, None
        c, _ = lax.scan(body, jnp.float32(0), jnp.arange(4, dtype=jnp.float32))
        return c
    got, exp = both(f, x8)
    assert got == exp and got.count("step") == 4, (got, exp)

@case
def debug_print_ordered():
    def f(x):
        for k in range(3):
            jax.debug.print("k={} v={}", k, x[k], ordered=True)
        return x * 2
    got, exp = both(f, x8)
    assert got == exp and got.splitlines()[0].startswith("k=0"), (got, exp)

@case
def debug_print_under_grad():
    def f(x):
        def loss(v):
            jax.debug.print("v={}", v)
            return jnp.sum(v ** 2)
        return jax.grad(loss)(x)
    got, exp = both(f, x8)
    assert got == exp and "v=" in got

@case
def debug_callback_side_effect():
    seen = []
    def f(x):
        y = jnp.exp(x)
        jax.debug.callback(lambda a, b: seen.append((np.asarray(a).copy(), int(b))), y, jnp.int32(7))
        return y
    on(metal(), f, x8)
    assert len(seen) == 1 and seen[0][1] == 7, seen
    assert_close(seen[0][0], np.exp(x8.astype(np.float64)), ulps=2.4, name="debug.callback exp")

@case
def callback_exception_surfaces():
    def bad(a):
        raise ValueError("boom from host")
    f = jax.jit(lambda x: jax.pure_callback(bad, jax.ShapeDtypeStruct(x.shape, x.dtype), x) + 1)
    try:
        np.asarray(f(jax.device_put(x8, metal())))
    except Exception as e:  # noqa: BLE001
        assert "boom from host" in str(e), e
    else:
        raise AssertionError("no exception")
    # The device is still usable afterwards.
    np.testing.assert_array_equal(np.asarray(jax.jit(lambda x: x + 1)(jax.device_put(x8, metal()))), x8 + 1)

@case
def wrong_shape_reported():
    f = jax.jit(lambda x: jax.pure_callback(lambda a: np.zeros(3, np.float32), jax.ShapeDtypeStruct(x.shape, x.dtype), x))
    try:
        np.asarray(f(jax.device_put(x8, metal())))
    except Exception as e:  # noqa: BLE001
        assert "Incorrect output shape" in str(e), e
    else:
        raise AssertionError("no exception")

@case
def many_calls_same_executable():
    f = jax.jit(lambda x: jax.pure_callback(lambda a: np.asarray(a) + 1, jax.ShapeDtypeStruct(x.shape, x.dtype), x) * 2)
    x = jax.device_put(x8, metal())
    for _ in range(50):
        x = f(x)
    assert_close(np.asarray(x), x8.astype(np.float64) * 2**50 + (2**51 - 2), ulps=1, name="50 calls")

@pytest.mark.parametrize("fn", CASES, ids=[f.__name__ for f in CASES])
def test_callback(fn):
    fn()
