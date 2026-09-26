"""Quick op-coverage sweep on metal: each case runs in isolation. Floating
results are compared with a float64 CPU reference in ulps
(tests/metal_testing.py); tolerances are ~2x the measured error."""
import numpy as np
import jax, jax.numpy as jnp
import pytest

from metal_testing import check

pytestmark = pytest.mark.metal

CASES = {}
def case(name):
    def deco(fn):
        CASES[name] = fn
        return fn
    return deco

f32 = jnp.float32
bf16 = jnp.bfloat16
rng = np.random.default_rng(0)
@case("matmul f32 64x64")
def _():
    a = np.arange(64*64, dtype=np.float32).reshape(64, 64) / 4096; b = a.T + 1
    check(lambda a, b: a @ b, a, b, ulps=4.5, normwise=True, name="matmul")
@case("matmul bf16")
def _():
    a = jnp.ones((32, 48), bf16); b = jnp.ones((48, 16), bf16)
    np.testing.assert_array_equal(np.asarray(a @ b).astype(np.float32), 48.0)
@case("batched matmul")
def _():
    a = jnp.ones((4, 8, 16), f32); b = jnp.ones((4, 16, 8), f32)
    np.testing.assert_array_equal(np.asarray(jnp.einsum("bij,bjk->bik", a, b)), 16.0)
@case("dense layer + relu + softmax")
def _():
    x = np.linspace(-1, 1, 32*16, dtype=np.float32).reshape(32, 16); w = np.full((16, 8), 0.1, np.float32)
    check(lambda x, w: jax.nn.softmax(jax.nn.relu(x @ w + 0.5), axis=-1), x, w, ulps=1, normwise=True, name="dense")
@case("transpose + reshape")
def _():
    x = jnp.arange(24, dtype=f32).reshape(2, 3, 4)
    np.testing.assert_array_equal(np.asarray(jnp.transpose(x, (2, 0, 1)).reshape(-1)), np.asarray(x).transpose(2, 0, 1).reshape(-1))
@case("reduce max/argmax")
def _():
    x = jnp.array([[1., 5., 2.], [7., 0., 3.]], f32)
    assert float(jnp.max(x)) == 7.0 and int(jnp.argmax(x)) == 3, (jnp.max(x), jnp.argmax(x))
@case("cumsum")
def _():
    np.testing.assert_array_equal(np.asarray(jnp.cumsum(jnp.arange(10, dtype=f32))), np.cumsum(np.arange(10, dtype=np.float32)))
@case("fori_loop")
def _():
    r = jax.lax.fori_loop(0, 10, lambda i, c: c + i, jnp.float32(0))
    assert float(r) == 45.0, r
@case("while_loop dynamic")
def _():
    r = jax.lax.while_loop(lambda c: c[0] < 100, lambda c: (c[0] * 2, c[1] + 1), (jnp.int32(1), jnp.int32(0)))
    assert int(r[1]) == 7, r
@case("scan")
def _():
    _, ys = jax.lax.scan(lambda c, x: (c + x, c * x), jnp.float32(1), jnp.arange(5, dtype=f32))
    np.testing.assert_array_equal(np.asarray(ys), [0., 1., 4., 12., 28.])
@case("cond")
def _():
    r = jax.lax.cond(True, lambda x: x + 1, lambda x: x - 1, jnp.float32(3))
    assert float(r) == 4.0
@case("gather / indexing")
def _():
    x = jnp.arange(100, dtype=f32); idx = jnp.array([3, 7, 42])
    np.testing.assert_array_equal(np.asarray(x[idx]), [3., 7., 42.])
@case("scatter add")
def _():
    x = jnp.zeros(8, f32).at[jnp.array([1, 1, 5])].add(jnp.array([1., 2., 3.], f32))
    np.testing.assert_array_equal(np.asarray(x), [0, 3, 0, 0, 0, 3, 0, 0])
@case("dynamic_slice / update")
def _():
    x = jnp.arange(10, dtype=f32)
    y = jax.lax.dynamic_update_slice(x, jnp.array([-1., -2.], f32), (jnp.int32(4),))
    np.testing.assert_array_equal(np.asarray(jax.lax.dynamic_slice(y, (jnp.int32(3),), (4,))), [3., -1., -2., 6.])
@case("sort")
def _():
    x = jnp.array([5., 1., 4., 2., 3.], f32)
    np.testing.assert_array_equal(np.asarray(jnp.sort(x)), [1., 2., 3., 4., 5.])
@case("random threefry")
def _():
    k = jax.random.PRNGKey(0); u = jax.random.uniform(k, (1000,))
    m = float(u.mean()); assert 0.4 < m < 0.6, m
@case("bf16 elementwise")
def _():
    x = np.linspace(0, 1, 64, dtype=np.float32).astype(bf16)
    check(lambda x: jnp.tanh(x) * 2, x, ulps=1, name="bf16 tanh")
@case("int32 ops")
def _():
    x = jnp.arange(10, dtype=jnp.int32)
    np.testing.assert_array_equal(np.asarray((x * 3) % 4 + (x // 2)), (np.arange(10) * 3) % 4 + np.arange(10) // 2)
@case("f16 math")
def _():
    x = np.linspace(0.1, 2.0, 32, dtype=np.float16)
    check(lambda x: jnp.log(x) + jnp.sqrt(x), x, ulps=7.2, name="f16 log+sqrt")
@case("grad of mlp")
def _():
    def loss(w, x): return jnp.sum(jnp.tanh(x @ w) ** 2)
    w = rng.standard_normal((8, 4)).astype(np.float32) * 0.3; x = rng.standard_normal((16, 8)).astype(np.float32)
    check(jax.grad(loss), w, x, ulps=1.8, normwise=True, name="grad mlp")
@case("conv2d")
def _():
    x = rng.standard_normal((1, 8, 8, 3)).astype(np.float32); w = rng.standard_normal((3, 3, 3, 4)).astype(np.float32)
    check(lambda x, w: jax.lax.conv_general_dilated(x, w, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")),
          x, w, ulps=4.2, normwise=True, name="conv2d")
@case("iota + where + select")
def _():
    x = jnp.arange(12, dtype=f32)
    np.testing.assert_array_equal(np.asarray(jnp.where(x % 2 == 0, x, -x)), [i if i % 2 == 0 else -i for i in range(12)])
@case("large elementwise 4M")
def _():
    x = jnp.ones((2048, 2048), f32)
    assert float(jnp.sum(x * 2 + 1)) == 3 * 2048 * 2048


@pytest.mark.parametrize("name", list(CASES))
def test_op(name):
    CASES[name]()
