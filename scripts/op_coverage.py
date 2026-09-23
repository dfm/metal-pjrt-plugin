"""Quick op-coverage sweep on the Metal backend: each case runs in isolation
and reports pass/fail with the first line of the error."""
import faulthandler, sys, traceback
import numpy as np
import jax, jax.numpy as jnp

CASES = {}
def case(name):
    def deco(fn):
        CASES[name] = fn
        return fn
    return deco

f32 = jnp.float32
@case("matmul f32 64x64")
def _():
    a = jnp.arange(64*64, dtype=f32).reshape(64, 64) / 4096; b = a.T + 1
    np.testing.assert_allclose(np.asarray(a @ b), np.asarray(a) @ np.asarray(b), rtol=1e-4, atol=1e-4)
@case("matmul bf16")
def _():
    a = jnp.ones((32, 48), jnp.bfloat16); b = jnp.ones((48, 16), jnp.bfloat16)
    np.testing.assert_allclose(np.asarray(a @ b).astype(np.float32), 48.0)
@case("batched matmul")
def _():
    a = jnp.ones((4, 8, 16), f32); b = jnp.ones((4, 16, 8), f32)
    np.testing.assert_allclose(np.asarray(jnp.einsum("bij,bjk->bik", a, b)), 16.0)
@case("dense layer + relu + softmax")
def _():
    x = jnp.linspace(-1, 1, 32*16, dtype=f32).reshape(32, 16); w = jnp.ones((16, 8), f32) * 0.1
    y = jax.nn.softmax(jax.nn.relu(x @ w + 0.5), axis=-1)
    np.testing.assert_allclose(np.asarray(y).sum(-1), 1.0, rtol=1e-5)
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
    np.testing.assert_array_equal(np.asarray(ys), [0., 1., 4., 12., 32.])  # c: 1,1,2,4,7 -> c*x
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
    x = jnp.linspace(0, 1, 64, dtype=jnp.bfloat16)
    np.testing.assert_allclose(np.asarray(jnp.tanh(x) * 2).astype(np.float32), np.tanh(np.asarray(x).astype(np.float32)) * 2, rtol=2e-2)
@case("int32 ops")
def _():
    x = jnp.arange(10, dtype=jnp.int32)
    np.testing.assert_array_equal(np.asarray((x * 3) % 4 + (x // 2)), (np.arange(10) * 3) % 4 + np.arange(10) // 2)
@case("f16 math")
def _():
    x = jnp.linspace(0.1, 2.0, 32, dtype=jnp.float16)
    np.testing.assert_allclose(np.asarray(jnp.log(x) + jnp.sqrt(x)).astype(np.float32), np.log(np.asarray(x).astype(np.float32)) + np.sqrt(np.asarray(x).astype(np.float32)), rtol=1e-2)
@case("grad of mlp")
def _():
    def loss(w, x): return jnp.sum(jnp.tanh(x @ w) ** 2)
    w = jnp.ones((8, 4), f32) * 0.1; x = jnp.ones((16, 8), f32)
    g = jax.grad(loss)(w, x); assert np.isfinite(np.asarray(g)).all()
@case("conv2d (expected unsupported)")
def _():
    x = jnp.ones((1, 8, 8, 3), f32); w = jnp.ones((3, 3, 3, 4), f32)
    y = jax.lax.conv_general_dilated(x, w, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC"))
    assert y.shape == (1, 8, 8, 4)
@case("iota + where + select")
def _():
    x = jnp.arange(12, dtype=f32)
    np.testing.assert_array_equal(np.asarray(jnp.where(x % 2 == 0, x, -x)), [i if i % 2 == 0 else -i for i in range(12)])
@case("large elementwise 4M")
def _():
    x = jnp.ones((2048, 2048), f32)
    assert float(jnp.sum(x * 2 + 1)) == 3 * 2048 * 2048

def main():
    print("backend:", jax.default_backend())
    ok = 0
    for name, fn in CASES.items():
        if name.startswith("SKIP"):
            print(f"SKIP {name}", flush=True); continue
        print(f"RUN  {name}", flush=True)
        # A GPU-side hang would not return to Python; dump stacks and exit.
        faulthandler.dump_traceback_later(180, exit=True)
        try:
            fn(); ok += 1; print(f"PASS {name}", flush=True)
        except Exception as e:  # noqa
            msg = str(e).strip().splitlines()[0][:160] if str(e).strip() else type(e).__name__
            print(f"FAIL {name}: {type(e).__name__}: {msg}", flush=True)
        finally:
            faulthandler.cancel_dump_traceback_later()
    print(f"{ok}/{len(CASES)} passed")
    return 0 if ok == len(CASES) else 1

if __name__ == "__main__":
    sys.exit(main())
