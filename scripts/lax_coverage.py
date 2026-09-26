"""Exercise (nearly) every jax.lax primitive on the current backend, one at a
time, and report pass/fail with the error class. Used for the coverage audit.

  JAX_PLATFORMS=metal python scripts/lax_coverage.py
"""
import faulthandler, sys
import numpy as np
import jax, jax.numpy as jnp
from jax import lax

f32 = jnp.float32
CASES = []
def case(name):
    def deco(fn):
        CASES.append((name, fn)); return fn
    return deco

def ref(fn, *args):
    """Run fn on CPU too and compare."""
    with jax.default_device(jax.devices("cpu")[0]):
        expect = np.asarray(jax.jit(fn)(*[jax.device_put(a, jax.devices("cpu")[0]) for a in args]))
    got = np.asarray(jax.jit(fn)(*args))
    np.testing.assert_allclose(got, expect, rtol=2e-3, atol=2e-3)

A = lambda *s: jnp.linspace(-2, 2, int(np.prod(s)), dtype=f32).reshape(*s)

# ---- elementwise unary ----
for op in ["abs","neg","sign","floor","ceil","round","exp","exp2","expm1","log","log1p","tanh","logistic",
           "sin","cos","tan","asin","acos","atan","sinh","cosh","asinh","acosh","atanh","sqrt","rsqrt","cbrt",
           "erf","erfc","erf_inv","lgamma","digamma","is_finite","square","reciprocal"]:
    def mk(op):
        def fn():
            x = A(64) if op not in ("acosh",) else A(64) + 3
            if op in ("log","log1p","sqrt","rsqrt","lgamma","digamma"): x = jnp.abs(x) + 0.1
            if op in ("asin","acos","atanh","erf_inv"): x = x / 3
            ref(getattr(lax, op), x)
        return fn
    case(f"lax.{op}")(mk(op))
for op in ["bessel_i0e","bessel_i1e","igamma","igammac","polygamma","zeta","random_gamma_grad","betainc"]:
    def mk(op):
        def fn():
            x = jnp.abs(A(32)) + 0.5; y = jnp.abs(A(32)) + 1.5
            if op in ("bessel_i0e","bessel_i1e"): ref(getattr(lax, op), x)
            elif op == "polygamma": ref(getattr(lax, op), jnp.ones(32, f32), y)
            elif op == "betainc": ref(getattr(lax, op), x, y, jnp.full(32, 0.3, f32))
            else: ref(getattr(lax, op), x, y)
        return fn
    case(f"lax.{op}")(mk(op))
# ---- binary ----
for op in ["add","sub","mul","div","rem","pow","max","min","atan2","nextafter"]:
    def mk(op):
        def fn():
            x = A(64); y = A(64)[::-1] + 0.3
            if op == "pow": x = jnp.abs(x) + 0.1
            ref(getattr(lax, op), x, y)
        return fn
    case(f"lax.{op} f32")(mk(op))
for op in ["add","mul","div","rem","max","shift_left","shift_right_arithmetic","shift_right_logical",
           "bitwise_and","bitwise_or","bitwise_xor","population_count","clz"]:
    def mk(op):
        def fn():
            x = jnp.arange(-20, 44, dtype=jnp.int32); y = (jnp.arange(64, dtype=jnp.int32) % 5) + 1
            if op in ("population_count","clz"): ref(getattr(lax, op), x)
            elif op.startswith("shift"): ref(getattr(lax, op), x, y % 8)
            else: ref(getattr(lax, op), x, y)
        return fn
    case(f"lax.{op} int32")(mk(op))
case("lax.integer_pow")(lambda: ref(lambda x: lax.integer_pow(x, 3), A(64)))
case("lax.eq/lt/select")(lambda: ref(lambda x, y: lax.select(lax.lt(x, y), x, lax.mul(y, 2.0)), A(64), A(64)[::-1]))
case("lax.clamp")(lambda: ref(lambda x: lax.clamp(-1.0, x, 1.0), A(64)))
# ---- dtype conversions ----
for src, dst in [("f32","i32"),("f32","bf16"),("f32","f16"),("i32","f32"),("f32","u8"),("i32","bool"),("f32","i64"),("i64","f32")]:
    def mk(src, dst):
        def fn():
            d = {"f32": f32, "i32": jnp.int32, "bf16": jnp.bfloat16, "f16": jnp.float16, "u8": jnp.uint8, "bool": jnp.bool_, "i64": jnp.int64}
            x = (A(64) * 50).astype(d[src]) if src != "i64" else jnp.arange(64, dtype=jnp.int32)
            ref(lambda v: lax.convert_element_type(v, d[dst]).astype(f32), x)
        return fn
    case(f"convert {src}->{dst}")(mk(src, dst))
case("lax.bitcast_convert_type")(lambda: ref(lambda x: lax.bitcast_convert_type(x, jnp.int32), A(64)))
case("lax.reduce_precision")(lambda: ref(lambda x: lax.reduce_precision(x, 5, 10), A(64)))
# ---- shape ops ----
case("lax.broadcast_in_dim")(lambda: ref(lambda x: lax.broadcast_in_dim(x, (4, 8, 3), (1,)), A(8)))
case("lax.reshape")(lambda: ref(lambda x: lax.reshape(x, (4, 16)), A(64)))
case("lax.transpose 3d")(lambda: ref(lambda x: lax.transpose(x, (2, 0, 1)), A(4, 5, 6)))
case("lax.rev")(lambda: ref(lambda x: lax.rev(x, (0, 1)), A(4, 6)))
case("lax.pad")(lambda: ref(lambda x: lax.pad(x, 0.5, [(1, 2, 1), (0, 1, 0)]), A(4, 6)))
case("lax.concatenate")(lambda: ref(lambda x, y: lax.concatenate([x, y], 1), A(4, 3), A(4, 5)))
case("lax.slice")(lambda: ref(lambda x: lax.slice(x, (1, 2), (4, 6), (1, 2)), A(5, 8)))
case("lax.dynamic_slice")(lambda: ref(lambda x: lax.dynamic_slice(x, (jnp.int32(1), jnp.int32(2)), (3, 4)), A(5, 8)))
case("lax.dynamic_update_slice")(lambda: ref(lambda x: lax.dynamic_update_slice(x, jnp.ones((2, 2), f32), (jnp.int32(1), jnp.int32(2))), A(5, 8)))
case("lax.squeeze/expand_dims")(lambda: ref(lambda x: lax.expand_dims(lax.squeeze(x, (1,)), (0,)), A(4, 1, 6)))
case("lax.iota")(lambda: ref(lambda: lax.iota(jnp.int32, 100).astype(f32)))
case("lax.full_like")(lambda: ref(lambda x: lax.full_like(x, 3.0), A(64)))
case("lax.gather")(lambda: ref(lambda x, i: x[i], A(64), jnp.array([3, 5, 7, 63])))
case("lax.gather 2d")(lambda: ref(lambda x, i: x[i, :], A(16, 8), jnp.array([3, 5])))
case("lax.scatter")(lambda: ref(lambda x: x.at[jnp.array([1, 3])].set(9.0), A(64)))
case("lax.scatter_add")(lambda: ref(lambda x: x.at[jnp.array([1, 1, 3])].add(1.0), A(64)))
case("lax.scatter_mul/min/max")(lambda: ref(lambda x: x.at[jnp.array([2])].mul(2.0).at[jnp.array([3])].min(-9.0).at[jnp.array([4])].max(9.0), A(64)))
# ---- reductions ----
for name, fn in [("reduce_sum", lambda x: jnp.sum(x, 1)), ("reduce_max", lambda x: jnp.max(x, 0)), ("reduce_min", lambda x: jnp.min(x)),
                 ("reduce_prod", lambda x: jnp.prod(x / 2, 1)), ("reduce_and", lambda x: jnp.all(x > 0, 1)), ("argmin", lambda x: jnp.argmin(x, 1)),
                 ("reduce_or", lambda x: jnp.any(x > 0, 0)), ("mean/var", lambda x: jnp.var(x, 1)), ("logsumexp", lambda x: jax.nn.logsumexp(x, 1)),
                 ("reduce big 1M", lambda x: jnp.sum(x)), ("reduce column 2048x64", lambda x: jnp.sum(x, 0))]:
    def mk(name, fn):
        def run():
            x = A(1024, 1024) if "1M" in name else (A(2048, 64) if "2048" in name else A(32, 64))
            ref(fn, x)
        return run
    case(f"lax.{name}")(mk(name, fn))
case("lax.cumsum/cumprod/cummax")(lambda: ref(lambda x: lax.cumsum(x, 1) + lax.cummax(x, 0) + lax.cumprod(x / 4 + 1, 1), A(16, 32)))
case("lax.reduce_window (maxpool)")(lambda: ref(lambda x: lax.reduce_window(x, -jnp.inf, lax.max, (2, 2), (2, 2), "VALID"), A(8, 8)))
case("lax.reduce_window sum")(lambda: ref(lambda x: lax.reduce_window(x, 0.0, lax.add, (3,), (1,), "SAME"), A(64)))
case("select_and_scatter (maxpool grad)")(lambda: ref(lambda x: jax.grad(lambda v: jnp.sum(lax.reduce_window(v, -jnp.inf, lax.max, (2, 2), (2, 2), "VALID")))(x), A(8, 8)))
# ---- linear algebra / library ----
case("lax.dot_general f32")(lambda: ref(lambda x, y: lax.dot_general(x, y, (((1,), (0,)), ((), ()))), A(32, 48), A(48, 16)))
case("dot int32")(lambda: ref(lambda x, y: (x @ y).astype(f32), jnp.arange(64, dtype=jnp.int32).reshape(8, 8), jnp.ones((8, 8), jnp.int32)))
case("dot f16 -> f32 out")(lambda: ref(lambda x, y: jnp.matmul(x.astype(jnp.float16), y.astype(jnp.float16), preferred_element_type=f32), A(32, 32), A(32, 32)))
case("dot batched bf16")(lambda: ref(lambda x, y: jnp.einsum("bij,bjk->bik", x.astype(jnp.bfloat16), y.astype(jnp.bfloat16)).astype(f32), A(4, 16, 16), A(4, 16, 16)))
case("matvec / outer")(lambda: ref(lambda x, y: jnp.outer(x @ y, y), A(16, 16), A(16)))
case("conv 2d NHWC")(lambda: ref(lambda x, w: lax.conv_general_dilated(x, w, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")), A(2, 8, 8, 3), A(3, 3, 3, 4)))
case("conv 2d strided NCHW")(lambda: ref(lambda x, w: lax.conv(x, w, (2, 2), "VALID"), A(1, 3, 9, 9), A(4, 3, 3, 3)))
case("conv grad")(lambda: ref(lambda x, w: jax.grad(lambda a, b: jnp.sum(lax.conv(a, b, (1, 1), "SAME") ** 2), argnums=(0, 1))(x, w)[1], A(1, 2, 6, 6), A(3, 2, 3, 3)))
case("cholesky")(lambda: ref(lambda x: jnp.linalg.cholesky(x @ x.T + 4 * jnp.eye(8)), A(8, 8)))
case("triangular_solve")(lambda: ref(lambda x, b: lax.linalg.triangular_solve(jnp.tril(x) + 5 * jnp.eye(8), b, left_side=True, lower=True), A(8, 8), A(8, 3)))
case("lu / solve")(lambda: ref(lambda x, b: jnp.linalg.solve(x + 8 * jnp.eye(8), b), A(8, 8), A(8, 2)))
case("qr")(lambda: ref(lambda x: jnp.abs(jnp.linalg.qr(x)[1]), A(8, 6)))
case("eigh")(lambda: ref(lambda x: jnp.linalg.eigh(x @ x.T)[0], A(6, 6)))
case("svd")(lambda: ref(lambda x: jnp.linalg.svd(x, compute_uv=False), A(6, 4)))
case("inv / det")(lambda: ref(lambda x: jnp.linalg.det(x + 8 * jnp.eye(6)), A(6, 6)))
case("fft")(lambda: ref(lambda x: jnp.abs(jnp.fft.fft(x)), A(64)))
case("rfft2")(lambda: ref(lambda x: jnp.abs(jnp.fft.rfft2(x)), A(8, 16)))
# ---- sort / search ----
case("lax.sort 1d")(lambda: ref(lambda x: lax.sort(x), A(64)[::-1]))
case("lax.sort keyed 2 operands")(lambda: ref(lambda x: lax.sort((x, jnp.arange(64, dtype=jnp.int32)), num_keys=1)[1].astype(f32), A(64)[::-1]))
case("argsort")(lambda: ref(lambda x: jnp.argsort(x).astype(f32), A(64)[::-1]))
case("top_k")(lambda: ref(lambda x: lax.top_k(x, 5)[0], A(64)))
case("searchsorted")(lambda: ref(lambda x: jnp.searchsorted(jnp.sort(x), jnp.array([0.0, 1.0])).astype(f32), A(64)))
case("jnp.unique via sort")(lambda: ref(lambda x: jnp.sort(jnp.floor(x)), A(64)))
# ---- control flow ----
case("lax.cond")(lambda: ref(lambda x: lax.cond(x[0] > 0, lambda v: v * 2, lambda v: v - 1, x), A(8)))
case("lax.switch")(lambda: ref(lambda x: lax.switch(jnp.int32(2), [lambda v: v, lambda v: v * 2, lambda v: v * 3], x), A(8)))
case("lax.while_loop")(lambda: ref(lambda x: lax.while_loop(lambda c: c[1] < 5, lambda c: (c[0] * 1.5, c[1] + 1), (x, jnp.int32(0)))[0], A(8)))
case("lax.fori_loop")(lambda: ref(lambda x: lax.fori_loop(0, 8, lambda i, v: v + i, x), A(8)))
case("lax.scan 100 steps")(lambda: ref(lambda x: lax.scan(lambda c, v: (c * 0.9 + v, c), x, jnp.arange(100, dtype=f32))[1].sum(0), A(8)))
case("lax.map")(lambda: ref(lambda x: lax.map(lambda v: jnp.sin(v) * 2, x), A(16, 4)))
case("vmap")(lambda: ref(lambda x: jax.vmap(lambda v: jnp.dot(v, v))(x), A(16, 8)))
case("grad through while")(lambda: ref(lambda x: jax.grad(lambda v: jnp.sum(lax.fori_loop(0, 4, lambda i, a: jnp.tanh(a) * 1.1, v)))(x), A(8)))
case("checkpoint/remat")(lambda: ref(lambda x: jax.grad(lambda v: jnp.sum(jax.checkpoint(lambda a: jnp.sin(a) ** 2)(v)))(x), A(64)))
# ---- random ----
case("random uniform")(lambda: (lambda u: np.testing.assert_(0.45 < float(u.mean()) < 0.55))(jax.random.uniform(jax.random.PRNGKey(1), (4096,))))
case("random normal")(lambda: (lambda u: np.testing.assert_(abs(float(u.std()) - 1) < 0.1))(jax.random.normal(jax.random.PRNGKey(2), (4096,))))
case("random bits threefry")(lambda: ref(lambda k: jax.random.bits(k, (16,)).astype(f32), jax.random.PRNGKey(3)))
case("random categorical/choice")(lambda: (lambda c: np.testing.assert_(c.shape == (100,)))(jax.random.choice(jax.random.PRNGKey(4), 10, (100,))))
case("rng_bit_generator")(lambda: (lambda r: np.testing.assert_(r[1].shape == (8,)))(lax.rng_bit_generator(jnp.zeros(4, jnp.uint32), (8,), jnp.uint32)))
# ---- misc ----
case("lax.erf_inv grad / custom_jvp")(lambda: ref(lambda x: jax.grad(lambda v: jnp.sum(jax.nn.gelu(v)))(x), A(64)))
case("jnp.linalg.norm")(lambda: ref(lambda x: jnp.linalg.norm(x, axis=1), A(8, 8)))
case("softmax cross entropy")(lambda: ref(lambda x: -jnp.mean(jax.nn.log_softmax(x)[jnp.arange(8), jnp.arange(8) % 4]), A(8, 4)))
case("layernorm fwd+bwd")(lambda: ref(lambda x: jax.grad(lambda v: jnp.sum(((v - v.mean(-1, keepdims=True)) / jnp.sqrt(v.var(-1, keepdims=True) + 1e-5)) ** 3))(x), A(16, 64)))
case("attention block")(lambda: ref(lambda q: jax.nn.softmax(q @ q.T / 8.0, -1) @ q, A(32, 64)))
case("int64 ops (x64 off -> i32)")(lambda: ref(lambda x: (x.astype(jnp.int64) * 3).astype(f32), jnp.arange(16, dtype=jnp.int32)))
case("f64 (x64 off -> f32)")(lambda: ref(lambda x: x.astype(jnp.float64) * 2, A(16)))
case("complex64 (expected unsupported)")(lambda: ref(lambda x: jnp.abs(x + 1j * x), A(16)))
case("int8 / uint8 math")(lambda: ref(lambda x: (x.astype(jnp.int8) * 2 + x.astype(jnp.uint8)).astype(f32), jnp.arange(16, dtype=jnp.int32)))
case("int4 (expected unsupported)")(lambda: ref(lambda x: x.astype(jnp.int4).astype(f32), jnp.arange(16, dtype=jnp.int32) % 7))
case("float8 e4m3")(lambda: ref(lambda x: x.astype(jnp.float8_e4m3fn).astype(f32), A(16)))
case("bf16 reduce")(lambda: ref(lambda x: jnp.sum(x.astype(jnp.bfloat16), 1).astype(f32), A(16, 64)))
case("host callback io_callback")(lambda: ref(lambda x: jax.experimental.io_callback(lambda v: np.asarray(v) * 2, jax.ShapeDtypeStruct((8,), f32), x), A(8)))
case("pure_callback")(lambda: ref(lambda x: jax.pure_callback(lambda v: np.sin(np.asarray(v)), jax.ShapeDtypeStruct((8,), f32), x), A(8)))
case("debug.print")(lambda: jax.jit(lambda x: (jax.debug.print("dbg {}", x[0]), x * 2)[1])(A(8)).block_until_ready())
case("device_put / copy")(lambda: np.testing.assert_array_equal(np.asarray(jax.device_put(np.arange(5.0, dtype=np.float32)) + 1), np.arange(5.0) + 1))
case("donation")(lambda: np.testing.assert_array_equal(np.asarray(jax.jit(lambda x: x + 1, donate_argnums=0)(jnp.zeros(8, f32))), np.ones(8)))
case("many-arg fusion (>31 buffers)")(lambda: ref(lambda *xs: sum(xs), *[A(16) + i for i in range(40)]))
case("concatenate 40 operands")(lambda: ref(lambda *xs: jnp.concatenate(xs), *[A(4) + i for i in range(40)]))

def main():
    import jax.experimental  # noqa
    print("backend:", jax.default_backend(), flush=True)
    results = []
    for name, fn in CASES:
        # Stack dump on a slow case, never exit (see op_coverage.py).
        faulthandler.dump_traceback_later(240, repeat=True)
        try:
            fn(); results.append((name, "PASS", ""))
        except Exception as e:  # noqa
            first = (str(e).strip().splitlines() or [""])[0][:200]
            results.append((name, "FAIL", f"{type(e).__name__}: {first}"))
        finally:
            faulthandler.cancel_dump_traceback_later()
        print(f"{results[-1][1]} {name} {results[-1][2]}", flush=True)
    n_ok = sum(r[1] == "PASS" for r in results)
    print(f"\n{n_ok}/{len(results)} passed", flush=True)
    return 0 if n_ok == len(results) else 1

if __name__ == "__main__":
    sys.exit(main())
