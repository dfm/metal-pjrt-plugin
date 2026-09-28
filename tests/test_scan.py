"""Numerics of the metal$scan FFI kernel against a float64 CPU reference
(integer cases: CPU in the same precision), a check that the rewriter
fired, and metal$scan called directly through jax.ffi.ffi_call (handlers
are registered statically in C++ under platform "METAL"; nothing is
registered from Python). Also softmax / log_softmax in every precision;
those run as XLA's fusions (the metal$softmax rewriter was removed,
docs/performance.md). And lax.associative_scan with a gradient.
"""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from metal_testing import assert_close, check, cpu, f64_reference, metal, run_on

pytestmark = pytest.mark.metal

CASES = {}  # name -> (fn, x, target, expect_rewrite, op)


def add(name, fn, x, target, op, expect_rewrite=True):
    CASES[name] = (fn, x, target, expect_rewrite, op)


def host(a, dtype):
    # Convert on the host: inputs are created with numpy/ml_dtypes so no
    # device-side f32->bf16 convert runs.
    return np.asarray(a).astype(jnp.dtype(dtype))


rng = np.random.default_rng(0)
for dtype, shape in [(jnp.float32, (33, 1000)), (jnp.float16, (33, 1000)),
                     (jnp.bfloat16, (33, 1000)), (jnp.float32, (4, 16385))]:
    x = host(rng.standard_normal(shape) * 4, dtype)
    nm = f"{jnp.dtype(dtype).name} {shape}"
    add(f"softmax {nm}", lambda x: jax.nn.softmax(x, axis=-1), x,
        None, "softmax")
    add(f"log_softmax {nm}", lambda x: jax.nn.log_softmax(x, axis=-1), x,
        None, "log_softmax")
x = rng.standard_normal((16, 256)).astype(np.float32)
x[0, 3] = -np.inf
x[1, :] = -np.inf
x[2, 5] = np.nan
add("softmax with -inf/nan rows", lambda x: jax.nn.softmax(x, -1), x,
    None, "softmax")

for dtype in (jnp.float32, jnp.float16, jnp.bfloat16, jnp.int32):
    for shape in [(4096, 64), (8, 1000), (3, 4097), (5, 3, 130), (100000,),
                  (2, 3)]:
        if shape == (100000,) and dtype in (jnp.float16, jnp.bfloat16):
            shape = (3000,)  # keep the half-precision sums in range
        if dtype == jnp.int32:
            x = host(rng.integers(-3, 4, shape), dtype)
        else:
            x = host(rng.standard_normal(shape) * 0.5 + 1.0, dtype)
        nm = f"{jnp.dtype(dtype).name} {shape}"
        n = shape[-1]
        # Few long rows stay with XLA (MetalScanRewriter::kLongRowLength,
        # kMinRowsForLongRows).
        fire = np.prod(shape) // n >= 32 or n <= 4096
        for rev in (False, True):
            add(f"cumsum rev={rev} {nm}",
                lambda x, rev=rev: jax.lax.cumsum(x, axis=x.ndim - 1, reverse=rev),
                x, "metal$scan", "cumsum", fire)
            add(f"cummax rev={rev} {nm}",
                lambda x, rev=rev: jax.lax.cummax(x, axis=x.ndim - 1, reverse=rev),
                x, "metal$scan", "exact", fire)
            add(f"cummin rev={rev} {nm}",
                lambda x, rev=rev: jax.lax.cummin(x, axis=x.ndim - 1, reverse=rev),
                x, "metal$scan", "exact", fire)
        if shape[-1] <= 1000 and dtype != jnp.int32:
            y = host(1.0 + 0.001 * rng.standard_normal(shape), dtype)
            add(f"cumprod {nm}", lambda x: jax.lax.cumprod(x, axis=x.ndim - 1),
                y, "metal$scan", "cumprod")
add("cumprod int32 (wraps)", lambda x: jnp.cumprod(x, axis=1),
    rng.integers(-3, 4, (16, 300)).astype(np.int32), "metal$scan", "exact")
xn = rng.standard_normal((4, 200)).astype(np.float32)
xn[1, 50] = np.nan
add("cummax with nan", lambda x: jax.lax.cummax(x, axis=1), xn, "metal$scan",
    "exact")

# Ulps of the output dtype, measured (METAL_TEST_REPORT_ULPS=1, M3) and
# doubled; per op and dtype, max over the shapes. Sums and log_softmax are
# normwise (their small outputs come from cancellation). softmax's large
# elementwise error is exp amplifying the rounding of x - max (inputs up to
# |x| ~ 16 here); CPU float32 does the same.
ULPS = {
    ("softmax", "float32"): 62, ("softmax", "float16"): 19,
    ("softmax", "bfloat16"): 1, ("log_softmax", "float32"): 2.1,
    ("log_softmax", "float16"): 2, ("log_softmax", "bfloat16"): 1,
    ("cumsum", "float32"): 4.5, ("cumsum", "float16"): 2.3,
    ("cumsum", "bfloat16"): 1.6, ("cumprod", "float32"): 43,
    ("cumprod", "float16"): 1.1, ("cumprod", "bfloat16"): 1,
}
NORMWISE = {"log_softmax", "cumsum"}


@pytest.mark.parametrize("name", list(CASES))
def test_fused_kernel(name):
    fn, x, target, expect_rewrite, op = CASES[name]
    f = jax.jit(fn)
    if target is not None:
        hlo = f.lower(jax.device_put(x, metal())).compile().as_text()
        assert (target in hlo) == expect_rewrite, f"rewritten={target in hlo}"
    got = np.asarray(f(jax.device_put(x, metal())))
    if np.issubdtype(x.dtype, np.integer):
        want = run_on(cpu(), fn, x)  # x64 would promote and change wrapping
    else:
        want = f64_reference(fn, x)
    dt = jnp.dtype(x.dtype).name
    assert_close(got, want, ULPS.get((op, dt), 0), op in NORMWISE, name=name)


def scan(x, op, reverse):
    return jax.ffi.ffi_call(
        "metal$scan", jax.ShapeDtypeStruct(x.shape, x.dtype))(
            x, op=op, reverse=reverse, row_length=np.int64(x.shape[-1]))


def scan_ref(x, op, reverse):
    x = x.astype(np.float64)
    if reverse:
        x = x[..., ::-1]
    y = {"add": np.cumsum, "mul": np.cumprod, "max": np.maximum.accumulate,
         "min": np.minimum.accumulate}[op](x, axis=-1)
    return y[..., ::-1] if reverse else y


X = np.random.default_rng(0).normal(size=(8, 1000)).astype(np.float32)


@pytest.mark.parametrize("op,reverse", [("add", False), ("add", True),
                                        ("max", False), ("min", True)])
def test_scan_ffi_call(op, reverse):
    y = jax.jit(scan, static_argnums=(1, 2))(X, op, reverse)
    assert_close(np.asarray(y), scan_ref(X, op, reverse),
                 ulps=4.5 if op == "add" else 0, normwise=op == "add",
                 name=f"scan {op} reverse={reverse}")


def test_ffi_call_composes_with_emitted_kernels():
    z = jax.jit(lambda x: scan(x * 2.0, "add", False) * 3.0)(X)
    assert_close(np.asarray(z), scan_ref(X * 2.0, "add", False) * 3.0,
                 ulps=4.5, normwise=True, name="cumsum*3")


def _linear_recurrence_loss(a, b):
    # x_t = a_t x_{t-1} + b_t by lax.associative_scan (the parallel-solver
    # pattern of Kalman-style models), batched over rows.
    def combine(left, right):
        return right[0] * left[0], right[0] * left[1] + right[1]
    _, x = jax.lax.associative_scan(combine, (a, b), axis=-1)
    return jnp.mean(x ** 2)


@pytest.mark.parametrize("n", [1000, 20000])
def test_associative_scan_value_and_grad(n):
    r = np.random.default_rng(n)
    a = r.uniform(0.9, 0.999, (4, n)).astype(np.float32)
    b = r.standard_normal((4, n)).astype(np.float32)
    fn = jax.value_and_grad(_linear_recurrence_loss, argnums=(0, 1))
    # Measured (M3, normwise, value / gradients): n=1000 4.5 ulps (CPU
    # float32 4.1), n=20000 2.4 (CPU 2.5).
    check(fn, a, b, ulps=10, normwise=True, name=f"associative_scan n={n}")
