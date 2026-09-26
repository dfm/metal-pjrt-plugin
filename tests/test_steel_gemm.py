"""f16 / bf16 GEMMs through JAX (steel kernels) against a float64 CPU
reference, normwise in ulps of the output dtype (tests/metal_testing.py)."""
import numpy as np
import jax, jax.numpy as jnp
import pytest

from metal_testing import check

pytestmark = pytest.mark.metal

CASES = {}  # name -> (kind, fn, args)
rng = np.random.default_rng(0)
for dt in (jnp.bfloat16, jnp.float16):
    d = jnp.dtype(dt).name
    host = lambda *s: rng.standard_normal(s).astype(dt)
    for (m, k, n) in ((64, 64, 64), (100, 37, 50), (1, 300, 7), (257, 129, 65), (512, 1024, 256)):
        a, b, bias = host(m, k), host(k, n), host(n)
        CASES[f"{d} {m}x{k}x{n}"] = ("matmul", lambda x, y: x @ y, (a, b))
        CASES[f"{d} {m}x{k}x{n} A.T B.T"] = ("matmul", lambda x, y: y.T @ x.T, (a, b))
        CASES[f"{d} {m}x{k}x{n} f32 out"] = ("f32 out", lambda x, y: jnp.matmul(x, y, preferred_element_type=jnp.float32), (a, b))
        CASES[f"{d} {m}x{k}x{n} +bias relu"] = ("matmul", lambda x, y, c: jax.nn.relu(x @ y + c), (a, b, bias))
    x, y = host(6, 33, 70), host(6, 70, 45)
    CASES[f"{d} batched"] = ("matmul", lambda x, y: jnp.einsum("bij,bjk->bik", x, y), (x, y))
    CASES[f"{d} batched bcast"] = ("matmul", lambda x, y: jnp.einsum("bij,jk->bik", x, y[0]), (x, y))
    CASES[f"{d} grad"] = ("grad", lambda x, y: jax.grad(lambda u, v: jnp.sum((u @ v).astype(jnp.float32) ** 2))(x[0], y[0]), (x, y))

# Normwise ulps of the output dtype, measured (METAL_TEST_REPORT_ULPS=1, M3)
# and doubled; per dtype and kind, max over the shapes.
ULPS = {
    ("bfloat16", "matmul"): 1.9, ("bfloat16", "f32 out"): 12, ("bfloat16", "grad"): 1,
    ("float16", "matmul"): 1.9, ("float16", "f32 out"): 24, ("float16", "grad"): 1,
}


@pytest.mark.parametrize("name", list(CASES))
def test_steel_gemm(name):
    kind, fn, args = CASES[name]
    check(fn, *args, ulps=ULPS.get((name.split()[0], kind), 0), normwise=True, name=name)


@pytest.mark.xfail(strict=True, reason=(
    "bug: a 16-bit -> f32 dot small enough to stay in a loop fusion rounds "
    "each product to the input type (XLA's elemental EmitMulAdd, "
    "elemental_hlo_to_mlir.cc:494-501); JAX's LaxTest::testDotPreferredElement2"))
@pytest.mark.parametrize("dt", ["bfloat16", "float16"])
def test_small_mixed_precision_dot(dt):
    a = rng.standard_normal((4, 3)).astype(dt)
    b = rng.standard_normal((3, 6)).astype(dt)
    check(lambda x, y: jnp.matmul(x, y, preferred_element_type=jnp.float32),
          a, b, ulps=24, normwise=True, name=f"{dt} 4x3x6 f32 out")
