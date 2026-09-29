"""f16 / bf16 GEMMs through JAX (steel kernels) against a float64 CPU
reference, normwise in ulps of the output dtype (tests/metal_testing.py)."""
import re

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
    # Few rows in the x @ W.T layout (W stored [n, k]): MetalBlasLt runs
    # these on the wide gemv (metal_pjrt/blas/gemv.h, 2..8 rows, either side,
    # K >= 512) or steel's 16-row tile (9..48 rows, or K < 512).
    for (m, k, n) in ((2, 256, 300), (5, 1024, 96), (8, 516, 1030), (13, 512, 200), (40, 128, 70)):
        x, w, bias = host(m, k), host(n, k), host(n)
        CASES[f"{d} few rows {m}x{k}x{n}"] = ("matmul", lambda x, w: x @ w.T, (x, w))
        CASES[f"{d} few cols {n}x{k}x{m}"] = ("matmul", lambda x, w: w @ x.T, (x, w))
        CASES[f"{d} few rows {m}x{k}x{n} f32 out"] = ("f32 out", lambda x, w: jnp.matmul(x, w.T, preferred_element_type=jnp.float32), (x, w))
        CASES[f"{d} few rows {m}x{k}x{n} +bias gelu"] = ("matmul", lambda x, w, c: jax.nn.gelu(x @ w.T + c), (x, w, bias))
    # K % 4 != 0: the gemv refuses it, so steel's 16-row tile runs it.
    x, w = host(4, 1022), host(300, 1022)
    CASES[f"{d} few rows 4x1022x300"] = ("matmul", lambda x, w: x @ w.T, (x, w))
    # A transposed (x stored [k, m]): never the gemv; steel's 16-row tile.
    a, y = host(200, 24), host(200, 90)
    CASES[f"{d} few rows A.T 24x200x90"] = ("matmul", lambda a, y: a.T @ y, (a, y))
    x, w = host(3, 4, 512), host(3, 96, 512)
    CASES[f"{d} few rows batched"] = ("matmul", lambda x, w: jnp.einsum("bmk,bnk->bmn", x, w), (x, w))
    CASES[f"{d} few rows batched bcast"] = ("matmul", lambda x, w: jnp.einsum("bmk,nk->bmn", x, w[0]), (x, w))
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


# Small enough that GemmRewriter leaves it a kDot for the loop emitter, which
# multiplies in the operand type; MetalDotOperandUpcaster upcasts the operands
# (JAX's LaxTest::testDotPreferredElement2).
@pytest.mark.parametrize("dt", ["bfloat16", "float16"])
def test_small_mixed_precision_dot(dt):
    a = rng.standard_normal((4, 3)).astype(dt)
    b = rng.standard_normal((3, 6)).astype(dt)
    check(lambda x, y: jnp.matmul(x, y, preferred_element_type=jnp.float32),
          a, b, ulps=24, normwise=True, name=f"{dt} 4x3x6 f32 out")


def _gemm_operand_types(hlo):
    """Element types of the operands of each __cublas$lt$matmul."""
    types = dict(re.findall(r"(%[\w.\-]+) = (\w+)\[", hlo))
    return [[types[o] for o in re.findall(r"%[\w.\-]+", ops)]
            for ops in re.findall(
                r"custom-call\(([^)]*)\), custom_call_target=\"__cublas\$lt\$matmul\"", hlo)]


@pytest.mark.parametrize("dt,hlo_dt", [("bfloat16", "bf16"), ("float16", "f16")])
def test_mixed_precision_dot_routing(dt, hlo_dt):
    """The upcast only touches dots left after GemmRewriter: a large 16-bit
    dot with f32 output still reaches the GEMM with 16-bit operands."""
    f = jax.jit(lambda x, y: jnp.matmul(x, y, preferred_element_type=jnp.float32))
    big = jnp.ones((256, 256), dt)
    assert _gemm_operand_types(f.lower(big, big).compile().as_text()) == [[hlo_dt, hlo_dt]]
    small = jnp.ones((4, 3), dt), jnp.ones((3, 6), dt)
    hlo = f.lower(*small).compile().as_text()
    assert "__cublas" not in hlo
    assert re.search(r"f32\[4,3\]\S* convert\(", hlo), hlo


def test_no_dot_merger():
    """Dots sharing an operand stay separate GEMMs (xla_gpu_dot_merger_
    threshold_mb = 0 in ApplyMetalDefaults): XLA's DotMerger would copy the
    weights into one concatenated operand on every call."""
    x = jnp.ones((4, 256), jnp.bfloat16)
    ws = [jnp.ones((512, 256), jnp.bfloat16) for _ in range(3)]
    hlo = jax.jit(lambda x, ws: [x @ w.T for w in ws]).lower(x, ws).compile().as_text()
    assert len(_gemm_operand_types(hlo)) == 3, hlo
    assert "concatenate" not in hlo, hlo


def test_unsupported_gemm_fails_at_compile_time():
    """CheckPostGemmRewriter: an int8 GEMM (no Metal kernel) is refused when
    compiling, naming the JAX op and the source line."""
    x = jnp.ones((64, 48), jnp.int8)
    f = jax.jit(lambda a, b: jax.lax.dot(a, b, preferred_element_type=jnp.int32))
    with pytest.raises(Exception, match=r"matmul s8 x s8 -> s32 is not supported.*dot_general.* at .*test_steel_gemm.py:\d+"):
        f.lower(x, x.T).compile()


@pytest.mark.parametrize("dt", ["bfloat16", "float16"])
def test_gemm_past_steel_limits_fails_at_compile_time(dt):
    """f16/bf16 GEMMs run only on steel, whose index math is 32-bit: a row
    of more than INT32_MAX / 256 elements is refused when compiling (nothing
    is allocated), naming the limit and the op, and the device stays usable."""
    k = (1 << 23) + 16
    f = jax.jit(lambda a, b: jnp.matmul(a, b, preferred_element_type=jnp.float32))
    a = jax.ShapeDtypeStruct((2, k), jnp.dtype(dt))
    b = jax.ShapeDtypeStruct((k, 2), jnp.dtype(dt))
    with pytest.raises(Exception, match=r"index in 32 bits.*leading dimension.* exceeds 8388607\. Split the matmul or use float32 .*dot_general.* at .*test_steel_gemm.py:\d+"):
        f.lower(a, b).compile()
    x = jnp.ones((64, 64), dt)
    np.testing.assert_array_equal(np.asarray(f(x, x)), 64.0)
