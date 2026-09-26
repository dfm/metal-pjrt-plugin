"""GEMM epilogue fusion (bias / ReLU / GELU, with and without aux output) on
metal against a float64 CPU reference, and a check that XLA really fused
them into the `__cublas$lt$matmul` custom call (the `"epilogue":"..."` in
the optimized HLO's backend config).
"""
import re

import numpy as np
import jax, jax.numpy as jnp
import pytest

from metal_testing import assert_close, f64_reference, metal

pytestmark = pytest.mark.metal


def gelu(x):  # the tanh approximation XLA's GemmRewriter matches
    return jax.nn.gelu(x, approximate=True)

CASES = [
    # name, fn(x, w, b), expected epilogue in the optimized HLO
    ("bias", lambda x, w, b: x @ w + b, "BIAS"),
    ("relu", lambda x, w, b: jax.nn.relu(x @ w), "RELU"),
    ("bias+relu", lambda x, w, b: jax.nn.relu(x @ w + b), "BIAS_RELU"),
    ("gelu", lambda x, w, b: gelu(x @ w), "GELU"),
    ("bias+gelu", lambda x, w, b: gelu(x @ w + b), "BIAS_GELU"),
    # The pre-activation value is also returned: GELU_AUX / BIAS_GELU_AUX.
    ("gelu+aux", lambda x, w, b: (gelu(x @ w), (x @ w) * 3.0), "GELU_AUX"),
    ("bias+gelu+aux",
     lambda x, w, b: (gelu(x @ w + b), (x @ w + b) * 3.0), "BIAS_GELU_AUX"),
    # Transposed result (n x m): the bias runs along m.
    ("transposed bias+gelu",
     lambda x, w, b: gelu(jnp.einsum("mk,kn->nm", x, w) + b[: x.shape[0]]),
     "BIAS_GELU"),
    # Batched (3-D) matmul with vector bias.
    ("batched bias+gelu",
     lambda x, w, b: gelu(jnp.einsum("bik,kj->bij", x.reshape(2, -1, x.shape[1]), w) + b),
     "BIAS_GELU"),
]
DTYPES = {"float32": jnp.float32, "float16": jnp.float16, "bfloat16": jnp.bfloat16}

# Normwise ulps of the output dtype, measured (METAL_TEST_REPORT_ULPS=1, M3)
# and doubled; per dtype, max over the cases.
ULPS = {"float32": 7.3, "float16": 2.5, "bfloat16": 2.4}


@pytest.mark.parametrize("dtype", list(DTYPES))
@pytest.mark.parametrize("i", range(len(CASES)), ids=[c[0] for c in CASES])
def test_epilogue(i, dtype):
    name, fn, want = CASES[i]
    rng = np.random.default_rng(i)
    m, k, n = 64, 48, 96
    x = rng.standard_normal((m, k)).astype(np.float32)
    w = (rng.standard_normal((k, n)) / np.sqrt(k)).astype(np.float32)
    b = rng.standard_normal((n,)).astype(np.float32)
    args = [np.asarray(a).astype(DTYPES[dtype]) for a in (x, w, b)]
    f = jax.jit(fn)
    on_metal = [jax.device_put(a, metal()) for a in args]
    hlo = f.lower(*on_metal).compile().as_text()
    fused = set(re.findall(r'"epilogue":"([A-Z_]+)"', hlo))
    assert want in fused, f"expected epilogue {want}, found {sorted(fused) or 'none'}"
    got = jax.tree.map(np.asarray, f(*on_metal))
    assert_close(got, f64_reference(fn, *args), ULPS[dtype], normwise=True,
                 name=f"{name} [{dtype}]")
