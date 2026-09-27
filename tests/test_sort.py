"""Sorts on metal against CPU, bit for bit. Sorts of more than 16384
elements with a simple comparator go through XLA's SortRewriter to the MSL
radix sort (metal_pjrt_plugin/ffi/cub_sort_ffi.cc; exhaustive type sweep in
//metal_pjrt_plugin/ffi:cub_sort_test); rows of <= 64 and smaller sorts stay
with MetalSortExpander. The inputs carry duplicates, +-0 and NaNs, so
stability and the NaN/zero order are checked too.
"""
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from metal_testing import cpu, metal, run_on

pytestmark = pytest.mark.metal


def floats(n, seed=0):
    x = np.random.default_rng(seed).standard_normal(n).astype(np.float32)
    x[::7] = x[::7].round()
    x[::97] = 0.0
    x[::101] = -0.0
    x[::103] = np.nan
    x[::107] = -np.inf
    return x


CASES = {
    "sort": (jnp.sort, lambda n: (floats(n),)),
    "argsort": (jnp.argsort, lambda n: (floats(n),)),
    "argsort descending": (lambda x: jnp.argsort(x, descending=True),
                           lambda n: (floats(n),)),
    "sort_key_val": (lax.sort_key_val,
                     lambda n: (np.random.default_rng(1).integers(0, 50, n).astype(np.int32),
                                floats(n))),
    "sort int32": (jnp.sort, lambda n: (np.random.default_rng(2).integers(
        -2**31, 2**31 - 1, n, dtype=np.int64).astype(np.int32),)),
    "sort rows": (lambda x: jnp.sort(x.reshape(16, -1), axis=-1),
                  lambda n: (floats(n),)),
}


def same_bits(got, want):
    for g, w in zip(jax.tree.leaves(got), jax.tree.leaves(want)):
        assert g.dtype == w.dtype and g.shape == w.shape
        np.testing.assert_array_equal(g.view(np.uint8), w.view(np.uint8))


def targets(fn, *args):
    with jax.default_device(metal()):
        text = jax.jit(fn).lower(*args).compile().as_text()
    return set(re.findall(r'custom_call_target="([^"]+)"', text))


@pytest.mark.parametrize("n", [20000, 1 << 20])
@pytest.mark.parametrize("name", list(CASES))
def test_radix_sort_matches_cpu(name, n):
    fn, make = CASES[name]
    args = make(n)
    assert any(t.startswith("xla.gpu.ext.cub_sort_") for t in targets(fn, *args))
    same_bits(run_on(metal(), fn, *args), run_on(cpu(), fn, *args))


def test_tiny_rows_stay_bitonic():
    # 1000 rows of 32 and tinygp's (rows, 4) int32 key/value sorts: more than
    # 16384 elements, but the bitonic network is faster than one threadgroup
    # per row (docs/performance.md).
    x = floats(32000).reshape(1000, 32)
    k = np.random.default_rng(3).integers(0, 5, (12500, 4)).astype(np.int32)
    v = np.arange(50000, dtype=np.int32).reshape(12500, 4)
    sort_rows = lambda a: jnp.sort(a, axis=-1)
    pairs = lambda a, b: lax.sort((a, b), dimension=1, num_keys=1)
    for fn, args in ((sort_rows, (x,)), (pairs, (k, v))):
        assert not any("cub_sort" in t for t in targets(fn, *args))
        same_bits(run_on(metal(), fn, *args), run_on(cpu(), fn, *args))


def test_disable_cubsort(monkeypatch):
    # METAL_PJRT_DISABLE_REWRITES=cubsort: MetalSortExpander takes every sort.
    x = floats(20000)

    def called():
        jax.clear_caches()
        return {t for t in targets(jnp.argsort, x) if "cub_sort" in t}

    monkeypatch.setenv("METAL_PJRT_DISABLE_REWRITES", "cubsort")
    assert called() == set()
    same_bits(run_on(metal(), jnp.argsort, x), run_on(cpu(), jnp.argsort, x))
    monkeypatch.delenv("METAL_PJRT_DISABLE_REWRITES")
    assert called() == {"xla.gpu.ext.cub_sort_pairs"}
