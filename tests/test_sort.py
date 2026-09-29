"""Sorts on metal against CPU, bit for bit. Sorts of more than 16384
elements with a simple comparator go through XLA's SortRewriter to the MSL
radix sort (metal_pjrt/ffi/cub_sort_ffi.cc; exhaustive type sweep of the
kernels in //metal_pjrt/ffi:radix_sort_test, every key type JAX can make
here through the handler below); rows of <= 64 and smaller sorts stay
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


def key_bits(dtype, n, seed=3):
    # Random bits: NaNs (with payloads), infinities, +-0 and denormals come
    # up among the float keys, the extremes among the integer ones.
    bits = np.random.default_rng(seed).integers(
        0, 256, n * jnp.dtype(dtype).itemsize, dtype=np.uint8)
    return bits.view(jnp.dtype(dtype))


# The handler maps each XLA key type to the kernels' key order: every key
# type JAX makes with x64 off, keys only and (integer keys SortRewriter
# pairs) with 16-bit values.
@pytest.mark.parametrize("dtype", ["int8", "uint8", "int16", "uint16",
                                   "float16", "bfloat16", "int32", "uint32",
                                   "float32"])
def test_radix_sort_key_types(dtype):
    x = key_bits(dtype, 20000)
    assert any(t.startswith("xla.gpu.ext.cub_sort_") for t in targets(jnp.sort, x))
    same_bits(run_on(metal(), jnp.sort, x), run_on(cpu(), jnp.sort, x))
    if dtype in ("uint8", "uint16", "int32", "uint32"):
        v = key_bits("float16", 20000, seed=4)
        fn = lambda k, v: lax.sort((k, v), num_keys=1, is_stable=True)
        assert any(t == "xla.gpu.ext.cub_sort_pairs" for t in targets(fn, x, v))
        same_bits(run_on(metal(), fn, x, v), run_on(cpu(), fn, x, v))


def test_tiny_rows_stay_bitonic():
    # 1000 rows of 32 and (rows, 4) int32 key/value sorts: more than
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


def test_top_k_short_rows_stay_bitonic():
    # XLA turns top_k into a sort after our pre-hook; short rows are
    # decomposed early so they stay bitonic (10-23x faster than radix here).
    for shape, k in (((4096, 8), 2), ((12500, 4), 1)):
        x = floats(shape[0] * shape[1]).reshape(shape)
        fn = lambda a: lax.top_k(a, k)
        assert not any("cub_sort" in t for t in targets(fn, x))
        same_bits(run_on(metal(), fn, x), run_on(cpu(), fn, x))
    x = floats(64 * 4096).reshape(64, 4096)  # long rows: still radix
    assert any("cub_sort" in t for t in targets(lambda a: lax.top_k(a, 8), x))


DISABLE_CUBSORT_CHILD = r"""
import numpy as np, jax, jax.numpy as jnp
x = np.random.default_rng(0).standard_normal(20000).astype(np.float32)
text = jax.jit(jnp.argsort).lower(x).compile().as_text()
print("cub_sort" in text, bool((np.asarray(jax.jit(jnp.argsort)(x)) == np.argsort(x, kind="stable")).all()))
"""


@pytest.mark.parametrize("value", [None, "cubsort"])
def test_disable_cubsort(value):
    # METAL_PJRT_DISABLE_REWRITES=cubsort (read once per process):
    # MetalSortExpander takes every sort.
    import os
    from metal_testing import run_python
    env = {k: v for k, v in os.environ.items()
           if k != "METAL_PJRT_DISABLE_REWRITES"}
    if value is not None:
        env["METAL_PJRT_DISABLE_REWRITES"] = value
    out = run_python(DISABLE_CUBSORT_CHILD, dict(env, JAX_PLATFORMS="mtl"))
    assert out.returncode == 0, out.stderr[-3000:]
    assert out.stdout.split() == [str(value is None), "True"]


def _run_hlo_text(text, backend_name, *args):
    # JAX's public API only builds strict (LT/GT) comparators; compile the
    # edited StableHLO directly (private API, as jax.jit does underneath).
    from jax._src import compiler, xla_bridge
    from jax._src.lib import xla_client as xc
    backend = xla_bridge.get_backend(backend_name)
    dev = backend.devices()[0]
    exe = backend.compile_and_load(
        text, xc.DeviceList((dev,)),
        compiler.get_compile_options(num_replicas=1, num_partitions=1))
    out = exe.execute_sharded([jax.device_put(a, dev) for a in args])
    return [np.asarray(o[0]) for o in out.disassemble_into_single_device_arrays()]


def test_non_strict_comparator_keeps_a_permutation():
    # A LE comparator says a <= b and b <= a for equal keys; the bitonic
    # network used to duplicate and drop elements then. Keys match CPU; the
    # values' order among equal keys is unspecified, but they must be the
    # same multiset per row.
    k = np.random.default_rng(0).integers(0, 4, (5, 19)).astype(np.int32)
    v = np.arange(95, dtype=np.int32).reshape(5, 19)
    sort = lambda k, v: lax.sort((k, v), dimension=1, num_keys=1, is_stable=False)
    text = jax.jit(sort).lower(k, v).as_text()
    le = re.sub(r"stablehlo\.compare\s+LT,", "stablehlo.compare LE,", text)
    assert le != text
    got_k, got_v = _run_hlo_text(le, "mtl", k, v)
    want_k, want_v = _run_hlo_text(le, "cpu", k, v)
    np.testing.assert_array_equal(got_k, want_k)
    np.testing.assert_array_equal(np.sort(got_v, axis=1), np.sort(want_v, axis=1))
