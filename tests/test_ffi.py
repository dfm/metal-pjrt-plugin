"""jax.ffi.ffi_call reaches handlers registered in the Metal plugin
(metal_pjrt_plugin/ffi). Handlers are registered statically in C++ under
platform "METAL"; nothing is registered from Python. Uses the production
metal$scan handler. The reference is float64 numpy.
"""
import jax
import numpy as np
import pytest

from metal_testing import assert_close

pytestmark = pytest.mark.metal


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


def test_composes_with_emitted_kernels():
    z = jax.jit(lambda x: scan(x * 2.0, "add", False) * 3.0)(X)
    assert_close(np.asarray(z), scan_ref(X * 2.0, "add", False) * 3.0,
                 ulps=4.5, normwise=True, name="cumsum*3")
