"""jax.ffi.ffi_call reaches handlers registered in the Metal plugin
(metal_pjrt_plugin/ffi). Handlers are registered statically in C++ under
platform "METAL"; nothing is registered from Python. Uses the production
metal$softmax handler (the test-only metal$test_scale is covered by
//metal_pjrt_plugin/ffi:ffi_test). The reference is float64 numpy.
"""
import jax
import numpy as np
import pytest

from metal_testing import assert_close

pytestmark = pytest.mark.metal


def softmax_ref(x, log):
    x = x.astype(np.float64)
    z = x - x.max(axis=-1, keepdims=True)
    lse = np.log(np.exp(z).sum(axis=-1, keepdims=True))
    return z - lse if log else np.exp(z - lse)


def softmax(x, log):
    return jax.ffi.ffi_call(
        "metal$softmax", jax.ShapeDtypeStruct(x.shape, x.dtype))(
            x, log=log, row_length=np.int64(x.shape[-1]))


X = np.random.default_rng(0).normal(size=(8, 1000)).astype(np.float32)


@pytest.mark.parametrize("log", [False, True])
def test_softmax_ffi_call(log):
    y = jax.jit(softmax, static_argnums=1)(X, log)
    # exp amplifies the rounding of x - max (CPU float32: 5.6 ulps);
    # log_softmax outputs near 0 come from cancellation: normwise.
    assert_close(np.asarray(y), softmax_ref(X, log), ulps=2.2 if log else 13,
                 normwise=log, name=f"softmax log={log}")


def test_composes_with_emitted_kernels():
    z = jax.jit(lambda x: softmax(x * 2.0, False) * 3.0)(X)
    assert_close(np.asarray(z), softmax_ref(X * 2.0, False) * 3.0, ulps=21,
                 name="softmax*3")
