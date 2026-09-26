"""Checks that jax.ffi.ffi_call reaches handlers registered in the Metal
plugin (metal_pjrt_plugin/ffi). Handlers are registered statically in C++
under platform "METAL"; nothing is registered from Python. Uses the
production metal$softmax handler (the test-only metal$test_scale is covered
by //metal_pjrt_plugin/ffi:ffi_test).

  JAX_PLATFORMS=metal python scripts/ffi_check.py
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np


def softmax_ref(x, log):
    x = x.astype(np.float64)
    z = x - x.max(axis=-1, keepdims=True)
    lse = np.log(np.exp(z).sum(axis=-1, keepdims=True))
    return z - lse if log else np.exp(z - lse)


def main() -> int:
    assert jax.default_backend() == "metal", jax.default_backend()
    x = np.random.default_rng(0).normal(size=(8, 1000)).astype(np.float32)

    def softmax(x, log):
        return jax.ffi.ffi_call(
            "metal$softmax", jax.ShapeDtypeStruct(x.shape, x.dtype))(
                x, log=log, row_length=np.int64(x.shape[-1]))

    for log in (False, True):
        y = jax.jit(softmax, static_argnums=1)(x, log)
        np.testing.assert_allclose(np.asarray(y), softmax_ref(x, log),
                                   rtol=1e-5, atol=1e-6)
    # Composes with emitted kernels in the same executable.
    z = jax.jit(lambda x: softmax(x * 2.0, False) * 3.0)(x)
    np.testing.assert_allclose(np.asarray(z), softmax_ref(x * 2.0, False) * 3.0,
                               rtol=1e-5, atol=1e-6)
    print("ffi_check: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
