"""Checks that XLA FFI custom calls reach handlers registered in the Metal
plugin (metal_pjrt_plugin/ffi). Handlers are registered statically in C++
under platform "METAL"; nothing is registered from Python.

  JAX_PLATFORMS=metal python scripts/ffi_check.py
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np


def main() -> int:
    jax.config.update("jax_enable_compilation_cache", False)
    assert jax.default_backend() == "metal", jax.default_backend()
    x = jnp.arange(1000, dtype=jnp.float32)

    def scale(x, s):
        return jax.ffi.ffi_call(
            "metal$test_scale", jax.ShapeDtypeStruct(x.shape, x.dtype))(
                x, scale=np.float32(s))

    y = jax.jit(scale, static_argnums=1)(x, 2.5)
    np.testing.assert_allclose(np.asarray(y), np.arange(1000) * 2.5, rtol=1e-6)
    # Composes with emitted kernels in the same executable.
    z = jax.jit(lambda x: scale(x + 1.0, 3.0) * 2.0)(x)
    np.testing.assert_allclose(np.asarray(z), (np.arange(1000) + 1) * 6.0,
                               rtol=1e-6)
    print("ffi_check: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
