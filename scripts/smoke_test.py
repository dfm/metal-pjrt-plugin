"""End-to-end smoke test for the Metal PJRT plugin.

Usage (after building the dylib and installing this package):
  JAX_PLATFORMS=metal python scripts/smoke_test.py
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np


def main() -> int:
    print("backend:", jax.default_backend(), jax.devices())
    assert jax.default_backend() == "metal", jax.default_backend()

    # 1. Transfers only.
    x = jnp.arange(16, dtype=jnp.float32)
    np.testing.assert_array_equal(np.asarray(x), np.arange(16, dtype=np.float32))
    print("transfer ok")

    # 2. A loop fusion: elementwise chain.
    @jax.jit
    def f(a, b):
        return jnp.exp(a) * b + 1.0

    a = jnp.linspace(0.0, 1.0, 1024, dtype=jnp.float32)
    b = jnp.full((1024,), 2.0, dtype=jnp.float32)
    got = np.asarray(f(a, b))
    want = np.exp(np.asarray(a)) * 2.0 + 1.0
    np.testing.assert_allclose(got, want, rtol=1e-5)
    print("elementwise fusion ok")

    # 3. A reduction fusion.
    @jax.jit
    def g(a):
        return jnp.sum(a * a, axis=0)

    m = jnp.ones((256, 64), dtype=jnp.float32)
    np.testing.assert_allclose(np.asarray(g(m)), np.full((64,), 256.0), rtol=1e-5)
    print("reduction fusion ok")

    # 4. Constants and broadcasting.
    @jax.jit
    def h(a):
        return a + jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)

    np.testing.assert_allclose(
        np.asarray(h(jnp.zeros((8, 4), dtype=jnp.float32))),
        np.tile([1.0, 2.0, 3.0, 4.0], (8, 1)))
    print("constants ok")
    print("ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
