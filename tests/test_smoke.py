"""End-to-end smoke test for the Metal PJRT plugin:
  scripts/device_lock.py -- .venv/bin/python -m pytest tests/test_smoke.py
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from metal_testing import check

pytestmark = pytest.mark.metal


def test_backend_is_openmetal():
    assert jax.default_backend() == "openmetal", jax.default_backend()


def test_transfer():
    x = jnp.arange(16, dtype=jnp.float32)
    np.testing.assert_array_equal(np.asarray(x), np.arange(16, dtype=np.float32))


def test_elementwise_fusion():
    a = np.linspace(0.0, 1.0, 1024, dtype=np.float32)
    b = np.full((1024,), 2.0, dtype=np.float32)
    check(lambda a, b: jnp.exp(a) * b + 1.0, a, b, ulps=2.7, name="exp*b+1")


def test_reduction_fusion():
    m = jnp.ones((256, 64), dtype=jnp.float32)
    np.testing.assert_array_equal(
        np.asarray(jax.jit(lambda a: jnp.sum(a * a, axis=0))(m)), 256.0)


def test_constants_and_broadcasting():
    h = jax.jit(lambda a: a + jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32))
    np.testing.assert_array_equal(
        np.asarray(h(jnp.zeros((8, 4), dtype=jnp.float32))),
        np.tile([1.0, 2.0, 3.0, 4.0], (8, 1)))
