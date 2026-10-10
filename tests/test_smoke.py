# Copyright 2026 The jax-graft Authors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end smoke test for the Metal PJRT plugin:
  scripts/device_lock.py -- .venv/bin/python -m pytest tests/test_smoke.py
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from metal_testing import check, run_python

pytestmark = pytest.mark.metal


def test_backend_is_mtl():
    # Nothing selects it: conftest leaves JAX_PLATFORMS unset.
    assert jax.default_backend() == "mtl", jax.default_backend()


@pytest.mark.parametrize("platforms,want", [(None, "mtl"), ("cpu", "cpu")])
def test_default_backend(platforms, want):
    # Installed: mtl is JAX's default backend, as JAX's other GPU plugins
    # are; JAX_PLATFORMS=cpu keeps CPU.
    import os
    env = {k: v for k, v in os.environ.items() if k != "JAX_PLATFORMS"}
    if platforms is not None:
        env["JAX_PLATFORMS"] = platforms
    code = (
        "import jax, jax.numpy as jnp\n"
        "y = jax.jit(lambda x: x * 2 + 1)(jnp.arange(4.0))\n"
        "assert y.tolist() == [1.0, 3.0, 5.0, 7.0]\n"
        "print(jax.default_backend(), next(iter(y.devices())).platform)\n")
    out = run_python(code, env)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.split() == [want, want], out.stdout


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


def test_grad_of_mlp():
    rng = np.random.default_rng(0)
    def loss(w, x): return jnp.sum(jnp.tanh(x @ w) ** 2)
    w = rng.standard_normal((8, 4)).astype(np.float32) * 0.3
    x = rng.standard_normal((16, 8)).astype(np.float32)
    check(jax.grad(loss), w, x, ulps=1.8, normwise=True, name="grad mlp")


def test_large_elementwise():
    x = jnp.ones((2048, 2048), jnp.float32)
    assert float(jnp.sum(x * 2 + 1)) == 3 * 2048 * 2048
