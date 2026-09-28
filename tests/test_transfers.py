"""Host <-> device transfers without staging
(should_stage_host_to_device_transfers=False): exact round trips, and the
source/destination numpy arrays' lifetimes.

JAX lets the H2D copy run after device_put returns, so the plugin copies
a source below 16 MB into a host temporary before returning, and waits for
the copy of a larger one: a caller refilling its numpy buffer right after
device_put (the usual data-loader pattern) must not change the device value.
"""
import gc

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.metal


@pytest.mark.parametrize("n", [0, 1, 7, 1 << 12, (1 << 22) + 3])
def test_round_trip(n):
    a = np.random.default_rng(n).standard_normal(n).astype(np.float32)
    x = jax.device_put(a)
    np.testing.assert_array_equal(np.asarray(x), a)
    np.testing.assert_array_equal(np.asarray(x * 2), a * 2)


@pytest.mark.parametrize("mb", [1, 32])
@pytest.mark.parametrize("view", ["transposed", "sliced", "size-1 dims"])
def test_round_trip_views(view, mb):
    a = np.arange(mb << 18, dtype=np.float32).reshape(512, -1)
    v = {"transposed": a.T, "sliced": a[:, ::2],
         "size-1 dims": a.reshape(512, 1, -1, 1)}[view]
    np.testing.assert_array_equal(np.asarray(jax.device_put(v)), v)


@pytest.mark.parametrize("mb", [1, 32])  # below / above the 16 MB snapshot limit
@pytest.mark.parametrize("busy", [False, True], ids=["idle", "busy"])
@pytest.mark.parametrize("may_alias", [None, False])
def test_source_mutated_right_after_device_put(busy, may_alias, mb):
    slow = jax.jit(lambda a: jax.lax.fori_loop(0, 16, lambda i, a: (a @ a) * 1e-3, a))
    a = jnp.ones((2048, 2048))
    slow(a).block_until_ready()
    buf = np.empty(mb << 18, np.float32)
    xs = []
    for i in range(6):
        pending = slow(a) if busy else None  # ~100 ms of GPU work queued first
        buf[:] = i
        xs.append(jax.device_put(buf, may_alias=may_alias))
        buf[-1024:] = -1  # the tail first: a copy still running reads it last
        buf[:] = -1
        del pending
    for i, x in enumerate(xs):
        assert np.all(np.asarray(x) == i), (i, np.unique(np.asarray(x))[:4])


def test_source_mutated_after_transfer():
    a = np.arange(1 << 20, dtype=np.float32)
    x = jax.device_put(a).block_until_ready()
    a[:] = -1
    np.testing.assert_array_equal(np.asarray(x), np.arange(1 << 20, dtype=np.float32))


def test_source_freed_before_transfer():
    # The temporary is dropped by Python right away; later host allocations
    # would reuse its memory if the transfer did not keep it alive.
    xs = [jax.device_put(np.full(1 << 20, float(i), np.float32)) for i in range(8)]
    junk = [np.full(1 << 20, -1.0, np.float32) for _ in range(8)]
    for i, x in enumerate(xs):
        assert np.all(np.asarray(x) == i), i
    del junk


def test_destination_freed_early():
    f = jax.jit(lambda x: x + 1)
    x = jnp.zeros(1 << 20)
    for i in range(8):
        y = f(x)
        y.copy_to_host_async()
        h = np.asarray(y)
        del y
        gc.collect()
        assert np.all(h == 1)
        del h
        x = f(x) - 1
