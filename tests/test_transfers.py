# Copyright 2026 The jax-graft Authors
# SPDX-License-Identifier: Apache-2.0

"""Host <-> device transfers without staging
(should_stage_host_to_device_transfers=False): exact round trips, and the
source/destination numpy arrays' lifetimes.

JAX lets the H2D copy run after device_put returns, so the plugin copies
the source into a host temporary before returning or, from min(256 MB,
max(16 MB, reclaimable memory / 8)) up, waits for the copy: a caller
refilling its numpy buffer right after device_put (the usual data-loader
pattern) must not change the device value.
"""
import ctypes
import gc
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from metal_testing import run_python

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


def mutate_right_after_device_put(busy, may_alias, mb):
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
        h = np.asarray(x)
        bad = np.flatnonzero(h != i)
        # On a failure, what the device got says what went wrong: a later
        # put's value (staging reused / freed early), -1 (the caller's
        # buffer read after device_put returned) or anything else (read
        # before the copy).
        assert bad.size == 0, dict(put=i, wrong=bad.size, first=bad[:3],
                                   last=bad[-3:], values=np.unique(h[bad])[:8])


# Snapshot path: below min(256 MB, max(16 MB, reclaimable memory / 8)).
@pytest.mark.parametrize("mb", [1, 32])
@pytest.mark.parametrize("busy", [False, True], ids=["idle", "busy"])
@pytest.mark.parametrize("may_alias", [None, False])
def test_source_mutated_right_after_device_put(busy, may_alias, mb):
    mutate_right_after_device_put(busy, may_alias, mb)


def test_source_mutated_right_after_large_device_put():
    # The waiting path, at 32 MB with the snapshot cap forced down to 16 MB.
    code = (f"import sys; sys.path.insert(0, {os.path.dirname(__file__)!r})\n"
            "import test_transfers as t\n"
            "for busy in (False, True):\n"
            "    for may_alias in (None, False):\n"
            "        t.mutate_right_after_device_put(busy, may_alias, 32)\n")
    out = run_python(code, dict(os.environ, METAL_PJRT_SNAPSHOT_MAX_MB="16"))
    assert out.returncode == 0, out.stderr[-3000:]


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


def _host_port_send_refs():
    lib = ctypes.CDLL(None)
    lib.mach_task_self.restype = lib.mach_host_self.restype = ctypes.c_uint32
    task, host = lib.mach_task_self(), lib.mach_host_self()  # +1 ref itself
    refs = ctypes.c_uint32()
    assert lib.mach_port_get_refs(task, host, 0, ctypes.byref(refs)) == 0  # MACH_PORT_RIGHT_SEND
    return refs.value - 1  # (the ref taken just now stays; the next call's -1 accounts for it)


def test_device_puts_do_not_leak_host_port_refs():
    # Each mach_host_self() adds a send-right reference; at 65534 the name
    # overflows and the system memory query (hence the allocation guard)
    # fails. 1 MB and 24 MB puts both query it (allocation guard; snapshot
    # threshold), so they must release what they take.
    a, b = np.ones(1 << 18, np.float32), np.ones(6 << 20, np.float32)
    jax.device_put(a).block_until_ready()
    before = _host_port_send_refs()
    for _ in range(200):
        jax.device_put(a).block_until_ready()
    for _ in range(20):
        jax.device_put(b).block_until_ready()
    grew = _host_port_send_refs() - before - 1  # our own call's reference
    assert grew <= 0, f"{grew} host port send refs leaked by 220 device_puts"


@pytest.mark.parametrize("put", ["zero_size", "copy"])
def test_transfer_behind_a_compute_backlog(put):
    # ~1 s of short GEMM chains queued on the compute stream, then a transfer
    # that XLA orders after it: a zero-byte device-to-device copy (only waits
    # for the destination's allocation, then records an event) and
    # device_put(x, may_alias=False) of an x still being computed. Neither
    # may wait on the GPU (that counts against the watchdog): with one queue
    # per device they are simply behind the backlog.
    chain = jax.jit(lambda x, w: jax.lax.fori_loop(
        0, 16, lambda i, x: jnp.tanh(x @ w), x, unroll=True))
    k = jax.random.PRNGKey(0)
    x = jax.random.normal(k, (1024, 1024), jnp.float32)
    w = jax.random.normal(k, (1024, 1024), jnp.float32) / 32
    chain(x, w).block_until_ready()
    one = float("inf")
    for _ in range(3):
        t = time.perf_counter()
        chain(x, w).block_until_ready()
        one = min(one, time.perf_counter() - t)
    # Each call (and so each of its command buffers) is far below the
    # watchdog; together they queue ~1 s.
    assert one < 0.1, one
    empty = jnp.zeros((0, 3)).block_until_ready()
    y = x
    for _ in range(int(1.0 / one)):
        y = chain(y, w)
    if put == "zero_size":
        z = jax.device_put(empty, may_alias=False)
        assert z.block_until_ready().shape == (0, 3)
    else:
        c = jax.device_put(y, may_alias=False)
        np.testing.assert_array_equal(np.asarray(c), np.asarray(y))
    cpu = jax.devices("cpu")[0]
    np.testing.assert_allclose(
        np.asarray(chain(x, w)),
        np.asarray(chain(jax.device_put(x, cpu), jax.device_put(w, cpu))),
        atol=1e-5)


def test_pinned_host_and_strided_puts():
    # XLA's host memory space (pinned_host arrays; the pool that linearizes
    # non-dense host arrays) is plain host memory the GPU can copy to and
    # from, outside the device budget.
    dev = jax.devices()[0]
    pinned = jax.sharding.SingleDeviceSharding(dev, memory_kind="pinned_host")
    x = jnp.arange(1 << 20, dtype=jnp.float32)
    live = _memory_stats()["live"]
    h = jax.device_put(x, pinned)
    assert h.sharding.memory_kind == "pinned_host"
    np.testing.assert_array_equal(np.asarray(h), np.asarray(x))
    assert _memory_stats()["live"] == live
    back = jax.device_put(h, jax.sharding.SingleDeviceSharding(dev))
    np.testing.assert_array_equal(np.asarray(back), np.asarray(x))
    np.testing.assert_array_equal(
        np.asarray(jax.device_put(h, pinned, may_alias=False)), np.asarray(x))
    a = np.arange(256 * 64, dtype=np.float32).reshape(256, 64)
    np.testing.assert_array_equal(np.asarray(jax.device_put(a.T)), a.T)
    np.testing.assert_array_equal(np.asarray(jax.device_put(a[::2, 1::3])),
                                  a[::2, 1::3])


def test_memory_kinds_inside_jit():
    # device_put to a memory kind inside jit takes XLA's host offloading (as
    # on cuda), and the result reports the kind it was put in.
    dev = jax.devices()[0]
    host = jax.sharding.SingleDeviceSharding(dev, memory_kind="pinned_host")
    device = jax.sharding.SingleDeviceSharding(dev, memory_kind="device")
    x = jax.device_put(jnp.arange(8, dtype=jnp.float32), device)
    xh = jax.device_put(np.arange(8, dtype=np.float32), host)
    to_host = jax.jit(lambda a: jax.device_put(a * 2, host))(x)
    assert to_host.sharding.memory_kind == "pinned_host"
    np.testing.assert_array_equal(np.asarray(to_host), 2 * np.arange(8))
    to_dev = jax.jit(lambda a: jax.device_put(a, device) + 1)(xh)
    assert to_dev.sharding.memory_kind == "device"
    np.testing.assert_array_equal(np.asarray(to_dev), np.arange(8) + 1)
    both = jax.jit(lambda a: (jax.device_put(a, host), a + 1))(x)
    assert [b.sharding.memory_kind for b in both] == ["pinned_host", "device"]
    out = jax.jit(lambda a: a * 3, out_shardings=host)(x)
    assert out.sharding.memory_kind == "pinned_host"
    np.testing.assert_array_equal(np.asarray(out), 3 * np.arange(8))


def _memory_stats():
    import jax_graft
    lib = ctypes.CDLL(str(jax_graft._get_library_path()))
    out = (ctypes.c_uint64 * 8)()
    assert lib.metal_pjrt_memory_stats(0, out) == 0
    return dict(zip(("live", "cached", "budget", "hits", "misses", "pressure",
                     "kernels", "kernel_msl_bytes"), out))
