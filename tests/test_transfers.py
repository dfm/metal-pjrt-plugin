"""Host <-> device transfers without staging
(should_stage_host_to_device_transfers=False): exact round trips, and the
source/destination numpy arrays' lifetimes.

JAX lets the H2D copy run after device_put returns, so the plugin copies
the source into a host temporary before returning or, from min(256 MB,
max(16 MB, reclaimable memory / 8)) up, waits for the copy: a caller
refilling its numpy buffer right after device_put (the usual data-loader
pattern) must not change the device value. Behind queued GPU work the
runtime stages a host-to-device copy through a GPU copy (within 64 MB in
flight) instead of a host task; metal_pjrt_transfer_stats counts both.
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


def transfer_stats():
    """{staged, host_task, staging_bytes}: host-to-device copies staged
    through a GPU copy, those run as host tasks, staging bytes in flight."""
    import metal_pjrt_plugin
    lib = ctypes.CDLL(str(metal_pjrt_plugin._get_library_path()))
    out = (ctypes.c_uint64 * 3)()
    assert lib.metal_pjrt_transfer_stats(0, out) == 0
    return dict(zip(("staged", "host_task", "staging_bytes"), out))


def mutate_right_after_device_put(busy, may_alias, mb):
    slow = jax.jit(lambda a: jax.lax.fori_loop(0, 16, lambda i, a: (a @ a) * 1e-3, a))
    a = jnp.ones((2048, 2048))
    slow(a).block_until_ready()
    buf = np.empty(mb << 18, np.float32)
    xs = []
    before = transfer_stats()
    for i in range(6):
        pending = slow(a) if busy else None  # ~100 ms of GPU work queued first
        buf[:] = i
        xs.append(jax.device_put(buf, may_alias=may_alias))
        buf[-1024:] = -1  # the tail first: a copy still running reads it last
        buf[:] = -1
        del pending
    after = transfer_stats()
    for i, x in enumerate(xs):
        h = np.asarray(x)
        bad = np.flatnonzero(h != i)
        # On a failure, what the device got says what went wrong: a later
        # put's value (staging reused / freed early), -1 (the caller's
        # buffer read after device_put returned) or anything else (read
        # before the copy).
        assert bad.size == 0, dict(put=i, wrong=bad.size, first=bad[:3],
                                   last=bad[-3:], values=np.unique(h[bad])[:8])
    return {k: after[k] - before[k] for k in ("staged", "host_task")}


# Snapshot path: below min(256 MB, max(16 MB, reclaimable memory / 8)).
@pytest.mark.parametrize("mb", [1, 32])
@pytest.mark.parametrize("busy", [False, True], ids=["idle", "busy"])
@pytest.mark.parametrize("may_alias", [None, False])
def test_source_mutated_right_after_device_put(busy, may_alias, mb):
    d = mutate_right_after_device_put(busy, may_alias, mb)
    # Behind queued work a small put is staged, never a host task. (At 32 MB
    # the snapshot-or-wait choice depends on free memory, and a waited put
    # finds the stream idle.)
    if busy and mb == 1:
        assert d == {"staged": 6, "host_task": 0}, d


def slow_fn(n):
    # Stays all ones (no inf/nan) for a ones matrix of size 2048.
    return jax.jit(lambda a: jax.lax.fori_loop(0, n, lambda i, a: (a @ a) / 2048, a))


def test_staged_put_waits_for_readers_of_the_destination():
    # x's buffer is freed while queued work still reads it; the next put of
    # the same size gets it back from the cache, and its GPU copy must wait
    # for that work (XLA orders the transfer after the compute stream).
    slow = slow_fn(16)
    late_read = jax.jit(lambda a, x: slow(a).sum() * 0 + x * 2)
    a = jnp.ones((2048, 2048))
    late_read(a, jnp.zeros(1 << 18)).block_until_ready()
    before = transfer_stats()
    for i in range(4):
        x = jax.device_put(np.full(1 << 18, float(i), np.float32))
        x.block_until_ready()
        y = late_read(a, x)  # ~100 ms, then reads x
        del x
        z = jax.device_put(np.full(1 << 18, -7.0, np.float32))
        assert np.all(np.asarray(y) == 2 * i), i
        assert np.all(np.asarray(z) == -7), i
    d = transfer_stats()
    assert d["staged"] - before["staged"] >= 4, (before, d)


def test_staging_in_flight_is_bounded():
    # 8 x 12 MB puts (snapshotted: below 16 MB) behind ~300 ms of GPU work:
    # at most 64 MB is staged at once, beyond it a put runs as a host task,
    # and every value arrives.
    slow = slow_fn(48)
    a = jnp.ones((2048, 2048))
    slow(a).block_until_ready()
    before = transfer_stats()
    pending = slow(a)
    xs, peak = [], 0
    for i in range(8):
        xs.append(jax.device_put(np.full(3 << 20, float(i), np.float32)))
        peak = max(peak, transfer_stats()["staging_bytes"])
    after = transfer_stats()
    del pending
    for i, x in enumerate(xs):
        assert np.all(np.asarray(x) == i), i
    staged = after["staged"] - before["staged"]
    host_task = after["host_task"] - before["host_task"]
    assert peak <= 64 << 20, peak
    # (A put whose stream has nothing to wait for copies on the calling
    # thread, counted by neither.)
    assert 1 <= staged <= 5 and host_task >= 1 and staged + host_task <= 8, (
        staged, host_task)
    # Released by the command buffers' completion handlers, which may run
    # just after the values are ready.
    for _ in range(200):
        if transfer_stats()["staging_bytes"] == 0:
            break
        time.sleep(0.01)
    assert transfer_stats()["staging_bytes"] == 0


def test_source_mutated_right_after_large_device_put():
    # The waiting path, at 32 MB with the snapshot cap forced down to 16 MB.
    code = (f"import sys; sys.path.insert(0, {os.path.dirname(__file__)!r})\n"
            "import test_transfers as t\n"
            "for busy in (False, True):\n"
            "    for may_alias in (None, False):\n"
            "        t.mutate_right_after_device_put(busy, may_alias, 32)\n")
    out = run_python(code, dict(os.environ, JAX_PLATFORMS="mtl",
                                METAL_PJRT_SNAPSHOT_MAX_MB="16"))
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
