"""Buffer donation (donate_argnums) on mtl: the plugin adds "mtl" to JAX's
list of platforms with donation (metal_pjrt_plugin.initialize), and XLA's
GPU client aliases a donated input to an output.

Checked here: the output reuses the donated buffer and the input is
deleted; the donated buffer is never handed out again while the output is
alive (the runtime's size-class cache would give the same MTLBuffer to the
next same-size allocation), and a loop of donated steps neither leaks nor
double counts; ordering against GPU work, a host LAPACK call and a host
callback that still read the donated buffer; donating a fresh device_put on
both of its paths; partial and pytree donation; a donation with no
matching output (copied, still correct).
"""
import ctypes
import os
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import metal_pjrt_plugin
from metal_testing import run_python

pytestmark = pytest.mark.metal

MB = 1 << 20


def test_mtl_has_donation():
    from jax._src.interpreters import mlir
    jax.devices("mtl")  # the plugin is initialized with the backend
    assert "mtl" in mlir._platforms_with_donation
    assert mlir._platforms_with_donation.count("mtl") == 1


def test_donated_input_becomes_the_output():
    f = jax.jit(lambda x: x * 2 + 1, donate_argnums=0)
    x = jnp.arange(1 << 20, dtype=jnp.float32)
    ptr = x.unsafe_buffer_pointer()
    lowered = f.lower(x).as_text()
    assert "tf.aliasing_output = 0" in lowered, lowered[:500]
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # "donated buffers were not usable"
        y = f(x)
    assert x.is_deleted()
    assert y.unsafe_buffer_pointer() == ptr
    np.testing.assert_array_equal(
        np.asarray(y), np.arange(1 << 20, dtype=np.float32) * 2 + 1)


def test_use_after_donation_raises():
    f = jax.jit(lambda x: x + 1, donate_argnums=0)
    x = jnp.ones(1024)
    f(x).block_until_ready()
    with pytest.raises(RuntimeError, match="deleted"):
        np.asarray(x)
    with pytest.raises(RuntimeError, match="deleted"):
        f(x)


def test_donation_while_a_reader_is_in_flight():
    # ~100 ms of GPU work reading `a` is queued, then `a` is donated to a
    # jit that overwrites it: the reader must see the old values.
    slow = jax.jit(lambda a: jax.lax.fori_loop(
        0, 16, lambda i, m: (m @ a) * 1e-3 + a, a))
    overwrite = jax.jit(lambda a: jnp.full_like(a, -7.0), donate_argnums=0)
    base = jax.random.normal(jax.random.PRNGKey(0), (2048, 2048))
    want = np.asarray(slow(base))
    for _ in range(3):
        a = base + 0.0  # a fresh buffer to donate
        r = slow(a)
        b = overwrite(a)
        np.testing.assert_array_equal(np.asarray(r), want)
        assert np.all(np.asarray(b) == -7.0)


def test_donation_while_host_tasks_read():
    # A host LAPACK call (n > 32 takes the host path after a stream sync)
    # and a host callback read `a`; `a` is then donated right away.
    overwrite = jax.jit(lambda a: jnp.full_like(a, np.nan), donate_argnums=0)
    rng = np.random.default_rng(0)
    m = rng.standard_normal((96, 96)).astype(np.float32)
    spd = m @ m.T + 96 * np.eye(96, dtype=np.float32)
    chol = jax.jit(jnp.linalg.cholesky)
    want = np.asarray(chol(jnp.asarray(spd)))
    host_sum = jax.jit(lambda a: jax.pure_callback(
        lambda v: np.asarray(v.sum(), np.float32),
        jax.ShapeDtypeStruct((), jnp.float32), a))
    for _ in range(3):
        a = jnp.asarray(spd) + 0.0
        l = chol(a)
        s = host_sum(a)
        b = overwrite(a)
        np.testing.assert_array_equal(np.asarray(l), want)
        np.testing.assert_allclose(float(s), spd.sum(), rtol=1e-6)
        assert np.all(np.isnan(np.asarray(b)))


def donate_fresh_device_put(mb):
    """device_put a numpy buffer, donate the result at once, refill the
    numpy buffer: the donated step must see the device_put values."""
    f = jax.jit(lambda x: x * 3.0, donate_argnums=0)
    buf = np.empty(mb * MB // 4, np.float32)
    for i in range(4):
        buf[:] = i
        y = f(jax.device_put(buf))
        buf[:] = -1
        assert np.all(np.asarray(y) == 3.0 * i), (i, np.unique(np.asarray(y))[:4])


def test_donate_fresh_device_put_snapshot():
    donate_fresh_device_put(1)


def test_donate_fresh_device_put_awaited():
    # The waiting path, at 32 MB with the snapshot cap forced down to 16 MB.
    code = (f"import sys; sys.path.insert(0, {os.path.dirname(__file__)!r})\n"
            "import test_donation as t\n"
            "t.donate_fresh_device_put(32)\n")
    out = run_python(code, dict(os.environ, JAX_PLATFORMS="mtl",
                                METAL_PJRT_SNAPSHOT_MAX_MB="16"))
    assert out.returncode == 0, out.stderr[-3000:]


def test_donation_with_a_callback():
    f = jax.jit(lambda x: x * 2 + jax.pure_callback(
        np.sin, jax.ShapeDtypeStruct(x.shape, x.dtype), x), donate_argnums=0)
    a = np.linspace(0, 1, 4096, dtype=np.float32)
    x = jnp.asarray(a)
    ptr = x.unsafe_buffer_pointer()
    y = f(x)
    assert x.is_deleted() and y.unsafe_buffer_pointer() == ptr
    np.testing.assert_allclose(np.asarray(y), a * 2 + np.sin(a), rtol=1e-6)


def test_partial_and_pytree_donation():
    f = jax.jit(lambda x, y: (x + y, y * 2), donate_argnums=0)
    x, y = jnp.ones(4096), jnp.full(4096, 2.0)
    s, d = f(x, y)
    assert x.is_deleted() and not y.is_deleted()
    assert np.all(np.asarray(s) == 3) and np.all(np.asarray(d) == 4)
    assert np.all(np.asarray(y) == 2)

    g = jax.jit(lambda p: {"a": p["a"] + 1, "b": p["b"] * p["a"]},
                donate_argnums=0)
    p = {"a": jnp.full(4096, 3.0), "b": jnp.full(4096, 5.0)}
    ptrs = {k: v.unsafe_buffer_pointer() for k, v in p.items()}
    q = g(p)
    assert all(v.is_deleted() for v in p.values())
    assert {q[k].unsafe_buffer_pointer() for k in q} == set(ptrs.values())
    assert np.all(np.asarray(q["a"]) == 4) and np.all(np.asarray(q["b"]) == 15)


def test_donation_without_a_matching_output():
    # No output has the donated input's shape: JAX warns, XLA copies, and
    # the result is still right.
    f = jax.jit(lambda x: x[:512] * 2, donate_argnums=0)
    x = jnp.arange(1024.0)
    with pytest.warns(UserWarning, match="donated buffers were not usable"):
        y = f(x)
    np.testing.assert_array_equal(np.asarray(y), np.arange(512.0) * 2)


def test_donated_buffer_is_not_recycled_while_the_output_lives():
    # In a fresh process, so the memory stats are this test's alone.
    code = r"""
import ctypes, gc
import numpy as np, jax, jax.numpy as jnp
import metal_pjrt_plugin
lib = ctypes.CDLL(str(metal_pjrt_plugin._get_library_path()))
def stats():
    out = (ctypes.c_uint64 * 6)()
    assert lib.metal_pjrt_memory_stats(0, out) == 0
    return dict(zip(("live", "cached", "budget", "hits", "misses", "pressure"), out))
MB = 1 << 20
n = 4 * MB  # 16 MB of f32
step = jax.jit(lambda x, i: x * 0.5 + i, donate_argnums=0)
fill = jax.jit(lambda i: jnp.full((n,), i, jnp.float32))

# The donated buffer moves to the output; same-size allocations and a cache
# drop (the memory-pressure hook) must not touch it.
x = jnp.full((n,), 2.0, jnp.float32).block_until_ready()
y = step(x, 1.0).block_until_ready()
assert x.is_deleted()
others = [fill(float(i)).block_until_ready() for i in range(4)]
del others; gc.collect()
lib.metal_pjrt_memory_pressure(1)
more = [fill(-1.0).block_until_ready() for _ in range(4)]
lib.metal_pjrt_memory_pressure(0)
more += [fill(-2.0).block_until_ready() for _ in range(4)]
assert np.all(np.asarray(y) == 2.0), np.unique(np.asarray(y))[:4]
del more, y; gc.collect()

# The KV-cache pattern: a state donated every step. Live memory stays at
# one state after warm-up (no leak, no double count).
x = jnp.zeros((n,), jnp.float32)
x = step(x, 1.0).block_until_ready()
live0 = stats()["live"]
for i in range(50):
    x = step(x, 1.0)
x.block_until_ready()
s = stats()
assert s["live"] == live0, (live0 // MB, s["live"] // MB, s)
assert s["live"] < live0 + 2 * MB
# x_{k+1} = x_k / 2 + 1 from 1 approaches 2.
assert np.allclose(np.asarray(x), 2.0), np.unique(np.asarray(x))[:4]
del x; gc.collect()
print("live MB after", stats()["live"] // MB)
print("OK")
"""
    out = run_python(code, dict(os.environ, JAX_PLATFORMS="mtl"))
    assert out.returncode == 0, (out.stdout[-2000:], out.stderr[-3000:])
    assert "OK" in out.stdout, out.stdout
