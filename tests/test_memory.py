"""Memory policy (runtime Device::Allocate behind XLA's platform allocator):
a workload that outgrows the budget gets RESOURCE_EXHAUSTED and the process
keeps working; memory goes back to the system after a computation; a
memory-pressure warning drops the cache.

Each test runs in a fresh process, so the budget can be made artificially
small (JAX_MTL_MEMORY_FRACTION) and nothing here gets near physical
memory. No timeouts: never kill a process with GPU work in flight.
"""
import os

import pytest

from metal_testing import run_python

pytestmark = pytest.mark.metal

PRELUDE = r"""
import ctypes, gc, time
import numpy as np, jax, jax.numpy as jnp
import metal_pjrt_plugin
lib = ctypes.CDLL(str(metal_pjrt_plugin._get_library_path()))
def stats():
    out = (ctypes.c_uint64 * 6)()
    assert lib.metal_pjrt_memory_stats(0, out) == 0
    return dict(zip(("live", "cached", "budget", "hits", "misses", "pressure"), out))
def footprint():
    libc = ctypes.CDLL(None)
    buf = (ctypes.c_uint64 * 64)(); count = ctypes.c_uint32(128)
    assert libc.task_info(ctypes.c_uint32.in_dll(libc, "mach_task_self_"), 22,  # TASK_VM_INFO
                          ctypes.byref(buf), ctypes.byref(count)) == 0
    return buf[18]  # phys_footprint
MB = 1 << 20
jnp.zeros(1).block_until_ready()
"""


def run_child(code, **env):
    out = run_python(PRELUDE + code,
                     dict(os.environ, JAX_PLATFORMS="mtl", **env))
    assert out.returncode == 0, (out.stdout[-2000:], out.stderr[-3000:])
    return out.stdout


def small_budget_fraction(budget_mb=512):
    """JAX_MTL_MEMORY_FRACTION giving a ~budget_mb budget (the default
    budget is half of RAM, capped by the GPU's recommended working set)."""
    ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    return str(budget_mb * 2**20 / (ram / 2))


def test_growing_workload_gets_resource_exhausted():
    out = run_child(r"""
s = stats()
assert 400 * MB < s["budget"] < 600 * MB, s
make = jax.jit(lambda i: jnp.full((16 * MB,), i, jnp.float32))  # 64 MB
held = []
try:
    for i in range(64):
        held.append(make(float(i)).block_until_ready())
    raise SystemExit("no error")
except jax.errors.JaxRuntimeError as e:
    # XLA's generic message, then the runtime's reason.
    assert "RESOURCE_EXHAUSTED" in str(e) and "memory budget" in str(e), e
print("held", len(held), "live MB", stats()["live"] // MB)
assert len(held) * 64 * MB <= s["budget"]
# One program whose output alone exceeds the budget.
try:
    jnp.ones((256 * MB,), jnp.float32).block_until_ready()
    raise SystemExit("no error")
except jax.errors.JaxRuntimeError as e:
    assert "RESOURCE_EXHAUSTED" in str(e), e
del held
gc.collect()
x = make(3.0) * 2
assert float(x[123]) == 6.0 and float(jnp.sum(x[:1000])) == 6000.0
print("after: OK")
""", JAX_MTL_MEMORY_FRACTION=small_budget_fraction())
    assert "after: OK" in out, out


def test_system_memory_guard_says_why():
    # A reserve larger than any Mac's RAM: every allocation of 1 MB or more is
    # refused by the system memory guard, smaller ones still work.
    out = run_child(r"""
try:
    jnp.ones((16 * MB,), jnp.float32).block_until_ready()
    raise SystemExit("no error")
except jax.errors.JaxRuntimeError as e:
    assert "RESOURCE_EXHAUSTED" in str(e) and "system memory guard" in str(e), e
    print(str(e).splitlines()[0])
assert float(jnp.sum(jnp.ones(1000))) == 1000.0
print("OK")
""", METAL_PJRT_SYSTEM_MEMORY_RESERVE_MB=str(1 << 30))
    assert "OK" in out, out


def test_memory_returned_after_compute():
    out = run_child(r"""
base = footprint()
x = jax.random.normal(jax.random.PRNGKey(0), (32 * MB,))  # 128 MB
f = jax.jit(lambda x: jnp.sum(jnp.sin(x) * jnp.exp(x)) + jnp.sum(jnp.cos(x) ** 2))
for _ in range(3):
    f(x).block_until_ready()
del x
gc.collect()
s = stats()
assert s["cached"] >= 128 * MB, s
peak = footprint()
time.sleep(4.0)  # Device::kCacheIdleRelease (2 s) + timer period and slack
s = stats()
idle = footprint()
print(f"footprint MB: base {base // MB} peak {peak // MB} idle {idle // MB}; {s}")
assert s["cached"] == 0 and s["live"] < 8 * MB, s
assert idle < base + 100 * MB, (base // MB, idle // MB)
""")
    print(out)


def test_memory_pressure_drops_the_cache():
    # Calls the handler the DISPATCH_SOURCE_TYPE_MEMORYPRESSURE source runs.
    # A real notification: `sudo memory_pressure -S -l warn` (manual).
    out = run_child(r"""
x = jnp.ones((16 * MB,)); y = (x * 2).block_until_ready()
del x, y
assert stats()["cached"] >= 64 * MB, stats()
lib.metal_pjrt_memory_pressure(1)
s = stats()
assert s["cached"] == 0 and s["pressure"] == 1, s
z = (jnp.ones((16 * MB,)) + 1).block_until_ready(); del z
assert stats()["cached"] == 0, stats()  # frees release while under pressure
lib.metal_pjrt_memory_pressure(0)
z = (jnp.ones((16 * MB,)) + 1).block_until_ready(); del z
assert stats()["cached"] >= 64 * MB, stats()
print("OK")
""")
    assert "OK" in out
