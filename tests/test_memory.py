# Copyright 2026 The jax-graft Authors
# SPDX-License-Identifier: Apache-2.0

"""Memory policy (runtime Device::Allocate behind XLA's platform allocator):
a workload that outgrows the budget gets RESOURCE_EXHAUSTED and the process
keeps working; memory goes back to the system after a computation; a
memory-pressure warning drops the cache; critical pressure refuses.

Each test runs in a fresh process, so the budget can be made artificially
small (METAL_PJRT_MEMORY_FRACTION) and nothing here gets near physical
memory. No timeouts: never kill a process with GPU work in flight.
"""
import os

import pytest

from metal_testing import run_python

pytestmark = pytest.mark.metal

PRELUDE = r"""
import ctypes, gc, time
import numpy as np, jax, jax.numpy as jnp
import jax_graft
lib = ctypes.CDLL(str(jax_graft._get_library_path()))
def stats():
    out = (ctypes.c_uint64 * 8)()
    assert lib.metal_pjrt_memory_stats(0, out) == 0
    return dict(zip(("live", "cached", "budget", "hits", "misses", "pressure",
                     "kernels", "kernel_msl_bytes"), out))
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
                     dict(os.environ, **env))
    assert out.returncode == 0, (out.stdout[-2000:], out.stderr[-3000:])
    return out.stdout


def small_budget_fraction(budget_mb=512):
    """METAL_PJRT_MEMORY_FRACTION giving a ~budget_mb budget (the default
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
    # Sizes in MB, and the setting that raises the budget, with a value.
    assert (" MB of its " in str(e)
            and "METAL_PJRT_MEMORY_FRACTION=" in str(e)), e
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
""", METAL_PJRT_MEMORY_FRACTION=small_budget_fraction())
    assert "after: OK" in out, out


def test_critical_memory_pressure_refuses():
    # The only system-level limit. Critical pressure is faked with the test
    # hook (never provoked for real): allocations of 1 MB or more are refused
    # after the cache is dropped, smaller ones still work, and allocating
    # resumes once the level is back to normal.
    out = run_child(r"""
lib.metal_pjrt_testing_memory_pressure(2)
try:
    try:
        jnp.ones((16 * MB,), jnp.float32).block_until_ready()
        raise SystemExit("no error")
    except jax.errors.JaxRuntimeError as e:
        assert "RESOURCE_EXHAUSTED" in str(e), e
        assert "metal-pjrt-plugin: in jit_broadcast_in_dim: Metal: allocating" in str(e), e
        assert "refused: the system is under critical memory pressure" in str(e), e
        assert "not this process's budget" in str(e), e
        print(str(e).splitlines()[0])
    assert float(jnp.sum(jnp.ones(1000))) == 1000.0
finally:
    lib.metal_pjrt_testing_memory_pressure(0)
assert float(jnp.ones((16 * MB,), jnp.float32)[123]) == 1.0
print("OK")
""")
    assert "OK" in out, out


def test_refusal_names_its_executable():
    # big_temp's ~512 MB of scratch is over the 256 MB budget, so it is
    # refused before anything is allocated. XLA's executable_name names the
    # last computation the error reached (the eager ops on big_temp's result);
    # the plugin's text names the one whose allocation was refused.
    out = run_child(r"""
@jax.jit
def big_temp(x):
    h = jnp.tanh(x @ x.T)
    return jnp.tanh(h @ h) @ x
x = jnp.ones((8192, 1024), jnp.float32)
try:
    big_temp(x).block_until_ready()
    raise SystemExit("no error")
except jax.errors.JaxRuntimeError as e:
    assert "metal-pjrt-plugin: in jit_big_temp: Metal: allocating" in str(e), e
try:
    int(jnp.argmax(big_temp(x), -1).sum())
    raise SystemExit("no error")
except jax.errors.JaxRuntimeError as e:
    assert ("metal-pjrt-plugin: in jit_big_temp (an earlier asynchronous "
            "computation; the error surfaced in jit_") in str(e), e
    assert "memory budget" in str(e), e
print("OK")
""", METAL_PJRT_MEMORY_FRACTION=small_budget_fraction(256))
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


def test_dropped_executables_release_their_kernels():
    # Compiled kernels belong to their executables: once the executables are
    # gone (jax.clear_caches: JAX's own caches hold every executable a
    # jitted function compiled), their pipelines, libraries and MSL go too,
    # and malloc'd memory stops growing with the programs run. Measured over
    # a second round of programs (the first grows ~110 KB per program of
    # one-time state, the second ~50, later ones ~10; kept kernels ~470).
    out = run_child(r"""
class MStats(ctypes.Structure):
    _fields_ = [(n, ctypes.c_size_t) for n in
                ("total", "chunks_used", "used", "chunks_free", "free")]
libc = ctypes.CDLL(None)
libc.mstats.restype = MStats
ops = (jnp.tanh, jnp.sin, jnp.exp, jnp.log1p, jnp.cos, jnp.sqrt, jnp.abs,
       jnp.negative)
def f(x):
    for op in ops:
        y = op(x * 0.5)
        x = y / (1.0 + jnp.sum(y * y, axis=-1, keepdims=True))
    return x
def run(p):  # a distinct shape: distinct kernels (8 or more)
    g = jax.jit(f)
    g(jnp.ones((16, 16 + p))).block_until_ready()
    return g
n = 32
run(-1)  # one-time state (Metal's shader cache)
jax.clear_caches(); gc.collect()
base = stats()
for r in range(2):
    base_used = libc.mstats().used
    held = [run(p) for p in range(r * n, (r + 1) * n)]
    grown = stats()
    del held
    jax.clear_caches(); gc.collect()
    s, used = stats(), libc.mstats().used
    print(f"kernels {base['kernels']} -> {grown['kernels']} -> {s['kernels']}; "
          f"malloc used MB {base_used / MB:.1f} -> {used / MB:.1f}")
    assert grown["kernels"] >= base["kernels"] + n * 8, (base, grown)
    assert s["kernels"] == base["kernels"], (base, s)
    assert s["kernel_msl_bytes"] == base["kernel_msl_bytes"], (base, s)
per_program = (used - base_used) / n
assert per_program < 200 * 1024, per_program / 1024  # ~470 KB if kept
print("OK")
""")
    print(out)
    assert "OK" in out


def test_memory_pressure_drops_the_cache():
    # Calls the handler the DISPATCH_SOURCE_TYPE_MEMORYPRESSURE source runs.
    # A real notification: `sudo memory_pressure -S -l warn` (manual).
    out = run_child(r"""
x = jnp.ones((16 * MB,)); y = (x * 2).block_until_ready()
del x, y
assert stats()["cached"] >= 64 * MB, stats()
lib.metal_pjrt_testing_memory_pressure(1)
try:
    s = stats()
    assert s["cached"] == 0 and s["pressure"] == 1, s
    z = (jnp.ones((16 * MB,)) + 1).block_until_ready(); del z
    assert stats()["cached"] == 0, stats()  # frees release while under pressure
finally:
    lib.metal_pjrt_testing_memory_pressure(0)
z = (jnp.ones((16 * MB,)) + 1).block_until_ready(); del z
assert stats()["cached"] >= 64 * MB, stats()
print("OK")
""")
    assert "OK" in out


@pytest.mark.parametrize("fraction", ["abc", "0", "0.5x", "1.5"])
def test_bad_settings_warn_and_keep_defaults(fraction):
    # A value that does not parse is ignored with a warning (strtoull/atof
    # would read a typo as 0); a fraction above 1 is allowed, with a warning.
    out = run_python(PRELUDE + r"""
print("budget", stats()["budget"])
print(jax.jit(lambda x: x + 1)(1.0))
""", dict(os.environ, METAL_PJRT_MEMORY_FRACTION=fraction,
          METAL_PJRT_QUARANTINE_STRIKES="two",
          METAL_PJRT_DISABLE_REWRITES="scan,bogus"))
    assert out.returncode == 0, out.stderr[-3000:]
    assert "2.0" in out.stdout, out.stdout
    for want in ("Ignoring METAL_PJRT_QUARANTINE_STRIKES=two",
                 'Ignoring "bogus" in METAL_PJRT_DISABLE_REWRITES'):
        assert want in out.stderr, (want, out.stderr[-3000:])
    budget = int(out.stdout.split("budget ")[1].split()[0])
    half_ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // 2
    if fraction == "1.5":
        assert "METAL_PJRT_MEMORY_FRACTION=1.5 is above 1" in out.stderr
        # Past half of RAM, capped at the working set (~2/3 of RAM or more
        # on Apple GPUs).
        assert budget > half_ram, budget
    else:
        assert (f"Ignoring METAL_PJRT_MEMORY_FRACTION={fraction} (not a number "
                "> 0); using 1, a budget of ") in out.stderr, out.stderr[-3000:]
        assert 0 < budget <= half_ram, budget


def test_jax_memory_stats():
    # jax.Device.memory_stats() comes from the runtime (the plugin's
    # PJRT_Device_MemoryStats): live bytes (buffer lengths, page-rounded),
    # their peak, the budget as the limit, live + cached as the pool.
    out = run_child(r"""
dev = jax.devices("mtl")[0]
m0 = dev.memory_stats()
assert m0 is not None
for k in ("bytes_in_use", "peak_bytes_in_use", "bytes_limit", "num_allocs",
          "pool_bytes"):
    assert k in m0, m0
assert m0["bytes_limit"] == stats()["budget"], (m0, stats())
assert m0["bytes_in_use"] == stats()["live"], (m0, stats())
x = jax.device_put(np.ones(16 * MB, np.uint8), dev).block_until_ready()
m1 = dev.memory_stats()
assert m1["bytes_in_use"] >= m0["bytes_in_use"] + 16 * MB, (m0, m1)
assert m1["peak_bytes_in_use"] >= m1["bytes_in_use"], m1
assert m1["num_allocs"] > m0["num_allocs"], (m0, m1)
assert m1["pool_bytes"] >= m1["bytes_in_use"], m1
del x
gc.collect()
m2 = dev.memory_stats()
assert m2["bytes_in_use"] <= m1["bytes_in_use"] - 16 * MB, (m1, m2)
assert m2["peak_bytes_in_use"] == m1["peak_bytes_in_use"], (m1, m2)
print("OK")
""")
    assert "OK" in out, out
