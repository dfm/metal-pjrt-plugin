"""Dispatch-bound programs: long chains of tiny kernels and a scan with tiny
state, where per-launch host cost and command-buffer batching dominate.

The GPU runs these tiny kernels in one of two performance states (~4.2 vs
~2.3 us of GPU time per dispatch, METAL_PJRT_TRACE=1), chosen by the OS from
the duty cycle: chain16 always lands in the slow one, chain256 in the fast
one, and chain64 flips between them from call to call (~410 vs ~720 us wall).
So each line reports p10 / median / p90 over many calls; compare A/B arms on
all three, not the median alone (docs/performance.md, 2026-09-27).
"""
import os, statistics, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import jax, jax.numpy as jnp
from common import emit

x = jnp.ones((1024,), jnp.float32)


def measure(fn, warmup, iters):
    for _ in range(warmup):
        jax.block_until_ready(fn())
    t = []
    for _ in range(iters):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        t.append((time.perf_counter() - t0) * 1e6)
    q = statistics.quantiles(t, n=10)
    return q[0], statistics.median(t), q[-1], min(t)


def report(name, steps, fn, warmup, iters):
    p10, p50, p90, best = measure(fn, warmup, iters)
    per = f" ({p50 / steps:.1f} us/step)" if steps else ""
    print(f"{jax.default_backend():8s} {name}: median {p50:7.0f} us, best {best:7.0f} us, "
          f"p10 {p10:7.0f}, p90 {p90:7.0f}{per}", flush=True)
    if os.environ.get("BENCH_OUT"):
        emit(os.environ.get("BENCH_LABEL", jax.default_backend()), name, p50 / 1e3,
             best / 1e3, {"p10_ms": round(p10 / 1e3, 3), "p90_ms": round(p90 / 1e3, 3)})


def chain(n):
    def f(x):
        for i in range(n):
            x = jnp.sin(x) * 1.001 + jnp.sum(x) * 1e-4  # one fusion + reduction each
        return x
    return jax.jit(f)


for n in (16, 64, 256):
    f = chain(n)
    report(f"chain of {n:3d} steps", n, lambda: f(x), warmup=5, iters=200)

# Sequential scan with tiny state: a while loop whose body is a few fusions.
def scan_body(carry, u):
    a, b = carry
    a = a * 0.99 + u
    b = b + jnp.sin(a)
    return (a, b), b
us = jnp.ones((2000, 8), jnp.float32)
g = jax.jit(lambda us: jax.lax.scan(scan_body, (jnp.zeros(8), jnp.zeros(8)), us)[1])
report("scan 2000 steps", 0, lambda: g(us), warmup=3, iters=30)
