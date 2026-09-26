"""Dispatch-bound programs: long chains of tiny kernels and a scan with tiny
state, where per-launch host cost and command-buffer batching dominate."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import jax, jax.numpy as jnp
from common import timeit

x = jnp.ones((1024,), jnp.float32)

def chain(n):
    def f(x):
        for i in range(n):
            x = jnp.sin(x) * 1.001 + jnp.sum(x) * 1e-4  # one fusion + reduction each
        return x
    return jax.jit(f)

for n in (16, 64, 256):
    f = chain(n)
    ms, best = timeit(lambda: f(x), sync=jax.block_until_ready, warmup=5, iters=50)
    print(f"{jax.default_backend():8s} chain of {n:3d} steps: median {ms*1000:7.0f} us, best {best*1000:7.0f} us "
          f"({ms*1000/n:.1f} us/step)")

# Sequential scan with tiny state: a while loop whose body is a few fusions.
def scan_body(carry, u):
    a, b = carry
    a = a * 0.99 + u
    b = b + jnp.sin(a)
    return (a, b), b
us = jnp.ones((2000, 8), jnp.float32)
g = jax.jit(lambda us: jax.lax.scan(scan_body, (jnp.zeros(8), jnp.zeros(8)), us)[1])
ms, best = timeit(lambda: g(us), sync=jax.block_until_ready, warmup=3, iters=10)
print(f"{jax.default_backend():8s} scan 2000 steps (unrolled by XLA? no): median {ms*1000:7.0f} us, best {best*1000:7.0f} us")
