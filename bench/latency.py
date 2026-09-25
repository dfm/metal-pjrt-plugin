"""Fixed per-call overhead: tiny arrays, many calls."""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import jax, jax.numpy as jnp
from common import timeit
x = jnp.ones((1024,), jnp.float32)
f = jax.jit(lambda x: x * 2.0 + 1.0)
ms, best = timeit(lambda: f(x), sync=jax.block_until_ready, warmup=10, iters=200)
print(f"{jax.default_backend():8s} jit(x*2+1) on 1K floats: median {ms*1000:.0f} us, best {best*1000:.0f} us")
g = jax.jit(lambda x: jnp.sum(jnp.exp(x)) + jnp.sum(x * x))  # two kernels + tiny outputs
ms, best = timeit(lambda: g(x), sync=jax.block_until_ready, warmup=10, iters=200)
print(f"{jax.default_backend():8s} two-kernel program: median {ms*1000:.0f} us, best {best*1000:.0f} us")
