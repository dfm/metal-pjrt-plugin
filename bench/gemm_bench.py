# Same GEMM cases as bench/jax_bench.py, without its unrelated setup.
import os, sys
sys.path.insert(0, "/Users/dfm/src/dfm/jax-plugin/bench")
# openmetal is opt-in (not the default backend); JAX_PLATFORMS=cpu for the CPU arm.
os.environ.setdefault("JAX_PLATFORMS", "openmetal")
import jax, jax.numpy as jnp
from common import timeit, emit
B = os.environ.get("BENCH_LABEL", jax.default_backend())
key = jax.random.PRNGKey(0)
def run(name, fn, *args):
    jfn = jax.jit(fn)
    ms, best = timeit(lambda: jfn(*args), sync=jax.block_until_ready)
    emit(B, name, ms, best)
a = jax.random.normal(key, (2048, 2048), jnp.float32)
run("matmul f32 2048", lambda a: a @ a, a)
a16 = jax.random.normal(key, (2048, 2048), jnp.bfloat16)
run("matmul bf16 2048", lambda a: a @ a, a16)
h16 = jax.random.normal(key, (2048, 2048), jnp.float16)
run("matmul f16 2048", lambda a: a @ a, h16)
run("matmul bf16 2048 a@a.T", lambda a: a @ a.T, a16)
ab = jax.random.normal(key, (16, 512, 512), jnp.float32)
run("batched matmul 16x512", lambda a: jnp.einsum("bij,bjk->bik", a, a), ab)
ab16 = ab.astype(jnp.bfloat16)
run("batched matmul bf16 16x512", lambda a: jnp.einsum("bij,bjk->bik", a, a), ab16)
