# Copyright 2026 The metal-pjrt-plugin Authors
# SPDX-License-Identifier: Apache-2.0
"""Few-row bf16 GEMMs on the default JAX backend (batched LLM decode's case).

  JAX_PLATFORMS=mtl,cpu .venv/bin/python examples/llm/skinny_jax.py

(prefix it with `scripts/device_lock.py --` when other GPU jobs may run;
skinny_mlx.py is the same measurement in MLX.)

28 GEMMs x[M, 1024] @ W.T with bf16 W [6144, 1024], one jitted program,
for M = 1..64, two ways: "list" returns all 28 products of the same x
(what a decode step's layers look like to XLA, which may merge them);
"chained" feeds each product into the next (x <- (x @ W.T)[:, :1024]),
so neither side can overlap or merge them. Prints p10 ms per call over
--rounds rounds of --bursts bursts of --calls calls (JSON rows) and the
achieved weight bandwidth. The weights are 28 distinct arrays: identical
copies would stay in the GPU's cache.
"""
import argparse, json, time

import jax
import jax.numpy as jnp
import numpy as np

L, K, N = 28, 1024, 6144


def time_ms(f, args, bursts, calls):
    ts = []
    for _ in range(bursts):
        for _ in range(calls):
            t0 = time.perf_counter()
            jax.block_until_ready(f(*args))
            ts.append((time.perf_counter() - t0) * 1e3)
    return ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="1,2,4,8,16,32,64")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--bursts", type=int, default=3)
    ap.add_argument("--calls", type=int, default=10)
    args = ap.parse_args()
    keys = jax.random.split(jax.random.key(0), L)
    Ws = [(jax.random.normal(k, (N, K), jnp.float32) * 0.02).astype(jnp.bfloat16) for k in keys]
    plain = jax.jit(lambda x, Ws: [x @ W.T for W in Ws])

    @jax.jit
    def chained(x, Ws):
        for W in Ws:
            x = (x @ W.T)[:, :K]
        return x

    fns = {"list": plain, "chained": chained}
    rows = [int(m) for m in args.rows.split(",")]
    xs = {m: jnp.ones((m, K), jnp.bfloat16) * 0.01 for m in rows}
    for m in rows:                                    # compile and warm up
        for f in fns.values():
            time_ms(f, (xs[m], Ws), 1, 3)
    ts = {(m, k): [] for m in rows for k in fns}
    for _ in range(args.rounds):                      # rounds interleave the cases
        for m in rows:
            for k, f in fns.items():
                ts[m, k] += time_ms(f, (xs[m], Ws), args.bursts, args.calls)
    for (m, k), t in ts.items():
        p10 = float(np.percentile(t, 10))
        print(json.dumps({"impl": f"jax-{jax.devices()[0].platform}", "M": m, "mode": k,
                          "p10_ms": round(p10, 2), "median_ms": round(float(np.median(t)), 2),
                          "weight_GB_s": round(L * N * K * 2 / 1e9 / p10 * 1e3, 1)}))


if __name__ == "__main__":
    main()
