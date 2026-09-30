# Copyright 2026 The metal-pjrt-plugin Authors
# SPDX-License-Identifier: Apache-2.0
"""skinny_jax.py's measurement in MLX, for comparison.

Needs MLX (not a dependency of this repo; see mlx_baseline.py):

  ~/.venvs/mlx/bin/python examples/llm/skinny_mlx.py

(prefix it with `scripts/device_lock.py --` when other GPU jobs may run).
Same GEMMs, modes and statistics as skinny_jax.py; each call ends with
mx.eval, MLX's equivalent of block_until_ready.
"""
import argparse, json, time

import mlx.core as mx
import numpy as np

L, K, N = 28, 1024, 6144


def time_ms(f, bursts, calls):
    ts = []
    for _ in range(bursts):
        for _ in range(calls):
            t0 = time.perf_counter()
            f()
            ts.append((time.perf_counter() - t0) * 1e3)
    return ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="1,2,4,8,16,32,64")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--bursts", type=int, default=3)
    ap.add_argument("--calls", type=int, default=10)
    args = ap.parse_args()
    mx.random.seed(0)
    Ws = [(mx.random.normal((N, K)) * 0.02).astype(mx.bfloat16) for _ in range(L)]
    mx.eval(Ws)
    rows = [int(m) for m in args.rows.split(",")]
    xs = {m: mx.ones((m, K), mx.bfloat16) * 0.01 for m in rows}

    def plain(x):
        mx.eval([x @ W.T for W in Ws])

    def chained(x):
        for W in Ws:
            x = (x @ W.T)[:, :K]
        mx.eval(x)

    fns = {"list": plain, "chained": chained}
    for m in rows:
        for f in fns.values():
            time_ms(lambda: f(xs[m]), 1, 3)
    ts = {(m, k): [] for m in rows for k in fns}
    for _ in range(args.rounds):
        for m in rows:
            for k, f in fns.items():
                ts[m, k] += time_ms(lambda: f(xs[m]), args.bursts, args.calls)
    for (m, k), t in ts.items():
        p10 = float(np.percentile(t, 10))
        print(json.dumps({"impl": "mlx", "M": m, "mode": k, "p10_ms": round(p10, 2),
                          "median_ms": round(float(np.median(t)), 2),
                          "weight_GB_s": round(L * N * K * 2 / 1e9 / p10 * 1e3, 1)}))


if __name__ == "__main__":
    main()
