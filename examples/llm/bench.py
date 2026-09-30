# Copyright 2026 The metal-pjrt-plugin Authors
# SPDX-License-Identifier: Apache-2.0
"""Prefill and decode throughput of the Qwen3 example on the current backend.

  JAX_PLATFORMS=mtl,cpu .venv/bin/python examples/llm/bench.py

(prefix it with `scripts/device_lock.py --` when other GPU jobs may run).

Prints one JSON row per case (bench/common.py's format, with commit and
versions), with the median and p10 / p90 of `--iters` timed runs after
warmup; compile time is reported separately and never part of a timing
(for prompts over generate.PREFILL_CHUNK tokens, which run as several
chunk programs, `first_call_s` is the first call's time, compile included).
Every timed region ends with `jax.block_until_ready` on all outputs.

  prefill_T     prompt of T random tokens -> first token on the host: time
                to first token without compile (includes the prompt upload
                and the KV cache allocation, both inside the jitted prefill).
  decode        `--steps` tokens with generate.py's loop: one dispatch per
                token, the next step dispatched before the previous token is
                read back (what streaming generation does). ms is per token.
  decode_sync   the same without the lookahead: dispatch, then read back.
                The difference to `decode` is the host round trip.
  decode_fused  all steps in one jitted while loop: the device-only cost.
With --batch B, tok_s counts all B sequences.

Decode starts after a `--context`-token prompt (prefilled outside the
timing). Attention reads a window of the `--max-len` cache that covers the
context (generate.Engine.window).
"""
import argparse, os, sys, time

import jax
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "bench"))
import common  # noqa: E402
import generate  # noqa: E402


def timed(fn, *, warmup, iters, setup=lambda: None):
    """Wall times in ms of fn(setup()); setup is not timed."""
    for _ in range(warmup):
        jax.block_until_ready(fn(setup()))
    ts = []
    for _ in range(iters):
        arg = setup()
        jax.block_until_ready(arg)
        t0 = time.perf_counter()
        jax.block_until_ready(fn(arg))
        ts.append((time.perf_counter() - t0) * 1e3)
    return np.array(ts)


def compile_s(jitted, *args):
    t0 = time.perf_counter()
    jitted.lower(*args).compile()
    return round(time.perf_counter() - t0, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--prompts", default="16,128,512")
    ap.add_argument("--context", type=int, default=128)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--quant", choices=["int8", "int4"], default=None)
    ap.add_argument("--batch", type=int, default=1,
                    help="sequences decoded together (the same prompt in each row)")
    ap.add_argument("--cases", default="prefill,decode,decode_sync,decode_fused")
    args = ap.parse_args()
    cases = set(args.cases.split(","))

    eng = generate.Engine(args.model, max_len=args.max_len, quant=args.quant,
                          batch=args.batch)
    backend = jax.devices()[0].platform
    rng = np.random.default_rng(0)
    key = jax.random.key(0)
    extra = {"model": args.model, "max_len": args.max_len, "quant": args.quant or "bf16"}
    if args.batch > 1:
        extra["batch"] = args.batch

    def emit(case, ts, scale=1.0, **more):
        ts = ts / scale
        p10, med, p90 = np.percentile(ts, [10, 50, 90])
        common.emit(backend, case, med, ts.min(),
                    {**extra, "p10_ms": round(p10, 3), "p90_ms": round(p90, 3), **more})

    if "prefill" in cases:
        for T in map(int, args.prompts.split(",")):
            if T > args.max_len:
                continue
            ids = rng.integers(0, 150000, T).tolist()
            tokens = np.zeros((args.batch, generate.bucket(T)), np.int32)
            fn = lambda _: int(np.asarray(eng.prefill(ids, key)[0])[0])
            if T <= generate.PREFILL_CHUNK:
                c = {"compile_s": compile_s(eng._prefill, eng.params, tokens, np.int32(T), key)}
            else:
                t0 = time.perf_counter()
                fn(None)
                c = {"first_call_s": round(time.perf_counter() - t0, 2)}
            ts = timed(fn, warmup=2, iters=args.iters)
            emit(f"prefill_{T}", ts, tok_s=round(args.batch * T / np.median(ts) * 1e3, 1), **c)

    S = args.steps
    ids = rng.integers(0, 150000, args.context).tolist()
    fresh = lambda: eng.prefill(ids, key)   # the cache is donated: new state per run

    n0 = len(ids)

    def loop(state, lookahead):
        nxt = eng.step(state, n0)
        for i in range(S):
            state, nxt = nxt, None
            if lookahead and i + 1 < S:
                nxt = eng.step(state, n0 + i + 1)
            int(np.asarray(state[0])[0])
            if not lookahead and i + 1 < S:
                nxt = eng.step(state, n0 + i + 1)
        return state

    def fused(state):
        out, n, state = eng.fused(state, n0, S, False)
        return out, state

    more = {"context": args.context, "steps": S}
    for case, fn in (("decode", lambda s: loop(s, True)),
                     ("decode_sync", lambda s: loop(s, False)),
                     ("decode_fused", fused)):
        if case not in cases:
            continue
        c = (compile_s(eng._fused, eng.params, *fresh(), S, eng.window(n0 + S), False)
             if case == "decode_fused"
             else compile_s(eng._step, eng.params, *fresh(), eng.window(n0 + 1)))
        ts = timed(fn, warmup=1, iters=max(3, args.iters // 2), setup=fresh)
        emit(case, ts, S, tok_s=round(args.batch * S / np.median(ts) * 1e3, 1),
             compile_s=c, **more)

    mem = jax.devices()[0].memory_stats() or {}
    if "peak_bytes_in_use" in mem:
        common.emit(backend, "peak_memory_gb", 0, 0,
                    {**extra, "value": round(mem["peak_bytes_in_use"] / 1e9, 2)})


if __name__ == "__main__":
    main()
