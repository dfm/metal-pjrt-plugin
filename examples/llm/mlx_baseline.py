# Copyright 2026 The metal-pjrt-plugin Authors
# SPDX-License-Identifier: Apache-2.0
"""The same prefill and decode measurements with mlx-lm, for comparison.

Needs mlx-lm, which is not a dependency of this repo; run it from its own
environment (it reads the same Hugging Face cache):

  uv venv ~/.venvs/mlx && uv pip install --python ~/.venvs/mlx/bin/python mlx-lm
  scripts/device_lock.py -- ~/.venvs/mlx/bin/python examples/llm/mlx_baseline.py

Prints rows shaped like bench.py's: prefill_T (one prefill of T random
tokens) and decode (per-token time over `--steps` greedy tokens after a
`--context`-token prompt, EOS suppressed so every run is the same length).
mlx-lm's KV cache grows with the context, where the JAX example's is
allocated at max_len up front.

--kl-ref ref.npz (from check.py --save-ref) scores the model on
check.py's footing instead: top-1 agreement, KL divergence from the float32
reference and perplexity, teacher-forced in chunks of 128 tokens.
"""
import argparse, json, statistics, time

import mlx.core as mx
import mlx_lm
import numpy as np
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache

PREFILL_STEP = 2048   # mlx_lm.generate's default prefill_step_size


def median_best(fn, warmup=2, iters=5):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts), min(ts)


def kl_report(model, path, chunk=128):
    d = np.load(path)
    ids, lr = d["ids"], d["ref"].astype(np.float32)
    cache = make_prompt_cache(model)
    lps = []
    for i in range(0, len(ids), chunk):
        logits = model(mx.array(ids[i:i + chunk])[None], cache=cache)[0].astype(mx.float32)
        lps.append(np.array(logits - mx.logsumexp(logits, axis=-1, keepdims=True)))
    lp = np.concatenate(lps)
    kl = np.sum(np.exp(lr) * (lr - lp), -1, dtype=np.float64)
    rows = np.arange(len(ids) - 1)
    ppl = float(np.exp(-lp[rows, ids[1:]].mean()))
    ppl_ref = float(np.exp(-lr[rows, ids[1:]].mean()))
    return {"case": "quality", "tokens": len(ids),
            "top1_agree": round(float(np.mean(lp.argmax(-1) == lr.argmax(-1))), 3),
            "kl_mean": float(f"{kl.mean():.3g}"), "kl_p99": float(f"{np.percentile(kl, 99):.3g}"),
            "kl_max": float(f"{kl.max():.3g}"), "ppl": round(ppl, 3), "ppl_ref": round(ppl_ref, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--prompts", default="16,128,512")
    ap.add_argument("--context", type=int, default=128)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--q-bits", type=int, default=0,
                    help="quantize in memory (mlx.nn.quantize, group 64; 0: bf16)")
    ap.add_argument("--kl-ref", help="score against check.py --save-ref output instead")
    args = ap.parse_args()
    model, tok = mlx_lm.load(args.model)
    if args.q_bits:
        import mlx.nn as nn
        nn.quantize(model, group_size=64, bits=args.q_bits)
        mx.eval(model.parameters())
    rng = np.random.default_rng(0)
    info = {"backend": "mlx-lm", "model": args.model,
            "quant": f"q{args.q_bits}" if args.q_bits else "bf16", "mlx": mx.__version__,
            "mlx_lm": mlx_lm.__version__}
    if args.kl_ref:
        print(json.dumps({**info, **kl_report(model, args.kl_ref)}))
        return

    for T in map(int, args.prompts.split(",")):
        ids = mx.array(rng.integers(0, 150000, T))[None]

        def prefill():
            # In chunks of PREFILL_STEP, as mlx-lm's generate does.
            cache = make_prompt_cache(model)
            for i in range(0, T - PREFILL_STEP, PREFILL_STEP):
                model(ids[:, i:i + PREFILL_STEP], cache=cache)
                mx.eval([c.state for c in cache])
            start = max(0, (T - 1) // PREFILL_STEP * PREFILL_STEP)
            mx.eval(mx.argmax(model(ids[:, start:], cache=cache)[:, -1], -1))

        ms, best = median_best(prefill)
        print(json.dumps({**info, "case": f"prefill_{T}", "ms": round(ms, 3),
                          "best_ms": round(best, 3), "tok_s": round(T / best * 1e3, 1)}))

    eos = list(tok.eos_token_ids)
    no_eos = lambda toks, logits: logits.at[:, eos].add(-float("inf"))
    prompt = mx.array(rng.integers(0, 150000, args.context))

    def decode():
        # generate_step prefills, then pipelines decode with mx.async_eval.
        gen = generate_step(prompt, model, max_tokens=args.steps + 1,
                            logits_processors=[no_eos])
        next(gen)                     # first token: the prefill
        t0 = time.perf_counter()
        n = sum(1 for _ in gen)
        return (time.perf_counter() - t0) * 1e3 / n

    times = [decode() for _ in range(4)][1:]
    ms, best = statistics.median(times), min(times)
    print(json.dumps({**info, "case": "decode", "ms": round(ms, 3), "best_ms": round(best, 3),
                      "context": args.context, "steps": args.steps,
                      "tok_s": round(1e3 / best, 1)}))
    print(json.dumps({**info, "case": "peak_memory_gb",
                      "value": round(mx.get_peak_memory() / 1e9, 2)}))


if __name__ == "__main__":
    main()
