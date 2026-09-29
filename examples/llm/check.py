"""Check the Qwen3 example on the default backend against a float32 CPU run.

  JAX_PLATFORMS=mtl,cpu .venv/bin/python examples/llm/check.py [--quant int8|int4]
      [--text-file FILE --tokens N] [--save-ref ref.npz]

(prefix it with `scripts/device_lock.py --` when other GPU jobs may run).

Runs a text through the model teacher-forced and compares the next-token
distributions with the same code in float32 on CPU: top-1 agreement, KL
divergence from the reference, and the perplexity of the text under both.
The text goes through the KV cache in chunks of `--chunk` tokens (the
prefill path); a short text also goes one token at a time (the decode
path). With bf16 weights this is a correctness check (exit status 1 on a
mismatch): bf16 differs from float32 by a KL of ~1e-3, and a broken kernel
shows up as top-1 disagreement or a KL orders of magnitude larger. With
--quant it measures the quantization's cost and always exits 0.

--save-ref writes the token ids and the reference log-probabilities for
mlx_baseline.py --kl-ref, which scores mlx-lm on the same footing.
"""
import argparse, functools, os, sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate, qwen3  # noqa: E402

TEXT = ("The James Webb Space Telescope observes in the infrared, which lets it "
        "see through dust and detect light from the earliest galaxies. Its "
        "primary mirror is made of eighteen hexagonal segments")


def log_probs_chunked(params, ids, cfg, dtype, chunk):
    """[len(ids), V] float32 log-probabilities, `chunk` tokens per call."""
    n = len(ids)
    padded = -(-n // chunk) * chunk
    cache = qwen3.init_cache(cfg, 1, padded, dtype)
    f = jax.jit(lambda p, t, c, pos: (lambda l, c: (jax.nn.log_softmax(l, -1), c))(
        *qwen3.forward(p, t, c, pos, cfg, all_logits=True)))
    toks = np.zeros(padded, np.int32)
    toks[:n] = ids
    out = []
    for i in range(0, padded, chunk):
        lp, cache = f(params, toks[None, i:i + chunk], cache, np.int32(i))
        out.append(np.asarray(lp[0]))
    return np.concatenate(out)[:n]


def log_probs_stepwise(params, ids, cfg, dtype):
    cache = qwen3.init_cache(cfg, 1, generate.bucket(len(ids)), dtype)
    f = jax.jit(functools.partial(qwen3.forward, cfg=cfg))
    out = []
    for i, t in enumerate(ids):
        logits, cache = f(params, np.array([[t]], np.int32), cache, np.int32(i))
        out.append(np.asarray(jax.nn.log_softmax(logits[0], -1)))
    return np.stack(out)


def stats(lp, lr, ids):
    """Top-1 agreement, KL(ref || got) per token, perplexities of the text."""
    agree = float(np.mean(lp.argmax(-1) == lr.argmax(-1)))
    kl = np.sum(np.exp(lr) * (lr - lp), -1, dtype=np.float64)
    nxt = np.asarray(ids[1:])
    rows = np.arange(len(nxt))
    ppl = float(np.exp(-lp[rows, nxt].mean()))
    ppl_ref = float(np.exp(-lr[rows, nxt].mean()))
    return agree, kl, ppl, ppl_ref


def report(name, lp, lr, ids):
    agree, kl, ppl, ppl_ref = stats(lp, lr, ids)
    print(f"{name:>9}: top-1 agree {agree:.3f}, KL mean {kl.mean():.2e} "
          f"p99 {np.percentile(kl, 99):.2e} max {kl.max():.2e}, "
          f"perplexity {ppl:.3f} (float32 {ppl_ref:.3f})")
    return agree, kl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--quant", choices=["int8", "int4"], default=None)
    ap.add_argument("--text-file", help="evaluate on this file instead of a short sentence")
    ap.add_argument("--tokens", type=int, default=512, help="with --text-file: first N tokens")
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--save-ref", help="write ids and float32 log-probs (npz)")
    args = ap.parse_args()
    path = qwen3.model_dir(args.model)
    cfg = qwen3.Config.from_json(os.path.join(path, "config.json"))
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(path, "tokenizer.json"))
    if args.text_file:
        with open(args.text_file) as f:
            ids = tok.encode(f.read()).ids[: args.tokens]
    else:
        ids = tok.encode(TEXT).ids
    chunk = min(args.chunk, generate.bucket(len(ids)))

    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        ref = log_probs_chunked(qwen3.load_params(path, cfg, jnp.float32, cpu), ids, cfg,
                                jnp.float32, chunk)
    if args.save_ref:
        np.savez(args.save_ref, ids=np.array(ids), ref=ref.astype(np.float16))
    dev = jax.devices()[0]
    params = qwen3.load_params(path, cfg, jnp.bfloat16, dev, quant=args.quant)
    print(f"{len(ids)} tokens, {dev.platform} bfloat16"
          f"{' with ' + args.quant + ' weights' if args.quant else ''} vs cpu float32")
    results = [report(f"chunk {chunk}", log_probs_chunked(params, ids, cfg, jnp.bfloat16, chunk),
                      ref, ids)]
    if len(ids) <= 64:
        results.append(report("stepwise", log_probs_stepwise(params, ids, cfg, jnp.bfloat16),
                              ref, ids))
    if args.quant:
        return
    ok = all(agree >= 0.9 and kl.max() < 0.05 for agree, kl in results)
    print("OK" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
