"""LoRA fine-tuning of Qwen3 on WikiSQL (text to SQL), in pure JAX.

  scripts/device_lock.py -- env JAX_PLATFORMS=mtl,cpu \\
      .venv/bin/python examples/lora/train.py --iters 200 --save adapters.npz --test 100

Defaults are mlx_lm.lora's (Qwen3-0.6B, rank 8, scale 20, the last 16
layers, all seven projections, batch 4, learning rate 1e-5), with AdamW
(weight decay 0.01) and the loss on the reply only (mlx-lm's
--mask-prompt). Prints one JSON line per report / validation / test, like
mlx_lm.lora's log: train loss over the last `--steps-per-report` steps,
steps/s, trained tokens/s, and the median step time (steps that compiled
a new batch shape are counted apart, in `compile_s`).

--test N greedily generates the SQL for the first N test questions with
the adapters merged into the weights and reports the exact-match rate,
before (`--test-base`) and after training.
"""
import argparse, json, os, statistics, sys, time

import jax
import jax.numpy as jnp
import numpy as np
import optax

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "llm"))
import generate, lora, qwen3  # noqa: E402


def evaluate(eval_fn, adapters, params, data, batch_size, pad_to):
    total, ntok = 0.0, 0
    for i in range(0, len(data) - batch_size + 1, batch_size):
        loss, n = eval_fn(adapters, params, *lora.pad(data[i:i + batch_size], pad_to))
        total += float(loss) * int(n)
        ntok += int(n)
    return total / ntok


def exact_match(eng, records, n, max_new=96):
    hits = 0
    for r in records[:n]:
        out, _ = eng.generate(eng.encode(r["prompt"]), max_new, fused=True)
        hits += eng.tok.decode(out).strip() == r["completion"].strip()
    return hits / n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--learning-rate", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--scale", type=float, default=20.0)
    ap.add_argument("--num-layers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pad-to", type=int, default=32,
                    help="pad batches to 1 + a multiple of this (mlx-lm: 32)")
    ap.add_argument("--grad-checkpoint", action="store_true",
                    help="recompute each adapted layer in the backward pass (less memory)")
    ap.add_argument("--steps-per-report", type=int, default=10)
    ap.add_argument("--steps-per-eval", type=int, default=50)
    ap.add_argument("--save", help="write the adapters here (.npz)")
    ap.add_argument("--test", type=int, default=0, help="exact match on N test questions")
    ap.add_argument("--test-base", action="store_true", help="also test the base model")
    args = ap.parse_args()

    path = qwen3.model_dir(args.model)
    cfg = qwen3.Config.from_json(os.path.join(path, "config.json"))
    lcfg = lora.LoraConfig(rank=args.rank, scale=args.scale, num_layers=args.num_layers)
    params = qwen3.load_params(path, cfg, jnp.bfloat16)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(path, "tokenizer.json"))
    train = [lora.tokenize(tok, r) for r in lora.wikisql("train")]
    valid = [lora.tokenize(tok, r) for r in lora.wikisql("valid")]

    adapters = lora.init_adapters(jax.random.key(args.seed), cfg, lcfg)
    opt = optax.adamw(args.learning_rate, b1=0.9, b2=0.999, eps=1e-8,
                      weight_decay=args.weight_decay)
    opt_state = opt.init(adapters)
    step = lora.make_step(cfg, lcfg, opt, args.grad_checkpoint)
    eval_fn = lora.make_eval(cfg, lcfg)
    n_params = sum(x.size for x in jax.tree.leaves(adapters))
    emit = lambda **kw: print(json.dumps({"backend": jax.devices()[0].platform, **kw}),
                              flush=True)
    emit(event="start", trainable_params=n_params, iters=args.iters)

    seen = set()
    losses, ntoks, times, compile_s = [], [], [], 0.0
    t_train = time.perf_counter()
    epochs = -(-args.iters * args.batch_size // len(train))
    for it, (tokens, offsets, lengths) in enumerate(
            lora.batches(train, args.batch_size, args.seed, epochs, args.pad_to), start=1):
        if it > args.iters:
            break
        if it == 1 or it % args.steps_per_eval == 0 or it == args.iters:
            t = time.perf_counter()
            val = evaluate(eval_fn, adapters, params, valid, args.batch_size, args.pad_to)
            emit(event="val", iter=it, val_loss=round(val, 4),
                 val_s=round(time.perf_counter() - t, 2))
            t_train += time.perf_counter() - t
        t0 = time.perf_counter()
        adapters, opt_state, loss, n = step(adapters, opt_state, params, tokens,
                                            offsets, lengths)
        loss, n = float(loss), int(n)        # syncs, as mlx_lm.lora does
        dt = time.perf_counter() - t0
        if tokens.shape[1] in seen:
            times.append(dt)
        else:
            seen.add(tokens.shape[1])
            compile_s += dt
        losses.append(loss)
        ntoks.append(n)
        if it % args.steps_per_report == 0 or it == args.iters:
            w = times[-args.steps_per_report:] or [dt]
            emit(event="train", iter=it,
                 train_loss=round(statistics.fmean(losses[-args.steps_per_report:]), 4),
                 step_ms=round(statistics.median(w) * 1e3, 2),
                 it_s=round(1 / statistics.fmean(w), 2),
                 tokens_s=round(sum(ntoks[-len(w):]) / sum(w), 1))
    total = time.perf_counter() - t_train
    emit(event="done", iters=args.iters, wall_s=round(total, 1), compile_s=round(compile_s, 1),
         step_ms_median=round(statistics.median(times) * 1e3, 2) if times else None,
         shapes=sorted(seen))
    if args.save:
        lora.save(args.save, adapters, lcfg)

    if args.test:
        del params, step, eval_fn, opt_state   # the Engine loads its own copy
        records = lora.wikisql("test")
        eng = generate.Engine(args.model, max_len=512)
        if args.test_base:
            emit(event="test", model="base", exact_match=exact_match(eng, records, args.test))
        eng.params = lora.merge(eng.params, adapters, cfg, lcfg)
        emit(event="test", model="lora", exact_match=exact_match(eng, records, args.test))


if __name__ == "__main__":
    main()
