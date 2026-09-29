"""The same fine-tune with mlx_lm.lora, for comparison.

Needs mlx-lm, which is not a dependency of this repo; run it from its own
environment. The data comes from lora.py (this repo's environment has the
parquet reader):

  .venv/bin/python examples/lora/lora.py export ~/.cache/metal-pjrt-examples/wikisql
  scripts/device_lock.py -- ~/.venvs/mlx/bin/python examples/lora/mlx_baseline.py \\
      --data ~/.cache/metal-pjrt-examples/wikisql --iters 200

Runs `python -m mlx_lm lora` with train.py's settings (a YAML config:
AdamW with bias correction as optax's, --mask-prompt, rank 8, scale 20,
the last 16 layers, batch 4, the whole validation set per evaluation) and
turns its log into train.py's JSON lines. mlx-lm reports steps/s and
tokens/s averaged over each report window, and its peak Metal memory.
"""
import argparse, json, os, re, subprocess, sys, tempfile

TRAIN = re.compile(r"Iter (\d+): Train loss ([\d.]+), Learning Rate [\d.e+-]+, "
                   r"It/sec ([\d.]+), Tokens/sec ([\d.]+), Trained Tokens \d+, "
                   r"Peak mem ([\d.]+) GB")
VAL = re.compile(r"Iter (\d+): Val loss ([\d.]+), Val took ([\d.]+)s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--learning-rate", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--scale", type=float, default=20.0)
    ap.add_argument("--num-layers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps-per-report", type=int, default=10)
    ap.add_argument("--steps-per-eval", type=int, default=50)
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--adapter-path", default=None)
    ap.add_argument("--test", type=int, default=0, help="exact match on N test questions")
    args = ap.parse_args()

    import yaml
    work = tempfile.mkdtemp(prefix="mlx_lora_")
    config = {
        "model": args.model, "train": True, "fine_tune_type": "lora", "data": args.data,
        "seed": args.seed, "num_layers": args.num_layers, "batch_size": args.batch_size,
        "iters": args.iters, "val_batches": -1, "learning_rate": args.learning_rate,
        "steps_per_report": args.steps_per_report, "steps_per_eval": args.steps_per_eval,
        "adapter_path": args.adapter_path or os.path.join(work, "adapters"),
        "save_every": 10 ** 9, "max_seq_length": 2048, "mask_prompt": True,
        "grad_checkpoint": args.grad_checkpoint, "optimizer": "adamw",
        "optimizer_config": {"adamw": {"weight_decay": args.weight_decay,
                                       "bias_correction": True}},
        "lora_parameters": {"rank": args.rank, "dropout": 0.0, "scale": args.scale},
    }
    path = os.path.join(work, "config.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(config, f)
    proc = subprocess.Popen([sys.executable, "-m", "mlx_lm", "lora", "-c", path],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    emit = lambda **kw: print(json.dumps({"backend": "mlx-lm", **kw}), flush=True)
    log = []
    for line in proc.stdout:
        log.append(line)
        if m := TRAIN.search(line):
            it, loss, its, toks, mem = m.groups()
            emit(event="train", iter=int(it), train_loss=float(loss),
                 step_ms=round(1e3 / float(its), 2), it_s=float(its), tokens_s=float(toks),
                 peak_mem_gb=float(mem))
        elif m := VAL.search(line):
            it, loss, secs = m.groups()
            emit(event="val", iter=int(it), val_loss=float(loss), val_s=float(secs))
    if proc.wait():
        sys.exit("mlx_lm lora failed:\n" + "".join(log[-30:]))
    if args.test:
        emit(event="test", model="lora",
             exact_match=exact_match(args.model, config["adapter_path"], args.data, args.test))


def exact_match(model_name, adapter_path, data, n, max_new=96):
    """Greedy SQL for the first n test questions, prompted as train.py's
    test does (chat template, thinking off), compared with the reference."""
    import mlx_lm
    model, tok = mlx_lm.load(model_name, adapter_path=adapter_path)
    with open(os.path.join(data, "test.jsonl")) as f:
        records = [json.loads(l) for l in f][:n]
    hits = 0
    for r in records:
        ids = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}],
                                      add_generation_prompt=True, enable_thinking=False)
        out = mlx_lm.generate(model, tok, prompt=ids, max_tokens=max_new)
        hits += out.strip() == r["completion"].strip()
    return hits / n


if __name__ == "__main__":
    main()
