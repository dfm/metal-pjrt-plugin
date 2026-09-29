"""Interleaved rounds of train.py and mlx_baseline.py; prints a markdown table.

  .venv/bin/python examples/lora/compare.py --data ~/.cache/metal-pjrt-examples/wikisql \\
      --mlx-python ~/.venvs/mlx/bin/python --wrap "scripts/device_lock.py --"

Each round runs a short fine-tune (`--iters` steps, no validation after
the first) with each implementation in turn, so drift of the machine's
state (other load, heat) hits both alike. Per run, the step time is the
median over the report windows after the first (which holds compilation
for JAX, kernel builds for MLX); the table shows the median over rounds,
the trained tokens/s, and mlx-lm's peak Metal memory. --wrap prefixes each
run's command, e.g. with the device lock.
"""
import argparse, json, os, shlex, statistics, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def run(cmd, env=None):
    out = subprocess.run(cmd, capture_output=True, text=True, env=env)
    rows = [json.loads(l) for l in out.stdout.splitlines() if l.startswith("{")]
    train = [r for r in rows if r.get("event") == "train"][1:]
    if out.returncode or not train:
        return None, out.stderr[-600:]
    return train, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="lora.py export directory")
    ap.add_argument("--mlx-python", required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--wrap", default="")
    args = ap.parse_args()
    wrap = shlex.split(args.wrap)
    common = ["--iters", str(args.iters), "--steps-per-report", "10",
              "--steps-per-eval", str(10 ** 6)] + (["--grad-checkpoint"] * args.grad_checkpoint)
    arms = {
        "jax": (wrap + [sys.executable, os.path.join(HERE, "train.py"), *common],
                {**os.environ, "JAX_PLATFORMS": "mtl,cpu"}),
        "mlx-lm": (wrap + [args.mlx_python, os.path.join(HERE, "mlx_baseline.py"),
                           "--data", args.data, *common], None),
    }
    res = {k: {"ms": [], "tok_s": [], "mem": []} for k in arms}
    for r in range(args.rounds):
        for name, (cmd, env) in arms.items():
            train, err = run(cmd, env)
            if train is None:
                print(f"round {r + 1}: {name} failed: {err}", file=sys.stderr)
                continue
            res[name]["ms"].append(statistics.median(t["step_ms"] for t in train))
            res[name]["tok_s"].append(statistics.median(t["tokens_s"] for t in train))
            res[name]["mem"] += [t["peak_mem_gb"] for t in train if "peak_mem_gb" in t]
            print(f"round {r + 1}: {name} {res[name]['ms'][-1]:.0f} ms/step", file=sys.stderr,
                  flush=True)
    med = lambda v: f"{statistics.median(v):.0f}" if v else "failed"
    print(f"Qwen3-0.6B LoRA on WikiSQL, batch 4, {args.iters} steps per run, "
          f"grad checkpoint {'on' if args.grad_checkpoint else 'off'}, "
          f"median of {args.rounds} interleaved rounds\n")
    print("| | step (ms) | trained tokens/s | peak Metal memory (GB) |")
    print("|---|---|---|---|")
    for name, v in res.items():
        mem = f"{max(v['mem']):.2f}" if v["mem"] else ""
        print(f"| {name} | {med(v['ms'])} | {med(v['tok_s'])} | {mem} |")


if __name__ == "__main__":
    main()
