"""The airbench94 comparison protocol: JAX (mtl) against PyTorch on MPS.

  .venv/bin/python examples/cifar/compare.py --torch-python ~/.venvs/torch/bin/python \\
      [--sampler PATH/gpuf] [--wrap "scripts/device_lock.py --"] [--runs 5]

1. One untimed full-length warm-up run (JAX), so that the timed runs start
   at the GPU's steady state: a fanless Mac's GPU throttles after ~3
   minutes of load, and the first run would otherwise get the fast minutes.
2. Then JAX and PyTorch alternate, J T J T ..., `--runs` each, seeds 1..N,
   one process per run (each does its own one-epoch warm-up, untimed).
3. With `--sampler`, a GPU sampler runs for the whole session (every
   `--sample-s` seconds a line "unix_time gpuW=.. cpuW=.. active=..%
   meanP=.. | states"); each run's rows get the mean GPU power and
   P-state over its timed part.

Acceptance (printed): the JAX runs' times within ~3% of each other
((max - min) / min), and the JAX mean below the PyTorch mean of the same
session. Raw rows go to stdout as JSON lines, the table at the end.
"""
import argparse, json, os, re, shlex, statistics, subprocess, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def run(cmd, env=None):
    t0 = time.time()
    out = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=ROOT)
    rows = [json.loads(l) for l in out.stdout.splitlines() if l.startswith("{")]
    return rows, out, t0, time.time()


def sampler_window(path, t0, t1):
    """Mean gpuW and meanP of the sampler lines stamped within [t0, t1]."""
    if not path or not os.path.exists(path):
        return {}
    w, p = [], []
    for line in open(path):
        m = re.match(r"(\d+) gpuW=([\d.]+).*meanP=([\d.]+)", line)
        if m and t0 <= int(m.group(1)) <= t1:
            w.append(float(m.group(2)))
            p.append(float(m.group(3)))
    return {"gpu_w": round(statistics.mean(w), 2), "mean_pstate": round(statistics.mean(p), 2)} if w else {}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--torch-python", help="python of an environment with torch "
                    "(omit to run JAX only, e.g. to test the harness)")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--epochs", type=float, default=None, help="passed to both (testing)")
    ap.add_argument("--sampler", help="GPU sampler binary (e.g. an IOReport P-state sampler)")
    ap.add_argument("--sample-s", type=int, default=10)
    ap.add_argument("--wrap", default="", help="command prefix for each run, e.g. the device lock")
    ap.add_argument("--log", default=os.path.join(tempfile.gettempdir(), "airbench_gpu_samples.txt"),
                    help="where the sampler writes (default: a temporary file)")
    args = ap.parse_args()
    wrap = shlex.split(args.wrap)
    env = {**os.environ, "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS", "mtl,cpu")}
    tenv = {**os.environ, "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.8",
            "PYTORCH_MPS_LOW_WATERMARK_RATIO": "0.6"}
    extra = ["--epochs", str(args.epochs)] if args.epochs else []
    jax_cmd = lambda seed: wrap + [sys.executable, os.path.join(HERE, "airbench.py"),
                                   "--runs", "1", "--seed", str(seed)] + extra
    torch_cmd = lambda seed: wrap + [args.torch_python, os.path.join(HERE, "torch_baseline.py"),
                                     "--runs", "1", "--seed", str(seed)] + extra

    sampler = None
    if args.sampler:
        sampler = subprocess.Popen([args.sampler, str(args.sample_s)], stdout=open(args.log, "w"),
                                   stderr=subprocess.STDOUT)
    rows = []
    try:
        rows_w, out, t0, t1 = run(jax_cmd(0), env)
        if out.returncode:
            sys.exit(f"warm-up run failed:\n{out.stderr[-2000:]}")
        print(json.dumps({"warmup_run": True, "wall_s": round(t1 - t0, 1)}), flush=True)
        arms = [("jax", jax_cmd, env)] + ([("torch", torch_cmd, tenv)] if args.torch_python else [])
        for seed in range(1, args.runs + 1):
            for name, cmd, e in arms:
                rs, out, t0, t1 = run(cmd(seed), e)
                res = [r for r in rs if "seed" in r and "acc" in r]
                if out.returncode or not res:
                    row = {"impl": name, "seed": seed, "error": out.stderr.strip().splitlines()[-1:]}
                else:
                    r = res[-1]
                    # The timed part is the last train_s + eval_s seconds before exit.
                    row = {"impl": name, "seed": seed, "acc": r["acc"], "train_s": r["train_s"],
                           **sampler_window(args.log if sampler else None,
                                            t1 - r["train_s"] - r["eval_s"], t1 - r["eval_s"])}
                rows.append(row)
                print(json.dumps(row), flush=True)
    finally:
        if sampler:
            sampler.terminate()

    print()
    print("| impl | seed | train s | acc | GPU W | mean P-state |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['impl']} | {r['seed']} | {r.get('train_s', 'error')} | {r.get('acc', '')} "
              f"| {r.get('gpu_w', '')} | {r.get('mean_pstate', '')} |")
    t = {k: [r["train_s"] for r in rows if r["impl"] == k and "train_s" in r] for k in ("jax", "torch")}
    if t["jax"]:
        spread = (max(t["jax"]) - min(t["jax"])) / min(t["jax"])
        summ = {"jax_mean_s": round(statistics.mean(t["jax"]), 1),
                "jax_spread": round(spread, 3), "consistent": spread <= 0.03}
        if t["torch"]:
            summ.update(torch_mean_s=round(statistics.mean(t["torch"]), 1),
                        faster=statistics.mean(t["jax"]) < statistics.mean(t["torch"]))
        print()
        print(json.dumps({"summary": True, **summ}))


if __name__ == "__main__":
    main()
