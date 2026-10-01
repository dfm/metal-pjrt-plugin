# Copyright 2026 The metal-pjrt-plugin Authors
# SPDX-License-Identifier: Apache-2.0
"""The airbench94 comparison protocol: JAX (mtl) against PyTorch on MPS.

  .venv/bin/python examples/cifar/compare.py --torch-python ~/.venvs/torch/bin/python \\
      [--sampler PATH/gpuf] [--wrap "scripts/device_lock.py --"] [--runs 5]

1. One untimed full-length warm-up run (JAX), so that the timed runs start
   at the GPU's steady state: a fanless Mac's GPU throttles after ~3
   minutes of load, and the first run would otherwise get the fast minutes.
2. Then JAX and PyTorch alternate, J T J T ..., `--runs` each, seeds 1..N,
   one process per run (each does its own one-epoch warm-up, untimed).
   With --one-process, a second phase follows: JAX's --runs runs in one
   process, then PyTorch's in one process, which shows run-over-run
   accumulation (memory, caches) that fresh processes hide.
3. With `--sampler`, a GPU sampler runs for the whole session (every
   `--sample-s` seconds a line "unix_time gpuW=.. cpuW=.. active=..%
   meanP=.. | states"); each run's rows get the mean GPU power and
   P-state over its timed part.

Summary (per phase): each side's mean and spread ((max - min) / min) of
the run times, and the ratio of the means. Raw rows go to stdout as JSON
lines, the table at the end.
"""
import argparse, json, os, re, shlex, statistics, subprocess, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def run(cmd, env=None):
    """Runs cmd; returns (rows, returncode, stderr tail, t0). Each JSON row
    printed on stdout gets "_t", the time it arrived, so that runs within
    one process can be matched to the sampler."""
    t0 = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                         env=env, cwd=ROOT)
    rows = []
    for line in p.stdout:
        if line.startswith("{"):
            rows.append({**json.loads(line), "_t": time.time()})
    err = p.stderr.read()
    p.wait()
    return rows, p.returncode, err.strip().splitlines()[-1:], t0


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
    ap.add_argument("--one-process", action="store_true",
                    help="after the alternating phase, a second phase: JAX's --runs runs in "
                         "one process, then PyTorch's in one process (shows run-over-run "
                         "accumulation that fresh processes hide)")
    ap.add_argument("--jax-variants", default="jax=",
                    help="JAX arms as NAME=EXTRA_ARGS pairs separated by ';', each run in "
                         "turn with airbench.py's extra arguments, e.g. "
                         "'noremat=;remat=--remat' (default: one arm, 'jax')")
    ap.add_argument("--log", default=os.path.join(tempfile.gettempdir(), "airbench_gpu_samples.txt"),
                    help="where the sampler writes (default: a temporary file)")
    args = ap.parse_args()
    wrap = shlex.split(args.wrap)
    env = {**os.environ, "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS", "mtl,cpu")}
    tenv = {**os.environ, "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.8",
            "PYTORCH_MPS_LOW_WATERMARK_RATIO": "0.6"}
    extra = ["--epochs", str(args.epochs)] if args.epochs else []
    variants = [v.split("=", 1) for v in args.jax_variants.split(";") if v]
    jax_cmd = lambda seed, n=1, more=(): wrap + [sys.executable, os.path.join(HERE, "airbench.py"),
                                                 "--runs", str(n), "--seed", str(seed)] + extra + list(more)
    torch_cmd = lambda seed, n=1: wrap + [args.torch_python,
                                          os.path.join(HERE, "torch_baseline.py"),
                                          "--runs", str(n), "--seed", str(seed)] + extra

    def record(name, phase, rs, rc, err):
        res = [r for r in rs if "seed" in r and "acc" in r]
        if rc or not res:
            res_rows = [{"impl": name, "phase": phase, "error": err}]
        else:
            # A run's timed part ends eval_s before its row was printed.
            res_rows = [{"impl": name, "phase": phase, "seed": r["seed"], "acc": r["acc"],
                         "train_s": r["train_s"],
                         **sampler_window(args.log if sampler else None,
                                          r["_t"] - r["eval_s"] - r["train_s"],
                                          r["_t"] - r["eval_s"])} for r in res]
        for row in res_rows:
            rows.append(row)
            print(json.dumps(row), flush=True)

    sampler = None
    if args.sampler:
        sampler = subprocess.Popen([args.sampler, str(args.sample_s)], stdout=open(args.log, "w"),
                                   stderr=subprocess.STDOUT)
    rows = []
    try:
        _, rc, err, t0 = run(jax_cmd(0), env)
        if rc:
            sys.exit(f"warm-up run failed: {err}")
        print(json.dumps({"warmup_run": True, "wall_s": round(time.time() - t0, 1)}), flush=True)
        arms = [(name, lambda seed, n=1, more=tuple(shlex.split(v)): jax_cmd(seed, n, more), env)
                for name, v in variants]
        arms += [("torch", torch_cmd, tenv)] if args.torch_python else []
        for seed in range(1, args.runs + 1):
            for name, cmd, e in arms:
                record(name, "alternating", *run(cmd(seed), e)[:3])
        if args.one_process:
            for name, cmd, e in arms:
                record(name, "one-process", *run(cmd(1, args.runs), e)[:3])
    finally:
        if sampler:
            sampler.terminate()

    print()
    print("| phase | impl | seed | train s | acc | GPU W | mean P-state |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['phase']} | {r['impl']} | {r.get('seed', '')} | {r.get('train_s', 'error')} "
              f"| {r.get('acc', '')} | {r.get('gpu_w', '')} | {r.get('mean_pstate', '')} |")
    print()
    names = [n for n, _ in variants] + ["torch"]
    spread = lambda v: round((max(v) - min(v)) / min(v), 3)
    for phase in ("alternating", "one-process"):
        t = {k: [r["train_s"] for r in rows if r["impl"] == k and r["phase"] == phase
                 and "train_s" in r] for k in names}
        summ = {"phase": phase}
        for k, v in t.items():
            if v:
                summ[f"{k}_mean_s"] = round(statistics.mean(v), 1)
                summ[f"{k}_spread"] = spread(v)
                if t["torch"] and k != "torch":
                    summ[f"{k}_over_torch"] = round(statistics.mean(v) / statistics.mean(t["torch"]), 3)
        if len(summ) > 1:
            print(json.dumps({"summary": True, **summ}))


if __name__ == "__main__":
    main()
