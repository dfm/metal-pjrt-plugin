# Copyright 2026 The metal-pjrt-plugin Authors
# SPDX-License-Identifier: Apache-2.0
"""Interleaved rounds of bench.py and mlx_baseline.py; prints a markdown table.

  .venv/bin/python examples/llm/compare.py --wrap "scripts/device_lock.py --" \\
      --mlx-python /path/to/env-with-mlx-lm/bin/python

Each round runs every (implementation, precision) arm once, in turn, so
slow drift of the machine (thermal state, other load) hits every arm alike;
the table shows the median over rounds. Each arm is its own process;
--wrap prefixes each arm's command, e.g. with the device lock (taken per
arm rather than for the whole comparison).
"""
import argparse, collections, json, os, shlex, statistics, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
QUANTS = {"bf16": ([], ["--q-bits", "0"]),
          "int8": (["--quant", "int8"], ["--q-bits", "8"]),
          "int4": (["--quant", "int4"], ["--q-bits", "4"])}


def run(cmd, env=None):
    out = subprocess.run(cmd, capture_output=True, text=True, env=env)
    rows = [json.loads(l) for l in out.stdout.splitlines() if l.startswith("{")]
    if out.returncode or not rows:
        sys.exit(f"failed: {' '.join(cmd)}\n{out.stderr[-2000:]}")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mlx-python", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--prompts", default="16,128,512")
    ap.add_argument("--quants", default="bf16,int8,int4")
    ap.add_argument("--wrap", default="", help="command prefix for each arm")
    args = ap.parse_args()
    lock = shlex.split(args.wrap)
    env = None
    common = ["--model", args.model, "--prompts", args.prompts]
    ms = collections.defaultdict(list)       # (impl, quant, case) -> [ms]
    for r in range(args.rounds):
        for q in args.quants.split(","):
            jflags, mflags = QUANTS[q]
            for impl, cmd, e in (
                    ("jax", [sys.executable, os.path.join(HERE, "bench.py"), *common, *jflags,
                             "--cases", "prefill,decode,decode_fused"], env),
                    ("mlx-lm", [args.mlx_python, os.path.join(HERE, "mlx_baseline.py"),
                                *common, *mflags], None)):
                for row in run(lock + cmd, e):
                    if not row["case"].startswith("peak_"):
                        ms[impl, q, row["case"]].append(row["ms"])
                print(f"round {r + 1}: {impl} {q} done", file=sys.stderr, flush=True)

    cases = [f"prefill_{t}" for t in args.prompts.split(",")] + ["decode", "decode_fused"]
    print(f"{args.model}, median of {args.rounds} interleaved rounds; "
          "prefill: ms (tokens/s), decode: ms/token (tokens/s)\n")
    print("| case | " + " | ".join(f"{i} {q}" for q in args.quants.split(",")
                                    for i in ("jax", "mlx-lm")) + " |")
    print("|---" * (1 + 2 * len(args.quants.split(","))) + "|")
    for c in cases:
        cells = []
        for q in args.quants.split(","):
            for impl in ("jax", "mlx-lm"):
                v = ms.get((impl, q, c))
                if not v:
                    cells.append("")
                    continue
                m = statistics.median(v)
                n = int(c.split("_")[1]) if c.startswith("prefill") else 1
                cells.append(f"{m:.1f} ({n / m * 1e3:.0f})")
        print(f"| {c} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
