# Copyright 2026 The metal-pjrt-plugin Authors
# SPDX-License-Identifier: Apache-2.0

"""Merge BENCH_OUT jsonl files into a markdown table: per backend the median
over rows (rounds) of each case's median wall ms; for metal also the GPU ms
(from the "metal-gpu" rows, a separate METAL_PJRT_TRACE=1 pass, since the
trace adds ~0.1 ms to sub-ms wall times) and TFLOPS at the best wall time
where the case has a flop count. The
header lists the commits, versions and knobs the rows were taken with."""
import collections, json, statistics, sys

rows = collections.defaultdict(lambda: collections.defaultdict(list))
cases, backends, info = [], [], collections.defaultdict(set)
for path in sys.argv[1:]:
    for line in open(path):
        r = json.loads(line)
        if r["case"] not in cases: cases.append(r["case"])
        if r["backend"] not in backends and r["backend"] != "metal-gpu":
            backends.append(r["backend"])
        rows[r["case"]][r["backend"]].append(r)
        for k in ("commit", "jax", "mlx", "platform_version"):
            if k in r: info[f"{r['backend']} {k}"].add(str(r[k]))
        if r.get("knobs"): info[f"{r['backend']} knobs"].add(json.dumps(r["knobs"], sort_keys=True))


def med(rs, key):
    v = [r[key] for r in rs if key in r and not r.get("error")]
    return statistics.median(v) if v else None


for b in backends:
    if not any(k.startswith(b + " ") for k in info):
        print(f"- {b}: no commit/version metadata (a run from before 2026-09-27)")
for k in sorted(info):
    print(f"- {k}: {', '.join(sorted(info[k]))}")
nrounds = max(len(by[b]) for by in rows.values() for b in by)
print(f"- medians of {nrounds} round(s); wall ms unless noted\n")
extra = ["metal GPU ms", "metal TFLOPS"]
print("| case | " + " | ".join(backends + extra) + " |")
print("|---|" + "---|" * (len(backends) + len(extra)))
for case in cases:
    by = rows[case]
    cells = []
    for b in backends:
        rs = by.get(b)
        m = med(rs, "ms") if rs else None
        cells.append("-" if not rs else ("ERR" if m is None else f"{m:.2f}"))
    g, t = med(by.get("metal-gpu", []), "gpu_ms"), med(by.get("metal", []), "tflops")
    cells += ["-" if g is None else f"{g:.2f}", "-" if t is None else f"{t:.2f}"]
    print(f"| {case} | " + " | ".join(cells) + " |")
