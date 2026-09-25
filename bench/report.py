"""Merge BENCH_OUT jsonl files into a markdown table (ms, median)."""
import json, sys, collections
rows = collections.OrderedDict(); backends = []
for path in sys.argv[1:]:
    for line in open(path):
        r = json.loads(line)
        rows.setdefault(r["case"], {})[r["backend"]] = r
        if r["backend"] not in backends: backends.append(r["backend"])
print("| case | " + " | ".join(backends) + " |")
print("|---|" + "---|" * len(backends))
for case, by in rows.items():
    cells = []
    for b in backends:
        r = by.get(b)
        cells.append("-" if r is None else ("ERR" if r.get("error") else f"{r['ms']:.2f}"))
    print(f"| {case} | " + " | ".join(cells) + " |")
