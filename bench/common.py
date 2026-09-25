"""Shared timing helpers for the benchmark scripts."""
import json, os, statistics, sys, time


def timeit(fn, *, warmup=3, iters=10, sync=lambda r: None):
    """Median wall time in ms of fn(); sync(result) blocks until complete."""
    for _ in range(warmup):
        sync(fn())
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        sync(fn())
        times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times), min(times)


def emit(backend, case, ms, best, extra=None):
    rec = {"backend": backend, "case": case, "ms": round(ms, 3), "best_ms": round(best, 3)}
    if extra:
        rec.update(extra)
    line = json.dumps(rec)
    print(line, flush=True)
    out = os.environ.get("BENCH_OUT")
    if out:
        with open(out, "a") as f:
            f.write(line + "\n")


def selected(name):
    only = os.environ.get("BENCH_ONLY")
    return (not only) or any(s in name for s in only.split(","))
