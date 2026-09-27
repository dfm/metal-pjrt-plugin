"""Shared timing helpers for the benchmark scripts."""
import datetime, functools, json, os, re, statistics, subprocess, sys, tempfile, time


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


def gpu_ms(fn, *, iters=5, sync=lambda r: None):
    """Mean GPU time in ms of fn() on the Metal plugin: the sum of its command
    buffers' GPUEndTime - GPUStartTime, read from the METAL_PJRT_TRACE=1 log
    lines (fd 2 is captured while it runs). None without the trace."""
    if os.environ.get("METAL_PJRT_TRACE", "0") in ("", "0"):
        return None
    time.sleep(0.05)  # completion handlers of earlier calls log late
    sys.stderr.flush()
    with tempfile.TemporaryFile() as tmp:
        saved = os.dup(2)
        os.dup2(tmp.fileno(), 2)
        try:
            for _ in range(iters):
                sync(fn())
            time.sleep(0.05)
        finally:
            os.dup2(saved, 2)
            os.close(saved)
        tmp.seek(0)
        text = tmp.read().decode(errors="replace")
    return sum(float(m) for m in re.findall(r"gpu_ms=([0-9.eE+-]+)", text)) / iters


@functools.cache
def run_info():
    """Commit, library versions and knobs, recorded in every result row."""
    repo = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    try:
        commit = subprocess.run(["git", "-C", repo, "describe", "--always", "--dirty"],
                                capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = "?"
    info = {"commit": commit, "date": datetime.datetime.now().isoformat(timespec="seconds")}
    for mod in ("jax", "jaxlib", "mlx.core"):
        m = sys.modules.get(mod)
        if m is not None:
            info[mod.split(".")[0]] = getattr(m, "__version__", "?")
    jax = sys.modules.get("jax")
    if jax is not None:
        info["platform_version"] = jax.devices()[0].client.platform_version
    info["knobs"] = {k: v for k, v in sorted(os.environ.items())
                     if (k.startswith(("METAL_PJRT_", "JAX_METAL_", "MLX_")) or
                         k in ("XLA_FLAGS", "JAX_PLATFORMS")) and
                     k not in ("JAX_METAL_DEVICE_LOCK_HELD", "JAX_METAL_LOCKED")}
    return info


def emit(backend, case, ms, best, extra=None, *, gpu=None, flops=None):
    rec = {"backend": backend, "case": case, "ms": round(ms, 3), "best_ms": round(best, 3)}
    if gpu is not None:
        rec["gpu_ms"] = round(gpu, 3)
    if flops:
        rec["tflops"] = round(flops / (best * 1e9), 3)  # at the best wall time
    if extra:
        rec.update(extra)
    rec.update(run_info())
    line = json.dumps(rec)
    print(line, flush=True)
    out = os.environ.get("BENCH_OUT")
    if out:
        with open(out, "a") as f:
            f.write(line + "\n")


def selected(name):
    only = os.environ.get("BENCH_ONLY")
    return (not only) or any(s in name for s in only.split(","))
