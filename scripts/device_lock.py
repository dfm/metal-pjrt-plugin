#!/usr/bin/env python3
"""Run a command while holding the Metal device lock.

Only one GPU-heavy process (test suite, benchmark, sweep) should run at a
time on an 8 GB unified-memory machine: concurrent runs over-commit memory
and, under thrashing, GPU command buffers hit the watchdog. Usage:

  scripts/device_lock.py -- <command> [args...]
"""
import fcntl, os, pathlib, subprocess, sys, time

LOCK = pathlib.Path(os.environ.get("JAX_METAL_DEVICE_LOCK",
                                   pathlib.Path.home() / ".cache" / "jax_metal" / "device.lock"))


def main(argv):
    if argv and argv[0] == "--":
        argv = argv[1:]
    if not argv:
        print(__doc__); return 2
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with open(LOCK, "w") as f:
        t0 = time.time()
        while True:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB); break
            except BlockingIOError:
                if time.time() - t0 < 1:
                    print(f"[device_lock] waiting for {LOCK} (another GPU job is running)", file=sys.stderr, flush=True)
                time.sleep(2)
        f.write(f"{os.getpid()} {' '.join(argv)}\n"); f.flush()
        return subprocess.call(argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
