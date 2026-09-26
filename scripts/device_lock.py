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
    # Open without truncating: the file names the current holder, and only
    # the process that holds the lock may rewrite it.
    with open(LOCK, "a+") as f:
        t0 = time.time()
        while True:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB); break
            except BlockingIOError:
                if time.time() - t0 < 1:
                    f.seek(0)
                    holder = f.read().strip() or "unknown"
                    print(f"[device_lock] waiting for {LOCK} (held by: {holder})", file=sys.stderr, flush=True)
                time.sleep(2)
        f.seek(0); f.truncate()
        f.write(f"{os.getpid()} {' '.join(argv)}\n"); f.flush()
        return subprocess.call(argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
