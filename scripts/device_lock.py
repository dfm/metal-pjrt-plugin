#!/usr/bin/env python3
"""Run a command while holding the Metal device lock.

Only one GPU-heavy process (test suite, benchmark, sweep) should run at a
time on an 8 GB unified-memory machine: concurrent runs over-commit memory
and, under thrashing, GPU command buffers hit the watchdog. Usage:

  scripts/device_lock.py -- <command> [args...]

The lock is ~/.cache/openmetal/device.lock (JAX_OPENMETAL_DEVICE_LOCK
overrides). Transition from before the platform was renamed "openmetal":
the lock also takes the old ~/.cache/jax_metal/device.lock first (creating
it if needed, so an older checkout cannot take it unnoticed later), so this
script and an older checkout's exclude each other.

Re-entrant: the child's environment carries JAX_OPENMETAL_DEVICE_LOCK_HELD=
<holder pid>, and a nested device_lock.py (e.g. run_jax_tests.sh, which
locks itself, run under device_lock.py) whose ancestor is that live holder
runs the command without locking again. Any other value (stale, exported by
hand) is ignored and the lock is acquired as usual.
"""
import fcntl, os, pathlib, subprocess, sys, time

LOCK = pathlib.Path(os.environ.get("JAX_OPENMETAL_DEVICE_LOCK",
                                   pathlib.Path.home() / ".cache" / "openmetal" / "device.lock"))
OLD_LOCK = pathlib.Path.home() / ".cache" / "jax_metal" / "device.lock"


def acquire(path, argv):
    # Open without truncating: the file names the current holder, and only
    # the process that holds the lock may rewrite it.
    f = open(path, "a+")
    t0 = time.time()
    while True:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB); break
        except BlockingIOError:
            if time.time() - t0 < 1:
                f.seek(0)
                holder = f.read().strip() or "unknown"
                print(f"[device_lock] waiting for {path} (held by: {holder})", file=sys.stderr, flush=True)
            time.sleep(2)
    f.seek(0); f.truncate()
    f.write(f"{os.getpid()} {' '.join(argv)}\n"); f.flush()
    return f  # held until this process exits


def held_by_ancestor():
    try:
        holder = int(os.environ.get("JAX_OPENMETAL_DEVICE_LOCK_HELD", ""))
    except ValueError:
        return False
    pid = os.getppid()
    while pid > 1:
        if pid == holder:
            return True
        out = subprocess.run(["ps", "-o", "ppid=", "-p", str(pid)],
                             capture_output=True, text=True).stdout.strip()
        pid = int(out) if out else 0
    return False


def main(argv):
    if argv and argv[0] == "--":
        argv = argv[1:]
    if not argv:
        print(__doc__); return 2
    if held_by_ancestor():
        return subprocess.call(argv)
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    OLD_LOCK.parent.mkdir(parents=True, exist_ok=True)
    held = []
    # Always the old lock first, then the new one: a fixed order, so two
    # instances of this script cannot deadlock.
    if OLD_LOCK.resolve() != LOCK.resolve():
        held.append(acquire(OLD_LOCK, argv))
    held.append(acquire(LOCK, argv))
    # Tells tests/conftest.py the lock is held.
    env = dict(os.environ, JAX_OPENMETAL_DEVICE_LOCK_HELD=str(os.getpid()))
    return subprocess.call(argv, env=env)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
