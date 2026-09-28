#!/usr/bin/env python3
"""Run a command while holding the Metal device lock.

Only one GPU-heavy process (test suite, benchmark, sweep) should run at a
time on an 8 GB unified-memory machine: concurrent runs over-commit memory
and, under thrashing, GPU command buffers hit the watchdog. Usage:

  scripts/device_lock.py -- <command> [args...]

The lock is ~/.cache/metal-pjrt/device.lock. Transition from before the
platform was renamed (from "metal"):
the lock also takes the old ~/.cache/jax_metal/device.lock first (creating
it if needed, so an older checkout cannot take it unnoticed later), so this
script and an older checkout's exclude each other.

Re-entrant: the child's environment carries JAX_MTL_DEVICE_LOCK_HELD=
<holder pid>, and a nested device_lock.py (e.g. run_jax_tests.sh, which
locks itself, run under device_lock.py) whose ancestor is that live holder
runs the command without locking again. Any other value (stale, exported by
hand) is ignored and the lock is acquired as usual.

The command is never killed from here: Ctrl-C reaches it directly (same
process group) and this script waits for it to exit (subprocess.call would
SIGKILL it on KeyboardInterrupt, with GPU work possibly in flight). The
command inherits the locked files, so it keeps holding the lock even if
this script dies first.
"""
import fcntl, os, pathlib, signal, subprocess, sys, time

LOCK = pathlib.Path.home() / ".cache" / "metal-pjrt" / "device.lock"
# Remove at the next pin bump, not before 2026-10-31 (git bisect to commits
# before the rename takes only this lock).
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
        holder = int(os.environ.get("JAX_MTL_DEVICE_LOCK_HELD", ""))
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


def run(argv, env=None, keep_fds=()):
    signal.signal(signal.SIGINT, lambda *_: None)  # the child handles Ctrl-C
    return subprocess.Popen(argv, env=env, pass_fds=keep_fds).wait()


def main(argv):
    if argv and argv[0] == "--":
        argv = argv[1:]
    if not argv:
        print(__doc__); return 2
    if held_by_ancestor():
        return run(argv)
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    OLD_LOCK.parent.mkdir(parents=True, exist_ok=True)
    held = []
    # Always the old lock first, then the new one: a fixed order, so two
    # instances of this script cannot deadlock.
    held.append(acquire(OLD_LOCK, argv))
    held.append(acquire(LOCK, argv))
    # Tells tests/conftest.py the lock is held.
    env = dict(os.environ, JAX_MTL_DEVICE_LOCK_HELD=str(os.getpid()))
    return run(argv, env, [f.fileno() for f in held])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
