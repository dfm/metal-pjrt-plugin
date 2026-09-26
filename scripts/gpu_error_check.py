#!/usr/bin/env python3
"""GPU errors reach Python as exceptions, never as wrong values.

Runs a small JAX program in fresh processes with
METAL_PJRT_FAIL_COMMAND_BUFFER=n (the runtime treats the n-th committed
command buffer as failed; testing only, no real GPU fault) for several n, and
checks that every step either returns the right values or raises, and that
each injected failure surfaces. Without injection every step must succeed.

  scripts/device_lock.py -- .venv/bin/python scripts/gpu_error_check.py
"""
import os, subprocess, sys

CHILD = r"""
import numpy as np, jax, jax.numpy as jnp
x = jax.device_put(np.arange(4096, dtype=np.float32))
f = jax.jit(lambda x: jnp.sin(x) * 2.0)
g = jax.jit(lambda y: y + 1.0)
want = np.sin(np.arange(4096, dtype=np.float32)) * 2.0
def step(name, fn, want):
    try:
        ok = np.allclose(np.asarray(fn()), want, atol=1e-5)
        print(f"{name}: {'OK' if ok else 'WRONG'}")
    except Exception as e:
        print(f"{name}: RAISED {str(e).splitlines()[0][:120]}")
y = f(x)
step("f(x)", lambda: y.block_until_ready(), want)
step("np.asarray(f(x))", lambda: y, want)
step("g(f(x))", lambda: g(y), want + 1)
step("fresh f(x)", lambda: f(x), want)
step("fresh g(f(x))", lambda: g(f(x)), want + 1)
"""


def run(n):
    env = dict(os.environ, JAX_PLATFORMS="metal",
               METAL_PJRT_FAIL_COMMAND_BUFFER=str(n))
    # No timeout: never kill a process with GPU work in flight. The runtime's
    # own waits are bounded.
    out = subprocess.run([sys.executable, "-c", CHILD], env=env,
                         capture_output=True, text=True)
    return [l for l in out.stdout.splitlines() if ": " in l], out.returncode


def main():
    failures = 0
    for n in range(0, 9):
        lines, rc = run(n)
        raised = sum("RAISED" in l for l in lines)
        bad = [l for l in lines if "WRONG" in l]
        # n > 0: the injected failure must surface (the program commits more
        # than 8 command buffers); n == 0: nothing may fail.
        ok = (rc == 0 and len(lines) == 5 and not bad and
              (raised > 0 if n > 0 else raised == 0))
        failures += not ok
        print(f"{'PASS' if ok else 'FAIL'} fail command buffer {n}: "
              f"{raised}/5 steps raised" + ("".join("\n  " + l for l in bad)))
        if rc != 0 or len(lines) != 5:
            print(f"  exit code {rc}; output: {lines}")
    print(f"\n{9 - failures}/9 passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
