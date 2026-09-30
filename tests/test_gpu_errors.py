# Copyright 2026 The metal-pjrt-plugin Authors
# SPDX-License-Identifier: Apache-2.0

"""GPU errors reach Python as exceptions, never as wrong values.

Runs a small JAX program in a fresh process with
METAL_PJRT_FAIL_COMMAND_BUFFER=n (the runtime treats the n-th committed
command buffer as failed; testing only, no real GPU fault) and checks that
no step returns wrong values, that the injected failure surfaces, and that
it is sticky: every step after the first one that raises raises too.
Without injection (n=0) every step must succeed.
"""
import os

import pytest

from metal_testing import run_python

pytestmark = pytest.mark.metal

CHILD = r"""
import numpy as np, jax, jax.numpy as jnp
x = jax.device_put(np.arange(4096, dtype=np.float32))
f = jax.jit(lambda x: jnp.sin(x) * 2.0)
g = jax.jit(lambda y: y + 1.0)
want = np.sin(np.arange(4096, dtype=np.float64)) * 2.0
def step(name, fn, want):
    try:
        # sin on large arguments: a few float32 ulps of 2.
        ok = np.allclose(np.asarray(fn()), want, rtol=0, atol=1e-6)
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


@pytest.mark.parametrize("n", range(9))
def test_injected_command_buffer_failure(n):
    env = dict(os.environ, JAX_PLATFORMS="mtl",
               METAL_PJRT_FAIL_COMMAND_BUFFER=str(n))
    # No timeout: never kill a process with GPU work in flight. The runtime's
    # own waits are bounded.
    out = run_python(CHILD, env)
    lines = [l for l in out.stdout.splitlines() if ": " in l]
    assert out.returncode == 0 and len(lines) == 5, (out.returncode, lines, out.stderr[-2000:])
    assert not [l for l in lines if "WRONG" in l], lines
    raised = ["RAISED" in l for l in lines]
    # n > 0: the injected failure must surface (the program commits more
    # than 8 command buffers) and stay; n == 0: nothing may fail.
    if n == 0:
        assert not any(raised), lines
    else:
        assert any(raised), lines
        assert all(raised[raised.index(True):]), lines
        # The sticky error says what to do. (n = 1, 2 fail the device_put
        # itself: every step then fails on the poisoned input, with XLA's
        # "Unknown predetermined error".)
        if n > 2:
            assert "restart the Python process" in lines[-1], lines


def test_array_constant_after_failure_raises():
    """A new executable whose array constants are uploaded on first run, after
    the device error is sticky, raises instead of aborting the process (XLA's
    constant upload used to CHECK_OK its BlockHostUntilDone)."""
    child = r"""
import numpy as np, jax
x = np.ones(4, np.float32)
try:
    jax.jit(lambda x: x * 2.0)(x).block_until_ready()
    print("first: OK")
except Exception as e:
    print("first: RAISED")
c = np.array([1.0, 5.0, 2.0, 7.0], np.float32)  # an XLA constant global
try:
    jax.jit(lambda x: x + c)(x).block_until_ready()
    print("constant: OK")
except Exception as e:
    print("constant: RAISED", str(e).splitlines()[0][:120])
"""
    env = dict(os.environ, JAX_PLATFORMS="mtl",
               METAL_PJRT_FAIL_COMMAND_BUFFER="1")
    out = run_python(child, env)
    assert out.returncode == 0, (out.returncode, out.stdout, out.stderr[-2000:])
    assert "constant: RAISED" in out.stdout, (out.stdout, out.stderr[-2000:])
