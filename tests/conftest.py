# Copyright 2026 The jax-graft Authors
# SPDX-License-Identifier: Apache-2.0

"""pytest setup for tests/.

Tests marked `metal` run on the Metal device, so the run must hold the
device lock (one GPU job at a time):

  scripts/device_lock.py -- .venv/bin/python -m pytest tests
  .venv/bin/python -m pytest tests -m "not metal"   # host-only tests

Runs are hermetic: no persistent compilation cache, x64 off, and CPU next to
metal for references (tests/metal_testing.py), whatever the environment
says; and a run refuses to start with XLA_FLAGS or plugin settings
(METAL_PJRT_*, e.g. the memory fraction) in the
environment (tests that need one set it for a child process).
"""
import importlib.util
import os
import pathlib

# No JAX_PLATFORMS: the tests run on JAX's default backend, which the
# installed plugin makes mtl (checked below for tests marked metal).
os.environ.pop("JAX_PLATFORMS", None)
os.environ.update(JAX_ENABLE_X64="0", JAX_ENABLE_COMPILATION_CACHE="false")

# Plugin variables that change nothing a test checks: where the GPU reset
# log lives (a scratch dir for runs from a fresh checkout), trace logging
# and the device lock's own marker.
ALLOWED_PLUGIN_VARS = {"METAL_PJRT_STATE_DIR", "METAL_PJRT_TRACE",
                       "METAL_PJRT_DEVICE_LOCK_HELD"}

import pytest


def pytest_configure(config):
    leaked = sorted(k for k in os.environ if k == "XLA_FLAGS" or (
        k.startswith("METAL_PJRT_")
        and k not in ALLOWED_PLUGIN_VARS))
    if leaked:
        raise pytest.UsageError(
            f"unset {', '.join(leaked)}: the tests assume the defaults (tests "
            "that need a setting set it for a child process)")


def lock_held():
    # device_lock.py's own check: the holder named in the environment is a
    # live ancestor of this process (a stale or hand-set value does not count).
    path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "device_lock.py"
    spec = importlib.util.spec_from_file_location("device_lock", path)
    device_lock = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(device_lock)
    return device_lock.held_by_ancestor()


@pytest.hookimpl(trylast=True)  # after -m / -k deselection
def pytest_collection_modifyitems(config, items):
    # test_gpu_errors starts child JAX processes: run it before this process
    # holds memory from other tests, so the children's allocation guard sees
    # it free.
    items.sort(key=lambda item: "test_gpu_errors.py" not in item.nodeid)
    if not any(item.get_closest_marker("metal") for item in items):
        return
    if not lock_held():
        raise pytest.UsageError(
            "tests marked metal need the device lock: scripts/device_lock.py "
            "-- .venv/bin/python -m pytest ... (or deselect them: -m 'not metal')")
    import jax
    backend = jax.default_backend()
    print(f"\njax backend: {backend} {jax.devices()}", flush=True)
    if backend != "mtl":
        raise pytest.UsageError(f"default backend is {backend!r}, not 'mtl'")
