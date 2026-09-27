"""pytest setup for tests/.

Tests marked `metal` run on the Metal device, so the run must hold the
device lock (one GPU job at a time):

  scripts/device_lock.py -- .venv/bin/python -m pytest tests
  .venv/bin/python -m pytest tests -m "not metal"   # host-only tests

Runs are hermetic: no persistent compilation cache, x64 off, and CPU next to
metal for references (tests/metal_testing.py).
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "openmetal,cpu")
os.environ.setdefault("JAX_ENABLE_X64", "0")
os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "false")

import pytest


@pytest.hookimpl(trylast=True)  # after -m / -k deselection
def pytest_collection_modifyitems(config, items):
    # test_gpu_errors starts child JAX processes: run it before this process
    # holds memory from other tests, so the children's allocation guard sees
    # it free.
    items.sort(key=lambda item: "test_gpu_errors.py" not in item.nodeid)
    if not any(item.get_closest_marker("metal") for item in items):
        return
    if not os.environ.get("JAX_OPENMETAL_DEVICE_LOCK_HELD"):
        raise pytest.UsageError(
            "tests marked metal need the device lock: scripts/device_lock.py "
            "-- .venv/bin/python -m pytest ... (or deselect them: -m 'not metal')")
    import jax
    backend = jax.default_backend()
    print(f"\njax backend: {backend} {jax.devices()}", flush=True)
    if backend != "openmetal":
        raise pytest.UsageError(f"default backend is {backend!r}, not 'openmetal'")
