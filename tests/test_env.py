# Copyright 2026 The jax-graft Authors
# SPDX-License-Identifier: Apache-2.0

"""Host-only checks of how the plugin's environment variables are parsed
(the C++ side is metal_pjrt/runtime/env_test.cc; test_memory.py runs the
runtime with bad values)."""
import os
import pathlib
import subprocess
import sys

import pytest

from jax_graft import _env_flag

ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("value, want", [
    (None, False), ("", False), ("0", False), ("false", False),
    ("False", False), ("NO", False), (" off ", False), ("\toff\n", False),
    # ASCII whitespace only, as absl::StripAsciiWhitespace: a no-break
    # space is part of the value.
    ("\u00a0off", True),
    ("1", True), ("true", True), ("yes", True), ("ON", True), ("2", True)])
def test_env_flag(monkeypatch, value, want):
    # One convention for every boolean, the same as EnvFlag in C++
    # (METAL_PJRT_DISABLE_LAPACK=false once disabled LAPACK).
    if value is None:
        monkeypatch.delenv("METAL_PJRT_ENV_TEST", raising=False)
    else:
        monkeypatch.setenv("METAL_PJRT_ENV_TEST", value)
    assert _env_flag("METAL_PJRT_ENV_TEST") is want


@pytest.mark.parametrize("value, want", [("3", "3"), ("two", "0"), ("-1", "0")])
def test_gpu_health_strikes(tmp_path, value, want):
    # A bad METAL_PJRT_QUARANTINE_STRIKES keeps the default instead of
    # crashing the script. An empty state dir: no reset log is touched.
    env = dict(os.environ, METAL_PJRT_STATE_DIR=str(tmp_path),
               METAL_PJRT_QUARANTINE_STRIKES=value)
    out = subprocess.run(
        [sys.executable, "-c",
         "import runpy, sys; "
         "print(runpy.run_path(sys.argv[1])['STRIKES'])",
         str(ROOT / "scripts" / "gpu_health.py")],
        env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == [want], out.stdout
    assert ("ignoring" in out.stderr) == (want != value), out.stderr
