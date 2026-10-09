# Copyright 2026 The metal-pjrt-plugin Authors
# SPDX-License-Identifier: Apache-2.0

"""Lowering for "mtl" without a GPU: the frontend's lowering rules (FFT,
LAPACK linear algebra, complex convolutions, host callbacks, checkify,
donation) against the installed JAX, plus a smoke test of everyday programs
through JAX's generic rules.

The frontend registers its rules with JAX's private APIs and never lets a
failure break plugin initialization (it logs a warning and the primitive
then has no "mtl" rule), so a JAX release can break them silently; this is
what the nightly workflow runs against JAX nightly. The programs are lowered
in a child process with JAX_PLATFORMS=cpu (tests/lowering_probe.py): the
platform and its rules are registered, but no Metal client is created. What
the compiler does with the lowered module needs the GPU suite.

No GPU needed: .venv/bin/python -m pytest tests/test_lowering.py
"""
import json
import os
import pathlib
import subprocess
import sys

import pytest

PROBE = pathlib.Path(__file__).with_name("lowering_probe.py")

FFT = {"metal$fft"}
CALLBACK = {"xla_ffi_python_metal_callback"}

# case -> custom-call targets the lowered module must contain exactly (an
# empty set: none) and StableHLO ops it must contain.
EXPECTED = {
    "fft c64": (FFT, set()),
    "fft2 c64": (FFT, set()),
    "rfft f32": (FFT, set()),
    "irfft c64": (FFT, set()),
    # The C++ MetalConvRewriter makes metal$conv later; a complex one is
    # expanded into real convolutions here (checked below).
    "conv f32": (set(), {"convolution"}),
    "conv c64": (set(), {"convolution"}),
    "lu f32": ({"metal$lapack_getrf"}, set()),
    "qr f32": ({"metal$lapack_geqrf", "metal$lapack_orgqr"}, set()),
    "eigh f32": ({"metal$lapack_syevd"}, set()),
    # complex64 takes JAX's TPU rule (Jacobi via XLA's EighExpander).
    "eigh c64": ({"Eigh"}, set()),
    "svd f32": ({"metal$lapack_gesdd"}, set()),
    "svd f32 values": ({"metal$lapack_gesdd_novec"}, set()),
    # HLO ops the C++ MetalLinalgRewriter turns into metal$ custom calls.
    "cholesky f32": (set(), {"cholesky"}),
    "triangular_solve f32": (set(), {"triangular_solve"}),
    "solve f32": ({"metal$lapack_getrf"}, set()),
    "debug.print": (CALLBACK, set()),
    "pure_callback": (CALLBACK, set()),
    "io_callback ordered": (CALLBACK, set()),
    "checkify debug_check": (set(), set()),
    "donation": (set(), set()),
    "sort": (set(), {"sort"}),
    "argsort": (set(), {"sort"}),
    "top_k": (set(), set()),
    "cumsum": (set(), set()),
    "scan": (set(), {"while"}),
    "while_loop": (set(), {"while"}),
    "gather": (set(), {"gather"}),
    "scatter-add": (set(), {"scatter"}),
    "random": (set(), set()),
    "matmul bf16": (set(), {"dot_general"}),
    "grad of conv": (set(), {"convolution"}),
    "softmax": (set(), {"exponential"}),
}


@pytest.fixture(scope="module")
def probe():
    env = {k: v for k, v in os.environ.items() if not k.startswith("METAL_PJRT_")}
    env["JAX_PLATFORMS"] = "cpu"
    out = subprocess.run([sys.executable, str(PROBE)], env=env,
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-4000:]
    return json.loads(out.stdout)


def test_no_metal_client_and_no_warnings(probe):
    # A rule that could not be registered is only a logged warning.
    assert probe["warnings"] == []
    assert probe["backends"] == ["cpu"]


def test_every_case_is_checked(probe):
    assert set(probe["results"]) == set(EXPECTED)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_lowers_for_mtl(probe, name):
    result = probe["results"][name]
    assert "error" not in result, result.get("traceback", result.get("error"))
    targets, ops = EXPECTED[name]
    assert set(result["targets"]) == targets
    assert ops <= set(result["ops"]), sorted(result["ops"])


def test_complex_convolution_is_expanded(probe):
    assert not probe["results"]["conv c64"]["complex_conv"]


def test_donation_aliases_the_input(probe):
    # JAX drops donate_argnums on platforms outside
    # mlir._platforms_with_donation, which the frontend extends.
    assert probe["results"]["donation"]["aliased"]
