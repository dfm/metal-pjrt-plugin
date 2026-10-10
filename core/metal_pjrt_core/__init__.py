# Copyright 2026 The jax-graft Authors
# SPDX-License-Identifier: Apache-2.0

"""metal-pjrt-core: the PJRT plugin library of metal-pjrt-plugin.

Only the library, next to this file (a real file in the wheel, built by
scripts/build_wheel.sh; a link into bazel-bin made by scripts/install_dev.sh
in a source checkout). jax_graft finds it here and registers it with
JAX; nothing in this package is for users.
"""

import pathlib

__all__ = ["library_path"]

LIBRARY_NAME = "pjrt_c_api_mtl_plugin.dylib"


def library_path() -> pathlib.Path:
    """Where the library should be (it may be missing or a dangling link)."""
    return pathlib.Path(__file__).resolve().parent / LIBRARY_NAME
