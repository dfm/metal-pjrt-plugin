#!/bin/bash
# Copyright 2026 The jax-graft Authors
# SPDX-License-Identifier: Apache-2.0

# Build the plugin dylib, link it into the metal-pjrt-core package (core/)
# and install both packages editable, for development.
#   scripts/install_dev.sh            # build + link + pip install -e core -e .[test]
#   scripts/install_dev.sh --no-build # just link an existing build
# Creates .venv (Python 3.12) if it does not exist: with uv if installed,
# else with python3.12 from PATH. Never installs into another Python.
set -euo pipefail
cd "$(dirname "$0")/.."
# Before a two-hour build: the things that would only fail at its end.
if [[ "$(uname -s)-$(uname -m)" != "Darwin-arm64" ]]; then
  echo "metal-pjrt-plugin builds only on Apple Silicon Macs (this is $(uname -s) $(uname -m))" >&2
  exit 1
fi
if [[ "${1:-}" != "--no-build" ]]; then
  command -v bazel >/dev/null || { echo "need bazel (brew install bazelisk)" >&2; exit 1; }
  # The SDK xcode-select points at; Bazel may pick another installed Xcode,
  # so an old one here is a warning, not an error.
  SDK=$(xcrun --show-sdk-version 2>/dev/null || true)
  if [[ -z "$SDK" ]]; then
    echo "no macOS SDK found: install the Xcode command-line tools (xcode-select --install)" >&2
    exit 1
  elif [[ "${SDK%%.*}" -lt 26 ]]; then
    echo "warning: the selected macOS SDK is $SDK; the plugin needs the macOS 26 SDK or later (its Metal headers reference symbols new in macOS 26), so the link will fail unless Bazel finds a newer Xcode" >&2
  fi
fi
if [[ ! -x .venv/bin/python ]]; then
  if command -v uv >/dev/null; then
    uv venv --python 3.12 .venv
  elif command -v python3.12 >/dev/null; then
    python3.12 -m venv .venv
  else
    echo "need uv or python3.12 (metal-pjrt-plugin requires Python >= 3.12)" >&2
    exit 1
  fi
fi
if [[ "${1:-}" != "--no-build" ]]; then
  bazel build //metal_pjrt/pjrt:pjrt_c_api_mtl_plugin.dylib
fi
SRC=bazel-bin/metal_pjrt/pjrt/pjrt_c_api_mtl_plugin.dylib
[[ -f "$SRC" ]] || { echo "missing $SRC" >&2; exit 1; }
ln -sf "$(pwd)/$SRC" core/metal_pjrt_core/pjrt_c_api_mtl_plugin.dylib
# Where the link was before the library moved to metal-pjrt-core.
rm -f metal_pjrt_plugin/pjrt_c_api_mtl_plugin.dylib
# Leftovers of earlier names, each of whose jax_plugins entry point would
# load the same dylib a second time: dists "jax-metal-pjrt" (platform
# "metal"), "jax-openmetal" and "openmetal_pjrt_plugin" (platform
# "openmetal"), "metal-pjrt-plugin" (jax-graft's earlier name), and the
# generated files left in their jax_plugins/metal and jax_plugins/openmetal
# package directories (JAX also path-scans jax_plugins/*).
OLD_DISTS=(jax-metal-pjrt jax-openmetal openmetal_pjrt_plugin metal-pjrt-plugin)
for d in jax_plugins/metal jax_plugins/openmetal; do
  rm -f "$d"/*.dylib
  rm -rf "$d/__pycache__"
  rmdir "$d" 2>/dev/null || true
done
rmdir jax_plugins 2>/dev/null || true
# On sys.path when python runs from here.
rm -rf jax_metal_pjrt.egg-info jax_openmetal.egg-info openmetal_pjrt_plugin.egg-info \
  metal_pjrt_plugin.egg-info
if command -v uv >/dev/null; then
  uv pip uninstall --python .venv/bin/python "${OLD_DISTS[@]}" >/dev/null 2>&1 || true
  uv pip install --python .venv/bin/python -e core -e ".[test]" >/dev/null
else
  .venv/bin/python -m pip uninstall -y "${OLD_DISTS[@]}" >/dev/null 2>&1 || true
  .venv/bin/python -m pip install -e core -e ".[test]" >/dev/null
fi
echo "installed; try: scripts/device_lock.py -- .venv/bin/python -m pytest tests/test_smoke.py"
