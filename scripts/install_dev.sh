#!/bin/bash
# Build the plugin dylib and link it into the Python package for development.
#   scripts/install_dev.sh            # build + link + pip install -e .[test]
#   scripts/install_dev.sh --no-build # just link an existing build
# Creates .venv (Python 3.12) if it does not exist: with uv if installed,
# else with python3.12 from PATH. Never installs into another Python.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -x .venv/bin/python ]]; then
  if command -v uv >/dev/null; then
    uv venv --python 3.12 .venv
  elif command -v python3.12 >/dev/null; then
    python3.12 -m venv .venv
  else
    echo "need uv or python3.12 (jax 0.11.2 requires Python >= 3.12)" >&2
    exit 1
  fi
fi
if [[ "${1:-}" != "--no-build" ]]; then
  bazel build //metal_pjrt/pjrt:pjrt_c_api_openmetal_plugin.dylib
fi
SRC=bazel-bin/metal_pjrt/pjrt/pjrt_c_api_openmetal_plugin.dylib
[[ -f "$SRC" ]] || { echo "missing $SRC" >&2; exit 1; }
ln -sf "$(pwd)/$SRC" jax_plugins/openmetal/pjrt_c_api_openmetal_plugin.dylib
# Leftovers of earlier names: the pre-rename "metal" platform's package
# directory (only its generated files), dist "jax-metal-pjrt", whose
# jax_plugins entry point would load the same dylib a second time as
# platform "metal", and dist "jax-openmetal" (this package before it became
# openmetal_pjrt_plugin), whose entry point would register it twice.
OLD_DISTS=(jax-metal-pjrt jax-openmetal)
rm -f jax_plugins/metal/pjrt_c_api_metal_plugin.dylib
rm -rf jax_plugins/metal/__pycache__
rmdir jax_plugins/metal 2>/dev/null || true
# On sys.path when python runs from here.
rm -rf jax_metal_pjrt.egg-info jax_openmetal.egg-info
if command -v uv >/dev/null; then
  uv pip uninstall --python .venv/bin/python "${OLD_DISTS[@]}" >/dev/null 2>&1 || true
  uv pip install --python .venv/bin/python -e ".[test]" >/dev/null
else
  .venv/bin/python -m pip uninstall -y "${OLD_DISTS[@]}" >/dev/null 2>&1 || true
  .venv/bin/python -m pip install -e ".[test]" >/dev/null
fi
echo "installed; try: scripts/device_lock.py -- .venv/bin/python -m pytest tests/test_smoke.py"
