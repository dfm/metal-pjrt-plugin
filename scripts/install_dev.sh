#!/bin/bash
# Build the plugin dylib and link it into the Python package for development.
#   scripts/install_dev.sh            # build + link + pip install -e .
#   scripts/install_dev.sh --no-build # just link an existing build
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "${1:-}" != "--no-build" ]]; then
  bazel build //metal_pjrt_plugin/pjrt:pjrt_c_api_metal_plugin.dylib
fi
SRC=bazel-bin/metal_pjrt_plugin/pjrt/pjrt_c_api_metal_plugin.dylib
[[ -f "$SRC" ]] || { echo "missing $SRC" >&2; exit 1; }
ln -sf "$(pwd)/$SRC" jax_plugins/metal/pjrt_c_api_metal_plugin.dylib
if [[ -x .venv/bin/python ]] && command -v uv >/dev/null; then
  uv pip install --python .venv/bin/python -e . >/dev/null
else
  python3 -m pip install -e . >/dev/null
fi
echo "installed; try: scripts/device_lock.py -- .venv/bin/python -m pytest tests/test_smoke.py"
