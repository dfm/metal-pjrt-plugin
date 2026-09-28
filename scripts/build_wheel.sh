#!/bin/bash
# Build the metal-pjrt-plugin wheel with the plugin dylib inside it as package
# data (a real file, not the dev symlink into bazel-bin).
#   scripts/build_wheel.sh            # bazel build, then the wheel in dist/
#   scripts/build_wheel.sh --no-build # wheel from the existing bazel-bin dylib
# The wheel is py3-none-macosx_<MACOS_MIN>_arm64: pure Python plus a dylib
# loaded through ctypes/PJRT, so it does not depend on the Python version.
# MACOS_MIN is the oldest macOS the plugin is tested on (see the README).
set -euo pipefail
cd "$(dirname "$0")/.."
MACOS_MIN=26_0
if [[ "${1:-}" != "--no-build" ]]; then
  bazel build //metal_pjrt/pjrt:pjrt_c_api_mtl_plugin.dylib
fi
DYLIB=bazel-bin/metal_pjrt/pjrt/pjrt_c_api_mtl_plugin.dylib
[[ -f "$DYLIB" ]] || { echo "missing $DYLIB" >&2; exit 1; }
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
cp pyproject.toml README.md LICENSE "$STAGE"/
mkdir -p "$STAGE/metal_pjrt_plugin"
cp metal_pjrt_plugin/*.py "$STAGE/metal_pjrt_plugin/"
cp -L "$DYLIB" "$STAGE/metal_pjrt_plugin/"
chmod u+w "$STAGE/metal_pjrt_plugin/pjrt_c_api_mtl_plugin.dylib"
cat > "$STAGE/setup.cfg" <<EOF
[bdist_wheel]
python_tag = py3
plat_name = macosx_${MACOS_MIN}_arm64
EOF
mkdir -p dist
uv build --wheel --out-dir dist "$STAGE"
ls -l dist/metal_pjrt_plugin-*-py3-none-macosx_${MACOS_MIN}_arm64.whl
