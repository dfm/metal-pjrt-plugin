#!/bin/bash
# Copyright 2026 The metal-pjrt-plugin Authors
# SPDX-License-Identifier: Apache-2.0

# Build the two wheels in dist/:
# - metal-pjrt-core (core/): the plugin dylib as package data (a real file,
#   not the dev symlink into bazel-bin), py3-none-macosx_<MACOS_MIN>_arm64:
#   a dylib loaded through ctypes/PJRT, so it depends on neither the Python
#   nor the jax version.
# - metal-pjrt-plugin (the root): the pure-Python frontend, py3-none-any.
#   scripts/build_wheel.sh                 # bazel build, then both wheels
#   scripts/build_wheel.sh --no-build      # both, from the existing bazel-bin dylib
#   scripts/build_wheel.sh --frontend-only # only metal-pjrt-plugin (any OS)
# MACOS_MIN is the dylib's deployment target (.bazelrc, --macos_minimum_os);
# the script checks that the two agree. Release the two separately
# (docs/development.md, "Packages and releases").
set -euo pipefail
cd "$(dirname "$0")/.."
MACOS_MIN=26_0
DYLIB=bazel-bin/metal_pjrt/pjrt/pjrt_c_api_mtl_plugin.dylib
MODE="${1:-}"
case "$MODE" in
  ""|--no-build|--frontend-only) ;;
  *) echo "usage: $0 [--no-build | --frontend-only]" >&2; exit 2 ;;
esac

STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
mkdir -p dist

build_core() {
  if [[ "$MODE" == "" ]]; then
    bazel build //metal_pjrt/pjrt:pjrt_c_api_mtl_plugin.dylib
  fi
  [[ -f "$DYLIB" ]] || { echo "missing $DYLIB" >&2; exit 1; }
  local minos
  minos=$(otool -l "$DYLIB" | awk '/LC_BUILD_VERSION/ {f=1} f && $1 == "minos" {print $2; f=0}')
  if [[ "${minos//./_}" != "$MACOS_MIN" ]]; then
    echo "$DYLIB is built for macOS ${minos:-?} or later, but the wheel would be tagged macosx_${MACOS_MIN}; rebuild it, or change MACOS_MIN and .bazelrc together" >&2
    exit 1
  fi
  # The library is loaded into any Python through ctypes/PJRT: it must not
  # link libpython.
  if otool -L "$DYLIB" | grep -qi python; then
    echo "$DYLIB links Python; metal-pjrt-core must not depend on the Python version" >&2
    exit 1
  fi
  local core="$STAGE/core"
  mkdir -p "$core/metal_pjrt_core"
  cp core/pyproject.toml core/README.md LICENSE THIRD_PARTY_NOTICES "$core"/
  cp core/metal_pjrt_core/*.py "$core/metal_pjrt_core/"
  cp -L "$DYLIB" "$core/metal_pjrt_core/"
  chmod u+w "$core/metal_pjrt_core/pjrt_c_api_mtl_plugin.dylib"
  cat > "$core/setup.cfg" <<CFG
[bdist_wheel]
python_tag = py3
plat_name = macosx_${MACOS_MIN}_arm64
CFG
  uv build --wheel --out-dir dist "$core"
  ls -l dist/metal_pjrt_core-*-py3-none-macosx_"${MACOS_MIN}"_arm64.whl
}

build_frontend() {
  local plugin="$STAGE/plugin"
  mkdir -p "$plugin/metal_pjrt_plugin"
  cp pyproject.toml README.md LICENSE THIRD_PARTY_NOTICES "$plugin"/
  cp metal_pjrt_plugin/*.py "$plugin/metal_pjrt_plugin/"
  uv build --wheel --out-dir dist "$plugin"
  ls -l dist/metal_pjrt_plugin-*-py3-none-any.whl
}

[[ "$MODE" == "--frontend-only" ]] || build_core
build_frontend
