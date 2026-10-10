#!/bin/bash
# Copyright 2026 The jax-graft Authors
# SPDX-License-Identifier: Apache-2.0

# Regenerates patches/0003-macos-build-fixes.patch: portability fixes needed to
# compile XLA's GPU stack on macOS (size_t and uint64_t are distinct types on
# Darwin, so an override declared with size_t does not match).
set -euo pipefail
XLA=${1:?path to pristine xla tree}
OUT=$(cd "$(dirname "$0")" && pwd)/patches/0003-macos-build-fixes.patch
WORK=$(mktemp -d)
FILES=(xla/backends/gpu/runtime/record_ffi.cc)
for f in "${FILES[@]}"; do
  mkdir -p "$WORK/a/$(dirname $f)" "$WORK/b/$(dirname $f)"
  cp "$XLA/$f" "$WORK/a/$f"; cp "$XLA/$f" "$WORK/b/$f"
done
B=$WORK/b
perl -0pi -e 's|size_t number_of_shared_bytes\(\) const override|uint64_t number_of_shared_bytes() const override|' $B/xla/backends/gpu/runtime/record_ffi.cc
cd $WORK && (diff -ruN a b > "$OUT" || true)
for f in "${FILES[@]}"; do cmp -s "a/$f" "b/$f" && echo "WARNING: no change in $f"; done
echo "wrote $OUT ($(grep -c '^@@' "$OUT") hunks)"
rm -rf "$WORK"
