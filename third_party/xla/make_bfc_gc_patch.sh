#!/bin/bash
# Regenerates patches/0004-bfc-garbage-collection.patch: enable BFC garbage
# collection for the GPU client's allocators. On unified memory the pool is
# system RAM, so a pool that never shrinks pushes the machine into swap; with
# garbage collection, a refused growth (our allocation guard) frees unused
# regions and retries.
set -euo pipefail
XLA=${1:?path to pristine xla tree}
OUT=$(cd "$(dirname "$0")" && pwd)/patches/0004-bfc-garbage-collection.patch
WORK=$(mktemp -d)
FILES=(xla/pjrt/gpu/gpu_helpers.cc)
for f in "${FILES[@]}"; do
  mkdir -p "$WORK/a/$(dirname $f)" "$WORK/b/$(dirname $f)"
  cp "$XLA/$f" "$WORK/a/$f"; cp "$XLA/$f" "$WORK/b/$f"
done
B=$WORK/b
perl -0pi -e 's|  opts.allow_growth = !preallocate;\n|  opts.allow_growth = !preallocate;\n  // metal-pjrt-plugin: return unused regions when growth is refused.\n  opts.garbage_collection = true;\n|g' $B/xla/pjrt/gpu/gpu_helpers.cc
cd $WORK && (diff -ruN a b > "$OUT" || true)
for f in "${FILES[@]}"; do cmp -s "a/$f" "b/$f" && echo "WARNING: no change in $f"; done
echo "wrote $OUT ($(grep -c '^@@' "$OUT") hunks)"
rm -rf "$WORK"
