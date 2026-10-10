#!/bin/bash
# Copyright 2026 The metal-pjrt-plugin Authors
# SPDX-License-Identifier: Apache-2.0

# Regenerates patches/0004-custom-fusion-emitter-factory.patch: lets a backend
# outside XLA's tree supply the emitter for kCustom fusions of kind
# "__custom_fusion" (upstream's CustomFusion only returns Unimplemented since
# the custom-kernel-fusion registry was removed). MetalCompiler registers the
# emitter for its row-normalization fusions. Without a registered factory, or
# when it returns null, behavior is unchanged.
set -euo pipefail
XLA=${1:?path to pristine xla tree}
OUT=$(cd "$(dirname "$0")" && pwd)/patches/0004-custom-fusion-emitter-factory.patch
WORK=$(mktemp -d)
FILES=(xla/backends/gpu/codegen/fusions.h xla/backends/gpu/codegen/fusions.cc)
for f in "${FILES[@]}"; do
  mkdir -p "$WORK/a/$(dirname $f)" "$WORK/b/$(dirname $f)"
  cp "$XLA/$f" "$WORK/a/$f"; cp "$XLA/$f" "$WORK/b/$f"
done
B=$WORK/b
perl -0pi -e 's|(// Returns the emitter for the given fusion.\n)|// Emitter for kCustom fusions of kind "__custom_fusion", supplied by a\n// backend outside XLA (set once, before compiling). Returns null for fusions\n// it does not handle, which then get CustomFusion (Unimplemented).\nusing CustomFusionEmitterFactory =\n    std::unique_ptr<FusionInterface> (*)(const FusionInfo& fusion_info);\nvoid SetCustomFusionEmitterFactory(CustomFusionEmitterFactory factory);\n\n$1|' $B/xla/backends/gpu/codegen/fusions.h
perl -0pi -e 's|(std::unique_ptr<FusionInterface> GetFusionEmitter\(\n    const FusionInfo& fusion_info\) \{\n)|namespace {\nstd::atomic<CustomFusionEmitterFactory> custom_fusion_emitter_factory{\n    nullptr};\n}  // namespace\n\nvoid SetCustomFusionEmitterFactory(CustomFusionEmitterFactory factory) {\n  custom_fusion_emitter_factory.store(factory);\n}\n\n$1|' $B/xla/backends/gpu/codegen/fusions.cc
perl -0pi -e 's|(    case HloFusionAnalysis::EmitterFusionKind::kCustomFusion:\n)(      return std::make_unique<CustomFusion>\(\);\n)|$1    {\n      if (CustomFusionEmitterFactory factory =\n              custom_fusion_emitter_factory.load()) {\n        if (auto emitter = factory(fusion_info)) return emitter;\n      }\n$2    }\n|' $B/xla/backends/gpu/codegen/fusions.cc
perl -0pi -e 's|(#include <memory>\n)|#include <atomic>\n$1|' $B/xla/backends/gpu/codegen/fusions.cc
cd $WORK && (diff -ruN a b > "$OUT" || true)
for f in "${FILES[@]}"; do cmp -s "a/$f" "b/$f" && echo "WARNING: no change in $f"; done
echo "wrote $OUT ($(grep -c '^@@' "$OUT") hunks)"
rm -rf "$WORK"
