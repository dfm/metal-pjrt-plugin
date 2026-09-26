#!/bin/bash
# Regenerates patches/0005-command-buffers-on-metal.patch: let the command
# buffer conversion pass run on the Metal platform. Upstream clears every
# enabled command type for devices with a OneAPI compute capability (SYCL has
# no command buffers yet); the Metal platform reports OneAPI too (see
# docs/integration-notes.md), with device vendor "Apple", and implements
# se::CommandBuffer (stream_executor/metal_command_buffer.*). Which command
# types are converted is still chosen by xla_gpu_enable_command_buffer, which
# MetalCompiler's ApplyMetalDefaults sets.
set -euo pipefail
XLA=${1:?path to pristine xla tree}
OUT=$(cd "$(dirname "$0")" && pwd)/patches/0005-command-buffers-on-metal.patch
WORK=$(mktemp -d)
FILES=(xla/backends/gpu/runtime/command_buffer_conversion_pass.cc)
for f in "${FILES[@]}"; do
  mkdir -p "$WORK/a/$(dirname $f)" "$WORK/b/$(dirname $f)"
  cp "$XLA/$f" "$WORK/a/$f"; cp "$XLA/$f" "$WORK/b/$f"
done
B=$WORK/b
perl -0pi -e 's|  if \(device_info.gpu_compute_capability\(\).IsOneAPI\(\)\) \{\n    config.enabled_commands.clear\(\);|  // metal-pjrt-plugin: the Metal platform also reports a OneAPI compute\n  // capability (device vendor "Apple") and can implement command buffers.\n  if (device_info.gpu_compute_capability().IsOneAPI() &&\n      device_info.device_vendor() != "Apple") {\n    config.enabled_commands.clear();|' $B/xla/backends/gpu/runtime/command_buffer_conversion_pass.cc
cd $WORK && (diff -ruN a b > "$OUT" || true)
for f in "${FILES[@]}"; do cmp -s "a/$f" "b/$f" && echo "WARNING: no change in $f"; done
echo "wrote $OUT ($(grep -c '^@@' "$OUT") hunks)"
rm -rf "$WORK"
