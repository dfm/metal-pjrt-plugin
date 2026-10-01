#!/bin/bash
# Copyright 2026 The metal-pjrt-plugin Authors
# SPDX-License-Identifier: Apache-2.0

# Runs every backend ROUNDS times, interleaved (round 1: metal, metal-gpu,
# cpu, mlx; round 2: ...), appending to bench/results/<label>.jsonl, then
# writes the table (medians over rounds) to bench/results/table.md; neither
# is tracked. Every row records the commit, versions and METAL_PJRT_*/XLA
# knobs (bench/common.py).
#   BENCH_BACKENDS="metal metal-gpu cpu mlx"  (default: all)
#   BENCH_ROUNDS=3  BENCH_ONLY=<case substrings>
#   BENCH_BAZEL_SHUTDOWN=1  stop this workspace's Bazel server first (its JVM
#     holds memory the allocation guard then refuses; off by default, since
#     it would kill another session's build)
# Compare against MLX only on a freshly booted, idle machine.
set -uo pipefail
cd "$(dirname "$0")/.."
# Serialize against other GPU jobs (see scripts/device_lock.py): re-run
# under the lock unless a live ancestor holds it (not just an env var set).
if ! .venv/bin/python -c 'import sys; sys.path.insert(0, "scripts"); import device_lock; sys.exit(not device_lock.held_by_ancestor())'; then
  exec scripts/device_lock.py -- "$0" "$@"
fi
# Booleans as everywhere (docs/development.md): unset, "", 0, false, no,
# off (any case) are off.
flag() { case "$(printf %s "${1:-}" | tr -d ' \t\n\v\f\r' | tr '[:upper:]' '[:lower:]')" in ""|0|false|no|off) return 1 ;; esac; }
# Refuse to time a GPU that was reset since boot (it runs slow until reboot).
if ! flag "${BENCH_ALLOW_DEGRADED:-}"; then
  scripts/gpu_health.py --strict || { echo "set BENCH_ALLOW_DEGRADED=1 to run anyway" >&2; exit 1; }
fi
# The Bazel server JVM holds a lot of memory after a build; the allocation
# guard then refuses large pools (nanoGPT train step needs a 1.2 GB chunk).
if flag "${BENCH_BAZEL_SHUTDOWN:-}" && command -v bazel >/dev/null; then
  bazel shutdown >/dev/null 2>&1
fi
mkdir -p bench/results
ONLY=${BENCH_ONLY:-}
BACKENDS=${BENCH_BACKENDS:-metal metal-gpu cpu mlx}
ROUNDS=${BENCH_ROUNDS:-3}
files=()
for label in $BACKENDS; do
  f=bench/results/$label.jsonl
  rm -f "$f"; files+=("$f")
done
quiet() { grep -v -i "warning\|experimental" | grep -v '^{' | tail -3; }
for round in $(seq 1 "$ROUNDS"); do
  echo "round $round" >&2
  for label in $BACKENDS; do
    case $label in
      metal) BENCH_OUT=bench/results/metal.jsonl BENCH_LABEL=metal BENCH_ONLY=$ONLY JAX_PLATFORMS=mtl .venv/bin/python bench/jax_bench.py 2>&1 | quiet ;;
      metal-gpu) BENCH_OUT=bench/results/metal-gpu.jsonl BENCH_LABEL=metal-gpu METAL_PJRT_TRACE=1 BENCH_ONLY=$ONLY JAX_PLATFORMS=mtl .venv/bin/python bench/jax_bench.py 2>/dev/null | quiet ;;
      cpu) BENCH_OUT=bench/results/cpu.jsonl BENCH_LABEL=cpu BENCH_ONLY=$ONLY JAX_PLATFORMS=cpu .venv/bin/python bench/jax_bench.py 2>&1 | quiet ;;
      mlx) BENCH_OUT=bench/results/mlx.jsonl BENCH_ONLY=$ONLY .venv/bin/python bench/mlx_bench.py 2>&1 | quiet ;;
      *) echo "unknown backend $label" >&2; exit 1 ;;
    esac
  done
done
.venv/bin/python bench/report.py "${files[@]}" | tee bench/results/table.md
