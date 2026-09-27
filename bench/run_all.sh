#!/bin/bash
# Runs every backend ROUNDS times, interleaved (round 1: metal, metal-gpu,
# cpu, jax-mps, mlx; round 2: ...), appending to bench/results/<label>.jsonl,
# then writes the table (medians over rounds). Every row records the commit,
# versions and METAL_PJRT_*/XLA knobs (bench/common.py).
#   BENCH_BACKENDS="metal metal-gpu cpu jax-mps mlx"  (default: all)
#   BENCH_ROUNDS=3  BENCH_ONLY=<case substrings>
# Compare against MLX / jax-mps only on a freshly booted, idle machine.
set -uo pipefail
cd "$(dirname "$0")/.."
# Serialize against other GPU jobs (see scripts/device_lock.py).
if [[ -z "${JAX_OPENMETAL_LOCKED:-}" ]]; then exec env JAX_OPENMETAL_LOCKED=1 scripts/device_lock.py -- "$0" "$@"; fi
# Refuse to time a GPU that was reset since boot (it runs slow until reboot).
if [[ -z "${BENCH_ALLOW_DEGRADED:-}" ]]; then
  scripts/gpu_health.py --strict || { echo "set BENCH_ALLOW_DEGRADED=1 to run anyway" >&2; exit 1; }
fi
# The Bazel server JVM holds a lot of memory after a build; the allocation
# guard then refuses large pools (nanoGPT train step needs a 1.2 GB chunk).
command -v bazel >/dev/null && bazel shutdown >/dev/null 2>&1
mkdir -p bench/results
ONLY=${BENCH_ONLY:-}
BACKENDS=${BENCH_BACKENDS:-metal metal-gpu cpu jax-mps mlx}
ROUNDS=${BENCH_ROUNDS:-3}
files=()
for label in $BACKENDS; do
  f=bench/results/${label/jax-mps/mps}.jsonl
  rm -f "$f"; files+=("$f")
done
quiet() { grep -v -i "warning\|experimental" | grep -v '^{' | tail -3; }
for round in $(seq 1 "$ROUNDS"); do
  echo "round $round" >&2
  for label in $BACKENDS; do
    case $label in
      metal) BENCH_OUT=bench/results/metal.jsonl BENCH_LABEL=metal BENCH_ONLY=$ONLY JAX_PLATFORMS=openmetal .venv/bin/python bench/jax_bench.py 2>&1 | quiet ;;
      metal-gpu) BENCH_OUT=bench/results/metal-gpu.jsonl BENCH_LABEL=metal-gpu METAL_PJRT_TRACE=1 BENCH_ONLY=$ONLY JAX_PLATFORMS=openmetal .venv/bin/python bench/jax_bench.py 2>/dev/null | quiet ;;
      cpu) BENCH_OUT=bench/results/cpu.jsonl BENCH_LABEL=cpu BENCH_ONLY=$ONLY JAX_PLATFORMS=cpu .venv/bin/python bench/jax_bench.py 2>&1 | quiet ;;
      jax-mps) BENCH_OUT=bench/results/mps.jsonl BENCH_LABEL=jax-mps BENCH_ONLY=$ONLY JAX_PLATFORMS=mps .venv-mps/bin/python bench/jax_bench.py 2>&1 | quiet ;;
      mlx) BENCH_OUT=bench/results/mlx.jsonl BENCH_ONLY=$ONLY .venv/bin/python bench/mlx_bench.py 2>&1 | quiet ;;
      *) echo "unknown backend $label" >&2; exit 1 ;;
    esac
  done
done
.venv/bin/python bench/report.py "${files[@]}" | tee bench/results/table.md
