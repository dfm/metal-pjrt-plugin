#!/bin/bash
# Runs every backend and writes bench/results/<label>.jsonl, then a table.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p bench/results
ONLY=${BENCH_ONLY:-}
for label in metal cpu; do
  rm -f bench/results/$label.jsonl
  BENCH_OUT=bench/results/$label.jsonl BENCH_LABEL=$label BENCH_ONLY=$ONLY JAX_PLATFORMS=$label .venv/bin/python bench/jax_bench.py 2>&1 | grep -v -i "warning\|experimental" | tail -3
done
rm -f bench/results/mps.jsonl
BENCH_OUT=bench/results/mps.jsonl BENCH_LABEL=jax-mps BENCH_ONLY=$ONLY JAX_PLATFORMS=mps .venv-mps/bin/python bench/jax_bench.py 2>&1 | grep -v -i "warning\|experimental" | tail -3
rm -f bench/results/mlx.jsonl
BENCH_OUT=bench/results/mlx.jsonl BENCH_ONLY=$ONLY .venv/bin/python bench/mlx_bench.py 2>&1 | tail -3
.venv/bin/python bench/report.py bench/results/metal.jsonl bench/results/mps.jsonl bench/results/mlx.jsonl bench/results/cpu.jsonl | tee bench/results/table.md
