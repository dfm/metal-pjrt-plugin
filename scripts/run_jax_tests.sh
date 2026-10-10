#!/bin/bash
# Copyright 2026 The metal-pjrt-plugin Authors
# SPDX-License-Identifier: Apache-2.0

# Run files from JAX's own test suite on the Metal backend, safely:
#  - one process, no xdist (concurrent GPU processes over-commit memory)
#  - no timeout that exits: pytest-timeout (either method) ends the process
#    with GPU work in flight, which is how the driver got wedged once. A slow
#    test gets a stack dump (faulthandler_timeout) and runs on; GPU hangs end
#    with the runtime's bounded waits (watchdog -> error)
#  - device lock so benchmarks/sweeps cannot run at the same time
#  - no persistent compilation cache: JAX's tests expect none
#    (LaxTest::testAsarray0 fails otherwise), and runs stay hermetic
# scripts/jax_tests_plugin.py refuses to run unless the default backend is
# mtl and compares failures with scripts/jax_known_failures/<file>.txt:
# the exit status is non-zero only for new failures.
# Tests are labelled as run on "gpu" (JAX_TESTS_DUT=gpu, the default; set it
# empty for JAX's unlabelled behaviour, where many tests skip on mtl).
# PYTHON picks the interpreter (default .venv/bin/python; scripts/jax_grid.py
# uses one venv per JAX version) and JAX_TESTS_DIR the JAX checkout, whose
# tests must be those of the installed jax (jax.version._git_hash).
# Usage: scripts/run_jax_tests.sh tests/lax_test.py [pytest args]
set -uo pipefail
R=$(cd "$(dirname "$0")/.." && pwd)
T=${JAX_TESTS_DIR:-$HOME/.cache/metal-pjrt/jax-tests}
PY=${PYTHON:-$R/.venv/bin/python}
FILE=${1:?test file relative to the jax repo, e.g. tests/lax_test.py}; shift
"$R/scripts/gpu_health.py" >&2  # warn about resets / quarantined kernels
if ! "$PY" -c "import absl, hypothesis" 2>/dev/null; then
  echo "run_jax_tests.sh: JAX's tests need absl-py and hypothesis in $PY:" >&2
  echo "  uv pip install --python $PY absl-py hypothesis" >&2
  exit 2
fi
# From the JAX checkout: `python -m pytest` puts the working directory first
# on sys.path, and in this repository that would be the source frontend
# instead of an installed wheel under test.
cd "$T"
exec "$R/scripts/device_lock.py" -- env -u JAX_PLATFORMS JAX_NUM_GENERATED_CASES=${JAX_NUM_GENERATED_CASES:-3} JAX_ENABLE_X64=0 \
  JAX_ENABLE_COMPILATION_CACHE=false JAX_TESTS_DUT=${JAX_TESTS_DUT-gpu} \
  PYTHONPATH="$R/scripts${PYTHONPATH:+:$PYTHONPATH}" \
  "$PY" -m pytest "$FILE" -p jax_tests_plugin -p no:cacheprovider -p no:xdist \
  -p no:timeout -o faulthandler_timeout=${PYTEST_TIMEOUT:-180} -q --tb=no -rN "$@"
