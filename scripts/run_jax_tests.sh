#!/bin/bash
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
# Usage: scripts/run_jax_tests.sh tests/lax_test.py [pytest args]
set -uo pipefail
cd "$(dirname "$0")/.."
T=${JAX_TESTS_DIR:-$HOME/.cache/metal-pjrt/jax-tests}
FILE=${1:?test file relative to the jax repo, e.g. tests/lax_test.py}; shift
scripts/gpu_health.py >&2  # warn about resets / quarantined kernels
exec scripts/device_lock.py -- env JAX_PLATFORMS=mtl,cpu JAX_NUM_GENERATED_CASES=${JAX_NUM_GENERATED_CASES:-3} JAX_ENABLE_X64=0 \
  JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH="$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}" \
  .venv/bin/python -m pytest "$T/$FILE" -p jax_tests_plugin -p no:cacheprovider -n 0 \
  -p no:timeout -o faulthandler_timeout=${PYTEST_TIMEOUT:-180} -q --tb=no -rN "$@"
