#!/bin/bash
# Overnight "build spike": measure whether XLA's PJRT + GPU compiler stack builds
# on macOS arm64 without CUDA, and how long / how much memory it takes on this
# machine. Stages are ordered from "certainly builds" to "the open question",
# and --keep_going means a failure in one target does not stop the others.
#
# Usage:  caffeinate -is nohup scripts/build_spike.sh > /dev/null 2>&1 &
# Logs:   build-logs/spike-<timestamp>.log (build) and .mem.log (memory samples)
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p build-logs
STAMP=$(date +%Y%m%d-%H%M)
LOG=build-logs/spike-$STAMP.log
MEMLOG=build-logs/spike-$STAMP.mem.log
exec > >(tee -a "$LOG") 2>&1

EXTRA=${BAZEL_EXTRA:---config=public_cache}

echo "=== build spike start $(date)"
sw_vers | tr '\n' ' '; echo
clang --version | head -1
echo "RAM: $(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 )) GB, cores: $(sysctl -n hw.ncpu)"
echo "extra bazel flags: $EXTRA"

# Memory sampler: total RSS of all processes, swap usage, every 60 s.
(
  while true; do
    printf '%s rss_total_mb=%s swap=%s\n' "$(date +%T)" \
      "$(ps -A -o rss= | awk '{s+=$1} END {printf "%d", s/1024}')" \
      "$(sysctl -n vm.swapusage | awk '{print $6}')"
    sleep 60
  done
) >> "$MEMLOG" 2>&1 &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

stage() {
  local name=$1; shift
  echo; echo "=== STAGE $name start $(date)"; echo "targets: $*"
  local t0=$SECONDS
  bazel build --keep_going $EXTRA "$@"
  local rc=$?
  echo "=== STAGE $name exit=$rc elapsed=$(( (SECONDS - t0) / 60 )) min $(date)"
}

# Stage 1: platform-agnostic core that jaxlib's own macOS CPU build already
# compiles: PJRT C API wrapper, StreamExecutor client, HLO passes, StableHLO.
stage core \
  @xla//xla/pjrt/c:pjrt_c_api_wrapper_impl \
  @xla//xla/pjrt:pjrt_stream_executor_client \
  @xla//xla/stream_executor:platform_manager \
  @xla//xla/stream_executor:stream_executor_h \
  @xla//xla/hlo/pass:hlo_pass_pipeline \
  @xla//xla/service:compiler \
  @xla//xla/hlo/translate:stablehlo

# Stage 2: the open question. Does the GPU compiler and its client build with
# no CUDA/ROCm/SYCL configured?
stage gpu \
  @xla//xla/service/gpu:gpu_compiler \
  @xla//xla/pjrt/gpu:se_gpu_pjrt_client \
  @xla//xla/pjrt/c:pjrt_c_api_gpu_internal

echo; echo "=== disk cache size: $(du -sh ~/.cache/metal-pjrt-plugin/bazel-disk 2>/dev/null | cut -f1)"
bazel shutdown
echo "=== build spike end $(date)"
