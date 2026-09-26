# Performance notes

Methodology: `bench/jax_bench.py` (same jitted programs on every JAX backend,
warmup, median of 10 with `block_until_ready`), `bench/mlx_bench.py` (same
workloads in MLX with `mx.compile`), `bench/latency.py` (fixed per-call
cost), `bench/run_all.sh` to produce `bench/results/table.md`. Machine: M3,
10-core GPU, 8 GB. Per-command-buffer GPU timing: `METAL_PJRT_TRACE=1`.

## Findings so far (2026-09-24)

1. **Buffer page faults dominated everything.** A fresh `MTLBuffer` costs
   about 60 us per MB on first touch (3.8 ms for 64 MB) plus 0.8 ms to
   release. With the plain platform allocator every output was a fresh
   buffer. Switching the PJRT client to XLA's BFC pool (grow on demand, no
   preallocation) took the nanoGPT training step from 3257 ms to 203 ms and
   attention forward from 58 ms to 7 ms.
2. **Emitted elementwise kernels run at memory bandwidth.** Replaying the
   emitted `x*2` kernel standalone at XLA's launch geometry gives 1.58 ms for
   128 MB of traffic (81 GB/s), the same as hand-written MSL. Threadgroup
   size, fast-math and the `char*` pointer convention make no difference.
   XLA bakes the block size into the index math, so launch geometry must
   match what XLA chose.
3. **Fixed per-execution overhead is about 0.4 ms** on top of GPU time, and
   about 165 us for tiny programs, the same as jax-mps and MLX. One execution
   is one command buffer with all its dispatches, plus an empty command
   buffer on the event stream.
4. **cumsum was 5 memory passes**: XLA's ReduceWindowRewriter builds a
   hierarchical scan out of slices and adds. `MetalScanRewriter` now turns
   minor-dim cumsum/cumprod/cummax/cummin into one `metal$scan` FFI kernel.
   GPU time for cumsum 4096x4096 f32 (METAL_PJRT_TRACE): 3.5 ms (5 kernels)
   -> 1.46 ms (1 kernel, ~88 GB/s). Threadgroup size 64-512 made no
   difference; 1024 exceeded the pipeline's limit (now clamped).
5. **softmax was 3 passes** (max reduce, sum reduce, elementwise).
   `MetalSoftmaxRewriter` now emits one `metal$softmax` kernel (online
   max/sum, row re-read from cache). GPU time for softmax 8192x1024 f32:
   1.45 ms -> 0.75 ms. Wall-clock numbers for both still need a rerun on an
   idle GPU (the 2026-09-24 runs were disturbed by other GPU jobs timing out).
6. **GEMM parity with MLX on f32** via MPS; bf16 is slower because MPS has
   no bf16 and we stage through f32 conversions.

## Where we stand vs MLX (medians, ms; lower is better)

See `bench/results/table.md` for the current full table. Highlights after
the allocator fix: nanoGPT train step 203 (MLX 252, jax-mps 276), attention
fwd+bwd 12.9 (MLX 18.6), layernorm fwd+bwd 3.6 (MLX 13.4), reductions ahead;
elementwise chain and transpose within 1.3x; cumsum 3x behind; bf16 GEMM
1.2x behind.

## Next levers, in order

1. Scans over non-minor axes and few-long-row scans (a decoupled
   look-back scan across threadgroups); logcumsumexp.
2. Softmax with a fused producer/consumer or masking (attention).
3. bf16 GEMM without staging (MLX-style steel kernels or Metal 4 primitives).
4. Untracked hazard mode with explicit barriers (CPU-side encode cost).
5. XLA launch-dimension and tiling heuristics tuned for Apple GPUs via a
   proper Apple compute capability.

## Update 2026-09-25 (early morning)

- Fused softmax and single-pass scan via FFI custom calls: kernel time 1.45 ->
  0.75 ms (softmax 8192x1024) and 3.5 -> 1.46 ms (cumsum 4096x4096). Wall
  clock not re-measured yet (see below).
- Native bf16/f16 GEMM (simdgroup-matrix kernels ported from MLX's steel
  design) is implemented but unverified on device.
- **Caveat:** killing a `pytest -n 2` run of JAX's lax_test with SIGALRM left
  worker processes stuck exiting inside the GPU driver, after which every
  Metal command buffer in every process timed out (`kIOGPUCommandBuffer
  CallbackErrorTimeout`). Measurements taken after that point are invalid; a
  reboot clears it. Lesson: never kill GPU-using test workers with signals
  while command buffers are in flight; use pytest timeouts per test instead.
- Runtime changes prompted by this: a failed command buffer is now reported
  once and the stream recovers (a stuck error state made whole test runs
  fail), and command buffers are committed after 2^29 dispatched threads so a
  batch of heavy kernels cannot approach the watchdog.

## Memory and reset policy (2026-09-25, after the wedge)

Root cause of the wedge was memory exhaustion, not a GPU fault: a jetsam
event preceded the first watchdog timeout by 40 minutes, with ~100 MB free,
2.6 GB wired, the browser holding ~5 GB and two JAX test processes each
entitled to 70% of the GPU working set. On unified memory the GPU stalls on
paged-out memory until the watchdog fires; killing workers with in-flight
work then left them stuck inside the driver.

Policy now:

- **Budget as a ceiling, guard as the protection.** The pool may grow to
  min(recommended working set, 3/4 of physical RAM); regions are only mapped
  when used. (A first version derived the budget from free memory at startup;
  on a machine busy with builds that came out at 1.5 GB and starved ordinary
  workloads, so the static budget is generous and the dynamic guard below
  does the real work.)
- **Allocation guard.** Any allocation of 1 MB or more is refused with
  RESOURCE_EXHAUSTED if it would leave less than 512 MB reclaimable, so a
  program fails cleanly instead of pushing the machine into swap.
- **Watchdog resets are sticky.** A command buffer that fails with a timeout,
  access-revoked or device-removed error marks the device lost for the
  process; all further GPU work fails with FAILED_PRECONDITION telling the
  user to restart. Fault errors (bad pointer, page fault) are reported once
  to the waiting caller and the stream recovers.
- **Smaller command buffers**: 32 dispatches or 2^27 dispatched threads.
- **One GPU job at a time.** `scripts/device_lock.py` serializes test suites,
  benchmarks and sweeps; `scripts/run_jax_tests.sh` runs JAX's tests in one
  process with per-test in-process timeouts (pytest-timeout, thread method)
  and never signal-kills workers.

## Dispatch-bound programs: decision (2026-09-26)

Small-kernel programs (JAX scans, tinygp's quasiseparable solvers) are bound
by per-dispatch cost: XLA thunk work plus ~30-40 us in our runtime per
launch, and ~1 ms per command-buffer round trip. On CUDA the same programs
are fast because XLA converts runs of thunks into CUDA graphs and replays
them; that conversion is off for us (OneAPI capability + our defaults).

Options considered:
- MPSGraph for program subsets: rejected. It is the approach Apple's
  jax-metal took; the library is opaque, has a documented record of bugs and
  unpredictable performance, and still issues roughly one kernel per op for
  the slices/concats/small matmuls these programs consist of.
- Metal-specific region emitters (associative scan as one kernel, in-kernel
  while loops for small-state loops): high leverage for scan-shaped programs
  but workload-specific. Deferred; not adding a scan primitive for now.
- StreamExecutor `CommandBuffer` implemented on Metal indirect command
  buffers: chosen. General, mirrors what XLA already does on CUDA/ROCm, and
  Metal's ICB is the native equivalent (record once, replay with one call,
  per-command barriers). Design: one argument convention for all kernels
  (pointers read from a per-executable address table, so recorded commands
  never change and `Update` is a memcpy of addresses), copies and memsets as
  compute kernels, serial barriers, untracked resources with `useResources`.
  Gate: a standalone spike must show ICB replay is clearly cheaper per
  command than direct encoding before the plumbing is built.

### ICB spike results (2026-09-26, degraded GPU; re-measure after a reboot)

`metal_pjrt_plugin/runtime/icb_spike.cc` (target `:icb_spike`, run under the
device lock) replays 10,000 dependent one-threadgroup dispatches. All variants
produced correct results, including ICB replay with `setBarrier` ordering.
Microseconds per command, range over 5 repetitions:

| variant | CPU | GPU |
|---|---|---|
| direct encoding, tracked buffers (today's path) | 0.10-0.39 | 3.2-4.7 |
| direct encoding, untracked + `useResources` | 0.11-0.18 | 2.5-4.6 |
| direct encoding, address-table kernels | 0.07-0.09 | 2.1-4.0 |
| ICB replay, buffers bound per command | 0.001-0.003 | 4.2-6.0 |
| ICB replay, address-table kernels | 0.001-0.007 | 6.0-9.0 |
| ICB record (one-time) | 0.05-0.13 | - |
| ICB rebind of every binding (`Update`) | 0.03-0.06 | - |
| address-table rewrite (`Update` as a memcpy) | 0.002-0.003 | - |

Reading: raw metal-cpp encoding costs 0.1-0.2 us per dispatch, so the
30-40 us we pay per kernel today is XLA thunk execution plus our resolve and
argument packing, not Metal encoding. An ICB with a barrier per command costs
1-2 us more GPU time per dispatch than direct encoding (3-4 us more with the
address table), and once host overhead is gone these programs are GPU-latency
bound, so ICB replay would make them slower. The gate in the decision above
was not met. Open question for the `CommandBuffer` backend: software replay
(record resolved commands, re-encode them in a tight loop on submit) keeps
the CPU win without the ICB's GPU cost; the address-table convention is still
worth keeping as an option since directly encoded table kernels were the
cheapest on the GPU. Independently of the backend, XLA patch 0005 lets
`xla_gpu_enable_command_buffer` take effect on Metal (the conversion pass
cleared every command type for OneAPI devices); `ApplyMetalDefaults` still
clears the set until a backend exists. Only `FUSION` should be enabled then:
the pass does not fall back if a backend returns Unimplemented while
recording.

## Command buffers: software replay (2026-09-26)

Implemented, following the spike: `stream_executor/metal_command_buffer.{h,cc}`
implements StreamExecutor's `CommandBuffer` on top of
`rt::CommandList` + `rt::Stream::Replay` (runtime/metal_runtime.{h,cc}).

- **Record** (`CreateLaunch`, `CreateMemcpyD2D`, `CreateMemset`,
  `CreateEmptyCmd`): arguments are converted once (StreamExecutor kernel args
  to device pointers) and resolved to `(MTLBuffer, offset)` immediately.
  Argument-buffer kernels get their GPU-address table and residency list
  built here too.
- **Submit**: one lock, one loop over the commands, encoding each into the
  stream's open command buffer with the same encoders the direct path uses
  (`EncodeLaunch`, `EncodeCopy`, `EncodeFill`). The stream's batching
  policy (op and thread caps, early commit when the GPU is idle) applies
  unchanged, so a long recorded sequence still cannot approach the watchdog.
- **Update**: XLA calls `Update` + `UpdateLaunch`/`UpdateMemcpyD2D`/
  `UpdateMemset` whenever a referenced allocation's address changed (with the
  default allocator only constants are persistent, so this is common); each
  touched command is re-resolved. Replay also re-resolves everything when the
  device's allocation generation moved (something was freed since), so a
  freed-and-reused address can never point at a stale `MTLBuffer`.
- **Ordering**: commands replay in recording order, which is thunk order and
  therefore a topological order of XLA's dependency graph; the dependency
  lists are accepted and ignored. `xla_gpu_command_buffer_scheduling_mode` is
  set to SERIALIZE to match.
- **Memsets** with a non-uniform pattern use a built-in fill kernel
  (`Device::FillKernel`) instead of the old host-callback path, which split
  the command buffer and cost a round trip.
- **Enabled command types**: FUSION only (kernel and custom-kernel thunks,
  device copies, memsets). Everything else (`CreateChildCommand`, case/while,
  host callbacks, host transfers, DNN graphs) returns Unimplemented, and XLA
  does not fall back when recording fails, so those types must stay off.
  Controls: `METAL_PJRT_COMMAND_BUFFERS=0` disables the conversion,
  `METAL_PJRT_MIN_GRAPH_SIZE=n` overrides XLA's minimum run length (5).
- Not done: Metal indirect command buffers (slower on the GPU per the spike),
  concurrent replay of independent commands.

## Reset log and kernel quarantine (2026-09-26)

Why: the scatter min/max hang (an upstream MLIR while-lowering bug, fixed in
codegen/msl_emitter.cc) was re-run by nine full lax suites over one night,
each of which reset the GPU; after that many resets the driver leaves the GPU
about ten times slower per dispatch until a reboot. Nothing stopped the next
run from doing it again. Now:

- The runtime appends every watchdog reset it observes to
  `~/.cache/jax_metal/gpu_resets.jsonl` (`METAL_PJRT_STATE_DIR` overrides)
  with the kernels that were in the command buffer that timed out. A buffer
  that only waited on another stream (as happens when the producer stream's
  batch is slow on a degraded GPU) contributes no suspects.
- `Device::CreateKernel` refuses a kernel seen in two or more resets
  (`METAL_PJRT_QUARANTINE_STRIKES`, 0 disables) with FAILED_PRECONDITION at
  executable load time, before any GPU work, naming the log. A compiler bug
  therefore costs at most two resets instead of one per run.
- `scripts/gpu_health.py` summarizes resets since boot and quarantined
  kernels; `--clear` forgets them. `bench/run_all.sh` refuses to take timings
  on a GPU reset since boot (`BENCH_ALLOW_DEGRADED=1` overrides) and
  `scripts/run_jax_tests.sh` prints the summary before a run; the runtime
  logs a warning at device creation too.
- Still unavoidable: the first reset a non-terminating kernel causes. Metal
  has no per-kernel timeout shorter than the watchdog.

### Measurements after the reboot (2026-09-26, healthy GPU)

ICB spike re-run: direct encoding 0.11 us CPU / 0.80 us GPU per dependent
tiny dispatch; ICB replay 0.003 us CPU / 1.2-1.4 us GPU. Same conclusion.

The bounded-wait change from the morning had a bug: when the fence was
already signaled but the command buffer's status was not yet Completed,
Synchronize slept 1 ms per check. Every host sync (while-loop predicates,
host FFI calls, device-to-host copies) paid it. Fixed (wait for completion
once the fence is signaled, which cannot hang). tinygp parallel solver,
value+grad, Metal vs CPU, ms:

| n | before fix | after fix, cb off | after fix, cb on | CPU |
|---|---|---|---|---|
| 1000 | 252 | 38 | 35 | 4.6 |
| 20000 | 423 | 122 | 106 | 74 |
| 200000 | 742 | 238 | 231 | 668 |

Command buffers on their own: chain of 256 tiny fusions 3.4 vs 3.9 us per
kernel; a single-kernel call still costs ~170 us end to end; a 2000-step
scan costs ~50 us per iteration either way (being profiled). Remaining
tinygp gaps: predict on the parallel solver (27 s at n=200000) and the
sequential solver (~1 ms per step) are dominated by something else; the
parallel solver's value+grad still spends most of its time in ~200 command
buffers per call, 60 of them host syncs from the host-side LAPACK FFI
(getrf + two triangular solves per 4x4 batched solve) and small argsorts.
