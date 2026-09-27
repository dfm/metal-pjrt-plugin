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
   (2026-09-27: the softmax rewriter was removed after an end-to-end A/B;
   see "Metal-only rewrites: end-to-end A/B" below.)
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
  process and never kills or exits it on a timeout: a slow test gets a stack
  dump (`faulthandler_timeout`) and GPU hangs end with the runtime's bounded
  waits. (pytest-timeout was dropped: both its methods end the process with
  GPU work in flight.)

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
  `~/.cache/openmetal/gpu_resets.jsonl` (`METAL_PJRT_STATE_DIR` overrides)
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
- Narrowed (2026-09-26, night): the runtime's built-in fill/copy kernels
  (in nearly every command buffer) are listed under `"builtins"` and never
  struck or refused, and strikes count only for resets since boot, so a
  reboot lifts a quarantine. Each record carries `"build"` (the dylib's
  LC_UUID) for diagnostics only: it changes with every unrelated rebuild, so
  keying strikes on it would let a twice-struck kernel reset the GPU again
  after any rebuild. A kernel whose source changes gets a new key anyway; a
  fix outside the kernel source (runtime, launch dimensions) needs
  `scripts/gpu_health.py --clear`. Not done: per-encoder blame via
  `MTLCommandBufferDescriptor.errorOptions = EncoderExecutionStatus`. It
  costs nothing measurable (`bench/dispatch_bench` raw round trip, 1
  dispatch: 96-107 us with it vs 97-111 us without, GPU time 1.8-2.1 vs 1.8
  us), but it reports a state per encoder and a command buffer is mostly one
  serial compute encoder (splitting encoders was a measured cost), so it
  cannot narrow blame within the usual buffer; what it reports on a watchdog
  timeout could not be checked without causing one.

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

## Dispatch-bound programs, round two (2026-09-26, afternoon)

Profiling a 2000-step scan (`bench/dispatch_bound.py`) with `sample` showed
the main thread 80% blocked on the GPU and Metal's own command-queue
submission thread 98% busy: the early-commit policy produced ~700 command
buffers per call, and the driver spends ~150 us of a dedicated thread
submitting each one. Copies (dynamic-slice thunks) also alternated blit and
compute encoders inside every iteration, and each encoder boundary is a
separate GPU "kick" (~10 us of GPU time). Three changes:

- **Time-paced early commits**: at most one early commit per 500 us
  (`kEarlyCommitIntervalUs`, `METAL_PJRT_EARLY_COMMIT_US`); syncs and the
  op/thread caps still commit immediately.
- **Copies and fills as compute kernels** up to 16 MB (`Device::Builtin`
  kernels), so a stream of kernels and copies stays in one compute encoder;
  blit for larger ones.
- **Small sorts unrolled** in MetalSortExpander (sort dimension <= 64: every
  substage is a fusion, no while loop). LU pivot inversion in
  `jnp.linalg.solve` sorts rows of J elements.
- **Small dense linalg on the GPU**: matrices up to 32x32 in
  `metal$cholesky`, `metal$triangular_solve` and `metal$lapack_getrf` run
  as one GPU thread per matrix (per right-hand-side column for solves)
  instead of a synchronizing host LAPACK call (`METAL_PJRT_SMALL_LINALG=0`
  restores the host path). A batched 4x4 `solve` cost three GPU round trips.

Results (ms; Metal vs CPU):

| program | before | after |
|---|---|---|
| scan, 2000 steps, tiny state | 100 | 7.4 |
| tinygp parallel value+grad n=1000 | 35 | 3.2 (CPU 4.9) |
| tinygp parallel value+grad n=20000 | 106 | 6.8 (CPU 77) |
| tinygp parallel value+grad n=200000 | 231 | 68 (CPU 703) |
| tinygp parallel predict n=200000 | 27000 | 1960 (CPU 295) |
| tinygp sequential value+grad n=20000 | 18700 | 1670 (CPU 64) |

GPU time per op is now ~1 us for these loops (0.8 us is the floor measured
for dependent one-threadgroup dispatches), and the sequential programs are
GPU-bound there: `predict` and the sequential solver run ~10 tiny kernels per
element in a scan. Going further means fewer, fatter kernels per iteration
(fusing a scan body into one kernel, or in-kernel loops), which is the
deferred region-emitter work, not runtime overhead.

## Command buffers: removed (2026-09-26, evening)

With the runtime fixes above in place, the software-replay `CommandBuffer`
was re-measured against thunk-by-thunk execution on the same build: nanoGPT
train step 184 vs 196 ms (6% better), tinygp parallel value+grad n=200000
52 vs 54 ms (a wash), and the loop-heavy programs 5-30% *worse* (scan 8.1 vs
6.2 ms, tinygp n=1000 3.1 vs 2.6 ms) because XLA's command-buffer thunk does
its own per-execution bookkeeping on top of the replay, which now costs more
than the ~3 us launches it replaces. The design argument for it rested on a
per-kernel host cost of 30-40 us that turned out to be the Synchronize
sleep and the command-buffer submission storm, not launch overhead.

Removed: `stream_executor/metal_command_buffer.{h,cc}`, the runtime
`CommandList`/`Replay`, XLA patch 0005, the Bazel visibility override, and
the ICB spike binary (its numbers stay recorded above). Kept: the shared
encoders, built-in copy/fill kernels, time-paced commits, unrolled small
sorts and GPU small-matrix linalg, which are what actually moved the numbers.
`ApplyMetalDefaults` clears the command types as before.

## Lazy waits (2026-09-26, evening)

A command buffer that only waits on an event still counts its waiting time
against the GPU watchdog; on a slow producer it times out on its own, which
is what the third reset of the day was. Waits (WaitForEvent, WaitForStream,
the wait for a host task) are now recorded and encoded lazily, immediately
before the next GPU work on the stream. A Synchronize with pending waits
satisfies them on the host, and a host task inherits the waits pending when
it was enqueued. Result: no wait-only command buffers exist at all. tinygp
parallel value+grad at n=1000: 97 command buffers per call (8 wait-only)
-> 6 (none), same timings otherwise; the JAX device-to-host pattern (wait
for the compute stream, copy, block) no longer costs a GPU round trip.

Hold rule (2026-09-26, night): lazy encoding still let a command buffer
wait on the GPU for a host task that had not run yet (a slow host task then
counts against the watchdog). `Stream::Commit` now waits on the host for
any unsignaled host-task value the buffer waits on before committing it.
Host-task workers never take the stream lock, so this cannot deadlock with
them; a host task that waits for its own stream (Synchronize, an event
recorded after it, a commit waiting for it) gets FAILED_PRECONDITION
instead of hanging. `Device::unsignaled_host_task_waits_committed()` counts
violations and the runtime tests assert it stays 0.

Bug fixed with it: `WaitForStream(other)` waited only for `other`'s last
command buffer, not for a host task enqueued after it, so "A: host-to-device
copy (a host task); B: wait for A; B: kernel" could run the kernel before
the copy (silent wrong results; XLA orders transfers this way in
`pjrt_stream_executor_client.cc`, `transfer_manager.cc` and
`local_device_state.cc`). It now waits for the highest value `other` has
issued, and inherits the waits `other` has not encoded yet (A waits for C,
B waits for A with nothing launched on A in between: B used to run before
C's host task). `RecordEvent` publishes the event's value only after its
buffer is committed, so no one can wait on a value whose signaling buffer
is still held behind a host task.

## GPU errors through events (2026-09-26, night)

A failed command buffer used to force-signal its events and report OK to
every waiter but its own stream's Synchronize, so PJRT consumed a faulted
kernel's output as valid. Now every value signaled on a stream fence or an
Event has a status (`rt::Device` value table; `metal_runtime.h` has the
rules): a buffer or host task fails if it failed itself, if a value it waited
for failed, or if the previous value on its stream failed. Host waits and
host tasks see that status; a host task ordered after failed work does not
run and hands the error to its `error_cb`, which is how XLA marks result
buffers' definition events failed, so `block_until_ready`, `np.asarray` and
dependent computations raise (`INTERNAL: Metal command buffer failed ...
[executable_name=...]`). `MetalEvent::PollForStatus` returns kError.

- Sticky for the device: watchdog timeouts, revoked/removed devices (as
  before) and page faults. Everything else (out of memory, invalid resource,
  ...) fails the buffer, what waits for it, and everything later on its
  stream (same-stream dependence is by stream order), until a Synchronize
  (`BlockHostUntilDone`) on that stream reports it. **PJRT waits on events
  and never calls BlockHostUntilDone on the compute stream, so under JAX one
  such failure fails every later execution on that stream until the process
  restarts: effectively sticky.** Cutting the chain earlier would need
  data-flow tracking; CUDA makes such errors fatal for the whole context.
- Completion handlers never block: a handler records its buffer's own
  status and the values it depended on, and the final status is resolved
  lazily (memoized) by whoever asks, so a handler can never wait on another
  handler queued behind it. A host task ordered after failed work runs
  anyway when it has no error callback (XLA's plain callbacks free memory or
  complete transfers) and passes the error on; with one, it is skipped and
  the callback gets the error.
- Known limitation: a dependency on a value whose Event or Stream was
  already destroyed resolves as OK (its records go with it), so the error is
  lost if a failed stream or Event dies before its dependents are queried;
  rare with PJRT, which keeps its streams and pooled events alive.
- `BlockHostUntilDone` no longer puts the StreamExecutor stream into its
  error state on a GPU failure (as on CUDA; XLA CHECKs `ok()` on pooled
  streams); a failing host callback without `error_cb` still does.
- Cost: a host query of a signaled value waits for the completion handler
  that settles it. bench/latency.py, interleaved A/B in one build: jit(x*2+1)
  median 173-176 us vs 157-168 us without the wait, two-kernel program 162
  vs 143-152 us (+8-12 us per synchronizing round trip); dispatch_bound.py
  unchanged. A fast path (treat signaled values as OK while no failure was
  ever recorded) would remove it, but it assumes a failing buffer never runs
  its trailing signals before its handler, which may not hold for page
  faults (the GPU may continue past a faulting access). Kept the sound
  version; the fast path is a 5-line change if the latency matters more.
- Tests: runtime tests inject failures (`FailNextCommandBufferForTesting`,
  no real GPU fault); `metal_executor_test` checks error_cb, events and
  BlockHostUntilDone; `tests/test_gpu_errors.py` runs a JAX program with
  `METAL_PJRT_FAIL_COMMAND_BUFFER=n` (testing only: the n-th committed
  command buffer counts as failed) for n = 1..8 and checks that every step
  either returns correct values or raises, and that each failure surfaces.

## Persistent compilation cache (2026-09-26, night)

JAX's persistent compilation cache now works for "openmetal" (how:
`docs/integration-notes.md`, PJRT client). First call in a fresh process
(jit + compile or cache load + one run, `block_until_ready`), healthy GPU,
Metal system shader cache warm; cache written with
`JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0`:

| program | no cache | cold (miss + write) | warm (hit) | entry |
|---|---|---|---|---|
| MLP train step 784-512-512-10, batch 128 | 64-68 ms | 72 ms | 41-42 ms | 39 KB |
| nanoGPT train step (bench/jax_bench.py) | 695-816 ms | 788 ms | 418-431 ms | 714 KB |

Steady-state nanoGPT step is ~190 ms, so a warm first call is roughly half
executable load (deserialize, `newLibraryWithSource` hitting the system
shader cache, buffer setup) and half the run; the hit saves ~350 ms of XLA
compilation. Process start to first MLP result, which also covers the ~20
small jits of parameter init, went 1463 -> 455 ms. The lax coverage sweep (now
`tests/test_lax.py`) ran in 7 s cold, 2 s warm. With the very first Metal shader compile in a
boot (system shader cache cold) the MLP first call was 541 ms without the
cache.

Caveat: JAX only writes entries whose compile took longer than
`jax_persistent_cache_min_compile_time_secs` (default 1 s). Both programs
above compile in under a second on Metal, so with the defaults they are
not cached; the plugin does not change that threshold.

## Metal-only rewrites: end-to-end A/B (2026-09-27)

`METAL_PJRT_DISABLE_REWRITES=softmax,scan` vs default, same build, arms
interleaved, 3 rounds, medians in ms (min..max of the 3 round medians).
Which FFI target fires is read from the compiled HLO.

| case | rewrites on | off | off/on | fires |
|---|---|---|---|---|
| cumsum 4096x4096 | 1.77-1.78 | 3.82-4.20 | 2.15 | scan |
| cumprod 4096x4096 | 1.77-1.80 | 3.79-3.80 | 2.13 | scan |
| cumsum bf16 4096x4096 | 1.05-1.06 | 3.43 | 3.24 | scan |
| cumsum fwd+bwd 4096x4096 | 4.67-4.69 | 6.68-6.71 | 1.43 | scan |
| cumsum / cumprod 65536x64 (short rows) | 0.69-0.88 | 1.13-1.36 | 1.5-1.6 | scan |
| cumsum / cumprod 256x16384 | 0.69-0.90 | 1.22-1.43 | 1.75-1.94 | scan |
| cumsum 8x262144, 1x1M, axis 0 | = | = | 1.00 | - |
| logcumsumexp (all shapes) | = | = | 1.00 (1x1M 1.18) | - |
| softmax / log_softmax 65536x64 | 0.66-0.70 | 1.04-1.07 | 1.5 | softmax |
| softmax / log_softmax 8192x1024 | 1.05-1.06 | 1.74-1.91 | 1.7 | softmax |
| softmax / log_softmax 1024x16384 | 1.78-1.80 | 3.15-3.35 | 1.8 | softmax |
| softmax fwd+bwd 8192x1024 | 2.08-2.10 | 2.07-2.10 | 1.00 | - |
| attention fwd 8x8x512x64 | 5.40-5.48 | 5.36-5.40 | 0.99 | softmax |
| attention fwd+bwd | 11.9-12.1 | 12.1-12.3 | 1.01 | - |
| nanoGPT fwd (loss) | 59.8-60.3 | 59.5-60.4 | 1.00 | softmax (6) |
| nanoGPT train step | 182.4-186.6 | 181.5-184.1 | 0.99 | none |
| tinygp quasisep / parallel 20000 value+grad | 1638 / 6.1 | 1639 / 6.1 | 1.00 | none |
| rest of bench/jax_bench.py, dispatch_bound.py | | | 0.95-1.09, noise | - |

The softmax matcher needs every intermediate (max, exp, sum) to have no
outside users, so it never fires under autodiff: the nanoGPT train step
(111 custom calls, all GEMMs) and every fwd+bwd case keep XLA's fusions.
Where it fires inside a real program (attention and nanoGPT forward) the
gain is zero; it only wins on a standalone softmax. Accuracy (ulps against
f64, max over tests/test_fused_kernels.py shapes, on / off / CPU f32):
f32 softmax 30.6 / 30.6 / 30.3, log_softmax 1.04 / 1.03 / 1.04; f16
softmax 5.0 / 9.5 / 9.5; bf16 equal. Decision: **softmax rewriter deleted**
(roadmap 2.3), with its FFI handler and tests. The scan rewriter wins 1.4-3.2x
on every minor-dim cumsum/cumprod it matches, fwd and bwd, and is at least
as accurate (f16 cumsum 1.15 vs 2.08 ulps, bf16 0.78 vs 1.51, f32 equal):
**kept**.

## Benchmark hygiene (2026-09-27)

- `bench/run_all.sh` runs `BENCH_ROUNDS` (default 3) rounds with the
  backends interleaved inside each round (`BENCH_BACKENDS`, default `metal
  metal-gpu cpu jax-mps mlx`); `bench/report.py` takes the median over
  rounds. Every row records the commit (`git describe --dirty`), JAX / MLX
  versions, the plugin's platform version and the `METAL_PJRT_*`,
  `JAX_OPENMETAL_*`, `XLA_FLAGS`, `JAX_PLATFORMS` knobs; the table header lists
  them. MLX and jax-mps were not re-run (that needs a freshly booted idle
  machine); their columns in `bench/results/table.md` are the older runs
  and are marked as having no metadata.
- GPU time: `metal-gpu` is a second metal pass with `METAL_PJRT_TRACE=1`;
  `common.gpu_ms` captures fd 2 while it runs the case 5 more times and sums
  the command buffers' `GPUEndTime - GPUStartTime` from the trace lines. A
  separate pass because the trace adds ~0.1-0.3 ms to sub-millisecond wall
  times. No runtime change was needed. TFLOPS (at the best wall time) for
  GEMMs, attention (the two einsums; 3x for fwd+bwd), cholesky (n^3/3),
  solve (2n^3/3 + 2n^2 k) and qr (8n^3/3, R and Q).
- Linear algebra cases (f32): cholesky, solve, qr at 128 / 512 / 2048, eigh
  at 128 / 512. All run on the host through the LAPACK FFI (GPU time ~0).
  Metal pays ~0.7-0.8 ms per call at n = 128 (cholesky 0.86 vs 0.03 ms on
  CPU, GPU time 0.03 ms, so the cost is host-side; not profiled, the
  synchronous round trip around the LAPACK call is the suspect, roadmap
  3.1), and matches CPU at 2048 (cholesky 6.8 vs 8.6, qr 111 vs 112 ms).
- softmax 8192x1024 is now 1.91 ms (was 1.24 with the deleted rewriter;
  MLX 0.97), the price of the decision above for standalone softmax.

### dispatch_bound chain64 is bimodal: GPU performance state

`METAL_PJRT_TRACE=1`, one line per call (`bench/dispatch_bound.py`'s
chain, 150 calls each): the command-buffer split (16 ops at the first
early commit, then 112) is the same in both modes, but the GPU time per
dispatch is either ~4.2 us or ~2.3 us:

| chain | calls slow / fast | GPU us per dispatch (slow / fast) | wall us, median (slow / fast) |
|---|---|---|---|
| 16 steps (32 dispatches) | 150 / 0 | 4.3 / - | 373 / - |
| 64 steps (128) | 127 / 23 | 4.1 / 2.3 | 753 / 512 (traced) |
| 256 steps (512) | 0 / 150 | - / 1.8 | - / 1459 (traced) |

Forcing one command buffer per call (`METAL_PJRT_EARLY_COMMIT_US=1000000`)
or an early commit every time (`=0`) does not remove it (both modes appear
under `=0`: 252 vs 456 us GPU for the same 112 ops). So it is not the
500 us commit pacing but the GPU clock / performance state, which macOS
picks from the duty cycle: a 128-dispatch burst every ~0.5-0.8 ms sits at
the governor's threshold, 32 dispatches never ramp it, 512 always do. There
is no public API to pin it. Fix in the benchmark, not the runtime:
`dispatch_bound.py` now times 200 calls and prints p10 / median / p90 with
the best, and its docstring says to compare A/B arms on all of them (the
median alone flips between the modes run to run: 409 vs 510 us back to
back on the same build).

### Kernels launched with one thread per threadgroup (report only)

The "22 of 125" count came from a whole-process dump (the benchmark's
parameter init included). By module: 21 of the 22 are JAX's PRNG setup
(`jit__normal`, `jit__threefry_seed`, `jit__threefry_fold_in`: scalar key
arithmetic and tiny concatenates, run once at init). The nanoGPT train step
itself has 51 emitted kernels (plus 111 GEMM custom calls), and exactly one
is single-threaded: `loop_negate_fusion`, `f32[]` = -mean(loss), one
element. It costs one dispatch (~2-4 us of GPU time) in a ~180 ms step.
Nothing to parallelize.

## GEMM epilogue inside steel (2026-09-27, roadmap 2.1)

BlasLt epilogues (bias, ReLU, GELU, SiLU, aux) are applied in steel's
`store_result` instead of a second kernel over D: v = alpha AB + beta C +
bias[col], aux = v, D = act(v), in f32 with one rounding. GEMMs with an
epilogue go to steel for f16/bf16 (as before) and now for f32 up to
batch*m*n*k = 2^33; larger f32 ones and `METAL_PJRT_GEMM=mps` keep MPS + the
second pass. f32 without an epilogue stays on MPS.

A/B, same machine, 2-3 interleaved rounds, 30 calls each, p10 / median /
p90 ms, "before" = MPS + second pass for f32, steel + second pass for bf16:

| case (f32) | before | fused steel |
|---|---|---|
| relu(xW+b) 128x256x256 | 0.26-0.42 / 0.42-0.43 / 0.44-0.48 | 0.24-0.40 / 0.24-0.41 / 0.41-0.42 |
| relu(xW+b) 512x1024x1024 | 0.74-0.76 / 0.79-0.97 / 0.98-1.03 | 0.68-0.71 / 0.88-1.00 / 0.94-1.32 |
| relu(xW+b) 2048x1024x4096 | 6.81-6.90 / 6.83-6.91 / 6.85-7.10 | 6.50-6.86 / 6.53-6.91 / 6.64-7.36 |
| gelu(xW+b) 2048x1024x4096 | 6.78-6.90 / 6.84-6.91 / 6.85-6.94 | 6.67-6.86 / 6.70-6.89 / 6.71-6.91 |
| xW+b 2048x1024x4096 | 6.81-6.88 / 6.85-6.91 / 6.91-6.95 | 6.49-6.58 / 6.50-6.61 / 6.52-6.71 |
| relu(xW+b) 8192x1024x1024 | 6.82-6.88 / 6.87-6.91 / 6.90-7.01 | 6.92-7.02 / 6.97-7.07 / 7.00-7.32 |
| relu(xW+b) 1024x1024x1024 | 1.13-1.17 / 1.20 / 1.21-1.22 | 1.10-1.11 / 1.12-1.13 / 1.13-1.14 |
| relu(xW+b) 4096x1024x4096 (steel, no threshold) | 13.38 / 13.42-13.44 / 13.51 | 13.61-13.87 / 13.68-14.08 / 13.74-14.95 |
| relu(xW+b) 4096x2048x4096 (idem) | 24.88-24.91 / 24.90-25.04 / 25.05-25.17 | 26.80 / 26.88-27.20 / 27.35-27.41 |
| relu(xW+b) 2048x4096x4096 (idem) | 24.38-24.43 / 24.45-24.52 / 24.55-24.89 | 27.62-27.63 / 27.70-27.83 / 28.00-28.22 |
| relu(xW+b) 8192x4096x1024 (idem) | 24.33-24.36 / 24.38-24.44 / 24.51-24.66 | 28.07-28.23 / 28.14-28.66 / 28.31-28.86 |
| relu(xW+b) 4096x4096x4096 (idem) | 48.3-49.3 / 48.4-49.6 / 49.1-50.2 | 55.4-56.3 / 55.6-56.5 / 56.8-57.3 |
| MLP 2-layer fwd+bwd 2048x1024x4096 | 33.2-33.4 / 33.4-33.5 / 33.4-33.7 | 33.6-34.1 / 33.8-34.6 / 34.2-35.4 |
| nanoGPT fwd (loss), median | 58.4-61.8 | 56.5-57.6 |
| nanoGPT train step, median | 180.4-185.6 | 180.3-181.9 |

steel's f32 GEMM is 2-15% slower than MPS's from batch*m*n*k ~ 1.7e10 up,
more than the second pass saves, hence the 2^33 threshold; with it those
shapes measure as before (4096x1024x4096 13.43 vs 13.43, 8192x4096x1024
24.50 vs 24.46). At 2^33 the shape decides (2048x1024x4096 3-5% faster,
8192x1024x1024 ~1.5% slower). In the MLP fwd+bwd only the forward GEMMs have
an epilogue; the ~1% there is within its round-to-round spread.

| case (bf16) | before (steel + pass) | fused |
|---|---|---|
| relu(xW+b) 512x1024x1024 | 0.88-0.91 median | 0.71-0.77 |
| relu(xW+b) 2048x1024x4096 | 6.19 | 5.88-5.90 |
| gelu(xW+b) 2048x1024x4096 | 6.35 | 6.06-6.07 |
| relu(xW+b) 8192x1024x1024 | 6.27-6.28 | 5.92 |
| relu(xW+b) 4096x4096x4096 | 45.35-45.37 | 44.63-44.75 |
| MLP 2-layer fwd+bwd | 29.56-29.62 | 29.20-29.29 |

Numerics: the f32 outputs of fused steel and MPS + second pass are bitwise
identical (gelu(xW+b) at 64x48x96, 512x1024x256, 2048x1024x4096; and every
f32 line of the METAL_TEST_REPORT_ULPS=1 pytest report): both accumulate
in f32 in the same order, and an f32 D holds the accumulator exactly, so
the second pass never added a rounding. f16/bf16 improve because the bias
is added before the single rounding (normwise ulps, pass -> fused): bias
0.74 -> 0.50 (f16), 0.52 -> 0.49 (bf16); bias+gelu 0.70 -> 0.49 / 0.71 ->
0.48; test_steel_gemm "+bias relu" 0.61-0.93 -> 0.34-0.50 (CPU float32
0.61-0.93).
