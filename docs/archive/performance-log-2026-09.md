# Performance notes

> Archived: a dated log (September 2026), not maintained. Names, environment
> variables and files in it are those of its date, and several were renamed
> or removed since. Current numbers are in `docs/performance.md`.

Methodology: `bench/jax_bench.py` (same jitted programs on every JAX backend,
warmup, median of 10 with `block_until_ready`), `bench/mlx_bench.py` (same
workloads in MLX with `mx.compile`), `bench/dispatch_bound.py` (fixed
per-call cost and chains of tiny kernels; it absorbed `bench/latency.py`),
`bench/run_all.sh` to produce `bench/results/table.md`. Machine: M3,
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
  fail; superseded: any GPU failure is now sticky for the process, see "GPU
  errors: sticky for the process" below), and command buffers are committed after 2^29 dispatched threads so a
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
  min(recommended working set, 3/4 of physical RAM; since then 1/2); regions
  are only mapped when used (since 2026-09-27 the runtime enforces it, see
  "Memory policy" below). (A first version derived the budget from free memory at startup;
  on a machine busy with builds that came out at 1.5 GB and starved ordinary
  workloads, so the static budget is generous and the dynamic guard below
  does the real work.)
- **Allocation guard.** Any allocation of 1 MB or more is refused with
  RESOURCE_EXHAUSTED if it would leave less than 512 MB reclaimable, so a
  program fails cleanly instead of pushing the machine into swap
  (`METAL_PJRT_SYSTEM_MEMORY_RESERVE_MB` changes the 512, for tests).
- **Watchdog resets are sticky.** A command buffer that fails with a timeout,
  access-revoked or device-removed error marks the device lost for the
  process; all further GPU work fails with FAILED_PRECONDITION telling the
  user to restart. Fault errors (bad pointer, page fault) are reported once
  to the waiting caller and the stream recovers. (Since 2026-09-27 every
  GPU failure is sticky; see "GPU errors" below.)
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
  (Superseded: command buffers and both knobs were removed; see "Command
  buffers: removed" below.)
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
  (`kEarlyCommitIntervalUs`); syncs and the
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
  instead of a synchronizing host LAPACK call. A batched 4x4 `solve` cost
  three GPU round trips.

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

## GPU errors: sticky for the process (2026-09-27)

Any GPU failure is sticky for the process; restart it. The first failed
command buffer, whatever the error (watchdog timeout, page fault, out of
memory, invalid resource, ...), or a failed host callback without an
`error_cb`, sets the device's error (`rt::Device::error()`, like a CUDA
context error). From then on every `BlockHostUntilDone`, event wait and
`PollForStatus` (kError) and every new launch returns it, and a host
callback with an `error_cb` does not run: the `error_cb` gets the error,
which is how XLA marks result buffers' definition events failed, so
`block_until_ready`, `np.asarray` and later computations raise (`INTERNAL:
Metal command buffer failed ... (Metal device 0; no further GPU work is
accepted in this process, restart it)`, or XLA's own "Unknown predetermined
error" for buffers whose definition failed). A host callback without an
`error_cb` still runs (XLA's plain callbacks free memory or complete
transfers). The StreamExecutor stream error state (`ok()`) is never set, as
on CUDA: XLA CHECKs it on pooled streams. This is what PJRT effectively did
before anyway: it never synchronizes its compute stream, so a failure used
to fail every later execution on that stream too.

What keeps a failure from hanging or crashing anything:

- A failed buffer's completion handler records the error, then
  force-signals the buffer's fence and events, so no waiter hangs.
  Handlers never block.
- Host waits are bounded: they give up once the device has failed (work
  committed after a reset may never run).
- No CHECK or abort on any error path.
- Watchdog timeouts and revoked/removed devices are still recorded in the
  reset log and feed the kernel quarantine (the only place the error code
  still matters).

Soundness (no wrong values): a host waiter learns of a failure without
waiting for completion handlers, which Metal does not order across queues.
After it sees a signal it checks the command buffers still in flight on the
device (`Device::CheckInFlight`): any buffer whose fence value is signaled
has run to its end, so its final status is waited for (`waitUntilCompleted`)
and an error recorded. Buffers signal their fence before their events, so a
waiter that saw an event value covers the buffer that signaled it, and any
buffer on another queue it depended on (runtime test
`EventSignalsFollowTheFenceSignal` guards the order).

Replaced: batch B's per-value status table (62f81f1, c7a86d0, 5353799:
every signaled value carried a status resolved through its waits and its
stream predecessor, reported once per stream, then the stream recovered),
the per-stream diagnostics dump (`DebugState`, `DumpStreams`), and the
host-callback error state in `MetalStream`. A failed buffer still logs its
own waits (with their signaled values), kernels and memory state. Net
-257 lines of runtime and adapter code (-294 with the tests).

Cost: none saved. Waiting for a signaled buffer's final status costs the
same as waiting for its handler (Metal sets the status just before running
handlers). bench/latency.py, three builds interleaved over 10 rounds, p10 /
median / p90 of the per-round medians: jit(x*2+1) 171/174/182 us before,
170/174/186 us now; two-kernel 165/165/199 before, 164/165/204 now.
dispatch_bound.py unchanged within noise (3 rounds). Skipping that wait
(checking only statuses already final) gives 159/168/170 and 154/156/211
us, i.e. 6-9 us per synchronizing round trip, but is sound only if a
failed buffer never runs its own trailing signals (then waiters are woken
by its handler, after the error is recorded). Metal's messages say such
buffers are "aborted", but that was not verified (no real fault was
provoked), so the sound version is kept; the fast one is a one-line change.

Tests (no real GPU fault; `FailNextCommandBufferForTesting` and
`METAL_PJRT_FAIL_COMMAND_BUFFER=n` fail a buffer the way an aborted one
fails: its work runs, its signals do not, and its handler records the error
and force-signals them): runtime tests for stickiness across streams,
events and host tasks, host-callback errors, the hold rule; the executor
test for error_cb, events and BlockHostUntilDone; `tests/test_gpu_errors.py`
runs a JAX program with n = 0..8 and checks that no step returns wrong
values, the failure surfaces and every later step raises too.

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
not cached; the plugin does not change that threshold. (Since 2026-09-27 the
plugin sets no cache directory either; the README says how to turn the
cache on.)

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
f64, max over tests/test_scan.py shapes, on / off / CPU f32):
f32 softmax 30.6 / 30.6 / 30.3, log_softmax 1.04 / 1.03 / 1.04; f16
softmax 5.0 / 9.5 / 9.5; bf16 equal. Decision: **softmax rewriter deleted**
(roadmap 2.3), with its FFI handler and tests. The scan rewriter wins 1.4-3.2x
on every minor-dim cumsum/cumprod it matches, fwd and bwd, and is at least
as accurate (f16 cumsum 1.15 vs 2.08 ulps, bf16 0.78 vs 1.51, f32 equal):
**kept**.

## Benchmark hygiene (2026-09-27)

- `bench/run_all.sh` runs `BENCH_ROUNDS` (default 3) rounds with the
  backends interleaved inside each round (`BENCH_BACKENDS`, default `metal
  metal-gpu cpu mlx`; the jax-mps arm was dropped later, MLX is the one
  reference); `bench/report.py` takes the median over
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

Forcing one command buffer per call (`METAL_PJRT_EARLY_COMMIT_US=1000000`,
a knob since removed) or an early commit every time (`=0`) does not remove it (both modes appear
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
epilogue go to steel for f16/bf16 (as before); f32 ones keep MPS + the
second pass. (Since 2026-09-27 f16/bf16 GEMMs run only on steel:
`METAL_PJRT_GEMM` and MPS's bf16 staging are gone, and shapes past steel's
32-bit index limits are refused at compile time.)

f32 routing, removed (2026-09-27): 7704d02 also sent f32 GEMMs with an
epilogue to steel up to batch*m*n*k = 2^33. It was a third GEMM path and a
size threshold tuned on one M3 for 0-7% on mid shapes, with the nanoGPT
train step unchanged (180-182 ms either way), so it was dropped in the
simplification pass; f32 is MPS + second pass again (bitwise identical
results, see below). The f32 rows below are the measurements behind that.

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
more than the second pass saves, hence the (now removed) 2^33 threshold;
with it those shapes measured as before (4096x1024x4096 13.43 vs 13.43, 8192x4096x1024
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

## Memory policy: a caching platform allocator (2026-09-27, roadmap 3.3)

The default allocator is now XLA's pass-through `platform` allocator over
`rt::Device::Allocate`, which caches freed buffers by size (the
`JAX_OPENMETAL_ALLOCATOR=bfc` A/B arm below was removed afterwards: the
cache won on every axis). Policy:

- Lengths are rounded (powers of two up to a 16 KB page, whole pages above);
  a request reuses a cached buffer of at most min(2x, +2 pages) its length,
  as MLX does. Reuse is immediate, ordered by the compute stream as with BFC
  (XLA treats both as stream-ordered allocators); releasing a buffer that
  in-flight work still uses is safe because command buffers retain what they
  bind.
- Live + cached stays within the budget (half of RAM, capped by the GPU's
  recommended working set, times `JAX_OPENMETAL_MEMORY_FRACTION`, now applied
  by the runtime for both allocators): a miss evicts least recently freed
  buffers first, then fails with RESOURCE_EXHAUSTED. The 512 MB system guard
  still applies to every new buffer; when it refuses, the whole cache is
  dropped and the guard asked again. Buffers freed while work was in flight
  can only go once that work ends, so if some are still cached the
  allocation waits (at most 1 s, not after a device error) for the work
  outstanding at that moment, drops the cache and asks once more.
- XLA reports every refusal as "Out of memory while trying to allocate N
  with allocator ..."; the plugin's `PJRT_Error_Message` appends the
  runtime's reason (memory budget or system memory guard, with the numbers),
  so it reaches Python, not only stderr.
- A busy machine makes the guard refuse. Right after a Bazel build (server
  still up) the nanoGPT train step's first 1.17 GiB temp allocation was
  refused with 1.36 GB free or reclaimable (1.24 GB already live in the
  process, budget 4 GB); the budget never was the limit. Free memory on an
  8 GB Mac with a browser open swings by ~1 GB between runs (1.7-3.1 GB
  before a run here), so the same step can pass a minute later: batch G's
  round-1-only failure. The benchmarks run `bazel shutdown` first.
- Cached buffers unused for 2 s are released by a libdispatch timer (armed
  only while the cache is not empty), and all of them on a
  `DISPATCH_SOURCE_TYPE_MEMORYPRESSURE` warning, after which frees release
  directly until the level is normal again. `metal_pjrt_memory_stats` and
  `metal_pjrt_memory_pressure` (exported for ctypes, tests/test_memory.py)
  read the counters and run the pressure handler. A real notification:
  `sudo memory_pressure -S -l warn`.

Measured on the 8 GB M3 (idle machine, `bazel shutdown` first), each arm a
fresh process, interleaved over 3 rounds; time is the median of 10 steps,
footprint is `phys_footprint` (what Activity Monitor shows), "idle" 3.5 s
after the last array was deleted:

| workload | allocator | step ms | peak MB | after compute MB | idle MB |
|---|---|---|---|---|---|
| nanoGPT train step | bfc | 183.6-184.8 | 2058-2899 | 2058-2899 | 2004-2137 |
| | platform, no cache | 217-245 | 2231-2328 | 666-670 | 572-576 |
| | platform + cache | 182.6-186.8 | 1891-1908 | 1891-1908 | 523-530 |
| tinygp vg (quasisep-par 200k + dense 3000) | bfc | 198.3-214.6 | 967-1317 | 967-1317 | 918-1267 |
| | platform, no cache | 210.9-212.1 | 1197-1207 | 432-442 | 432-442 |
| | platform + cache | 199.7-201.5 | 878-886 | 878-886 | 392-407 |

BFC never gave anything back (patch 0004 only did when growth was refused),
so a finished nanoGPT run held ~2-2.9 GB, about a quarter to a third of this
machine's RAM. With the cache, the steady state is BFC's (1123 of 1295
nanoGPT allocations were cache hits) and the memory goes back within ~3 s.
The idle ~0.5 GB is the process itself (XLA, compiled programs, Python);
fresh processes start at ~150 MB. Dispatch-bound programs (latency.py,
dispatch_bound.py, 6 interleaved rounds, median of per-round medians in us,
bfc / cache): jit(x*2+1) 188 / 188, two-kernel 170 / 172, chain16 273 / 276,
chain64 414 / 414, chain256 1349 / 1364, scan 8480 / 8610 (bimodal in both).
The bench suite (5 interleaved rounds) is unchanged within its spread
(nanoGPT train step 180.3 / 178.3 ms, fwd 58.3 / 58.0; largest moves are
sub-ms bimodal rows: qr 128 0.70 / 1.00 with ranges 0.64-1.32 / 0.68-1.27).

Re-measured at the end of the batch (after 9b57467: cached buffers are
released only once every command buffer and host task that existed at the
free has finished; 44a1105: device_put copies its source): nanoGPT
178.7-180.0 ms (bfc 179.6-181.7), peak 1893-1897 MB (bfc 2508-2896), idle
515-519 MB (bfc 2454-2843); tinygp 188.6-192.6 ms (bfc 190.5-195.0), idle
391-399 MB (bfc 920-1223). Nothing stayed pinned by pending work: cached
was 0 MB at idle in every run. Under a 512 MB budget
(JAX_OPENMETAL_MEMORY_FRACTION) a loop of 64 MB outputs held 7, then got
RESOURCE_EXHAUSTED ("Out of memory while trying to allocate 64.00MiB";
the runtime's reason is logged), and the process kept working.

The plain platform allocator was also leaking: every MPS GEMM autoreleased
MPSMatrix objects (retaining their buffers) into a pool that XLA's threads
never drain (fixed in 9545c5b; nanoGPT reached 5.5 GB and then
RESOURCE_EXHAUSTED before). A loop over the other Metal paths (blit copy,
LAPACK, small-n GPU cholesky, steel, scan, sort, host callbacks, small H2D;
200 calls each with caching off) shows no growth.

## Host LAPACK as a stream host task: decided against (2026-09-27, roadmap 3.1)

The host LAPACK handlers (n > 32, `linalg/lapack_ffi.cc`) synchronize the
stream and run on the thunk thread. Tried: validate on the thunk thread, then
enqueue the LAPACK work with `rt::Stream::HostCallback` (pointers and sizes
captured by value, workspace in `std::vector` inside the task, info > 0 ->
NaN, so no data can fail the task and set the sticky error). It worked (the
tests below pass with it) but is not faster, so it was not kept.

Interleaved, 16 rounds, fresh process per arm and round, per-call wall time
in us pooled over rounds (p10 / median / p90); CPU is `JAX_PLATFORMS=cpu`:

| program | CPU | sync (kept) | host task |
|---|---|---|---|
| cholesky 128 | 26 / 28 / 29 | 270 / 313 / 680 | 246 / 299 / 681 |
| solve 128x16 | 50 / 56 / 58 | 314 / 348 / 362 | 314 / 354 / 384 |
| qr 128 | 150 / 151 / 156 | 527 / 544 / 594 | 523 / 571 / 614 |
| 100x cho_solve 64 (no block between) | 456 / 460 / 478 | 1663 / 2086 / 2213 | 1818 / 2345 / 2490 |
| pure_callback 1K (unchanged path) | | 179 / 196 / 218 | 178 / 196 / 231 |

6 more rounds: time until `jit(cholesky)(x)` returns (not blocking) 135 /
173 / 361 sync vs 169 / 179 / 370 host task; 20 unblocked steps of
{cholesky + l @ l.T, independent 1024^2 matmul} 24.6 / 24.8 / 25.3 ms vs
23.0 / 23.4 / 26.0.

Why it cannot approach "CPU + one dispatch":

- The programs put GPU work on both sides of the LAPACK call.
  `jnp.linalg.cholesky` is transpose fusion (symmetrize) -> `metal$cholesky`
  -> select fusion (mask), two command buffers (METAL_PJRT_TRACE). The
  LAPACK work needs the first buffer's result and the second needs LAPACK's,
  so every call pays two dependent GPU round trips (~130 us each on this M3)
  whether the thunk thread waits or a worker thread does.
- The thunk thread does not get free either: under the hold rule (never
  commit a buffer that waits on an unsignaled host task), the executable's
  final commit (its RecordEvent) host-waits for the task, so `f(x)` returns
  no earlier. Removing that wait needs the deferred held-buffer commit queue
  that the design postponed, and would still leave the two round trips.
- A chain of LAPACK-only programs (cho_solve: two triangular solves and no
  GPU kernel) never waited for the GPU before (synchronizing an idle stream
  is cheap). As host tasks it pays a worker wakeup and a CPU-signaled fence
  per call: +12% at the median and p90.

What would reach the target instead is not running the small fusions
around the call on the GPU (or folding them into the handler). That is a
separate item, not scheduled.

Python host callbacks stay synchronous too (coordinator decision, not
measured): with the hold rule a commit host-waits for unfinished host tasks
on the calling thread. If that thread held the GIL (jaxlib paths we cannot
audit; the py_client sources are not in the pinned checkout), a callback
task that needs the GIL would deadlock for good, in a process with GPU work
in flight that cannot be killed safely. Making them asynchronous needs the
same deferred held-buffer commit path, just for Python tasks.

Tests added (they exercise the synchronous path and passed with the host
task too): `test_host_lapack_in_order` (cholesky, triangular solve and LU at
n = 33, 100, 500, 2000 with GPU work before and after, vs float64
numpy/scipy) and `test_lapack_failures_are_values` (non-PD cholesky, a
singular solve, eigh/svd of a NaN matrix give NaN as on CPU, a singular LU
stays finite, a malformed `metal$cholesky` call raises INVALID_ARGUMENT, and
an unrelated jit still works after each). Both failed under mutations of
the host-task version (task run without waiting; info > 0 made an error,
which set the sticky error and killed the child).

## Sort via SortRewriter: MSL radix sort (2026-09-27, roadmap 2.2)

`ApplyMetalDefaults` now enables `xla_gpu_enable_cub_radix_sort`, and the
`xla.gpu.ext.cub_sort_{keys,pairs}` targets run an MSL LSD radix sort
(`ffi/cub_sort_ffi.cc`): 4-bit digits, 256 threads x 8 items per tile,
stable in-tile ranking by one scan over per-thread digit counts. Rows of at
most 2048 are sorted by one threadgroup each in threadgroup memory (one
dispatch); longer rows take hist / scan / scatter dispatches per digit pass
(24 for 32-bit keys), ping-ponging through scratch.

Interleaved in one process (the arm is `METAL_PJRT_DISABLE_REWRITES`, read
per compile), 8 rounds x 15 calls, per-call wall us pooled, p10 / median /
p90. `bitonic` = `METAL_PJRT_DISABLE_REWRITES=cubsort` (MetalSortExpander
only, the previous behaviour):

| program | radix | bitonic |
|---|---|---|
| jnp.sort f32 20k | 296 / 307 / 541 | 13839 / 14014 / 14707 |
| jnp.sort f32 100k | 424 / 429 / 448 | 20483 / 20688 / 21092 |
| jnp.sort f32 1M | 2491 / 2557 / 2677 | 107699 / 108463 / 109864 |
| jnp.sort i32 1M | 1630 / 1644 / 1710 | 107981 / 108687 / 111015 |
| argsort f32 20k | 297 / 311 / 316 | 13573 / 14004 / 14167 |
| argsort f32 1M | 3757 / 3843 / 4085 | 107834 / 108335 / 110687 |
| sort rows f32 16x65536 | 2302 / 2400 / 2509 | 69070 / 69632 / 70425 |
| sort rows f32 64x4096 | 648 / 665 / 672 | 12866 / 13343 / 13541 |
| sort rows f32 200x1000 | 376 / 386 / 394 | 8571 / 8686 / 8864 |
| argsort rows f32 1000x128 | 916 / 964 / 1011 | 3897 / 3960 / 4123 |
| argsort rows f32 300x100 | 425 / 434 / 455 | 3489 / 3567 / 3724 |
| sort rows f32 1000x65 | 891 / 946 / 995 | 3943 / 4092 / 4252 |
| sort rows f32 1000x32 (bitonic in both) | 262 / 266 / 270 | 301 / 310 / 314 |
| sort rows f32 5000x8 (bitonic in both) | 229 / 239 / 242 | 240 / 250 / 260 |
| tinygp key/value i32 12500x4 (bitonic in both) | 203 / 211 / 216 | 199 / 210 / 214 |
| tinygp key/value i32 100000x4 (bitonic in both) | 336 / 341 / 353 | 340 / 348 / 365 |

16M elements: jnp.sort 41 ms, argsort 60 ms (bit-identical to numpy).

SortRewriter's non-CUDA rule takes every simple sort of more than 16384
elements however short its rows, and before the small-row fix the radix sort
lost badly there (one 256-thread threadgroup per row): 1000x32 942 us vs
307, 5000x8 3560 vs 250, tinygp's 12500x4 8386 vs 213 and 100000x4 64959
vs 353. So `RunHloPasses` first expands sorts with rows of <= 64 (the
bitonic network's straight-line case) and more than 16384 elements with
`MetalSortExpander`; SortRewriter never sees them. Smaller sorts are left to
the later expansion as before. tinygp (`bench/tinygp_bench.py`, 3
interleaved rounds per arm) and the nanoGPT train step (no sorts) are
unchanged.
