# Design: Metal as a fourth XLA:GPU platform

For contributors changing the compiler or the runtime: how the plugin is
put together and the policies the runtime follows. Users want the README.

## The production GPU plugin, and what changes for Metal

The JAX CUDA backend is four layers, and only the bottom two are CUDA-specific:

1. **C API shim** (`xla/pjrt/c/pjrt_c_api_gpu_internal.cc`): parses client
   options, picks the platform name from a compile-time macro (CUDA, ROCm or
   SYCL), and chains the PJRT extensions (FFI, custom call, profiler, stream,
   layouts, memory descriptions, cross-host transfers, shardings, ABI version).
2. **Client** (`xla/pjrt/gpu/se_gpu_pjrt_client.cc`): per-device
   `LocalDeviceState`, allocators, host memory spaces, async dispatch. Only
   NCCL collectives and fabric handles are CUDA-specific.
3. **StreamExecutor platform** (`xla/stream_executor/<platform>/`): allocate,
   memcpy, streams, events, kernel launch, device description, plus the optional
   BLAS/DNN interfaces that cuBLAS and cuDNN plug into.
4. **Compiler subclass** of `GpuCompiler`: overrides post-layout-assignment
   passes, autotuning passes, target triple/data layout, and the final
   LLVM-IR-to-binary step. The base class owns the entire HLO pipeline.

Metal support is therefore:

- `stream_executor/metal/`: `MetalPlatform`, `MetalExecutor`, streams as
  `MTLCommandQueue`, events as `MTLSharedEvent`, shared-storage `MTLBuffer`s,
  kernel launch from an `MTLLibrary`, over an XLA-free runtime layer
  (`metal_pjrt/runtime/`, "Runtime" below). XLA's `CommandBuffer`
  (its CUDA-graph abstraction) is deliberately not implemented: a
  software-replay version was built, measured to be a wash once the
  runtime's per-launch overhead was fixed, and removed to keep the platform
  small (`docs/performance.md`, "Measured and rejected").
- `MetalCompiler : GpuCompiler` with Triton, cuDNN passes, autotuning and
  collectives disabled.
- XLA's GPU C API shim built for Metal (patch 0001 names the platform),
  behind the plugin's own small `GetPjrtApi` (`pjrt/metal_pjrt_api.cc`),
  which drops the ABI-version extension so JAX's persistent cache works and
  is done with `device_put`'s host data before returning.
- The Python package `metal_pjrt_plugin`, modeled on JAX's
  `jax_plugins/cuda`: a top-level package found through its `jax_plugins`
  entry point, like `jax_cuda12_plugin`.

Intel's extension for OpenXLA did exactly this for SYCL out of tree. The
device reports a OneAPI compute capability, so XLA takes its generic
non-NVIDIA branches; `metal_pjrt/xla_tripwire` fails when a pin bump
changes any of them (`docs/integration-notes.md`).

## Codegen

`GpuCompiler` bottoms out in LLVM IR and there is no LLVM Metal target. The
in-tree Intel compiler shows the shape of the plumbing (LLVM IR -> SPIR-V via
`spirv_backend.cc`), but two facts rule out "translate the final IR":

- SPIRV-Cross only consumes shader-flavored SPIR-V; the LLVM backend emits
  OpenCL-flavored kernel SPIR-V for pointer-based kernels.
- MSL has no `goto` and no labeled statements (verified on this machine), so
  turning a CFG back into C requires reconstructing structured control flow.

XLA's MLIR emitters keep `scf.for`/`scf.if` structure until the
second-to-last lowering pass, and MLIR ships an EmitC dialect with a C++
printer. So the kernel path is: `MetalCompiler` supplies its own
`KernelCompiler` (`MetalKernelCompiler`, via a virtual
`GpuCompiler::CreateKernelCompiler` factory that is our one small codegen
patch to XLA), whose `CompileMlirToLlvm` runs the
lowering pipeline up to (not including) SCFToControlFlow, converts what
remains (func/scf/arith/math/vector/gpu plus the LLVM-dialect memory ops that
LowerTensors introduced) to EmitC, and prints MSL. The text rides to
`CompileTargetBinary` inside a stub LLVM module and becomes the kernel
"binary"; the executor compiles it with `newLibraryWithSource` at load time.
The binary's first line, `#include <metal_pjrt/msl_prelude.metal>`, stands
for the ~9 KB prelude of helpers, which the runtime puts back before
compiling: XLA keeps two copies of every kernel thunk's binary for the
executable's life (the thunk's and its serialized form, one per thunk even
where thunks share a kernel), so a full prelude in each was ~40 MB of a
Qwen3-0.6B train step's ~270 MB (`docs/performance.md`, host memory).
MSL dumps (`--xla_dump_to`) have the prelude expanded, so a dumped `.metal`
file compiles on its own with `xcrun metal`.
Details and the exact contract are in `docs/integration-notes.md`.

Runtime shader compilation works with command-line tools only (verified:
192 ms for a trivial kernel). No cache of our own is needed across processes:
Metal's system shader cache keys on the source, so the same MSL compiles in
~1.3 ms in a later process (186 ms the first time). Within a
process every kernel (emitted, FFI, steel, MPS staging, built-ins) comes
from one cache in `rt::Device` (`GetKernel`), keyed by the MSL source, the
function name and any function constants. Library kernels (FFI, steel, MPS
staging, built-ins) stay for the process; kernels the compiler emitted are
counted (`AcquireKernel`, taken by the StreamExecutor's `LoadKernel` and
dropped with the thunk that owns it), so a dropped executable takes its
pipelines, libraries and cached MSL text with it (~40 KB per kernel). XLA compilation
itself is cached across processes by JAX's persistent compilation cache
when the user turns it on (`docs/integration-notes.md`, PJRT client).

## Library ops

`GpuCompiler` rewrites dots into library custom calls (on OneAPI every one
is `__cublas$lt$matmul`), and the StreamExecutor BLAS interface is the plug
point. f32 GEMMs run on Metal Performance Shaders' matrix kernels (a bias
or activation epilogue is a second MSL pass); f16/bf16 GEMMs run on "steel"
kernels ported from MLX (MIT), with the epilogue applied in their store, and
2..8 rows of x W^T with K >= 512 (small-batch LLM decode) on MLX's wide
gemv, which streams the weight once per <= 5 rows.
There is no DNN library: convolutions of 4 Mflop and more (1-D and 2-D,
f32/f16/bf16, ungrouped) go to `metal$conv`, MLX's steel convolution kernels
(`metal_pjrt/conv`), through the plugin's own rewriter in
`OptimizeHloConvolutionCanonicalization` (CUDA's cuDNN canonicalization
hook); the rest run as XLA-emitted loop kernels.
Other custom calls follow CUDA's shape, where CUDA also has a library call:
XLA's SortRewriter targets an MSL radix sort, dense linear algebra goes to
Accelerate LAPACK on the shared buffers (GPU kernels up to 32x32), Python
callbacks to an FFI handler (`docs/callbacks.md`). The one Metal-only
rewrite is `metal$scan` for minor-dimension cumulative ops (1.4-3.2x).
MPSGraph is not used: it is what Apple's jax-metal compiles whole programs
with, and it is opaque and has a record of bugs.

## Deliberately off

Triton, cuDNN fusion passes, autotuning, collectives (a stub for one
device), float64 (Metal has none: f64 arithmetic is refused at compile
time; double-float emulation is an idea in `docs/roadmap.md`).

## Runtime

`metal_pjrt/runtime/metal_runtime.h` is the XLA-free layer every
Metal call goes through (`rt::Device`, `rt::Stream`, `rt::Event`); its
header comments are the reference. The policies, in the present tense:

**Command-buffer batching.** A command buffer costs hundreds of
microseconds of CPU and GPU-scheduler time (the driver spends ~150 us of a
dedicated thread submitting each), so a stream packs many dispatches into
one. It commits when the GPU has nothing of ours left to run and at
least 16 ops are encoded, if 500 us have passed since the last commit;
explicit syncs commit at once. Caps keep each buffer far from the GPU
watchdog: 1024 ops, 2^27 dispatched threads, or 2e11 GEMM flops. GEMMs
(steel launches and MPS through `EncodeExternal`) launch few threads for
their work, so they are charged an estimate instead, `blas::GemmWork`:
2mnk per batch, or 64 flops per byte of A, B and C for bandwidth-bound
(GEMV-like) shapes. The budget is <= 0.1 s of GEMM on the M3 (2.0-3.1
TFLOPS measured over f32/f16/bf16 shapes), a >= 5x margin for a throttled
base M1 and for shapes that run less efficiently; an op that would take the
open buffer over it starts a new one, and a single larger GEMM gets a
buffer of its own. A buffer that ran longer than 1 s on the GPU is logged
(rate-limited) with its kernels: it came close to the watchdog. Eight chained 4096^3 f32 GEMMs used to share one
371 ms buffer; now each buffer holds two (93 ms). Two gaps remain: the
budget splits only between dispatches, so one kernel longer than the
watchdog still resets the GPU (one 16384^3 f32 GEMM is ~8.8e12 flops, 3-4 s
in one dispatch here); and non-GEMM heavy kernels (emitted fusions,
especially convolutions left to XLA's loop emitter) are charged only their
thread count, so a large convolution there can be a multi-second single
kernel (`metal$conv` charges its flops and splits its launches at half the
budget). Copies and uniform fills up to 16 MB run as built-in
compute kernels so kernels and copies share one compute encoder (an encoder
switch costs ~10 us of GPU time); larger ones use the blit engine.

**Queue (since 2026-09-30).** Each device has one Metal command queue;
every stream encodes into its open command buffer, and buffers are
committed in encode order. Every committed buffer ends by signaling the
device timeline (a shared event) with its sequence number. A command
buffer never waits on the GPU for anything (a waiting buffer counts
against the watchdog, and a wait has no timeout): GPU work that depends on
other GPU work is behind it in the queue, so waits on GPU work cost
nothing, and GPU work that depends on a host task is encoded only after
the task has run. The encoding thread waits for that on the host without
holding the queue lock, so other streams keep encoding.
`Device::gpu_waits_encoded()` is 0 by construction (nothing encodes a
wait); tests check it through `metal_pjrt_sync_stats`.

A stream is a handle: the timeline value of its last GPU op, the waits its
next op must respect (`WaitForEvent`, `WaitForStream`, its own host tasks)
and a worker thread for its host tasks. An event is a recorded work point:
a timeline value plus host-task values. Host work (device-to-host copies,
host-to-device copies behind pending work, XLA callbacks) runs on the
stream's worker once everything the stream enqueued or waits for is done;
the open buffer is committed when such a task is enqueued (and when an
event is recorded), so no host wait depends on the next commit. Host
transfers run inline when the stream has nothing pending. A host task (or
its error callback) that calls into its own stream, or waits for itself
or later work of its stream, gets FAILED_PRECONDITION.

**Sticky GPU errors.** The first failed command buffer (watchdog timeout,
page fault, out of memory, anything) or failed host task without an error
callback sets the device's error, as a CUDA context error does. From then on
every wait, poll, host task and launch returns it, a host callback's
`error_cb` gets it (which is how XLA fails result buffers), a host callback
without one still runs (XLA's free memory or finish transfers), and nothing
more is committed. The message says to restart the process. Nothing hangs: a
failed buffer's completion handler records the error and force-signals its
timeline value, waits on the GPU timeline give up once the device has
failed, and no error path CHECKs or aborts. A transfer task skips its copy
after a failure (XLA may already be freeing its buffers), and waiters on a
host task wait for it to end even then. Only resets (timeout, access revoked,
device removed) feed the reset log.

**Failures seen by host waiters.** Soundness (never consume a failed
buffer's output) does not wait for completion handlers: after a host
waiter sees the timeline signal it checks the buffers still in flight
(`Device::CheckInFlight`); one whose timeline value is signaled has run to
its end, so its final status is waited for and an error recorded. The
wait costs nothing measurable; skipping it would save 6-9 us per
synchronizing round trip but is sound only if an aborted buffer never runs
its trailing signal, which is unverified (`docs/roadmap.md`, Next).

**Memory.** Allocations are shared-storage `MTLBuffer`s from a size-class
cache behind XLA's pass-through `platform` allocator; a fresh buffer costs
~60 us per MB in first-touch page faults, and training steps repeat their
sizes.

- Lengths are rounded (powers of two from 256 bytes up to a 16 KB page,
  whole pages above); a request reuses a cached buffer of at most min(2x,
  +2 pages) its length. Reuse is immediate, ordered by the compute stream
  as XLA assumes for its stream-ordered allocators; command buffers retain
  what they bind. Buffers the host fills right away, off any stream
  (module constants, FFT tables), take only a cached buffer whose work has
  all ended (its release ticket, see `Device::BeginWork`), else a fresh
  one: a reused buffer that queued kernels still read would see the host's
  bytes.
- Budget: live + cached stays within half of physical RAM, capped by the
  GPU's recommended working set, times `METAL_PJRT_MEMORY_FRACTION`. A
  miss evicts least recently freed buffers first, then fails with
  RESOURCE_EXHAUSTED. The compiler sees the recommended working set as the
  device size, so compiled programs do not vary with load.
- System memory: the budget is the limit, as PyTorch MPS caps its own
  allocations and otherwise lets macOS page. The only system-level refusal
  is at critical memory pressure (`kern.memorystatus_vm_pressure_level`,
  read per allocation of 1 MB or more; critical is where jetsam starts
  killing processes): the cache is dropped and the level read again; if
  cached buffers are still held by work in flight, the allocation waits
  (at most 1 s, not after a device error) for that work, drops the cache
  and reads once more, then fails with RESOURCE_EXHAUSTED. MPS's internal
  staging copies bypass the cache but get the same check. The first
  allocation at warning or worse logs a warning. A stricter guard (refuse
  below 512 MB of free pages) existed because swapping wedged the GPU on
  2026-09-25 (two processes each allowed 70% of the working set, ~100 MB
  free, workers killed mid-GPU-work); low free pages are macOS's normal
  state, and it refused 25 of 28 airbench94 runs at 50-70% free. That
  wedge is now covered by the half-of-RAM budget, bounded waits instead
  of kills, and the GPU reset log.
- Release tickets: every command buffer holds a work ticket from creation
  to completion, every host task from enqueue to its end. A cached buffer
  is released (evicted, trimmed or dropped) only once all work that existed
  when it was freed has ended, since host tasks hold raw pointers and
  argument buffers hold GPU addresses.
- Host transfers check their device side: a copy to or from a pointer that
  is not inside a live allocation fails with INTERNAL and a log line naming
  the nearest allocations. XLA keeps a transfer's buffer alive until the
  copy is done, so this means the allocation table lost a buffer XLA still
  holds (a use after free); writing through the pointer would corrupt
  whatever reused the memory.
- Frees check their pointer: freeing anything but the start of a live
  allocation (unknown, interior, already freed: the last 1024 frees are
  remembered to name a double free) or with a size other than the one
  allocated is INTERNAL, logged with a backtrace, and nothing is cached.
  (XLA's own allocator path frees without a size; the size is checked
  where one is passed: module constants, FFT tables, XLA's memory
  allocators.) A new buffer at an address that is still a live key is
  logged. `METAL_PJRT_DEBUG_FREE_QUARANTINE=1` (debugging) keeps freed
  buffers out of reuse until 64 later frees and 100 ms have passed, and
  reports a pointer into one with the backtrace of its free.
- Stale transfer pointers: a watched risk, not closed (2026-09-30).
  - Seen only with an experiment that staged busy host-to-device copies as
    GPU copies: Resolve failed for copy destinations (2 of 7 runs of
    `test_source_mutated_right_after_device_put[False-idle-1]`), i.e. XLA
    apparently copying into memory that was no longer allocated.
  - A probe with that experiment's check alone (resolve the destination
    of every host-to-device copy enqueued on a busy stream, log, continue
    on the normal path) found 0 failures in 600 runs of that test on two
    later trees (before and after XLA's host memory moved out of the
    device allocator; busy enqueues in every run), and 0 in 50 runs of all
    of test_transfers.
  - The mechanism is not identified. A control run on the experiment's
    parent tree was not done (a multi-hour rebuild); it is left for the
    owner.
  - The tripwires (the unresolved-pointer INTERNAL; frees checked for
    start, size where known and double frees, with generations; the debug
    free quarantine) catch a stale pointer only while its memory is not
    live: a stale write into a buffer that has been reused stays silent.
  - The unresolved-pointer check has been live on every host transfer
    through every full gate since it landed (2026-09-30), with no
    INTERNAL.
  - One wrong-value failure of
    `test_source_mutated_right_after_device_put[False-busy-1]` on
    2026-09-28 predates the experiment and is not explained by it; it has
    not recurred (~15.6k puts since).
- Idle trim and pressure: a libdispatch timer, armed only while the cache
  is not empty, releases buffers unused for 2 s, so an idle process gives
  its memory back; a `DISPATCH_SOURCE_TYPE_MEMORYPRESSURE` warning releases
  them all, and frees release directly until the level is normal.
- XLA reports every refusal as "Out of memory while trying to allocate N";
  the plugin's `PJRT_Error_Message` appends the runtime's reason (budget or
  critical system memory pressure, with the numbers) so it reaches Python, with the executable
  whose allocation was refused (its wrapped `PJRT_LoadedExecutable_Execute`
  names refusals made during the call; XLA allocates an execution's
  buffers on the calling thread). XLA's own `executable_name` payload names
  the last computation the error reached instead (each consumer of a failed
  output overwrites it).
- XLA's host memory space (its host BFC pools, which never shrink:
  linearizing host arrays that are not dense row-major, and `pinned_host`
  arrays) is plain anonymous host memory wrapped without copying in an
  `MTL::Buffer` (`Device::AllocateHost`), so device copies reach it. It is
  outside the device budget, the buffer cache and the device allocation
  table. XLA writes through the pool's pointer unchecked
  (`AllocateLinearizeDest`), so the only failure left is a failed `mmap`;
  beyond maxBufferLength the pages are host-only. The staging pool keeps
  about the largest non-dense put it has served and reuses it (measured
  2026-09-30: after a 128 MB transposed `device_put` and its deletion the
  process kept ~132 MB of host pool, and a second such put did not grow
  it).

**Transfers.** Host-to-device and device-to-host copies are a `memcpy` on
the calling thread when the stream is idle, and a host task otherwise; there
is no staging. (Staging a busy stream's copy as a GPU blit was tried and
reverted: its command buffer waited on the GPU for XLA's allocation event,
i.e. the whole compute backlog, which counts against the watchdog and reset
the GPU. A GPU-side wait has no timeout.) JAX lets XLA read the source after `device_put` returns, and
a data loader that refilled its array changed what the device got. So
`device_put` of dense host data snapshots it into a malloc'd buffer before
returning (as CUDA does for pageable memory), which does not block. From
min(256 MB, max(16 MB, reclaimable memory / 8)) up it hands XLA the
caller's data instead and waits until XLA is done with it: no second host copy (512 MB:
peak footprint +512 MB instead of +1041 MB), but the caller waits behind
already queued GPU work (the copy waits for XLA's allocation event on the
compute stream). Strided and sub-byte inputs take XLA's synchronous path,
which stages a copy.

**Reset log and quarantine.** A watchdog reset hits every process, and after
several the driver leaves the GPU ~10x slower per dispatch until a reboot.
The runtime appends each reset it observes to
`~/.cache/metal-pjrt/gpu_resets.jsonl` (`METAL_PJRT_STATE_DIR`) with the
kernels in the buffer that timed out (built-in fill/copy kernels are
listed apart and never blamed) and
the plugin build (diagnostics only), once per incident (a reset fails every
buffer in flight; the first one records it). The quarantine is opt-in
(off by default since 2026-09-30): with `METAL_PJRT_QUARANTINE_STRIKES=n`,
a kernel seen in n resets since boot is refused by
`Device::GetKernel`/`AcquireKernel` (at load time, and on later cache hits) with
FAILED_PRECONDITION, so a compiler bug costs at most n resets, not one per
run; a reboot or deleting the reset log lifts it (`scripts/gpu_health.py
--clear` does that in a source checkout; the wheel does not ship it). Strikes are not
keyed by build (it changes on every rebuild); a changed kernel source gets
a new key anyway. The first reset of a non-terminating kernel is
unavoidable: Metal has no per-kernel timeout.

## How this differs from MLX

From a source-level read of MLX and jax-mps (September 2026): MLX is an
eager interpreter over a lazy graph;
`mx.compile` fuses only elementwise chains, every call re-walks the graph
and encodes primitive by primitive, and memory comes from a caching
allocator with no planning. Its kernels (steel GEMM, Winograd conv, fused
inference attention, quantized matmul) are hand-tuned. An XLA backend wins
where fusion and a static executable matter (reductions fused with
producers, optimizer updates, layer-norm backward, static buffer
assignment) and trails where MLX has tuned library kernels (convolutions,
attention inference, quantized matmul) and on compile latency.

## History and risks

The original milestones (build XLA's GPU compiler on macOS without CUDA; a
StreamExecutor platform; MLIR -> EmitC -> MSL codegen; GEMM via MPS and
steel; a Python package run against JAX's tests and benchmarks) are all
done. Standing risks:

- Building XLA on an 8 GB laptop: low concurrency, a shared disk cache and
  JAX's public remote cache keep rebuilds to minutes; a cold build takes
  ~2 hours.
- CUDA coupling in `xla/service/gpu` and XLA's OneAPI branches: three
  patches against the pinned XLA (`third_party/xla/patches`) and the
  tripwire test on every pin bump.
- MSL codegen gaps: 32-bit atomics only, no f64, 32 KB threadgroup memory,
  32-wide SIMD, launch-dimension mapping.
