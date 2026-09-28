# Design: Metal as a fourth XLA:GPU platform

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
Details and the exact contract are in `docs/integration-notes.md`.

Runtime shader compilation works with command-line tools only (verified:
192 ms for a trivial kernel). No cache of our own is needed across processes:
Metal's system shader cache keys on the source, so the same MSL compiles in
~1.3 ms in a later process (186 ms first; measured 2026-09-26). Within a
process every kernel (emitted, FFI, steel, MPS staging, built-ins) comes
from one cache in `rt::Device` (`GetKernel`), keyed by the MSL source, the
function name and any function constants. XLA compilation
itself is cached across processes by JAX's persistent compilation cache
when the user turns it on (`docs/integration-notes.md`, PJRT client).

## Library ops

`GpuCompiler` rewrites dots into library custom calls (on OneAPI every one
is `__cublas$lt$matmul`), and the StreamExecutor BLAS interface is the plug
point. f32 GEMMs run on Metal Performance Shaders' matrix kernels (a bias
or activation epilogue is a second MSL pass); f16/bf16 GEMMs run on "steel"
kernels ported from MLX (MIT), with the epilogue applied in their store.
There is no DNN library path: convolutions run as XLA-emitted loop kernels.
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
one. It commits when the GPU has nothing of the stream's left to run and at
least 16 ops are encoded, if 500 us have passed since its last commit;
explicit syncs commit at once. Caps keep each buffer far from the GPU
watchdog: 1024 ops, 2^27 dispatched threads, or 2e11 GEMM flops. GEMMs
(steel launches and MPS through `EncodeExternal`) launch few threads for
their work, so they are charged an estimate instead, `blas::GemmWork`:
2mnk per batch, or 64 flops per byte of A, B and C for bandwidth-bound
(GEMV-like) shapes. The budget is <= 0.1 s of GEMM on the M3 (2.0-3.1
TFLOPS measured over f32/f16/bf16 shapes), a >= 5x margin for a throttled
base M1 and for shapes that run less efficiently; a single larger GEMM
gets a buffer of its own. Eight chained 4096^3 f32 GEMMs used to share one
371 ms buffer; now each buffer holds two (93 ms). Two gaps remain: the
budget splits only between dispatches, so one kernel longer than the
watchdog still resets the GPU (one 16384^3 f32 GEMM is ~8.8e12 flops, 3-4 s
in one dispatch here); and non-GEMM heavy kernels (emitted fusions,
especially convolutions through XLA's loop emitter) are charged only their
thread count, so a large convolution can be a multi-second single kernel. Copies and uniform fills up to 16 MB run as built-in
compute kernels so kernels and copies share one compute encoder (an encoder
switch costs ~10 us of GPU time); larger ones use the blit engine.

**Lazy waits.** Waits (`WaitForEvent`, `WaitForStream`, a host task) are
recorded and encoded only just before the stream's next GPU work, and a
`Synchronize` with pending waits satisfies them on the host, so no command
buffer ever only waits (a waiting buffer counts against the watchdog).
`WaitForStream(other)` waits for the highest value `other` has issued,
including host tasks, and inherits the waits `other` has not encoded yet.
`RecordEvent` publishes its value only once the signaling buffer is
committed.

**Hold rule.** A command buffer that waits on a host task that has not run
yet is not committed: `Stream::Commit` waits on the host for the task
first, so a slow host task never sits on the GPU timeline. Host-task
workers never take the stream lock, so this cannot deadlock with them; a
task (or its error callback) that calls into its own stream gets
FAILED_PRECONDITION before taking the lock.
`Device::unsignaled_host_task_waits_committed()` counts violations and the
runtime tests assert it stays 0.

**Sticky GPU errors.** The first failed command buffer (watchdog timeout,
page fault, out of memory, anything) or failed host task without an error
callback sets the device's error, as a CUDA context error does. From then on
every wait, poll, host task and launch returns it, a host callback's
`error_cb` gets it (which is how XLA fails result buffers), a host callback
without one still runs (XLA's free memory or finish transfers), and nothing
more is committed. The message says to restart the process. Nothing hangs: a
failed buffer's completion handler records the error and force-signals the
buffer's fence and events, host waits give up once the device has failed,
and no error path CHECKs or aborts. Only resets (timeout, access revoked,
device removed) feed the reset log.

**Fence before events.** Soundness (never consume a failed buffer's output)
does not wait for completion handlers, which Metal does not order across
queues. Every buffer signals its stream's fence before its events, so after
a host waiter sees a signal it checks the buffers still in flight
(`Device::CheckInFlight`): one whose fence value is signaled has run to its
end, so its final status is waited for and an error recorded. That covers
the buffer that signaled and every buffer on another queue it depended on
(runtime test `EventSignalsFollowTheFenceSignal`). The wait costs nothing
measurable; skipping it would save 6-9 us per synchronizing round trip but
is sound only if an aborted buffer never runs its trailing signals, which
is unverified (`docs/roadmap.md`, Next).

**Memory.** Allocations are shared-storage `MTLBuffer`s from a size-class
cache behind XLA's pass-through `platform` allocator; a fresh buffer costs
~60 us per MB in first-touch page faults, and training steps repeat their
sizes.

- Lengths are rounded (powers of two from 256 bytes up to a 16 KB page,
  whole pages above); a request reuses a cached buffer of at most min(2x,
  +2 pages) its length. Reuse is immediate, ordered by the compute stream
  as XLA assumes for its stream-ordered allocators; command buffers retain
  what they bind.
- Budget: live + cached stays within half of physical RAM, capped by the
  GPU's recommended working set, times `JAX_MTL_MEMORY_FRACTION`. A
  miss evicts least recently freed buffers first, then fails with
  RESOURCE_EXHAUSTED. The compiler sees the recommended working set as the
  device size, so compiled programs do not vary with load.
- System guard: an allocation of 1 MB or more is refused if it would leave
  less than 512 MB free, inactive, speculative or purgeable
  (`METAL_PJRT_SYSTEM_MEMORY_RESERVE_MB`), because a GPU touching swapped-out
  pages stalls until the watchdog resets it. On a refusal the cache is
  dropped and the guard asked again; if cached buffers are still held by
  work in flight, the allocation waits (at most 1 s, not after a device
  error) for that work, drops the cache and asks once more. MPS's internal
  staging copies bypass the cache but pass the guard.
- Release tickets: every command buffer holds a work ticket from creation
  to completion, every host task from enqueue to its end. A cached buffer
  is released (evicted, trimmed or dropped) only once all work that existed
  when it was freed has ended, since host tasks hold raw pointers and
  argument buffers hold GPU addresses.
- Idle trim and pressure: a libdispatch timer, armed only while the cache
  is not empty, releases buffers unused for 2 s, so an idle process gives
  its memory back; a `DISPATCH_SOURCE_TYPE_MEMORYPRESSURE` warning releases
  them all, and frees release directly until the level is normal.
- XLA reports every refusal as "Out of memory while trying to allocate N";
  the plugin's `PJRT_Error_Message` appends the runtime's reason (budget or
  system guard, with the numbers) so it reaches Python.
- Only the `pinned_host` memory kind uses XLA's host BFC pool (never
  shrinks).

**Transfers.** Host-to-device and device-to-host copies are a `memcpy` on
the calling thread when the stream is idle, and a host task otherwise; there
is no staging. JAX lets XLA read the source after `device_put` returns, and
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
kernels in the buffer that timed out (none if it only waited on another
stream; built-in fill/copy kernels are listed apart and never blamed) and
the plugin build (diagnostics only). A kernel seen in two resets since boot
(`METAL_PJRT_QUARANTINE_STRIKES`, 0 disables) is refused by
`Device::GetKernel` (at load time, and on later cache hits) with
FAILED_PRECONDITION, so a compiler bug costs at most two resets, not one per
run; a reboot or `scripts/gpu_health.py --clear` lifts it. Strikes are not
keyed by build (it changes on every rebuild); a changed kernel source gets
a new key anyway. The first reset of a non-terminating kernel is
unavoidable: Metal has no per-kernel timeout.

## How this differs from MLX

`docs/archive/mlx-comparison.md` has the source-level read (MLX 59d600b,
jax-mps 7fd54c6). In short: MLX is an eager interpreter over a lazy graph;
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
done; `docs/archive/` has the record. Standing risks:

- Building XLA on an 8 GB laptop: low concurrency, a shared disk cache and
  JAX's public remote cache keep rebuilds to minutes; a cold build takes
  ~2 hours.
- CUDA coupling in `xla/service/gpu` and XLA's OneAPI branches: three
  patches against the pinned XLA (`third_party/xla/patches`) and the
  tripwire test on every pin bump.
- MSL codegen gaps: 32-bit atomics only, no f64, 32 KB threadgroup memory,
  32-wide SIMD, launch-dimension mapping.
