# Design: Metal as a fourth XLA:GPU platform

For contributors changing the compiler or the runtime. Users should start
with the README.

## The idea

JAX's CUDA backend has four layers, and only the bottom two are
CUDA-specific: the PJRT C API shim, the client
(`se_gpu_pjrt_client.cc`), the StreamExecutor platform (allocation,
streams, events, kernel launch, BLAS/DNN plug points) and a
`GpuCompiler` subclass, whose base class owns the whole HLO pipeline. So
Metal support is:

- `stream_executor/metal/`: `MetalPlatform` and `MetalExecutor`, over an
  XLA-free runtime ([Runtime](#runtime)). Streams map to command queues,
  events to `MTLSharedEvent`, memory to shared-storage `MTLBuffer`s.
  XLA's `CommandBuffer` (CUDA graphs) is deliberately not implemented: a
  replay version measured as a wash and was removed
  ([`performance.md`](performance.md), "Measured and rejected").
- `MetalCompiler : GpuCompiler`, with Triton, cuDNN passes, autotuning and
  collectives off.
- XLA's GPU C API shim built for Metal, behind the plugin's own small
  `GetPjrtApi` (`pjrt/metal_pjrt_api.cc`), which drops the ABI-version
  extension so JAX's persistent cache works and finishes with
  `device_put`'s host data before returning.
- The Python package `metal_pjrt_plugin`, modeled on `jax_plugins/cuda`.

Like Intel's out-of-tree SYCL backend, the device reports a OneAPI
compute capability, so XLA takes its generic non-NVIDIA branches.
`metal_pjrt/xla_tripwire` fails when a pin bump changes any of them
([`integration-notes.md`](integration-notes.md)).

## What loading the plugin changes in JAX

The plugin uses private `jax._src` APIs, so it's built for exactly
`jax==0.11.2`/`jaxlib==0.11.2` and warns at discovery on anything else.

With `JAX_PLATFORMS` unset, JAX initializes every installed backend, this
one included; if the Metal device can't be set up, it fails quietly and
CPU programs carry on. `JAX_PLATFORMS=cpu` creates no Metal device, but
JAX still imports the plugin. So in every JAX process in the environment
the plugin library is loaded and changes JAX's private state:

- wraps three functions (the compilation-cache check and host-callback
  lowering), passing other backends straight through;
- registers lowering rules for "mtl" only;
- adds "mtl" to the platforms that support buffer donation;
- marks the "mtl" backend as allowed to fail quietly.

## Codegen

There's no LLVM Metal target, and translating final LLVM IR is out:
SPIRV-Cross only reads shader-flavored SPIR-V, and MSL has no `goto`, so
a CFG would need its structure rebuilt.

XLA's MLIR emitters keep `scf.for`/`scf.if` structure until the
second-to-last lowering pass, and MLIR has an EmitC dialect with a C++
printer. So `MetalCompiler` supplies its own `MetalKernelCompiler`
(through a virtual `GpuCompiler::CreateKernelCompiler`, our one codegen
patch to XLA). It runs the lowering pipeline up to SCFToControlFlow,
converts the rest to EmitC, and prints MSL. The text rides to
`CompileTargetBinary` in a stub LLVM module and becomes the kernel
"binary", which the executor compiles with `newLibraryWithSource` at load
time.

The binary starts with `#include <metal_pjrt/msl_prelude.metal>` in place
of a ~9 KB helper prelude, which the runtime pastes back before
compiling. XLA keeps two copies of every kernel's binary, so inlining the
prelude cost ~40 MB of host memory in a Qwen3-0.6B train step. MSL dumps
(`--xla_dump_to`) have it expanded, so a dumped `.metal` file compiles on
its own with `xcrun metal`.

Shaders compile at run time with only the command-line tools installed.
Metal's system shader cache keys on the source, so a kernel compiled in
one process compiles in ~1 ms in the next; we keep no cache of our own
across processes. Within a process every kernel comes from one cache in
`rt::Device` (`GetKernel`), keyed by source, function name and constants.
Library kernels live for the process; emitted kernels are
reference-counted (`AcquireKernel`) and go away with the executable that
owns them.

## Library ops

`GpuCompiler` rewrites dots into `__cublas$lt$matmul` custom calls, and
the StreamExecutor BLAS interface is the plug point:

- **f32 GEMMs** run on Metal Performance Shaders; a bias or activation
  epilogue is a second MSL pass.
- **f16/bf16 GEMMs** run on "steel" kernels ported from MLX, with the
  epilogue fused into the store. Small-batch decode shapes (2-8 rows, K
  >= 512) use MLX's wide gemv.
- **Convolutions** of 4 Mflop and up (1-D and 2-D, ungrouped,
  f32/f16/bf16) go to `metal$conv`, MLX's steel convolution kernels,
  rewritten in the hook CUDA uses for cuDNN. Everything else is an
  XLA-emitted loop kernel. There is no DNN library.
- **Sort, linear algebra and callbacks** follow CUDA's shape: an MSL
  radix sort, Accelerate LAPACK on the shared buffers (GPU kernels up to
  32x32), and an FFI handler for Python callbacks
  ([`callbacks.md`](callbacks.md)).

The one Metal-only rewrite is `metal$scan` for minor-dimension cumulative
ops. MPSGraph, which Apple's jax-metal uses, isn't: it's opaque and has a
record of bugs.

Deliberately off: Triton, cuDNN fusion, autotuning, collectives (a
one-device stub) and float64, which Metal lacks; f64 arithmetic is
refused at compile time.

## Runtime

`metal_pjrt/runtime/metal_runtime.h` is the XLA-free layer every Metal
call goes through (`rt::Device`, `rt::Stream`, `rt::Event`). Its header
comments are the reference; these are the policies.

### Command-buffer batching

A command buffer costs hundreds of microseconds of CPU and scheduler time,
so a stream packs many dispatches into one. It commits when the GPU is
out of our work and at least 16 ops are encoded, if 500 us have passed
since the last commit; explicit syncs commit at once.

Caps keep each buffer far from the GPU watchdog: 1024 ops, 2^27 threads,
or 2e11 GEMM flops. GEMMs are charged an estimate (`blas::GemmWork`)
rather than their thread count. The budget is about 0.1 s of GEMM on an
M3, a >= 5x margin for slower GPUs. A single larger GEMM gets a buffer of
its own, and a buffer that runs over 1 s is logged with its kernels.

Two gaps remain. The budget only splits between dispatches, so one kernel
longer than the watchdog still resets the GPU (a 16384^3 f32 GEMM takes
3-4 s here). And heavy non-GEMM kernels, like large convolutions left to
the loop emitter, are charged only their thread count; `metal$conv`
charges its flops and splits its launches.

### Queue

Since 2026-09-30, each device has one Metal command queue. Every stream
encodes into the open command buffer, buffers commit in encode order, and
each ends by signaling the device timeline (a shared event) with its
sequence number.

**A command buffer never waits on the GPU.** A waiting buffer counts
against the watchdog, and a GPU wait has no timeout. GPU work that
depends on other GPU work is simply behind it in the queue. GPU work that
depends on a host task is encoded only after the task has run; the
encoding thread waits on the host without holding the queue lock. There
is no API for encoding a GPU wait.

A stream is a handle: the timeline value of its last op, the waits its
next op must respect, and a worker thread for host tasks (device-to-host
copies, copies behind pending work, XLA callbacks). An event is a
timeline value plus host-task values. The open buffer commits whenever a
host task is enqueued or an event recorded, so no host wait depends on a
future commit. A host task (or its error callback) must not call into
its own stream or wait for that stream's later work: those calls get
FAILED_PRECONDITION instead of deadlocking. Nor may it call into any
other stream, since an encoding thread may be waiting for the task (no
XLA path does this; it isn't checked).

### Sticky GPU errors

The first failed command buffer (watchdog timeout, page fault, anything)
or failed host task sets the device's error, like a CUDA context error.
From then on every wait, poll, host task and launch returns it, nothing
more is committed, and the message says to restart the process. Host
callbacks still run, since XLA uses them to free memory and fail result
buffers.

Nothing hangs: a failed buffer's completion handler force-signals its
timeline value, timeline waits give up once the device has failed, and no
error path aborts. Host waiters don't rely on completion handlers to
avoid consuming a failed buffer's output: after seeing the timeline
signal, they check the buffers still in flight
(`Device::CheckInFlight`). Skipping that check would save a few
microseconds but is sound only if an aborted buffer never signals, which
is unverified ([`roadmap.md`](roadmap.md)).

### Memory

Allocations are shared-storage `MTLBuffer`s from a size-class cache
behind XLA's pass-through allocator. Fresh buffers cost first-touch page
faults and training steps repeat their sizes, so caching pays.

- **Size classes.** Lengths round to powers of two (at least 256 bytes)
  up to a 16 KB page, whole pages above. A request reuses a cached buffer
  of at most min(2x its length, its length + 2 pages, i.e. +32 KB). Reuse is ordered by the compute stream, as XLA assumes.
  Buffers the host fills right away (module constants, FFT tables) only
  take a cached buffer whose work has all ended.
- **Budget.** Live plus cached memory stays within half of physical RAM,
  capped by the GPU's recommended working set, times
  `METAL_PJRT_MEMORY_FRACTION`. A miss evicts the least recently freed
  buffers, then fails with RESOURCE_EXHAUSTED. The compiler always sees
  the working set as the device size, so programs don't vary with load.
- **Critical pressure.** Like PyTorch MPS, the plugin caps its own
  allocations and otherwise lets macOS page. The one system-level refusal
  is at critical memory pressure (where jetsam starts killing processes),
  checked on allocations of 1 MB or more. The plugin drops its cache,
  waits up to 1 s for in-flight work holding cached buffers, and if the
  level is still critical fails with RESOURCE_EXHAUSTED. The first
  allocation at warning level logs a warning. (A stricter free-pages
  guard, added after swapping wedged the GPU on 2026-09-25, refused most
  ordinary training runs and was replaced by the budget, bounded waits
  and the reset log.)
- **Release tickets.** Every command buffer and host task holds a work
  ticket until it ends. A cached buffer is released only once all work
  that existed when it was freed has ended, since host tasks hold raw
  pointers and argument buffers hold GPU addresses.
- **Idle trim.** Buffers unused for 2 s are released, so an idle process
  gives memory back; a memory-pressure warning releases them all.
- **Tripwires.** A host copy to a pointer outside any live allocation,
  and a free of anything but the start of a live allocation (or with the
  wrong size, or twice), fail with INTERNAL and a log line, since either
  means a use after free. `METAL_PJRT_DEBUG_FREE_QUARANTINE=1` holds
  freed buffers out of reuse to catch late writes. These catch a stale
  pointer only while its memory isn't reused: a stale write into a
  reused buffer stays silent. Open risk: an experiment (since reverted)
  once saw XLA copy into freed memory, mechanism unidentified; a probe
  with only its check found 0 hits in 600 runs. Separately, one
  wrong-value failure of
  `test_source_mutated_right_after_device_put[False-busy-1]` on
  2026-09-28 is unexplained and hasn't recurred.
- **Error messages.** XLA reports every refusal as "Out of memory while
  trying to allocate N"; the plugin appends the reason (budget or
  critical pressure, with numbers) and the executable that was refused.
- **Host memory.** XLA's host pools (for linearizing non-dense arrays and
  `pinned_host`) are anonymous host memory wrapped as `MTLBuffer`s
  (`Device::AllocateHost`), outside the budget and the cache. They never
  shrink.

### Transfers

Host copies are a `memcpy` on the calling thread when the stream is idle,
and a host task otherwise. There's no GPU staging: a staged copy waited on
the GPU for XLA's allocation event and reset the GPU.

JAX lets XLA read the source after `device_put` returns, so a data loader
that refilled its array could change what the device got. So
`device_put` of dense host data **snapshots it** into a malloc'd buffer
before returning, as CUDA does for pageable memory. Above min(256 MB,
max(16 MB, reclaimable memory / 8)) it instead hands XLA the caller's
data and waits until XLA is done, which avoids a second large host copy.
Strided and sub-byte inputs take XLA's synchronous staging path.

### Reset log and quarantine

A watchdog reset hits every process, and several can leave the GPU slow
until a reboot. The runtime appends each reset it sees to
`~/.cache/metal-pjrt/gpu_resets.jsonl` (`METAL_PJRT_STATE_DIR` moves it),
once per incident, with the kernels in the buffer that timed out
(built-in copy and fill kernels are never blamed).

With `METAL_PJRT_QUARANTINE_STRIKES=n` (off by default), a kernel seen in
n resets since boot is refused with FAILED_PRECONDITION, so a compiler
bug costs at most n resets. A reboot or deleting the log lifts it
(`scripts/gpu_health.py --clear` in a checkout). The first reset from a
kernel that never terminates is unavoidable: Metal has no per-kernel
timeout.

## How this differs from MLX

From reading the MLX and jax-mps sources (September 2026): MLX is an
eager interpreter over a lazy graph. `mx.compile` fuses only elementwise
chains, each call re-encodes primitive by primitive, and memory comes
from a caching allocator without planning. Its kernels (GEMM,
convolution, attention, quantized matmul) are hand-tuned.

An XLA backend has the edge where fusion and a static executable matter:
reductions fused with producers, optimizer updates, static buffer
assignment. It's behind where MLX has tuned library kernels and on
compile latency.

## Risks

- **Building XLA on an 8 GB laptop.** A cold build takes about 1.5-2
  hours; a shared local disk cache keeps rebuilds to minutes. (JAX's
  public remote cache had no hits for this build, measured 2026-10-01.)
- **CUDA coupling** in XLA's GPU code: three patches against the pinned
  XLA (`third_party/xla/patches`) and a tripwire test on every pin bump.
- **MSL limits**: 32-bit atomics only, no f64, 32 KB threadgroup memory,
  32-wide SIMD.
