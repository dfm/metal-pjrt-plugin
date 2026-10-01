# Roadmap

Decisions made, options turned down, and what's next. Measurements live in
[`performance.md`](performance.md); [`CHANGELOG.md`](../CHANGELOG.md) has
the dated record.

The guiding rules: mirror what XLA does on CUDA, measure before building,
delete what doesn't earn its keep, and avoid new upstream patches unless
there's no other way.

Where it stands: the minimum viable product is done (an opt-in platform,
a wheel, a fresh-clone install, a user README), along with every review
item not listed below as declined, next or deferred.

## Decisions

The maintainer's standing decisions; `CHANGELOG.md` says when each was
made.

- **Names.** JAX platform "mtl" (`jax.devices("mtl")`,
  `JAX_PLATFORMS=mtl,cpu`, environment variables `METAL_PJRT_*`), import
  package `metal_pjrt_plugin`, PyPI name `metal-pjrt-plugin`, Bazel tree
  `metal_pjrt/`, state directory `~/.cache/metal-pjrt`. XLA-internal names
  stay "METAL". The name "metal" collided with Apple's jax-metal.
- **Opt-in.** CPU stays JAX's default backend.
- **Compilation cache.** The plugin never sets `jax_compilation_cache_dir`
  or its thresholds; the README says how to turn the cache on.
- **Versions.** A jax/jaxlib other than the one the plugin was built for
  gets a warning naming the versions to install, and the plugin loads
  anyway.
- **Launch (2026-09-30).** Source only: no wheel on a release and no PyPI.
  The deployment target is macOS 26.0. `THIRD_PARTY_NOTICES` carries the
  license texts of everything linked into the library.
- **Accuracy.** Match XLA:CPU. Where Metal's math and XLA:CPU disagree,
  follow CPU, including its flushing of subnormal inputs and outputs; a
  prelude special case stays only if it brings mtl closer to CPU. Metal's
  biased `exp`, `sin`, `cos` and `log` are accepted as documented in
  [`accuracy.md`](accuracy.md); the prelude's polynomials for |x| < 0.125
  stay. Known gaps are listed there, not chased.
- **Memory limits as in PyTorch MPS (2026-09-29).** The per-process budget
  is the limit and macOS pages below it; the only system-wide refusal is
  at critical memory pressure. The free-pages guard this replaced refused
  25 of 28 airbench94 runs on an 8 GB Mac with 50-70% of memory free.
- **Quarantine strikes** count per boot, so an unchanged buggy kernel can
  cause n more resets after each reboot. They aren't keyed by plugin
  build, which changes on every rebuild and would release an unchanged
  buggy kernel; the kernel key (MSL hash and name) already tracks real
  changes.
- **int8 GEMM** (s8 x s8 -> s32) stays refused at compile time. The
  alternatives were a size-capped elementwise fallback (a watchdog risk
  above the cap) or an int8 steel GEMM; going through the f32 GEMM is
  exact only for K below ~1040.
- **One of each.** The size-class cache is the only device allocator.
  f16/bf16 GEMMs run only on steel, and shapes outside its range are
  refused at compile time. MLX is the one benchmark reference.
- **A/Bs** of sub-millisecond programs interleave the arms and report
  p10 / median / p90.

## Declined

- **Our own PJRT client** (2026-09-27). It would remove only ~15 of patch
  0001's ~200 lines, there's no client class to subclass at this pin, and
  zero-copy host import would need a ~400-line client fork.
- **Host LAPACK as a stream host task** (2026-09-27, built and measured):
  no faster, because each call still pays two dependent GPU round trips.
- **Asynchronous Python callbacks.** A callback on the stream's host-task
  worker could make a commit wait for a task that needs the GIL, and the
  LAPACK experiment showed the host-task path buys nothing here.
- **The softmax rewriter** (2026-09-27): it never fires under autodiff.
- **XLA command buffers and Metal indirect command buffers** (removed
  2026-09-26): a wash or slower once the runtime was fixed.
- **MPSGraph for whole subprograms**, as jax-metal does: closed source,
  can't be debugged or fused into, and still about one kernel per op for
  scan-shaped programs. (MPSGraph *per op* is still an idea; see below.)
- **An upstream `AppleComputeCapability` patch**: a tripwire test and
  post-GEMM checks guard XLA's OneAPI branches instead. Also declined as
  codegen routes: LLVM IR -> AIR, and SPIR-V -> SPIRV-Cross.
- **Borrowing from XLA:CPU despite unified memory**: reporting device
  memory as CPU memory (buffer views and CPU copy paths assume
  synchronously written memory, so readers would race GPU writes), running
  HLO subgraphs through XLA:CPU (two compilers and a partitioner), and
  `pinned_host` memory kinds (a copy between two views of the same RAM).
- **Smaller ones**: `const` on kernel arguments, an on-disk kernel cache,
  per-encoder reset blame (all measured, see
  [`performance.md`](performance.md#measured-and-dropped)), and purgeable
  cached buffers (the 2 s idle release does the job).

## Before the repository goes public

- **A clean-clone build** with empty Bazel caches on the release commit,
  then the full suites against that library, since a source build is the
  only install path. Done on 2026-10-01 from a GitHub clone with empty
  caches, following the README verbatim on the 8 GB M3: a cold build of 95
  minutes (5685 s, ~0.5 GB downloaded), 8.2 GB of output plus a 4.5 GB
  disk cache; host tests 13/13, device tests 13/13, pytest 905 passed and 2
  expected failures, JAX's lax tests 996 passed with 13 known failures and
  no new ones. Redo it on the release commit.
- **Stopping a program mid-computation**, and how long the GPU watchdog
  allows: both untested. The advice meanwhile: stop a GPU job with Ctrl-C
  (`device_lock.py` passes it to the job), never `kill -9`, and let jobs
  finish when you can. This needs someone at the
  machine, because a wrong guess resets the GPU.
- **A second Apple GPU and macOS version.** Every tolerance and
  performance number comes from one M3 on macOS 26.2; other GPUs or macOS
  versions (a different Metal compiler and math library) may need
  retuned tolerances.
- **CI** (`.github/workflows/ci.yml`) has never run: its jobs skip private
  repositories, so the first run is the one after publication, a cold
  build on a hosted runner.

## Untested

- **The GPU-error path** is tested only with injected failures. Real
  watchdog resets happened during development (the last on 2026-09-30,
  from a change since reverted), but none of them exercised that path, and
  the watchdog's timeout was never measured.
- **Running next to jax-metal** is argued from the code, never tried.
- **Work tickets after a sticky error**: whether every ticket ends is
  untested. If not, cached buffers stay until the process exits.
- **Critical memory pressure.** The refusal and its retry (drop the
  cache, wait for in-flight work, read the level again) and the real
  memory-pressure notification (`sudo memory_pressure -S -l warn`) haven't
  been tested at a real pressure level; the tests fake the level.
- **MLX comparisons** haven't been re-run on an idle, freshly booted
  machine.
- **The wheel** is untested on other machines and not stripped (a 236 MB
  library; `strip -x` would save ~90 MB).

## Next

- **The memory cache at warn pressure.** While pressure is at warn, the
  pressure handler releases every cached buffer on each free, so the cache
  is effectively off. Trimming to a fraction instead might help (an
  airbench94 run that spent most of its time at warn took 257 s, against
  218 s for one that didn't).
- **Giant GEMMs.** Tile GEMMs over ~1e12 flops over N or batch into
  several dispatches, so the command-buffer budget can split them; today
  one is a single multi-second kernel that can trip the watchdog (see
  [`design.md`](design.md#runtime)).
- **Housekeeping.** Drop the old `~/.cache/jax_metal/device.lock` from
  `scripts/device_lock.py` at the next pin bump (not before 2026-10-31).
- **Upstreaming candidates.**
  - Patch 0001's `gpu_module_globals.cc` hunk: on any backend, a failed
    wait after uploading an executable's constants aborts the process
    (`CHECK_OK`) instead of returning the error. Returning instead means
    callers must keep the constant storage alive until the stream drains.
  - `GetBatchRowColumnShape` (`matmul_utils.cc`) accumulates dimensions in
    `int`: 2^31 aborts and 2^32 becomes a 0-row GEMM (CUDA too). The plugin
    refuses such GEMMs at compile time (`CheckGemmGroupsFitInt32`).
  - Patches 0002 and 0003.
  - Letting a PJRT plugin declare donation support. JAX hard-codes the
    platforms that get buffer donation, so the plugin appends "mtl" to a
    private list at start-up (pinned by `tests/test_jax_private_api.py`).
  - Sharing one kernel binary between thunks (and serializing on demand),
    which would remove most of XLA's per-executable MSL copies (see
    [`performance.md`](performance.md#host-memory-per-executable)).

## Low priority (2026-10-01)

- **Don't overfit to airbench.** Further convolution work (the weight
  gradients run at 1.5-2.6 TFLOP/s against ~3.0 for the forward and input
  gradients) should use general rules and kernels, never per-shape tables,
  and be judged on a broad set of shapes (ResNet 3x3 and 1x1, RGB stems,
  strided, depthwise and grouped, 1-D, several batch sizes, f32 and bf16)
  with no regressions.
- **A broader benchmark sweep**: ResNet-18/50 and small-transformer train
  steps and a U-Net (upsampling, transposed convolutions), with MLX and
  PyTorch MPS references under the thermal protocol. Their shapes become
  the convolution shape set.
- **MPSGraph per op**, the way XLA:GPU uses cuDNN: a custom call running a
  cached fixed-shape `MPSGraphExecutable` inside our command buffer (for
  example for the conv weight gradient). Measure first: PyTorch MPS per op
  against the plugin on the shape set, and prototype only if MPSGraph is
  ahead broadly. Concerns: encoding into the single queue through
  `MPSCommandBuffer`, per-shape compile latency and caching, charging opaque
  kernels against the watchdog budget, and closed source.
- **GPU linear algebra beyond 32x32**, to avoid the host sync, not for raw
  speed. Measured 2026-10-01 against PyTorch 2.14.1's MPS kernels:
  Accelerate wins or ties up to n = 1024, including batched 256 x 64^2
  Cholesky and LU (0.55 vs 1.15 ms); PyTorch's GPU Cholesky reaches parity
  only at n = 1024. The one clear GPU win is batched small SVD by one-sided
  Jacobi (1.8-2.4x). Our real cost above 32x32 is the full GPU sync inside
  jitted programs. If pursued: extend our GPU Cholesky and LU past 32x32
  along PyTorch's blocked right-looking design (a register- or
  threadgroup-resident panel factor, staged row swaps, TRSM, a
  `simdgroup_matrix` Schur/SYRK update, the batch on a grid dimension, a
  streaming pivot search for tall panels; PyTorch's
  `aten/src/ATen/native/mps/kernels/LinearAlgebra.metal`, BSD-3, so a
  notice if ported); batched Jacobi SVD and eigh for small matrices (one
  threadgroup per matrix, circle-method rotations); and choose GPU or CPU
  on batch x size, not size alone. Not worth copying: their QR (an O(n^4)
  orgqr and an apparent barrier bug) or their per-matrix batching of
  `MPSMatrixSolveTriangular`.
- **Winograd convolutions** (not now, 2026-10-01): about 2% of the
  airbench step, forward only. MLX's F(6,3) Winograd runs airbench94's
  15x15 64->256 forward (bf16, N = 1024) in 16.34 ms against 22.45 ms here,
  its 7x7 layers in about the same time, and its 3x3 layer slower; MLX has
  no Winograd gradients. A port would save ~6 ms of a 308 ms step. If the
  forward ever matters more: MLX v0.32.3's `conv.metal` transforms and
  `winograd_conv_2D_gpu` as a `ConvPath::kWinograd` (3x3, stride 1, C and O
  multiples of 32, output 13x13 or larger), its 64-point GEMM on our steel
  GEMM, the input chunked over N to 64 MB. About 600-700 lines with tests,
  2-3 days. Its bf16 intermediate stages lose ~2-3 bits, so it would need
  its own tolerance, a 5-seed bf16 airbench94 accuracy check and an off
  switch.
- **An implicit-GEMM weight gradient**: saves only the unfold, at most a
  third of the 31x31 layer's time.
- **The no-wait variant of `CheckInFlight`**: dropping its
  `waitUntilCompleted` would save 6-9 us per synchronizing round trip, but
  is sound only if an aborted command buffer never runs its trailing
  signals, which is unverified. Not until a profile shows it matters.
- **examples/lora** could use coarser length buckets (`--pad-to`), since
  each length is its own executable.

## Ideas

Unscheduled directions, with enough context to pick one up cold.

- **Untracked resources, `MTLResidencySet` and a concurrent encoder**, with
  barriers from `xla/runtime/execution_graph.h`. Measure first.
- **A dedicated MLIR -> MSL printer** instead of EmitC plus macro shims
  (`#define _Float16 half`).
- **Triton IR -> MSL** (`CompileTritonToLlvm` already reaches
  `MetalKernelCompiler`): reduction and elementwise tiles first, then
  `tt.dot` -> `simdgroup_matrix`. Several weeks; only if fusion quality
  matters.
- **More convolutions on fast paths**: grouped and depthwise, 3-D,
  negative low padding (sliced today), and convolutions under 4 Mflop all
  run on XLA's loop emitter. Grouped convolutions are converted to
  ordinary ones first (as on XLA:CPU), so they're correct, but the kernel
  gradient of a grouped convolution becomes a 3-D dilated convolution on
  the loop emitter. Measured 2026-09-30: the weight gradient of x
  f32[8,32,32,64], 3x3, 4 groups takes 2.26 ms p10, against 1.44 ms for
  the dense one with 4x the flops.
- **FFT gaps**: MLX's Bluestein twiddle-table path for n = 4096 with many
  rows (not ported, unmeasured); lengths above 2^24 (powers of two) or
  2^23 - 1, which raise an error; complex128.
- **Tuned library kernels.** The one thing jax-metal gets from MPSGraph
  that we don't is Apple's tuned convolution and attention kernels. In the
  order worth trying:
  1. MPSGraph as a per-op library (above). Targets: convolutions and their
     gradients, SDPA (macOS 15+). Days, not weeks.
  2. More MLX steel kernels (MIT; the f16/bf16 GEMM port of 2026-09-25
     shows the process): quantized matmul, decode-shaped attention,
     gather-matmul and segmented reductions.
  3. Metal 4 MetalPerformancePrimitives (`matmul2d` / `conv2d` in-shader,
     macOS 26) inside generated fusions: the only route to a tuned GEMM
     with an arbitrary XLA epilogue fused in.
  4. With two or more implementations per op, XLA's GEMM and convolution
     autotuner in front of them (today the choice is fixed: f32 on MPS,
     f16/bf16 on steel).
- **float64 by double-float emulation** (jax-metal has no f64 either): a
  (hi, lo) float pair with error-free transforms in the emitter, about 48
  mantissa bits (~1e-14 relative) and f32's exponent range. Start with
  add, sub, mul, div, sqrt, fma, compare, convert and reduce, and refuse
  the rest at compile time. Storage stays 8 bytes.
- **Small fusions around host LAPACK** moved off the GPU (or folded into
  the handler): the only way small factorizations get close to CPU plus
  one dispatch.
- **Radix sort for rows of 65 to ~2048**, which use one threadgroup per
  row today (1000x65: 0.95 ms). A multi-row path would help if someone
  needs it.
