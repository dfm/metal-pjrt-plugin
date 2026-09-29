# Roadmap

The design review's phased plan, with a status line per item and commit
hashes, is kept verbatim in `docs/archive/roadmap-2026-09-27.md`; the
measurements are in `docs/performance.md` and
`docs/archive/performance-log-2026-09.md`.

Guiding rules: mirror what XLA does on CUDA; measure before building;
delete what does not earn its keep; no new upstream patches unless
unavoidable.

Where it stands: the minimum viable product is done (opt-in platform,
wheel, fresh-clone install, user README), and so is every review item not
listed below as decided against, next or deferred.

## Decisions

The maintainer's standing decisions; `CHANGELOG.md` has when each was made.

- Names: JAX platform "mtl" (`jax.devices("mtl")`, `JAX_PLATFORMS=mtl,cpu`,
  env `METAL_PJRT_*`), import package `metal_pjrt_plugin`, PyPI dist
  `metal-pjrt-plugin`, C++ Bazel tree `metal_pjrt/`, state dir
  `~/.cache/metal-pjrt`; the XLA-internal names stay "METAL". "metal"
  collided with Apple's jax-metal (history in `CHANGELOG.md`).
- Opt-in: CPU stays JAX's default backend.
- JAX's compilation-cache settings are left to the user: the plugin never
  sets `jax_compilation_cache_dir` or the thresholds; the README says how
  to turn the cache on.
- Land the minimum viable product before numerical side quests; known
  accuracy gaps are listed in `docs/accuracy.md`, not chased.
- Quarantine strikes count per boot, so an unchanged buggy kernel gets two
  more resets after each reboot (not counted across boots until
  `gpu_health.py --clear`). They are not keyed by plugin build: the build
  changes on every rebuild and would unquarantine an unchanged buggy
  kernel (the kernel key, MSL hash + name, already tracks real changes).
  The build is recorded for diagnostics only.
- A jax/jaxlib version other than the one the plugin was built for gets a
  warning naming the versions to install, and the plugin loads anyway.
- int8 GEMM (s8 x s8 -> s32) stays refused at compile time. The ways out
  were a size-capped elemental fallback (watchdog risk above the cap) or
  an int8 steel GEMM; int8 through the f32 GEMM is exact only for
  K < ~1040.
- The system memory guard stays strict: it counts only free + inactive +
  speculative + purgeable pages, minus a 512 MB reserve, not compressible
  memory, because swapping is what wedged the GPU. The cost: on an 8 GB
  Mac with a browser open it refused nanoGPT's 1.17 GiB step allocation
  with 1.36 GB free (the README says so).
- Metal's biased `exp` / `sin` / `cos` / `log` are accepted as documented
  in `docs/accuracy.md`; the prelude's polynomials for |x| < 0.125 stay,
  and nothing further is planned.
- A/Bs of sub-millisecond programs interleave the arms and report
  p10/median/p90 (GPU performance states make single medians bimodal).
- The size-class cache is the only device allocator (no BFC option).
  f16/bf16 GEMMs run only on steel, with its out-of-range shapes refused
  at compile time. MLX is the one benchmark reference.

## Decided against

- **Our own PJRT client** (3b209b5): it would remove only ~15
  of patch 0001's ~200 lines; there is no `StreamExecutorGpuClient` class
  to subclass at this pin; zero-copy host import would need a ~400-line
  client fork. The small `GetPjrtApi` wrapper stays.
- **Host LAPACK as a stream host task** (59ec97d, built and measured):
  not faster (cholesky 128 median 313 -> 299 us, same p90; 100 unblocked
  `cho_solve` +12%). Each call still pays two dependent GPU round trips
  and the hold rule keeps the dispatch thread waiting.
- **Async Python callbacks**: stay synchronous. A callback on the stream's
  host-task worker could make a commit wait for a task that needs the GIL,
  and the host-LAPACK experiment above showed the host-task path buys
  nothing here.
- **The softmax rewriter** (fcbf5ce): 1.5-1.8x on standalone softmax, but
  it never fires under autodiff and gave nothing where it fired. Cost:
  standalone softmax 8192x1024 1.24 -> 1.91 ms (MLX 0.97).
- **XLA command buffers and Metal indirect command buffers** (removed
  bdb9c4c): measured; a wash or slower once the runtime was fixed.
- **MPSGraph for program subsets**: what jax-metal does; opaque, a record of
  bugs, and still about one kernel per op for scan-shaped programs.
- **An `AppleComputeCapability` upstream patch**: the tripwire test and
  post-GEMM checks guard the OneAPI branches instead. Also rejected as
  codegen routes: LLVM IR -> AIR, SPIR-V -> SPIRV-Cross.
- From XLA:CPU, despite unified memory: reporting device memory as on-CPU
  (buffer-protocol views and the CPU copy fast paths assume synchronously
  written memory, so readers would race in-flight GPU writes); running HLO
  subgraphs through XLA:CPU (two compilers and a partitioner); host
  offloading / `pinned_host` memory kinds (a copy between two views of the
  same RAM).
- Smaller ones: `const` on kernel arguments, an on-disk kernel cache and
  per-encoder reset blame (measured, `docs/performance.md`); purgeable
  cached buffers (not built: the 2 s idle release does the job).

## Next

Open decision for the maintainer:

- **Accuracy policy**: match numpy/IEEE on valid inputs, or XLA:CPU?
  `docs/accuracy.md`, "Policy", has the question and the cases it decides.

Still on hold:

- macOS CI (and a remote cache).
- The minimum macOS version the wheel declares.
- Full third-party notices for the statically linked XLA dependencies:
  before a release, alongside CI.

Housekeeping: drop the old `~/.cache/jax_metal/device.lock` in
`scripts/device_lock.py` at the next pin bump (not before 2026-10-31).

Watchdog: tile giant GEMMs (over ~1e12 flops) over N or batch into several
dispatches, so the command-buffer flops budget can split them; today one
such GEMM is a single multi-second kernel (docs/design.md, Runtime).

Upstreaming candidate: patch 0001's `gpu_module_globals.cc` hunk
(`CHECK_OK(stream->BlockHostUntilDone())` -> `ABSL_RETURN_IF_ERROR`). On
any backend, a failed wait after uploading an executable's constants
aborts the process instead of returning the error. Returning an error instead
of CHECK means the constant copies may still be queued: callers must keep
the executable's constant storage alive until the stream drains.

Upstreaming candidate: `GetBatchRowColumnShape` (`matmul_utils.cc`)
accumulates dims in int: 2^31 aborts, 2^32 becomes a 0-row GEMM (affects
CUDA too). The plugin refuses such GEMMs at compile time
(`CheckGemmGroupsFitInt32`, tripwired).

## Deferred / unverified

- The README's "Unverified" item lists what was never tried (a real GPU
  fault or watchdog timeout, jax-metal side by side, other Macs). Beyond
  that: whether every work ticket ends after a sticky error is untested
  (if not, cached buffers stay until the process exits).
- The out-of-memory retry (drop the cache, wait for in-flight work, retry)
  and the real memory-pressure notification (`sudo memory_pressure -S -l
  warn`) are untested; the pressure handler runs through a test hook.
- MLX comparison numbers not re-run on an idle, freshly booted machine.
- Patches 0002 and 0003 not sent upstream.
- The wheel is untested elsewhere and not stripped (236 MB dylib; `strip
  -x` would save ~90 MB).
- **The 6-9 us no-wait variant**, until a profile shows it matters: drop
  `waitUntilCompleted` in `CheckInFlight` and save 6-9 us per
  synchronizing round trip. Sound only if an aborted command buffer never
  runs its own trailing signals, which is unverified.

## Deferred / ideas

Unscheduled directions, with enough context to pick one up cold.

- **Untracked resources + `MTLResidencySet` + a concurrent encoder**, with
  barriers from `xla/runtime/execution_graph.h`. Measure first.
- **A dedicated MLIR -> MSL printer** instead of EmitC plus macro shims
  (`#define _Float16 half`).
- **Triton IR -> MSL** (`CompileTritonToLlvm` already reaches
  `MetalKernelCompiler`): reductions/elementwise tiles first, then `tt.dot`
  -> `simdgroup_matrix`. Multi-week; only if fusion quality matters.
- **Convolutions**, the largest workload gap (cnn fwd+bwd 6.2 vs MLX
  2.3 ms): no library path today.
- **Upstream: donation for plugin platforms.** JAX hard-codes the
  platforms that get buffer donation (`mlir._platforms_with_donation`); the
  plugin appends "mtl" to the private list at initialization (pinned by
  `tests/test_jax_private_api.py`). A JAX PR letting a PJRT plugin declare
  donation support (e.g. a `register_plugin` option) would remove that.
- **A native FFT** (the dense-DFT lowering is O(n^2) per axis; MPS has no
  FFT for arbitrary sizes). **c64** in the emitter (`complex<f32>` as
  `float2`).
- **Tuned library kernels.** Apple's jax-metal hands whole programs to
  MPSGraph; the one thing it gets that we do not is Apple's tuned
  convolution and attention kernels. In the order worth trying:
  1. MPSGraph as a per-op library, as XLA:GPU uses cuDNN: a custom call
     running a fixed-shape `MPSGraphExecutable` on our `MTLBuffer`s (no
     copies; the bug reputation is about whole-program use). Needs an
     `MPSCommandBuffer` over ours and an on-disk executable cache. Targets:
     conv and its grads, SDPA (macOS 15+), FFT. Days, not weeks.
  2. More MLX steel kernels (MIT; the f16/bf16 GEMM port 31a9623 shows the
     process): quantized matmul, decode-shaped attention, steel conv as a
     fallback, gather-matmul and segmented reductions.
  3. Metal 4 MetalPerformancePrimitives (`matmul2d` / `conv2d` in-shader,
     macOS 26) inside generated fusions: the only route to a tuned GEMM with
     an arbitrary XLA epilogue fused in.
  4. With two or more implementations per op, XLA's GEMM / conv autotuner
     in front of them (today: f32 on MPS, f16/bf16 on steel, fixed).
- **float64 by double-float emulation** (jax-metal rejects f64): a (hi, lo)
  float pair with error-free transforms in the emitter, about 48 mantissa
  bits (~1e-14 relative) and f32's exponent range. Start with add / sub /
  mul / div / sqrt / fma / compare / convert / reduce; refuse the rest at
  compile time. Storage stays 8 bytes.
- **Small fusions around host LAPACK** off the GPU (or folded into the
  handler): the only way small factorizations approach CPU + one dispatch.
- **Rows 65..~2048 in the radix sort** use one threadgroup per row
  (1000x65: 0.95 ms); a multi-row path would help if someone needs it.
