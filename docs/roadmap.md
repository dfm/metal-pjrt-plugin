# Roadmap

Decisions made, options decided against, and what is next. The
measurements are in `docs/performance.md`; `CHANGELOG.md` has the dated
record.

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
- Memory limits as in PyTorch MPS (2026-09-29): the per-process budget is
  the limit and macOS pages below it; the only system-level refusal is at
  critical memory pressure. The free-pages guard it replaced refused 25 of
  28 airbench94 runs on an 8 GB Mac at 50-70% free (`docs/design.md`).
- Metal's biased `exp` / `sin` / `cos` / `log` are accepted as documented
  in `docs/accuracy.md`; the prelude's polynomials for |x| < 0.125 stay,
  and nothing further is planned.
- Accuracy policy: match XLA:CPU. Where Metal's math and XLA:CPU
  disagree, follow CPU, including its flushing of subnormal inputs and
  outputs (the subnormal `log` fix was reverted); a prelude special case
  stays only when it brings mtl closer to CPU. The rest are known gaps in
  `docs/accuracy.md`.
- A/Bs of sub-millisecond programs interleave the arms and report
  p10/median/p90 (GPU performance states make single medians bimodal).
- The size-class cache is the only device allocator (no BFC option).
  f16/bf16 GEMMs run only on steel, with its out-of-range shapes refused
  at compile time. MLX is the one benchmark reference.

## Decided against

- **Our own PJRT client** (2026-09-27): it would remove only ~15
  of patch 0001's ~200 lines; there is no `StreamExecutorGpuClient` class
  to subclass at this pin; zero-copy host import would need a ~400-line
  client fork. The small `GetPjrtApi` wrapper stays.
- **Host LAPACK as a stream host task** (2026-09-27, built and measured):
  not faster (cholesky 128 median 313 -> 299 us, same p90; 100 unblocked
  `cho_solve` +12%). Each call still pays two dependent GPU round trips
  and the hold rule keeps the dispatch thread waiting.
- **Async Python callbacks**: stay synchronous. A callback on the stream's
  host-task worker could make a commit wait for a task that needs the GIL,
  and the host-LAPACK experiment above showed the host-task path buys
  nothing here.
- **The softmax rewriter** (2026-09-27): 1.5-1.8x on standalone softmax, but
  it never fires under autodiff and gave nothing where it fired. Cost:
  standalone softmax 8192x1024 1.24 -> 1.91 ms (MLX 0.97).
- **XLA command buffers and Metal indirect command buffers** (removed
  2026-09-26): measured; a wash or slower once the runtime was fixed.
- **MPSGraph for program subsets**: what jax-metal does; closed source (it
  cannot be debugged or fused into), and still about one kernel per op for
  scan-shaped programs.
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

Before the repository goes public:

- A build from a clean clone with empty Bazel caches on the release commit,
  then the full suites against that library: the source build is the only
  install path at launch.
- What happens when a program is stopped mid-computation (Ctrl-C,
  `kill -9`), and how long the GPU watchdog allows: both untested (README,
  "GPU safety"). Needs someone at the machine; a wrong guess resets the GPU.
- The suites on a second Apple GPU and macOS version: every tolerance was
  measured on one M3.
- CI (`.github/workflows/ci.yml`) has never run: its jobs are skipped on a
  private repository, so its first run is the one after publication, a cold
  build on a hosted runner.
- `bench/results/table.md` (predated the platform rename, no MLX version):
  dropped (2026-10-01): generated locally by `bench/run_all.sh`.

Decided (2026-09-30): the launch is source only (no wheel on a release, no
PyPI); the deployment target is macOS 26.0; `THIRD_PARTY_NOTICES` carries
the license texts of everything linked into the library.

Small-M bf16/f16 GEMM: done (2026-09-29: MLX's wide gemv for 2..8 rows, a
16-row steel tile to 48, DotMerger off). 28 independent [M,1024]x[1024,6144]
GEMMs at M=2: 15.03 -> 4.47 ms p10 (MLX 4.07); Qwen3-0.6B bf16 decode at
batch 2: 26.71 -> 13.88 ms. The gemv needs K >= 512 (a sweep over batch,
rows and K: below that the 16-row tile wins by up to 2.8x), so batched
decode attention, 1024 x [4..8, 128] x [128, 128]^T, is 1.9-2.0x faster
(6.9 -> 3.5 ms per 8 GEMMs; `docs/performance.md`).

Host memory per compiled executable: attributed (`docs/performance.md`,
host memory). "Unchanged after del + gc" was JAX's own cache, which keeps
every executable of a live jitted function (`jax.clear_caches()` or
deleting the function frees it: malloc 885 -> 136 MB for 3 LoRA steps).
The plugin's parts are fixed: compiled kernels now go with their
executables (the runtime cache kept them, ~40 KB each, for the process),
and emitted kernels carry a one-line stand-in for the ~9 KB MSL prelude
(-40 MB per LoRA executable, 272 -> 232 MB of malloc). The rest is XLA's:
two copies of each kernel thunk's MSL (the thunk's and the executable's
serialized thunks), the HLO module and annotations; footprint also keeps
~300 MB of freed-but-dirty malloc pages after a compile (fragmentation, as
on CPU). Left open: an upstream change so thunks share one kernel binary
(and the serialized copy is made on demand); examples/lora could use
coarser length buckets (`--pad-to`; each length is an executable).

OOM diagnostics: done (2026-09-29). A refused allocation's RESOURCE_EXHAUSTED
now names the executable that asked for it, also when the error surfaces in
a later eager op.

Conv weight gradient: done (2026-09-29 vectorized unfold, 2026-09-29 GEMM tile
and split-K sized together). bf16 N=1024 31x31 24->64: 53.8 -> 18.2 ms;
airbench94 to 94% in 229-259 s vs torch-MPS 254 +/- 20 s, at 3.3 vs 5.9 GB
peak (`docs/performance.md`). Deferred: an implicit-GEMM weight gradient
(saves only the unfold, at most a third on 31x31). The system memory
guard that refused 3 of 5 airbench94 runs at 1.1-1.4 GB free is gone:
done, only critical memory pressure refuses now (see Decisions).

Winograd convolutions: not now (2026-10-01): ~2% of the airbench step,
forward only. MLX's F(6,3) Winograd runs airbench94's 15x15 64->256
forward (bf16, N = 1024) in 16.34 ms vs our 22.45 (1.37x), but its 7x7
layers only 1.04x and the 3x3 one slower; MLX has no Winograd weight or
input gradient (ours beat its weight gradients). Ported, it would save
~6 ms of the 308 ms step (`docs/performance.md`, the airbench94 step
without rematerialization). The port plan, if the forward ever matters
more: MLX v0.32.3's `conv.metal` transforms and `winograd_conv_2D_gpu`
as a `ConvPath::kWinograd` (3x3, stride 1, C and O multiples of 32,
output 13x13 or larger), its 64-point GEMM on our steel GEMM, the input
chunked over N to 64 MB; ~600-700 lines with tests, 2-3 days; the
intermediate bf16 stages lose ~2-3 bits, so it needs its own conv_test
tolerance, a 5-seed bf16 airbench94 accuracy check and an off switch.

Low priority (dfm, 2026-10-01; no work now):

- Don't overfit to airbench. Further convolution work (the weight
  gradients, ~104 ms of the step at 1.5-2.6 vs ~3.0 TFLOP/s for the
  forward and input gradients) uses general rules and kernels, never
  per-shape tables, and is judged on a broad convolution shape set
  (ResNet 3x3 / 1x1, RGB stems, strided, depthwise / grouped, 1-D, batch
  sizes, f32 / bf16) with no regressions.
- A broader benchmark sweep: ResNet-18/50 train steps, a small transformer
  train step, a U-Net (upsampling, transposed convolutions), with MLX and
  torch-MPS references under the thermal protocol; their shapes become the
  convolution shape set.
- MPSGraph per op (as XLA:GPU uses cuDNN; option 1 of "Tuned library
  kernels" below): a custom call running a cached fixed-shape
  `MPSGraphExecutable` encoded into our command buffer (e.g. the conv
  weight gradient). Measure first: torch-MPS per op (forward, input and
  weight gradients) against ours on the shape set; prototype only if
  MPSGraph wins broadly. Concerns: encoding into the single queue through
  `MPSCommandBuffer`, per-shape compile latency and caching, the work
  budget (opaque kernels: charge their flops), closed source.

Memory cache at warn pressure: the pressure handler releases every cached
buffer on each free while pressure is at warn, so the cache is off. Consider
trimming to a fraction instead (airbench94 run 2 spent most of its time at
warn: 257 vs 218 s for run 1).

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

- The README's "Unverified" item lists what was never tested (the error
  path after a real GPU fault or watchdog timeout, jax-metal side by side,
  other Macs). Beyond
  that: whether every work ticket ends after a sticky error is untested
  (if not, cached buffers stay until the process exits).
- The critical-pressure refusal and its retry (drop the cache, wait for
  in-flight work, read the level again) and the real memory-pressure
  notification (`sudo memory_pressure -S -l warn`) are untested at a real
  level; tests fake the level through a test hook.
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
- **Convolutions**: 1-D/2-D f32/f16/bf16 on MLX's steel kernels
  (`metal$conv`; cnn fwd+bwd 5.5 -> 1.56 ms p10, MLX 1.42). Not yet:
  grouped / depthwise, Winograd, 3-D, negative low padding (sliced), and
  small convolutions under 4 Mflop (loop emitter).
- **Grouped-convolution kernel gradients**: batch-group convolutions are
  converted to ordinary ones first (ConvolutionGroupConverter, as on
  XLA:CPU), so they are correct; the converted kernel gradient of a
  grouped (non-depthwise) convolution is a 3-spatial-dim dilated
  convolution that runs on the loop emitter (slow but correct). A faster
  path (a grouped `metal$conv` or a direct batch-group kernel) is future
  work. Measured 2026-09-30: grad wrt w of x f32[8,32,32,64], 3x3, groups
  4: 2.26 ms p10 vs 1.44 ms dense (4x the flops, on `metal$conv`).
- **Upstream: donation for plugin platforms.** JAX hard-codes the
  platforms that get buffer donation (`mlir._platforms_with_donation`); the
  plugin appends "mtl" to the private list at initialization (pinned by
  `tests/test_jax_private_api.py`). A JAX PR letting a PJRT plugin declare
  donation support (e.g. a `register_plugin` option) would remove that.
- **FFT**: native on MLX's FFT kernels (`metal$fft`, one call per axis;
  0.6-1.3x MLX's time, 2.5-44x faster than the dense DFT it replaces).
  Not yet: MLX's Bluestein twiddle-table path for n = 4096 with many rows
  (not ported, unmeasured); lengths above 2^24 (powers of two) or 2^23 - 1
  fall back to the O(n^2) DFT; no complex128.
- **Tuned library kernels.** Apple's jax-metal hands whole programs to
  MPSGraph; the one thing it gets that we do not is Apple's tuned
  convolution and attention kernels. In the order worth trying:
  1. MPSGraph as a per-op library, as XLA:GPU uses cuDNN: a custom call
     running a fixed-shape `MPSGraphExecutable` on our `MTLBuffer`s (no
     copies). Needs an
     `MPSCommandBuffer` over ours and an on-disk executable cache. Targets:
     conv and its grads, SDPA (macOS 15+). Days, not weeks.
  2. More MLX steel kernels (MIT; the f16/bf16 GEMM port 2026-09-25 shows the
     process): quantized matmul, decode-shaped attention, gather-matmul
     and segmented reductions (steel conv is done: `metal$conv`).
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
