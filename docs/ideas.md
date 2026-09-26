# Ideas

Unscheduled directions, with enough context to pick one up cold. Nothing here
is committed work; see `docs/design.md` for the plan. Dated so stale entries
are easy to spot.

## Tuned library kernels (2026-09-25)

Context: jax-metal (Apple's closed plugin, inspected at 0.1.1) does no kernel
generation of its own. It lowers MHLO into a private `mps` MLIR dialect,
serializes it to MLIR bytecode, and hands the whole program to
`-[MPSGraphExecutable initWithMLIRBytecode:executableDescriptor:]`. Every
fusion, schedule and kernel comes from MPSGraph's closed compiler. That is why
its op coverage is exactly the `mps` dialect and why f64 is rejected by a type
verifier. The one thing it gets for free that we do not is Apple's tuned
convolution and attention kernels. Three ways to close that gap, in the order
worth trying:

### 1. MPSGraph as a per-op library

Use the public MPSGraph API the way XLA:GPU uses cuDNN: a rewriter turns
conv / attention HLO into a library custom call, and a thunk runs a
fixed-shape `MPSGraphExecutable` built from a two-node graph.

- `MPSGraphTensorData` wraps an existing `MTLBuffer`, so no copies.
- Executables encode onto an `MPSCommandBuffer`, not a bare
  `MTLCommandBuffer`. The thunk has to wrap ours and stay inside the
  per-command-buffer work budget from the GPU-wedge fix.
- Compile latency is real. Metal's system shader cache already makes a
  repeated `newLibraryWithSource` cheap (186 ms first, 1.3 ms in a later
  process for a small kernel); an MPSGraph executable would need its own
  on-disk cache.
- Targets: conv forward / data grad / weights grad; scaled dot-product
  attention (first-class op on macOS 15+); FFT, which we have no other source
  for.
- The bug reputation noted in `design.md` is about whole-program use, where
  shape inference, control flow and fusion are all exercised. A single-op,
  fixed-shape executable touches almost none of that surface.

Rough cost: a thunk, a rewriter, and a cache. Days, not weeks.

### 2. More MLX steel kernels

The bf16/f16 GEMM port (commit 31a9623) shows the process: MLX already
JIT-compiles these kernels from MSL strings with function constants, so they
drop into `newLibraryWithSource` with a few hundred lines of host-side tiling
logic each. MIT licence, keep the attribution header. Worth taking, in order:

- Quantized matmul (`qmm`). No Apple library equivalent; matters for
  inference on 8 to 16 GB machines.
- Decode-shaped attention. MPSGraph's SDPA is tuned for training shapes;
  MLX's small-batch inference kernels are much faster there.
- Steel conv (implicit GEMM and Winograd) as a fallback for shapes MPSGraph
  handles badly. Lower priority if idea 1 lands.
- Gather-matmul and segmented reductions. Small.

### 3. Metal 4 MetalPerformancePrimitives inside fusions

macOS 26 ships in-shader `matmul2d` / `conv2d` tensor primitives. Unlike the
two options above, these can live inside our generated fusion kernels, which
is the only route to a tuned GEMM with an arbitrary XLA epilogue fused in.
Touches the emitter, so this is the follow-up once the library path exists.

### Autotuning across sources

Once there are two or more implementations per op (MPS matrix kernels, steel,
MPSGraph), wire XLA's existing GEMM / conv autotuner in front of them so the
per-shape choice stops being the `METAL_PJRT_GEMM` environment variable.

## float64 by double-float emulation (2026-09-25)

Metal has no double type and jax-metal rejects f64 outright, so this would be
a genuine differentiator. Because we generate the MSL, f64 can be lowered to
a (hi, lo) float2 pair with error-free transforms (two-sum, two-prod via fma)
in the emitter, with the precision loss documented: two f32 give about 48
mantissa bits (~1e-14 relative, not IEEE double's 53) and keep f32's
exponent range (overflow near 3e38, denormals below 1e-38). Transcendentals need
their own double-float implementations; start with add / sub / mul / div /
sqrt / fma / compare / convert and reduce, which cover most of what people
want f64 for (accumulators, orbital mechanics, GP kernels). Reject anything
else at compile time rather than silently narrowing. Storage stays 8 bytes
per element so host transfers are plain memcpy.
