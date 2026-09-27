# How MLX executes, and where an XLA-based backend should land relative to it

Findings from reading MLX at commit 59d600b (2026-09-17) and jax-mps at
7fd54c6 (2026-09-15). File references are into those trees.

## MLX's execution model in one paragraph

MLX is an eager-dispatch interpreter over a lazily built graph. `eval()` walks
the DAG producers-first and calls each primitive's hand-written `eval_gpu`,
which picks a kernel by size/layout heuristics and encodes one or a few Metal
dispatches. Command buffers are cut by an op and size budget. Memory comes from
a dynamic caching allocator with no planning. The only automatic fusion is
`compile()`, which fuses chains of elementwise ops and broadcasts into
generated MSL. Nothing else is fused: not reductions, not matmul epilogues,
not transposes. Every matmul, conv, attention and norm kernel is custom MSL;
MPS is never used.

## The details that matter for performance

**Graph and eval** (`mlx/array.h:490-526`, `mlx/transforms.cpp:80-345`)
- Shapes are computed eagerly at op construction; only data is lazy.
- `eval` does two full traversals (in-degree count, then a width-limited BFS
  tape) with hash-map bookkeeping per node, then encodes each node on the
  calling thread. Nodes are detached after evaluation, so the graph is
  consumed as it runs.
- Back-pressure: at most 10 command buffers in flight
  (`transforms.cpp:25, 271-286`).

**Command buffers** (`mlx/backend/metal/device.cpp`)
- One `MTLCommandQueue` per stream, one concurrent compute encoder reused
  across primitives. Hazard tracking is off; barriers are inserted manually
  by tracking prev/next input and output sets (`:345-409`).
- A command buffer is committed every 40 ops (base/Pro chips) or 50 (Max/Ultra),
  or when input element counts exceed a budget (`:511-514, 602-624`).

**Fusion** (`mlx/compile.cpp`)
- `compile()` traces, runs simplify (scalar-constant dedup, no-op removal,
  CSE) and then fuses only unary/binary/ternary/broadcast chains. Depth is
  capped at 11 and inputs at 24 (`:24-25, 861-906`). Reduce, matmul,
  transpose, reshape, slice, gather and scatter all end a region.
- The `Compiled` primitive is lowered to generated MSL text with contiguous,
  strided and large-index variants, JIT-compiled and cached by a name that
  encodes the tape (`mlx/backend/metal/compiled.cpp:16-359`).
- Even when compiled, every call re-materializes the lazy graph via
  `compile_replace` and goes through the normal eval path: per-primitive
  encoding, allocation and bookkeeping (`compile.cpp:1047-1112`). There is
  no ahead-of-time executable and no command-buffer replay.
- Matmul + bias is fused only when the user calls `addmm` explicitly. There is
  no pattern matching (`mlx/ops.cpp:5818-5945`).

**Kernels** (`mlx/backend/metal/kernels/`)
- Steel GEMM uses `simdgroup_matrix` 8x8 tiles, six precompiled tile shapes,
  split-K, and an alpha/beta epilogue via function constants
  (`steel/gemm/mma.h`, `steel_gemm_fused.metal:21-33`). On macOS 26.2+ with
  the newest GPUs it switches to MetalPerformancePrimitives `matmul2d`
  ("NAX" kernels, `matmul.cpp:206-228`, `device.cpp:947-966`).
- Conv is channels-last with Winograd, implicit GEMM and im2col paths chosen
  by channel alignment (`conv.cpp:1400-1480`).
- Fused attention exists for inference only; training and the VJP use the
  unfused fallback (`scaled_dot_product_attention.cpp:843-847, 1022-1030`).
- `fast::rms_norm`, `layer_norm`, `rope` are single hand-written kernels; they
  are MLX's only reduction-plus-elementwise fusions.
- Kernels are precompiled into a metallib with `xcrun metal` by default;
  compiled elementwise regions, custom kernels and gather/scatter are always
  JIT-compiled from source.

**Memory** (`mlx/backend/metal/allocator.cpp`)
- Every buffer is shared storage, untracked. Host pointer is `contents()`;
  there are no host/device copies at all.
- Size-bucketed cache allocator with LRU. Buffers return to the cache only when
  the command buffer that used them completes, so nothing is reused within an
  in-flight command buffer. No static planning; every `eval_gpu` mallocs.

**Per-op host overhead** (`mlx/backend/metal/eval.cpp:29-70`, `device.cpp:345-423`)
- Per node: shared_ptr allocations, an autorelease pool, hash-set inserts for
  hazard tracking, kernel-name string building plus a hashed lookup under a
  shared mutex, `setBytes` for shapes/strides, allocator mutex, and a
  completion-handler block. No argument buffers, no Metal 4 command APIs.

## How jax-mps drives MLX

- Parses StableHLO, runs StableHLO simplification and folding plus five
  pattern passes (bias-add to `addmm`, softmax, layer norm, RMS norm,
  broadcast-identity cleanup). No canonicalize, no CSE
  (`src/pjrt_plugin/stablehlo_parser.cc:80-119`, `passes/`).
- First execute records the whole entry function with `mx::compile` and
  replays it afterwards (`mlx_executable.cc:921-971`). So per call it still
  pays MLX's graph re-materialization and per-op dispatch.
- Execute blocks on `eval` by default; async dispatch is opt-in
  (`mlx_executable.cc:1018-1020`). A global mutex serializes every PJRT call.
- `while` runs as a CPU-stream primitive: full GPU sync at loop entry, and a
  sync plus host readback per iteration for dynamic loops (about 500 us each
  by their own comment). Counted loops avoid the readback and flush every 64
  iterations (`ops/control_flow.cc:181-192, 441-548`).
- Host transfers copy; no donation; f64 narrowed to f32; cholesky, eigh, qr,
  svd, triangular_solve run on the CPU.
- 83 op handlers; about 95% of JAX's test suite passes.

## Where an XLA-based backend should win

1. **Reduction and elementwise-heavy code.** XLA fuses reductions with their
   producers and consumers, does multi-output fusion, and horizontally fuses
   independent small ops (optimizer updates over all parameters become a few
   kernels). MLX fuses none of these. On a 100 GB/s M3 every unfused op is a
   full trip through memory, so layer-norm/softmax backward, losses and Adam
   steps are where the largest wins should be.
2. **Host overhead.** An XLA executable is a static thunk sequence with
   pre-assigned buffers. Execution is: walk thunks, encode dispatches, commit
   once. No graph rebuild, no allocator, no shape recomputation, no hash-map
   hazard tracking. MetalHLO attributes its remaining 1.25x gap to native MLX
   on nanoGPT to host overhead; jax-mps has strictly more host overhead than
   native MLX. Small-batch and small-model workloads are where this shows.
3. **Static memory planning.** Buffer assignment with liveness gives lower
   peak memory and zero allocator traffic. Matters on 8 to 16 GB machines.
4. **Layout assignment and algebraic simplification.** Global layout choice
   (NHWC for conv), transpose folding into dot operands, dot strength
   reduction, constant folding, CSE, while-loop simplification and invariant
   code motion. MLX has none of these at graph level.
5. **Async dispatch, donation and zero-copy transfers.** PJRT's async model
   plus unified memory means parameters update in place and host transfers
   are pointer handoffs. jax-mps copies on transfer and never donates.

## Where MLX-based approaches should stay ahead, at least initially

1. **Hand-tuned library kernels.** Steel GEMM, Winograd conv, fused inference
   attention, quantized matmuls, and the Metal 4 NAX path are years of tuning.
   Starting from MPS matrix/CNN kernels is probably close on large GEMMs
   (MetalHLO found MPS ahead of its own kernels there) but behind on
   attention and quantized inference. Vendoring MLX's kernels as a library
   behind XLA custom calls is an option; MLX is MIT-licensed and jax-mps
   shows it links fine.
2. **Per-kernel codegen quality.** MLX's generated elementwise kernels are
   near roofline (work-per-thread vectorization, dimension collapsing). A
   generic LLVM to SPIR-V to MSL path will be worse per kernel until tuned.
   Fewer kernels should outweigh this, but not by a fixed factor.
3. **Compile latency.** XLA's pipeline takes seconds, and each fused kernel
   costs about 200 ms of Metal shader compilation. A persistent kernel cache
   keyed by MSL hash is mandatory, not optional.
4. **Dynamic while loops.** XLA's while thunk also reads the predicate back
   each iteration, so this is parity, not a win. Static trip counts (scan,
   fori_loop) are fully on-device in both.

## Expected picture

Training steps dominated by normalization, losses, optimizers and small
matmuls: clear win, plausibly 2x or more over jax-mps. Large-GEMM-dominated
workloads: near parity, set by the GEMM library. Transformer inference with
fused attention and quantized weights: MLX stays ahead until library kernels
are added. The benchmarks to track are ResNet18/CIFAR training (jax-mps: 3.7x
over CPU on an M4 Air; MetalHLO: 8.7x on an M5 Pro) and nanoGPT training
(MetalHLO: 1.25x slower than native MLX). Matching native MLX on nanoGPT
is the concrete target, since the gap there is exactly the host overhead a
static executable removes.
