# XLA op coverage audit (XLA @ 91888df6)

For users: the README's support table is the short version. JAX's own
`lax_test.py` passes 996 cases on mtl; its 13 known failures
(`scripts/jax_known_failures/lax_test.txt`, run with
`JAX_NUM_GENERATED_CASES=3` and x64 off) are int4, dot precision
algorithms and JAX's `dce_sink` test handler (an FFI handler not
registered for Metal).

How each thing XLA:GPU can emit reaches the Metal backend, and whether it
works. "Verified" means observed in `tests/test_lax.py` (against a float64
CPU reference, with tolerances in ulps; int4 is an expected failure). File
references are into the XLA tree. Missing pieces worth building (int8
GEMM) are in `docs/roadmap.md`.

## Execution paths the backend implements

1. **MLIR emitter kernels** (loop, reduction, transpose, concatenate, scatter,
   in-place dynamic-update-slice) translated to MSL. Everything the elemental
   MLIR emitter handles (`codegen/emitters/elemental_hlo_to_mlir.cc:938-1205`)
   works, except f64 / complex128 and sub-byte integers. complex64
   values are carried as MSL `float2` (the same 8 bytes, real part
   first).
   Kernels with more than 31 buffer arguments (Metal's argument-table limit)
   take them through an argument buffer of GPU addresses (up to 512).
2. **GEMM** via the BlasLt thunk backed by Metal Performance Shaders (f32)
   and steel MSL kernels (f16/bf16). On an
   OneAPI-reporting device every dot the rewriter accepts becomes
   `__cublas$lt$matmul` (`gemm_rewriter.cc:2318`).
3. **Generic runtime thunks**: copies, memset, while, conditional, call,
   sequential, host transfers, events.

Anything else either hits a legacy LLVM-IR emitter (error: "function bodies
but no embedded MSL"), a library plugin we don't provide, or the thunk
emitter's default case ("Unsupported instruction opcode").

## Table

| Op or target | Path in XLA:GPU | Metal | Notes |
|---|---|---|---|
| elementwise, broadcast, reshape, transpose, slice, concat, iota, pad, reverse, map, clamp, select, convert, bitcast-convert, reduce-precision, compare | loop / transpose / concat MLIR emitters | OK | verified, incl. complex64; f64 excluded |
| reduce | reduction MLIR emitter | OK | verified incl. 1M and column reductions |
| reduce-window, cumulative ops | elemental MLIR (+ scan rewriters); cumsum/cumprod/cummax/cummin over the minor dim -> `MetalScanRewriter` -> `metal$scan` | OK | verified incl. reverse, f32/f16/bf16/s32 (`tests/test_scan.py`) |
| softmax / log-softmax over the minor dim | XLA's reduction + loop fusions (the `metal$softmax` rewriter was removed: no end-to-end win) | OK | verified f32/f16/bf16 (`tests/test_scan.py`) |
| gather, dynamic-slice | elemental MLIR | OK | verified |
| dynamic-update-slice | in-place DUS emitter or loop | OK | verified |
| scatter | scatter MLIR emitter | OK | verified for f32; uses atomics, Metal has 32-bit atomics only: a 64-bit combining scatter, or any complex64 scatter, without `unique_indices` is refused at compile time (`CheckPostGemmRewriter`); overwrite and `unique_indices` s64 scatters run |
| select-and-scatter | expander to scatter + reduce-window | OK | verified (maxpool grad) |
| stochastic-convert, logistic, batch-norm | HLO expanders | OK | expanders run in the shared pipeline |
| convolution | 1-D/2-D f32/f16/bf16 ungrouped, >= 4 Mflop: `MetalConvRewriter` -> `metal$conv` FFI (MLX's steel kernels, `metal_pjrt/conv`; forward, input and weight gradients). Else FusionWrapper, loop emitter, naive `EmitDotLoop`. A complex64 convolution is three real ones first (JAX's cpu/gpu rule, registered for mtl in `_lowerings.py`) | OK | vs a float64 CPU reference in f32/f16/bf16, forward and both gradients, NHWC/NCHW/1-D, strides, dilations, padding (`tests/test_conv.py`, `conv:conv_test`); `METAL_PJRT_DISABLE_REWRITES=conv` sends every convolution to the loop emitter (slow: the cnn fwd+bwd step 5.5 vs 1.6 ms) |
| dot, f16/bf16/f32 | BlasLt thunk over MPS (f32) / steel (f16/bf16; 2..8 rows or columns of x W^T with K >= 512 and a multiple of 4: MLX's wide gemv, `blas:gemv`); XLA's DotMerger is off (`ApplyMetalDefaults`: it would copy the weights of dots sharing an input into one operand on every call) | OK | verified incl. batched and few-row shapes (`tests/test_steel_gemm.py`, `blas:gemv_test`). For every type, a GEMM whose batch, row or column group has more than INT32_MAX elements is refused at compile time (XLA's matmul config counts them in 32 bits; `CheckGemmGroupsFitInt32`). f16/bf16 GEMMs past steel's 32-bit index limits (a row of more than 8388607 elements, a dimension or batch count over INT32_MAX) are refused at compile time too. GEMMs mixing types (e.g. f16 x f16 -> bf16) are refused ("both operands must be f32, f16 or bf16 of one type, the result that type or f32"). Integer dots GemmRewriter turns into GEMMs (s8 x s8 -> s32, at every size) are refused at compile time ("Metal: matmul s8 x s8 -> s32 is not supported"); other integer dots (e.g. s32, or s8 -> s8) stay kDot and run in elemental loops. Open decision: size-capped elemental fallback (watchdog risk for large K) vs an int8 steel GEMM; int8 via f32 GEMM is exact only for K < ~1040 |
| dot, f64 / s8 to s32 | rewriter still emits BlasLt | NO | `CheckPostGemmRewriter` refuses the GEMM at compile time, naming the op (s8 x s8 -> s32 is a GEMM at every size); f64 GEMMs are refused as f64 arithmetic by the same check |
| dot, c64 | `MetalComplexDotExpander` (`compiler/passes/complex_dot.h`; start of `RunHloPasses`, and again after `TriangularSolveExpander` for the dots XLA's complex cholesky / triangular_solve / qr expanders make) turns each complex64 dot into four real f32 dots (Ar.Br - Ai.Bi, Ar.Bi + Ai.Br; not Gauss's three, for accuracy parity with cuBLAS and CPU) with the same dimension numbers and precision config, which then take the f32 GEMM or loop-emitter paths | OK | against CPU: small and GEMM-sized, matvec, batched einsum, conj / transpose, HIGHEST precision, real @ complex, gradients, complex conv (`tests/test_lax.py` "c64 ..." cases, 0.6-4.3 ulps normwise, the same as CPU float32). complex64 cholesky, triangular_solve, qr (`tests/test_lax.py`, 2.4 ulps normwise, CPU 3.9) and ragged_dot (a probe) run through XLA's expanders; LU (solve, inv, det) is refused: its pivoting scatter on complex needs 64-bit atomics. c128 is refused with the f64 arithmetic (`CheckPostGemmRewriter`) |
| dot with fused epilogue (bias, relu, gelu, matrix bias) | rewriter fuses on OneAPI (`gemm_rewriter.cc:1806-2169`) | OK | applied in steel's store (bias, relu / tanh-gelu / silu, aux; f16/bf16); f32: MPS GEMM + one MSL epilogue kernel; verified by `tests/test_epilogue.py` and `blas:metal_blas_lt_test` |
| ragged-dot, scaled-dot | rewriters to dense dots | OK / untested for fp8 | fp8 GEMMs are untested; `CheckBlasLtTypes` refuses any GEMM that is not f32, f16 or bf16 |
| dot precision algorithms `TF32_*`, `F16_F16_F16`, `BF16_BF16_BF16` | XLA's algorithm check | NO | XLA: "Unsupported algorithm on the current device(s)"; the `BF16_BF16_F32` family runs (`test_small_dot_bf16_algorithm`) |
| sort, argsort, top_k, searchsorted, unique | Simple comparators on more than 16384 elements: XLA's SortRewriter -> `xla.gpu.ext.cub_sort_{keys,pairs}` FFI, an MSL LSD radix sort (`metal_pjrt/ffi/cub_sort_ffi.cc`, `radix_sort.h`). Everything else, and rows of <= 64 (pre-expanded in `RunHloPasses`; top_k with such rows is decomposed to a sort there first, since XLA would otherwise make it a sort only after that point): `MetalSortExpander` (`metal_pjrt/compiler/passes`), a bitonic network (straight-line up to 64 per row, else a while loop of gather + elementwise compare-and-swap); TopK decomposes back to sort on OneAPI | OK | bit-identical to CPU incl. 16M elements, batched, +-0/NaN and stability (`tests/test_sort.py`, `ffi:radix_sort_test`); `METAL_PJRT_DISABLE_REWRITES=cubsort` sends every sort to the bitonic network; complex64 keys and values (lexicographic, real then imaginary; SortRewriter's radix sort takes no complex comparator) go to `MetalSortExpander`, bit-identical to CPU incl. ties, NaN / inf, `argsort` and `sort_key_val` (`tests/test_lax.py` "c64 sort ...") |
| rng-bit-generator | Philox/ThreeFry expander | OK | verified via jax.random |
| rng (HLO kRng) | RngExpander emits rng-get-and-update-state, legacy IR | NO | `jax.lax.rng_uniform` emits it; refused naming the op (`CheckPostGemmRewriter`). `jax.random` does not use it |
| cholesky | `MetalLinalgRewriter` -> `metal$cholesky` FFI (Accelerate `spotrf`, f32); CholeskyExpander otherwise | OK | host LAPACK on the unified-memory buffers after a stream sync; `METAL_PJRT_DISABLE_LAPACK=1` restores the expander. `tests/test_linalg.py` |
| triangular-solve | `MetalLinalgRewriter` -> `metal$triangular_solve` FFI (Accelerate `cblas_strsm`, f32); `TriangularSolveExpander` otherwise | OK | all side/uplo/transpose/unit-diagonal variants, batched, checked vs CPU |
| lu / geqrf / householder_product / eigh / svd | JAX lowerings in `metal_pjrt_plugin/_linalg_lowerings.py` -> `metal$lapack_{getrf,geqrf,orgqr,syevd,gesdd}` FFI (f32) | OK | column-major operand/result layouts requested from XLA (like jaxlib CPU); other dtypes/options fall back to the pure-JAX / Qr / Eigh expander paths |
| fft | `metal_pjrt_plugin/_lowerings.py` lowers `fft` to one `metal$fft` FFI call per axis (MLX's FFT kernels, `metal_pjrt/fft`: Stockham, Rader, Bluestein and four-step plans, complex64 / float32; lengths above the kernels' limits (2^24 for powers of two, else 2^23 - 1) raise NotImplementedError), or, for complex128, a symbolic batch (`jax.export`) and `METAL_PJRT_DISABLE_FFT=1`, to a dense DFT (real matmuls against in-graph twiddles, O(n^2) per axis, n <= 46340). The HLO `fft` op itself is refused (`CheckPostGemmRewriter`, after the simplifier has folded the platform conditional of a multi-platform export): XLA's FftThunk is cuFFT-only | OK | against numpy per plan, multi-dimensional, vmap, non-last axes, gradients (`tests/test_fft.py`); each plan against a double reference in `fft:fft_test` |
| eig, schur, hessenberg, tridiagonal, geqp3 | no lowering registered for mtl (none is platform-independent in JAX; TPU lacks them too) | NO | JAX: "MLIR translation rule for primitive 'eig' not found for platform mtl" |
| cuDNN conv / norm / attention | DNN thunks | not produced | XLA's conv rewriter is not run (`MetalConvRewriter` targets an FFI handler instead) |
| Triton fusions | Triton | not produced | gated to CUDA/ROCm |
| custom fusions (CUTLASS), PTX custom kernels | custom kernel thunks | NO | not produced |
| PadToStatic / SliceToDynamic | legacy LLVM IR | NO | only with dynamic shapes |
| copy, slice-copy fusions | D2D memcpy thunks | OK | |
| while, conditional, call, tuple, GTE, bitcast, parameter | control-flow / no thunk | OK | verified |
| constants | constants module, resolved at load | OK | verified |
| replica-id, partition-id, rng seed | small host thunks | OK | |
| all-reduce, all-gather, reduce-scatter, all-to-all, collective-permute, broadcast | degenerate on one device: D2D copies | OK single device | ragged-all-to-all is never degenerate: NO |
| send/recv (device) | collective P2P | NO | |
| host send/recv, infeed/outfeed, host-execute | host transfer thunks | NO | need PjRt callbacks and SE infeed/outfeed |
| copy-start/done | async copy thunks | OK | memcpy + events |
| FFI custom calls | CustomCallThunk, handler looked up for platform "METAL" (canonical "metal") | OK for handlers in the plugin | `metal_pjrt/ffi` (`metal$scan`, `metal$conv`, `metal$fft`, the radix sort, the Python callback handler) and `metal_pjrt/linalg`; `tests/test_scan.py` calls `metal$scan` through `jax.ffi.ffi_call`; jaxlib's GPU handlers are cuda/rocm only and live in another binary; XLA's assert/debug-print intrinsics register under "cuda" |

## Element types

Nothing demotes f64 or complex. `CheckPostGemmRewriter`
(`compiler/passes/hlo_checks.h`, end of post-layout optimization) refuses
f64 arithmetic and scatters that would need 64-bit atomics (a combiner on
64-bit elements without `unique_indices`) with an error naming the JAX op
and source line, e.g. "Metal: scatter with a combiner on 64-bit elements
needs 64-bit atomics, which Metal does not have: scatter-add.5
(jit(f)/scatter-add) at f.py:7" (for complex elements: "Metal: scatter of
complex values needs 64-bit atomics ..."). f64 and complex128 data movement that needs a GPU kernel
(transpose, broadcast, concatenate, gather, pad, reverse, select, iota) is
refused the same way; contiguous slices, reshapes and dynamic slices and
updates are copies and run. Other f64 kernels (a strided or column slice) still
fail in the MSL emitter ("MSL emitter: unsupported f64 type (Metal has no
double precision) in op 'llvm.load' ..."). The check runs late on
purpose: an f32 -> f64 -> f32 chain is simplified away and runs. complex128
counts as f64 there. complex64 works: the MSL emitter carries
`complex<f32>` (and the `!llvm.struct<(f32, f32)>` LowerTensors stores it
as) as `float2`, in buffers, constants, shared memory, loop-carried values
and helper calls, with the arithmetic already expanded to f32 by
ConvertComplexToStandard (`tests/test_lax.py` "c64 ..." cases; JAX's
`lax_test.py` complex cases pass); complex dots become real dots
(`MetalComplexDotExpander`) and complex sorts run on the bitonic network.
Refused, naming the op: complex scatters without `unique_indices` (XLA
compare-and-swaps complex elements in 64 bits, even to overwrite). bf16 and f8 conversions
are expanded to integer math by XLA and round exactly (0 ulps against CPU).
The mismatch seen earlier came from the test converting back to f32 inside
the same jit: XLA's GPU pipeline removes f32 -> bf16/f16 -> f32 pairs
(`SimplifyFPConversions` under `xla_allow_excess_precision`, on by default,
as on CUDA), so the value never rounds. Sub-byte integers (s4/u4) are
rejected by the emitter ("MSL emitter: unsupported sub-byte / odd-width
integer type in op 'arith.trunci' ..."). `docs/troubleshooting.md` lists
these messages for users.

A 16-bit -> f32 dot (`preferred_element_type`) small enough that XLA keeps it
as a kDot (e.g. 4x3 @ 3x6) used to round every product to the input type:
XLA's elemental `EmitMulAdd` multiplies in the operand type before
converting to the f32 accumulator (`elemental_hlo_to_mlir.cc:494-501`;
relative error bf16 1.4e-2, f16 6.7e-4). `MetalDotOperandUpcaster`
(`compiler/passes/dot_upcast.cc`) now upcasts the operands of every dot
GemmRewriter left behind and fuses the converts with the dot, so these
match the GEMM path; JAX's `testDotPreferredElement2` passes. The same
holds for the bf16 x bf16 -> f32 dots that XLA's DotAlgorithmRewriter makes
of `ALG_DOT_BF16_BF16_F32` and its `_X3`/`_X6`/`_X9` variants (a small
`BF16_BF16_F32_X6` dot had relative error 2.5e-3). It runs after
GemmRewriter, so large 16-bit dots still reach the steel GEMM with 16-bit
operands (`tests/test_steel_gemm.py`).
