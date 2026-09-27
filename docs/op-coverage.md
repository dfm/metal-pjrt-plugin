# XLA op coverage audit (XLA @ 91888df6)

How each thing XLA:GPU can emit reaches the Metal backend, and whether it
works. "Verified" means observed in `tests/test_lax.py` (against a float64
CPU reference, with tolerances in ulps; 166/167 cases pass on 2026-09-26, int4
is the known failure). File references are into the XLA tree.

## Execution paths the backend implements

1. **MLIR emitter kernels** (loop, reduction, transpose, concatenate, scatter,
   in-place dynamic-update-slice) translated to MSL. Everything the elemental
   MLIR emitter handles (`codegen/emitters/elemental_hlo_to_mlir.cc:938-1205`)
   works, except f64 and complex element types and sub-byte integers.
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
| elementwise, broadcast, reshape, transpose, slice, concat, iota, pad, reverse, map, clamp, select, convert, bitcast-convert, reduce-precision, compare | loop / transpose / concat MLIR emitters | OK | verified; f64 and complex excluded |
| reduce | reduction MLIR emitter | OK | verified incl. 1M and column reductions |
| reduce-window, cumulative ops | elemental MLIR (+ scan rewriters); cumsum/cumprod/cummax/cummin over the minor dim -> `MetalScanRewriter` -> `metal$scan` | OK | verified incl. reverse, f32/f16/bf16/s32 (`tests/test_fused_kernels.py`) |
| softmax / log-softmax over the minor dim | XLA's reduction + loop fusions (the `metal$softmax` rewriter was removed: no end-to-end win) | OK | verified f32/f16/bf16 (`tests/test_fused_kernels.py`) |
| gather, dynamic-slice | elemental MLIR | OK | verified |
| dynamic-update-slice | in-place DUS emitter or loop | OK | verified |
| scatter | scatter MLIR emitter | OK | verified for f32; uses atomics, Metal has 32-bit atomics only: a 64-bit combining scatter without `unique_indices` is refused at compile time (`CheckPostGemmRewriter`); overwrite and `unique_indices` s64 scatters run |
| select-and-scatter | expander to scatter + reduce-window | OK | verified (maxpool grad) |
| stochastic-convert, logistic, batch-norm | HLO expanders | OK | expanders run in the shared pipeline |
| convolution | FusionWrapper, loop emitter, naive `EmitDotLoop` | OK, slow | verified correct; no library path since conv canonicalization is a no-op |
| dot, f16/bf16/f32 | BlasLt thunk over MPS / steel | OK | verified incl. batched. Integer dots GemmRewriter turns into GEMMs (s8 x s8 -> s32, at every size) are refused at compile time ("Metal BlasLt: unsupported types S8 x S8 -> S32"); other integer dots (e.g. s32, or s8 -> s8) stay kDot and run in elemental loops. Open decision: size-capped elemental fallback (watchdog risk for large K) vs an int8 steel GEMM; int8 via f32 GEMM is exact only for K < ~1040 |
| dot, f64 / c64 / c128 / s8 to s32 | rewriter still emits BlasLt | NO | `CheckPostGemmRewriter` refuses the GEMM at compile time, naming the op (s8 x s8 -> s32 is a GEMM at every size); f64 GEMMs are refused as f64 arithmetic by the same check |
| dot with fused epilogue (bias, relu, gelu, matrix bias) | rewriter fuses on OneAPI (`gemm_rewriter.cc:1806-2169`) | OK | applied in steel's store (bias, relu / tanh-gelu / silu, aux; f16/bf16); f32 and `METAL_PJRT_GEMM=mps`: MPS GEMM + one MSL epilogue kernel; verified by `tests/test_epilogue.py`, `blas:metal_blas_lt_test` and `blas:metal_blas_lt_mps_test` |
| ragged-dot, scaled-dot | rewriters to dense dots | OK / NO for fp8 | fp8 Lt paths unsupported |
| sort, argsort, top_k, searchsorted, unique | `MetalSortExpander` (`metal_pjrt_plugin/compiler/passes`) rewrites kSort pre-layout into a bitonic network: while loop of gather + elementwise compare-and-swap; TopK decomposes back to sort on OneAPI | OK | verified incl. 1e6 elements and batched; `ApplyMetalDefaults` sets `xla_gpu_enable_cub_radix_sort=false` so no CUB calls |
| rng-bit-generator | Philox/ThreeFry expander | OK | verified via jax.random |
| rng (HLO kRng) | RngExpander emits rng-get-and-update-state, legacy IR | NO | JAX does not emit this |
| cholesky | `MetalLinalgRewriter` -> `metal$cholesky` FFI (Accelerate `spotrf`, f32); CholeskyExpander otherwise | OK | host LAPACK on the unified-memory buffers after a stream sync; `METAL_PJRT_DISABLE_LAPACK=1` restores the expander. `tests/test_linalg.py` |
| triangular-solve | `MetalLinalgRewriter` -> `metal$triangular_solve` FFI (Accelerate `cblas_strsm`, f32); `TriangularSolveExpander` otherwise | OK | all side/uplo/transpose/unit-diagonal variants, batched, checked vs CPU |
| lu / geqrf / householder_product / eigh / svd | JAX lowerings in `jax_plugins/openmetal/linalg_lowerings.py` -> `metal$lapack_{getrf,geqrf,orgqr,syevd,gesdd}` FFI (f32) | OK | column-major operand/result layouts requested from XLA (like jaxlib CPU); other dtypes/options fall back to the pure-JAX / Qr / Eigh expander paths |
| fft | `jax_plugins/openmetal/lowerings.py` lowers `fft` to a dense DFT (real matmuls against in-graph twiddles), so XLA's FftThunk (cuFFT) is never reached | OK, O(n^2) per axis | verified (fft, rfft2); a native FFT is still missing |
| cuDNN conv / norm / attention | DNN thunks | not produced | conv rewriter is a no-op |
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
| FFI custom calls | CustomCallThunk, handler looked up for platform "METAL" (canonical "metal") | OK for handlers in the plugin | `metal_pjrt_plugin/ffi` (`metal$scan`, the Python callback handler) and `metal_pjrt_plugin/linalg`; `tests/test_ffi.py` calls `metal$scan` through `jax.ffi.ffi_call`; jaxlib's GPU handlers are cuda/rocm only and live in another binary; XLA's assert/debug-print intrinsics register under "cuda" |

## Element types

Nothing demotes f64 or complex. `CheckPostGemmRewriter`
(`compiler/passes/hlo_checks.h`, end of post-layout optimization) refuses
f64 arithmetic and scatters that would need 64-bit atomics (a combiner on
64-bit elements without `unique_indices`) with an error naming the JAX op
and source line, e.g. "Metal: scatter with a combiner on 64-bit elements
needs 64-bit atomics, which Metal does not have: scatter-add.5
(jit(f)/scatter-add) at f.py:7". Data movement on f64 passes the check but
still fails in the MSL emitter (no f64 loads). The check runs late on
purpose: an f32 -> f64 -> f32 chain is simplified away and runs. Complex is
not checked: complex values inside a fusion lower fine (`abs(fft(x))` runs);
only complex kernel buffers fail, in the emitter
("unsupported non-trivial unrealized_conversion_cast"). bf16 and f8 conversions
are expanded to integer math by XLA and round exactly (0 ulps against CPU).
The mismatch seen earlier came from the test converting back to f32 inside
the same jit: XLA's GPU pipeline removes f32 -> bf16/f16 -> f32 pairs
(`SimplifyFPConversions` under `xla_allow_excess_precision`, on by default,
as on CUDA), so the value never rounds. Sub-byte integers (s4/u4) are
rejected by the emitter.

A 16-bit -> f32 dot (`preferred_element_type`) small enough that XLA keeps it
as a kDot (e.g. 4x3 @ 3x6) used to round every product to the input type:
XLA's elemental `EmitMulAdd` multiplies in the operand type before
converting to the f32 accumulator (`elemental_hlo_to_mlir.cc:494-501`;
relative error bf16 1.4e-2, f16 6.7e-4). `MetalDotOperandUpcaster`
(`compiler/passes/dot_upcast.cc`) now upcasts the operands of every dot
GemmRewriter left behind and fuses the converts with the dot, so these
match the GEMM path; JAX's `testDotPreferredElement2` passes. It runs after
GemmRewriter, so large 16-bit dots still reach the steel GEMM with 16-bit
operands (`tests/test_steel_gemm.py`).

## Fixes ranked by payoff

1. ~~Sort~~ (done: `MetalSortExpander` bitonic network).
2. ~~`TriangularSolveExpander` in MetalCompiler~~ (done; f32 solves and
   Cholesky go to LAPACK / small-matrix GPU kernels via `MetalLinalgRewriter`).
3. ~~`xla_gpu_enable_cub_radix_sort=false` by default~~ (done).
4. ~~BlasLt epilogues~~ (done).
5. ~~Argument buffers for kernels with more than 31 buffers~~ (done).
6. ~~bf16/f8 conversion rounding~~ (not a bug: excess precision, above).
7. ~~Reject f64 and complex in the BLAS thunk explicitly~~ (done:
   UNIMPLEMENTED from `metal_blas.cc`).
8. A native FFT (the dense-DFT lowering is O(n^2); MPS has no FFT for
   arbitrary sizes).
