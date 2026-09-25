# XLA op coverage audit (XLA @ 91888df6)

How each thing XLA:GPU can emit reaches the Metal backend, and whether it
works. "Verified" means observed in `scripts/lax_coverage.py` on 2026-09-23
(146/167 cases pass). File references are into the XLA tree.

## Execution paths the backend implements

1. **MLIR emitter kernels** (loop, reduction, transpose, concatenate, scatter,
   in-place dynamic-update-slice) translated to MSL. Everything the elemental
   MLIR emitter handles (`codegen/emitters/elemental_hlo_to_mlir.cc:938-1205`)
   works, except f64 and complex element types and sub-byte integers.
   Kernels with more than 31 buffer arguments (Metal's argument-table limit)
   take them through an argument buffer of GPU addresses (up to 512).
2. **GEMM** via the BlasLt thunk backed by Metal Performance Shaders. On an
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
| reduce-window, cumulative ops | elemental MLIR (+ scan rewriters) | OK | verified |
| gather, dynamic-slice | elemental MLIR | OK | verified |
| dynamic-update-slice | in-place DUS emitter or loop | OK | verified |
| scatter | scatter MLIR emitter | OK | verified for f32; uses atomics, Metal has 32-bit atomics only, so 64-bit scatter-add is unverified |
| select-and-scatter | expander to scatter + reduce-window | OK | verified (maxpool grad) |
| stochastic-convert, logistic, batch-norm | HLO expanders | OK | expanders run in the shared pipeline |
| convolution | FusionWrapper, loop emitter, naive `EmitDotLoop` | OK, slow | verified correct; no library path since conv canonicalization is a no-op |
| dot, f16/bf16/f32 | BlasLt thunk over MPS | OK | verified incl. batched, int8/int32 fell back to elemental loops |
| dot, f64 / c64 / c128 / s8 to s32 | rewriter still emits BlasLt | NO | BLAS thunk must reject; today those dtypes fail earlier |
| dot with fused epilogue (bias, relu, gelu, matrix bias) | rewriter fuses on OneAPI (`gemm_rewriter.cc:1806-2169`) | OK | MPS GEMM + one MSL epilogue kernel (bias, relu / tanh-gelu / silu, aux); verified by `scripts/epilogue_check.py` and `blas:metal_blas_lt_test` |
| ragged-dot, scaled-dot | rewriters to dense dots | OK / NO for fp8 | fp8 Lt paths unsupported |
| sort, argsort, top_k, searchsorted, unique | `MetalSortExpander` (`metal_pjrt_plugin/compiler/passes`) rewrites kSort pre-layout into a bitonic network: while loop of gather + elementwise compare-and-swap; TopK decomposes back to sort on OneAPI | OK | verified incl. 1e6 elements and batched; `ApplyMetalDefaults` sets `xla_gpu_enable_cub_radix_sort=false` so no CUB calls |
| rng-bit-generator | Philox/ThreeFry expander | OK | verified via jax.random |
| rng (HLO kRng) | RngExpander emits rng-get-and-update-state, legacy IR | NO | JAX does not emit this |
| cholesky | CholeskyExpander | OK | larger than 128 emits triangular-solve, expanded too; verified n=8, 200 |
| triangular-solve | `TriangularSolveExpander` in MetalCompiler's pre-layout hook | OK | verified incl. LU solve and n=200 Cholesky/solve |
| QR, eigh custom calls | QrExpander / EighExpander | OK at XLA level | JAX's own lowering for eigh/svd has no rule for platform "metal"; separate issue |
| fft | FftThunk (cuFFT) | NO | verified failing (complex reaches the emitter first) |
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
| other custom calls (FFI) | need a handler registered for platform "metal" | NO | jaxlib registers GPU handlers for cuda/rocm only; also XLA's own assert/debug-print intrinsics register under "cuda" |

## Element types

The emitters accept every XLA type; nothing demotes f64 or complex, so they
reach the MSL translation and are rejected there. bf16 and f8 conversions
are expanded to integer math by XLA; a rounding mismatch against CPU for
f32 to bf16 and f32 to f8e4m3 was observed and needs investigation. Sub-byte
integers (s4/u4) are rejected by the emitter.

## Fixes ranked by payoff

1. Sort as an MSL kernel (unblocks sort, argsort, top_k, searchsorted, unique).
2. `TriangularSolveExpander` in MetalCompiler (unblocks solve, LU, large Cholesky).
3. `xla_gpu_enable_cub_radix_sort=false` by default.
4. ~~BlasLt epilogues~~ (done).
5. ~~Argument buffers for kernels with more than 31 buffers~~ (done).
6. bf16/f8 conversion rounding.
7. Reject f64 and complex in the BLAS thunk explicitly.
8. FFT (would need a Metal FFT library; MPS has none for arbitrary sizes).
