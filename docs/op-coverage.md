# Op coverage

The [FAQ](faq.md#what-works)'s support table is the short version. This page has the full
picture: each feature with the tests behind it, then every operation
XLA:GPU can emit and the path it takes on Metal.

JAX's own `lax_test.py` passes 996 cases on mtl. Its 13 known failures
(`scripts/jax_known_failures/lax_test.txt`, run with
`JAX_NUM_GENERATED_CASES=3` and x64 off) are int4, dot precision
algorithms, and JAX's `dce_sink` test handler (an FFI handler that isn't
registered for Metal).

## What a "no" looks like

An unsupported feature should always fail at compile time, never give a
wrong answer. Please report one that doesn't. The errors come in two
kinds:

- `Metal: ...` names the JAX operation and its source line.
- `MSL emitter: unsupported ...` comes from the kernel translator, which
  has no code for some type, and names an internal operation instead of
  yours.

[`troubleshooting.md`](troubleshooting.md#unsupported-operations-unimplemented)
lists the messages and what to do about each.

## By feature

| Feature | Works? | Details | Tests (`tests/`) |
|---|---|---|---|
| `jit`, `grad`, `vmap`, `checkpoint`, control flow (`scan`, `while_loop`, `cond`, `switch`) | yes | | `test_grad.py`, `test_lax.py`, `test_smoke.py` |
| Elementwise math, reductions, broadcasting, gather, scatter, cumulative ops | yes | Metal has only 32-bit atomics, so a scatter-add/min/max on 64-bit elements, or a complex64 scatter that does more than add or overwrite (e.g. multiply), is refused unless `unique_indices=True` | `test_lax.py`, `test_scan.py` |
| `jax.random` | yes | | `test_lax.py` |
| Matmul, f32 / f16 / bf16 | yes | f32 on Metal Performance Shaders; f16/bf16 on native "steel" kernels (ported from MLX) with bias and activation fused into the store. Few-row shapes (small-batch decode) have their own kernel | `test_steel_gemm.py`, `test_epilogue.py` |
| Matmul, int8 x int8 -> int32 and mixed types (e.g. f16 x f16 -> bf16) | no | refused at compile time. Small integer dots that XLA keeps as loops do run | `test_lax.py` ("dot int32") |
| Matmul, dot precision algorithms `TF32_TF32_F32`, `F16_F16_F16`, `BF16_BF16_BF16` | no | XLA refuses them ("Unsupported algorithm on the current device(s)"). The `BF16_BF16_F32` family works | `test_lax.py` |
| Matmul, fp8 | untested | XLA turns an fp8 dot it can't hand to a library into an f16 one; fp8 conversions work | `test_lax.py` |
| Matmul and sort of complex64 | yes | a complex matmul runs as four real f32 ones; sorts are bit-identical to CPU | `test_lax.py` |
| Convolutions | yes | 1-D and 2-D f32/f16/bf16, forward and gradients, on MLX's steel convolution kernels. Grouped, 3-D, other types and tiny ones run on XLA's (slow) loop emitter. Complex64 convolutions work | `test_conv.py`, `test_lax.py` |
| Sorting (`sort`, `argsort`, `top_k`, `searchsorted`) | yes | a GPU radix sort for large arrays; bit-identical to CPU | `test_sort.py`, `test_lax.py` |
| Linear algebra in f32 (`cholesky`, `solve`, `triangular_solve`, `lu`, `qr`, `eigh`, `svd`, `inv`, `det`), with gradients | yes | Accelerate's LAPACK on the host, on the shared memory (small matrices on GPU kernels). f16/bf16 linear algebra is untested | `test_linalg.py` |
| `eig`, `schur`, `hessenberg`, `tridiagonal` | no | no lowering for mtl (JAX: "MLIR translation rule for primitive 'eig' not found for platform mtl") | |
| FFT (`jnp.fft`: fft, rfft, irfft, fftn, ...) | yes | MLX's FFT kernels, complex64 / float32, with gradients and vmap. Lengths up to 2^24 for powers of two, otherwise up to 2^23 - 1; longer ones raise NotImplementedError. complex128 is refused like float64 | `test_fft.py` |
| `pure_callback`, `io_callback`, `jax.debug.print`, `jax.debug.callback` | yes | synchronous; sub-byte dtypes such as int4 are refused ([`callbacks.md`](callbacks.md)) | `test_callbacks.py` |
| `checkify` | partly, untested | functionalized checks (`checkify.checkify`) use JAX's generic path; `debug=True` checks are dropped, as on TPU | |
| Several devices (`pmap`, sharding) | no | the plugin exposes one device | |
| f32, f16, bf16, integer and bool types | yes | | `test_lax.py` |
| float64 | no | Apple GPUs have no double type. Transfers of f64 arrays work, and so do copies inside `jit` (reshapes, contiguous slices). f64 arithmetic and other f64 data movement are refused at compile time; a strided f64 slice fails in the kernel translator instead. With `jax_enable_x64` on, keep f64 work on the CPU | `test_lax.py` |
| complex64 | yes | arithmetic, math, data movement, reductions, matmul, sort, scatter-add and overwrite (and so LU: `solve`, `inv`, `det`; sparse arrays), `cholesky`, `triangular_solve`, `qr` and transfers. Not supported: other scatter combiners without unique indices (above). complex128 is refused like float64 | `test_lax.py`, `test_linalg.py` |
| int4 / uint4 | no | fail in the kernel translator | `test_lax.py` (expected failure) |
| Buffer donation (`donate_argnums`) | yes | the donated input's memory becomes the output, as on CUDA; JAX's own donation tests in `api_test.py` pass | `test_donation.py` |
| JAX's persistent compilation cache | yes, opt-in | see the [FAQ](faq.md#how-do-i-turn-on-the-compilation-cache) | `test_callbacks.py`, `test_compilation_cache.py` |

## How XLA ops reach Metal

File and line references are into the pinned XLA (`third_party/PINS.md`,
XLA @ 91888df6). "Verified" means checked in `tests/test_lax.py` against
float64 CPU. Missing pieces worth building are in
[`roadmap.md`](roadmap.md).

The backend has three execution paths:

1. **MLIR emitter kernels** (loop, reduction, transpose, concatenate,
   scatter, in-place dynamic-update-slice), translated to MSL. Everything
   the elemental emitter handles works except f64, complex128 and
   sub-byte integers. complex64 is carried as MSL `float2`. Kernels with
   more than 31 buffers (Metal's limit) get them through an argument
   buffer (up to 512).
2. **GEMM** through the BlasLt thunk: MPS for f32, steel for f16/bf16. On
   a OneAPI-reporting device every accepted dot becomes
   `__cublas$lt$matmul` (`gemm_rewriter.cc:2318`).
3. **Generic runtime thunks**: copies, memset, while, conditional, call,
   host transfers, events.

Anything else hits a legacy LLVM-IR emitter ("function bodies but no
embedded MSL"), a library we don't provide, or "Unsupported instruction
opcode".

| Op or target | Path | Metal | Notes |
|---|---|---|---|
| elementwise, broadcast, reshape, transpose, slice, concat, iota, pad, reverse, map, clamp, select, convert, bitcast-convert, reduce-precision, compare | loop / transpose / concat emitters | OK | verified, including complex64 |
| reduce | reduction emitter | OK | verified, including 1M-element and column reductions |
| reduce-window, cumulative ops | elemental; minor-dim cumsum/cumprod/cummax/cummin -> `metal$scan` | OK | `tests/test_scan.py` |
| softmax / log-softmax | XLA's reduction + loop fusions | OK | `tests/test_scan.py` |
| gather, dynamic-slice, dynamic-update-slice | elemental / in-place DUS | OK | verified |
| scatter | scatter emitter, with atomics | OK | 64-bit combining scatters need `unique_indices` (refused by `CheckPostGemmRewriter`). A 64-bit integer overwrite with repeated indices is a plain 8-byte store (XLA's own lowering); no torn values observed under stress (`test_scatter_64bit_overwrite_with_repeated_indices`), though Metal doesn't document 8-byte store atomicity. complex64: an add becomes two f32 scatter-adds (`MetalComplexScatterSplitter`; about 3x the operand traffic of an in-place scatter, from the real/imag split and recombine); an overwrite is one 8-byte store where XLA, also on CUDA, would compare-and-swap (`metal_kernel_compiler.cc`), with no torn values observed under stress either (`test_complex_scatter_overwrite_with_repeated_indices`); other combiners need `unique_indices` |
| select-and-scatter | non-overlapping unpadded max-pool gradients -> `metal$pool_max_bwd` (one pass, no atomics); otherwise XLA's expander | OK | bit-identical to CPU, including ties and NaN ("max-pool grad" cases); `METAL_PJRT_DISABLE_REWRITES=pool` turns it off |
| stochastic-convert, logistic, batch-norm | HLO expanders | OK | |
| convolution | 1-D/2-D f32/f16/bf16, ungrouped, >= 4 Mflop -> `metal$conv` (`metal_pjrt/conv`); otherwise the loop emitter. complex64 becomes three real convolutions | OK | `tests/test_conv.py`, `conv:conv_test`; `METAL_PJRT_DISABLE_REWRITES=conv` sends all to the loop emitter |
| dot, f32/f16/bf16 | BlasLt over MPS or steel; 2..8 rows with K >= 512 (a multiple of 4) on MLX's wide gemv. XLA's DotMerger is off: it would copy shared weights on every call | OK | `tests/test_steel_gemm.py`, `blas:gemv_test`. Refused at compile time: groups over INT32_MAX elements (`CheckGemmGroupsFitInt32`), f16/bf16 past steel's 32-bit index limits, mixed types. Open decision: an int8 GEMM ([`roadmap.md`](roadmap.md)) |
| dot, f64, s8 x s8 -> s32 | BlasLt | NO | refused at compile time, naming the op. Other integer dots (s32, s8 -> s8) stay loops and run |
| dot, c64 | `MetalComplexDotExpander`: four real f32 dots (not Gauss's three, for parity with cuBLAS and CPU), also for the dots of XLA's complex cholesky / triangular_solve / qr expanders | OK | 0.6-4.3 ulps normwise, as CPU |
| dot with fused epilogue (bias, relu, gelu, matrix bias) | fused by the rewriter on OneAPI | OK | applied in steel's store; f32 is MPS plus one epilogue kernel. `tests/test_epilogue.py` |
| ragged-dot, scaled-dot | rewriters to dense dots | OK / fp8 untested | |
| dot precision algorithms `TF32_*`, `F16_F16_F16`, `BF16_BF16_BF16` | XLA's algorithm check | NO | `BF16_BF16_F32` runs |
| sort, argsort, top_k, searchsorted, unique | over 16384 elements with simple comparators: SortRewriter -> an MSL radix sort (`metal_pjrt/ffi`). Otherwise, and rows of <= 64: `MetalSortExpander`, a bitonic network | OK | bit-identical to CPU, including 16M elements, +-0 / NaN and stability (`tests/test_sort.py`). complex64 keys go to the bitonic network. `METAL_PJRT_DISABLE_REWRITES=cubsort` turns off the radix sort |
| rng-bit-generator | Philox / ThreeFry expander | OK | via `jax.random` |
| rng (HLO kRng) | legacy IR | NO | `jax.lax.rng_uniform`; refused naming the op |
| mulhi, 64-bit | i128 multiply | NO | refused naming the op; 8 to 32 bits work |
| cholesky, triangular-solve | `metal$cholesky` / `metal$triangular_solve` (Accelerate, f32) | OK | host LAPACK on the shared buffers after a stream sync; `METAL_PJRT_DISABLE_LAPACK=1` restores XLA's expanders. `tests/test_linalg.py` |
| lu, geqrf, householder_product, eigh, svd | `_linalg_lowerings.py` -> `metal$lapack_*` (f32) | OK | other dtypes fall back to JAX's generic paths |
| fft | `_lowerings.py` -> one `metal$fft` call per axis (`metal_pjrt/fft`). complex128, a symbolic batch or `METAL_PJRT_DISABLE_FFT=1` take a dense DFT (n <= 46340). The HLO `fft` op itself is refused: XLA's is cuFFT-only | OK | `tests/test_fft.py`, `fft:fft_test` |
| eig, schur, hessenberg, tridiagonal, geqp3 | no lowering for mtl (TPU lacks them too) | NO | |
| cuDNN, Triton, CUTLASS, PTX custom kernels | | not produced | |
| PadToStatic / SliceToDynamic | legacy IR | NO | dynamic shapes only |
| copies, control flow, constants, replica-id, partition-id | memcpy / control-flow / host thunks | OK | |
| collectives (all-reduce, all-gather, ...) | degenerate copies on one device | OK, one device | ragged-all-to-all: NO |
| send / recv, infeed / outfeed, host-execute | | NO | |
| copy-start / done | memcpy + events | OK | |
| FFI custom calls | handlers for platform "METAL" | the plugin's own | jaxlib's GPU handlers are cuda/rocm only |

## Element types

`CheckPostGemmRewriter` (`compiler/passes/hlo_checks.h`) refuses what
Metal can't do, naming the JAX op and source line, for example:

    Metal: scatter with a combiner on 64-bit elements needs 64-bit atomics,
    which Metal does not have: scatter-add.5 (jit(f)/scatter-add) at f.py:7

- **f64 and complex128**: arithmetic, and data movement that needs a
  kernel (transpose, broadcast, concatenate, gather, pad, reverse, select,
  iota), are refused. Contiguous slices, reshapes and dynamic slices are
  plain copies and run. Strided f64 slices fail in the MSL emitter. The
  check runs late, so an f32 -> f64 -> f32 chain is simplified away and
  runs.
- **complex64** is carried as `float2` everywhere; the arithmetic is
  expanded to f32 before the emitter. JAX's complex `lax_test.py` cases
  pass.
- **bf16 and f8 conversions** round exactly. XLA removes
  f32 -> bf16/f16 -> f32 pairs inside one `jit` (`xla_allow_excess_precision`,
  as on CUDA), so such a round trip never rounds.
- **Sub-byte integers** (s4/u4) fail in the MSL emitter.
- **Small 16-bit -> f32 dots** that XLA keeps as loops used to round every
  product to the input type. `MetalDotOperandUpcaster` upcasts their
  operands, so they match the GEMM path (JAX's `testDotPreferredElement2`
  passes), including the dots of `ALG_DOT_BF16_BF16_F32` and its
  `_X3`/`_X6`/`_X9` variants.
