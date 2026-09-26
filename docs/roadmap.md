# Roadmap after the 2026-09-26 design review

A multi-agent design review (architecture, codegen, runtime, GEMM/linalg/FFI,
Python/testing) plus follow-up discussion. The core bet (Metal as a fourth
XLA:GPU platform, MLIR emitters -> EmitC -> MSL) holds up; the work below is
at the edges. Items already done in parallel (1 ms Synchronize sleep, commit
storm, encoder switches, small linalg on the GPU, command buffers removed with
patch 0005 and the visibility override) are not repeated here.

Guiding rules: mirror what XLA does on CUDA; measure before building; delete
what does not earn its keep; no new upstream patches unless unavoidable.

## Phase 0: trust (silent wrong results first)

1. **Compilation cache key.** JAX keys on the PJRT platform version, which for
   a OneAPI capability is `"oneapi " + runtime_version`
   (`se_gpu_pjrt_client.cc:262-277`); we set `runtime_version` to 0.0.0
   (`metal_executor.cc:381`), so the key is the constant `"oneapi 2026.0"` and
   a rebuilt plugin loads executables compiled by the old one. Encode a build
   ID plus a fingerprint of the compile-time `METAL_PJRT_*` settings into
   `runtime_version` in C++. Then drop the Python variant-directory hashing in
   `jax_plugins/metal/__init__.py` (it also misses `JAX_COMPILATION_CACHE_DIR`).
   No XLA patch needed.
2. **GPU errors through events.** A failed command buffer force-signals its
   events and `Event::WaitOnHost` returns OK (`metal_runtime.cc:613`), so a
   faulted kernel's output is consumed as valid by anything that waits on the
   event (most of PJRT). Record a status per signaled event value and return
   it from `WaitOnHost`/`PollForStatus`; make faults sticky per device, as
   CUDA does for illegal-address errors.
3. **Narrow the kernel quarantine** (keep it; resets are a design flaw).
   Today every kernel in a timed-out command buffer gets a strike, including
   built-ins, and strikes persist across processes and boots. Never strike
   built-ins; count strikes per boot and per plugin build; attribute with
   `MTLCommandBufferDescriptor.errorOptions = EncoderExecutionStatus` so only
   faulted/pending encoders are blamed.
4. **Known bugs.**
   - Verified: 3-lane vectors map to `float3`/`half3` (16/8 bytes) but
     `ByteSize` says 12/6 (`msl_emitter.cc:379`, `:455`); latent.
   - Verified: `DoBlasGemmStridedBatched` infers an f32 output from C's size
     (`metal_blas.cc:354`); delete the path or make it an error (BlasLt is the
     live path).
   - To triage (reported by review, not verified): completion handler captures
     `this` and can outlive the `Stream` after device loss; `DebugState` reads
     other streams' containers unlocked on the error path; the host-callback
     worker can lower a force-signaled fence (`setSignaledValue` without a
     monotonic check); MPS staging buffers bypass the pool and the allocation
     guard; `device_lock.py` truncates the holder's record by opening with
     `"w"` before locking; test timeouts `os._exit` with GPU work in flight.
5. **Tests that run.** The README's `bazel test --test_tag_filters=local
   //metal_pjrt_plugin/...` matches nothing (device tests are `manual`, which
   `//...` excludes); add a `test_suite` naming them. Turn the `scripts/*_check.py`
   scripts into a pytest suite (`metal` marker, run under the device lock).
   Check in an expected-failures manifest for `run_jax_tests.sh` (946 pass /
   63 known fails) so regressions show. Tighten tolerances (ulp-based per op,
   f64 CPU reference); fix `tinygp_check.py` passing Metal whenever CPU is NaN.
6. **Hygiene.** Untrack the dylib symlink (it points at an absolute
   `bazel-bin` path). Move `metal$test_scale` out of the production FFI
   library and `dispatch_bench.cc` out of `runtime/`. Fix stale docs: README
   status line, `design.md` SPIR-V milestone and persistent-cache claim,
   `metal_pjrt_plugin/BUILD.bazel` header, `__init__.py` docstring,
   `op-coverage.md` fft/fixes list, `ideas.md` float-float precision (two f32
   give ~48 bits with f32's exponent range, not ~106 bits).

## Phase 1: measure (cheap, decides Phase 2)

1. **A/B the Metal-only rewrites** with `METAL_PJRT_DISABLE_REWRITES=softmax,scan`
   end to end: bench suite, nanoGPT train step, attention fwd+bwd. Recorded
   gains are kernel time on synthetic shapes only. Dump HLO for a train step
   to see whether the softmax matcher fires at all (it requires every
   intermediate to have no outside users).
2. **Benchmark hygiene.** Interleave backends, record commit/JAX version/knobs
   per row, add GPU-time and TFLOPS columns, add linalg cases. Re-run MLX and
   jax-mps on an idle, freshly booted machine; the last runs look degraded.

## Phase 2: custom calls in CUDA's shape

Rule: a custom call is fine where CUDA also has one (cuBLASLt, cuSolver, CUB
sort, callbacks); it costs us only where CUDA would have fused.

1. **GEMM epilogue inside the kernel.** Apply bias/activation in steel's
   `store_result` (`steel_gemm_msl.h:200-244`) instead of a second pass over D
   (`metal_blas.cc:573-622`). Route f32 GEMMs with an epilogue to steel f32,
   or skip bias fusion for MPS so XLA's loop fusion absorbs it.
2. **Sort via SortRewriter.** Re-enable `xla_gpu_enable_cub_radix_sort` and
   implement the `__cub$DeviceRadixSort{Keys,Pairs}` targets as Metal FFI
   handlers (MSL radix sort). `MetalSortExpander` stays as the fallback for
   comparators SortRewriter rejects.
3. **Delete softmax/scan rewriters** unless Phase 1.1 shows a real win. Any
   FFI kernel that survives compiles its PSO once (FFI instantiate stage)
   rather than rebuilding and hashing MSL text per execution.
4. **Linalg layer ownership.** C++ owns anything visible as HLO (cholesky,
   triangular solve); Python owns only primitives with no HLO op (lu, qr,
   eigh, svd). One kill switch.

## Phase 3: unified memory

1. **Host work on the GPU timeline.** Replace `Stream::Synchronize()` in
   LAPACK (n > 32) and Python callbacks with an rt-layer
   `EnqueueHostTask(stream, fn)`: GPU signals event value n, a host queue runs
   the task on the same buffers (via `MTLSharedEventListener`, no blocked
   thread), signals n+1, and the GPU waits on n+1. The dispatch thread never
   blocks. Needs Phase 0.2 for error propagation. Later: f64/complex/FFT
   fallbacks via Accelerate on the same path.
   Python callbacks as XLA:CPU does them: the CPU handler passes operand
   pointers straight to the callback, while jaxlib's
   `xla_ffi_python_gpu_callback` copies operands D2H, waits, and copies
   results H2D. Register our own handler for that target under platform
   "METAL" that runs as a host task and hands the callback numpy views of the
   `MTLBuffer` contents, with no copies. Check jaxlib's handler signature
   first (its source is not in the pinned XLA checkout).
2. **Transfers without a drain.** H2D/D2H: when the stream has nothing in
   flight, plain `memcpy` on the calling thread; otherwise copy to staging and
   enqueue a copy kernel. D2H (`ToLiteral`/`CopyRawToHost`) as on XLA:CPU: a
   `memcpy` from `contents()` once the buffer's ready event has fired, with no
   blit into staging. Stop at a copy; do not hand out zero-copy views (see
   "Decided against").
3. **Memory policy: a pool that follows system pressure, not a startup
   fraction.** Today: BFC grows on demand up to `min(RAM/2,
   recommendedMaxWorkingSetSize)` fixed at startup (`metal_runtime.cc:175`),
   never returns freed regions unless growth is refused (patch 0004), and a
   per-growth guard refuses when free+inactive+speculative < 512 MB. Keep
   three numbers separate:
   - *Compile-time size* (`device_memory_size`, feeds the scheduler's 80%
     memory limit, `gpu_hlo_schedule.cc:1035`): stays
     `recommendedMaxWorkingSetSize`. Stable, so compiled programs don't vary
     with machine load.
   - *Live bytes*: no fixed fraction. Admit new memory while live + cached
     stays under `recommendedMaxWorkingSetSize` (the OS's GPU working-set
     number) and the kernel's memory-pressure level is normal.
   - *Cached-free bytes*: bounded by a cache limit, marked purgeable-volatile
     once their last-use fence passes (contents of free memory don't matter,
     so the OS may take them for free), dropped entirely on a
     `DISPATCH_SOURCE_TYPE_MEMORYPRESSURE` warning.
   On a miss over the limit: drop the cache, wait for in-flight work (pending
   frees), retry, then fail with a clear RESOURCE_EXHAUSTED. Implement it
   behind `MetalExecutor::Allocate` and select XLA's `platform` allocator
   (`se_gpu_pjrt_client.cc:1368`, a pure passthrough), so no XLA patch is
   needed and 0004 goes away; keep `bfc` selectable for A/B. Size-class
   caching (exact size for large buffers, since training steps repeat
   allocation sizes) keeps the steady state free of first-touch page faults,
   which is why BFC was adopted. Reuse on a different stream must wait for
   the chunk's last-use fence. Prior art to read first: MLX's Metal allocator
   (memory and cache limits relative to the recommended working set) and
   PyTorch's MPS allocator (watermark ratios, purgeable cached buffers).
   Measure first: log live/cached bytes and pressure across a nanoGPT train
   step and tinygp, and A/B against BFC on time and peak footprint.
4. **Own the PJRT entry point.** Provide our own `GetPjrtApi` (adapted from
   `pjrt_c_api_gpu_internal.cc`) creating a `MetalPjRtClient :
   StreamExecutorGpuClient`. Removes most of patch 0001 and enables zero-copy
   import: override `BufferFromHostBufferSupportsZeroCopy` (CPU-only today,
   `common_pjrt_client.cc:132`) and `ImportForeignMemory` with
   `newBufferWithBytesNoCopy` for page-aligned host memory. First check which
   host-buffer semantics JAX requests for non-CPU `device_put`.
5. **Later, measure first:** untracked buffers + `MTLResidencySet` + concurrent
   encoder with barriers only on dependency edges. BFC packs tensors into a
   few big buffers, so tracked hazards make every dispatch a full barrier.
   Reuse `xla/runtime/execution_graph.h` for the dependency edges: it is the
   shared XLA code that XLA:CPU's thunk executor uses to relax the sequential
   schedule into a DAG from buffer-slice reads/writes, so we don't write that
   analysis ourselves.

## Phase 4: OneAPI contract without upstream patches

Reporting OneAPI is the right choice (Intel's branches are "generic non-NVIDIA
GPU"); the problem is that nothing checks it. We do not plan an
`AppleComputeCapability` patch.

1. **Tripwire test.** A Bazel test over the pinned XLA sources that hashes the
   `IsOneAPI()`/`IsIntelGpu()` branches Metal relies on (GEMM routing
   `gemm_rewriter.cc:2318`, TopK skip `topk_specializer.cc:373`,
   `ExecutableAbiVersion`, `algorithm_util.cc:283`, atomics and vectorization
   in `lower_tensors`/`vectorize_loads_stores`) and fails on any change or new
   call site. Include the `AddLoweringPasses` prefix copied into
   `msl_emitter.h:24`.
2. **Post-conditions in MetalCompiler**: after the base post-layout pipeline,
   assert every dot became `__cublas$lt$matmul` and no unspecialized TopK
   remains.
3. **Upstream 0002 and 0003** (small, uncontroversial). 0001 shrinks with 3.4;
   0004 goes with 3.3.

## Phase 5: codegen robustness

1. Emit `[[max_total_threads_per_threadgroup(N)]]` (or set it on the pipeline
   descriptor) from the launch dimensions, so a register-heavy fusion cannot
   compile and then fail at launch (`metal_kernel.cc:118`).
2. Measure cold vs warm `newLibrary` time on nanoGPT; if it matters, per-kernel
   async compile during `CompileTargetBinary` plus `MTLBinaryArchive`, keyed by
   a strong hash (today FNV-64).
3. Early HLO legality pass (f64, complex, i64 atomics) with JAX op names in
   the error, instead of MLIR op names after fusion.
4. `const`/`noalias` on read-only kernel arguments; 16-byte vector accesses for
   f16/bf16/i8 via `uint4` + `as_type`; make the inttoptr address-space default
   an error.
5. Longer term: a dedicated MLIR -> MSL printer instead of EmitC plus macro
   shims (`#define _Float16 half`).

## Phase 6: strategic, gated on measurement

1. **Triton IR -> MSL.** `KernelCompiler::CompileTritonToLlvm` already reaches
   `MetalKernelCompiler` (it delegates to `inner_` today). Add
   `SoftmaxRewriterTriton` in our own hook (`IsTritonEnabled` is false for
   OneAPI, `gpu_compiler.cc:363`) and lower reductions/elementwise tiles first
   (softmax, normalization), then `tt.dot` -> `simdgroup_matrix` (fused GEMM
   prologues/epilogues, could replace steel). Hard parts: layout conversions
   and structured control flow. Multi-week; only if Phase 1 shows fusion
   quality matters on real models.
2. Convolution path (largest workload gap: cnn fwd+bwd 5.97 vs 1.93 ms).
3. c64 in the emitter (`complex<f32>` as `float2`); f64 via Phase 3.1.
4. Distribution: wheel with the dylib as package data, macOS CI with our own
   remote cache, version-range check at init (private `jax._src` imports are
   the real pin), coexistence with Apple's `jax-metal` (same platform name),
   and whether `priority=500` (default backend without f64) is right.

## Decided against

Command buffers / ICBs (measured, removed); MPSGraph partitioning; an
associative-scan primitive (deferred); an `AppleComputeCapability` upstream
patch (replaced by Phase 4.1-4.2); LLVM IR -> AIR; SPIR-V -> SPIRV-Cross.

From XLA:CPU, despite unified memory: reporting device memory as on-CPU
(`IsMemorySpaceOnCpu`), which would enable buffer-protocol views and the CPU
copy fast paths (`common_pjrt_client.cc:3402-3415`) that assume synchronously
written memory, so readers would race in-flight GPU writes; running HLO
subgraphs through XLA:CPU on the host timeline (two compilers plus a
partitioner, same objection as MPSGraph; host fallbacks stay library calls
via 3.1); building out host offloading / `pinned_host` memory kinds (a copy
between two views of the same RAM).
