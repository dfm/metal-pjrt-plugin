"""Tripwire over the pinned XLA sources the Metal plugin depends on.

The plugin relies on specific behaviour of XLA code it does not own: OneAPI
branches it inherits by reporting a OneAPI compute capability, a pass
pipeline prefix copied into codegen/msl_emitter.cc, and assumptions of its
own HLO passes. Each site below is snapshotted (whitespace-collapsed lines
around an anchor) into golden.txt next to this file. When the XLA pin moves
and a site changes, this test fails with a diff: re-check the "why" line
against the new code, adapt the plugin if needed, then regenerate:

  bazel run //metal_pjrt_plugin/xla_tripwire:xla_tripwire_test -- --update
  # or, outside Bazel:
  python3 metal_pjrt_plugin/xla_tripwire/tripwire.py \
      --xla-root "$(bazel info output_base)/external/xla+" --update

The file list in BUILD.bazel must name every file used here, except the
INVISIBLE ones below.
"""
import argparse
import difflib
import os
import re
import sys
from dataclasses import dataclass


@dataclass
class Site:
    file: str
    anchor: str  # substring of the first line of interest (first match)
    why: str
    before: int = 3
    after: int = 8
    end: str = ""  # if set, the window ends at the first line containing it


GR = "xla/backends/gpu/transforms/gemm_rewriter.cc"
GC = "xla/service/gpu/gpu_compiler.cc"
MKE = "xla/backends/gpu/codegen/emitters/mlir_kernel_emitter.cc"

SITES = [
    # --- OneAPI branches the plugin inherits ---
    Site(GR, "absl::StatusOr<absl::string_view> GetNonFp8GemmCustomCallTarget(", before=0, after=8,
         why="Every GEMM becomes __cublas$lt$matmul on OneAPI (MetalBlasLt is the only GEMM path)."),
    Site("xla/backends/gpu/transforms/topk_specializer.cc", 'custom_call_target() != "TopK"', before=1, after=3,
         why="TopkSpecializer skips OneAPI, so TopkDecomposer turns TopK into a sort (a surviving TopK custom call has no Metal handler and fails at thunk emission)."),
    Site("xla/service/algorithm_util.cc", "const bool is_sycl = gpu_compute_capability.IsOneAPI();", before=0, after=0,
         why="bf16 dot algorithms allowed on OneAPI (ALG_DOT_BF16_BF16_F32*)."),
    Site("xla/service/algorithm_util.cc", "if (!is_cuda_ge_ampere && !is_rocm_bf16 && !is_sycl)", before=1, after=2,
         why="ALG_DOT_BF16_BF16_F32 accepted on OneAPI."),
    Site("xla/service/algorithm_util.cc", "return (is_cuda_ge_ampere || is_rocm_bf16 || is_sycl) &&", before=3, after=2,
         why="ALG_DOT_BF16_BF16_F32_X3/X6/X9 accepted on OneAPI."),
    Site("xla/codegen/emitters/transforms/lower_tensors.cc", "if (device_spec_.IsIntelGpu()) {", before=3, after=3,
         why="Atomic fadd lowers to the SPIR-V form, which the MSL emitter translates."),
    Site("xla/codegen/emitters/transforms/vectorize_loads_stores.cc", "if (device_spec_.IsIntelGpu() && (IsSubByteIntOrFloatType(element_type) ||", before=0, after=3,
         why="No sub-byte vector loads on OneAPI."),
    Site("xla/codegen/emitters/transforms/vectorize_loads_stores.cc", "if (device_spec_.IsIntelGpu() && IsSubByteIntOrFloatType(element_type))", before=0, after=2,
         why="No sub-byte vector stores on OneAPI."),
    Site("xla/backends/gpu/codegen/emitters/transpose.cc", "bool use_scalar_ops = device.gpu_compute_capability().IsOneAPI() &&", before=0, after=2,
         why="Scalar shared-memory ops for sub-byte types on OneAPI."),
    Site(GC, "if (gpu_version.IsOneAPI()) {", before=1, after=2,
         why="F4E2M1FN compares upcast on OneAPI."),
    # --- PJRT / jaxlib behaviour the runtime and the compile cache rely on ---
    Site("xla/pjrt/gpu/se_gpu_pjrt_client.cc", "if (cc.oneapi_compute_capability() != nullptr) {", before=0, after=3,
         why="platform_version = 'oneapi ' + runtime_version: the persistent-cache key carries the plugin build and env (29b8494)."),
    Site("xla/pjrt/c_api_client/pjrt_c_api_client.cc", "PjRtCApiExecutable::GetAbiVersion() const {", before=0, after=7,
         why="No AbiVersion extension -> Unimplemented, which metal_pjrt_api.cc relies on by dropping the extension (e21d29d)."),
    Site("xla/pjrt/c/pjrt_c_api_gpu_internal.cc", "static PJRT_AbiVersion_Extension abi_version_extension =", before=3, after=8,
         why="The extension chain metal_pjrt_api.cc copies and filters (removes the AbiVersion node) (e21d29d)."),
    Site("xla/python/pjrt_ifrt/pjrt_executable.cc", "GetXlaExecutableVersion(", before=0, after=31,
         why="IFRT falls back to a version-less XlaExecutableVersion on Unimplemented; otherwise Metal (no TPU/CUDA/ROCm id) can't serialize, and the persistent cache breaks (e21d29d). A jaxlib bump can change this silently."),
    Site("xla/pjrt/se/stream_executor_executable.cc", "StreamExecutorExecutable::Deserialize(", before=0, after=11,
         why="Deserialize checks only the client name, so a cached Metal executable loads without an ABI check (e21d29d; the cache key is the only guard)."),
    Site("xla/pjrt/se/pjrt_stream_executor_client.cc", "auto deleted = stream_->DoHostCallback([chunk_ptr]() { delete chunk_ptr; });", before=1, after=1,
         why="Host callback without error_cb that must still run after a GPU failure (host tasks without on_error run anyway, 48ac173, sticky device error)."),
    Site("xla/pjrt/gpu/se_gpu_pjrt_client.cc", "return stream->DoHostCallback(", before=0, after=8,
         why="Host callback without error_cb that must still run after a GPU failure (48ac173, sticky device error)."),
    Site("xla/pjrt/se/buffer_sequencing_event.cc", "bool BufferSequencingEvent::IsComplete() {", before=0, after=7,
         why="IsComplete is PollForStatus() == kComplete; MetalEvent::PollForStatus returns kError for failed work (48ac173, sticky device error)."),
    Site("xla/backends/gpu/transforms/sort_rewriter.cc", "return Product(operand_shape.dimensions()) > 16384;", before=2, after=0,
         why="Non-CUDA CUB-sort threshold counts total elements; RunHloPasses pre-expands rows <= 64 above it so SortRewriter never sees them (cub_sort_ffi.cc)."),
    # --- Pipeline order the plugin's passes rely on ---
    Site(GC, "pipeline.AddPass<TopkSpecializer>(gpu_version);", before=1, after=1,
         why="TopK is specialized/decomposed in RunOptimizationPasses, before the post-layout checks."),
    Site(GC, "ABSL_RETURN_IF_ERROR(OptimizeHloConvolutionCanonicalization(", before=15, after=5,
         why="MetalCompiler's OptimizeHloConvolutionCanonicalization hook runs TriangularSolveExpander and MetalSortExpander: after RunOptimizationPasses (CholeskyExpander, TopkDecomposer, StableSortExpander, SortSimplifier) and before layout assignment."),
    Site(GC, "pre_spmd_pipeline.AddPass<TopkDecomposer>(", before=2, after=8,
         why="top_k becomes a sort in the pre-SPMD pipeline, after RunHloPasses' metal-small-sorts decomposed the short-row ones (else SortRewriter takes them; metal_compiler.cc)."),
    Site(GC, "pipeline.AddPass<PermutationSortExpander>();", before=0, after=5,
         why="SortRewriter's first call site (RunOptimizationPasses); metal-small-sorts runs before it, so rows <= 64 never reach it."),
    Site("xla/backends/gpu/transforms/sort_rewriter.cc", "bool AreOperandTypesSupportedByCub(", before=0, after=36,
         why="The key/value types SortRewriter hands to the CUB custom call: ffi/cub_sort_ffi.cc must handle a superset (cub_sort_test sweeps them)."),
    Site("xla/hlo/transforms/expanders/cholesky_expander.cc", "/*block_size=*/128,", before=2, after=1,
         why="metal_compiler.cc assumes block size 128 (Cholesky of n <= 128 emits no triangular solve) when LAPACK is disabled."),
    Site("xla/stream_executor/integrations/tf_allocator_adapter.cc", "absl::Status MemoryAllocationError(", before=0, after=16,
         why="pjrt/metal_pjrt_api.cc ErrorMessage matches 'Out of memory while trying to allocate <HumanReadableNumBytes> ' to append the runtime's refusal reason."),
    Site(GC, "// SortRewriter needs to run before StableSortExpander.", before=0, after=4,
         why="SortRewriter only with xla_gpu_enable_cub_radix_sort (ApplyMetalDefaults turns it on unless DISABLE_REWRITES=cubsort); runs before MetalSortExpander."),
    Site(GC, "// Run target-specific HLO optimization passes after layout assignment.", before=0, after=11,
         why="OptimizeHloPostLayoutAssignment runs before fusion: MetalDotOperandUpcaster and CheckPostGemmRewriter run after GemmRewriter and before priority fusion."),
    Site(GC, "void AddGemmRewriterPasses(", before=0, after=12,
         why="Bias fusion always on (no async dot); GEMM epilogues reach MetalBlasLt."),
    # --- Codegen prefix copied into msl_emitter.cc (AddMslLoweringPasses) ---
    Site(MKE, "void AddLoweringPasses(", before=0, end="createSCFToControlFlowPass",
         why="AddMslLoweringPasses (codegen/msl_emitter.cc) copies this up to SCFToControlFlow; keep them identical for the OneAPI branch."),
    # --- Assumptions of MetalDotOperandUpcaster (compiler/passes/dot_upcast.h) ---
    Site("xla/codegen/emitters/elemental_hlo_to_mlir.cc", "absl::StatusOr<Value> EmitMulAdd(", before=0, after=12,
         why="Loop-emitted dots multiply in the operand type; if this upcasts itself, the upcaster may be unnecessary."),
    Site(GR, "IsMatrixMultiplicationTooSmallForRewriting(", before=6, after=5,
         why="Small dots stay kDot (xla_gpu_gemm_rewrite_size_threshold); the upcaster targets exactly those."),
    Site("xla/backends/gpu/transforms/priority_fusion.cc", "bool IsFusible(const HloInstruction& instr) {", before=0, end="default:",
         why="kDot is not fusible, so the upcaster builds the convert+dot fusion itself."),
]

# Files Bazel can't list as data (their package's default visibility is
# private). Read from the same XLA checkout as the listed files: every pin
# change also changes listed files, so the test still reruns.
INVISIBLE = {"xla/pjrt/gpu/se_gpu_pjrt_client.cc"}

ONEAPI_RE = re.compile(r"IsOneAPI\(\)|IsIntelGpu\(\)|is_sycl")


def collapse(line):
    return " ".join(line.split())


def snapshot(site, text):
    lines = text.splitlines()
    idx = [i for i, l in enumerate(lines) if site.anchor in l]
    if not idx:
        return [f"!! anchor not found: {site.anchor}"]
    i = idx[0]
    lo = max(0, i - site.before)
    if site.end:
        hi = next((j for j in range(i, len(lines)) if site.end in lines[j]), None)
        if hi is None:
            return [f"!! end anchor not found: {site.end}"]
    else:
        hi = min(len(lines) - 1, i + site.after)
    return [c for c in (collapse(l) for l in lines[lo:hi + 1]) if c]


def render(read):
    out = []
    for s in SITES:
        out += [f"### {s.file} @ {s.anchor}", f"# why: {s.why}"]
        out += snapshot(s, read(s.file)) + [""]
    out.append("### IsOneAPI()/IsIntelGpu()/is_sycl occurrences per file")
    for f in sorted({s.file for s in SITES}):
        out.append(f"{f}: {len(ONEAPI_RE.findall(read(f)))}")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xla-root", help="XLA source root (else the Bazel data files)")
    ap.add_argument("--golden", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden.txt"))
    ap.add_argument("--update", action="store_true")
    ap.add_argument("files", nargs="*", help="XLA files (Bazel $(rootpaths))")
    a = ap.parse_args()

    by_rel = {}
    for p in a.files:
        m = re.search(r"(?:^|/)xla\+?/(xla/.*)$", p)
        if m:
            by_rel[m.group(1)] = p

    root = a.xla_root
    if root is None and by_rel:
        some_rel, some_path = next(iter(by_rel.items()))
        root = os.path.realpath(some_path)[:-len(some_rel)]

    def read(rel):
        if a.xla_root or rel in INVISIBLE:
            path = os.path.join(root, rel)
        else:
            path = by_rel.get(rel)
        if path is None:
            sys.exit(f"{rel} is not in the test's data (add it to BUILD.bazel)")
        with open(path, encoding="utf-8") as f:
            return f.read()

    got = render(read)
    golden = a.golden
    ws = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
    if a.update and ws:  # bazel run: write the source tree's golden.
        golden = os.path.join(ws, "metal_pjrt_plugin/xla_tripwire/golden.txt")
    if a.update:
        with open(golden, "w", encoding="utf-8") as f:
            f.write(got)
        print(f"wrote {golden}")
        return 0
    with open(golden, encoding="utf-8") as f:
        want = f.read()
    if got == want:
        print(f"OK: {len(SITES)} XLA sites unchanged")
        return 0
    sys.stdout.writelines(difflib.unified_diff(
        want.splitlines(True), got.splitlines(True), "golden.txt", "pinned XLA"))
    print("\nXLA sites the Metal plugin depends on changed. Re-check each changed "
          "site's 'why', adapt the plugin, then rerun with --update (see the "
          "docstring of tripwire.py).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
