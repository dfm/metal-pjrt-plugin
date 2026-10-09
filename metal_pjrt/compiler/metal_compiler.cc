// Copyright 2026 The metal-pjrt-plugin Authors
// SPDX-License-Identifier: Apache-2.0

#include "metal_pjrt/compiler/metal_compiler.h"

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string_view>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/numbers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/statusor.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"
#include "metal_pjrt/codegen/metal_kernel_compiler.h"
#include "metal_pjrt/codegen/msl_kernel.h"
#include "metal_pjrt/codegen/msl_llvm_bridge.h"
#include "metal_pjrt/compiler/compile_settings.h"
#include "metal_pjrt/compiler/report_bug.h"
#include "metal_pjrt/runtime/constants_container.h"
#include "metal_pjrt/stream_executor/metal_platform_id.h"
#include "metal_pjrt/compiler/passes/complex_dot.h"
#include "metal_pjrt/compiler/passes/complex_scatter.h"
#include "metal_pjrt/compiler/passes/dot_upcast.h"
#include "metal_pjrt/compiler/passes/hlo_checks.h"
#include "metal_pjrt/compiler/passes/conv_rewriter.h"
#include "metal_pjrt/compiler/passes/pool_rewriter.h"
#include "metal_pjrt/compiler/passes/scan_rewriter.h"
#include "metal_pjrt/compiler/passes/sort_expander.h"
// --- begin linalg (Accelerate LAPACK) ---
#include "metal_pjrt/linalg/linalg_rewriter.h"
// --- end linalg ---
#include "xla/hlo/transforms/simplifiers/convolution_group_converter.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/hlo/transforms/simplifiers/tuple_simplifier.h"
#include "xla/service/call_inliner.h"
#include "xla/service/dump.h"
#include "xla/service/triangular_solve_expander.h"
#include "xla/service/topk_rewriter.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/service/gpu/target_constants.h"

namespace xla {
namespace gpu {

namespace {

// XLA emits constants as global variables with array initializers. Turn each
// one into a named byte blob for the constants container.
absl::StatusOr<std::vector<uint8_t>> SerializeConstantsModule(
    llvm::Module* llvm_module) {
  const llvm::DataLayout& dl = llvm_module->getDataLayout();
  std::vector<metal_pjrt::rt::ConstantBlob> blobs;
  for (llvm::GlobalVariable& gv : llvm_module->globals()) {
    if (!gv.hasInitializer()) continue;
    llvm::Constant* init = gv.getInitializer();
    uint64_t size = dl.getTypeAllocSize(init->getType());
    metal_pjrt::rt::ConstantBlob blob;
    blob.name = gv.getName().str();
    blob.data.assign(size, 0);
    if (llvm::isa<llvm::ConstantAggregateZero>(init) ||
        llvm::isa<llvm::UndefValue>(init)) {
      // zeros
    } else if (auto* cds = llvm::dyn_cast<llvm::ConstantDataSequential>(init)) {
      llvm::StringRef raw = cds->getRawDataValues();
      if (raw.size() > size) {
        return absl::InternalError(
            absl::StrCat("constant ", blob.name, " initializer larger than "
                         "its allocation", metal_pjrt::kReportBug));
      }
      std::copy(raw.begin(), raw.end(), blob.data.begin());
    } else if (auto* ci = llvm::dyn_cast<llvm::ConstantInt>(init)) {
      llvm::APInt v = ci->getValue();
      uint64_t words = (size + 7) / 8;
      for (uint64_t w = 0; w < words && w < v.getNumWords(); ++w) {
        uint64_t word = v.getRawData()[w];
        for (int b = 0; b < 8 && w * 8 + b < size; ++b) {
          blob.data[w * 8 + b] = static_cast<uint8_t>(word >> (8 * b));
        }
      }
    } else if (auto* cf = llvm::dyn_cast<llvm::ConstantFP>(init)) {
      llvm::APInt bits = cf->getValueAPF().bitcastToAPInt();
      uint64_t word = bits.getZExtValue();
      for (uint64_t b = 0; b < size && b < 8; ++b) {
        blob.data[b] = static_cast<uint8_t>(word >> (8 * b));
      }
    } else {
      return absl::UnimplementedError(absl::StrCat(
          "Metal constants module: unsupported initializer kind for global ",
          blob.name));
    }
    blobs.push_back(std::move(blob));
  }
  return metal_pjrt::rt::SerializeConstants(blobs);
}

// METAL_PJRT_DISABLE_REWRITES=scan (or all) turns off the metal$scan
// rewriter, =cubsort XLA's SortRewriter (radix sort; MetalSortExpander then
// takes every sort), =conv the metal$conv rewriter (the loop emitter then
// takes every convolution), =pool the metal$pool_max_bwd rewriter (XLA's
// SelectAndScatterExpander then takes every max-pool gradient), for A/B
// comparisons and bisecting. LAPACK has its own
// switch, METAL_PJRT_DISABLE_LAPACK (LapackDisabled()), shared with the
// Python lowerings. All are read once (compile_settings.h).
const metal_pjrt::CompileSettings& Settings() {
  return metal_pjrt::GetCompileSettings();
}

}  // namespace

void ApplyMetalDefaults(DebugOptions& debug_options) {
  debug_options.set_xla_gpu_enable_cub_radix_sort(Settings().cub_sort);
  debug_options.set_xla_gpu_enable_triton_gemm(false);
  // No DotMerger: it turns dots that share an operand (x @ Wq.T, x @ Wk.T)
  // into one dot against a concatenation of the others, which copies every
  // weight on each call when the weights are parameters. 28 bf16 [8, 1024] x
  // [1024, 6144] GEMMs sharing x: 12.2 -> 4.9 ms (the copy is twice the
  // GEMMs' own traffic); bench/jax_bench.py unchanged (docs/performance.md).
  debug_options.set_xla_gpu_dot_merger_threshold_mb(0);
  // No dynamic-slice fusion: DynamicSliceFusionRewriterV2 (on by default)
  // wraps a GEMM or any FFI custom call whose operand is a slice, or whose
  // result goes into a dynamic-update-slice, and hands the call pointers
  // into the sliced buffers. For dus(x, f(slice(x, i)), i) the result
  // aliases the operand, which the handlers here are not written for:
  // x.at[:n].set(x[:n] @ r) compiled to a GEMM reading slice(x) and writing
  // in place into x (seen in the HLO with the fusion forced on). Without it
  // XLA copies the slices.
  debug_options.set_xla_gpu_enable_dynamic_slice_fusion(false);
  // No command buffers: XLA's conversion pass already clears the command
  // types for OneAPI-capability devices, and a software-replay
  // implementation was measured (docs/performance.md) to be a wash once the
  // runtime's per-launch overhead was fixed, so it was removed.
  debug_options.clear_xla_gpu_enable_command_buffer();
}

MetalCompiler::MetalCompiler()
    : GpuCompiler(stream_executor::metal::kMetalPlatformId,
                  spir::TargetTriple(), spir::DataLayout()) {}

absl::Status MetalCompiler::OptimizeHloConvolutionCanonicalization(
    HloModule* hlo_module, const se::GpuComputeCapability& gpu_version,
    se::dnn::VersionInfo dnn_version,
    const se::SemanticVersion& toolkit_version,
    CompilationStats* compilation_stats) {
  // This hook runs after RunOptimizationPasses (CholeskyExpander,
  // TopkDecomposer, StableSortExpander and SortSimplifier are done) and
  // before layout assignment, so it is where we replace ops that only have
  // legacy or library emitters in XLA:GPU: convolutions go to metal$conv
  // (MLX's steel kernels; the rest stay on the loop emitter), in place of
  // CUDA's cuDNN canonicalization.
  HloPassPipeline pipeline("metal-expanders", compilation_stats);
  // batch_group_count > 1 (the kernel gradient of a grouped or depthwise
  // convolution, in JAX) becomes an ordinary convolution with one more
  // spatial dimension, as on XLA:CPU (cpu_compiler.cc). On CUDA cuDNN takes
  // these; XLA's loop emitter sums over every batch group for each output
  // feature instead of the feature's own (indexing_analysis.cc,
  // ComputeOutputToInputConvolutionOpIndexing), so none may reach it.
  // CheckPostGemmRewriter refuses any that does.
  pipeline.AddPass<ConvolutionGroupConverter>(
      /*should_expand=*/[](HloInstruction*) { return true; },
      /*is_cost_viable=*/[](HloInstruction*) { return false; },
      /*convert_batch_groups_only=*/true);
  if (Settings().conv_rewrite) {
    pipeline.AddPass<MetalConvRewriter>();
  }
  // Same default block size as CholeskyExpander (128): Cholesky of n <= 128
  // emits no triangular-solve; larger ones emit blocked solves expanded here.
  pipeline.AddPass<TriangularSolveExpander>();
  // Complex dots XLA made after RunHloPasses' expansion: CholeskyExpander,
  // QrExpander, RaggedDotRewriter and TriangularSolveExpander above.
  pipeline.AddPass<MetalComplexDotExpander>();
  // complex64 scatter-adds with possibly repeated indices: two f32 ones
  // (Metal has no 64-bit atomics for XLA's complex compare-and-swap).
  pipeline.AddPass<MetalComplexScatterSplitter>();
  pipeline.AddPass<MetalSortExpander>();
  pipeline.AddPass<CallInliner>();
  pipeline.AddPass<TupleSimplifier>();
  pipeline.AddPass<HloDCE>();
  return pipeline.Run(hlo_module).status();
}

absl::StatusOr<std::unique_ptr<HloModule>> MetalCompiler::RunHloPasses(
    std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
    const CompileOptions& options) {
  ApplyMetalDefaults(module->mutable_config().mutable_debug_options());
  TF_RETURN_IF_ERROR(CheckBeforeOptimization(*module));
  {
    // Complex64 dots become real f32 dots (MetalBlasLt has no complex GEMM),
    // before the simplifier and GemmRewriter see them.
    HloPassPipeline pipeline("metal-complex-dots");
    pipeline.AddPass<MetalComplexDotExpander>();
    TF_RETURN_IF_ERROR(pipeline.Run(module.get()).status());
  }
  // --- begin linalg (Accelerate LAPACK) ---
  // kCholesky / kTriangularSolve -> metal$cholesky / metal$triangular_solve,
  // before CholeskyExpander / TriangularSolveExpander see them.
  if (!LapackDisabled()) {
    HloPassPipeline pipeline("metal-linalg");
    pipeline.AddPass<MetalLinalgRewriter>();
    TF_RETURN_IF_ERROR(pipeline.Run(module.get()).status());
  }
  // --- end linalg ---
  if (Settings().scan_rewrite || Settings().pool_rewrite) {
    // Before GpuCompiler's ReduceWindowRewriter / SelectAndScatterExpander.
    HloPassPipeline pipeline("metal-pre-optimization");
    if (Settings().scan_rewrite) pipeline.AddPass<MetalScanRewriter>();
    if (Settings().pool_rewrite) pipeline.AddPass<MetalPoolMaxBwdRewriter>();
    TF_RETURN_IF_ERROR(pipeline.Run(module.get()).status());
  }
  // Every sort ends up stable whatever its is_stable flag: SortRewriter's
  // radix sort is stable, and MetalSortExpander breaks ties on the original
  // index. On a key-only sort the flag only makes StableSortExpander add an
  // iota operand (and tie-break) that the bitonic network then carries next
  // to its own index.
  for (HloComputation* computation : module->computations()) {
    for (HloInstruction* instr : computation->instructions()) {
      if (instr->opcode() == HloOpcode::kSort && instr->operand_count() == 1) {
        Cast<HloSortInstruction>(instr)->set_is_stable(false);
      }
    }
  }
  if (Settings().cub_sort) {
    // SortRewriter (enabled in ApplyMetalDefaults) takes every simple sort of
    // more than 16384 elements (sort_rewriter.cc, non-CUDA default), however
    // short its rows. Rows of <= 64 stay with the straight-line bitonic
    // network: it measured 3-180x faster there than the radix sort's one
    // threadgroup per row (docs/performance.md). Smaller sorts are left
    // alone here and expanded later, as before.
    // top_k is still a kTopK here; XLA turns it into a sort later (TopkDecomposer
    // and TopkRewriter in the pre-SPMD pipeline), which SortRewriter would then
    // take. Decompose the same short-row ones now so they stay bitonic too
    // (radix measured ~40x slower on top_k of f32[4096,8]).
    HloPassPipeline pipeline("metal-small-sorts");
    constexpr int64_t kCubSortMinElements = 16384;  // sort_rewriter.cc
    pipeline.AddPass<TopkDecomposer>([](const HloInstruction* instr) {
      if (instr->opcode() != HloOpcode::kTopK) return false;
      const Shape& shape = instr->operand(0)->shape();
      return shape.dimensions(shape.dimensions().size() - 1) <=
                 kMaxUnrolledSortDim &&
             ShapeUtil::ElementsIn(shape) > kCubSortMinElements;
    });
    pipeline.AddPass<MetalSortExpander>(kMaxUnrolledSortDim,
                                        kCubSortMinElements);
    TF_RETURN_IF_ERROR(pipeline.Run(module.get()).status());
  }
  return GpuCompiler::RunHloPasses(std::move(module), stream_exec, options);
}

absl::Status MetalCompiler::OptimizeHloPostLayoutAssignment(
    HloModule* hlo_module, se::StreamExecutor* stream_exec,
    const CompileOptions& options, const GpuTopology& gpu_topology,
    const GpuAliasInfo* alias_info, tsl::thread::ThreadPool* thread_pool,
    CompilationStats* compilation_stats, mlir::MLIRContext* mlir_context) {
  TF_RETURN_IF_ERROR(GpuCompiler::OptimizeHloPostLayoutAssignment(
      hlo_module, stream_exec, options, gpu_topology, alias_info, thread_pool,
      compilation_stats, mlir_context));
  // After GemmRewriter, so only the dots left for the loop emitter change.
  HloPassPipeline pipeline("metal-post-gemm", compilation_stats);
  pipeline.AddPass<MetalDotOperandUpcaster>();
  TF_RETURN_IF_ERROR(pipeline.Run(hlo_module).status());
  return CheckPostGemmRewriter(*hlo_module);
}

void MetalCompiler::AddPaddingForGpublasGemms(
    HloPassPipeline& pipeline, const DebugOptions& debug_options,
    const se::GpuComputeCapability& gpu_version) {}

absl::Status MetalCompiler::AddConfigAssignerPass(
    HloPassPipeline* pipeline, HloModule* hlo_module,
    const se::GpuComputeCapability& gpu_version, const CompileOptions& options,
    tsl::thread::ThreadPool* thread_pool,
    stream_executor::StreamExecutor* stream_executor,
    const GpuTargetConfig* target_config, const AliasInfo* alias_info,
    mlir::MLIRContext* mlir_context,
    HloCostAnalysis::ShapeSizeFunction shape_size_fn,
    const MultiProcessKeyValueStore& key_value_store) {
  return absl::OkStatus();
}

std::unique_ptr<KernelCompiler> MetalCompiler::CreateKernelCompiler(
    LlvmIrCompiler llvm_compiler,
    const se::DeviceDescription& device_description,
    const DebugOptions& debug_options, tsl::thread::ThreadPool* thread_pool) {
  return std::make_unique<metal_pjrt::codegen::MetalKernelCompiler>(
      std::move(llvm_compiler), device_description, debug_options,
      thread_pool);
}

absl::StatusOr<GpuCompiler::BackendCompileResult>
MetalCompiler::CompileTargetBinary(
    const HloModuleConfig& module_config, llvm::Module* llvm_module,
    const stream_executor::DeviceDescription& device_description,
    bool relocatable, const HloModule* debug_module,
    std::optional<int> shard_number) {
  // Kernel modules arrive with their MSL already generated by
  // MetalKernelCompiler and embedded in the module. Everything else is a constants module.
  std::optional<std::string> msl =
      metal_pjrt::codegen::ExtractMslFromLlvmModule(*llvm_module);
  if (msl.has_value()) {
    // The binary keeps the prelude line (the runtime expands it); dumps and
    // asm_text (XLA only hands it to a user hook, then drops it) get the
    // prelude, so a dumped .metal file compiles on its own.
    std::string text = metal_pjrt::codegen::ExpandMslPrelude(*msl);
    if (DumpingEnabledForHloModule(debug_module ? debug_module->name() : "",
                                   module_config.debug_options()) &&
        debug_module) {
      DumpToFileInDirOrStdout(*debug_module, "",
                              shard_number.has_value()
                                  ? (std::to_string(*shard_number) + ".metal")
                                  : "metal",
                              text);
    }
    std::vector<uint8_t> bytes(msl->begin(), msl->end());
    return BackendCompileResult{/*asm_text=*/std::move(text), std::move(bytes)};
  }

  bool has_kernel = false;
  for (llvm::Function& f : *llvm_module) {
    if (!f.isDeclaration()) has_kernel = true;
  }
  if (has_kernel) {
    return absl::UnimplementedError(
        "Metal backend received an LLVM module with function bodies but no "
        "embedded MSL; this kernel was not produced by the MLIR emitters");
  }
  TF_ASSIGN_OR_RETURN(std::vector<uint8_t> constants,
                      SerializeConstantsModule(llvm_module));
  return BackendCompileResult{/*asm_text=*/"", std::move(constants)};
}

std::vector<std::string> MetalCompiler::GetLLVMCommandLineOptions(
    const DebugOptions& debug_options) const {
  return {};
}

}  // namespace gpu
}  // namespace xla
