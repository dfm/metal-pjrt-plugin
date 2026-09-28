#include "metal_pjrt_plugin/compiler/metal_compiler.h"

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
#include "metal_pjrt_plugin/codegen/metal_kernel_compiler.h"
#include "metal_pjrt_plugin/codegen/msl_llvm_bridge.h"
#include "metal_pjrt_plugin/compiler/compile_settings.h"
#include "metal_pjrt_plugin/runtime/constants_container.h"
#include "metal_pjrt_plugin/stream_executor/metal_platform_id.h"
#include "metal_pjrt_plugin/compiler/passes/dot_upcast.h"
#include "metal_pjrt_plugin/compiler/passes/hlo_checks.h"
#include "metal_pjrt_plugin/compiler/passes/scan_rewriter.h"
#include "metal_pjrt_plugin/compiler/passes/sort_expander.h"
// --- begin linalg (Accelerate LAPACK) ---
#include "metal_pjrt_plugin/linalg/linalg_rewriter.h"
// --- end linalg ---
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
                         "its allocation"));
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
// takes every sort), for A/B comparisons and bisecting. LAPACK has its own
// switch, METAL_PJRT_DISABLE_LAPACK (LapackDisabled()), shared with the
// Python lowerings. All are read once (compile_settings.h).
const metal_pjrt::CompileSettings& Settings() {
  return metal_pjrt::GetCompileSettings();
}

}  // namespace

void ApplyMetalDefaults(DebugOptions& debug_options) {
  debug_options.set_xla_gpu_enable_cub_radix_sort(Settings().cub_sort);
  debug_options.set_xla_gpu_enable_triton_gemm(false);
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
  // No convolution library, so nothing to canonicalize. This hook runs after
  // RunOptimizationPasses (CholeskyExpander, TopkDecomposer, StableSortExpander
  // and SortSimplifier are done) and before layout assignment, so it is where
  // we replace ops that only have legacy or library emitters in XLA:GPU.
  HloPassPipeline pipeline("metal-expanders", compilation_stats);
  // Same default block size as CholeskyExpander (128): Cholesky of n <= 128
  // emits no triangular-solve; larger ones emit blocked solves expanded here.
  pipeline.AddPass<TriangularSolveExpander>();
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
  // --- begin linalg (Accelerate LAPACK) ---
  // kCholesky / kTriangularSolve -> metal$cholesky / metal$triangular_solve,
  // before CholeskyExpander / TriangularSolveExpander see them.
  if (!LapackDisabled()) {
    HloPassPipeline pipeline("metal-linalg");
    pipeline.AddPass<MetalLinalgRewriter>();
    TF_RETURN_IF_ERROR(pipeline.Run(module.get()).status());
  }
  // --- end linalg ---
  if (Settings().scan_rewrite) {
    HloPassPipeline pipeline("metal-pre-optimization");
    pipeline.AddPass<MetalScanRewriter>();
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
    if (DumpingEnabledForHloModule(debug_module ? debug_module->name() : "",
                                   module_config.debug_options()) &&
        debug_module) {
      DumpToFileInDirOrStdout(*debug_module, "",
                              shard_number.has_value()
                                  ? (std::to_string(*shard_number) + ".metal")
                                  : "metal",
                              *msl);
    }
    std::vector<uint8_t> bytes(msl->begin(), msl->end());
    return BackendCompileResult{/*asm_text=*/*msl, std::move(bytes)};
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
