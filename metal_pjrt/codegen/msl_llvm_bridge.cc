// Copyright 2026 The jax-graft Authors
// SPDX-License-Identifier: Apache-2.0

#include "metal_pjrt/codegen/msl_llvm_bridge.h"

#include <algorithm>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Casting.h"

namespace metal_pjrt::codegen {

absl::Status EmbedMslInLlvmModule(llvm::Module& m, const MslKernel& kernel) {
  if (kernel.kernel_name.empty()) {
    return absl::InvalidArgumentError("MSL kernel has no name");
  }
  if (kernel.num_buffer_args < 0) {
    return absl::InvalidArgumentError("negative number of buffer arguments");
  }
  llvm::LLVMContext& ctx = m.getContext();
  std::string global_name =
      absl::StrCat(kMslSourceGlobalPrefix, kernel.kernel_name);
  if (m.getNamedGlobal(global_name) != nullptr ||
      m.getFunction(kernel.kernel_name) != nullptr) {
    return absl::AlreadyExistsError(
        absl::StrCat("kernel '", kernel.kernel_name, "' already in module"));
  }

  llvm::Constant* text = llvm::ConstantDataArray::getString(
      ctx, kernel.msl_source, /*AddNull=*/true);
  auto* source = new llvm::GlobalVariable(
      m, text->getType(), /*isConstant=*/true,
      llvm::GlobalValue::ExternalLinkage, text, global_name);
  source->setAlignment(llvm::Align(1));

  // Kernel stub: XLA looks the kernel up by name and annotates it.
  llvm::Type* ptr = llvm::PointerType::get(ctx, /*AddressSpace=*/1);
  std::vector<llvm::Type*> params(kernel.num_buffer_args, ptr);
  auto* fn_type = llvm::FunctionType::get(llvm::Type::getVoidTy(ctx), params,
                                          /*isVarArg=*/false);
  llvm::Function* fn = llvm::Function::Create(
      fn_type, llvm::GlobalValue::ExternalLinkage, kernel.kernel_name, m);
  fn->setCallingConv(llvm::CallingConv::SPIR_KERNEL);
  llvm::BasicBlock* entry = llvm::BasicBlock::Create(ctx, "entry", fn);
  llvm::ReturnInst::Create(ctx, entry);

  llvm::NamedMDNode* md = m.getOrInsertNamedMetadata(kMslNamedMetadata);
  md->addOperand(llvm::MDNode::get(
      ctx, {llvm::MDString::get(ctx, kernel.kernel_name),
            llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                llvm::Type::getInt32Ty(ctx), kernel.num_buffer_args)),
            llvm::ConstantAsMetadata::get(source)}));
  return absl::OkStatus();
}

std::optional<std::string> ExtractMslFromLlvmModule(const llvm::Module& m) {
  std::vector<std::pair<std::string, std::string>> parts;
  for (const llvm::GlobalVariable& gv : m.globals()) {
    if (!gv.getName().starts_with(kMslSourceGlobalPrefix) ||
        !gv.hasInitializer()) {
      continue;
    }
    const auto* data =
        llvm::dyn_cast<llvm::ConstantDataSequential>(gv.getInitializer());
    if (data == nullptr || !data->isString()) continue;
    llvm::StringRef text = data->getAsString();
    if (text.ends_with(llvm::StringRef("\0", 1))) text = text.drop_back();
    parts.emplace_back(gv.getName().str(), text.str());
  }
  if (parts.empty()) return std::nullopt;
  std::sort(parts.begin(), parts.end());
  std::string out;
  for (const auto& [name, text] : parts) {
    if (!out.empty()) out += "\n";
    out += text;
  }
  return out;
}

std::vector<EmbeddedMslKernel> ListMslKernels(const llvm::Module& m) {
  std::vector<EmbeddedMslKernel> kernels;
  const llvm::NamedMDNode* md = m.getNamedMetadata(kMslNamedMetadata);
  if (md == nullptr) return kernels;
  for (const llvm::MDNode* node : md->operands()) {
    if (node->getNumOperands() < 2) continue;
    const auto* name = llvm::dyn_cast<llvm::MDString>(node->getOperand(0));
    const auto* nargs =
        llvm::mdconst::dyn_extract<llvm::ConstantInt>(node->getOperand(1));
    if (name == nullptr || nargs == nullptr) continue;
    kernels.push_back(EmbeddedMslKernel{
        name->getString().str(), static_cast<int>(nargs->getSExtValue())});
  }
  return kernels;
}

}  // namespace metal_pjrt::codegen
