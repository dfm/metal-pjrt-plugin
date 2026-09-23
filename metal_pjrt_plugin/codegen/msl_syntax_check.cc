// Standalone MSL syntax/semantic checker: compiles MSL files with the system
// Metal compiler (runtime compilation, no Xcode toolchain needed) and, for
// every `kernel` function found, builds a compute pipeline state.
//
// Build (outside Bazel):
//   clang++ -std=c++17 -I <path>/metal-cpp msl_syntax_check.cc \
//       -framework Metal -framework Foundation -o msl_syntax_check
// Usage:
//   msl_syntax_check [--fast-math] file.metal [file2.metal ...]
// Exit status is non-zero if any file fails to compile.
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

namespace {

bool CheckFile(MTL::Device* device, const char* path, bool fast_math) {
  std::ifstream in(path);
  if (!in) {
    std::fprintf(stderr, "%s: cannot open\n", path);
    return false;
  }
  std::stringstream buf;
  buf << in.rdbuf();
  std::string src = buf.str();

  NS::Error* error = nullptr;
  MTL::CompileOptions* options = MTL::CompileOptions::alloc()->init();
  // XLA semantics need IEEE behavior (NaN/Inf, precise transcendentals).
  options->setFastMathEnabled(fast_math);
  MTL::Library* library = device->newLibrary(
      NS::String::string(src.c_str(), NS::UTF8StringEncoding), options, &error);
  options->release();
  if (!library) {
    std::fprintf(stderr, "%s: FAILED\n%s\n", path,
                 error ? error->localizedDescription()->utf8String()
                       : "(no error message)");
    return false;
  }
  if (error) {  // Warnings.
    std::fprintf(stderr, "%s: warnings:\n%s\n", path,
                 error->localizedDescription()->utf8String());
  }
  bool ok = true;
  NS::Array* names = library->functionNames();
  for (NS::UInteger i = 0; i < names->count(); ++i) {
    auto* name = names->object<NS::String>(i);
    MTL::Function* fn = library->newFunction(name);
    if (fn->functionType() == MTL::FunctionTypeKernel) {
      NS::Error* pso_error = nullptr;
      MTL::ComputePipelineState* pso =
          device->newComputePipelineState(fn, &pso_error);
      if (!pso) {
        std::fprintf(stderr, "%s: pipeline for '%s' FAILED: %s\n", path,
                     name->utf8String(),
                     pso_error ? pso_error->localizedDescription()->utf8String()
                               : "?");
        ok = false;
      } else {
        std::printf("%s: kernel '%s' OK (max threads %lu, simd width %lu)\n",
                    path, name->utf8String(),
                    (unsigned long)pso->maxTotalThreadsPerThreadgroup(),
                    (unsigned long)pso->threadExecutionWidth());
        pso->release();
      }
    }
    fn->release();
  }
  library->release();
  return ok;
}

}  // namespace

int main(int argc, char** argv) {
  NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
  MTL::Device* device = MTL::CreateSystemDefaultDevice();
  if (!device) {
    std::fprintf(stderr, "no Metal device\n");
    return 2;
  }
  bool fast_math = false;
  bool ok = true;
  int files = 0;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--fast-math") == 0) {
      fast_math = true;
      continue;
    }
    ++files;
    ok &= CheckFile(device, argv[i], fast_math);
  }
  if (files == 0) {
    std::fprintf(stderr, "usage: %s [--fast-math] file.metal...\n", argv[0]);
    ok = false;
  }
  pool->release();
  return ok ? 0 : 1;
}
