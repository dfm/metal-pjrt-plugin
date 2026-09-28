#include "metal_pjrt_plugin/compiler/compile_settings.h"

#include <cstdlib>
#include <string>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"

namespace metal_pjrt {

std::string CompileSettings::Fingerprint() const {
  return absl::StrCat("lapack=", lapack, ";scan=", scan_rewrite,
                      ";cubsort=", cub_sort);
}

const CompileSettings& GetCompileSettings() {
  static const CompileSettings settings = [] {
    CompileSettings s;
    const char* lapack = std::getenv("METAL_PJRT_DISABLE_LAPACK");
    s.lapack = lapack == nullptr || absl::string_view(lapack).empty() ||
               absl::string_view(lapack) == "0";
    if (const char* v = std::getenv("METAL_PJRT_DISABLE_REWRITES")) {
      for (absl::string_view name : absl::StrSplit(v, ',')) {
        if (name == "scan" || name == "all") s.scan_rewrite = false;
        if (name == "cubsort" || name == "all") s.cub_sort = false;
      }
    }
    return s;
  }();
  return settings;
}

}  // namespace metal_pjrt
