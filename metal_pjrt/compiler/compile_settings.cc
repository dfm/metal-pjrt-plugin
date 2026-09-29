#include "metal_pjrt/compiler/compile_settings.h"

#include <cstdlib>
#include <string>

#include "absl/log/log.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "metal_pjrt/runtime/env.h"

namespace metal_pjrt {

std::string CompileSettings::Fingerprint() const {
  return absl::StrCat("lapack=", lapack, ";scan=", scan_rewrite,
                      ";cubsort=", cub_sort, ";conv=", conv_rewrite,
                      ";fft=", fft);
}

const CompileSettings& GetCompileSettings() {
  static const CompileSettings settings = [] {
    CompileSettings s;
    s.lapack = !EnvFlag("METAL_PJRT_DISABLE_LAPACK");
    s.fft = !EnvFlag("METAL_PJRT_DISABLE_FFT");
    if (const char* v = std::getenv("METAL_PJRT_DISABLE_REWRITES")) {
      for (absl::string_view name :
           absl::StrSplit(v, ',', absl::SkipWhitespace())) {
        name = absl::StripAsciiWhitespace(name);
        if (name == "all") {
          s.scan_rewrite = s.cub_sort = s.conv_rewrite = false;
        } else if (name == "scan") {
          s.scan_rewrite = false;
        } else if (name == "cubsort") {
          s.cub_sort = false;
        } else if (name == "conv") {
          s.conv_rewrite = false;
        } else {
          LOG(WARNING) << "Ignoring \"" << name
                       << "\" in METAL_PJRT_DISABLE_REWRITES; the valid "
                          "values are scan, cubsort, conv and all";
        }
      }
    }
    return s;
  }();
  return settings;
}

}  // namespace metal_pjrt
