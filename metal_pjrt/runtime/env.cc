// Copyright 2026 The metal-pjrt-plugin Authors
// SPDX-License-Identifier: Apache-2.0

#include "metal_pjrt/runtime/env.h"

#include <cstdint>
#include <cstdlib>
#include <string>

#include "absl/log/log.h"
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/string_view.h"

namespace metal_pjrt {

bool EnvFlag(const char* name) {
  const char* v = std::getenv(name);
  if (v == nullptr) return false;
  const std::string s = absl::AsciiStrToLower(absl::StripAsciiWhitespace(v));
  return !(s.empty() || s == "0" || s == "false" || s == "no" || s == "off");
}

uint64_t EnvUint(const char* name, uint64_t default_value) {
  const char* v = std::getenv(name);
  if (v == nullptr || v[0] == '\0') return default_value;
  uint64_t n = 0;
  if (absl::SimpleAtoi(v, &n)) return n;
  LOG(WARNING) << "Ignoring " << name << "=" << v
               << " (not a non-negative integer); using " << default_value;
  return default_value;
}

uint64_t EnvMegabytes(const char* name, uint64_t default_mb) {
  constexpr uint64_t kMaxMb = UINT64_MAX >> 20;
  const uint64_t mb = EnvUint(name, default_mb);
  if (mb <= kMaxMb) return mb << 20;
  LOG(WARNING) << "Ignoring " << name << "=" << mb << " (more than " << kMaxMb
               << " MB); using " << default_mb;
  return default_mb << 20;
}

}  // namespace metal_pjrt
