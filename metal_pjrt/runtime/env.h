// Environment variables, parsed one way everywhere. Booleans: unset, "",
// "0", "false", "no" and "off" (any case) are off; anything else is on
// (metal_pjrt_plugin._env_flag is the Python twin). Numbers: a value that
// does not parse is ignored with a warning, never read as 0.
#ifndef METAL_PJRT_RUNTIME_ENV_H_
#define METAL_PJRT_RUNTIME_ENV_H_

#include <cstdint>

namespace metal_pjrt {

bool EnvFlag(const char* name);

// A non-negative integer, or `default_value` when unset or empty, or (with
// a LOG(WARNING)) when it does not parse.
uint64_t EnvUint(const char* name, uint64_t default_value);

}  // namespace metal_pjrt

#endif  // METAL_PJRT_RUNTIME_ENV_H_
