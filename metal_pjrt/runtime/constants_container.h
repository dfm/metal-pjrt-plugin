// Serialization of an XLA "constants module" (a set of named byte blobs) into
// the opaque binary that GpuExecutable carries and StreamExecutor::LoadModule
// receives. Shared by the compiler (writer) and the executor (reader); kept
// free of XLA dependencies.
//
// Layout (little endian):
//   char[8]  magic "MTLCONST"
//   u32      version (1)
//   u32      count
//   count x { u32 name_len; u32 reserved; u64 data_len; char name[name_len];
//             pad to 8; u8 data[data_len]; pad to 8 }
#ifndef METAL_PJRT_RUNTIME_CONSTANTS_CONTAINER_H_
#define METAL_PJRT_RUNTIME_CONSTANTS_CONTAINER_H_

#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace metal_pjrt {
namespace rt {

struct ConstantBlob {
  std::string name;
  std::vector<uint8_t> data;
};

inline constexpr char kConstantsMagic[8] = {'M', 'T', 'L', 'C',
                                            'O', 'N', 'S', 'T'};

inline std::vector<uint8_t> SerializeConstants(
    const std::vector<ConstantBlob>& blobs) {
  std::vector<uint8_t> out;
  auto put = [&out](const void* p, size_t n) {
    const uint8_t* b = static_cast<const uint8_t*>(p);
    out.insert(out.end(), b, b + n);
  };
  auto pad8 = [&out]() {
    while (out.size() % 8 != 0) out.push_back(0);
  };
  put(kConstantsMagic, 8);
  uint32_t version = 1, count = static_cast<uint32_t>(blobs.size());
  put(&version, 4);
  put(&count, 4);
  for (const ConstantBlob& b : blobs) {
    uint32_t name_len = static_cast<uint32_t>(b.name.size()), reserved = 0;
    uint64_t data_len = b.data.size();
    put(&name_len, 4);
    put(&reserved, 4);
    put(&data_len, 8);
    put(b.name.data(), b.name.size());
    pad8();
    put(b.data.data(), b.data.size());
    pad8();
  }
  return out;
}

// Returns false if the bytes are not a constants container.
inline bool IsConstantsContainer(const uint8_t* bytes, size_t size) {
  return size >= 16 && std::memcmp(bytes, kConstantsMagic, 8) == 0;
}

inline bool ParseConstants(const uint8_t* bytes, size_t size,
                           std::vector<ConstantBlob>* out) {
  if (!IsConstantsContainer(bytes, size)) return false;
  size_t pos = 8;
  auto get = [&](void* dst, size_t n) {
    if (pos + n > size) return false;
    std::memcpy(dst, bytes + pos, n);
    pos += n;
    return true;
  };
  auto align8 = [&]() { pos = (pos + 7) & ~static_cast<size_t>(7); };
  uint32_t version = 0, count = 0;
  if (!get(&version, 4) || !get(&count, 4) || version != 1) return false;
  out->clear();
  for (uint32_t i = 0; i < count; ++i) {
    uint32_t name_len = 0, reserved = 0;
    uint64_t data_len = 0;
    if (!get(&name_len, 4) || !get(&reserved, 4) || !get(&data_len, 8)) {
      return false;
    }
    ConstantBlob b;
    b.name.resize(name_len);
    if (!get(b.name.data(), name_len)) return false;
    align8();
    b.data.resize(data_len);
    if (data_len > 0 && !get(b.data.data(), data_len)) return false;
    align8();
    out->push_back(std::move(b));
  }
  return true;
}

}  // namespace rt
}  // namespace metal_pjrt

#endif  // METAL_PJRT_RUNTIME_CONSTANTS_CONTAINER_H_
