// The runtime's built-in kernels (rt::Device::BuiltinKernel). Fill: `v` is
// the pattern broadcast to 32 bits, `n` the number of elements (words or
// bytes) to write; byte i takes byte (i mod 4) of the pattern, which is right
// for 1-, 2- and 4-byte patterns alike. Copy: 16-byte vectors when both ends
// are 16-byte aligned, bytes otherwise.
#include <metal_stdlib>
using namespace metal;
kernel void xla_metal_fill32(device uint* p [[buffer(0)]],
                             constant uint& v [[buffer(1)]],
                             constant uint& n [[buffer(2)]],
                             uint i [[thread_position_in_grid]]) {
  if (i < n) p[i] = v;
}
kernel void xla_metal_fill8(device uchar* p [[buffer(0)]],
                            constant uint& v [[buffer(1)]],
                            constant uint& n [[buffer(2)]],
                            uint i [[thread_position_in_grid]]) {
  if (i < n) p[i] = (uchar)((v >> (8u * (i & 3u))) & 0xffu);
}
kernel void xla_metal_copy16(device const uint4* src [[buffer(0)]],
                             device uint4* dst [[buffer(1)]],
                             constant uint& n [[buffer(2)]],
                             uint i [[thread_position_in_grid]]) {
  if (i < n) dst[i] = src[i];
}
kernel void xla_metal_copy8(device const uchar* src [[buffer(0)]],
                            device uchar* dst [[buffer(1)]],
                            constant uint& n [[buffer(2)]],
                            uint i [[thread_position_in_grid]]) {
  if (i < n) dst[i] = src[i];
}
