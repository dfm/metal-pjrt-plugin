// Copyright 2026 The jax-graft Authors
// SPDX-License-Identifier: Apache-2.0

// The device buffer cache's lookup, without Metal (host-testable).
#ifndef METAL_PJRT_RUNTIME_BUFFER_CACHE_H_
#define METAL_PJRT_RUNTIME_BUFFER_CACHE_H_

#include <cstdint>
#include <map>

namespace metal_pjrt {
namespace rt {

// The cache entry to reuse for a buffer of `length` bytes: the smallest
// size in [length, limit), and among equal sizes the first freed. `by_size`
// maps sizes to entries with a `ticket` member: the last work ticket issued
// when the buffer was freed (Device::BeginWork). GPU work reuses any entry
// (later work is ordered after earlier work on the buffer). A buffer the
// host writes into right away must not be one that earlier GPU work may
// still read or write: with `host_write`, only entries whose ticket has
// ended (ticket < ended_below, Device::EndedBelow) qualify. end() if none.
template <typename Map>
typename Map::iterator FindCachedBuffer(Map& by_size, uint64_t length,
                                        uint64_t limit, bool host_write,
                                        uint64_t ended_below) {
  auto end = by_size.lower_bound(limit);
  for (auto it = by_size.lower_bound(length); it != end; ++it) {
    if (!host_write || it->second->ticket < ended_below) return it;
  }
  return by_size.end();
}

}  // namespace rt
}  // namespace metal_pjrt

#endif  // METAL_PJRT_RUNTIME_BUFFER_CACHE_H_
