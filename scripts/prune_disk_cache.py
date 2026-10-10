#!/usr/bin/env python3
# Copyright 2026 The jax-graft Authors
# SPDX-License-Identifier: Apache-2.0
"""Shrink a Bazel disk cache to a size cap, least recently used first.

  scripts/prune_disk_cache.py ~/.cache/metal-pjrt-plugin/bazel-disk --max-gb 6.5

Bazel's own disk cache GC (--experimental_disk_cache_gc_max_size) runs only
while the server sits idle between builds, which a CI job never does; CI
runs this before saving the cache, so that it fits GitHub's per-repository
cache quota. Bazel updates an entry's modification time when it uses it,
so the oldest modification times are the least recently used entries.
Deleting any entry is safe: a missing one is a cache miss, rebuilt.
"""
import argparse, os, sys


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cache", help="the --disk_cache directory")
    ap.add_argument("--max-gb", type=float, required=True, help="size cap in GB (1e9 bytes)")
    args = ap.parse_args()
    root = os.path.expanduser(args.cache)
    if not os.path.isdir(root):
        print(f"prune_disk_cache: no cache at {root}")
        return
    entries, total = [], 0
    for dirpath, _, files in os.walk(root):
        for f in files:
            p = os.path.join(dirpath, f)
            try:
                st = os.lstat(p)
            except FileNotFoundError:
                continue
            entries.append((st.st_mtime, st.st_size, p))
            total += st.st_size
    cap = int(args.max_gb * 1e9)
    removed = freed = 0
    if total > cap:
        entries.sort()                                   # oldest first
        for _, size, p in entries:
            if total - freed <= cap:
                break
            try:
                os.remove(p)
            except FileNotFoundError:
                continue
            freed += size
            removed += 1
    print(f"prune_disk_cache: {total / 1e9:.2f} GB in {len(entries)} files; removed {removed} "
          f"({freed / 1e9:.2f} GB), now {(total - freed) / 1e9:.2f} GB (cap {args.max_gb} GB)")


if __name__ == "__main__":
    sys.exit(main())
