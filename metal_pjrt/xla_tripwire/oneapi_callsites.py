# Copyright 2026 The jax-graft Authors
# SPDX-License-Identifier: Apache-2.0

"""Lists every line of the pinned XLA tree that branches on the OneAPI
capability or on the SYCL platform, and compares it with
oneapi_callsites.txt. The Metal platform reports a OneAPI compute
capability but is not the SYCL platform and targets SPIR, so it takes the
OneAPI side of IsOneAPI() / IsIntelGpu() / is_sycl /
oneapi_compute_capability() branches, the other side of kSyclPlatformId
ones, and the SPIR side of isSPIR*() ones; a new branch of any kind needs a
look. Not a Bazel test (it would have to glob another repository). Run
after moving the XLA pin:

  python3 metal_pjrt/xla_tripwire/oneapi_callsites.py \
      --xla-root "$(bazel info output_base)/external/xla+" [--update]

Entries are "file: line text" (whitespace collapsed, no line numbers), so
unrelated edits do not show up. Tests and the SYCL platform's own directory
(xla/stream_executor/sycl, whose platform is not built here) are skipped.
"""
import argparse
import difflib
import os
import re
import sys

PATTERN = re.compile(r"IsOneAPI\(\)|IsIntelGpu\(\)|is_sycl"
                     r"|oneapi_compute_capability\(\)|kSyclPlatformId"
                     r"|isSPIR")
GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "oneapi_callsites.txt")


def scan(root):
    out = []
    for d, _, files in os.walk(os.path.join(root, "xla")):
        if os.path.relpath(d, root).startswith("xla/stream_executor/sycl"):
            continue
        for f in files:
            if not f.endswith((".cc", ".h")) or f.endswith("_test.cc"):
                continue
            path = os.path.join(d, f)
            rel = os.path.relpath(path, root)
            with open(path, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if PATTERN.search(line):
                        out.append(f"{rel}: {' '.join(line.split())}\n")
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xla-root", required=True)
    ap.add_argument("--update", action="store_true")
    a = ap.parse_args()
    got = scan(a.xla_root)
    if a.update:
        with open(GOLDEN, "w", encoding="utf-8") as f:
            f.writelines(got)
        print(f"wrote {len(got)} call sites to {GOLDEN}")
        return 0
    with open(GOLDEN, encoding="utf-8") as f:
        want = f.readlines()
    if got == want:
        print(f"OK: {len(got)} OneAPI call sites unchanged")
        return 0
    sys.stdout.writelines(difflib.unified_diff(want, got, "oneapi_callsites.txt", "pinned XLA"))
    print("\nOneAPI call sites changed: check what each new branch does for "
          "Metal (add load-bearing ones to tripwire.py), then --update.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
