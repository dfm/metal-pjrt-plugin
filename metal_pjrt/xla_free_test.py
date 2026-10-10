# Copyright 2026 The jax-graft Authors
# SPDX-License-Identifier: Apache-2.0

"""The dispatch libraries' tests link no XLA, and the host tests no Metal.

conv_test, fft_test, scan_test, radix_sort_test and small_linalg_test run
the kernels on rt::Device alone, and lapack_host_test and fft_plan_test are
plain host tests; a dependency on XLA (or, for the host tests, on metal-cpp,
the runtime or the Metal framework) creeping back in would undo that. Reads
the genquery outputs named in argv: all these tests' transitive deps, the
host tests' deps, the rules among those that link Metal/QuartzCore, and the
direct deps of //metal_pjrt/ffi:metal_ffi. Controls keep a label-format
change from silencing a check: the XLA pattern must match among metal_ffi's
direct deps, and both Metal patterns (metal-cpp, the runtime/kernels
packages) among the device dispatch tests' deps (reached through
metal_runtime).
"""
import re
import sys

XLA = re.compile(r"^@@?xla[+/]")  # @xla//..., @@xla+//..., @@xla++ext+repo//...
# @metal_cpp//..., canonical @@+_repo_rules+metal_cpp//...
METAL_CPP = re.compile(r"(^|[+@])metal_cpp//")
METAL_PKGS = re.compile(r"^//metal_pjrt/(runtime|kernels)[:/]")


def labels(path):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main(dispatch_deps, host_deps, host_metal_linkopts,
         metal_ffi_direct_deps):
    failures = []
    control = [l for l in labels(metal_ffi_direct_deps) if XLA.match(l)]
    if not control:
        failures.append("control: no XLA label among metal_ffi's direct deps; "
                        "did the label format change?")
    for name, pattern in (("metal-cpp", METAL_CPP), ("runtime/kernels", METAL_PKGS)):
        if not any(pattern.search(l) for l in labels(dispatch_deps)):
            failures.append(f"control: no {name} label among the dispatch "
                            "tests' deps; did the label format change?")
    xla = [l for l in labels(dispatch_deps) if XLA.match(l)]
    if xla:
        failures.append(f"XLA in the dispatch tests' deps: {xla[:10]}")
    metal = [l for l in labels(host_deps)
             if METAL_CPP.search(l) or METAL_PKGS.match(l)]
    if metal:
        failures.append(f"Metal in the host tests' deps: {metal[:10]}")
    linkopts = labels(host_metal_linkopts)
    if linkopts:
        failures.append(f"Metal framework linkopts in the host tests' deps: "
                        f"{linkopts}")
    for f in failures:
        print("FAIL:", f)
    if not failures:
        print(f"OK: {len(labels(dispatch_deps))} labels without XLA, "
              f"{len(labels(host_deps))} without Metal; control found "
              f"{len(control)} XLA labels")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
