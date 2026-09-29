"""The dispatch libraries' tests link no XLA, and lapack_host_test no Metal.

scan_test, radix_sort_test and small_linalg_test run the kernels on
rt::Device alone, and lapack_host_test is a plain host test; a dependency
on XLA (or, for lapack_host_test, on metal-cpp, the runtime or the Metal
framework) creeping back in would undo that. Reads the genquery outputs
named in argv: the new tests' transitive deps, lapack_host_test's deps, the
rules among those that link Metal/QuartzCore, and (as a control that the
label pattern still matches) the direct deps of //metal_pjrt/ffi:metal_ffi.
"""
import re
import sys

XLA = re.compile(r"^@@?xla[+/]")  # @xla//..., @@xla+//..., @@xla++ext+repo//...
NO_METAL = re.compile(r"^(@@?metal_cpp[+/]|//metal_pjrt/(runtime|kernels)[:/])")


def labels(path):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main(dispatch_deps, lapack_host_deps, lapack_host_metal_linkopts,
         metal_ffi_direct_deps):
    failures = []
    control = [l for l in labels(metal_ffi_direct_deps) if XLA.match(l)]
    if not control:
        failures.append("control: no XLA label among metal_ffi's direct deps; "
                        "did the label format change?")
    xla = [l for l in labels(dispatch_deps) if XLA.match(l)]
    if xla:
        failures.append(f"XLA in the dispatch tests' deps: {xla[:10]}")
    metal = [l for l in labels(lapack_host_deps) if NO_METAL.match(l)]
    if metal:
        failures.append(f"Metal in lapack_host_test's deps: {metal[:10]}")
    linkopts = labels(lapack_host_metal_linkopts)
    if linkopts:
        failures.append(f"Metal framework linkopts in lapack_host_test's deps: "
                        f"{linkopts}")
    for f in failures:
        print("FAIL:", f)
    if not failures:
        print(f"OK: {len(labels(dispatch_deps))} labels without XLA, "
              f"{len(labels(lapack_host_deps))} without Metal; control found "
              f"{len(control)} XLA labels")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
