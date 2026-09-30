# Copyright 2026 The metal-pjrt-plugin Authors
# SPDX-License-Identifier: Apache-2.0

"""Every C++ device test (a cc_test with tags = DEVICE_TEST_TAGS) is listed
in //metal_pjrt:device_tests. A test_suite silently skips manual tests
it doesn't name, which is how the old README command came to run nothing.
A text scan of the BUILD files, so no Bazel server starts. No GPU needed."""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent / "metal_pjrt"


def test_device_tests_suite_lists_every_device_test():
    device_tests = set()
    for build in ROOT.rglob("BUILD.bazel"):
        pkg = build.parent.relative_to(ROOT.parent).as_posix()
        for rule in re.split(r"\n(?=\w+\()", build.read_text()):
            if rule.startswith("cc_test(") and "DEVICE_TEST_TAGS" in rule:
                name = re.search(r'name = "([^"]+)"', rule).group(1)
                device_tests.add(f"//{pkg}:{name}")
    suite = (ROOT / "BUILD.bazel").read_text()
    listed = set(re.findall(r'"(//metal_pjrt/[^"]+)"',
                            suite[suite.index('name = "device_tests"'):]))
    assert device_tests, "no device tests found; did DEVICE_TEST_TAGS move?"
    assert device_tests == listed, (
        f"missing from device_tests: {sorted(device_tests - listed)}; "
        f"listed but not device tests: {sorted(listed - device_tests)}")
