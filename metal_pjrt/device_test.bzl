"""Settings shared by tests that need a Metal device.

Device tests are listed in //metal_pjrt:device_tests; run them with

  scripts/device_lock.py -- bazel test //metal_pjrt:device_tests

Tags: "manual" keeps them out of `//...` (which must build and pass without a
GPU); "local" runs them outside the sandbox; "exclusive" stops Bazel from
running them in parallel with each other or anything else (one GPU job at a
time on 8 GB of unified memory). Give each one `timeout = "eternal"` too:
Bazel kills a test that overruns its timeout, with GPU work in flight. GPU
hangs are ended by the runtime's bounded waits (watchdog -> error), not by
Bazel.
"""

DEVICE_TEST_TAGS = [
    "exclusive",
    "local",
    "manual",
]
