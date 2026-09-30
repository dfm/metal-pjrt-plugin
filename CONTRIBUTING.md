# Contributing

Issues and pull requests are welcome. The project is young and has been
developed and tested on one machine (M3, 8 GB, macOS 26.2), so reports from
other Apple GPUs and macOS versions are especially useful.

## Reporting a problem

Please include:

- the Mac (chip, memory) and macOS version (`sw_vers`);
- the plugin commit, and the `jax` and `jaxlib` versions;
- the full error text. [`docs/troubleshooting.md`](docs/troubleshooting.md)
  maps the plugin's messages to what to do; an error that says
  "metal-pjrt-plugin bug; please report it" is always worth reporting, with
  the HLO dump it asks for;
- a small program that reproduces it, and what the same program gives with
  `JAX_PLATFORMS=cpu`.

A wrong result matters more than anything else here: if mtl and CPU
disagree beyond rounding, please report it even without a small
reproduction.

A test that fails only on its tolerance on other hardware is useful too:
the tolerances in `tests/` are about twice the error measured on the M3.
Run it with `METAL_TEST_REPORT_ULPS=1` and `pytest -s` and include the
printed errors.

If the GPU was reset (the plugin logs it, and every later GPU operation in
the process fails), include `~/.cache/metal-pjrt/gpu_resets.jsonl`.

## Changing the code

[`docs/development.md`](docs/development.md) covers the layout, building,
tests, benchmarks and environment variables;
[`docs/design.md`](docs/design.md) the design and the runtime's policies.
In short:

- `scripts/install_dev.sh` builds and installs into `.venv` (the first
  build of XLA takes about two hours on 8 GB).
- Run everything that touches the GPU under `scripts/device_lock.py --
  <command>`, one job at a time, and do not kill a process that has GPU
  work in flight (README, "GPU safety").
- `scripts/device_lock.py -- .venv/bin/python -m pytest tests` and
  `scripts/device_lock.py -- bazel test //metal_pjrt:device_tests` should
  pass; `bazel test //metal_pjrt/...` runs the tests that need no GPU.
- A fix for a wrong result comes with a test that compares against CPU
  (`tests/metal_testing.py`).
- Measure before adding machinery, and say what was measured.

Contributions are accepted under the project's license, Apache-2.0
(`LICENSE`). Code ported from another project keeps that project's
copyright notice in the file and gets an entry in `THIRD_PARTY_NOTICES`.
