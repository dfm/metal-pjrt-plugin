# Copyright 2026 The jax-graft Authors
# SPDX-License-Identifier: Apache-2.0

"""pytest plugin used by scripts/run_jax_tests.sh for JAX's own test files.

- Before any test runs, prints the backend and refuses to run unless it is
  mtl, so pass/fail counts can never silently come from CPU.
- Compares failures against scripts/jax_known_failures/<file>.txt (test ids
  without the file part, '#' comments). New failures make the run fail;
  known failures don't; known failures that now pass are reported so the
  manifest can be trimmed. A file without a manifest has no known failures.
  Test ids depend on the run's settings, so a manifest's "# config: K=V ..."
  line must match the run (environment, or jax.config for JAX_ENABLE_X64);
  otherwise the comparison is skipped and pytest's own exit status stands.
- JAX_TESTS_DUT=gpu (run_jax_tests.sh's default) labels the device under
  test "gpu" (jax_test_dut; mtl then matches the tags {gpu, oneapi}), so
  tests gated on jtu.test_device_matches(["gpu"]) run instead of skipping.
  It is a jax.config flag defined in jax._src.test_util: setting
  JAX_TEST_DUT in the environment does nothing.
- JAX_TESTS_UPDATE_MANIFEST=1 rewrites the manifests of the files that ran:
  known failures that still fail keep their comment block, ones that pass
  are dropped, new failures are added under "# TODO review:" and their
  error, and the config line becomes the run's. Review the diff: a manifest
  lists what is allowed to fail.
"""
import os
import pathlib

import pytest

MANIFESTS = pathlib.Path(__file__).resolve().parent / "jax_known_failures"

_failed = {}   # nodeid -> first line of the error
_passed = set()
_manifests = {}  # nodeid file part -> (known-failure ids, config mismatch)
_report = {}


CONFIG_KEYS = ("JAX_NUM_GENERATED_CASES", "JAX_ENABLE_X64", "JAX_TESTS_DUT")


def pytest_configure(config):
    dut = os.environ.get("JAX_TESTS_DUT")
    if dut:
        import jax
        from jax._src import test_util  # noqa: F401 - defines jax_test_dut
        jax.config.update("jax_test_dut", dut)


def pytest_sessionstart(session):
    import jax
    backend = jax.default_backend()
    print(f"\njax backend: {backend} {jax.devices()}", flush=True)
    if backend != "mtl":
        raise pytest.UsageError(f"default backend is {backend!r}, not 'mtl'")
    print(f"jax {jax.__version__} ({(jax.version._git_hash or 'release')[:12]}), "
          f"test labels: {os.environ.get('JAX_TESTS_DUT') or 'none'}", flush=True)


def pytest_runtest_logreport(report):
    if report.failed:
        crash = getattr(report.longrepr, "reprcrash", None)
        msg = crash.message if crash else str(report.longrepr)
        _failed.setdefault(report.nodeid, (msg.strip().splitlines() or [""])[0])
    elif report.when == "call" and report.passed:
        _passed.add(report.nodeid)


def _manifest(path):
    """(known-failure ids, config mismatch or None) for the test file at
    `path` (nodeid file part)."""
    if path not in _manifests:
        f = MANIFESTS / (pathlib.PurePath(path).stem + ".txt")
        lines = f.read_text().splitlines() if f.exists() else []
        ids = {l.strip() for l in lines if l.strip() and not l.startswith("#")}
        mismatch = None
        for l in lines:
            if l.startswith("# config:"):
                for kv in l.split(":", 1)[1].split():
                    key, want = kv.split("=")
                    have = os.environ.get(key)
                    if key == "JAX_ENABLE_X64":
                        import jax
                        have = str(int(jax.config.jax_enable_x64))
                    if have != want:
                        mismatch = f"{f.name} expects {key}={want}, run has {have}"
        _manifests[path] = (ids, mismatch)
    return _manifests[path]


def _split(nodeid):
    path, _, test = nodeid.partition("::")
    return path, test


def _run_config():
    import jax
    values = {k: os.environ.get(k, "") for k in CONFIG_KEYS}
    values["JAX_ENABLE_X64"] = str(int(jax.config.jax_enable_x64))
    return " ".join(f"{k}={v}" for k, v in values.items() if v)


def _update_manifest(path):
    """Rewrites the manifest of the test file at `path` (nodeid file part)
    from this run's results (JAX_TESTS_UPDATE_MANIFEST=1)."""
    f = MANIFESTS / (pathlib.PurePath(path).stem + ".txt")
    failed = {_split(n)[1]: msg for n, msg in _failed.items() if _split(n)[0] == path}
    header, blocks, block = [], [], None
    for line in (f.read_text().splitlines() if f.exists() else []):
        if block is None and line.startswith("#") and not line.startswith("# config:"):
            header.append(line)            # the leading comment, up to a blank line
            continue
        if line.startswith("# config:"):
            continue
        if not line.strip():
            if block is None:
                block = [[], []]
            elif block[1]:
                blocks.append(block)
                block = [[], []]
            continue
        if block is None:
            block = [[], []]
        (block[0] if line.startswith("#") else block[1]).append(line.strip())
    if block and block[1]:
        blocks.append(block)
    if not header:
        header = [f"# Known failures of JAX's tests/{pathlib.PurePath(path).name} on mtl, as",
                  "# run by scripts/run_jax_tests.sh. One test id per line (without the",
                  "# file), '#' comments; scripts/jax_tests_plugin.py explains matching."]
    out = header + [f"# config: {_run_config()}"]
    known = set()
    for comments, ids in blocks:
        keep = [i for i in ids if i in failed]
        known.update(ids)
        if keep:
            out += [""] + comments + keep
    new = {}
    for i, msg in sorted(failed.items()):
        if i not in known:
            new.setdefault(msg[:150], []).append(i)
    for msg, ids in new.items():
        out += ["", f"# TODO review: {msg}"] + ids
    if failed:
        f.write_text("\n".join(out) + "\n")
    elif f.exists():
        f.unlink()


@pytest.hookimpl(tryfirst=True)
def pytest_sessionfinish(session, exitstatus):
    files = {_split(n)[0] for n in _failed.keys() | _passed}
    if os.environ.get("JAX_TESTS_UPDATE_MANIFEST") == "1":
        for path in sorted(files):
            _update_manifest(path)
        _manifests.clear()
    mismatches = sorted({m for p in files if (m := _manifest(p)[1])})
    if mismatches:
        _report.update(skipped=mismatches)
        return
    known = lambda n: _split(n)[1] in _manifest(_split(n)[0])[0]
    new = sorted(n for n in _failed if not known(n))
    fixed = sorted(n for n in _passed - _failed.keys() if known(n))
    _report.update(new=new, known=len(_failed) - len(new), fixed=fixed)
    if exitstatus == pytest.ExitCode.TESTS_FAILED and not new:
        session.exitstatus = pytest.ExitCode.OK


def pytest_terminal_summary(terminalreporter):
    if not _report:
        return
    tr = terminalreporter
    tr.section("mtl: known-failures manifest")
    if "skipped" in _report:
        for m in _report["skipped"]:
            tr.write_line(f"NOT COMPARED: {m} (test ids differ)", yellow=True)
        return
    tr.write_line(f"{_report['known']} known failures, "
                  f"{len(_report['new'])} new failures, "
                  f"{len(_report['fixed'])} unexpected passes "
                  f"(manifests: {MANIFESTS})")
    for n in _report["new"]:
        tr.write_line(f"NEW FAILURE {n}: {_failed[n][:200]}", red=True)
    for n in _report["fixed"]:
        tr.write_line(f"UNEXPECTED PASS {n} (remove it from the manifest)",
                      yellow=True)
