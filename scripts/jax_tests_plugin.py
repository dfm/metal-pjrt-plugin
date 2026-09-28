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
"""
import os
import pathlib

import pytest

MANIFESTS = pathlib.Path(__file__).resolve().parent / "jax_known_failures"

_failed = {}   # nodeid -> first line of the error
_passed = set()
_manifests = {}  # nodeid file part -> (known-failure ids, config mismatch)
_report = {}


def pytest_sessionstart(session):
    import jax
    backend = jax.default_backend()
    print(f"\njax backend: {backend} {jax.devices()}", flush=True)
    if backend != "mtl":
        raise pytest.UsageError(f"default backend is {backend!r}, not 'mtl'")


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


@pytest.hookimpl(tryfirst=True)
def pytest_sessionfinish(session, exitstatus):
    files = {_split(n)[0] for n in _failed.keys() | _passed}
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
