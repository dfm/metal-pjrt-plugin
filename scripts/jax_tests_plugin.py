"""pytest plugin used by scripts/run_jax_tests.sh for JAX's own test files.

- Before any test runs, prints the backend and refuses to run unless it is
  metal, so pass/fail counts can never silently come from CPU.
- Compares failures against scripts/jax_known_failures/<file>.txt (test ids
  without the file part, '#' comments). New failures make the run fail;
  known failures don't; known failures that now pass are reported so the
  manifest can be trimmed. A file without a manifest has no known failures.
"""
import pathlib

import pytest

MANIFESTS = pathlib.Path(__file__).resolve().parent / "jax_known_failures"

_failed = {}   # nodeid -> first line of the error
_passed = set()
_manifests = {}  # nodeid file part -> known-failure ids
_report = {}


def pytest_sessionstart(session):
    import jax
    backend = jax.default_backend()
    print(f"\njax backend: {backend} {jax.devices()}", flush=True)
    if backend != "metal":
        raise pytest.UsageError(f"default backend is {backend!r}, not 'metal'")


def pytest_runtest_logreport(report):
    if report.failed:
        crash = getattr(report.longrepr, "reprcrash", None)
        msg = crash.message if crash else str(report.longrepr)
        _failed.setdefault(report.nodeid, (msg.strip().splitlines() or [""])[0])
    elif report.when == "call" and report.passed:
        _passed.add(report.nodeid)


def _known(path):
    """Known-failure ids for the test file at `path` (nodeid file part)."""
    if path not in _manifests:
        f = MANIFESTS / (pathlib.PurePath(path).stem + ".txt")
        lines = f.read_text().splitlines() if f.exists() else []
        _manifests[path] = {l.strip() for l in lines
                            if l.strip() and not l.startswith("#")}
    return _manifests[path]


def _split(nodeid):
    path, _, test = nodeid.partition("::")
    return path, test


@pytest.hookimpl(tryfirst=True)
def pytest_sessionfinish(session, exitstatus):
    new = sorted(n for n in _failed if _split(n)[1] not in _known(_split(n)[0]))
    known = [n for n in _failed if n not in new]
    fixed = sorted(n for n in _passed - set(_failed)
                   if _split(n)[1] in _known(_split(n)[0]))
    _report.update(new=new, known=known, fixed=fixed)
    if exitstatus == pytest.ExitCode.TESTS_FAILED and not new:
        session.exitstatus = pytest.ExitCode.OK


def pytest_terminal_summary(terminalreporter):
    if not _report:
        return
    tr = terminalreporter
    tr.section("metal: known-failures manifest")
    tr.write_line(f"{len(_report['known'])} known failures, "
                  f"{len(_report['new'])} new failures, "
                  f"{len(_report['fixed'])} unexpected passes "
                  f"(manifests: {MANIFESTS})")
    for n in _report["new"]:
        tr.write_line(f"NEW FAILURE {n}: {_failed[n][:200]}", red=True)
    for n in _report["fixed"]:
        tr.write_line(f"UNEXPECTED PASS {n} (remove it from the manifest)",
                      yellow=True)
