#!/usr/bin/env python3
# Copyright 2026 The jax-graft Authors
# SPDX-License-Identifier: Apache-2.0

"""GPU tests across JAX versions: this repository's suite and JAX's own tests
on mtl (docs/development.md, "Testing across JAX versions").

  scripts/jax_grid.py                     # fast: the dev .venv as it is
  scripts/jax_grid.py --tier slow         # release: every version, every file
  scripts/jax_grid.py --tier slow --wheels ci        # main's latest CI wheels
  scripts/jax_grid.py --tier slow --wheels ci:<run-id> --jax 0.11.2 nightly
                                          # a ci.yml run, or the release run to approve
  scripts/jax_grid.py --files linalg_test fft_test --jax nightly

Each JAX version gets a venv (~/.cache/metal-pjrt/grid/venv-<version>) with
the plugin's wheels (built from bazel-bin with scripts/build_wheel.sh
--no-build, or downloaded from a ci.yml run with gh) and that jax/jaxlib,
and JAX's tests at the commit the installed jax was built from
(jax.version._git_hash). "dev" is the editable .venv, unchanged. Every GPU
step holds the device lock, one at a time, and waits while a Bazel build or
test runs (memory). Logs and summary.md go to
~/.cache/metal-pjrt/grid/<time>/. Exit status: non-zero if this suite
failed or a JAX file had failures not in scripts/jax_known_failures/.
"""
import argparse
import datetime
import json
import os
import pathlib
import re
import subprocess
import sys
import time
import urllib.request

REPO = pathlib.Path(__file__).resolve().parents[1]
GRID = pathlib.Path.home() / ".cache/metal-pjrt/grid"
NIGHTLY_INDEX = "https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/"
TEST_DEPS = ["absl-py", "hypothesis"]

# JAX test files (in JAX's tests/), run with the "gpu" label. FAST takes a
# few minutes; SLOW is every file the 2026-10-08 coverage sweep ran
# (about 25 minutes per JAX version).
FAST = ["lax_test", "lax_numpy_operators_test", "lax_numpy_reducers_test",
        "linalg_test", "fft_test", "python_callback_test"]
SLOW = sorted(set(FAST) | {
    "ann_test", "aot_test", "api_test", "array_test", "batching_test",
    "checkify_test", "core_test", "custom_api_test", "custom_linear_solve_test",
    "custom_root_test", "debug_nans_test", "debugging_primitives_test",
    "dtypes_test", "eigh_test", "export_test", "image_test", "jax_jit_test",
    "lax_autodiff_test", "lax_control_flow_test", "lax_numpy_einsum_test",
    "lax_numpy_indexing_test", "lax_numpy_test", "lax_scipy_special_functions_test",
    "lax_scipy_test", "lax_vmap_op_test", "layout_test", "memories_test",
    "nn_test", "ode_test", "pjit_test", "qdwh_test", "random_lax_test",
    "scaled_dot_test", "scipy_stats_test", "sparse_test", "svd_test",
    "transfer_guard_test", "unary_ops_accuracy_test",
})


def run(cmd, **kw):
    print("+", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, check=True, **kw)


def wait_for_bazel():
    while subprocess.run(["pgrep", "-f", r"bazel(-real)? (build|test)|bin/bazel (build|test)"],
                         capture_output=True).returncode == 0:
        print("waiting for a Bazel build/test to finish...", flush=True)
        time.sleep(30)


def jax_min():
    text = (REPO / "jax_graft/__init__.py").read_text()
    nums = re.search(r"_JAX_MIN = \(([\d, ]+)\)", text).group(1)
    return ".".join(n.strip() for n in nums.split(","))


def default_versions():
    """_JAX_MIN, the newest patch of each later minor release, and nightly."""
    with urllib.request.urlopen("https://pypi.org/pypi/jaxlib/json") as r:
        releases = json.load(r)["releases"]
    key = lambda v: tuple(int(x) for x in v.split("."))  # noqa: E731
    lo = key(jax_min())
    stable = sorted((v for v in releases if re.fullmatch(r"\d+\.\d+\.\d+", v)
                     and key(v) >= lo and releases[v]), key=key)
    newest = {}
    for v in stable:
        newest[key(v)[:2]] = v
    return [jax_min()] + [v for v in newest.values() if v != jax_min()] + ["nightly"]


def get_wheels(spec, out):
    out.mkdir(parents=True, exist_ok=True)
    if spec == "local":
        run([REPO / "scripts/build_wheel.sh", "--no-build"], cwd=REPO)
        src = REPO / "dist"
    else:
        run_id = spec.partition(":")[2]
        if not run_id:
            run_id = subprocess.run(
                ["gh", "run", "list", "--workflow", "ci.yml", "--branch", "main",
                 "--event", "push", "--status", "success", "--limit", "1",
                 "--json", "databaseId", "--jq", ".[0].databaseId"],
                cwd=REPO, check=True, capture_output=True, text=True).stdout.strip()
        src = out / f"ci-{run_id}"
        if not src.exists():
            # ci.yml's "wheels", or release.yml's "wheel-core"/"wheel-plugin".
            run(["gh", "run", "download", run_id, "--pattern", "wheel*", "--dir", src], cwd=REPO)
    newest = lambda pat: max(src.rglob(pat), key=os.path.getmtime, default=None)  # noqa: E731
    plugin = newest("jax_graft-*.whl")
    if plugin is None:
        sys.exit(f"no jax_graft wheel in {src}")
    # None: a release that reuses a published core; the frontend's exact
    # pin then installs it from PyPI.
    return newest("metal_pjrt_core-*.whl"), plugin


def make_venv(version, wheels):
    if version == "dev":
        py = REPO / ".venv/bin/python"
        run(["uv", "pip", "install", "-q", "--python", py, *TEST_DEPS])
        return py
    venv = GRID / f"venv-{version}"
    py = venv / "bin/python"
    if not py.exists():
        run(["uv", "venv", "-q", "--python", "3.12", venv])
    core, plugin = wheels
    run(["uv", "pip", "install", "-q", "--python", py, "--reinstall-package",
         "metal-pjrt-core", "--reinstall-package", "jax-graft",
         *([core] if core else []), f"{plugin}[test]", *TEST_DEPS])
    run([py, REPO / "scripts/check_installed_wheels.py"], cwd="/")
    if version == "nightly":
        run(["uv", "pip", "install", "-q", "--python", py, "--prerelease=allow",
             "--upgrade-package", "jax", "--upgrade-package", "jaxlib",
             "--extra-index-url", NIGHTLY_INDEX, "jax", "jaxlib"])
    else:
        run(["uv", "pip", "install", "-q", "--python", py,
             f"jax=={version}", f"jaxlib=={version}"])
    return py


def jax_tests(py):
    """JAX's sources for the installed jax (shallow, cached): the commit a
    nightly was built from (jax.version._git_hash), a release's tag."""
    version, sha = subprocess.run(
        [py, "-c", "import jax; print(jax.__version__, jax.version._git_hash)"],
        check=True, capture_output=True, text=True, cwd="/").stdout.split()
    ref = sha if sha != "None" else f"refs/tags/jax-v{version}"
    src = GRID / f"jax-src-{sha[:12] if sha != 'None' else version}"
    if not (src / "tests").exists():
        src.mkdir(parents=True, exist_ok=True)
        run(["git", "init", "-q"], cwd=src)
        run(["git", "fetch", "-q", "--depth", "1", "https://github.com/jax-ml/jax", ref], cwd=src)
        run(["git", "checkout", "-q", "FETCH_HEAD"], cwd=src)
    return version, src


def summary_line(log):
    lines = [l for l in log.read_text(errors="replace").splitlines() if l.strip()]
    tail = [l for l in lines if re.search(r"\d+ (passed|failed|skipped|error)", l)]
    news = sum(1 for l in lines if l.startswith("NEW FAILURE"))
    return (tail[-1].strip("= ") if tail else (lines[-1] if lines else "(no output)")), news


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tier", choices=["fast", "slow"], default="fast")
    ap.add_argument("--jax", nargs="+", help="versions: dev (the .venv), X.Y.Z, nightly "
                    "(default: fast: dev; slow: _JAX_MIN, newest of each later minor, nightly)")
    ap.add_argument("--wheels", default="local",
                    help="local (build from bazel-bin), ci (main's latest ci.yml run) or "
                    "ci:<run-id> (a ci.yml or release.yml run)")
    ap.add_argument("--files", nargs="+", help="JAX test files instead of the tier's (no .py)")
    ap.add_argument("--no-suite", action="store_true", help="skip this repository's tests")
    ap.add_argument("--update-manifests", action="store_true",
                    help="rewrite scripts/jax_known_failures/ from this run (review the diff)")
    args = ap.parse_args()
    versions = args.jax or (["dev"] if args.tier == "fast" else default_versions())
    files = args.files or (FAST if args.tier == "fast" else SLOW)
    out = GRID / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True)
    wheels = None if versions == ["dev"] else get_wheels(args.wheels, GRID / "wheels")
    rows, failed = [], False
    for v in versions:
        py = make_venv(v, wheels)
        version, src = jax_tests(py)
        label = f"{v} (jax {version})"
        print(f"\n=== {label}: {src}", flush=True)
        if not args.no_suite:
            wait_for_bazel()
            log = out / f"{v}-suite.log"
            # From outside the checkout for wheels (the installed frontend,
            # not the source tree), as in release.yml.
            cwd = REPO if v == "dev" else out
            with open(log, "w") as f:
                rc = subprocess.run([REPO / "scripts/device_lock.py", "--", py, "-m", "pytest",
                                     REPO / "tests", "-q", "-p", "no:cacheprovider", "-rfE"],
                                    cwd=cwd, stdout=f, stderr=subprocess.STDOUT).returncode
            line, _ = summary_line(log)
            rows.append((label, "this repository's tests", "ok" if rc == 0 else "FAIL", line))
            failed |= rc != 0
        env = dict(os.environ, PYTHON=str(py), JAX_TESTS_DIR=str(src))
        if args.update_manifests:
            env["JAX_TESTS_UPDATE_MANIFEST"] = "1"
        for name in files:
            if not (src / "tests" / f"{name}.py").exists():
                rows.append((label, name, "missing", "not in this JAX's tests"))
                continue
            wait_for_bazel()
            log = out / f"{v}-{name}.log"
            # -s: with the gpu label, debugging_primitives_test and the
            # effect-counting tests fail only under pytest's output capture
            # (on CPU too). -p no:randomly: stable order.
            with open(log, "w") as f:
                rc = subprocess.run([REPO / "scripts/run_jax_tests.sh", f"tests/{name}.py",
                                     "-s", "-p", "no:randomly"],
                                    env=env, stdout=f, stderr=subprocess.STDOUT).returncode
            line, news = summary_line(log)
            status = "ok" if rc == 0 else (f"{news} NEW" if news else "FAIL")
            rows.append((label, name, status, line))
            failed |= rc != 0
            print(f"{name}: {status} {line}", flush=True)
    table = ["| JAX | tests | status | result |", "|---|---|---|---|"]
    table += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows]
    (out / "summary.md").write_text("\n".join(table) + "\n")
    print("\n" + "\n".join(table) + f"\n\nlogs: {out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
