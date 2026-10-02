# Copyright 2026 The metal-pjrt-plugin Authors
# SPDX-License-Identifier: Apache-2.0

"""Host-only packaging checks (no GPU)."""
import importlib.metadata
import importlib.util
import json
import pathlib
import re
import tomllib

import pytest

import metal_pjrt_plugin

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _pyproject(path="pyproject.toml"):
    return tomllib.loads((ROOT / path).read_text())


def test_jax_requirement_matches_pyproject():
    deps = _pyproject()["project"]["dependencies"]
    want = metal_pjrt_plugin._jax_requirement()
    assert f"jax{want}" in deps and f"jaxlib{want}" in deps, (want, deps)


def test_core_dependency():
    # The frontend pins exactly the metal-pjrt-core in core/ (the build it
    # is tested with); the core has no Python dependencies (in particular
    # not jax), so one build serves every jax the frontend supports.
    deps = _pyproject()["project"]["dependencies"]
    core = _pyproject("core/pyproject.toml")["project"]
    pin = f"metal-pjrt-core=={core['version']};"
    assert any(d.startswith(pin) for d in deps), (pin, deps)
    assert core["name"] == "metal-pjrt-core"
    assert "dependencies" not in core, core.get("dependencies")


def test_one_dist_registers_mtl():
    # One installed dist, metal-pjrt-plugin, registers the plugin: a
    # leftover older dist (openmetal_pjrt_plugin, jax-openmetal,
    # jax-metal-pjrt) would load the dylib a second time.
    # scripts/install_dev.sh uninstalls those.
    old = {"openmetal-pjrt-plugin", "openmetal_pjrt_plugin", "jax-openmetal",
           "jax_openmetal", "jax-metal-pjrt", "jax_metal_pjrt"}
    ours = [ep for ep in importlib.metadata.entry_points(group="jax_plugins")
            if ep.value.startswith(("metal_pjrt_plugin", "jax_plugins.openmetal"))
            or ep.dist.name in old]
    assert [(ep.name, ep.value, ep.dist.name) for ep in ours] == [
        ("mtl", "metal_pjrt_plugin", "metal-pjrt-plugin")], ours


def test_version_warning(monkeypatch, caplog):
    # Below the lowest tested version: a warning naming it, and the plugin
    # still loads (initialize() goes on).
    import jax
    monkeypatch.setattr(jax, "__version__", "0.0.0")
    metal_pjrt_plugin._check_versions()
    assert "found jax 0.0.0" in caplog.text, caplog.text
    assert metal_pjrt_plugin._jax_requirement() in caplog.text, caplog.text


@pytest.mark.parametrize("version,warns", [
    ("0.9.2", True), ("0.9.99.dev1", True), ("0.10.0", False),
    ("0.12.0.dev20261001", False), ("0.11.2+local", False),
    ("0.13.0", False), ("0.13.0.dev1", False), ("1.0.0", False),
])
def test_version_range_edges(monkeypatch, caplog, version, warns):
    import jax
    import jaxlib
    monkeypatch.setattr(jax, "__version__", version)
    monkeypatch.setattr(jaxlib, "__version__", version)
    metal_pjrt_plugin._check_versions()
    assert bool(caplog.text) == warns, caplog.text


def test_version_warning_names_only_the_one_too_old(monkeypatch, caplog):
    import jax
    import jaxlib
    monkeypatch.setattr(jax, "__version__", "0.11.2")
    monkeypatch.setattr(jaxlib, "__version__", "0.9.2")
    metal_pjrt_plugin._check_versions()
    assert "found jaxlib 0.9.2;" in caplog.text, caplog.text
    assert "jax 0.11.2" not in caplog.text, caplog.text


def test_core_abi_version_matches():
    path = metal_pjrt_plugin._get_library_path()
    assert path is not None, metal_pjrt_plugin._missing_library_message()
    assert (metal_pjrt_plugin._core_abi_version(path)
            == metal_pjrt_plugin._CORE_ABI_VERSION)


def _initialize_registers(monkeypatch):
    """Runs initialize() with JAX's register_plugin stubbed; whether it
    registered."""
    registered = []
    import jax._src.xla_bridge as xb
    monkeypatch.setattr(xb, "register_plugin",
                        lambda *a, **k: registered.append(a))
    metal_pjrt_plugin.initialize()
    return bool(registered)


def test_core_abi_mismatch_is_refused(monkeypatch, caplog):
    # Another version of the private contract is a broken install, not an
    # untested combination: nothing is registered.
    monkeypatch.setattr(metal_pjrt_plugin, "_CORE_ABI_VERSION", 999)
    assert not _initialize_registers(monkeypatch)
    assert "frontend ABI version 999" in caplog.text, caplog.text
    assert "'mtl' platform is unavailable" in caplog.text, caplog.text


def test_pre_split_library_is_refused(monkeypatch, caplog):
    # A library without metal_pjrt_frontend_abi_version (built before the
    # packages were split) reads as version 0.
    class NoSymbol:
        def __init__(self, path):
            pass
    monkeypatch.setattr(metal_pjrt_plugin.ctypes, "CDLL", NoSymbol)
    assert not _initialize_registers(monkeypatch)
    assert "has version 0" in caplog.text, caplog.text


def test_unloadable_library_is_refused(monkeypatch, tmp_path, caplog):
    # A file that is not a loadable library: the reason is logged, nothing
    # is registered, and initialize() does not raise.
    lib = tmp_path / "pjrt_c_api_mtl_plugin.dylib"
    lib.write_bytes(b"not a Mach-O file")
    monkeypatch.setattr(metal_pjrt_plugin, "_library_candidate", lambda: lib)
    assert not _initialize_registers(monkeypatch)
    assert "could not be loaded" in caplog.text, caplog.text
    assert "'mtl' platform is unavailable" in caplog.text, caplog.text


def test_public_surface():
    # Users need nothing from the package but PLATFORM; the implementation
    # modules and the jax pin are private.
    assert metal_pjrt_plugin.__all__ == ["PLATFORM", "initialize", "__version__"]
    assert metal_pjrt_plugin.PLATFORM == "mtl"
    assert metal_pjrt_plugin.__version__ == importlib.metadata.version(
        "metal-pjrt-plugin")
    assert metal_pjrt_plugin.initialize.__doc__
    assert not hasattr(metal_pjrt_plugin, "JAX_VERSION")
    for name in ("callbacks", "lowerings", "linalg_lowerings"):
        assert importlib.util.find_spec(f"metal_pjrt_plugin.{name}") is None
        assert importlib.util.find_spec(f"metal_pjrt_plugin._{name}") is not None


@pytest.mark.parametrize("case", ["missing", "dangling", "no core"])
def test_missing_library_warning(monkeypatch, tmp_path, caplog, case):
    # The warning names the expected path (or the missing metal-pjrt-core),
    # whether it is a dangling bazel-bin link, the remedy and the
    # consequence; nothing is registered.
    lib = tmp_path / "pjrt_c_api_mtl_plugin.dylib"
    if case == "dangling":
        lib.symlink_to(tmp_path / "bazel-bin" / "gone.dylib")
    monkeypatch.setattr(metal_pjrt_plugin, "_library_candidate",
                        lambda: None if case == "no core" else lib)
    metal_pjrt_plugin.initialize()
    assert "'mtl' platform is unavailable" in caplog.text, caplog.text
    assert "scripts/install_dev.sh" in caplog.text, caplog.text
    assert ("which does not exist" in caplog.text) == (case == "dangling"), caplog.text
    if case == "no core":
        assert "metal-pjrt-core" in caplog.text, caplog.text
    else:
        assert str(lib) in caplog.text, caplog.text


def test_license_files_in_wheel():
    # PEP 639 license-files put both in each wheel's dist-info/licenses/
    # (setuptools >= 77); scripts/build_wheel.sh stages both for each. The
    # notices carry the MLX and metal-cpp licenses for code compiled into
    # the dylib.
    files = ["LICENSE", "THIRD_PARTY_NOTICES"]
    script = (ROOT / "scripts/build_wheel.sh").read_text()
    for path, stage in (("pyproject.toml", "PLUGIN"),
                        ("core/pyproject.toml", "CORE")):
        pyproject = _pyproject(path)
        assert pyproject["project"]["license-files"] == files, path
        assert pyproject["build-system"]["requires"] == ["setuptools>=77"], path
        staged = re.search(rf'^cp (.*) "\${stage}"/$', script, re.M)
        assert staged and set(files) <= set(staged.group(1).split()), stage
    notices = (ROOT / "THIRD_PARTY_NOTICES").read_text()
    for needle in ("Copyright © 2023 Apple Inc.",
                   "Copyright \u00a9 2024 Apple Inc.",
                   "Apache License\n                           Version 2.0"):
        assert needle in notices, needle
    # The statically linked projects (scripts/gen_third_party_notices.py):
    # every project of the manifest has its section.
    manifest = json.loads(
        (ROOT / "third_party/notices_manifest.json").read_text())
    for project in manifest["projects"]:
        assert f"\n{project['name']} ({project['url']})\n" in notices, project["name"]


def test_ci_jobs_skip_private_repos():
    # Cost guard (docs/development.md, "Continuous integration"): GitHub bills
    # macOS minutes on private repositories, so every job of every workflow
    # carries the guard and is skipped while the repository is private.
    guard = "if: ${{ !github.event.repository.private }}"
    workflows = sorted((ROOT / ".github/workflows").glob("*.y*ml"))
    assert workflows
    for wf in workflows:
        lines = wf.read_text().splitlines()
        start = lines.index("jobs:") + 1
        jobs, current = {}, None
        for line in lines[start:]:
            if line and not line[0].isspace() and not line.startswith("#"):
                break                                   # the next top-level key
            if re.match(r"^  [A-Za-z0-9_-]+:\s*$", line):
                current = line.strip()[:-1]
                jobs[current] = []
            elif current is not None:
                jobs[current].append(line.strip())
        assert jobs, wf.name
        for name, body in jobs.items():
            assert guard in body, f"{wf.name}: job {name} lacks `{guard}`"
