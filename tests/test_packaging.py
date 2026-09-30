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


def test_version_pin_matches_pyproject():
    deps = re.search(r"^dependencies = \[(.*)\]$",
                     (ROOT / "pyproject.toml").read_text(), re.M).group(1)
    assert f'"jax=={metal_pjrt_plugin._JAX_VERSION}"' in deps, deps
    assert f'"jaxlib=={metal_pjrt_plugin._JAX_VERSION}"' in deps, deps


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
    import jax
    monkeypatch.setattr(jax, "__version__", "0.0.0")
    metal_pjrt_plugin._check_versions()
    assert "found jax 0.0.0;" in caplog.text, caplog.text


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


@pytest.mark.parametrize("dangling", [False, True])
def test_missing_library_warning(monkeypatch, tmp_path, caplog, dangling):
    # The warning names the expected path, whether it is a dangling
    # bazel-bin link, the remedy and the consequence; nothing is registered.
    lib = tmp_path / "pjrt_c_api_mtl_plugin.dylib"
    if dangling:
        lib.symlink_to(tmp_path / "bazel-bin" / "gone.dylib")
    monkeypatch.setattr(metal_pjrt_plugin, "_library_candidate", lambda: lib)
    metal_pjrt_plugin.initialize()
    assert str(lib) in caplog.text, caplog.text
    assert "'mtl' platform is unavailable" in caplog.text, caplog.text
    assert "scripts/install_dev.sh" in caplog.text, caplog.text
    assert ("which does not exist" in caplog.text) == dangling, caplog.text


def test_license_files_in_wheel():
    # PEP 639 license-files put both in the wheel's dist-info/licenses/
    # (setuptools >= 77); scripts/build_wheel.sh stages both. The notices
    # carry the MLX and metal-cpp licenses for code compiled into the dylib.
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    files = ["LICENSE", "THIRD_PARTY_NOTICES"]
    assert pyproject["project"]["license-files"] == files
    assert pyproject["build-system"]["requires"] == ["setuptools>=77"]
    staged = re.search(r"^cp (.*) \"\$STAGE\"/$",
                       (ROOT / "scripts/build_wheel.sh").read_text(), re.M)
    assert staged and set(files) <= set(staged.group(1).split()), staged
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
