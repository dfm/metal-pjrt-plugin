"""Host-only packaging checks (no GPU)."""
import importlib.metadata
import pathlib
import re

import jax_plugins.openmetal as om

ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_version_pin_matches_pyproject():
    deps = re.search(r"^dependencies = \[(.*)\]$",
                     (ROOT / "pyproject.toml").read_text(), re.M).group(1)
    assert f'"jax=={om.JAX_VERSION}"' in deps, deps
    assert f'"jaxlib=={om.JAX_VERSION}"' in deps, deps


def test_one_dist_registers_openmetal():
    # One installed dist, openmetal_pjrt_plugin, registers the plugin: a
    # leftover older dist (jax-openmetal, jax-metal-pjrt) would load the
    # dylib a second time. scripts/install_dev.sh uninstalls those.
    old = {"jax-openmetal", "jax_openmetal", "jax-metal-pjrt", "jax_metal_pjrt"}
    ours = [ep for ep in importlib.metadata.entry_points(group="jax_plugins")
            if ep.value.startswith("jax_plugins.openmetal")
            or ep.dist.name in old]
    assert [(ep.name, ep.value, ep.dist.name) for ep in ours] == [
        ("openmetal", "jax_plugins.openmetal", "openmetal_pjrt_plugin")], ours


def test_version_warning(monkeypatch, caplog):
    import jax
    monkeypatch.setattr(jax, "__version__", "0.0.0")
    om._check_versions()
    assert "found jax 0.0.0;" in caplog.text, caplog.text
