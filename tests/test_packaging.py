"""Host-only packaging checks (no GPU)."""
import pathlib
import re

import jax_plugins.openmetal as om

ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_version_pin_matches_pyproject():
    deps = re.search(r"^dependencies = \[(.*)\]$",
                     (ROOT / "pyproject.toml").read_text(), re.M).group(1)
    assert f'"jax=={om.JAX_VERSION}"' in deps, deps
    assert f'"jaxlib=={om.JAX_VERSION}"' in deps, deps


def test_version_warning(monkeypatch, caplog):
    import jax
    monkeypatch.setattr(jax, "__version__", "0.0.0")
    om._check_versions()
    assert "found jax 0.0.0;" in caplog.text, caplog.text
