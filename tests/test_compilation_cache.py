"""Persistent compilation cache opt-in (jax_plugins/openmetal/__init__.py).

No GPU needed: .venv/bin/python -m pytest tests/test_compilation_cache.py
"""

import inspect

from jax._src import compilation_cache

import jax_plugins.openmetal as openmetal


def test_is_cache_used_still_has_local_platform_list():
    # Tripwire for JAX upgrades: the opt-in presents openmetal backends as "gpu"
    # to this check. If upstream changes it (e.g. adds "openmetal" or moves the
    # list), revisit _enable_persistent_cache.
    src = inspect.getsource(compilation_cache)
    assert 'supported_platforms = ["tpu", "gpu", "cpu", "neuron"]' in src
    assert "backend.platform in supported_platforms" in src
    assert "def is_cache_used(backend" in src


class _FakeBackend:
    def __init__(self, platform):
        self.platform = platform
        self.platform_version = "v"


def test_wrapper_presents_openmetal_as_gpu_and_leaves_cpu_alone(monkeypatch):
    seen = []

    def upstream(backend):
        seen.append(backend)
        return backend.platform in ["tpu", "gpu", "cpu", "neuron"]

    monkeypatch.setattr(compilation_cache, "is_cache_used", upstream)
    openmetal._install_is_cache_used_wrapper()
    wrapped = compilation_cache.is_cache_used
    assert wrapped is not upstream

    cpu = _FakeBackend("cpu")
    assert wrapped(cpu) is True
    assert seen[-1] is cpu  # CPU backend is passed through untouched

    mtl = _FakeBackend("openmetal")
    assert wrapped(mtl) is True
    proxy = seen[-1]
    assert proxy.platform == "gpu"
    assert proxy.platform_version == "v"  # other attributes forwarded
    assert not hasattr(proxy, "supports_executable_serialization")

    # Installing twice does not stack wrappers.
    openmetal._install_is_cache_used_wrapper()
    assert compilation_cache.is_cache_used is wrapped
