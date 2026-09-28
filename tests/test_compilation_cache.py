"""Persistent compilation cache opt-in (metal_pjrt_plugin/__init__.py).

No GPU work (the cache-key test creates the mtl client, so it is
marked metal): .venv/bin/python -m pytest tests/test_compilation_cache.py
"""

import inspect
import os

import pytest

from jax._src import compilation_cache

import metal_pjrt_plugin
from metal_testing import run_python


def test_is_cache_used_still_has_local_platform_list():
    # Tripwire for JAX upgrades: the opt-in presents mtl backends as "gpu"
    # to this check. If upstream changes it (e.g. adds "mtl" or moves the
    # list), revisit _install_is_cache_used_wrapper.
    src = inspect.getsource(compilation_cache)
    assert 'supported_platforms = ["tpu", "gpu", "cpu", "neuron"]' in src
    assert "backend.platform in supported_platforms" in src
    assert "def is_cache_used(backend" in src


class _FakeBackend:
    def __init__(self, platform):
        self.platform = platform
        self.platform_version = "v"


def test_wrapper_presents_mtl_as_gpu_and_leaves_cpu_alone(monkeypatch):
    seen = []

    def upstream(backend):
        seen.append(backend)
        return backend.platform in ["tpu", "gpu", "cpu", "neuron"]

    monkeypatch.setattr(compilation_cache, "is_cache_used", upstream)
    metal_pjrt_plugin._install_is_cache_used_wrapper()
    wrapped = compilation_cache.is_cache_used
    assert wrapped is not upstream

    cpu = _FakeBackend("cpu")
    assert wrapped(cpu) is True
    assert seen[-1] is cpu  # CPU backend is passed through untouched

    mtl = _FakeBackend("mtl")
    assert wrapped(mtl) is True
    proxy = seen[-1]
    assert proxy.platform == "gpu"
    assert proxy.platform_version == "v"  # other attributes forwarded
    assert not hasattr(proxy, "supports_executable_serialization")

    # Installing twice does not stack wrappers.
    metal_pjrt_plugin._install_is_cache_used_wrapper()
    assert compilation_cache.is_cache_used is wrapped


def test_plugin_sets_no_cache_dir():
    # Installing the plugin must not turn on JAX's persistent cache: its
    # initialize() runs whatever JAX_PLATFORMS says (CPU only here, no GPU).
    env = {k: v for k, v in os.environ.items()
           if k not in ("JAX_COMPILATION_CACHE_DIR", "JAX_ENABLE_COMPILATION_CACHE")}
    env["JAX_PLATFORMS"] = "cpu"
    out = run_python(
        "import jax, jax._src.xla_bridge as xb\n"
        "jax.devices()\n"
        "assert 'mtl' in xb._backend_factories, 'plugin not initialized'\n"
        "print(jax.config.jax_compilation_cache_dir)\n", env)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip() == "None", out.stdout


@pytest.mark.metal  # creates the mtl client (no GPU work)
def test_cache_key_fingerprints_parsed_settings():
    # platform_version carries the compile-time settings as the passes read
    # them: METAL_PJRT_DISABLE_LAPACK unset and "0" mean the same (one key),
    # "1" does not; likewise the REWRITES list by its effect.
    def version(**env):
        base = {k: v for k, v in os.environ.items()
                if k not in ("METAL_PJRT_DISABLE_LAPACK",
                             "METAL_PJRT_DISABLE_REWRITES")}
        out = run_python("import jax\n"
                         "print(jax.devices('mtl')[0].client.platform_version)\n",
                         dict(base, JAX_PLATFORMS="mtl", **env))
        assert out.returncode == 0, out.stderr[-2000:]
        return out.stdout.strip()

    unset = version()
    assert version(METAL_PJRT_DISABLE_LAPACK="0") == unset
    assert version(METAL_PJRT_DISABLE_LAPACK="") == unset
    assert version(METAL_PJRT_DISABLE_LAPACK="1") != unset
    assert version(METAL_PJRT_DISABLE_REWRITES="scan,cubsort") == \
        version(METAL_PJRT_DISABLE_REWRITES="all")
    assert version(METAL_PJRT_DISABLE_REWRITES="bogus") == unset
