"""JAX plugin registration for the Metal PJRT plugin.

Modeled on jax_plugins/cuda/__init__.py: registers the PJRT plugin dylib
(linked next to this file by scripts/install_dev.sh) under platform
"openmetal" (not "metal", which is Apple's jax-metal plugin),
then opts it into the persistent compilation cache and installs the
lowerings and host callbacks the plugin needs.
"""

import logging
import os
import pathlib

logger = logging.getLogger(__name__)

# The JAX platform name. It must equal MetalName() in
# third_party/xla/patches/0001-metal-pjrt-identity.patch (the PJRT client's
# platform_name, i.e. backend.platform, which JAX looks lowerings up by) and
# the PLATFORM of lowerings.py, linalg_lowerings.py and callbacks.py.
PLATFORM = "openmetal"
_PLUGIN_BASENAME = "pjrt_c_api_openmetal_plugin.dylib"


def _get_library_path() -> pathlib.Path | None:
    candidate = pathlib.Path(__file__).resolve().parent / _PLUGIN_BASENAME
    if candidate.exists():
        return candidate
    return None


class _AsGpuBackend:
    """Forwards to an openmetal backend but reports platform "gpu"."""

    platform = "gpu"

    def __init__(self, backend):
        self._backend = backend

    def __getattr__(self, name):
        return getattr(self._backend, name)


def _install_is_cache_used_wrapper():
    """Let JAX's persistent compilation cache serve the openmetal platform.

    jax._src.compilation_cache.is_cache_used only accepts the platforms in a
    local list (tpu/gpu/cpu/neuron; tests/test_compilation_cache.py trips if
    that changes). Metal executables serialize like GPU ones (the plugin drops
    the PJRT ABI-version extension, see metal_pjrt_api.cc), so ask with the
    backend presented as "gpu"; everything else, including the one-shot
    bookkeeping, is upstream's. The cache key hashes platform_version, which
    encodes the plugin build and its compile-time settings.
    """
    from jax._src import compilation_cache

    upstream = compilation_cache.is_cache_used
    if getattr(upstream, "_metal_wrapped", False):
        return

    def is_cache_used(backend):
        if getattr(backend, "platform", None) == PLATFORM:
            backend = _AsGpuBackend(backend)
        return upstream(backend)

    is_cache_used._metal_wrapped = True
    compilation_cache.is_cache_used = is_cache_used


def _enable_persistent_cache():
    """Opt in, and unless a cache directory is configured
    (jax_compilation_cache_dir / JAX_COMPILATION_CACHE_DIR) use
    ~/.cache/openmetal/compilation_cache. jax_enable_compilation_cache=False
    turns it off."""
    import jax

    _install_is_cache_used_wrapper()
    if jax.config.jax_compilation_cache_dir is None:
        cache = pathlib.Path.home() / ".cache" / "openmetal" / "compilation_cache"
        jax.config.update("jax_compilation_cache_dir", str(cache))


def initialize():
    import jax._src.xla_bridge as xb

    path = _get_library_path()
    if path is None:
        logger.warning("openmetal PJRT plugin library not found; skipping registration")
        return
    # The plugin is XLA's GPU PJRT client; these are its client-creation
    # options. "platform" is XLA's pass-through allocator: every buffer comes
    # from the runtime's Device::Allocate, which caches freed buffers by size
    # (a fresh MTLBuffer costs ~60 us/MB of page faults on first touch),
    # releases them after ~2 s unused or on a system memory-pressure warning,
    # and refuses (RESOURCE_EXHAUSTED) beyond the process budget or when the
    # system is short of memory. JAX_OPENMETAL_ALLOCATOR=bfc selects XLA's BFC
    # pool instead, which never returns memory to the system. The budget
    # (JAX_OPENMETAL_MEMORY_FRACTION scales it) is applied by the runtime, so
    # memory_fraction stays 1.
    # "platform_name" selects the StreamExecutor platform ("METAL"), not the
    # JAX/PJRT one (PLATFORM, from MetalName() in the XLA patch).
    options = {
        "platform_name": "METAL",
        "allocator": os.environ.get("JAX_OPENMETAL_ALLOCATOR", "platform"),
        "preallocate": False,
        "memory_fraction": 1.0,
        "visible_devices": [0],
        # Unified memory: H2D/D2H copy straight between the numpy array and
        # the MTLBuffer. Staging through XLA's pinned-host BFC pool would add
        # a memcpy and a pool that never shrinks. (XLA already skipped
        # staging because the executor reports every foreign pointer as host
        # memory, which IsHostMemoryPinned takes as pinned; this makes it
        # explicit.)
        "should_stage_host_to_device_transfers": False,
    }
    # Opt-in: a priority below CPU's (0), so installing the plugin does not
    # change JAX's default backend. Select it with
    # JAX_PLATFORMS=openmetal,cpu (or jax.config.update("jax_platforms", ...))
    # or use jax.devices("openmetal") explicitly.
    xb.register_plugin(PLATFORM, priority=-100, library_path=str(path), options=options)
    try:
        _enable_persistent_cache()
    except Exception as e:  # noqa: BLE001 - never break plugin init
        logger.warning("openmetal: persistent compilation cache unavailable: %s", e)
    # Lowering rules for primitives upstream only lowers on named platforms.
    from jax_plugins.openmetal import lowerings
    lowerings.register()
    # Host callbacks (pure_callback, io_callback, jax.debug.*).
    try:
        from jax_plugins.openmetal import callbacks
        callbacks.install(path)
    except Exception as e:  # noqa: BLE001 - never break plugin init
        logger.warning("openmetal: host callbacks unavailable: %s", e)
