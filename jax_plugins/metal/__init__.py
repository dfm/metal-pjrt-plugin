"""JAX plugin registration for the Metal PJRT plugin.

Modeled on jax_plugins/cuda/__init__.py. Not functional until the plugin
shared library exists.
"""

import logging
import os
import pathlib

logger = logging.getLogger(__name__)

_PLUGIN_BASENAME = "pjrt_c_api_metal_plugin.dylib"


def _get_library_path() -> pathlib.Path | None:
    candidate = pathlib.Path(__file__).resolve().parent / _PLUGIN_BASENAME
    if candidate.exists():
        return candidate
    return None


def _default_compilation_cache():
    """XLA compilation dominates first-call latency (Metal's own shader cache
    is system-wide already). Enable JAX's persistent cache unless configured."""
    import jax
    try:
        if jax.config.jax_compilation_cache_dir is None:
            cache = pathlib.Path(
                os.environ.get("JAX_METAL_CACHE_DIR", pathlib.Path.home() / ".cache" / "jax_metal"))
            cache.mkdir(parents=True, exist_ok=True)
            jax.config.update("jax_compilation_cache_dir", str(cache))
    except Exception as e:  # noqa: BLE001
        logger.warning("could not enable the persistent compilation cache: %s", e)


def initialize():
    import jax._src.xla_bridge as xb

    path = _get_library_path()
    if path is None:
        logger.warning("metal PJRT plugin library not found; skipping registration")
        return
    # The plugin is XLA's GPU PJRT client; these are its client-creation
    # options. The BFC pool matters even with unified memory: a fresh
    # MTLBuffer costs ~60 us/MB of page faults on first touch, so per-call
    # allocation of outputs dominated memory-bound kernels. The pool grows on
    # demand (no preallocation) up to memory_fraction of the working set.
    options = {
        "platform_name": "metal",
        "allocator": os.environ.get("JAX_METAL_ALLOCATOR", "bfc"),
        "preallocate": False,
        "memory_fraction": float(os.environ.get("JAX_METAL_MEMORY_FRACTION", "0.7")),
        "visible_devices": [0],
    }
    xb.register_plugin("metal", priority=500, library_path=str(path), options=options)
    _default_compilation_cache()
    # Lowering rules for primitives upstream only lowers on named platforms.
    from jax_plugins.metal import lowerings
    lowerings.register()
