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
    # demand (no preallocation) up to the process's budget, which the plugin
    # derives from free system memory at startup minus a reserve (unified
    # memory is shared with every other process on the machine); the
    # allocation-time guard in the runtime handles later pressure.
    options = {
        "platform_name": "metal",
        "allocator": os.environ.get("JAX_METAL_ALLOCATOR", "bfc"),
        "preallocate": False,
        "memory_fraction": float(os.environ.get("JAX_METAL_MEMORY_FRACTION", "1.0")),
        "visible_devices": [0],
    }
    xb.register_plugin("metal", priority=500, library_path=str(path), options=options)
    # Lowering rules for primitives upstream only lowers on named platforms.
    from jax_plugins.metal import lowerings
    lowerings.register()
    # Host callbacks (pure_callback, io_callback, jax.debug.*).
    try:
        from jax_plugins.metal import callbacks
        callbacks.install(path)
    except Exception as e:  # noqa: BLE001 - never break plugin init
        logger.warning("metal: host callbacks unavailable: %s", e)
