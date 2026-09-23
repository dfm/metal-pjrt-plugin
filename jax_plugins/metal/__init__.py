"""JAX plugin registration for the Metal PJRT plugin.

Modeled on jax_plugins/cuda/__init__.py. Not functional until the plugin
shared library exists.
"""

import importlib.metadata
import logging
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
    # options. "platform" selects the plain StreamExecutor allocator (no BFC
    # pool), which is the right default for unified memory.
    options = {
        "platform_name": "metal",
        "allocator": "platform",
        "visible_devices": [0],
    }
    xb.register_plugin("metal", priority=500, library_path=str(path), options=options)
