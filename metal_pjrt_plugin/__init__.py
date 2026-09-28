"""metal-pjrt-plugin: run JAX on Apple GPUs through Metal.

Installing the package registers a JAX platform named "mtl" (not "metal",
which is Apple's jax-metal plugin). It is opt-in: select it with
JAX_PLATFORMS=mtl,cpu, jax.config.update("jax_platforms", "mtl,cpu") before
the first use of a device or array, or pass jax.devices("mtl") explicitly.
Nothing in this package needs to be imported or called by users; JAX finds
it through its "jax_plugins" entry point. See the README for what works.

Implementation: modeled on jax_plugins/cuda/__init__.py, initialize()
registers the PJRT plugin dylib next to this file (packaged by
scripts/build_wheel.sh, or a link into bazel-bin made by
scripts/install_dev.sh), then lets JAX's persistent compilation cache serve
it (when the user configures a cache directory; the plugin never sets one)
and installs the lowerings and host callbacks the plugin needs.
"""

import dataclasses
import importlib.metadata
import logging
import os
import pathlib

__all__ = ["PLATFORM", "initialize", "__version__"]

logger = logging.getLogger(__name__)

try:
    __version__ = importlib.metadata.version("metal-pjrt-plugin")
except importlib.metadata.PackageNotFoundError:  # a bare source tree
    __version__ = "0+unknown"

# The JAX platform name. It must equal MetalName() in
# third_party/xla/patches/0001-metal-pjrt-identity.patch (the PJRT client's
# platform_name, i.e. backend.platform, which JAX looks lowerings up by) and
# the PLATFORM of _lowerings.py, _linalg_lowerings.py and _callbacks.py.
PLATFORM = "mtl"
_PLUGIN_BASENAME = "pjrt_c_api_mtl_plugin.dylib"
# The jax/jaxlib the plugin is built against (pyproject.toml pins the same;
# tests/test_packaging.py checks). The dylib is built from that jaxlib's XLA
# commit, and this package imports private jax._src modules, so the pin is
# real even where pip lets another version in.
_JAX_VERSION = "0.11.2"


def _env_flag(name: str) -> bool:
    """A boolean environment variable: unset, "", "0", "false", "no" and
    "off" (any case) are off, anything else on. The C++ twin is EnvFlag in
    metal_pjrt/runtime/env.h; both must agree (METAL_PJRT_DISABLE_LAPACK is
    read on both sides)."""
    v = os.environ.get(name, "").strip().lower()
    return v not in ("", "0", "false", "no", "off")


def _library_candidate() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent / _PLUGIN_BASENAME


def _get_library_path() -> pathlib.Path | None:
    candidate = _library_candidate()
    if candidate.exists():
        return candidate
    return None


def _missing_library_message() -> str:
    candidate = _library_candidate()
    if candidate.is_symlink():  # exists() was False: a dangling link
        why = (f"{candidate} is a link to {os.readlink(candidate)}, which does "
               "not exist (a development install whose bazel-bin output is "
               "gone). Rebuild and relink with scripts/install_dev.sh")
    else:
        why = (f"{candidate} does not exist. Reinstall metal-pjrt-plugin from "
               "a wheel (which ships it), or run scripts/install_dev.sh in a "
               "source checkout")
    return (f"metal-pjrt-plugin: the PJRT plugin library is missing, so the "
            f"'{PLATFORM}' platform is unavailable: {why}.")


class _AsGpuBackend:
    """Forwards to an mtl backend but reports platform "gpu"."""

    platform = "gpu"

    def __init__(self, backend):
        self._backend = backend

    def __getattr__(self, name):
        return getattr(self._backend, name)


def _install_is_cache_used_wrapper():
    """Let JAX's persistent compilation cache serve the mtl platform.

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


def _check_versions():
    import jax
    import jaxlib
    found = {"jax": jax.__version__, "jaxlib": jaxlib.__version__}
    other = {k: v for k, v in found.items() if v != _JAX_VERSION}
    if other:
        logger.warning(
            "metal-pjrt-plugin is built for jax and jaxlib %s, found %s; it may fail "
            "or compute wrong results. Install jax==%s jaxlib==%s.",
            _JAX_VERSION, ", ".join(f"{k} {v}" for k, v in other.items()),
            _JAX_VERSION, _JAX_VERSION)


def initialize():
    """Registers the "mtl" platform with JAX. Called by JAX's plugin
    discovery (the "jax_plugins" entry point), not by users."""
    import jax._src.xla_bridge as xb

    _check_versions()

    path = _get_library_path()
    if path is None:
        logger.warning(_missing_library_message())
        return
    # The plugin is XLA's GPU PJRT client; these are its client-creation
    # options. "platform" is XLA's pass-through allocator: every buffer comes
    # from the runtime's Device::Allocate, which caches freed buffers by size
    # (a fresh MTLBuffer costs ~60 us/MB of page faults on first touch),
    # releases them after ~2 s unused or on a system memory-pressure warning,
    # and refuses (RESOURCE_EXHAUSTED) beyond the process budget or when the
    # system is short of memory. The budget (JAX_MTL_MEMORY_FRACTION
    # scales it) is applied by the runtime, so memory_fraction stays 1.
    # "platform_name" selects the StreamExecutor platform ("METAL"), not the
    # JAX/PJRT one (PLATFORM, from MetalName() in the XLA patch).
    options = {
        "platform_name": "METAL",
        "allocator": "platform",
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
    # JAX_PLATFORMS=mtl,cpu (or jax.config.update("jax_platforms", ...))
    # or use jax.devices("mtl") explicitly.
    xb.register_plugin(PLATFORM, priority=-100, library_path=str(path), options=options)
    # With JAX_PLATFORMS unset, JAX initializes every registered backend,
    # and register_plugin makes a plugin's failure fatal (fail_quietly=False):
    # every CPU-only program with this package installed would fail with
    # "Unable to initialize backend 'mtl'" wherever the Metal client cannot
    # be created. The platform is opt-in, so fail quietly (logged at INFO;
    # jax.devices("mtl") then raises with the reason). Naming it in
    # JAX_PLATFORMS still fails loudly. tests/test_jax_private_api.py pins
    # the xla_bridge behaviour this relies on.
    registration = xb._backend_factories.get(PLATFORM)
    if registration is not None:
        xb._backend_factories[PLATFORM] = dataclasses.replace(
            registration, fail_quietly=True)
    # Only makes the cache usable for mtl. The cache directory is
    # process-wide JAX config (it would turn caching on for CPU too, and this
    # runs whenever the plugin is installed), so it is left to the user.
    try:
        _install_is_cache_used_wrapper()
    except Exception as e:  # noqa: BLE001 - never break plugin init
        logger.warning("metal-pjrt-plugin: persistent compilation cache unavailable: %s", e)
    # Lowering rules for primitives upstream only lowers on named platforms.
    from metal_pjrt_plugin import _lowerings
    _lowerings.register()
    # Host callbacks (pure_callback, io_callback, jax.debug.*).
    try:
        from metal_pjrt_plugin import _callbacks
        _callbacks.install(path)
    except Exception as e:  # noqa: BLE001 - never break plugin init
        logger.warning("metal-pjrt-plugin: host callbacks unavailable: %s", e)
