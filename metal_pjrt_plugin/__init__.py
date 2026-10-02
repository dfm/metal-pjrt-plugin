# Copyright 2026 The metal-pjrt-plugin Authors
# SPDX-License-Identifier: Apache-2.0

"""metal-pjrt-plugin: run JAX on Apple GPUs through Metal.

Installing the package registers a JAX platform named "mtl" (not "metal",
which is Apple's jax-metal plugin). It is opt-in: select it with
JAX_PLATFORMS=mtl,cpu, jax.config.update("jax_platforms", "mtl,cpu") before
the first use of a device or array, or pass jax.devices("mtl") explicitly.
Nothing in this package needs to be imported or called by users; JAX finds
it through its "jax_plugins" entry point. See docs/faq.md for what works.

Implementation: modeled on jax_plugins/cuda/__init__.py, initialize()
registers the PJRT plugin dylib of the metal-pjrt-core package (a separate
wheel, released rarely; a link into bazel-bin made by scripts/install_dev.sh
in a source checkout) once its frontend ABI version matches this package's,
then lets JAX's persistent compilation cache serve it (when the user
configures a cache directory; the plugin never sets one) and installs the
lowerings and host callbacks the plugin needs.
"""

import ctypes
import dataclasses
import importlib.metadata
import itertools
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
# The lowest jax and jaxlib this package is tested with: pyproject.toml
# requires the same (tests/test_packaging.py checks). This package imports
# private jax._src modules, so older versions get a warning (and still
# load). There is no upper bound: newer versions load silently, and testing
# against JAX nightly says when a new release is needed.
_JAX_MIN = (0, 10, 0)
# The version of the private contract with metal-pjrt-core's library
# (metal_pjrt_frontend_abi_version in metal_pjrt/pjrt/metal_pjrt_api.cc):
# FFI targets and their attributes, the callback trampoline, client options,
# shared environment variables and the platform name. A library with another
# version is not registered; it stays loaded (reading the version loads it,
# and unloading XLA is not safe), which is harmless.
_CORE_ABI_VERSION = 1


def _env_flag(name: str) -> bool:
    """A boolean environment variable: unset, "", "0", "false", "no" and
    "off" (any case) are off, anything else on. The C++ twin is EnvFlag in
    metal_pjrt/runtime/env.h; both must agree (METAL_PJRT_DISABLE_LAPACK is
    read on both sides)."""
    # ASCII whitespace only, as absl::StripAsciiWhitespace in EnvFlag.
    v = os.environ.get(name, "").strip(" \t\n\v\f\r").lower()
    return v not in ("", "0", "false", "no", "off")


def _library_candidate() -> pathlib.Path | None:
    """The library in the installed metal-pjrt-core, or None without it."""
    try:
        import metal_pjrt_core
    except ImportError:
        return None
    return metal_pjrt_core.library_path()


def _get_library_path() -> pathlib.Path | None:
    candidate = _library_candidate()
    if candidate is not None and candidate.exists():
        return candidate
    return None


def _missing_library_message() -> str:
    candidate = _library_candidate()
    if candidate is None:
        why = ("the metal-pjrt-core package, which holds it, is not "
               "installed. Install it (pip install metal-pjrt-core; "
               "metal-pjrt-plugin depends on it), or run "
               "scripts/install_dev.sh in a source checkout")
    elif candidate.is_symlink():  # exists() was False: a dangling link
        why = (f"{candidate} is a link to {os.readlink(candidate)}, which does "
               "not exist (a development install whose bazel-bin output is "
               "gone). Rebuild and relink with scripts/install_dev.sh")
    else:
        why = (f"{candidate} does not exist. Reinstall metal-pjrt-core, or run "
               "scripts/install_dev.sh in a source checkout")
    return (f"metal-pjrt-plugin: the PJRT plugin library is missing, so the "
            f"'{PLATFORM}' platform is unavailable: {why}.")


def _core_abi_version(path: pathlib.Path) -> int:
    """The library's frontend ABI version; 0 for a library from before the
    packages were split, which has none."""
    fn = getattr(ctypes.CDLL(str(path)), "metal_pjrt_frontend_abi_version", None)
    if fn is None:
        return 0
    fn.restype = ctypes.c_int
    fn.argtypes = []
    return fn()


def _abi_mismatch_message(path: pathlib.Path, found: int) -> str:
    try:
        core = importlib.metadata.version("metal-pjrt-core")
    except importlib.metadata.PackageNotFoundError:
        core = "unknown"
    return (f"metal-pjrt-plugin {__version__} needs a metal-pjrt-core library "
            f"with frontend ABI version {_CORE_ABI_VERSION}, but {path} "
            f"(metal-pjrt-core {core}) has version {found}, so the "
            f"'{PLATFORM}' platform is unavailable. Install versions that "
            "match (pip install -U metal-pjrt-plugin metal-pjrt-core), or "
            "rebuild with scripts/install_dev.sh in a source checkout.")


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


def _version_tuple(version: str) -> tuple[int, ...]:
    """(major, minor, patch) of a version string; a pre-release such as
    0.12.0.dev20261001 counts as its release."""
    parts = []
    for p in version.split(".")[:3]:
        digits = "".join(itertools.takewhile(str.isdigit, p))
        parts.append(int(digits or 0))
    return tuple(parts + [0] * (3 - len(parts)))


def _jax_requirement() -> str:
    return ">=" + ".".join(map(str, _JAX_MIN))


def _check_versions():
    import jax
    import jaxlib
    found = {"jax": jax.__version__, "jaxlib": jaxlib.__version__}
    other = {k: v for k, v in found.items() if _version_tuple(v) < _JAX_MIN}
    if other:
        logger.warning(
            "metal-pjrt-plugin %s needs jax and jaxlib %s, found %s; "
            "it may fail or compute wrong results.",
            __version__, _jax_requirement(),
            ", ".join(f"{k} {v}" for k, v in other.items()))


def initialize():
    """Registers the "mtl" platform with JAX. Called by JAX's plugin
    discovery (the "jax_plugins" entry point), not by users."""
    import jax._src.xla_bridge as xb

    _check_versions()

    path = _get_library_path()
    if path is None:
        logger.warning(_missing_library_message())
        return
    try:
        abi = _core_abi_version(path)
    except OSError as e:  # wrong architecture, newer macOS, a broken build
        logger.warning(
            "metal-pjrt-plugin: the PJRT plugin library %s could not be "
            "loaded, so the '%s' platform is unavailable: %s", path, PLATFORM, e)
        return
    if abi != _CORE_ABI_VERSION:
        logger.warning(_abi_mismatch_message(path, abi))
        return
    # The plugin is XLA's GPU PJRT client; these are its client-creation
    # options. "platform" is XLA's pass-through allocator: every buffer comes
    # from the runtime's Device::Allocate, which caches freed buffers by size
    # (a fresh MTLBuffer costs ~60 us/MB of page faults on first touch),
    # releases them after ~2 s unused or on a system memory-pressure warning,
    # and refuses (RESOURCE_EXHAUSTED) beyond the process budget or when the
    # system is short of memory. The budget (METAL_PJRT_MEMORY_FRACTION
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
    # Buffer donation: JAX lowers donate_argnums only for the platforms in
    # mlir._platforms_with_donation and silently drops it (copying the
    # input) on any other. XLA's GPU client, which this plugin is, aliases
    # donated inputs to outputs; tests/test_donation.py checks it on mtl and
    # tests/test_jax_private_api.py pins the list.
    try:
        from jax._src.interpreters import mlir
        if PLATFORM not in mlir._platforms_with_donation:
            mlir._platforms_with_donation.append(PLATFORM)
    except Exception as e:  # noqa: BLE001 - never break plugin init
        logger.warning("metal-pjrt-plugin: buffer donation unavailable: %s", e)
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
