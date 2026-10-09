# Copyright 2026 The metal-pjrt-plugin Authors
# SPDX-License-Identifier: Apache-2.0
"""Memory and pressure snapshots for airbench.py --profile-memory.

snapshot() returns a dict of:
  footprint_mb   the process's physical footprint (task_info TASK_VM_INFO,
                 what Activity Monitor calls Memory)
  pressure_sys   macOS's memory pressure level from sysctl
                 kern.memorystatus_vm_pressure_level (1 normal, 2 warn,
                 4 critical)
  and, when the mtl plugin is loaded, its device 0 counters from
  metal_pjrt_memory_stats, the plugin's test hook (not a stable API; it may
  change between versions): live_mb, cached_mb, budget_mb, cache_hits,
  cache_misses (cumulative: a miss is a fresh Metal allocation),
  pressure_plugin (0 normal, 1 warn, 2 critical as the plugin sees it),
  kernels (compiled kernels cached).
"""
import ctypes, ctypes.util

MB = 1 << 20
_FIELDS = ("live", "cached", "budget", "cache_hits", "cache_misses", "pressure_plugin",
           "kernels", "kernel_msl_bytes")
_lib = None


def _plugin():
    global _lib
    if _lib is None:
        try:
            import metal_pjrt_plugin
            _lib = ctypes.CDLL(str(metal_pjrt_plugin._get_library_path()))
            _lib.metal_pjrt_memory_stats.restype = ctypes.c_int
        except Exception:
            _lib = False
    return _lib or None


def plugin_stats():
    lib = _plugin()
    if lib is None:
        return {}
    out = (ctypes.c_uint64 * 8)()
    if lib.metal_pjrt_memory_stats(0, out) != 0:
        return {}
    s = dict(zip(_FIELDS, out))
    return {"live_mb": round(s["live"] / MB, 1), "cached_mb": round(s["cached"] / MB, 1),
            "budget_mb": round(s["budget"] / MB), "cache_hits": s["cache_hits"],
            "cache_misses": s["cache_misses"], "pressure_plugin": s["pressure_plugin"],
            "kernels": s["kernels"]}


def footprint_mb():
    libc = ctypes.CDLL(ctypes.util.find_library("c"))
    buf = (ctypes.c_uint64 * 64)()
    count = ctypes.c_uint32(128)
    task = ctypes.c_uint32.in_dll(libc, "mach_task_self_")
    if libc.task_info(task, 22, ctypes.byref(buf), ctypes.byref(count)) != 0:  # TASK_VM_INFO
        return None
    return round(buf[18] / MB, 1)                                            # phys_footprint


def pressure_sys():
    libc = ctypes.CDLL(ctypes.util.find_library("c"))
    v = ctypes.c_int(0)
    size = ctypes.c_size_t(ctypes.sizeof(v))
    if libc.sysctlbyname(b"kern.memorystatus_vm_pressure_level", ctypes.byref(v),
                         ctypes.byref(size), None, 0) != 0:
        return None
    return v.value


def snapshot():
    return {"footprint_mb": footprint_mb(), "pressure_sys": pressure_sys(), **plugin_stats()}
