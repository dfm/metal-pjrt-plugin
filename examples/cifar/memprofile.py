# Copyright 2026 The metal-pjrt-plugin Authors
# SPDX-License-Identifier: Apache-2.0
"""Memory and pressure snapshots for airbench.py --profile-memory.

snapshot() returns a dict of:
  footprint_mb   the process's physical footprint (task_info TASK_VM_INFO,
                 what Activity Monitor calls Memory)
  pressure_sys   macOS's memory pressure level from sysctl
                 kern.memorystatus_vm_pressure_level (1 normal, 2 warn,
                 4 critical)
  and, on mtl, device 0's jax memory_stats(): live_mb (buffers in use),
  peak_mb (their peak so far), cached_mb (freed buffers the plugin keeps
  for reuse), budget_mb (the plugin's memory budget), num_allocs
  (allocations so far).
"""
import ctypes, ctypes.util

MB = 1 << 20


def plugin_stats():
    import jax
    try:
        m = jax.devices("mtl")[0].memory_stats()
    except RuntimeError:  # no mtl backend
        return {}
    if not m:
        return {}
    return {"live_mb": round(m["bytes_in_use"] / MB, 1),
            "peak_mb": round(m["peak_bytes_in_use"] / MB, 1),
            "cached_mb": round((m["pool_bytes"] - m["bytes_in_use"]) / MB, 1),
            "budget_mb": round(m["bytes_limit"] / MB), "num_allocs": m["num_allocs"]}


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
