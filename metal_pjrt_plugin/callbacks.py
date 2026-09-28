"""Host (Python) callbacks on the "mtl" platform.

Makes ``jax.pure_callback``, ``jax.experimental.io_callback``,
``jax.debug.callback`` and ``jax.debug.print`` work under ``jit`` on metal.

Upstream, ``jax._src.callback.emit_python_callback`` rejects platforms other
than cpu/cuda/rocm/tpu/oneapi, and even if it did not, jaxlib only forwards
the executable's host callbacks (``FfiLoadedHostCallbacks`` FFI user data) to
the cpu/cuda/rocm/oneapi platform ids. So metal uses its own path:

* Lowering: ``emit_python_callback`` is wrapped so that, for a module lowered
  only for "mtl", the callback is registered in a process-global table
  under a fresh ``callback_id`` and a typed-FFI custom call
  ``xla_ffi_python_metal_callback`` (``has_side_effect`` as upstream, ordered
  effects threading a ``!stablehlo.token`` operand/result exactly like the
  cpu/gpu path) is emitted with that id as an attribute.
* Runtime: the handler (metal_pjrt/ffi/python_callback_ffi.cc)
  synchronizes the execution stream and calls the ctypes trampoline below,
  which looks up the id, wraps the (unified-memory, host-addressable)
  argument buffers as numpy arrays (copied, so the callee may keep them),
  calls the callable, and writes the results into the result buffers. A
  Python exception becomes an XLA error, i.e. a ``JaxRuntimeError`` at the
  call site.
* Lifetime: the wrapped callable is also added to the module's
  ``host_callbacks``, so the compiled executable owns it (jaxlib keeps it in a
  ``PyFfiLoadedHostCallback``; it never calls it on metal). The table holds
  only a weak reference; the entry disappears with the executable.
* Persistent compilation cache: ``callback_id`` values are per-process (they
  carry a random per-process salt), so an executable cached by another
  process would name a callback that does not exist here. Executables whose
  host callbacks include one of ours therefore bypass the persistent cache
  (``compiler.compile_or_get_cached`` is wrapped). Should one be loaded
  anyway, the handler fails with a clear "unknown callback id" error instead
  of calling the wrong function.

Limitations: one device (callbacks run once, not per shard); sub-byte dtypes
(int4, ...) are rejected as on upstream cpu/gpu; the callback runs on an XLA
execution thread while the stream is drained, so it must not wait on work
queued on the same metal stream after it (calling other jitted metal
functions from inside a callback is not supported).
"""

from __future__ import annotations

import ctypes
import dataclasses
import inspect
import itertools
import logging
import os
import secrets
import threading
import traceback
import weakref

import numpy as np

from metal_pjrt_plugin import PLATFORM  # "mtl"

logger = logging.getLogger(__name__)

TARGET = "xla_ffi_python_metal_callback"


class _Buffer(ctypes.Structure):
  _fields_ = [("data", ctypes.c_void_p), ("dtype", ctypes.c_int64),
              ("rank", ctypes.c_int64), ("dims", ctypes.POINTER(ctypes.c_int64))]


_TRAMPOLINE_TYPE = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_uint64, ctypes.c_int64, ctypes.POINTER(_Buffer),
    ctypes.c_int64, ctypes.POINTER(_Buffer), ctypes.c_void_p, ctypes.c_int64)


def _dtype_table():
  import ml_dtypes
  t = {1: np.bool_, 2: np.int8, 3: np.int16, 4: np.int32, 5: np.int64,
       6: np.uint8, 7: np.uint16, 8: np.uint32, 9: np.uint64, 10: np.float16,
       11: np.float32, 12: np.float64, 15: np.complex64, 16: ml_dtypes.bfloat16,
       18: np.complex128}
  for num, name in [(19, "float8_e5m2"), (28, "float8_e4m3"), (20, "float8_e4m3fn"),
                    (23, "float8_e4m3b11fnuz"), (29, "float8_e3m4"),
                    (24, "float8_e5m2fnuz"), (25, "float8_e4m3fnuz"),
                    (33, "float8_e8m0fnu")]:
    if hasattr(ml_dtypes, name):
      t[num] = getattr(ml_dtypes, name)
  return {k: np.dtype(v) for k, v in t.items()}


_DTYPES: dict[int, np.dtype] = {}

# callback_id -> weakref to the wrapped callable.
_table: dict[int, weakref.ref] = {}
_table_lock = threading.Lock()
_SALT = secrets.randbits(31) << 32
_counter = itertools.count(1)


def _register(fn) -> int:
  cb_id = _SALT | (next(_counter) & 0xFFFFFFFF)
  def _drop(_, cb_id=cb_id):
    with _table_lock:
      _table.pop(cb_id, None)
  with _table_lock:
    _table[cb_id] = weakref.ref(fn, _drop)
  return cb_id


def _view(buf: _Buffer):
  dtype = _DTYPES.get(buf.dtype)
  if dtype is None:
    raise TypeError(f"unsupported element type (XLA PrimitiveType {buf.dtype}) "
                    "for a metal host callback")
  shape = tuple(buf.dims[i] for i in range(buf.rank))
  n = int(np.prod(shape, dtype=np.int64))
  if n == 0 or not buf.data:
    return np.zeros(shape, dtype)
  raw = (ctypes.c_char * (n * dtype.itemsize)).from_address(buf.data)
  return np.frombuffer(raw, dtype).reshape(shape)


def _write_error(err_ptr, capacity, msg: str):
  data = msg.encode("utf-8", "replace")[: max(capacity - 1, 0)]
  ctypes.memmove(err_ptr, data, len(data))
  ctypes.memset(err_ptr + len(data), 0, 1)


def _trampoline(cb_id, nargs, args, nrets, rets, err_ptr, err_cap):
  try:
    with _table_lock:
      ref = _table.get(cb_id)
    fn = ref() if ref is not None else None
    if fn is None:
      raise RuntimeError(
          f"unknown metal host callback id {cb_id:#x}: the callable is gone. "
          "This happens if an executable containing a callback was loaded "
          "from a persistent compilation cache written by another process.")
    # Copy: the buffers are reused by XLA once the callback returns.
    in_vals = [np.array(_view(args[i])) for i in range(nargs)]
    outs = fn(*in_vals)
    outs = tuple(outs)
    if len(outs) != nrets:
      raise RuntimeError(f"callback returned {len(outs)} values, expected {nrets}")
    for i, out in enumerate(outs):
      dst = _view(rets[i])
      if dst.size:
        dst[...] = np.asarray(out, dtype=dst.dtype).reshape(dst.shape)
    return 0
  except BaseException as e:  # noqa: BLE001 - must not propagate into C
    try:
      msg = "".join(traceback.format_exception_only(type(e), e)).strip()
      tb = "".join(traceback.format_tb(e.__traceback__)[-3:])
      _write_error(err_ptr, err_cap, f"{msg}\n{tb}")
    except BaseException:  # noqa: BLE001
      pass
    return 1


_c_trampoline = _TRAMPOLINE_TYPE(_trampoline)  # keep alive for the process


# --------------------------------------------------------------------------
# Lowering
# --------------------------------------------------------------------------

def _metal_emit_python_callback(ctx, callback, token, operands, operand_avals,
                                result_avals, *, has_side_effect,
                                returns_token=True, partitioned=False,
                                sharding=None):
  from jax._src import config, core, dtypes, ffi
  from jax._src.interpreters import mlir
  from jax._src.sharding_impls import SdyArray, SdyArrayList

  del operand_avals, returns_token
  if partitioned and result_avals:
    raise ValueError("Partitioned callback not supported with return values.")

  def _wrapped_callback(*args):
    out_vals = callback(*args)
    if len(out_vals) != len(result_avals):
      raise RuntimeError(
          "Mismatched number of outputs from callback. "
          f"Expected: {len(result_avals)}, Actual: {len(out_vals)}")
    out_vals = tuple(dtypes.canonicalize_value(np.asarray(a)) for a in out_vals)
    for i, (out_val, out_aval) in enumerate(zip(out_vals, result_avals)):
      if out_val.shape != out_aval.shape:
        raise RuntimeError(
            f"Incorrect output shape for return value #{i}: "
            f"Expected: {out_aval.shape}, Actual: {out_val.shape}")
      if out_val.dtype != out_aval.dtype:
        raise RuntimeError(
            f"Incorrect output dtype for return value #{i}: "
            f"Expected: {out_aval.dtype}, Actual: {out_val.dtype}")
    return out_vals
  _wrapped_callback._metal_host_callback = True  # noqa: SLF001

  if token:
    operands = [token, *operands]
    # As upstream (jax/_src/callback.py): under Shardy, results besides the
    # token need a sharding for the token too.
    if (config.use_shardy_partitioner.value and sharding is not None
        and len(ctx.avals_out) > 0 and isinstance(sharding, SdyArrayList)):
      sharding = SdyArrayList((
          SdyArray(mesh_shape=(), dim_shardings=(),
                   logical_device_ids=sharding.shardings[0].logical_device_ids),
          *sharding.shardings))
    ctx = dataclasses.replace(
        ctx,
        avals_in=[core.abstract_token, *ctx.avals_in],
        avals_out=[core.abstract_token, *ctx.avals_out],
    )

  # The executable owns the callable (see module docstring).
  ctx.module_context.add_host_callback(_wrapped_callback)
  cb_id = _register(_wrapped_callback)
  result = ffi.build_ffi_lowering_function(
      TARGET, has_side_effect=has_side_effect,
  )(ctx, *operands, callback_id=np.uint64(cb_id))
  if sharding is not None:
    mlir.set_sharding(ctx.module_context, result, sharding)
  results = result.results
  if token:
    token, *results = results
  return results, token, _wrapped_callback


def _is_metal_module(ctx) -> bool:
  platforms = ctx.module_context.platforms
  return len(platforms) == 1 and platforms[0] == PLATFORM


_installed = False


def install(library_path) -> None:
  """Install the trampoline into the plugin and patch JAX's lowering. Call
  after the plugin is registered."""
  global _installed
  if _installed:
    return
  lib = ctypes.CDLL(os.fspath(library_path))
  lib.metal_pjrt_register_python_callback_trampoline.argtypes = [_TRAMPOLINE_TYPE]
  lib.metal_pjrt_register_python_callback_trampoline.restype = None
  lib.metal_pjrt_register_python_callback_trampoline(_c_trampoline)
  _DTYPES.update(_dtype_table())

  from jax._src import callback as jax_callback
  from jax._src import compiler
  from jax.interpreters import mlir as public_mlir

  upstream_emit = jax_callback.emit_python_callback

  def emit_python_callback(ctx, *args, **kwargs):
    if _is_metal_module(ctx):
      return _metal_emit_python_callback(ctx, *args, **kwargs)
    return upstream_emit(ctx, *args, **kwargs)
  emit_python_callback.__wrapped__ = upstream_emit
  emit_python_callback.__doc__ = upstream_emit.__doc__
  # Lowering rules in callback.py, debugging.py and checkify.py look the name
  # up on the module at call time; user lowering rules use the public alias.
  jax_callback.emit_python_callback = emit_python_callback
  public_mlir.emit_python_callback = emit_python_callback

  upstream_compile = compiler.compile_or_get_cached
  # Arguments by name, so a new upstream parameter passes through
  # (tests/test_jax_private_api.py pins the names used here).
  upstream_signature = inspect.signature(upstream_compile)

  def compile_or_get_cached(*args, **kwargs):
    a = upstream_signature.bind(*args, **kwargs).arguments
    if any(getattr(cb, "_metal_host_callback", False)
           for cb in a["host_callbacks"]):
      # Per-process callback ids must not go through the persistent cache.
      return compiler.backend_compile_and_load(
          a["backend"], a["computation"], a["executable_devices"],
          a["compile_options"], a["host_callbacks"])
    return upstream_compile(*args, **kwargs)
  compile_or_get_cached.__wrapped__ = upstream_compile
  compiler.compile_or_get_cached = compile_or_get_cached
  _installed = True
