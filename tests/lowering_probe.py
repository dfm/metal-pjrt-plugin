# Copyright 2026 The metal-pjrt-plugin Authors
# SPDX-License-Identifier: Apache-2.0

"""Lowers programs for "mtl" without a Metal device, for test_lowering.py.

Run as a child process with JAX_PLATFORMS=cpu: plugin discovery registers
the platform and the frontend's lowering rules, but no Metal client is
created, so this needs no GPU (only the dylib, for the ABI check). Prints
one JSON object: the frontend's warnings during initialization, the
backends that exist at the end, and per case either the lowered module's
custom-call targets and StableHLO ops or the error.
"""
import json
import logging
import re
import sys
import traceback

warnings = []


class _Collect(logging.Handler):
    def emit(self, record):
        if record.levelno >= logging.WARNING:
            warnings.append(record.getMessage())


logging.getLogger("metal_pjrt_plugin").addHandler(_Collect())

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax import lax  # noqa: E402
from jax.experimental import checkify, io_callback  # noqa: E402

jax.devices()  # discovery: initialize() registers "mtl" and its lowerings

f32 = lambda *s: jax.ShapeDtypeStruct(s, np.float32)  # noqa: E731
c64 = lambda *s: jax.ShapeDtypeStruct(s, np.complex64)  # noqa: E731


def _conv(x, w):
    return lax.conv_general_dilated(x, w, (1, 1), "SAME",
                                    dimension_numbers=("NHWC", "HWIO", "NHWC"))


def _debug_check(x):
    checkify.debug_check(jnp.all(x > 0), "x must be positive")
    return x + 1


def _host(x):
    return jax.pure_callback(lambda a: np.asarray(a) * 2, x, x)


def _io(x):
    return io_callback(lambda a: np.asarray(a), x, x, ordered=True)


def _print(x):
    jax.debug.print("x = {}", x)
    return x


def _scan(x):
    return lax.scan(lambda c, v: (c + v, c * v), 0.0, x)[1]


def _while(x):
    return lax.while_loop(lambda c: c[0] < 10, lambda c: (c[0] + 1, c[1] * 2), (0, x))[1]


# name -> (function, abstract arguments, jit options)
CASES = {
    "fft c64": (jnp.fft.fft, [c64(16)], {}),
    "fft2 c64": (jnp.fft.fft2, [c64(8, 16)], {}),
    "rfft f32": (jnp.fft.rfft, [f32(16)], {}),
    "irfft c64": (jnp.fft.irfft, [c64(9)], {}),
    "conv f32": (_conv, [f32(1, 8, 8, 3), f32(3, 3, 3, 4)], {}),
    "conv c64": (_conv, [c64(1, 8, 8, 3), c64(3, 3, 3, 4)], {}),
    "lu f32": (lax.linalg.lu, [f32(6, 6)], {}),
    "qr f32": (jnp.linalg.qr, [f32(6, 4)], {}),
    "eigh f32": (jnp.linalg.eigh, [f32(6, 6)], {}),
    "eigh c64": (jnp.linalg.eigh, [c64(6, 6)], {}),
    "svd f32": (jnp.linalg.svd, [f32(6, 4)], {}),
    "svd f32 values": (lambda a: jnp.linalg.svd(a, compute_uv=False), [f32(6, 4)], {}),
    "cholesky f32": (jnp.linalg.cholesky, [f32(6, 6)], {}),
    "triangular_solve f32": (
        lambda a, b: lax.linalg.triangular_solve(a, b, left_side=True, lower=True),
        [f32(6, 6), f32(6, 2)], {}),
    "solve f32": (jnp.linalg.solve, [f32(6, 6), f32(6, 2)], {}),
    "debug.print": (_print, [f32(4)], {}),
    "pure_callback": (_host, [f32(4)], {}),
    "io_callback ordered": (_io, [f32(4)], {}),
    "checkify debug_check": (_debug_check, [f32(4)], {}),
    "donation": (lambda x: x * 2, [f32(1024)], {"donate_argnums": 0}),
    # Generic rules: a broad smoke test of everyday programs.
    "sort": (jnp.sort, [f32(100)], {}),
    "argsort": (jnp.argsort, [f32(100)], {}),
    "top_k": (lambda x: lax.top_k(x, 5), [f32(100)], {}),
    "cumsum": (jnp.cumsum, [f32(100)], {}),
    "scan": (_scan, [f32(10)], {}),
    "while_loop": (_while, [f32(4)], {}),
    "gather": (lambda x, i: x[i], [f32(10, 4), jax.ShapeDtypeStruct((3,), np.int32)], {}),
    "scatter-add": (lambda x, i: x.at[i].add(1.0), [f32(10), jax.ShapeDtypeStruct((3,), np.int32)], {}),
    "random": (lambda k: jax.random.normal(jax.random.wrap_key_data(k), (8,)),
               [jax.ShapeDtypeStruct((2,), np.uint32)], {}),
    "matmul bf16": (jnp.matmul, [jax.ShapeDtypeStruct((8, 8), jnp.bfloat16)] * 2, {}),
    "grad of conv": (jax.grad(lambda x, w: jnp.sum(_conv(x, w) ** 2), argnums=(0, 1)),
                     [f32(1, 8, 8, 3), f32(3, 3, 3, 4)], {}),
    "softmax": (jax.nn.softmax, [f32(4, 16)], {}),
}


def lower(fn, args, opts):
    text = jax.jit(fn, **opts).trace(*args).lower(lowering_platforms=("mtl",)).as_text()
    return {
        "targets": sorted(set(re.findall(r"custom_call @([\w$.]+)", text))),
        "ops": sorted(set(re.findall(r"stablehlo\.(\w+)", text))),
        "complex_conv": any("stablehlo.convolution" in l and "complex" in l
                            for l in text.splitlines()),
        "aliased": "tf.aliasing_output" in text,
    }


results = {}
for name, (fn, args, opts) in CASES.items():
    try:
        results[name] = lower(fn, args, opts)
    except Exception as e:  # noqa: BLE001 - reported per case
        results[name] = {"error": f"{type(e).__name__}: {e}"[:2000],
                         "traceback": traceback.format_exc()[-4000:]}

from jax._src import xla_bridge as xb  # noqa: E402

json.dump({"jax": jax.__version__, "warnings": warnings,
           "backends": sorted(xb._backends), "results": results}, sys.stdout)
