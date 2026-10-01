# Copyright 2026 The metal-pjrt-plugin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tripwires for the private JAX APIs the plugin calls or replaces
(metal_pjrt_plugin). The pinned jax version is checked at import
(test_packaging.py); these fail when a JAX upgrade changes a signature or
removes a symbol, so the plugin's use of it can be re-checked. Parameter
names, kinds and defaults are compared; annotations are not.

No GPU needed: .venv/bin/python -m pytest tests/test_jax_private_api.py
"""
import dataclasses
import inspect
import os
import subprocess
import sys

import pytest
from jax._src import (callback, checkify, compilation_cache, compiler, config,
                      core, debugging, dispatch, dtypes, ffi)
from jax._src import xla_bridge as xb
from jax._src.interpreters import mlir
from jax._src.lax import convolution as lax_convolution
from jax._src.lax import fft as lax_fft
from jax._src.lax import linalg as ll
from jax._src.sharding_impls import SdyArray, SdyArrayList
from jax._src.tpu.linalg import eigh as tpu_eigh
from jax._src.tpu.linalg import svd as tpu_svd
from jax.interpreters import mlir as public_mlir

from metal_pjrt_plugin import _callbacks as callbacks


def params(fn):
    sig = inspect.signature(inspect.unwrap(fn))
    sig = sig.replace(
        parameters=[p.replace(annotation=inspect.Parameter.empty)
                    for p in sig.parameters.values()],
        return_annotation=inspect.Signature.empty)
    return str(sig)


SIGNATURES = {
    # _callbacks.py: replaced (emit_python_callback also on the public alias)
    # and called.
    "callback.emit_python_callback": (
        callback.emit_python_callback,
        "(ctx, callback, token, operands, operand_avals, result_avals, *, "
        "has_side_effect, returns_token=True, partitioned=False, "
        "sharding=None)"),
    "compiler.compile_or_get_cached": (
        compiler.compile_or_get_cached,
        "(backend, computation, devices, compile_options, host_callbacks, "
        "executable_devices, pgle_profiler=None)"),
    "compiler.backend_compile_and_load": (
        compiler.backend_compile_and_load,
        "(backend, module, executable_devices, options, host_callbacks)"),
    "ffi.build_ffi_lowering_function": (
        ffi.build_ffi_lowering_function,
        "(call_target_name, *, operand_layouts=None, result_layouts=None, "
        "backend_config=None, skip_ffi_layout_processing=False, "
        "**lowering_args)"),
    "mlir.set_sharding": (mlir.set_sharding, "(ctx, op, sharding)"),
    "mlir.ModuleContext.add_host_callback": (
        mlir.ModuleContext.add_host_callback, "(self, host_callback)"),
    "SdyArray": (
        SdyArray,
        "(*, mesh_shape, dim_shardings, logical_device_ids=None, "
        "replicated_axes=frozenset(), unreduced_axes=frozenset(), "
        "reduction_op=None)"),
    "SdyArrayList": (SdyArrayList, "(shardings)"),
    # __init__.py
    "compilation_cache.is_cache_used": (
        compilation_cache.is_cache_used, "(backend)"),
    "xb.register_plugin": (
        xb.register_plugin,
        "(plugin_name, *, priority=400, library_path=None, options=None, "
        "c_api=None, factory=None, make_topology=None)"),
    # _lowerings.py
    "mlir.register_lowering": (
        mlir.register_lowering,
        "(prim, rule, platform=None, inline=True, cacheable=True)"),
    "mlir.lower_fun": (mlir.lower_fun, "(fun, multiple_results=True)"),
    "checkify.check_lowering_rule_unsupported": (
        checkify.check_lowering_rule_unsupported, "(*a, debug, **k)"),
    "lax_convolution._conv_general_dilated_lower": (
        lax_convolution._conv_general_dilated_lower,
        "(ctx, lhs, rhs, *, window_strides, padding, lhs_dilation, "
        "rhs_dilation, dimension_numbers, feature_group_count, "
        "batch_group_count, precision, preferred_element_type, out_sharding, "
        "expand_complex_convolutions=False, **unused_kwargs)"),
    "dispatch._tpu_gpu_device_put_lowering": (
        dispatch._tpu_gpu_device_put_lowering,
        "(ctx, *xs, devices, srcs, copy_semantics)"),
    "debugging.debug_callback_lowering": (
        debugging.debug_callback_lowering,
        "(ctx, *args, effect, partitioned, callback, **params)"),
    "debugging.debug_print_lowering_rule": (
        debugging.debug_print_lowering_rule,
        "(ctx, *dyn_args, fmt, ordered, partitioned, in_tree, static_args, "
        "np_printoptions, has_placeholders, logging_record)"),
    # _linalg_lowerings.py: the fallbacks it calls with these arguments.
    "core.is_constant_shape": (core.is_constant_shape, "(s)"),
    "ll._linalg_ffi_lowering": (
        ll._linalg_ffi_lowering,
        "(target_name, avals_in=None, avals_out=None, "
        "operand_output_aliases=None, column_major=True, "
        "num_non_batch_dims=2, batch_partitionable=True)"),
    "ll._lu_python": (ll._lu_python, "(x)"),
    "ll._geqrf_lowering_rule": (ll._geqrf_lowering_rule, "(ctx, operand)"),
    "ll._householder_product_lowering": (
        ll._householder_product_lowering, "(ctx, a, taus)"),
    "tpu_eigh._eigh_tpu_lowering": (
        tpu_eigh._eigh_tpu_lowering,
        "(ctx, operand, *, lower, sort_eigenvalues, subset_by_index, "
        "algorithm)"),
    "tpu_svd._svd_tpu_lowering_rule": (
        tpu_svd._svd_tpu_lowering_rule,
        "(ctx, operand, *, full_matrices, compute_uv, subset_by_index, "
        "algorithm=None)"),
}


@pytest.mark.parametrize("name", list(SIGNATURES))
def test_signature(name):
    fn, want = SIGNATURES[name]
    assert params(fn) == want, f"{name} changed: re-check the plugin's use"


def test_metal_emit_python_callback_matches_upstream():
    # The mtl branch receives the same arguments as upstream's.
    assert (params(callbacks._metal_emit_python_callback)
            == params(callback.emit_python_callback))


def test_symbols_exist():
    # Used without a signature to pin.
    assert isinstance(core.abstract_token, core.AbstractToken)
    assert isinstance(config.use_shardy_partitioner.value, bool)
    assert callable(dtypes.canonicalize_value)
    assert public_mlir.emit_python_callback is not None
    assert {ll.EighImplementation.QR, ll.SvdAlgorithm.DEFAULT,
            ll.SvdAlgorithm.DIVIDE_AND_CONQUER}
    assert {lax_fft.FftType.RFFT, lax_fft.FftType.IRFFT, lax_fft.FftType.IFFT}
    prims = (ll.lu_p, ll.geqrf_p, ll.householder_product_p, ll.eigh_p,
             ll.svd_p, lax_fft.fft_p, checkify.check_p,
             debugging.debug_callback_p, debugging.debug_print_p,
             lax_convolution.conv_general_dilated_p, dispatch.device_put_p)
    assert [p.name for p in prims] == [
        "lu", "geqrf", "householder_product", "eigh", "svd", "fft", "check",
        "debug_callback", "debug_print", "conv_general_dilated", "device_put"]


def test_backend_init_failure_is_quiet_unless_selected():
    # __init__.py sets fail_quietly on the mtl registration: with
    # JAX_PLATFORMS unset JAX initializes every registered backend, and a
    # failing plugin must not break CPU-only programs. xla_bridge.backends()
    # takes fail_quietly from the registration only then; a platform named
    # in JAX_PLATFORMS always fails loudly.
    assert "fail_quietly" in {f.name for f in dataclasses.fields(
        xb.BackendRegistration)}
    src = inspect.getsource(xb.backends)
    assert "registration.fail_quietly" in src, src
    assert "fail_quietly_list = [False] * len(platforms)" in src, src
    xb._discover_and_register_pjrt_plugins()
    assert xb._backend_factories["mtl"].fail_quietly is True


def test_platforms_with_donation():
    # __init__.py appends "mtl" to this list: lower_jaxpr_to_module lowers
    # donate_argnums only for the platforms in it and drops the donation
    # (copying) on any other.
    assert isinstance(mlir._platforms_with_donation, list)
    assert {"cpu", "cuda"} <= set(mlir._platforms_with_donation)
    src = inspect.getsource(mlir.lower_jaxpr_to_module)
    assert "if p in _platforms_with_donation" in src, src


# A child whose mtl client creation fails (simulated before plugin
# discovery; no Metal device is created).
FAILING_CLIENT = r"""
import jax._src.xla_bridge as xb
upstream = xb.make_pjrt_c_api_client
def make(name, *a, **k):
    if name == "mtl":
        raise RuntimeError("simulated client failure")
    return upstream(name, *a, **k)
xb.make_pjrt_c_api_client = make
import jax, jax.numpy as jnp
try:
    print(jax.default_backend(), float(jnp.arange(4.0).sum()))
except RuntimeError as e:
    print("RAISED", e)
"""


@pytest.mark.parametrize("platforms", [None, "mtl,cpu"])
def test_failing_client(platforms):
    env = {k: v for k, v in os.environ.items() if k != "JAX_PLATFORMS"}
    if platforms is not None:
        env["JAX_PLATFORMS"] = platforms
    out = subprocess.run([sys.executable, "-c", FAILING_CLIENT], env=env,
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-3000:]
    if platforms is None:
        assert out.stdout.strip() == "cpu 6.0", (out.stdout, out.stderr[-3000:])
    else:
        assert "RAISED Unable to initialize backend 'mtl'" in out.stdout, out.stdout
        assert "simulated client failure" in out.stdout, out.stdout


def test_cache_bypass_only_for_mtl_callbacks():
    # The compile_or_get_cached wrapper (_callbacks.py) checks the platform
    # first and leaves anything it cannot bind to upstream, so it cannot
    # break another backend's compile.
    sig = inspect.signature(inspect.unwrap(compiler.compile_or_get_cached))
    ours = lambda: None  # noqa: E731
    ours._metal_host_callback = True
    other = lambda: None  # noqa: E731
    mtl = type("B", (), {"platform": "mtl"})()
    cpu = type("B", (), {"platform": "cpu"})()
    bypass = lambda *a, **k: callbacks._cache_bypass_arguments(sig, a, k)  # noqa: E731
    assert bypass(mtl, "hlo", [], "opts", [ours], []) is not None
    assert bypass(mtl, "hlo", [], "opts", host_callbacks=[other, ours],
                  executable_devices=[])["computation"] == "hlo"
    assert bypass(mtl, "hlo", [], "opts", [other], []) is None
    assert bypass(cpu, "hlo", [], "opts", [ours], []) is None
    assert bypass() is None
    assert bypass(mtl, "hlo", [], "opts", [ours], [], None, "extra") is None
