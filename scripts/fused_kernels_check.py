"""Numerics of the Metal FFI kernels (metal$softmax, metal$scan) against the
CPU backend, and a check that the rewriters fired.

  JAX_PLATFORMS=metal,cpu python scripts/fused_kernels_check.py
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_compilation_cache", False)
METAL = jax.devices("metal")[0]
CPU = jax.devices("cpu")[0]
FAIL = []


def check(name, fn, x, target, rtol, atol, expect_rewrite=True):
    f = jax.jit(fn)
    hlo = f.lower(jax.device_put(x, METAL)).compile().as_text()
    fired = target in hlo
    got = np.asarray(f(jax.device_put(x, METAL))).astype(np.float64)
    want = np.asarray(jax.jit(fn)(jax.device_put(x, CPU))).astype(np.float64)
    ok = np.allclose(got, want, rtol=rtol, atol=atol, equal_nan=True)
    err = np.nanmax(np.abs(got - want)) if got.size else 0.0
    status = "ok " if ok else "BAD"
    print(f"{status} {name:44s} rewritten={fired!s:5s} max_abs_err={err:.3g}")
    if not ok or fired != expect_rewrite:
        FAIL.append(name)


def host(a, dtype):
    # Convert on the host: inputs are created with numpy/ml_dtypes so no
    # device-side f32->bf16 convert runs.
    return np.asarray(a).astype(jnp.dtype(dtype))


def main() -> int:
    rng = np.random.default_rng(0)
    tol = {jnp.float32: (1e-5, 1e-6), jnp.float16: (1e-2, 1e-3),
           jnp.bfloat16: (2e-2, 1e-2)}
    for dtype in (jnp.float32, jnp.float16, jnp.bfloat16):
        for shape in [(7,), (64, 1024), (33, 1000), (4, 3, 17), (8, 16384),
                      (2, 4097), (128, 1), (4, 16385)]:
            x = host(rng.standard_normal(shape) * 4, dtype)
            rt, at = tol[dtype]
            nm = f"{jnp.dtype(dtype).name} {shape}"
            # n == 1 is simplified away by XLA; n > 16384 is not rewritten.
            fire = 1 < shape[-1] <= 16384
            check(f"softmax {nm}", lambda x: jax.nn.softmax(x, axis=-1), x,
                  "metal$softmax", rt, at, fire)
            check(f"log_softmax {nm}", lambda x: jax.nn.log_softmax(x, axis=-1),
                  x, "metal$softmax", rt, max(at, rt * 10), fire)
    x = rng.standard_normal((16, 256)).astype(np.float32)
    x[0, 3] = -np.inf
    x[1, :] = -np.inf
    x[2, 5] = np.nan
    check("softmax with -inf/nan rows", lambda x: jax.nn.softmax(x, -1), x,
          "metal$softmax", 1e-5, 1e-6)

    for dtype in (jnp.float32, jnp.float16, jnp.bfloat16, jnp.int32):
        for shape in [(4096, 64), (8, 1000), (3, 4097), (5, 3, 130), (100000,),
                      (2, 3)]:
            if shape == (100000,) and dtype in (jnp.float16, jnp.bfloat16):
                shape = (3000,)  # keep the half-precision sums in range
            if dtype == jnp.int32:
                x = host(rng.integers(-3, 4, shape), dtype)
            else:
                x = host(rng.standard_normal(shape) * 0.5 + 1.0, dtype)
            nm = f"{jnp.dtype(dtype).name} {shape}"
            n = shape[-1]
            ftol = {jnp.float32: 1e-4, jnp.float16: 2e-2, jnp.bfloat16: 5e-2,
                    jnp.int32: 0}[dtype]
            for rev in (False, True):
                check(f"cumsum rev={rev} {nm}",
                      lambda x: jax.lax.cumsum(x, axis=x.ndim - 1, reverse=rev),
                      x, "metal$scan", ftol, ftol * np.sqrt(n))
                check(f"cummax rev={rev} {nm}",
                      lambda x: jax.lax.cummax(x, axis=x.ndim - 1, reverse=rev),
                      x, "metal$scan", 0, 0)
                check(f"cummin rev={rev} {nm}",
                      lambda x: jax.lax.cummin(x, axis=x.ndim - 1, reverse=rev),
                      x, "metal$scan", 0, 0)
            if n <= 1000 and dtype != jnp.int32:
                y = host(1.0 + 0.001 * rng.standard_normal(shape), dtype)
                check(f"cumprod {nm}",
                      lambda x: jax.lax.cumprod(x, axis=x.ndim - 1), y,
                      "metal$scan", ftol * 10, ftol)
    xi = rng.integers(-3, 4, (16, 300)).astype(np.int32)
    check("cumprod int32 (wraps)", lambda x: jnp.cumprod(x, axis=1), xi,
          "metal$scan", 0, 0)
    xn = rng.standard_normal((4, 200)).astype(np.float32)
    xn[1, 50] = np.nan
    check("cummax with nan", lambda x: jax.lax.cummax(x, axis=1), xn,
          "metal$scan", 0, 0)
    print("FAILED:" if FAIL else "all ok", FAIL)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
