"""Signed-error sweep of float32 transcendentals, Metal vs CPU.

Max ulp error hides a *bias*: an error that is small per call but has the
same sign every time adds up coherently in a long recursion (tinygp's
20000-step quasisep scan, see docs/accuracy.md). For each function and each
|x| decade this prints, per backend, the mean signed error, the mean
absolute error and the max error, in ulps of the float32 result against a
float64 numpy reference of the same (rounded) inputs.

  scripts/device_lock.py -- .venv/bin/python bench/math_bias.py [fn ...]
"""
import sys

import jax
import jax.numpy as jnp
import numpy as np

N = 1 << 18
DECADES = [(1e-4, 1e-3), (1e-3, 1e-2), (1e-2, 0.125), (0.125, 1.0), (1.0, 10.0),
           (1.4e-45, 1.1754942e-38)]  # the last row: float32 subnormals
SPECIAL = np.array([0.0, -0.0, 1e-40, -1e-40, 1e-30, -1e-30], np.float32)
FUNCS = {  # name: (jax fn, numpy f64 reference, sign of the argument)
    "exp-": (jnp.exp, np.exp, -1),
    "exp+": (jnp.exp, np.exp, +1),
    "expm1-": (jnp.expm1, np.expm1, -1),
    "sin": (jnp.sin, np.sin, +1),
    "cos": (jnp.cos, np.cos, +1),
    "log": (jnp.log, np.log, +1),
    "log2": (jnp.log2, np.log2, +1),
    "log10": (jnp.log10, np.log10, +1),
    "tanh": (jnp.tanh, np.tanh, +1),
}


def errors(dev, f, ref, x):
    got = np.asarray(jax.jit(f, device=dev)(x)).astype(np.float64)
    want = ref(x.astype(np.float64))
    ulp = np.spacing(np.abs(want).astype(np.float32)).astype(np.float64)
    return (got - want) / ulp


def main(names):
    rng = np.random.default_rng(0)
    devs = [("metal", jax.devices("openmetal")[0]), ("cpu", jax.devices("cpu")[0])]
    print(f"{'fn':7s} {'|x| range':15s} " + " ".join(
        f"{d:>5s} mean/abs/max    " for d, _ in devs))
    for name in names:
        f, ref, sign = FUNCS[name]
        for lo, hi in DECADES:
            x = (sign * np.exp(rng.uniform(np.log(lo), np.log(hi), N))).astype(
                np.float32)
            cols = []
            for _, dev in devs:
                e = errors(dev, f, ref, x)
                cols.append(f"{e.mean():+6.3f} {np.abs(e).mean():5.3f} "
                            f"{np.abs(e).max():4.2f}")
            print(f"{name:7s} {f'[{lo:g}, {hi:g})':15s} "
                  + "   ".join(cols), flush=True)
        # Zeros and subnormals: sign must match numpy, value within 1 ulp.
        got = np.asarray(jax.jit(f, device=devs[0][1])(sign * SPECIAL))
        with np.errstate(all="ignore"):
            want = ref(sign * SPECIAL).astype(np.float32)
            off = np.abs(got.astype(np.float64) - want) > np.spacing(
                np.abs(want))
        bad = ((got != want) & off | (np.signbit(got) != np.signbit(want))) & ~(
            np.isnan(got) & np.isnan(want))
        print(f"{name:7s} +-0 / subnormals: "
              + ("OK" if not bad.any() else f"MISMATCH at {SPECIAL[bad]}"))


if __name__ == "__main__":
    main(sys.argv[1:] or list(FUNCS))
