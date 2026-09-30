# Copyright 2026 The metal-pjrt-plugin Authors
# SPDX-License-Identifier: Apache-2.0
"""Check airbench.py's training step on the default backend against CPU.

  JAX_PLATFORMS=mtl,cpu .venv/bin/python examples/cifar/check.py

(prefix it with `scripts/device_lock.py --` when other GPU jobs may run).

On one batch of 256 CIFAR-10 training images (random images if the data
is not downloaded yet), compares the logits, the loss and the gradient of
every parameter with float32 on CPU, computed in float32 and in bfloat16
on the default device, and in bfloat16 on CPU as a control. Reports the
relative error (|a - b| / |b| over the whole array) per parameter; exit
status 1 if float32 differs by more than 5e-3 anywhere (float32 itself is
up to 5e-4 from float64 on the worst-conditioned gradient), or if
bfloat16's error on the device is more than 3x bfloat16's error on CPU
for any array.
"""
import os, sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import airbench as ab  # noqa: E402


def batch(n=256):
    try:
        tx, ty, _, _ = ab.load_cifar10()
        return tx[:n], ty[:n], "CIFAR-10"
    except Exception:                      # offline: shapes are what matters
        rng = np.random.default_rng(0)
        return (rng.integers(0, 256, (n, 32, 32, 3), dtype=np.uint8),
                rng.integers(0, 10, n).astype(np.int32), "random")


def run(device, dtype, x_u8, y, params, stats):
    with jax.default_device(device):
        x = ab.normalize(jnp.asarray(x_u8), dtype)
        p, s = jax.device_put((params, stats), device)
        f = jax.jit(lambda p, s, x, y: jax.value_and_grad(
            lambda p: ab.loss_fn(p, s, x, y, dtype)[0])(p))
        loss, grads = f(p, s, x, jnp.asarray(y))
        logits = jax.jit(lambda p, s, x: ab.forward(p, s, x, True, dtype)[0])(p, s, x)
        return jax.device_get((logits, loss, grads))


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def main():
    x, y, src = batch()
    cpu, dev = jax.devices("cpu")[0], jax.devices()[0]
    with jax.default_device(cpu):
        params, stats = jax.device_get(
            ab.init_params(jax.random.key(0), ab.normalize(jnp.asarray(x), jnp.float32)))
    ref = run(cpu, jnp.float32, x, y, params, stats)
    ok, bf16 = True, {}
    # float32 tolerance: on real images the last conv layers' weight
    # gradients are sums with heavy cancellation; CPU float32 itself is
    # 5e-4 from float64 there (mtl: 1e-3), elsewhere ~1e-5.
    cases = [(dev, jnp.float32, 5e-3), (dev, jnp.bfloat16, None), (cpu, jnp.bfloat16, None)]
    for d, dtype, tol in cases:
        got = run(d, dtype, x, y, params, stats)
        errs = {"logits": rel(got[0], ref[0]), "loss": rel(got[1], ref[1])}
        flat_g, _ = jax.tree_util.tree_flatten_with_path(got[2])
        flat_r = jax.tree.leaves(ref[2])
        for (path, g), r in zip(flat_g, flat_r):
            if np.any(r):                  # the frozen whitening weight has no gradient
                errs["grad" + jax.tree_util.keystr(path)] = rel(g, r)
        worst = max(errs, key=errs.get)
        print(f"{d.platform} {jnp.dtype(dtype).name} vs cpu float32 ({src}, {len(y)} images): "
              f"loss {float(got[1]):.4f} (cpu {float(ref[1]):.4f}), worst relative error "
              f"{errs[worst]:.1e} ({worst})")
        for k, v in errs.items():
            print(f"    {k:<32} {v:.1e}")
        if tol is not None:
            ok &= errs[worst] <= tol
        else:
            bf16[d.platform] = errs
    # bfloat16 is judged against bfloat16 on CPU: the same rounding, so
    # the error should be of the same size, not identical.
    # (floored at 1e-3, so that entries both backends get almost exactly
    # right, like the loss, cannot dominate the ratio)
    ratio = max(bf16[dev.platform][k] / max(bf16["cpu"][k], 1e-3) for k in bf16["cpu"])
    print(f"bfloat16: {dev.platform} error / cpu error, worst over arrays: {ratio:.2f}")
    ok &= ratio <= 3.0
    print("OK" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
