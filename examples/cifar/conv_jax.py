# Copyright 2026 The metal-pjrt-plugin Authors
# SPDX-License-Identifier: Apache-2.0
"""airbench94's convolutions, one at a time: forward, input and weight gradients.

  JAX_PLATFORMS=mtl,cpu .venv/bin/python examples/cifar/conv_jax.py

(prefix it with `scripts/device_lock.py --` when other GPU jobs may run;
conv_torch.py is the same measurement in PyTorch on MPS.)

bf16, batch 1024, NHWC / HWIO, stride 1, the layer shapes of airbench.py's
network: the 2x2 VALID whitening convolution and the six 3x3 SAME ones.
Each operation is its own jitted program; prints the median ms of 10
calls after 3 warmup calls, and TFLOPS.
"""
import json, time

import jax
import jax.numpy as jnp
import numpy as np

SHAPES = [("whiten", 32, 3, 24, 2, "VALID"), ("g1c1", 31, 24, 64, 3, "SAME"),
          ("g1c2", 15, 64, 64, 3, "SAME"), ("g2c1", 15, 64, 256, 3, "SAME"),
          ("g2c2", 7, 256, 256, 3, "SAME"), ("g3c1", 7, 256, 256, 3, "SAME"),
          ("g3c2", 3, 256, 256, 3, "SAME")]
N = 1024


def median_ms(f, *args):
    for _ in range(3):
        jax.block_until_ready(f(*args))
    ts = []
    for _ in range(10):
        t0 = time.perf_counter()
        jax.block_until_ready(f(*args))
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts))


def main():
    for name, hw, ci, co, k, pad in SHAPES:
        x = jnp.ones((N, hw, hw, ci), jnp.bfloat16) * 0.1
        w = jnp.ones((k, k, ci, co), jnp.bfloat16) * 0.01
        conv = lambda x, w, pad=pad: jax.lax.conv_general_dilated(
            x, w, (1, 1), pad, dimension_numbers=("NHWC", "HWIO", "NHWC"))
        fwd = jax.jit(conv)
        y = fwd(x, w)
        dgrad = jax.jit(lambda x, w, g: jax.vjp(lambda x: conv(x, w), x)[1](g))
        wgrad = jax.jit(lambda x, w, g: jax.vjp(lambda w: conv(x, w), w)[1](g))
        flops = 2 * N * y.shape[1] * y.shape[2] * co * ci * k * k
        tf, td, tw = median_ms(fwd, x, w), median_ms(dgrad, x, w, y), median_ms(wgrad, x, w, y)
        print(json.dumps({"impl": f"jax-{jax.devices()[0].platform}", "conv": name,
                          "fwd_ms": round(tf, 2), "dgrad_ms": round(td, 2), "wgrad_ms": round(tw, 2),
                          "fwd_tflops": round(flops / tf / 1e9, 2),
                          "wgrad_tflops": round(flops / tw / 1e9, 2)}), flush=True)


if __name__ == "__main__":
    main()
