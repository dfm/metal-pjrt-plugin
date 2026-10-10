# Copyright 2026 The jax-graft Authors
# SPDX-License-Identifier: Apache-2.0
"""Check the LoRA gradients on the default backend against float32 on CPU.

  .venv/bin/python examples/lora/gradcheck.py

(prefix it with `scripts/device_lock.py --` when other GPU jobs may run).

One batch of two training examples; adapters with random (nonzero) B so
that both A and B get gradients. Prints the loss and the relative error
||g - g_ref|| / ||g_ref|| of the gradient, over all adapters and per
projection, for the bf16 base model on the default backend and, as the
control, the same bf16 model on CPU, and the float32 model on the default
backend; exit status 1 if bf16 on the default backend is more than 3x
further from float32 than CPU bf16 is, or float32 on it differs from CPU
float32 by more than 1e-3.
"""
import os, sys

import jax
import jax.numpy as jnp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lora, qwen3  # noqa: E402


def grads(params, adapters, batch, cfg, lcfg, device):
    with jax.default_device(device):
        f = jax.jit(jax.value_and_grad(lora.loss_fn, has_aux=True), static_argnums=(5, 6))
        (loss, _), g = f(jax.device_put(adapters, device), params, *batch, cfg, lcfg)
        return float(loss), jax.tree.map(np.asarray, g)


def rel(g, ref, keep=lambda path: True):
    num = den = 0.0
    for (path, a), b in zip(jax.tree_util.tree_leaves_with_path(g), jax.tree.leaves(ref)):
        if keep(jax.tree_util.keystr(path)):
            num += float(np.sum((a.astype(np.float64) - b) ** 2))
            den += float(np.sum(b.astype(np.float64) ** 2))
    return (num / den) ** 0.5


def main():
    path = qwen3.model_dir()
    cfg = qwen3.Config.from_json(os.path.join(path, "config.json"))
    lcfg = lora.LoraConfig()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(path, "tokenizer.json"))
    batch = lora.pad([lora.tokenize(tok, r) for r in lora.wikisql("train")[:2]])
    adapters = lora.init_adapters(jax.random.key(0), cfg, lcfg)
    keys = iter(jax.random.split(jax.random.key(1), 1000))
    adapters = jax.tree_util.tree_map_with_path(
        lambda p, x: 0.01 * jax.random.normal(next(keys), x.shape)
        if jax.tree_util.keystr(p).endswith("['b']") else x, adapters)

    cpu, dev = jax.devices("cpu")[0], jax.devices()[0]
    loss_ref, ref = grads(qwen3.load_params(path, cfg, jnp.float32, cpu), adapters, batch,
                          cfg, lcfg, cpu)
    loss_cpu, g_cpu = grads(qwen3.load_params(path, cfg, jnp.bfloat16, cpu), adapters, batch,
                            cfg, lcfg, cpu)
    loss_dev, g_dev = grads(qwen3.load_params(path, cfg, jnp.bfloat16, dev), adapters, batch,
                            cfg, lcfg, dev)
    # The same computation in float32 on the device: bf16 on the two
    # backends rounds in different places over 28 layers, so bf16 vs bf16
    # need not be closer than bf16 vs float32; float32 vs float32 isolates
    # the plugin's own error.
    loss_d32, g_d32 = grads(qwen3.load_params(path, cfg, jnp.float32, dev), adapters, batch,
                            cfg, lcfg, dev)
    print(f"loss: cpu float32 {loss_ref:.5f}, cpu bf16 {loss_cpu:.5f}, "
          f"{dev.platform} bf16 {loss_dev:.5f}, {dev.platform} float32 {loss_d32:.5f}")
    for name, g in (("cpu bf16", g_cpu), (f"{dev.platform} bf16", g_dev),
                    (f"{dev.platform} f32", g_d32)):
        per = {p: rel(g, ref, lambda s, p=p: f"['{p}']" in s) for p in lora.PROJECTIONS}
        per = {p: f"{e:.1e}" for p, e in per.items()}
        print(f"{name:>9}: gradient rel. error vs float32 {rel(g, ref):.2e}, by projection {per}")
    ok = rel(g_dev, ref) <= 3 * rel(g_cpu, ref) and rel(g_d32, ref) <= 1e-3
    print("OK" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
