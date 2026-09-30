# Copyright 2026 The metal-pjrt-plugin Authors
# SPDX-License-Identifier: Apache-2.0
"""LoRA fine-tuning of Qwen3 in pure JAX: data, adapters, loss, train step.

The base model is examples/llm/qwen3.py's, frozen in bfloat16. Adapters
follow mlx-lm's LoRALinear so that the two can be compared step by step:
for a projection y = x W^T, the adapted layer computes
y + scale * (x A) B with A [in, r] ~ U(-1/sqrt(in), 1/sqrt(in)) and B [r, out]
= 0 in float32, on every linear projection (q, k, v, o, gate, up, down) of
the last `num_layers` layers. The data pipeline reproduces mlx_lm.lora's
batches too (`batches`), including its data order and loss mask.

The adapter initialization and scaling, the batching and the loss mask
follow mlx-lm (https://github.com/ml-explore/mlx-lm, Copyright (c) 2023
Apple Inc., MIT License), reimplemented here in JAX.
"""
import dataclasses, json, math, os, sys

WIKISQL_REVISION = "886acf6d49be0dc2ee58fc3eb768d2dee1476da2"   # the one tested

import jax
import jax.numpy as jnp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "llm"))
import qwen3  # noqa: E402

PROJECTIONS = ("q", "k", "v", "o", "gate", "up", "down")


@dataclasses.dataclass(frozen=True)
class LoraConfig:
    rank: int = 8
    scale: float = 20.0
    num_layers: int = 16          # adapt the last num_layers layers (-1: all)
    keys: tuple = PROJECTIONS


# --- Data ------------------------------------------------------------------

def wikisql(split):
    """mlx-community/WikiSQL (from salesforce/WikiSQL, BSD-3-Clause), the
    dataset of mlx-lm's LoRA example, as {"prompt", "completion"} records:
    the table, columns and question, and the SQL query that answers it."""
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download
    path = snapshot_download("mlx-community/WikiSQL", repo_type="dataset",
                             revision=WIKISQL_REVISION)
    rows = pq.read_table(os.path.join(path, "data", f"{split}-00000-of-00001.parquet"))
    out = []
    for text in rows.column("text").to_pylist():
        prompt, _, completion = text.rpartition("\nA: ")
        out.append({"prompt": prompt, "completion": completion})
    return out


def tokenize(tok, record):
    """(tokens, offset): the Qwen3 chat template of the user prompt and the
    assistant reply (thinking off), and the index of the first token after
    the generation prompt, as mlx-lm's CompletionsDataset computes them.
    The loss covers the tokens from `offset` on (`batches`)."""
    head = f"<|im_start|>user\n{record['prompt']}<|im_end|>\n<|im_start|>assistant\n"
    full = head + f"<think>\n\n</think>\n\n{record['completion']}<|im_end|>\n"
    return tok.encode(full).ids, len(tok.encode(head).ids)


def batches(data, batch_size, seed=0, epochs=1, pad_to=32):
    """(tokens [B, L] int32, offsets [B], lengths [B]) per step, the batches
    mlx_lm.lora makes from the same records: consecutive groups of
    `batch_size` records (its sort by length is by the length of the raw
    record, a no-op), in the order of np.random.permutation after
    np.random.seed(seed), padded with zeros to 1 + a multiple of `pad_to`
    (mlx-lm: 32). Padding changes no loss (attention is causal and the
    padding is masked), only the number of batch shapes, each of which
    compiles once: fewer, longer shapes trade compute for compile time and
    memory (`train.py --pad-to`).

    mlx_lm.lora draws the next epoch's permutation after its validation
    passes have drawn their own, so only the first epoch's order matches.
    """
    groups = [data[i:i + batch_size] for i in range(0, len(data) - batch_size + 1, batch_size)]
    rng = np.random.RandomState(seed)
    for _ in range(epochs):
        for i in rng.permutation(len(groups)):
            yield pad(groups[i], pad_to)


def pad(group, pad_to=32):
    lengths = [len(t) for t, _ in group]
    L = 1 + pad_to * ((max(lengths) + pad_to - 1) // pad_to)
    tokens = np.zeros((len(group), L), np.int32)
    for j, (t, _) in enumerate(group):
        tokens[j, :len(t)] = t
    offsets = np.array([o for _, o in group], np.int32)
    return tokens, offsets, np.array(lengths, np.int32)


# --- Adapters --------------------------------------------------------------

def shapes(cfg):
    """{projection: (in, out)} of one layer's linear projections."""
    E, H, KV, D, F = (cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads,
                      cfg.head_dim, cfg.intermediate_size)
    return {"q": (E, H * D), "k": (E, KV * D), "v": (E, KV * D), "o": (H * D, E),
            "gate": (E, F), "up": (E, F), "down": (F, E)}


def adapted_layers(cfg, lcfg):
    n = cfg.num_hidden_layers
    return list(range(n - lcfg.num_layers if lcfg.num_layers >= 0 else 0, n))


def init_adapters(key, cfg, lcfg):
    """{layer index (str): {projection: {"a", "b"}}} in float32."""
    out = {}
    for i in adapted_layers(cfg, lcfg):
        layer = {}
        for p in lcfg.keys:
            n_in, n_out = shapes(cfg)[p]
            key, sub = jax.random.split(key)
            s = 1 / math.sqrt(n_in)
            layer[p] = {"a": jax.random.uniform(sub, (n_in, lcfg.rank), jnp.float32, -s, s),
                        "b": jnp.zeros((lcfg.rank, n_out), jnp.float32)}
        out[str(i)] = layer
    return out


def lora_delta(x, ab, scale):
    """scale * (x A) B in float32, cast to x's type (as mlx-lm does)."""
    z = (x.astype(jnp.float32) @ ab["a"]) @ ab["b"]
    return (scale * z).astype(x.dtype)


def save(path, adapters, lcfg):
    flat = {f"{i}.{p}.{w}": np.asarray(v) for i, layer in adapters.items()
            for p, ab in layer.items() for w, v in ab.items()}
    np.savez(path, __config__=json.dumps(dataclasses.asdict(lcfg)), **flat)


def load(path):
    z = np.load(path)
    lcfg = json.loads(str(z["__config__"]))
    lcfg = LoraConfig(**{**lcfg, "keys": tuple(lcfg["keys"])})
    out = {}
    for name in z.files:
        if name == "__config__":
            continue
        i, p, w = name.split(".")
        out.setdefault(i, {}).setdefault(p, {})[w] = jnp.asarray(z[name])
    return out, lcfg


def merge(params, adapters, cfg, lcfg):
    """Base params with W + scale * (A B)^T folded in (bf16), for generation
    with examples/llm's Engine."""
    order = {"qkv": ("q", "k", "v"), "o": ("o",), "gate_up": ("gate", "up"),
             "down": ("down",)}
    layers = list(params["layers"])
    for i, layer in adapters.items():
        lp = dict(layers[int(i)])
        for fused, parts in order.items():
            if not any(p in layer for p in parts):
                continue
            rows = []
            for p in parts:
                n_in, n_out = shapes(cfg)[p]
                d = (lcfg.scale * layer[p]["a"] @ layer[p]["b"]).T if p in layer \
                    else jnp.zeros((n_out, n_in), jnp.float32)
                rows.append(d)
            lp[fused] = (lp[fused].astype(jnp.float32) + jnp.concatenate(rows)).astype(
                lp[fused].dtype)
        layers[int(i)] = lp
    return {**params, "layers": layers}


# --- Model -----------------------------------------------------------------

def hidden(params, adapters, tokens, cfg, lcfg, grad_checkpoint=False):
    """Final hidden states [B, T, E] (after the last norm) of `tokens`
    [B, T] with the adapters applied: examples/llm/qwen3.forward without
    the KV cache and the output projection."""
    B, T = tokens.shape
    H, KV, D = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    G, eps = H // KV, cfg.rms_norm_eps
    positions = jnp.broadcast_to(jnp.arange(T), (B, T))
    mask = jnp.arange(T)[None, :] <= jnp.arange(T)[:, None]
    x = qwen3.embed(params["embed"], tokens)

    def layer(x, lp, ad):
        delta = lambda h, p: lora_delta(h, ad[p], lcfg.scale) if p in ad else 0

        h = qwen3.rms_norm(x, lp["attn_norm"], eps)
        q, k, v = jnp.split(qwen3.linear(h, lp["qkv"]), [H * D, (H + KV) * D], axis=-1)
        q = (q + delta(h, "q")).reshape(B, T, H, D)
        k = (k + delta(h, "k")).reshape(B, T, KV, D)
        v = (v + delta(h, "v")).reshape(B, T, KV, D)
        q = qwen3.rope(qwen3.rms_norm(q, lp["q_norm"], eps), positions, cfg.rope_theta)
        k = qwen3.rope(qwen3.rms_norm(k, lp["k_norm"], eps), positions, cfg.rope_theta)
        a = qwen3.attention(q.reshape(B, T, KV, G, D), k.transpose(0, 2, 1, 3),
                            v.transpose(0, 2, 3, 1), mask, D ** -0.5).reshape(B, T, H * D)
        x = x + qwen3.linear(a, lp["o"]) + delta(a, "o")

        h = qwen3.rms_norm(x, lp["mlp_norm"], eps)
        gate, up = jnp.split(qwen3.linear(h, lp["gate_up"]), 2, axis=-1)
        gate, up = gate + delta(h, "gate"), up + delta(h, "up")
        m = jax.nn.silu(gate) * up
        return x + qwen3.linear(m, lp["down"]) + delta(m, "down")

    # With grad_checkpoint, the backward pass recomputes each adapted layer
    # from its input instead of keeping its activations (mlx-lm's
    # --grad-checkpoint).
    ckpt = jax.checkpoint(layer)
    for i, lp in enumerate(params["layers"]):
        ad = adapters.get(str(i), {})
        x = (ckpt if grad_checkpoint and ad else layer)(x, lp, ad)
    return qwen3.rms_norm(x, params["norm"], eps)


CE_CHUNK = 32


@jax.custom_vjp
def cross_entropy(x, head, targets, weights):
    """sum(weights * -log p(targets)) for hidden states x [B, T, E] and the
    output projection `head` [V, E]; gradient with respect to x only.

    The logits are computed `CE_CHUNK` positions at a time and never kept:
    the forward pass also computes each chunk's gradient with respect to x,
    ((softmax - onehot) * weight) @ head, which is all the backward pass
    needs (the loss is the last thing computed, so its cotangent is a
    scalar). The float32 logits of a whole batch ([4, 256, 151936]) would
    take 620 MB, and as much again for their gradient.
    """
    return _ce_fwd(x, head, targets, weights)[0]


def _ce_fwd(x, head, targets, weights):
    loss, dx = 0.0, []
    for i in range(0, x.shape[1], CE_CHUNK):
        xc, tc, wc = x[:, i:i + CE_CHUNK], targets[:, i:i + CE_CHUNK], weights[:, i:i + CE_CHUNK]
        logits = qwen3.linear(xc, head).astype(jnp.float32)
        lse = jax.nn.logsumexp(logits, axis=-1)
        tgt = jnp.take_along_axis(logits, tc[..., None], axis=-1)[..., 0]
        loss = loss + jnp.sum(wc * (lse - tgt))
        onehot = jnp.arange(logits.shape[-1]) == tc[..., None]
        g = (jnp.exp(logits - lse[..., None]) - onehot) * wc[..., None]
        dx.append(jnp.einsum("btv,ve->bte", g.astype(x.dtype), head))
    return loss, jnp.concatenate(dx, axis=1)


def _ce_bwd(dx, g):
    return g * dx, None, None, None


cross_entropy.defvjp(_ce_fwd, _ce_bwd)


def loss_fn(adapters, params, tokens, offsets, lengths, cfg, lcfg, grad_checkpoint=False):
    """Mean cross-entropy over the target positions j with offset <= j <=
    length (mlx-lm's mask: the reply, plus the first padding position),
    and the number of those targets."""
    inputs, targets = tokens[:, :-1], tokens[:, 1:]
    x = hidden(params, adapters, inputs, cfg, lcfg, grad_checkpoint)
    j = jnp.arange(1, targets.shape[1] + 1)[None, :]
    mask = (j >= offsets[:, None]) & (j <= lengths[:, None])
    n = mask.sum()
    loss = cross_entropy(x, params.get("lm_head", params["embed"]), targets, mask / n)
    return loss, n


def make_step(cfg, lcfg, optimizer, grad_checkpoint=False):
    """The jitted train step: (adapters, opt_state, params, tokens, offsets,
    lengths) -> (adapters, opt_state, loss, ntokens). One compile per
    padded batch length."""
    grad = jax.value_and_grad(loss_fn, has_aux=True)

    def step(adapters, opt_state, params, tokens, offsets, lengths):
        (loss, n), g = grad(adapters, params, tokens, offsets, lengths, cfg, lcfg,
                            grad_checkpoint)
        updates, opt_state = optimizer.update(g, opt_state, adapters)
        adapters = jax.tree.map(lambda p, u: p + u, adapters, updates)
        return adapters, opt_state, loss, n

    return jax.jit(step, donate_argnums=(0, 1))


def make_eval(cfg, lcfg):
    return jax.jit(lambda adapters, params, tokens, offsets, lengths: loss_fn(
        adapters, params, tokens, offsets, lengths, cfg, lcfg))


def export(directory):
    """Write train/valid/test.jsonl ({"prompt", "completion"}) for
    mlx_lm.lora --data (mlx_baseline.py)."""
    os.makedirs(directory, exist_ok=True)
    for split in ("train", "valid", "test"):
        with open(os.path.join(directory, f"{split}.jsonl"), "w") as f:
            for r in wikisql(split):
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "export":
        sys.exit("usage: lora.py export DIR")
    export(sys.argv[2])
