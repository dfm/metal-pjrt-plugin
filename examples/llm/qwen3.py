"""Qwen3 dense decoder in pure JAX: weights, prefill, decode and sampling.

One forward function serves both prefill (T prompt tokens) and decode
(T = 1): it writes the new keys and values into a fixed-size KV cache at
position `pos` and attends over the whole cache with a causal mask, so every
shape is static and each (batch, T) pair compiles once.

The layers are a Python list, unrolled at trace time, each with its own
cache arrays. A `lax.scan` over stacked layers compiles faster but costs
far more per token: with the cache as the scan's input and output, XLA
copies every layer's cache slice in and out and zero-fills a new output
cache on each call (at max_len 4096 that doubled the decode step time).
"""
import dataclasses, glob, json, mmap, os, struct

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np


@dataclasses.dataclass(frozen=True)
class Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rope_theta: float
    rms_norm_eps: float
    tie_word_embeddings: bool

    @classmethod
    def from_json(cls, path):
        """A dense Qwen3 config. Refuses what this implementation lacks:
        rope scaling (YaRN) and quantized (e.g. FP8) checkpoints."""
        with open(path) as f:
            d = json.load(f)
        if d.get("model_type") != "qwen3":
            raise ValueError(f"not a dense Qwen3 checkpoint: model_type {d.get('model_type')!r}")
        if d.get("rope_scaling"):
            raise ValueError(f"rope_scaling {d['rope_scaling']!r} is not supported")
        if d.get("quantization_config"):
            raise ValueError("quantized checkpoints (e.g. FP8) are not supported; "
                             "use the bf16 one and --quant")
        return cls(**{k.name: d[k.name] for k in dataclasses.fields(cls)})


# The Hub revisions these examples were tested with. Other repos (any dense
# Qwen3) download their latest revision.
REVISIONS = {
    "Qwen/Qwen3-0.6B": "c1899de289a04d12100db370d81485cdf75e47ca",
    "Qwen/Qwen3-1.7B": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    "Qwen/Qwen3-4B": "1cfa9a7208912126459214e8b04321603b3df60c",
}


def model_dir(repo="Qwen/Qwen3-0.6B", revision=None):
    """Local snapshot of `repo` in the Hugging Face cache (downloaded once),
    at `revision` or else the tested one in REVISIONS."""
    from huggingface_hub import snapshot_download
    return snapshot_download(repo, revision=revision or REVISIONS.get(repo),
                             allow_patterns=["*.json", "*.safetensors"])


# --- Weights ---------------------------------------------------------------

_DTYPES = {"BF16": ml_dtypes.bfloat16, "F16": np.float16, "F32": np.float32}


def read_safetensors(path):
    """{name: np.ndarray} backed by an mmap of the file (no copy).

    The safetensors layout is an 8-byte little-endian header length, a JSON
    header of {name: {dtype, shape, data_offsets}}, then the raw data. Parsed
    here because the safetensors package's numpy loader has no bfloat16.
    """
    with open(path, "rb") as f:
        buf = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    (n,) = struct.unpack("<Q", buf[:8])
    header = json.loads(buf[8:8 + n])
    header.pop("__metadata__", None)
    out = {}
    for name, info in header.items():
        start, end = info["data_offsets"]
        if info["dtype"] not in _DTYPES:
            raise ValueError(f"{path}: tensor {name} has dtype {info['dtype']}; "
                             f"only {', '.join(_DTYPES)} checkpoints are supported")
        out[name] = np.frombuffer(buf, dtype=_DTYPES[info["dtype"]],
                                  count=int(np.prod(info["shape"])),
                                  offset=8 + n + start).reshape(info["shape"])
    return out


def load_params(path, cfg, dtype=jnp.bfloat16, device=None, quant=None):
    """Parameters as a pytree: a list of per-layer dicts.

    Matrices keep the checkpoint's [out, in] layout (see `linear`), with the
    q/k/v projections concatenated into one `qkv` matrix and gate/up into
    one `gate_up`: a decode step then makes fewer, larger matrix-vector
    products, which read memory faster. `quant` stores every matrix, the
    embedding included, quantized in groups of `GROUP` inputs: "int8"
    (`quantize_int8`) or "int4" (`quantize_int4`).
    """
    t = {}
    for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        t.update(read_safetensors(f))

    put = lambda a: jax.device_put(np.asarray(a).astype(dtype), device)

    def mat(*names):
        w = np.concatenate([t[n] for n in names]) if len(names) > 1 else t[names[0]]
        if quant is None:
            return put(w)
        fn = {"int8": quantize_int8, "int4": quantize_int4}.get(quant)
        if fn is None:
            raise ValueError(f"unknown quant {quant!r}")
        # In blocks of rows: the float32 temporaries of a whole embedding
        # would take several GB.
        parts = [fn(w[i:i + 8192]) for i in range(0, w.shape[0], 8192)]
        parts = [np.concatenate(a) for a in zip(*parts)]
        names = ("q", "s") if quant == "int8" else ("q4", "s", "b")
        return {n: jax.device_put(a, device) if a.dtype.kind in "iu" else put(a)
                for n, a in zip(names, parts)}

    layers = []
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}."
        w = lambda n: p + n + ".weight"
        layers.append({
            "attn_norm": put(t[w("input_layernorm")]),
            "qkv": mat(w("self_attn.q_proj"), w("self_attn.k_proj"), w("self_attn.v_proj")),
            "q_norm": put(t[w("self_attn.q_norm")]),
            "k_norm": put(t[w("self_attn.k_norm")]),
            "o": mat(w("self_attn.o_proj")),
            "mlp_norm": put(t[w("post_attention_layernorm")]),
            "gate_up": mat(w("mlp.gate_proj"), w("mlp.up_proj")),
            "down": mat(w("mlp.down_proj")),
        })
    params = {"embed": mat("model.embed_tokens.weight"), "layers": layers,
              "norm": put(t["model.norm.weight"])}
    if not cfg.tie_word_embeddings:
        params["lm_head"] = mat("lm_head.weight")
    return params


GROUP = 64


def quantize_int8(w, group=GROUP):
    """Symmetric int8 with one scale per `group` consecutive inputs.

    Returns q [out, in // group, group] and s [out, in // group, 1] with
    w ~= q * s. One scale per whole row is too coarse: a few large inputs
    set the scale for the row and the rest lose precision (on Qwen3-0.6B
    that gave a KL divergence ~20x that of the bf16 weights).
    """
    w = np.asarray(w, np.float32)
    w = w.reshape(w.shape[0], -1, group)
    s = np.maximum(np.abs(w).max(axis=-1, keepdims=True), 1e-12) / 127.0
    q = np.clip(np.rint(w / s), -127, 127).astype(np.int8)
    return q, s


def quantize_int4(w, group=GROUP):
    """Affine 4-bit with a scale and offset per `group` inputs (as MLX does).

    Returns packed [out, in // group, group // 2] uint8 holding element j of
    a group in the low nibble of byte j and element j + group/2 in the high
    nibble, and s, b [out, in // group, 1] with w ~= q * s + b, q in 0..15.
    (Metal kernels here cannot hold 4-bit types, so the packing is by hand.)
    """
    w = np.asarray(w, np.float32)
    w = w.reshape(w.shape[0], -1, group)
    lo, hi = w.min(axis=-1, keepdims=True), w.max(axis=-1, keepdims=True)
    # The grid is ported from MLX's affine_quantize (mlx/ops.cpp;
    # https://github.com/ml-explore/mlx, Copyright (c) 2023 Apple Inc., MIT
    # License): anchor it at the group's largest-magnitude
    # weight (the offset b) and stretch the step so that zero also falls on
    # the grid. A plain min..max grid has the same RMS error but loses more
    # accuracy (on 512 tokens: KL 0.33 against float32, where this gives
    # MLX's 0.24).
    neg = np.abs(lo) > np.abs(hi)
    s = np.maximum(hi - lo, 1e-12) / 15.0
    s = np.where(neg, s, -s)
    edge = np.where(neg, lo, hi)
    q0 = np.rint(edge / s)
    s = np.where(q0 != 0, edge / np.where(q0 != 0, q0, 1), s)
    b = np.where(q0 == 0, 0.0, edge)
    q = np.clip(np.rint((w - b) / s), 0, 15).astype(np.uint8)
    half = group // 2
    return q[..., :half] | (q[..., half:] << 4), s, b


def dequantize(w, dtype):
    """The [out, in] matrix of a quantized weight dict, in `dtype`."""
    if "q4" in w:
        p = w["q4"]
        q = jnp.concatenate([p & 0xF, p >> 4], axis=-1).astype(dtype)
        wd = q * w["s"] + w["b"]
    else:
        wd = w["q"].astype(dtype) * w["s"]
    return wd.reshape(wd.shape[0], -1)


# Up to this many rows (a decode step of up to this many sequences), a
# quantized product is a reduction over the stored weights; beyond it the
# weights are dequantized for a GEMM. A reduction reads the weights once
# per row, a dequantization writes and reads them at bf16 size: on
# Qwen3-0.6B int4, 2 rows took 8.2 ms against 40.9, 16 rows 42 against 33.
ROWS_MAX = 4


def linear(x, w):
    """x @ w.T for a [out, in] matrix `w`, bf16 or quantized.

    For one row (a decode step) XLA turns a product into a multiply-reduce
    kernel rather than a GEMM. Quantized weights up to ROWS_MAX rows take
    explicit reductions (`_matvec_int8`, `_matvec_int4`) that dequantize
    inside the kernel, so the weights are read at their stored size.
    """
    if isinstance(w, dict):
        if np.prod(x.shape[:-1]) <= ROWS_MAX:
            return (_matvec_int4 if "q4" in w else _matvec_int8)(x, w)
        w = dequantize(w, x.dtype)
    return jnp.einsum("...i,oi->...o", x, w)


def _matvec_int8(x, w):
    """x @ w.T for a few rows x and int8 weights (scale per group), one
    reduction kernel per product."""
    q = w["q"]
    O, G, g = q.shape
    xg = x.reshape(-1, 1, G, g).astype(jnp.float32)                          # [N, 1, G, g]
    t = xg * q.astype(jnp.float32) * w["s"].astype(jnp.float32)            # [N, O, G, g]
    return jnp.sum(t, axis=(2, 3)).reshape(*x.shape[:-1], O).astype(x.dtype)


def _matvec_int4(x, w):
    """x @ w.T for a few rows x and int4 weights, without unpacking to a matrix.

    With w = q * s + b per group: sum_i x_i w_oi = sum_g s_og (sum_j x_gj q_ogj)
    + sum_g b_og (sum_j x_gj). Splitting x by half-group matches the packing
    (low nibbles hold the first half), so each byte contributes
    x_lo * (p & 15) + x_hi * (p >> 4), and the scale is applied once per
    group rather than per weight. Unpacking with a concatenate and
    dequantizing each weight made the kernel compute-bound (no faster than
    bf16).
    """
    p = w["q4"]
    O, G, half = p.shape
    lead = x.shape[:-1]
    xg = x.reshape(-1, 1, G, 2, half).astype(jnp.float32)                  # [N, 1, G, 2, half]
    lo = (p & 0xF).astype(jnp.float32)
    hi = (p >> 4).astype(jnp.float32)
    # Everything, the offset term b_og * sum_j x_gj included, is one
    # reduction over (group, byte): one kernel per product (as separate
    # reductions it was three).
    xlo, xhi = xg[:, :, :, 0], xg[:, :, :, 1]                               # [N, 1, G, half]
    s = w["s"].astype(jnp.float32)
    b = w["b"].astype(jnp.float32)
    t = (xlo * lo + xhi * hi) * s + (xlo + xhi) * b                        # [N, O, G, half]
    return jnp.sum(t, axis=(2, 3)).reshape(*lead, O).astype(x.dtype)


def embed(w, tokens):
    if isinstance(w, dict):
        rows = dequantize({k: v[tokens.reshape(-1)] for k, v in w.items()}, w["s"].dtype)
        return rows.reshape(*tokens.shape, -1)
    return w[tokens]


# --- Model -----------------------------------------------------------------

def init_cache(cfg, batch, max_len, dtype=jnp.bfloat16, device=None):
    """Per-layer lists of keys [batch, kv head, position, head dim] and
    values [batch, kv head, head dim, position].

    Both put the axis attention reduces over last: q . k over the head dim,
    p . v over positions. Decode's reductions then read the cache in place;
    with values stored like keys, XLA transposed each layer's value window
    first (a copy per layer per token, 10% of an int4 decode step).
    """
    k = (batch, cfg.num_key_value_heads, max_len, cfg.head_dim)
    v = (batch, cfg.num_key_value_heads, cfg.head_dim, max_len)
    z = lambda s: [jnp.zeros(s, dtype, device=device) for _ in range(cfg.num_hidden_layers)]
    return {"k": z(k), "v": z(v)}


def rms_norm(x, w, eps):
    x32 = x.astype(jnp.float32)
    x32 = x32 * jax.lax.rsqrt(jnp.mean(x32 * x32, axis=-1, keepdims=True) + eps)
    return (x32 * w.astype(jnp.float32)).astype(x.dtype)


def rope(x, positions, theta):
    """Rotary embedding, Hugging Face's rotate-half layout. x: [B, T, H, D]."""
    d = x.shape[-1]
    inv_freq = 1.0 / theta ** (jnp.arange(0, d, 2, dtype=jnp.float32) / d)
    ang = positions[:, :, None].astype(jnp.float32) * inv_freq    # [B, T, D/2]
    cos = jnp.cos(ang)[:, :, None, :]
    sin = jnp.sin(ang)[:, :, None, :]
    x32 = x.astype(jnp.float32)
    x1, x2 = x32[..., : d // 2], x32[..., d // 2:]
    out = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
    return out.astype(x.dtype)


def attention(q, k, v, mask, scale):
    """Grouped-query attention: q [B, T, KV, G, D] (G query heads share each
    KV head), k [B, KV, S, D], v [B, KV, D, S] (see `init_cache`), mask
    [T, S]. Returns [B, T, KV, G, D].

    For one query (decode) it is written as multiply-reduce rather than
    einsum: XLA then emits reductions that read a window of the cache in
    place, where a GEMM would first copy the sliced window.
    """
    if q.shape[1] == 1:
        q32 = q[:, 0, :, :, None, :].astype(jnp.float32)                 # [B, KV, G, 1, D]
        s = jnp.sum(q32 * k[:, :, None].astype(jnp.float32), axis=-1) * scale   # [B, KV, G, S]
        p = jax.nn.softmax(jnp.where(mask[0], s, -jnp.inf), axis=-1)
        a = jnp.sum(p[..., None, :] * v[:, :, None].astype(jnp.float32), axis=-1)  # [B, KV, G, D]
        return a[:, None].astype(v.dtype)
    s = jnp.einsum("btkgd,bksd->bkgts", q, k, preferred_element_type=jnp.float32) * scale
    s = jnp.where(mask, s, -jnp.inf)
    p = jax.nn.softmax(s, axis=-1).astype(v.dtype)
    return jnp.einsum("bkgts,bkds->btkgd", p, v)


def forward(params, tokens, cache, pos, cfg, *, last=None, all_logits=False,
            window=None):
    """Run `tokens` [B, T] at positions pos .. pos+T-1.

    `window` (static) limits attention to the first `window` cache slots,
    which must cover pos + T: attention then reads only that much of the
    cache rather than all max_len slots.

    Returns (logits, cache): logits [B, V] for chunk index `last` (default
    T - 1; a padded prompt passes its true last index), or [B, T, V] with
    all_logits, in float32, and the cache with this chunk's keys and values
    written at `pos`. Padding after `last` writes junk into the cache past
    the prompt, which the causal mask hides until decode overwrites it.
    """
    B, T = tokens.shape
    S = window or cache["k"][0].shape[2]
    H, KV, D = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    G = H // KV
    eps = cfg.rms_norm_eps

    x = embed(params["embed"], tokens)                              # [B, T, E]
    positions = pos + jnp.broadcast_to(jnp.arange(T), (B, T))
    # Query t (absolute position pos + t) sees cache slots s <= pos + t.
    mask = jnp.arange(S)[None, :] <= (pos + jnp.arange(T))[:, None]  # [T, S]
    scale = D ** -0.5

    def layer(x, lp, k_cache, v_cache):
        h = rms_norm(x, lp["attn_norm"], eps)
        q, k, v = jnp.split(linear(h, lp["qkv"]), [H * D, (H + KV) * D], axis=-1)
        q = q.reshape(B, T, H, D)
        k = k.reshape(B, T, KV, D)
        v = v.reshape(B, T, KV, D)
        q = rope(rms_norm(q, lp["q_norm"], eps), positions, cfg.rope_theta)
        k = rope(rms_norm(k, lp["k_norm"], eps), positions, cfg.rope_theta)
        k_cache = jax.lax.dynamic_update_slice(k_cache, k.transpose(0, 2, 1, 3), (0, 0, pos, 0))
        v_cache = jax.lax.dynamic_update_slice(v_cache, v.transpose(0, 2, 3, 1), (0, 0, 0, pos))

        a = attention(q.reshape(B, T, KV, G, D), k_cache[:, :, :S], v_cache[..., :S],
                      mask, scale).reshape(B, T, H * D)
        x = x + linear(a, lp["o"])

        h = rms_norm(x, lp["mlp_norm"], eps)
        gate, up = jnp.split(linear(h, lp["gate_up"]), 2, axis=-1)
        x = x + linear(jax.nn.silu(gate) * up, lp["down"])
        return x, (k_cache, v_cache)

    ks, vs = [], []
    for lp, k_cache, v_cache in zip(params["layers"], cache["k"], cache["v"]):
        x, (k, v) = layer(x, lp, k_cache, v_cache)
        ks.append(k)
        vs.append(v)
    if not all_logits:
        x = jax.lax.dynamic_index_in_dim(x, T - 1 if last is None else last,
                                         axis=1, keepdims=False)
    x = rms_norm(x, params["norm"], eps)
    logits = linear(x, params.get("lm_head", params["embed"]))
    return logits.astype(jnp.float32), {"k": ks, "v": vs}


# --- Sampling --------------------------------------------------------------

def sample(logits, key, *, temperature=0.0, top_k=0, top_p=1.0):
    """Next token [B] from logits [B, V]; temperature 0 is greedy."""
    if temperature < 0:
        raise ValueError(f"temperature must be >= 0, got {temperature}")
    if not 0 <= top_k <= logits.shape[-1]:
        raise ValueError(f"top_k must be in [0, {logits.shape[-1]}], got {top_k}")
    if temperature == 0.0:
        return jnp.argmax(logits, axis=-1).astype(jnp.int32)
    logits = logits / temperature
    if not top_k and top_p >= 1.0:
        return jax.random.categorical(key, logits, axis=-1).astype(jnp.int32)
    if top_k:
        vals, idx = jax.lax.top_k(logits, top_k)
    else:
        idx = jnp.argsort(-logits, axis=-1)
        vals = jnp.take_along_axis(logits, idx, axis=-1)
    if top_p < 1.0:
        # Keep the smallest prefix whose probability reaches top_p (always
        # at least the most likely token).
        probs = jax.nn.softmax(vals, axis=-1)
        before = jnp.cumsum(probs, axis=-1) - probs
        vals = jnp.where(before < top_p, vals, -jnp.inf)
    choice = jax.random.categorical(key, vals, axis=-1)
    return jnp.take_along_axis(idx, choice[:, None], axis=-1)[:, 0].astype(jnp.int32)
