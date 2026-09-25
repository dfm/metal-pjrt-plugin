"""MLX equivalents of bench/jax_bench.py. Training steps use mx.compile as an
MLX user would."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from common import timeit, emit, selected

BACKEND = "mlx"
def sync(r):
    mx.eval(r)

def run(name, fn, *args, compile=True, **kw):
    if not selected(name):
        return
    try:
        cfn = mx.compile(fn) if compile else fn
        ms, best = timeit(lambda: cfn(*args), sync=sync, **kw)
        emit(BACKEND, name, ms, best)
    except Exception as e:  # noqa
        emit(BACKEND, name, float("nan"), float("nan"), {"error": f"{type(e).__name__}: {str(e).splitlines()[0][:120]}"})

x16m = mx.random.normal((16 * 1024 * 1024,)); mx.eval(x16m)
run("elementwise chain 16M", lambda x: mx.exp(x) * 0.5 + mx.sin(x) * mx.tanh(x), x16m)
x4k = mx.random.normal((4096, 4096)); mx.eval(x4k)
run("reduce rows 4096x4096", lambda x: mx.sum(x * x, axis=1), x4k)
run("reduce cols 4096x4096", lambda x: mx.sum(x * x, axis=0), x4k)
run("reduce all 16M", lambda x: mx.sum(x), x16m)
xs = mx.random.normal((8192, 1024)); mx.eval(xs)
run("softmax 8192x1024", lambda x: mx.softmax(x, axis=-1), xs)
def layernorm(x):
    m = mx.mean(x, -1, keepdims=True); v = mx.var(x, -1, keepdims=True)
    return (x - m) / mx.sqrt(v + 1e-5)
run("layernorm fwd 8192x1024", layernorm, xs)
run("layernorm fwd+bwd 8192x1024", lambda x: mx.grad(lambda v: mx.sum(layernorm(v) ** 2))(x), xs)
run("transpose 4096x4096", lambda x: x.T + 1.0, x4k)
run("cumsum 4096x4096 rows", lambda x: mx.cumsum(x, axis=1), x4k)

params = [mx.random.normal((1_000_000,)) for _ in range(50)]
grads = [p * 0.01 for p in params]
m = [mx.zeros_like(p) for p in params]; v = [mx.zeros_like(p) for p in params]
mx.eval(params, grads, m, v)
def adam(params, grads, m, v):
    b1, b2, lr, eps = 0.9, 0.999, 1e-3, 1e-8
    out = []
    for p, g, mi, vi in zip(params, grads, m, v):
        mi = b1 * mi + (1 - b1) * g; vi = b2 * vi + (1 - b2) * g * g
        out.append((p - lr * mi / (mx.sqrt(vi) + eps), mi, vi))
    return out
run("adam 50x1M params", adam, params, grads, m, v)

for n in (1024, 2048, 4096):
    a = mx.random.normal((n, n)); mx.eval(a)
    run(f"matmul f32 {n}", lambda a: a @ a, a, compile=False)
a16 = mx.random.normal((2048, 2048)).astype(mx.bfloat16); mx.eval(a16)
run("matmul bf16 2048", lambda a: a @ a, a16, compile=False)
ab = mx.random.normal((16, 512, 512)); mx.eval(ab)
run("batched matmul 16x512", lambda a: a @ a, ab, compile=False)

B, H, S, D = 8, 8, 512, 64
q = mx.random.normal((B, H, S, D)); mx.eval(q)
def attn(q, k, v):
    s = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(D)
    return mx.softmax(s, axis=-1) @ v
run("attention fwd 8x8x512x64", attn, q, q, q)
run("attention fwd+bwd", lambda q: mx.grad(lambda a: mx.sum(attn(a, a, a)))(q), q)

L, DM, NH, T, BS, VOCAB = 6, 384, 6, 256, 16, 4096
def init_gpt():
    s = 0.02
    p = {"wte": s * mx.random.normal((VOCAB, DM)), "wpe": s * mx.random.normal((T, DM)), "blocks": [], "lnf": mx.ones((DM,))}
    for i in range(L):
        p["blocks"].append({"ln1": mx.ones((DM,)), "qkv": s * mx.random.normal((DM, 3 * DM)), "proj": s * mx.random.normal((DM, DM)),
                            "ln2": mx.ones((DM,)), "fc": s * mx.random.normal((DM, 4 * DM)), "fc2": s * mx.random.normal((4 * DM, DM))})
    return p
def gpt_loss(p, idx, tgt):
    x = p["wte"][idx] + p["wpe"][:T]
    mask = mx.tril(mx.ones((T, T), dtype=mx.bool_))
    for b in p["blocks"]:
        h = layernorm(x) * b["ln1"]
        qkv = h @ b["qkv"]
        qh, kh, vh = [t.reshape(BS, T, NH, DM // NH).transpose(0, 2, 1, 3) for t in mx.split(qkv, 3, axis=-1)]
        s = (qh @ kh.transpose(0, 1, 3, 2)) / np.sqrt(DM // NH)
        s = mx.where(mask, s, -1e9)
        a = (mx.softmax(s, axis=-1) @ vh).transpose(0, 2, 1, 3).reshape(BS, T, DM)
        x = x + a @ b["proj"]
        h = layernorm(x) * b["ln2"]
        x = x + nn.gelu(h @ b["fc"]) @ b["fc2"]
    logits = (layernorm(x) * p["lnf"]) @ p["wte"].T
    logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    return -mx.mean(mx.take_along_axis(logp, tgt[..., None], axis=-1))
gp = init_gpt()
idx = mx.random.randint(0, VOCAB, (BS, T)); tgt = mx.random.randint(0, VOCAB, (BS, T))
from mlx.utils import tree_map
opt = tree_map(lambda x: mx.zeros_like(x), gp)
mx.eval(gp, idx, tgt, opt)
def gpt_step(p, opt, idx, tgt):
    loss, g = mx.value_and_grad(gpt_loss)(p, idx, tgt)
    new_opt = tree_map(lambda o, gg: 0.9 * o + 0.1 * gg, opt, g)
    new_p = tree_map(lambda pp, o: pp - 1e-3 * o, p, new_opt)
    return new_p, new_opt, loss
run("nanoGPT fwd (loss)", gpt_loss, gp, idx, tgt, iters=5)
run("nanoGPT train step", gpt_step, gp, opt, idx, tgt, iters=5)

xc = mx.random.normal((32, 32, 32, 3)); wc1 = 0.1 * mx.random.normal((32, 3, 3, 3)); wc2 = 0.1 * mx.random.normal((64, 3, 3, 32))
mx.eval(xc, wc1, wc2)
def cnn(x, w1, w2):
    h = nn.relu(mx.conv2d(x, w1, stride=1, padding=1))
    h = nn.relu(mx.conv2d(h, w2, stride=2, padding=1))
    return mx.mean(h ** 2)
run("cnn fwd 32x32x32", cnn, xc, wc1, wc2, iters=3)
run("cnn fwd+bwd", lambda x, w1, w2: mx.grad(cnn, argnums=(1, 2))(x, w1, w2), xc, wc1, wc2, iters=3)
