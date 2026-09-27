"""JAX benchmark workloads. Run under any backend:
  JAX_PLATFORMS=openmetal python bench/jax_bench.py
Cases are chosen to separate memory-bound fusion, reductions, GEMM, linear
algebra, and a small end-to-end transformer training step. Rows carry the
commit, versions and knobs (common.run_info), TFLOPS where the flop count is
standard, and on metal with METAL_PJRT_TRACE=1 the GPU time (common.gpu_ms).
"""
import functools, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import jax, jax.numpy as jnp
from common import timeit, emit, gpu_ms, selected

BACKEND = os.environ.get("BENCH_LABEL", jax.default_backend())
sync = lambda r: jax.block_until_ready(r)
f32 = jnp.float32
key = jax.random.PRNGKey(0)


def run(name, fn, *args, flops=None, **kw):
    if not selected(name):
        return
    try:
        jfn = jax.jit(fn)
        ms, best = timeit(lambda: jfn(*args), sync=sync, **kw)
        gpu = gpu_ms(lambda: jfn(*args), sync=sync) if jax.default_backend() == "openmetal" else None
        emit(BACKEND, name, ms, best, gpu=gpu, flops=flops)
    except Exception as e:  # noqa
        emit(BACKEND, name, float("nan"), float("nan"), {"error": f"{type(e).__name__}: {str(e).splitlines()[0][:120]}"})


# --- memory-bound fusions ---
x16m = jax.random.normal(key, (16 * 1024 * 1024,), f32)
run("elementwise chain 16M", lambda x: jnp.exp(x) * 0.5 + jnp.sin(x) * jnp.tanh(x), x16m)
x4k = jax.random.normal(key, (4096, 4096), f32)
run("reduce rows 4096x4096", lambda x: jnp.sum(x * x, axis=1), x4k)
run("reduce cols 4096x4096", lambda x: jnp.sum(x * x, axis=0), x4k)
run("reduce all 16M", lambda x: jnp.sum(x), x16m)
xs = jax.random.normal(key, (8192, 1024), f32)
run("softmax 8192x1024", lambda x: jax.nn.softmax(x, axis=-1), xs)
def layernorm(x):
    m = x.mean(-1, keepdims=True); v = x.var(-1, keepdims=True)
    return (x - m) / jnp.sqrt(v + 1e-5)
run("layernorm fwd 8192x1024", layernorm, xs)
run("layernorm fwd+bwd 8192x1024", lambda x: jax.grad(lambda v: jnp.sum(layernorm(v) ** 2))(x), xs)
run("transpose 4096x4096", lambda x: x.T + 1.0, x4k)
run("cumsum 4096x4096 rows", lambda x: jnp.cumsum(x, axis=1), x4k)

# --- optimizer step over many parameters (horizontal fusion) ---
params = [jax.random.normal(jax.random.fold_in(key, i), (1_000_000,), f32) for i in range(50)]
grads = [p * 0.01 for p in params]
m = [jnp.zeros_like(p) for p in params]; v = [jnp.zeros_like(p) for p in params]
def adam(params, grads, m, v):
    b1, b2, lr, eps = 0.9, 0.999, 1e-3, 1e-8
    out = []
    for p, g, mi, vi in zip(params, grads, m, v):
        mi = b1 * mi + (1 - b1) * g; vi = b2 * vi + (1 - b2) * g * g
        out.append((p - lr * mi / (jnp.sqrt(vi) + eps), mi, vi))
    return out
run("adam 50x1M params", adam, params, grads, m, v)

# --- GEMM ---
for n in (1024, 2048, 4096):
    a = jax.random.normal(key, (n, n), f32)
    run(f"matmul f32 {n}", lambda a: a @ a, a, flops=2 * n**3)
a16 = jax.random.normal(key, (2048, 2048), jnp.bfloat16)
run("matmul bf16 2048", lambda a: a @ a, a16, flops=2 * 2048**3)
ab = jax.random.normal(key, (16, 512, 512), f32)
run("batched matmul 16x512", lambda a: jnp.einsum("bij,bjk->bik", a, a), ab, flops=16 * 2 * 512**3)

# --- attention ---
B, H, S, D = 8, 8, 512, 64
q = jax.random.normal(key, (B, H, S, D), f32)
def attn(q, k, v):
    s = jnp.einsum("bhsd,bhtd->bhst", q, k) / np.sqrt(D)
    return jnp.einsum("bhst,bhtd->bhsd", jax.nn.softmax(s, -1), v)
attn_flops = 4 * B * H * S * S * D  # the two einsums
run("attention fwd 8x8x512x64", attn, q, q, q, flops=attn_flops)
run("attention fwd+bwd", lambda q: jax.grad(lambda a: jnp.sum(attn(a, a, a)))(q), q, flops=3 * attn_flops)

# --- linear algebra (f32; LAPACK FFI or GPU kernels on metal) ---
for n in (128, 512, 2048):
    a = jax.random.normal(jax.random.fold_in(key, n), (n, n), f32)
    spd = a @ a.T / n + jnp.eye(n, dtype=f32)
    b = jax.random.normal(key, (n, 16), f32)
    run(f"cholesky {n}", jnp.linalg.cholesky, spd, flops=n**3 / 3)
    run(f"solve {n}x16", jnp.linalg.solve, spd, b, flops=2 * n**3 / 3 + 2 * n * n * 16)
    run(f"qr {n}", jnp.linalg.qr, a, flops=8 * n**3 / 3)  # R and Q
    if n <= 512:
        run(f"eigh {n}", jnp.linalg.eigh, spd)

# --- nanoGPT-style training step: 6 layers, d=384, 6 heads, seq 256, batch 16 ---
L, DM, NH, T, BS, VOCAB = 6, 384, 6, 256, 16, 4096
def init_gpt(k):
    ks = jax.random.split(k, 2 + 6 * L)
    s = 0.02
    p = {"wte": s * jax.random.normal(ks[0], (VOCAB, DM), f32), "wpe": s * jax.random.normal(ks[1], (T, DM), f32), "blocks": []}
    for i in range(L):
        kk = ks[2 + 6 * i: 8 + 6 * i]
        p["blocks"].append({"ln1": jnp.ones(DM), "qkv": s * jax.random.normal(kk[0], (DM, 3 * DM), f32),
                            "proj": s * jax.random.normal(kk[1], (DM, DM), f32), "ln2": jnp.ones(DM),
                            "fc": s * jax.random.normal(kk[2], (DM, 4 * DM), f32), "fc2": s * jax.random.normal(kk[3], (4 * DM, DM), f32)})
    p["lnf"] = jnp.ones(DM)
    return p
def gpt_loss(p, idx, tgt):
    x = p["wte"][idx] + p["wpe"][:T]
    mask = jnp.tril(jnp.ones((T, T), bool))
    for b in p["blocks"]:
        h = layernorm(x) * b["ln1"]
        qkv = h @ b["qkv"]
        qh, kh, vh = [t.reshape(BS, T, NH, DM // NH).transpose(0, 2, 1, 3) for t in jnp.split(qkv, 3, -1)]
        s = jnp.einsum("bhsd,bhtd->bhst", qh, kh) / np.sqrt(DM // NH)
        s = jnp.where(mask, s, -1e9)
        a = jnp.einsum("bhst,bhtd->bhsd", jax.nn.softmax(s, -1), vh).transpose(0, 2, 1, 3).reshape(BS, T, DM)
        x = x + a @ b["proj"]
        h = layernorm(x) * b["ln2"]
        x = x + jax.nn.gelu(h @ b["fc"]) @ b["fc2"]
    logits = (layernorm(x) * p["lnf"]) @ p["wte"].T
    return -jnp.mean(jnp.take_along_axis(jax.nn.log_softmax(logits), tgt[..., None], -1))
gp = init_gpt(key)
idx = jax.random.randint(key, (BS, T), 0, VOCAB); tgt = jax.random.randint(jax.random.fold_in(key, 1), (BS, T), 0, VOCAB)
opt = jax.tree_util.tree_map(jnp.zeros_like, gp)
def gpt_step(p, opt, idx, tgt):
    loss, g = jax.value_and_grad(gpt_loss)(p, idx, tgt)
    new_opt = jax.tree_util.tree_map(lambda o, gg: 0.9 * o + 0.1 * gg, opt, g)
    new_p = jax.tree_util.tree_map(lambda pp, o: pp - 1e-3 * o, p, new_opt)
    return new_p, new_opt, loss
run("nanoGPT fwd (loss)", gpt_loss, gp, idx, tgt, iters=5)
run("nanoGPT train step", gpt_step, gp, opt, idx, tgt, iters=5)

# --- small CNN step (conv heavy; naive conv on metal today) ---
xc = jax.random.normal(key, (32, 32, 32, 3), f32)
wc1 = 0.1 * jax.random.normal(key, (3, 3, 3, 32), f32); wc2 = 0.1 * jax.random.normal(key, (3, 3, 32, 64), f32)
def cnn(x, w1, w2):
    h = jax.nn.relu(jax.lax.conv_general_dilated(x, w1, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")))
    h = jax.nn.relu(jax.lax.conv_general_dilated(h, w2, (2, 2), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")))
    return jnp.mean(h ** 2)
run("cnn fwd 32x32x32", cnn, xc, wc1, wc2, iters=3)
run("cnn fwd+bwd", lambda x, w1, w2: jax.grad(cnn, argnums=(1, 2))(x, w1, w2), xc, wc1, wc2, iters=3)
