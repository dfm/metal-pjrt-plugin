# Copyright 2026 The metal-pjrt-plugin Authors
# SPDX-License-Identifier: Apache-2.0
# Derived from airbench94 (https://github.com/KellerJordan/cifar10-airbench):
# Copyright (c) 2024 Keller Jordan. MIT License; the full notice is in
# LICENSE-airbench in this directory.
"""CIFAR-10 to 94% in pure JAX: a port of Keller Jordan's airbench94.

  JAX_PLATFORMS=mtl,cpu .venv/bin/python examples/cifar/airbench.py --runs 5

(prefix it with `scripts/device_lock.py --` when other GPU jobs may run).

The model, hyperparameters, augmentation and evaluation follow airbench94
(https://github.com/KellerJordan/cifar10-airbench, MIT): a frozen 2x2
patch-whitening convolution, three conv groups (64, 256, 256 channels) of
conv -> 2x2 max-pool -> BatchNorm -> GELU -> conv -> BatchNorm -> GELU, a
global max-pool and a linear layer; label smoothing 0.2, Nesterov SGD with
airbench's learning-rate and weight-decay scaling, a piecewise-linear
schedule, a lookahead EMA every 5 steps; random 2-pixel translation and
flips alternating each epoch; test-time augmentation over mirror and
translate. Images stay on the device for the whole run, and augmentation
runs there too.

Time is measured like airbench's: the whitening initialization and every
training step, synchronized at the end; compilation (done before timing)
and the final evaluation are reported separately.
"""
import argparse, functools, json, math, os, sys, time

import jax
import jax.numpy as jnp
import numpy as np
import optax

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from airbench_data import HYP, MEAN, STD, load_cifar10  # noqa: E402


# --- Data ------------------------------------------------------------------

def normalize(x_uint8, dtype):
    """Images are kept as uint8 (150 MB for the training set) and
    normalized inside the jitted programs, fused into the first
    convolution's input; normalized bf16 copies would take twice that, and
    the float32 intermediates of an eager normalization 614 MB each."""
    return ((x_uint8.astype(jnp.float32) / 255 - MEAN) / STD).astype(dtype)


def reflect_pad(x, p):
    return jnp.pad(x, ((0, 0), (p, p), (p, p), (0, 0)), mode="reflect")


@jax.jit
def first_epoch_images(x, key):
    """Epoch 0's set: a random half of the images flipped, reflect-padded
    for the random translations."""
    flip = jax.random.bernoulli(key, 0.5, (x.shape[0], 1, 1, 1))
    return reflect_pad(jnp.where(flip, x[:, :, ::-1], x), HYP["translate"])


@jax.jit
def epoch_images(padded, key, flip):
    """Every image cropped at a random offset in [0, 2 * translate], and
    the whole set flipped if `flip` (airbench flips every other epoch)."""
    n, hp, size = padded.shape[0], padded.shape[1], padded.shape[1] - 2 * HYP["translate"]
    off = jax.random.randint(key, (n, 2), 0, hp - size + 1)
    crop = lambda img, o: jax.lax.dynamic_slice(img, (o[0], o[1], 0), (size, size, img.shape[-1]))
    out = jax.vmap(crop)(padded, off)
    return jnp.where(flip, out[:, :, ::-1], out)


# --- Model -----------------------------------------------------------------

def conv(x, w, padding="SAME"):
    return jax.lax.conv_general_dilated(
        x, w.astype(x.dtype), (1, 1), padding, dimension_numbers=("NHWC", "HWIO", "NHWC"))


def max_pool(x, k):
    return jax.lax.reduce_window(x, -jnp.inf, jax.lax.max, (1, k, k, 1), (1, k, k, 1), "VALID")


def batch_norm(x, bias, stats, train, momentum, eps=1e-12):
    """BatchNorm without a scale (airbench freezes it at 1), in float32.
    Returns (y, new running stats)."""
    x32 = x.astype(jnp.float32)
    if train:
        mean = x32.mean(axis=(0, 1, 2))
        var = x32.var(axis=(0, 1, 2))
        n = x.shape[0] * x.shape[1] * x.shape[2]
        # PyTorch's convention: running stats move by (1 - momentum) and
        # keep the unbiased variance.
        new = {"mean": momentum * stats["mean"] + (1 - momentum) * mean,
               "var": momentum * stats["var"] + (1 - momentum) * var * n / (n - 1)}
    else:
        mean, var, new = stats["mean"], stats["var"], stats
    y = (x32 - mean) * jax.lax.rsqrt(var + eps) + bias
    return y.astype(x.dtype), new


REMAT = False   # --remat turns it on

# Called as EPOCH_HOOK(seed, epoch, steps, params) after the setup (epoch -1)
# and after each epoch, if set (--profile-memory sets one that syncs and
# prints a row per epoch). None: no extra synchronization.
EPOCH_HOOK = None


def conv_group(g, s, x, train):
    gelu = functools.partial(jax.nn.gelu, approximate=False)
    x = max_pool(conv(x, g["conv1"]), 2)
    x, s1 = batch_norm(x, g["bias1"], s[0], train, HYP["bn_momentum"])
    x = gelu(x)
    x, s2 = batch_norm(conv(x, g["conv2"]), g["bias2"], s[1], train, HYP["bn_momentum"])
    return gelu(x), (s1, s2)


def forward(params, stats, x, train, dtype=jnp.bfloat16):
    """Logits [N, 10] (float32) and the updated BatchNorm running stats."""
    gelu = functools.partial(jax.nn.gelu, approximate=False)
    x = normalize(x, dtype) if x.dtype == jnp.uint8 else x.astype(dtype)
    x = conv(x, params["whiten"]["w"], "VALID") + params["whiten"]["b"].astype(dtype)
    x = gelu(x)
    new_stats = []
    # With --remat the backward pass recomputes each group from its input
    # instead of keeping every intermediate: the step's scratch at batch
    # 1024 goes from 1.17 to 0.94 GB, for a run 27% longer (242 vs 190 s on
    # an M3, 2026-10-01). Without it the run fits the plugin's default
    # memory budget on an 8 GB Mac.
    group = jax.checkpoint(conv_group, static_argnums=(3,)) if REMAT and train else conv_group
    for g, s in zip(params["groups"], stats):
        x, s12 = group(g, s, x, train)
        new_stats.append(s12)
    x = max_pool(x, 3).reshape(x.shape[0], -1)
    logits = (x @ params["head"].astype(dtype)).astype(jnp.float32) * HYP["scaling_factor"]
    return logits, new_stats


@jax.jit
def init_params(key, train_x_norm):
    """airbench's initialization: PyTorch's default conv init with the first
    in-channels filters set to identity (dirac), whitening from data."""
    widths = (24,) + HYP["widths"]
    keys = jax.random.split(key, 2 * len(HYP["widths"]) + 1)

    def conv_init(k, cin, cout):
        bound = 1 / math.sqrt(cin * 9)        # kaiming_uniform(a=sqrt(5))
        w = jax.random.uniform(k, (3, 3, cin, cout), jnp.float32, -bound, bound)
        n = min(cin, cout)
        eye = jnp.zeros((3, 3, cin, n)).at[1, 1, jnp.arange(n), jnp.arange(n)].set(1.0)
        return w.at[..., :n].set(eye)

    groups, stats = [], []
    for i, (cin, cout) in enumerate(zip(widths[:-1], widths[1:])):
        groups.append({"conv1": conv_init(keys[2 * i], cin, cout), "bias1": jnp.zeros(cout),
                       "conv2": conv_init(keys[2 * i + 1], cout, cout), "bias2": jnp.zeros(cout)})
        st = lambda: {"mean": jnp.zeros(cout), "var": jnp.ones(cout)}
        stats.append((st(), st()))
    bound = 1 / math.sqrt(widths[-1])
    head = jax.random.uniform(keys[-1], (widths[-1], 10), jnp.float32, -bound, bound)
    params = {"whiten": {"w": whitening(train_x_norm), "b": jnp.zeros(24)},
              "groups": groups, "head": head}
    return params, stats


@jax.jit
def whitening(x, eps=5e-4):
    """The frozen first layer: 2x2 patches projected on the eigenvectors of
    their covariance, scaled to unit variance, with both signs (24 filters)."""
    x = x.astype(jnp.float32)
    p = jnp.stack([x[:, i:i + 31, j:j + 31] for i in range(2) for j in range(2)], axis=3)
    p = p.reshape(-1, 12)                                            # (i, j, c) per patch
    cov = p.T @ p / p.shape[0]
    evals, evecs = jnp.linalg.eigh(cov)
    w = (evecs / jnp.sqrt(evals + eps)).T[::-1]                      # [12, 12], largest first
    w = jnp.concatenate([w, -w])                                     # [24, 12]
    return w.reshape(24, 2, 2, 3).transpose(1, 2, 3, 0)              # HWIO


# --- Training --------------------------------------------------------------

@functools.lru_cache
def make_optimizer(total_steps):
    h = HYP
    kilostep_scale = 1024 * (1 + 1 / (1 - h["momentum"]))
    lr = h["lr"] / kilostep_scale
    wd = h["weight_decay"] * h["batch_size"] / kilostep_scale
    lr_b = lr * h["bias_scaler"]
    warm = int(total_steps * 0.23)

    def sched(step):
        up = 0.2 + 0.8 * step / warm
        down = 1.0 + (0.07 - 1.0) * (step - warm) / (total_steps - warm)
        return jnp.where(step < warm, up, down)

    def group(base_lr):
        # PyTorch SGD semantics: decay added to the gradient before
        # momentum; airbench sets decay = wd / lr so the step shrinks the
        # weights by sched * wd.
        return optax.chain(optax.add_decayed_weights(wd / base_lr),
                           optax.sgd(lambda s: base_lr * sched(s), h["momentum"], nesterov=True))

    def label(params):
        # airbench's bias group is the parameters named "norm"; the
        # whitening bias is in the other group.
        return {"whiten": {"w": "frozen", "b": "other"},
                "groups": [{"conv1": "other", "bias1": "bias", "conv2": "other", "bias2": "bias"}
                           for _ in params["groups"]],
                "head": "other"}

    return optax.multi_transform(
        {"bias": group(lr_b), "other": group(lr), "frozen": optax.set_to_zero()}, label)


def loss_fn(params, stats, x, y, dtype=jnp.bfloat16):
    logits, new_stats = forward(params, stats, x, train=True, dtype=dtype)
    labels = optax.smooth_labels(jax.nn.one_hot(y, 10), HYP["label_smoothing"])
    return optax.softmax_cross_entropy(logits, labels).sum(), new_stats


@functools.lru_cache
def make_train_step(tx, dtype=jnp.bfloat16):
    """One jitted step per (optimizer, dtype), shared by every run: the
    warmup run then compiles exactly the programs the timed runs use."""
    # No donation: after a lookahead the net and the ema share buffers
    # (and the parameters are only a few MB).
    #
    # The step picks its own batch (batch i of the epoch's `order`) and
    # carries its step counters as device values: a fresh host value per
    # step (an eager slice's indices, a Python float) made the next program
    # wait for the GPU to drain, ~60 ms of idle GPU per step.
    @jax.jit
    def step(params, stats, opt_state, imgs, labels, order, i, n, whiten_bias_on):
        bs = HYP["batch_size"]
        idx = jax.lax.dynamic_slice(order, (i * bs,), (bs,))
        x, y = imgs[idx], labels[idx]
        (loss, stats), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, stats, x, y,
                                                                          dtype)
        updates, opt_state = tx.update(grads, opt_state, params)
        # After whiten_bias_epochs the whitening bias is frozen, as in
        # airbench (requires_grad = False: no gradient, decay or momentum).
        updates["whiten"]["b"] = updates["whiten"]["b"] * whiten_bias_on
        return optax.apply_updates(params, updates), stats, opt_state, loss, i + 1, n + 1
    return step


@jax.jit
def lookahead(ema, params, stats, alpha, n):
    """airbench's LookaheadState.update with decay alpha[n] (a device table
    and a device step count, so no host value per call): ema <- lerp(ema,
    net, 1 - decay), then the net (weights and BatchNorm stats) is reset to
    the ema."""
    decay = alpha[n]
    net = (params, stats)
    ema = jax.tree.map(lambda e, p: e + (1 - decay) * (p - e), ema, net)
    return ema, ema[0], ema[1]


@functools.partial(jax.jit, static_argnums=(3, 4))
def predict(params, stats, x, tta_level, dtype=jnp.bfloat16):
    """Test-time augmentation: 0 none, 1 mirror, 2 mirror and translate."""
    f = lambda z: forward(params, stats, z, train=False, dtype=dtype)[0]
    mirror = lambda z: 0.5 * f(z) + 0.5 * f(z[:, :, ::-1])
    if tta_level == 0:
        return f(x)
    if tta_level == 1:
        return mirror(x)
    p = reflect_pad(x, 1)
    shifted = 0.5 * (mirror(p[:, 0:32, 0:32]) + mirror(p[:, 2:34, 2:34]))
    return 0.5 * mirror(x) + 0.5 * shifted


def evaluate(params, stats, test_x, test_y, tta_level, dtype=jnp.bfloat16, batch=500):
    # 500 images per call: with TTA level 2 (6 forward passes) a batch of
    # 2000 needs 1.8 GB of scratch, 500 needs 0.45 GB.
    correct = 0
    for i in range(0, test_x.shape[0], batch):
        logits = predict(params, stats, test_x[i:i + batch], tta_level, dtype)
        correct += int((jnp.argmax(logits, -1) == test_y[i:i + batch]).sum())
    return correct / test_x.shape[0]


def train(seed, data, epochs=HYP["epochs"], dtype=jnp.bfloat16, log=None, max_steps=None):
    """One airbench94 run. Returns a dict of accuracy and timings.

    `max_steps` stops early (the warmup run) while keeping the full run's
    schedule, so that every program is the one the full run uses."""
    train_x, train_y, test_x, test_y = data
    bs = HYP["batch_size"]
    steps_per_epoch = train_x.shape[0] // bs                        # drop_last
    total = math.ceil(steps_per_epoch * epochs)
    tx = make_optimizer(total)
    step = make_train_step(tx, jnp.dtype(dtype))
    stop = total if max_steps is None else min(total, max_steps)
    alpha = jnp.asarray(0.95 ** 5 * (np.arange(total + 1) / total) ** 3, jnp.float32)
    key = jax.random.key(seed)

    jax.block_until_ready(train_x)
    t0 = time.perf_counter()
    key, k = jax.random.split(key)
    params, stats = init_params(k, normalize(train_x[:5000], jnp.float32))
    opt_state = tx.init(params)
    ema = jax.tree.map(jnp.copy, (params, stats))
    # Epoch 0 flips a random half of the images; later epochs flip the
    # whole set every other epoch (airbench's alternating flip).
    key, k = jax.random.split(key)
    padded = first_epoch_images(train_x, k)
    n = 0                                                           # host copy of n_dev
    n_dev = jnp.zeros((), jnp.int32)
    if EPOCH_HOOK:
        EPOCH_HOOK(seed, -1, n, params)
    for epoch in range(math.ceil(epochs)):
        key, k1, k2 = jax.random.split(key, 3)
        imgs = epoch_images(padded, k1, epoch % 2 == 1)
        order = jax.random.permutation(k2, train_x.shape[0])
        wb = jnp.float32(epoch < HYP["whiten_bias_epochs"])   # host values once per epoch
        i_dev = jnp.zeros((), jnp.int32)
        for _ in range(steps_per_epoch):
            if n >= stop:
                break
            params, stats, opt_state, loss, i_dev, n_dev = step(
                params, stats, opt_state, imgs, train_y, order, i_dev, n_dev, wb)
            n += 1
            if n % 5 == 0:
                ema, params, stats = lookahead(ema, params, stats, alpha, n_dev)
        if log:
            log(epoch, float(loss) / bs)
        if EPOCH_HOOK:
            EPOCH_HOOK(seed, epoch, n, params)
    params, stats = ema               # the final update with decay 1: the net becomes the ema
    jax.block_until_ready(params)
    train_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    acc = evaluate(params, stats, test_x, test_y, HYP["tta_level"], dtype)
    return {"seed": seed, "acc": acc, "train_s": round(train_s, 3),
            "eval_s": round(time.perf_counter() - t1, 3), "steps": n}


def install_memory_profile():
    """EPOCH_HOOK printing one JSON row per epoch (`epoch_s` covers the
    epoch's steps and its augmentation, `steps_ms` = epoch_s / steps) with
    memprofile.snapshot()'s counters; the seed of the warmup run is 0."""
    import memprofile
    global EPOCH_HOOK
    last = {}

    def hook(seed, epoch, steps, params):
        jax.block_until_ready(params)
        now = time.perf_counter()
        row = {"profile": "epoch", "seed": seed, "epoch": epoch, "steps": steps}
        if epoch >= 0:
            dt = now - last["t"]
            row["epoch_s"] = round(dt, 3)
            row["step_ms"] = round(dt / max(steps - last["steps"], 1) * 1e3, 1)
        last.update(t=now, steps=steps)
        print(json.dumps({**row, **memprofile.snapshot()}), flush=True)

    EPOCH_HOOK = hook


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1, help="seed of the first run")
    ap.add_argument("--epochs", type=float, default=HYP["epochs"])
    ap.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--warm-full", action="store_true",
                    help="one untimed full-length run before the timed runs, so that they "
                         "start at the GPU's steady (on a fanless Mac: throttled) state")
    ap.add_argument("--profile-memory", action="store_true",
                    help="print a JSON row per epoch: time, macOS pressure, footprint, "
                         "the plugin's memory counters (syncs once per epoch)")
    ap.add_argument("--remat", action="store_true",
                    help="recompute activations in the backward pass: 0.23 GB less scratch, "
                         "a run ~27%% longer")
    args = ap.parse_args()
    global REMAT
    REMAT = args.remat
    dtype = jnp.dtype(args.dtype)

    tx_, ty_, vx_, vy_ = load_cifar10()
    data = tuple(jnp.asarray(a) for a in (tx_, ty_, vx_, vy_))
    log = (lambda e, l: print(f"  epoch {e}: loss {l:.4f}", file=sys.stderr)) if args.verbose else None
    if args.profile_memory:
        install_memory_profile()

    # Warmup: the first epoch of a full-length run, which compiles every
    # program the timed runs use (the same schedule, the same jitted step).
    steps_per_epoch = data[0].shape[0] // HYP["batch_size"]
    t0 = time.perf_counter()
    train(0, data, args.epochs, dtype, max_steps=steps_per_epoch)
    warm = time.perf_counter() - t0
    step = make_train_step(make_optimizer(math.ceil(steps_per_epoch * args.epochs)), dtype)
    compiled = step._cache_size()
    print(json.dumps({"backend": jax.devices()[0].platform, "warmup_run_s": round(warm, 2)}),
          flush=True)
    if args.warm_full:
        t0 = time.perf_counter()
        r = train(0, data, args.epochs, dtype)
        print(json.dumps({"backend": jax.devices()[0].platform, "warm_full_run_s":
                          round(time.perf_counter() - t0, 2), "warm_full_train_s": r["train_s"]}),
              flush=True)
    rows = []
    for r in range(args.runs):
        row = train(args.seed + r, data, args.epochs, dtype, log)
        row["backend"] = jax.devices()[0].platform
        if args.profile_memory:
            import memprofile
            row.update(memprofile.snapshot())
        rows.append(row)
        print(json.dumps(row), flush=True)
    # The timed runs must not have compiled anything new.
    assert step._cache_size() == compiled, "the train step recompiled during a timed run"
    accs = np.array([r["acc"] for r in rows])
    times = np.array([r["train_s"] for r in rows])
    print(json.dumps({"summary": True, "runs": len(rows), "acc_mean": round(float(accs.mean()), 4),
                      "acc_std": round(float(accs.std(ddof=1)), 4) if len(rows) > 1 else 0.0,
                      "runs_at_94": int((accs >= 0.94).sum()),
                      "train_s_mean": round(float(times.mean()), 2),
                      "train_s_std": round(float(times.std(ddof=1)), 2) if len(rows) > 1 else 0.0,
                      "std": "sample (ddof=1)"}))


if __name__ == "__main__":
    main()
