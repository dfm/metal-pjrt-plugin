"""The same airbench94 run in PyTorch on MPS, for comparison with airbench.py.

Needs PyTorch, which is not a dependency of this repo; run it from its own
environment (it reads the same CIFAR-10 cache as airbench.py):

  uv venv ~/.venvs/torch && uv pip install --python ~/.venvs/torch/bin/python torch numpy
  PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.6 \\
      ~/.venvs/torch/bin/python examples/cifar/torch_baseline.py --runs 5

(The watermarks cap PyTorch's MPS allocator, whose cache otherwise grows past
an 8 GB machine's memory over a run.)

This is airbench94 (https://github.com/KellerJordan/cifar10-airbench, MIT,
Keller Jordan) with the CUDA specifics replaced: device "mps", no CUDA
events (wall time with torch.mps.synchronize), the same data loading as
airbench.py. Model, hyperparameters, augmentation, lookahead and TTA are
airbench's, so the two scripts run the same algorithm.
"""
import argparse, json, math, os, sys, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from airbench_data import load_cifar10, HYP, MEAN, STD  # noqa: E402

DEV = torch.device("mps")
# airbench uses channels_last (cuDNN's fast layout); on MPS it costs memory
# (layout copies) and is off by default here: --channels-last turns it on.
CHANNELS_LAST = False


class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, momentum, eps=1e-12):
        super().__init__(num_features, eps=eps, momentum=1 - momentum)
        self.weight.requires_grad = False

    def forward(self, x):
        # MPS's normalization kernel needs the input in the parameters'
        # dtype (float32); cuDNN, which airbench runs on, accepts fp16 in.
        return super().forward(x.float()).to(x.dtype)


class Conv(nn.Conv2d):
    def __init__(self, cin, cout, kernel_size=3, padding="same", bias=False):
        super().__init__(cin, cout, kernel_size=kernel_size, padding=padding, bias=bias)

    def reset_parameters(self):
        super().reset_parameters()
        if self.bias is not None:
            self.bias.data.zero_()
        w = self.weight.data
        torch.nn.init.dirac_(w[:w.size(1)])


class ConvGroup(nn.Module):
    def __init__(self, cin, cout, momentum):
        super().__init__()
        self.conv1, self.pool = Conv(cin, cout), nn.MaxPool2d(2)
        self.norm1, self.conv2 = BatchNorm(cout, momentum), Conv(cout, cout)
        self.norm2, self.activ = BatchNorm(cout, momentum), nn.GELU()

    def forward(self, x):
        x = self.activ(self.norm1(self.pool(self.conv1(x))))
        return self.activ(self.norm2(self.conv2(x)))


class Mul(nn.Module):
    def __init__(self, s):
        super().__init__()
        self.s = s

    def forward(self, x):
        return x * self.s


def make_net(dtype):
    w = HYP["widths"]
    net = nn.Sequential(
        Conv(3, 24, kernel_size=2, padding=0, bias=True), nn.GELU(),
        ConvGroup(24, w[0], HYP["bn_momentum"]), ConvGroup(w[0], w[1], HYP["bn_momentum"]),
        ConvGroup(w[1], w[2], HYP["bn_momentum"]), nn.MaxPool2d(3), nn.Flatten(),
        nn.Linear(w[2], 10, bias=False), Mul(HYP["scaling_factor"]))
    net[0].weight.requires_grad = False
    net = net.to(DEV, dtype)
    if CHANNELS_LAST:
        net = net.to(memory_format=torch.channels_last)
    for m in net.modules():
        if isinstance(m, BatchNorm):
            m.float()
    return net


def init_whitening(layer, x, eps=5e-4):
    c, (h, w) = x.shape[1], layer.weight.shape[2:]
    p = x.unfold(2, h, 1).unfold(3, w, 1).transpose(1, 3).reshape(-1, c, h, w).float()
    flat = p.view(p.shape[0], -1)
    cov = flat.T @ flat / flat.shape[0]
    evals, evecs = torch.linalg.eigh(cov.cpu())       # eigh is not on MPS
    evals, evecs = evals.to(DEV), evecs.to(DEV)
    evals = evals.flip(0).view(-1, 1, 1, 1)
    evecs = evecs.T.reshape(c * h * w, c, h, w).flip(0)
    scaled = evecs / torch.sqrt(evals + eps)
    layer.weight.data[:] = torch.cat((scaled, -scaled)).to(layer.weight.dtype)


class Lookahead:
    def __init__(self, net):
        self.ema = {k: v.clone() for k, v in net.state_dict().items()}

    def update(self, net, decay):
        for e, p in zip(self.ema.values(), net.state_dict().values()):
            if p.dtype in (torch.half, torch.bfloat16, torch.float):
                e.lerp_(p, 1 - decay)
                p.copy_(e)


def batch_crop(images, crop_size):
    """airbench's per-image random crop, by masks over the (2r+1)^2 shifts."""
    r = (images.size(-1) - crop_size) // 2
    shifts = torch.randint(-r, r + 1, size=(len(images), 2), device=images.device)
    out = torch.empty((len(images), 3, crop_size, crop_size), device=images.device,
                      dtype=images.dtype)
    for sy in range(-r, r + 1):
        for sx in range(-r, r + 1):
            mask = (shifts[:, 0] == sy) & (shifts[:, 1] == sx)
            out[mask] = images[mask, :, r + sy:r + sy + crop_size, r + sx:r + sx + crop_size]
    return out


def infer(net, x, tta_level):
    f = lambda z: net(z.to(memory_format=torch.channels_last) if CHANNELS_LAST else z)
    mirror = lambda z: 0.5 * f(z) + 0.5 * f(z.flip(-1))
    if tta_level == 0:
        return f(x)
    if tta_level == 1:
        return mirror(x)
    p = F.pad(x, (1,) * 4, "reflect")
    return 0.5 * mirror(x) + 0.25 * (mirror(p[:, :, 0:32, 0:32]) + mirror(p[:, :, 2:34, 2:34]))


def train(seed, data, epochs, dtype):
    torch.manual_seed(seed)
    train_x, train_y, test_x, test_y = data
    bs, n_img = HYP["batch_size"], train_x.shape[0]
    steps_per_epoch = n_img // bs
    total = math.ceil(steps_per_epoch * epochs)
    h = HYP
    kilostep_scale = 1024 * (1 + 1 / (1 - h["momentum"]))
    lr = h["lr"] / kilostep_scale
    wd = h["weight_decay"] * bs / kilostep_scale
    lr_b = lr * h["bias_scaler"]
    loss_fn = nn.CrossEntropyLoss(label_smoothing=h["label_smoothing"], reduction="none")

    torch.mps.empty_cache()
    torch.mps.synchronize()
    t0 = time.perf_counter()
    net = make_net(dtype)
    init_whitening(net[0], train_x[:5000])
    norm_b = [p for k, p in net.named_parameters() if "norm" in k and p.requires_grad]
    other = [p for k, p in net.named_parameters() if "norm" not in k and p.requires_grad]
    opt = torch.optim.SGD([dict(params=norm_b, lr=lr_b, weight_decay=wd / lr_b),
                           dict(params=other, lr=lr, weight_decay=wd / lr)],
                          momentum=h["momentum"], nesterov=True)
    warm = int(total * 0.23)

    def get_lr(step):
        if step < warm:
            return 0.2 + 0.8 * step / warm
        return 1.0 + (0.07 - 1.0) * (step - warm) / (total - warm)

    sched = torch.optim.lr_scheduler.LambdaLR(opt, get_lr)
    alpha = 0.95 ** 5 * (torch.arange(total + 1) / total) ** 3
    look = Lookahead(net)
    flip = torch.rand(n_img, device=DEV) < 0.5
    base = torch.where(flip.view(-1, 1, 1, 1), train_x.flip(-1), train_x)
    pad = h["translate"]
    padded = F.pad(base, (pad,) * 4, "reflect")
    del base
    n = 0
    for epoch in range(math.ceil(epochs)):
        net[0].bias.requires_grad = epoch < h["whiten_bias_epochs"]
        imgs = None
        torch.mps.empty_cache()   # MPS caches freed blocks; on 8 GB they add up
        imgs = batch_crop(padded, 32)
        if epoch % 2 == 1:
            imgs = imgs.flip(-1)
        order = torch.randperm(n_img, device=DEV)
        for i in range(steps_per_epoch):
            if n >= total:
                break
            idx = order[i * bs:(i + 1) * bs]
            x = imgs[idx]
            out = net(x.contiguous(memory_format=torch.channels_last) if CHANNELS_LAST else x)
            loss = loss_fn(out.float(), train_y[idx]).sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            n += 1
            if n % 5 == 0:
                look.update(net, decay=alpha[n].item())
    look.update(net, decay=1.0)
    torch.mps.synchronize()
    train_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    net.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, test_x.shape[0], 2000):
            logits = infer(net, test_x[i:i + 2000], h["tta_level"])
            correct += (logits.argmax(-1) == test_y[i:i + 2000]).sum().item()
    torch.mps.synchronize()
    return {"seed": seed, "acc": correct / test_x.shape[0], "train_s": round(train_s, 3),
            "eval_s": round(time.perf_counter() - t1, 3), "steps": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1, help="seed of the first run")
    ap.add_argument("--epochs", type=float, default=HYP["epochs"])
    ap.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    ap.add_argument("--channels-last", action="store_true")
    args = ap.parse_args()
    global CHANNELS_LAST
    CHANNELS_LAST = args.channels_last
    dtype = getattr(torch, args.dtype)
    tx, ty, vx, vy = load_cifar10()
    # Normalized on the CPU: on MPS the float32 intermediates of the whole
    # training set (614 MB each) exhaust an 8 GB machine.
    np_dtype = {torch.float16: np.float16, torch.float32: np.float32}.get(dtype, np.float32)
    norm = lambda x: torch.from_numpy(np.ascontiguousarray(
        ((x.astype(np.float32) / 255 - MEAN) / STD).transpose(0, 3, 1, 2).astype(np_dtype))
    ).to(DEV).to(dtype)
    data = (norm(tx), torch.tensor(ty, device=DEV).long(), norm(vx), torch.tensor(vy, device=DEV).long())
    info = {"backend": "torch-mps", "torch": torch.__version__, "dtype": args.dtype}
    t0 = time.perf_counter()
    train(0, data, min(args.epochs, 1.0), dtype)                     # warmup
    print(json.dumps({**info, "warmup_run_s": round(time.perf_counter() - t0, 2)}), flush=True)
    rows = []
    for r in range(args.runs):
        row = {**info, **train(args.seed + r, data, args.epochs, dtype)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    accs = np.array([r["acc"] for r in rows])
    times = np.array([r["train_s"] for r in rows])
    print(json.dumps({**info, "summary": True, "runs": len(rows),
                      "acc_mean": round(float(accs.mean()), 4), "acc_std": round(float(accs.std()), 4),
                      "runs_at_94": int((accs >= 0.94).sum()),
                      "train_s_mean": round(float(times.mean()), 2),
                      "train_s_std": round(float(times.std()), 2)}))


if __name__ == "__main__":
    main()
