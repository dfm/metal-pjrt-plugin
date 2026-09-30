"""conv_jax.py's measurement in PyTorch on MPS, for comparison.

Needs PyTorch (not a dependency of this repo; see torch_baseline.py):

  ~/.venvs/torch/bin/python examples/cifar/conv_torch.py

(prefix it with `scripts/device_lock.py --` when other GPU jobs may run).
fp16 (as airbench runs), NCHW, the same shapes; forward, and the input and
weight gradients each alone (torch.autograd.grad with respect to one
input), median ms of 10 calls after 3 warmup calls, synchronized with
torch.mps.synchronize.
"""
import json, time

import numpy as np
import torch
import torch.nn.functional as F

SHAPES = [("whiten", 32, 3, 24, 2, 0), ("g1c1", 31, 24, 64, 3, "same"),
          ("g1c2", 15, 64, 64, 3, "same"), ("g2c1", 15, 64, 256, 3, "same"),
          ("g2c2", 7, 256, 256, 3, "same"), ("g3c1", 7, 256, 256, 3, "same"),
          ("g3c2", 3, 256, 256, 3, "same")]
N, DEV, DT = 1024, torch.device("mps"), torch.float16


def median_ms(f):
    for _ in range(3):
        f()
        torch.mps.synchronize()
    ts = []
    for _ in range(10):
        t0 = time.perf_counter()
        f()
        torch.mps.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts))


def main():
    for name, hw, ci, co, k, pad in SHAPES:
        x = (torch.ones(N, ci, hw, hw, device=DEV, dtype=DT) * 0.1).requires_grad_()
        w = (torch.ones(co, ci, k, k, device=DEV, dtype=DT) * 0.01).requires_grad_()
        y = F.conv2d(x, w, padding=pad)
        g = torch.ones_like(y)
        # dgrad and wgrad include their own forward (autograd needs the graph);
        # the forward's time is subtracted.
        tf = median_ms(lambda: F.conv2d(x, w, padding=pad))
        td = median_ms(lambda: torch.autograd.grad(F.conv2d(x, w, padding=pad), x, g)) - tf
        tw = median_ms(lambda: torch.autograd.grad(F.conv2d(x, w, padding=pad), w, g)) - tf
        flops = 2 * N * y.shape[-2] * y.shape[-1] * co * ci * k * k
        print(json.dumps({"impl": "torch-mps", "torch": torch.__version__, "conv": name,
                          "fwd_ms": round(tf, 2), "dgrad_ms": round(td, 2), "wgrad_ms": round(tw, 2),
                          "fwd_tflops": round(flops / tf / 1e9, 2),
                          "wgrad_tflops": round(flops / tw / 1e9, 2)}), flush=True)


if __name__ == "__main__":
    main()
