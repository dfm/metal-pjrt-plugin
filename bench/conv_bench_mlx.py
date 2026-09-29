"""MLX's convolutions on bench/conv_bench.cc's cases, timed the same way.

  .venv/bin/python bench/conv_bench_mlx.py [rounds] [f32|f16|bf16]

Each round evaluates a burst of 30 independent convolutions per case (30
inputs, so nothing is shared) and prints one JSON line per case with the
burst's wall time per convolution in ms. Run it interleaved with
bazel-bin/bench/conv_bench (GPU performance states make single runs
bimodal) under scripts/device_lock.py.
"""
import json, sys, time
import mlx.core as mx

BURST = 30
rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
tname = sys.argv[2] if len(sys.argv) > 2 else "f32"
dtype = {"f32": mx.float32, "f16": mx.float16, "bf16": mx.bfloat16}[tname]

# (name, input NHWC, weight OHWI, kwargs), as conv_bench.cc.
CASES = [
    ("conv1 fwd 3->32", (32, 32, 32, 3), (32, 3, 3, 3),
     dict(stride=1, padding=([1, 1], [1, 1]))),
    ("conv2 fwd 32->64 s2", (32, 32, 32, 32), (64, 3, 3, 32),
     dict(stride=2, padding=([0, 0], [1, 1]))),
    ("conv2 input grad", (32, 16, 16, 64), (32, 3, 3, 64),
     dict(stride=1, padding=([2, 2], [1, 1]), input_dilation=2, flip=True)),
]

# Weight gradients (MLX's vjp w.r.t. the weight: patches x cotangent) of
# the two forward convolutions.
WGRAD = {"conv1 weight grad": 0, "conv2 weight grad": 1}
CASES += [(name, CASES[i][1], CASES[i][2], CASES[i][3]) for name, i in WGRAD.items()]


def run(name, x, w, kw):
    if name not in WGRAD:
        return mx.conv_general(x, w, **kw)
    cot = mx.ones(mx.conv_general(x, w, **kw).shape, dtype=x.dtype)
    return mx.vjp(lambda w: mx.conv_general(x, w, **kw), [w], [cot])[1][0]


for name, xs, ws, kw in CASES:
    xs_ = [mx.random.normal(xs).astype(dtype) for _ in range(BURST)]
    w = mx.random.normal(ws).astype(dtype)
    mx.eval(xs_, w)
    for _ in range(3):
        mx.eval([run(name, x, w, kw) for x in xs_[:3]])
    for _ in range(rounds):
        t0 = time.perf_counter()
        outs = [run(name, x, w, kw) for x in xs_]
        mx.eval(outs)
        ms = (time.perf_counter() - t0) * 1e3 / BURST
        o = mx.conv_general(xs_[0], w, **kw).shape
        flops = 2 * o[0] * o[1] * o[2] * o[3] * ws[1] * ws[2] * ws[3]
        print(json.dumps({"backend": "mlx", "case": name, "type": tname,
                          "ms": round(ms, 4), "gflops": round(flops / (ms * 1e6), 1),
                          "out": list(o)}), flush=True)
