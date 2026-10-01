# CIFAR-10 to 94%: airbench94 in pure JAX

A port of Keller Jordan's [airbench94](https://github.com/KellerJordan/cifar10-airbench)
(MIT), the fastest known recipe for training a CIFAR-10 classifier to 94%
test accuracy, to plain JAX and optax, run on the Mac's GPU by this
plugin. It is an end-to-end training workload: data on the device, GPU
augmentation, a small ResNet-style CNN with BatchNorm, SGD with a
schedule, weight averaging and test-time augmentation. The metric is the
one airbench uses: time to 94% (training time, with the accuracy of every
run reported).

| file | what |
|---|---|
| `airbench.py` | model, training, evaluation and a command line |
| `airbench_data.py` | CIFAR-10 loading and the hyperparameters (numpy only, shared) |
| `check.py` | one training step's loss and gradients against CPU (float32 and bfloat16) |
| `torch_baseline.py` | the same algorithm in PyTorch on MPS (run from its own environment) |
| `compare.py` | the comparison protocol: both, interleaved, with an optional GPU power sampler |
| `conv_jax.py`, `conv_torch.py` | airbench's convolutions one at a time (forward, input and weight gradients), JAX vs PyTorch |
| `LICENSE-airbench` | airbench's MIT license (the files above are derived from it) |

## Run it

```sh
uv pip install --python .venv/bin/python -e '.[examples]'     # optax
export JAX_PLATFORMS=mtl,cpu
.venv/bin/python examples/cifar/check.py
.venv/bin/python examples/cifar/airbench.py --runs 5
# at a matched thermal state (fanless Macs throttle after ~3 min): an untimed
# full-length run first; the same flag exists in torch_baseline.py
.venv/bin/python examples/cifar/airbench.py --runs 5 --warm-full
# the comparison protocol: an untimed warm-up run, then JAX and PyTorch
# alternating, 5 runs each, optionally with a GPU power/P-state sampler
.venv/bin/python examples/cifar/compare.py --torch-python ~/.venvs/torch/bin/python \
    --wrap "scripts/device_lock.py --" [--sampler PATH] [--one-process] \
    [--jax-variants "noremat=;remat=--remat"]   # several JAX arms, interleaved
# per-epoch time, footprint, macOS memory pressure and the plugin's cache counters
.venv/bin/python examples/cifar/airbench.py --runs 5 --profile-memory
# PyTorch on MPS, in its own environment (not a dependency of this repo)
uv venv ~/.venvs/torch && uv pip install --python ~/.venvs/torch/bin/python torch numpy
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.6 \
    ~/.venvs/torch/bin/python examples/cifar/torch_baseline.py --runs 5
```

Prefix the GPU commands with `scripts/device_lock.py --` when other GPU
jobs may run. The first run downloads CIFAR-10 (163 MB; Krizhevsky,
"Learning Multiple Layers of Features from Tiny Images", 2009) from the
authors' site into `~/.cache/metal-pjrt-examples`; the data is not part of
this repository. Each script first does a one-epoch warmup run, which
compiles every program the timed runs use; its time is reported
separately (`warmup_run_s`). Accuracy and time spreads are sample
standard deviations (ddof=1).

## Results

Measured 2026-10-01 on an M3 MacBook Air (10-core GPU, 8 GB, no fan),
macOS 26.2, JAX 0.11.2 with optax and the plugin as of 2026-09-30, against
PyTorch 2.14.1 on MPS running the same algorithm (`torch_baseline.py`,
float16, as airbench runs), all in one session with `compare.py`: an
untimed full-length warm-up run, then the three arms alternating (3 seeds
each, one process per run), then each arm's 3 runs in one process; the GPU
power and performance state sampled every 10 s. Times are airbench's
training time per run, mean ± sample standard deviation; spread is
(max - min) / min; accuracy is on the full 10,000-image test set.

| phase | | JAX, bf16 (default) | JAX, bf16, `--remat` | PyTorch MPS, fp16 |
|---|---|---|---|---|
| alternating | time | 188.0 ± 1.2 s (spread 1.2%) | 239.7 ± 1.4 s (1.1%) | 219.7 ± 4.8 s (4.4%) |
| | GPU power, mean P-state | 6.1 W, 7.9 | 6.2 W, 7.9 | 6.6 W, 7.6 |
| one process | time | 190.5 ± 0.2 s (spread 0.2%) | 242.2 ± 0.3 s (0.2%) | 219.5 ± 1.7 s (1.6%) |
| | GPU power, mean P-state | 5.8 W, 7.6 | 6.0 W, 7.6 | 6.6 W, 7.6 |
| | test accuracy (seeds 1-3) | 94.03% (every seed) | 94.03% | 93.94% ± 0.03% |

At the same GPU power, JAX trains in 0.86-0.87x PyTorch's time (13-14%
faster), and its later runs in one process stay within 0.2% of the first.
Accuracy is 94.03% for each of seeds 1-3 by coincidence: the seeds' final
predictions differ on 370 test images, and each gets 9,403 right. Runs are
deterministic: a seed gives the same accuracy every time.

The runs start at the GPU's sustained state, not its first minutes: this
MacBook Air has no fan, and under load its GPU holds its top performance
state (~9 W) for about 3 minutes, then settles near 6 W; the untimed
warm-up run takes it there, and the arms alternate so that neither gets the
fast minutes. (A single JAX run from a cool start: 161 s.)

Memory: the default JAX run fits the plugin's default budget (half of RAM,
4 GiB here; its own allocations peaked at 1.7 GB at epoch ends) with a
peak footprint of 3.5 GB. `--remat` recomputes activations in the backward
pass instead (0.23 GB less scratch, a run 27% longer). PyTorch's footprint
was 3.9-5.9 GB in earlier runs (2026-09-29), with its MPS allocator capped
by PYTORCH_MPS_HIGH/LOW_WATERMARK_RATIO=0.8/0.6 to stay within 8 GB;
without a cap its cache grew past 9 GB.

History, in one line: the first port (2026-09-29) took ~260 s per run, with
the conv weight gradients at 2.5-5x the forward time, before the plugin's
faster weight gradients, its PyTorch-like memory limits and its single
command queue (2026-09-29/30).

For scale, airbench94 takes about 3.8 s on an A100; the M3 is ~76x slower
here, a ~3.5 TFLOPS GPU against a 312 TFLOPS (fp16 tensor core) one plus
the gaps below.

Correctness (`check.py`, one step on 256 CIFAR-10 images against CPU):
float32 loss identical to 7 digits and gradients within 6e-5 of CPU
float32, except the last two conv layers' weight gradients (2e-4, 1e-3),
which are ill-conditioned: CPU float32 itself is 1.3e-4 and 5.3e-4 away
from float64 there. In bfloat16, the gradients are ~20% from float32 on
both backends alike.

## What it took

The port reproduced airbench94's accuracy on the first complete run
(94.01%). Everything else was memory: an 8 GB Mac shared with other
applications has 1-2 GB free, and until 2026-09-29 the plugin's memory guard
refused any allocation that would leave less than 512 MB of it.

- **Keep the data as uint8 and normalize inside the jitted programs.**
  The training set is 150 MB as uint8 and 300 MB as bf16; an eager
  normalization also materializes float32 intermediates of 614 MB each.
  Normalizing inside the step fuses it into the first convolution's input.
- **Evaluate in small batches.** Test-time augmentation runs six forward
  passes per batch; at 2000 images that program needed 1.8 GB of scratch,
  at 500 images 0.45 GB.
- **Rematerialization is optional** (`--remat`, `jax.checkpoint` per conv
  group): the step's scratch at batch 1024 goes from 1.17 to 0.94 GB, for a
  run 27% longer. It was the default while the plugin's old memory guard
  refused the larger step; without the guard the default run fits.

Measured and not adopted: exact GELU vs the tanh approximation (no
measurable difference), float16 instead of bfloat16 (12% slower), and
computing the conv weight gradients as 9 shifted GEMMs (faster only for
the 7x7 and 3x3 layers, 4% of a step, not worth a custom gradient).

### Differences from airbench94

- Both: a one-epoch warmup run before the timed runs (airbench warms up
  with a full-length run on random labels); float32 eigendecomposition
  for the whitening layer (JAX through the plugin, PyTorch on the CPU, as MPS
  has no eigh).
- JAX: bfloat16 compute with float32 master weights and momentum
  (airbench and the PyTorch script keep the network in fp16, BatchNorm in
  float32); images kept as uint8 and normalized inside the jitted step;
  conv groups optionally rematerialized in the backward pass (--remat).
- PyTorch on MPS: BatchNorm casts its input to float32 (MPS's kernel
  rejects fp16 input with float32 parameters); NCHW instead of
  channels-last (channels-last grew the footprint past 9 GB);
  normalization done once on the CPU before the timed region;
  torch.mps.empty_cache() at every epoch, inside the timed region, and an
  allocator cap (PYTORCH_MPS_*_WATERMARK_RATIO), both to stay within 8 GB.
- Unchanged in both: the model, hyperparameters, schedule, lookahead,
  flip/translate augmentation (the crop by masks is airbench's own
  batch_crop) and test-time augmentation.

## What it found in the plugin

- **The convolutions are correct.** One training step's gradients in
  float32 match CPU float32 to within 6e-5, except two ill-conditioned
  weight gradients (2e-4, 1e-3; CPU float32 itself is 1.3e-4 and 5.3e-4
  from float64 there) (`check.py`); in
  bfloat16 they are as far from float32 as CPU bfloat16 is.
- **Conv weight gradients were the bottleneck of CNN training.** Per
  layer at batch 1024 (bf16), the forward and input-gradient
  convolutions ran near PyTorch MPS's speed, but the weight gradient took
  2.5-5x the forward time (31x31, 24 -> 64 channels: 54 ms, where
  PyTorch's whole backward is 25 ms): a reduction over batch x pixels
  (~1M) into a small output. Reported with a standalone benchmark, the
  plugin now sizes the GEMM tile and split-K together (since 2026-09-29): 18 ms
  for that layer.
- **The GPU idled ~60 ms per step** (of ~490 at full clock): every fresh
  host value (an eager slice's indices, a Python float) made the next
  program wait for the GPU to drain. The plugin's single command queue
  (2026-09-30) removed the wait; the example also keeps its step counters
  and batch selection on the device, so a step creates no host values.
- **The memory guard decided whether training ran at all** on a busy
  8 GB machine. It refused any allocation that would leave less than
  512 MB of free memory, so the step (0.9 GB of scratch) was refused
  whenever less than ~1.4 GB was free: 25 of 28 attempts. PyTorch MPS,
  which caps only its own allocations, ran with a 4-6 GB footprint and
  let macOS make room. The plugin now does the same (since 2026-09-29: a
  per-process budget, refusals only at critical system memory pressure),
  and every run in Results ran without a refusal, in the default budget.
- **An out-of-memory error named the wrong program** (the next eager op,
  not the program whose allocation failed). Reported with a repro; fixed.
