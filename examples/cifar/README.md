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

## Run it

```sh
uv pip install --python .venv/bin/python -e '.[examples]'     # optax
export JAX_PLATFORMS=mtl,cpu
.venv/bin/python examples/cifar/check.py
.venv/bin/python examples/cifar/airbench.py --runs 5
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

Measured 2026-09-29 on an M3 MacBook (10-core GPU, 8 GB), macOS 26.2,
JAX 0.11.2 with optax, with the plugin's faster conv weight gradients
(f0aca7b) and its PyTorch-like memory limits (c4bf2d6), both prompted by
this example, against PyTorch 2.14.0 on MPS running the same algorithm
(`torch_baseline.py`, float16, as airbench runs). Spreads are sample
standard deviations; accuracy is on the full 10,000-image test set.

| | JAX (mtl), bf16 | PyTorch MPS, fp16 |
|---|---|---|
| test accuracy (5 seeds) | 94.02% ± 0.05% (4 of 5 reached 94%) | 93.93% ± 0.11% (1 of 5) |
| time to train (airbench's measure) | 260 ± 13 s | 254 ± 20 s |
| warmup (first epoch, compiling every program) | 30 s | 28 s (no compile; MPS graph setup) |
| peak memory footprint | 3.3 GB | 5.9 GB |

The two columns were measured in different conditions: PyTorch's 5 seeds
in a quiet window (no other GPU jobs or builds), JAX's in one process on
the busy machine, with memory pressure at "warn" for part of the run.
There JAX's runs 2-5 took 262-271 s against 237 s for run 1, consistent
with the plugin releasing its buffer cache on every free at warn (a
roadmap item); in the earlier quiet window, before c4bf2d6, seeds 1 and 5
took 229 and 259 s. So "the same speed" is conservative for JAX: JAX and
PyTorch reach the same accuracy (JAX slightly higher and less variable)
in about the same time, and JAX uses 45% less memory. Runs are
deterministic: a seed gives the same accuracy every time.

Before the plugin's weight-gradient change, the conv weight gradients
took 2.5-5x the forward time and the step 545 ms (about 259 s per run);
per layer at batch 1024 (bf16) they went from 54, 28, 53, 39 and 7 ms to
18, 10, 26, 23 and 4.5 ms.

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
applications has 1-2 GB free, and until c4bf2d6 the plugin's memory guard
refused any allocation that would leave less than 512 MB of it.

- **Keep the data as uint8 and normalize inside the jitted programs.**
  The training set is 150 MB as uint8 and 300 MB as bf16; an eager
  normalization also materializes float32 intermediates of 614 MB each.
  Normalizing inside the step fuses it into the first convolution's input.
- **Evaluate in small batches.** Test-time augmentation runs six forward
  passes per batch; at 2000 images that program needed 1.8 GB of scratch,
  at 500 images 0.45 GB.
- **Rematerialize the conv groups** (`jax.checkpoint`): the step's scratch
  at batch 1024 goes from 1.17 to 0.94 GB for 7% more time per step
  (`--no-remat` turns it off). Finer checkpoints did not lower it further.

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
  conv groups rematerialized in the backward pass (default; --no-remat).
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
  plugin now sizes the GEMM tile and split-K together (f0aca7b): 18 ms
  for that layer, and end to end JAX went from ~259 s to 229-259 s.
- **The memory guard decided whether training ran at all** on a busy
  8 GB machine. It refused any allocation that would leave less than
  512 MB of free memory, so the step (0.9 GB of scratch) was refused
  whenever less than ~1.4 GB was free: 25 of 28 attempts. PyTorch MPS,
  which caps only its own allocations, ran with a 4-6 GB footprint and
  let macOS make room. The plugin now does the same (c4bf2d6: a
  per-process budget, refusals only at critical system memory pressure),
  and the 5 JAX seeds above ran in one go on the same busy machine.
- **An out-of-memory error named the wrong program** (the next eager op,
  not the program whose allocation failed). Reported with a repro.
