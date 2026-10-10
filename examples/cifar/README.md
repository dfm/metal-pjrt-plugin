# CIFAR-10 to 94%: airbench94 in pure JAX

A port of Keller Jordan's
[airbench94](https://github.com/KellerJordan/cifar10-airbench), the
fastest known recipe for training a CIFAR-10 classifier to 94% test
accuracy, to plain JAX and optax. It's a complete training workload: the
data lives on the GPU, augmentation runs there, and the model is a small
ResNet-style CNN with BatchNorm, trained with SGD, weight averaging and
test-time augmentation. On a MacBook Air it trains in about 3 minutes,
similar to the same algorithm in PyTorch on MPS.

## Run it

```sh
uv pip install --python .venv/bin/python -e '.[examples]'
.venv/bin/python examples/cifar/airbench.py --runs 5
```

The first run downloads CIFAR-10 (163 MB) from the authors' site into
`~/.cache/metal-pjrt-examples`. Each invocation starts with an untimed
one-epoch run that compiles everything, then trains `--runs` times and
reports each run's training time and test accuracy, as airbench does.
`--remat` recomputes activations in the backward pass, for less memory at
~27% more time.

| file | what it does |
|---|---|
| `airbench.py` | the model, training, evaluation and command line |
| `airbench_data.py` | CIFAR-10 loading and the hyperparameters (numpy only, shared by both versions) |
| `check.py` | compares one training step's loss and gradients with CPU |
| `torch_baseline.py`, `compare.py` | the same algorithm in PyTorch on MPS, and the protocol that compares the two |
| `conv_jax.py`, `conv_torch.py` | times airbench's convolutions one at a time, JAX vs PyTorch |
| `memprofile.py` | memory snapshots for `airbench.py --profile-memory` |
| `LICENSE-airbench` | airbench's MIT license, which covers the files derived from it |

## Results

On an M3 MacBook Air (8 GB, no fan), training time per run and test
accuracy over 3 seeds:

| | time | test accuracy |
|---|---|---|
| JAX, bf16 | 188 s | 94.03% |
| JAX, bf16, `--remat` | 240 s | 94.03% |
| PyTorch on MPS, fp16 | 220 s | 93.94% |

Both sides drew about the same GPU power (6-6.6 W). Runs are
deterministic: a seed gives the same accuracy every time. The default run
peaks at 3.5 GB of memory, within the plugin's default budget.

These times are for a GPU that has been under load for a while. A fanless
MacBook Air holds its top clock for about 3 minutes and then throttles,
so the comparison starts with an untimed warm-up run and alternates the
two sides. A single run from a cool start takes ~160 s. For scale,
airbench94 takes about 3.8 s on an A100.

`check.py` shows that one training step in float32 matches CPU to within
6e-5, except in two ill-conditioned weight gradients, where CPU float32 is
itself as far from float64. In bfloat16, both backends are ~20% from
float32.

Measured 2026-10-01 with JAX 0.11.2 and PyTorch 2.14.1, macOS 26.2.

## What it took

The port reproduced airbench94's accuracy on its first complete run.
Everything else was about memory: an 8 GB Mac shared with other
applications has only 1-2 GB free.

- **Keep the data as uint8 and normalize inside the jitted step.** The
  training set is 150 MB as uint8 and 300 MB as bf16, and normalizing it
  eagerly creates float32 intermediates of 614 MB each. Inside the step,
  the normalization fuses into the first convolution.
- **Evaluate in small batches.** Test-time augmentation runs six forward
  passes per batch: at 2000 images that needs 1.8 GB of scratch memory, at
  500 images 0.45 GB.
- **Keep step counters and batch selection on the device**, so a training
  step never waits on a value from Python.

Tried and not adopted: exact GELU instead of the tanh approximation (no
measurable difference), float16 instead of bfloat16 (12% slower), and
weight gradients computed as nine shifted GEMMs (faster only for layers
that are 4% of a step).

### Differences from airbench94

- Both versions warm up with a one-epoch run before timing (airbench uses
  a full-length run on random labels), and compute the whitening layer's
  eigendecomposition in float32 (PyTorch on the CPU, since MPS has no
  `eigh`).
- JAX computes in bfloat16 with float32 master weights and momentum.
  airbench and the PyTorch version keep the network in fp16 and BatchNorm
  in float32.
- PyTorch on MPS needs a few changes to fit in 8 GB: BatchNorm casts its
  input to float32 (MPS rejects fp16 input with float32 parameters), the
  layout is NCHW (channels-last used over 9 GB), normalization happens
  once on the CPU, and the allocator is capped and emptied every epoch.
- The model, hyperparameters, schedule, lookahead, augmentation and
  test-time augmentation are unchanged in both.

## Reproducing the numbers

PyTorch isn't a dependency of this repo, so it needs its own environment
(any location works; these commands use `~/.venvs/torch`).
`compare.py` runs the protocol behind the results: an untimed full-length
warm-up, then JAX and PyTorch alternating, one process per run.

```sh
.venv/bin/python examples/cifar/check.py
uv venv ~/.venvs/torch
uv pip install --python ~/.venvs/torch/bin/python torch numpy
.venv/bin/python examples/cifar/compare.py --torch-python ~/.venvs/torch/bin/python \
    --wrap "scripts/device_lock.py --" --one-process
```

Prefix the GPU commands with `scripts/device_lock.py --` when other GPU
jobs may run. Useful options:

- `compare.py --sampler PATH` records GPU power and performance state
  during the runs, given a sampler program (not part of this repo) that
  prints them; `--jax-variants "default=;remat=--remat"` adds JAX arms.
- `--warm-full` (`airbench.py` and `torch_baseline.py`) does an untimed
  full-length run first, to time a single side at the same thermal state.
- `airbench.py --profile-memory` prints per-epoch time, memory footprint,
  macOS memory pressure and the plugin's allocator counters.
- To run `torch_baseline.py` on its own, cap PyTorch's MPS allocator
  with `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.6`,
  as `compare.py` does; without a cap its cache grows past 9 GB.
