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
compiles every program; its time is reported separately (`warmup_run_s`).

## Results

Measured 2026-09-29 on an M3 MacBook (10-core GPU, 8 GB), macOS 26.2,
JAX 0.11.2 with optax, against PyTorch 2.14.0 on MPS running the same
algorithm (`torch_baseline.py`, float16, as airbench runs).

| | JAX (mtl), bf16 | PyTorch MPS, fp16 |
|---|---|---|
| test accuracy | 94.01% (1 run) | 93.93% ± 0.10% (5 seeds; 1 of 5 reached 94%) |
| time to train (airbench's measure) | 289 s | 251 ± 19 s (225 s in an earlier single run) |
| train step at batch 1024, steady state | 509 ms (545 with rematerialization) | ~460-500 ms (train time / 476 steps) |
| compile / warmup (1 epoch incl. compiling everything) | 36 s | 27-38 s (no compile; MPS graph setup) |
| peak memory footprint | 2.7-3.4 GB | 3.9-5.9 GB |

The JAX run above used rematerialization (the default, see below) and
synchronized once per epoch to log the loss. The planned 5-seed JAX
measurement did not complete: the plugin's memory guard refused the
training step on 13 attempts over two hours, whenever the machine
(with other applications open) had less than ~1.4 GB free; see "What it
found in the plugin". PyTorch, which has no such guard, completed its 5
seeds in the same conditions by going into swap.

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
applications has 1-2 GB free, and the plugin's memory guard refuses an
allocation that would leave less than 512 MB for the system.

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
- **Run the seeds in one process.** Once the first run's buffers are
  allocated, the plugin's buffer cache serves the later runs; separate
  processes each have to get past the memory guard again.

Measured and not adopted: exact GELU vs the tanh approximation (no
measurable difference), float16 instead of bfloat16 (12% slower), and
computing the conv weight gradients as 9 shifted GEMMs (faster only for
the 7x7 and 3x3 layers, 4% of a step, not worth a custom gradient).

The PyTorch baseline needed four changes from airbench's CUDA code to run
on MPS on this machine: BatchNorm computes in float32 (MPS's kernel
rejects fp16 input with float32 parameters), NCHW instead of
channels-last (with channels-last its footprint grew past 9 GB), a cap on
the MPS allocator's cache (`PYTORCH_MPS_HIGH_WATERMARK_RATIO`), and
airbench's crop by masks rather than a first attempt with advanced
indexing (whose int64 index tensors took 5 GB).

## What it found in the plugin

- **The convolutions are correct.** One training step's gradients in
  float32 match CPU float32 to ~1e-5 for every parameter (`check.py`); in
  bfloat16 they are as far from float32 as CPU bfloat16 is.
- **Conv weight gradients are the bottleneck of CNN training.** Per layer
  at batch 1024 (bf16), the forward and input-gradient convolutions run
  near PyTorch MPS's speed, but the weight gradient takes 2.5-5x the
  forward time (31x31, 24 -> 64 channels: 54 ms, where PyTorch's whole
  backward is 25 ms). It is a reduction over batch x pixels (~1M) into a
  small output. Reported with a standalone benchmark.
- **The memory guard decides whether training runs at all** on a busy
  8 GB machine. It refused the step (0.9 GB of scratch) whenever less than
  ~1.4 GB was free, which on this machine was most of the time; PyTorch
  MPS, which has no such guard, ran with a 4-6 GB footprint and pushed the
  machine into swap instead. The trade-off (refuse vs swap under GPU load)
  is an open decision for the plugin; the numbers are recorded there.
- **An out-of-memory error named the wrong program** (the next eager op,
  not the program whose allocation failed). Reported with a repro.
