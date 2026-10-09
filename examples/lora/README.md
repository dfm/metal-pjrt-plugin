# LoRA fine-tuning: Qwen3 on WikiSQL in pure JAX

Fine-tunes [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) with LoRA
adapters to turn a table description and a question into SQL, in plain
JAX and optax. The base model is the one from
[`examples/llm`](../llm), frozen in bfloat16; only the adapters (2.9 M
parameters) are trained. With the same data and settings as
[`mlx_lm.lora`](https://github.com/ml-explore/mlx-lm), it reaches the same
losses and accuracy at a similar speed.

## Run it

```sh
uv pip install --python .venv/bin/python -e '.[examples]'
JAX_PLATFORMS=mtl,cpu .venv/bin/python examples/lora/train.py \
    --iters 200 --grad-checkpoint --save adapters.npz --test 100 --test-base
```

This trains for 200 steps, saves the adapters, and then scores the base
model and the fine-tuned one on 100 test questions. The first run
downloads the model and the dataset from the Hugging Face Hub.
`--grad-checkpoint` recomputes each layer in the backward pass, trading
~15% more time for much less memory; on a machine with memory to spare,
leave it off.

The data is [`mlx-community/WikiSQL`](https://huggingface.co/datasets/mlx-community/WikiSQL)
(1000 training, 100 validation and 100 test examples from
[WikiSQL](https://github.com/salesforce/WikiSQL), BSD 3-Clause), which is
also mlx-lm's own LoRA example. Each example becomes a chat turn: the
user message holds the table, its columns and the question, and the reply
is the SQL. Only the reply counts toward the loss.

| file | what it does |
|---|---|
| `lora.py` | data loading and batching, adapters, loss, the jitted train step, saving and merging |
| `train.py` | the training loop and command line, and the test-set evaluation |
| `gradcheck.py` | compares the adapters' gradients with float32 on CPU |
| `mlx_baseline.py`, `compare.py` | the same fine-tune with `mlx_lm.lora`, and both side by side |

## Results

On an M3 MacBook Air (8 GB), 200 steps (0.8 epochs), validation loss on
all 100 validation examples and exact-match SQL on 100 test questions
(greedy decoding, adapters merged into the weights):

| | step 1 | 50 | 100 | 150 | 200 | exact match |
|---|---|---|---|---|---|---|
| JAX | 2.668 | 0.180 | 0.107 | 0.089 | 0.0815 | 45% |
| mlx-lm | 2.659 | 0.177 | 0.106 | 0.089 | 0.082 | 44% |

The base model gets none of the test questions right. With gradient
checkpointing, a step takes 0.6-0.75 s on both sides, and the JAX process
peaks at about 4 GB of memory, against mlx-lm's 2.3 GB. Most of that
difference is compiled code: each distinct batch shape compiles its own
train step. `--pad-to 128` pads batches to fewer shapes, which lowers the
peak to 3.4 GB at ~35% more time per epoch.

`gradcheck.py` shows the gradients are as accurate as bfloat16 allows: with
a bf16 base model they're ~10% from float32 on both the GPU and the CPU,
and with a float32 base model the GPU matches the CPU to better than 1e-4.

Measured 2026-09-29 and 2026-10-01 with JAX 0.11.2, optax 0.2.8 and
mlx-lm 0.31.3, macOS 26.2.

## What made it fast (and fit)

1. **Differentiate only the adapters.** `jax.value_and_grad` takes the
   adapters as its argument and the base weights as a constant, so the
   backward pass computes no weight gradients for the base model and
   stops at the lowest adapted layer.
2. **Fuse the cross-entropy.** Over a 152k-token vocabulary, the logits
   for one batch take 620 MB in float32, and their gradient as much again.
   `lora.cross_entropy` computes the logits 32 positions at a time and,
   since the loss is the last thing computed, their gradient right away
   (a `jax.custom_vjp`). No chunk of logits outlives its loop iteration,
   and nothing is recomputed.
3. **Checkpoint per layer** (`--grad-checkpoint`): each adapted layer is
   wrapped in `jax.checkpoint`, which cuts the step's scratch memory from
   0.9 GB to 122 MB.
4. **Keep the adapters in float32 and the base in bf16**, as mlx-lm does.
   The rank-8 products are cheap in float32, and small updates aren't
   rounded away.

## Limitations

- No QLoRA (LoRA on a quantized base), dropout or DoRA. Only batch size 4
  has been tested.
- The first epoch's data order matches mlx-lm's exactly; later epochs
  don't.
- The loss mask is mlx-lm's, which also scores the first padding position
  after each example. This keeps the two loss curves comparable.

## Reproducing the numbers

Both sides use `mlx_lm.lora`'s defaults: rank 8, scale 20, all seven
projections of the last 16 of 28 layers, batch 4, learning rate 1e-5,
seed 0, and AdamW with weight decay 0.01. The batches are mlx-lm's
exactly (same groups, order, padding and loss mask), and the tokenization
matches `transformers`' chat template token for token.

mlx-lm isn't a dependency of this repo, so it needs its own environment
(any location works; these commands use `~/.venvs/mlx`), and it reads the
dataset from a directory that `lora.py` exports:

```sh
export JAX_PLATFORMS=mtl,cpu
.venv/bin/python examples/lora/gradcheck.py
.venv/bin/python examples/lora/lora.py export ~/.cache/metal-pjrt-examples/wikisql
uv venv ~/.venvs/mlx
uv pip install --python ~/.venvs/mlx/bin/python mlx-lm
~/.venvs/mlx/bin/python examples/lora/mlx_baseline.py \
    --data ~/.cache/metal-pjrt-examples/wikisql --iters 200 --test 100
.venv/bin/python examples/lora/compare.py --data ~/.cache/metal-pjrt-examples/wikisql \
    --mlx-python ~/.venvs/mlx/bin/python --grad-checkpoint \
    --wrap "scripts/device_lock.py --"
```

Prefix the GPU commands with `scripts/device_lock.py --` when other GPU
jobs may run. `compare.py` alternates short runs of the two in rounds, so
that heat and background load hit both alike, and reports each side's
median step time. The speed above is from 3 rounds of 60 steps; on a
fanless laptop that throttles, individual rounds varied by up to 25%.
