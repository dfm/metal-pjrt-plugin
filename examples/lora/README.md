# LoRA fine-tuning: Qwen3 on WikiSQL in pure JAX

Fine-tunes [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) with LoRA
adapters to turn a table description and a question into SQL, in plain JAX
and optax on the Mac's GPU (this plugin), and compares it with
[`mlx_lm.lora`](https://github.com/ml-explore/mlx-lm) on the same data with
the same settings. The base model is `examples/llm/qwen3.py`'s, frozen in
bfloat16; only the adapters are trained (2.9 M parameters) and only they
have gradients and optimizer state.

| file | what |
|---|---|
| `lora.py` | data (WikiSQL, tokenization, mlx-lm-compatible batches), adapters, loss, jitted train step, save/load/merge |
| `train.py` | the training loop and command line; test-set exact match with the adapters merged into `examples/llm`'s generation engine |
| `gradcheck.py` | LoRA gradients on the GPU against float32 on CPU |
| `mlx_baseline.py` | the same fine-tune with `mlx_lm.lora` (run from its own environment) |
| `compare.py` | both, in interleaved rounds, as a markdown table |

**Data:** [`mlx-community/WikiSQL`](https://huggingface.co/datasets/mlx-community/WikiSQL),
1000 training / 100 validation / 100 test examples processed from
[salesforce/WikiSQL](https://github.com/salesforce/WikiSQL) (BSD 3-Clause
license); it is also mlx-lm's own LoRA example and `mlx_lm.lora`'s default
data. Each example becomes a chat turn: the user message is the table,
its columns and the question, the reply is the SQL; the loss covers only
the reply (mlx-lm's `--mask-prompt`).

## Run it

```sh
uv pip install --python .venv/bin/python -e '.[examples]'   # tokenizers, huggingface_hub, optax, pyarrow
export JAX_PLATFORMS=mtl,cpu
.venv/bin/python examples/lora/train.py \
    --iters 200 --save adapters.npz --test 100 --test-base [--grad-checkpoint]
.venv/bin/python examples/lora/gradcheck.py
# mlx-lm, in its own environment (not a dependency of this repo)
.venv/bin/python examples/lora/lora.py export ~/.cache/metal-pjrt-examples/wikisql
uv venv ~/.venvs/mlx && uv pip install --python ~/.venvs/mlx/bin/python mlx-lm
~/.venvs/mlx/bin/python examples/lora/mlx_baseline.py \
    --data ~/.cache/metal-pjrt-examples/wikisql --iters 200 --test 100
.venv/bin/python examples/lora/compare.py --data ~/.cache/metal-pjrt-examples/wikisql \
    --mlx-python ~/.venvs/mlx/bin/python --grad-checkpoint
```

Prefix the GPU commands with `scripts/device_lock.py --` (and give
`compare.py` `--wrap "scripts/device_lock.py --"`) when other GPU jobs may
run on the machine. The exported dataset is a cache; any directory works.

Settings (both sides): `mlx_lm.lora`'s defaults, i.e. rank 8, scale 20, all
seven projections (q, k, v, o, gate, up, down) of the last 16 of 28
layers, batch 4, learning rate 1e-5, seed 0; AdamW with weight decay 0.01
and bias correction (optax's; mlx-lm's is switched on in its config to
match). The batches are mlx-lm's exactly (`lora.batches`: same groups,
same shuffled order for the first epoch, same padding and loss mask), and
the tokenization matches `transformers`' chat template token for token.

## Results

M3 MacBook (10-core GPU, 8 GB), macOS 26.2, JAX 0.11.2, optax 0.2.8,
mlx-lm 0.31.3 (MLX 0.32.3), 2026-09-29.

**Quality**: 200 steps (0.8 epochs), validation loss on all 100
validation examples, and exact-match SQL on the first 100 test questions
(greedy, the adapters merged into the weights):

| | step 1 | 50 | 100 | 150 | 200 | exact match (base model: 0%) |
|---|---|---|---|---|---|---|
| JAX | 2.668 | 0.180 | 0.107 | 0.089 | 0.0815 | 45% |
| mlx-lm | 2.659 | 0.177 | 0.106 | 0.089 | 0.082 | 44% |

The training losses agree window by window as well (last window 0.069 vs
0.068). `gradcheck.py`: the adapters' gradients on the GPU differ from a
float32 CPU run by 9.7% (relative norm; random nonzero B on two examples),
the same bf16 CPU run by 10.7%: both are bf16 rounding, which happens in
different places on the two backends over 28 layers, so bf16 on the GPU
need not be closer to bf16 on CPU than to float32. With a float32 base
model on the GPU the gradients match CPU float32 to better than 1e-4 in
relative norm, and the loss to 1e-5 (6.67105 vs 6.67106; `gradcheck.py`),
which isolates the plugin's own error.

**Speed**, with gradient checkpointing, 60-step runs, median of 3
interleaved rounds (`compare.py`): trained tokens per second, the window
mean both sides report, JAX 184 and mlx-lm 187, i.e. the same speed
(mlx-lm 1.6% ahead). (An earlier version of this README compared step
times, "760 vs 781 ms", but those mixed JAX's window median with
mlx-lm's window mean; `train.py` now reports the window mean too, and a
re-measurement is pending.) The machine was shared with other GPU jobs
(serialized) and their memory, and single runs swung by up to 50%, so
only interleaved numbers are comparable.

**Memory**: the JAX process peaks at a 4.0-4.1 GB footprint with gradient
checkpointing (mlx-lm: 2.3 GB Metal peak; 3.1 GB without checkpointing).
Without checkpointing the JAX step needs a ~0.7-0.9 GB scratch buffer,
which the plugin's former system memory guard refused while other
processes held most of the 8 GB (the guard now refuses only at critical
system memory pressure, since 2026-09-29); with checkpointing it needs 122 MB. Most of the difference to mlx-lm is
compiled code: see the findings.

## What made it fast (and fit)

1. **Only the adapters are differentiated.** `jax.value_and_grad` takes
   the adapters as its argument and the bf16 base as a constant input, so
   the backward pass computes activation gradients through the adapted
   layers (and none below them) and no weight gradients for the base.
2. **A fused cross-entropy.** The loss over a 152k-word vocabulary needs
   `[batch, length, 151936]` float32 logits, 620 MB at length 256, and as
   much again for their gradient. `lora.cross_entropy` computes the logits
   32 positions at a time and, since the loss is the last thing computed,
   the gradient with respect to the hidden states right away,
   `((softmax - onehot) * weight) @ head` (a `jax.custom_vjp`). Nothing of
   the logits outlives its chunk and nothing is recomputed. The plain loss
   asked for a single 1.36 GB buffer (refused by the plugin's memory guard
   as it was before 2026-09-29); a
   `jax.checkpoint`ed chunked loss fit but recomputed the logits in the
   backward pass: at 4 x 161 tokens the fused version takes the step from
   711 to 626 ms and its scratch memory from 1.05 to 0.93 GB.
3. **Gradient checkpointing per layer** (`--grad-checkpoint`, as
   mlx-lm's): each adapted layer is a `jax.checkpoint`, recomputed in the
   backward pass; scratch memory 0.9 GB -> 122 MB for ~15% more time.
4. **Adapters in float32, base in bf16**, as mlx-lm: `(x A) B` in float32,
   cast to the activations' type, so the tiny rank-8 products cost
   little and the update is not rounded away.

## What it found in the plugin

- Nothing numerical: the loss curves match mlx-lm's to the third digit
  over 200 steps, and the gradients are as close to float32 as bf16 on
  CPU. No crashes or GPU errors.
- **Compiled programs cost host memory that is not given back.** Compiling
  the train step (about 3150 kernels) adds 160-570 MB of process footprint
  per batch shape, and deleting the executables frees none of it. With
  mlx-lm's padding (to multiples of 32) a run compiles 5 shapes for
  training and as many for validation, which is most of the gap between a
  4 GB footprint and mlx-lm's 2.3 GB. Reported with a repro; in the
  example, `--pad-to 128` (2 shapes) lowers the footprint to 3.4 GB at ~35%
  more time per epoch (longer padding). The size-class buffer cache is not
  the cause (an idle pause releases ~0.2 GB).
- The plugin's system memory guard refused the non-checkpointed step's
  scratch buffer twice while other processes held memory. The guard then
  refused any allocation that would leave less than 512 MB of free
  memory; since 2026-09-29 it works like PyTorch MPS's limits (a
  per-process budget, refusals only at critical system memory pressure),
  so `--grad-checkpoint` is a memory/speed trade-off rather than a
  requirement on a busy 8 GB Mac.

## Known gaps

- One epoch's data order matches mlx-lm; from the second epoch on,
  mlx-lm's validation passes shift its shuffle (they draw from the same
  global NumPy generator).
- The loss mask is mlx-lm's, which also scores the first padding position
  after each example (targets `offset <= j <= length`), to keep the two
  loss curves comparable.
- No LoRA for quantized bases (QLoRA), no dropout, no DoRA; batch size 4
  only tested; the 8 GB machine rules out full fine-tuning of the base.
