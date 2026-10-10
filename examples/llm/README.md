# LLM inference: Qwen3 in pure JAX

A small, self-contained implementation of
[Qwen3](https://huggingface.co/Qwen/Qwen3-0.6B) inference in plain JAX: no
Flax, no model library. It loads the Hugging Face checkpoint directly,
generates text with a KV cache, and can quantize the weights to int8 or
int4. Decoding runs at about the same speed as
[mlx-lm](https://github.com/ml-explore/mlx-lm) on the same machine.

## Run it

```sh
uv pip install --python .venv/bin/python -e '.[examples]'
.venv/bin/python examples/llm/generate.py \
    "Give me three facts about Jupiter." --quant int4
```

The first run downloads Qwen3-0.6B (1.4 GB) into the Hugging Face cache.
The weights are Apache-2.0, from the Qwen team, and aren't redistributed
here.

Useful options:

- `--model Qwen/Qwen3-1.7B`: any dense Qwen3 checkpoint in bf16, f16 or
  f32. FP8 checkpoints and YaRN `rope_scaling` aren't supported.
- `--quant int8` or `--quant int4`: weight-only quantization, in groups of
  64.
- `--temperature`, `--top-k`, `--top-p`: sampling (greedy by default).
- `--think`: Qwen3's reasoning mode (off by default).
- `--fused`: run the whole decode loop as one jitted `while_loop` (no
  streaming).

| file | what it does |
|---|---|
| `qwen3.py` | the model: weight loading, forward pass, quantization, sampling |
| `generate.py` | the jitted prefill and decode engine, and the command line |
| `check.py` | compares the output distributions with a float32 run on CPU |
| `bench.py` | measures prefill and decode speed |
| `mlx_baseline.py`, `compare.py` | the same measurements with mlx-lm, and both side by side |
| `skinny_jax.py`, `skinny_mlx.py` | few-row GEMMs (batched decode's case), JAX vs MLX |

## Results

On an M3 MacBook Air (8 GB), Qwen3-0.6B, decode after a 128-token prompt,
in ms per token:

| | bf16 | int8 | int4 |
|---|---|---|---|
| JAX | 13.5 | 7.6 | 5.0 |
| mlx-lm | 14.6 | 8.0 | 4.8 |

Prefill is similar too, except for short quantized prompts, where mlx-lm
is ahead: this example dequantizes each weight matrix to bf16 before
multiplying, a fixed cost per call that mlx-lm avoids by reading the
packed weights directly. Larger models behave the same way: Qwen3-1.7B
decodes at 11-12 ms per token in int4 on both, and Qwen3-4B fits on an
8 GB Mac in int4 at ~27 ms per token.

Quantization costs about what it costs in mlx-lm. Against float32 on CPU,
on the first 512 tokens of this repository's design notes:

| | KL from float32 | top-1 agreement | perplexity (float32: 64.9) |
|---|---|---|---|
| JAX bf16 | 5.7e-4 | 0.990 | 64.7 |
| mlx-lm bf16 | 7.5e-4 | 0.984 | 64.8 |
| JAX int8 | 2.4e-3 | 0.967 | 65.2 |
| mlx-lm 8-bit | 3.4e-3 | 0.963 | 65.4 |
| JAX int4 | 0.24 | 0.762 | 75.9 |
| mlx-lm 4-bit | 0.24 | 0.770 | 75.3 |

The float32 reference is this example's own model, so it scores how well
each side's reduced precision tracks full precision, not how faithful
`qwen3.py` is to Qwen3.

Peak memory with `generate.py` is 1.9 GB for Qwen3-0.6B in bf16 (1.1 GB
in int4) and 4.5 GB for Qwen3-1.7B (2.1 GB in int4).

Measured 2026-09-28 with JAX 0.11.2 and mlx-lm 0.31.3, macOS 26.2. For
the methodology and the full set of numbers, see
[Reproducing the numbers](#reproducing-the-numbers).

## What made it fast

The first version worked on the first try and decoded at 35 tokens per
second; the current one does 200 in int4. None of the changes needed
anything beyond ordinary JAX, and most apply to any XLA backend.

| step | ms/token |
|---|---|
| first version: `lax.scan` over stacked layers | 28.5 |
| unrolled layers, one cache per layer | 21.7 |
| fused q/k/v and gate/up products, attention over a window | 13.3 |
| int8 weights | 7.6 |
| int4 weights, dequantize then multiply | 14.4 |
| int4 weights, `_matvec_int4` | 5.3 |
| int4, values cached as `[..., dim, pos]` | 4.8 |

1. **Don't `scan` over layers that carry a KV cache.** With the cache as
   the scan's input and output, XLA copies each layer's cache in and out
   on every call, so the cost grows with the cache size. A Python loop
   over per-layer weights and caches writes one row per layer per token,
   and still compiles in about a second.
2. **Donate the cache**, or every decode step copies it.
3. **Keep the decode state on the device and dispatch ahead.** The token,
   the cache, the position and the RNG key are inputs and outputs of one
   jitted step, so each token costs one dispatch. The loop dispatches the
   next step before reading the current token back, which hides the host
   round trip. A fully fused `lax.while_loop` is no faster.
4. **For one token, a matmul is a reduction.** XLA turns a
   vector-matrix product into a multiply-reduce kernel. Small products
   read memory more slowly than large ones, so q/k/v and gate/up are each
   concatenated into one product.
5. **Attend over a window, not the whole cache.** Decode compiles one step
   per window size and reads only that much of the cache, so a 4096-token
   cache costs the same as a 256-token one for a short context. Long
   prompts are prefilled in 512-token chunks to keep the attention scores
   small.
6. **Put the reduced axis last.** Keys are cached as `[..., pos, dim]` and
   values as `[..., dim, pos]`, so both attention reductions run along the
   minor axis. Storing values like keys made XLA transpose them on every
   token.
7. **Quantization is almost free if the dequantization fuses into the
   reduction.** For int8 it does as written. For int4 (two weights per
   byte) the obvious unpack-and-dequantize version is no faster than
   bf16; `_matvec_int4` splits the input to match the packing and applies
   each group's scale once, which reads the weights at full bandwidth.
8. **Measure the quality, not just the speed.** `check.py` compares the
   next-token distributions with float32. Where the 4-bit grid sits
   matters: putting zero on the grid, as MLX does, took the KL divergence
   from 0.33 to 0.24.

Decode is now bound by memory bandwidth: each token reads every weight,
and the weight products run at about the M3's streaming bandwidth. What's
left is the ~570 small kernels per token (norms, RoPE, cache updates,
attention), about 1 ms in int4.

## Limitations

- One prompt at a time: no continuous batching, paged cache or prompt
  caching. The KV cache is allocated at `--max-len` up front.
- Each prompt-length bucket and attention window compiles once per
  process, in 1-2 s. `generate.py` compiles what a reply needs before it
  starts, but a long generation in your own code can pause when it
  crosses into a new window.

## Reproducing the numbers

Run one GPU job at a time; `scripts/device_lock.py` serializes them.

```sh
# numerics against float32 on CPU (exits 1 on a bf16 mismatch)
scripts/device_lock.py -- .venv/bin/python examples/llm/check.py --quant int4
# prefill and decode speed
scripts/device_lock.py -- .venv/bin/python examples/llm/bench.py --quant int4
```

mlx-lm isn't a dependency of this repo, so the comparison needs its own
environment (any location works; these commands use `~/.venvs/mlx`):

```sh
uv venv ~/.venvs/mlx
uv pip install --python ~/.venvs/mlx/bin/python mlx-lm
.venv/bin/python examples/llm/compare.py --mlx-python ~/.venvs/mlx/bin/python \
    --wrap "scripts/device_lock.py --"
```

`compare.py` runs every arm in interleaved rounds (one process each, so
heat and background load hit them alike) and prints the median as a
markdown table. The results above are the median of 3 rounds, with the
same weights quantized the same way on both sides.

The quality table scores the version of `docs/design.md` from the commit
that published it; the file has changed since:

```sh
git show f633328:docs/design.md > design.md
scripts/device_lock.py -- .venv/bin/python examples/llm/check.py \
    --text-file design.md --tokens 512 --save-ref ref.npz
scripts/device_lock.py -- ~/.venvs/mlx/bin/python examples/llm/mlx_baseline.py \
    --kl-ref ref.npz
```

Add `--quant int8` or `--quant int4` to `check.py`, and `--q-bits 8` or
`--q-bits 4` to `mlx_baseline.py`, for the quantized rows.
