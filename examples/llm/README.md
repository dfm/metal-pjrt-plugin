# LLM inference: Qwen3 in pure JAX

A small, self-contained implementation of [Qwen3](https://huggingface.co/Qwen/Qwen3-0.6B)
inference in plain JAX (no Flax, no model library), run on the Mac's GPU by
this plugin. It loads the Hugging Face checkpoint directly, generates text
with a KV cache, and optionally quantizes the weights to int8 or int4.
It doubles as a case study: what it takes to make LLM decoding fast in
JAX on Apple Silicon, measured against [mlx-lm](https://github.com/ml-explore/mlx-lm).

| file | what |
|---|---|
| `qwen3.py` | the model: weight loading (safetensors, parsed by hand), forward pass, quantization, sampling |
| `generate.py` | the engine (jitted prefill and decode) and a command line |
| `check.py` | numerical check against a float32 CPU run (KL divergence, perplexity) |
| `bench.py` | prefill and decode throughput |
| `mlx_baseline.py` | the same measurements with mlx-lm (run from its own environment) |
| `compare.py` | both, in interleaved rounds, as a markdown table |
| `skinny_jax.py`, `skinny_mlx.py` | few-row bf16 GEMMs (batched decode's case), JAX vs MLX |

## Run it

```sh
uv pip install --python .venv/bin/python -e '.[examples]'   # tokenizers, huggingface_hub
JAX_PLATFORMS=mtl,cpu .venv/bin/python examples/llm/generate.py \
    "Give me three facts about Jupiter." --quant int4
```

The first run downloads Qwen3-0.6B (1.4 GB) into the Hugging Face cache.
The Qwen3 weights are Apache-2.0 (Qwen team); they are downloaded from the
Hub, not redistributed here.
Options: `--model Qwen/Qwen3-1.7B` (any dense Qwen3 checkpoint in
bf16/f16/f32: no FP8 checkpoints, no YaRN `rope_scaling`; the loader
refuses them), `--quant int8|int4`,
`--temperature/--top-k/--top-p` (greedy by default), `--think` (Qwen3's
reasoning mode, off by default), `--fused` (the whole decode loop as one
jitted `while_loop`; no streaming).

Measurements (run one GPU job at a time; the repo's `scripts/device_lock.py`
serializes them):

```sh
export JAX_PLATFORMS=mtl,cpu
# numerics against float32 on CPU (exit 1 on a bf16 mismatch)
scripts/device_lock.py -- .venv/bin/python examples/llm/check.py [--quant int8|int4]
scripts/device_lock.py -- .venv/bin/python examples/llm/check.py --text-file docs/design.md --tokens 512
# throughput: prefill_T, decode (streaming), decode_sync, decode_fused
scripts/device_lock.py -- .venv/bin/python examples/llm/bench.py [--quant int4] [--batch 2]
# mlx-lm needs its own environment (it is not a dependency of this repo)
uv venv ~/.venvs/mlx && uv pip install --python ~/.venvs/mlx/bin/python mlx-lm
scripts/device_lock.py -- ~/.venvs/mlx/bin/python examples/llm/mlx_baseline.py --q-bits 4
# both, interleaved, as a markdown table
.venv/bin/python examples/llm/compare.py --mlx-python ~/.venvs/mlx/bin/python \
    --wrap "scripts/device_lock.py --"
```

## Results

Measured 2026-09-28 on an M3 MacBook (10-core GPU, 8 GB), macOS 26.2,
JAX 0.11.2, against mlx-lm 0.31.3 (MLX 0.32.3) with the same weights
quantized the same way (groups of 64). Qwen3-0.6B, median of 3
interleaved rounds (`compare.py`); prefill in ms (tokens/s), decode in
ms per token (tokens/s) after a 128-token prompt:

| | JAX bf16 | mlx-lm bf16 | JAX int8 | mlx-lm 8-bit | JAX int4 | mlx-lm 4-bit |
|---|---|---|---|---|---|---|
| prefill 16 tokens | 26.5 (603) | 28.6 (560) | 36.5 (438) | 23.0 (696) | 32.9 (486) | 23.4 (683) |
| prefill 128 | 49.6 (2579) | 63.1 (2030) | 60.1 (2128) | 62.5 (2048) | 57.0 (2245) | 64.5 (1983) |
| prefill 512 | 201.6 (2540) | 217.7 (2352) | 212.9 (2405) | 238.1 (2150) | 210.2 (2436) | 238.2 (2150) |
| decode, streaming | 13.5 (74) | 14.6 (69) | 7.6 (132) | 8.0 (124) | 5.0 (202) | 4.8 (209) |
| decode, fused loop | 13.4 (75) | | 7.5 (134) | | 4.8 (209) | |

Qwen3-1.7B, 2 interleaved rounds of the quantized arms: decode int8
19.4 ms/token (mlx-lm 21.5), int4 11.3 (11.8); prefill of 128 tokens
int8 175 ms (168), int4 168 (174); of 16 tokens 109-115 (45-46). In
bf16 (3.4 GB, ~40 ms/token at the bandwidth limit) both land at 40-47
ms/token, prefill 128 tokens 134 ms (158). On the 8 GB machine, rounds
that include the bf16 arms slow the arms after them (quantized decode
up to ~18 ms for JAX, ~13 for mlx-lm): measure large models one
precision at a time.

Qwen3-4B, which needs 8 GB in bf16, runs in int4 on the 8 GB machine
(peak footprint 3.7 GB): decode 26.9-28.5 ms/token (mlx-lm with
`mlx-community/Qwen3-4B-4bit`, the same format: 26.8), prefill of 128
tokens 434 ms (394-427), of 16 tokens 273 ms (99). Its float32 reference
does not fit in memory, so `check.py` cannot score it here.

Long contexts (Qwen3-0.6B, max_len 4096, single runs): decode after a
1024-token prompt, int4, 6.4 ms/token (mlx-lm 6.0); after 3072 tokens
10.1 (8.6). Prefill of a 3072-token prompt: bf16 1.89 s (mlx-lm 1.87),
int4 1.96 s (2.31). Attention reads a window of the cache a little
larger than the context (up to a quarter more, at least 256 slots),
where mlx-lm's cache is
exactly the context.

Quality, on the first 512 tokens of `docs/design.md` (as of f4e2691)
against float32 on CPU (`check.py --text-file`; mlx-lm via
`mlx_baseline.py --kl-ref`). The reference is this example's own model
run in float32 on the CPU, not an independent implementation: it scores
both sides fairly only as far as `qwen3.py` itself matches Qwen3 (its
bf16 run and mlx-lm's are equally close to it). The mlx-lm rows were
scored against a float16-rounded copy of the reference, JAX's against
float32; `check.py --save-ref` now writes float32, and a re-scoring is
pending.

| | KL from float32 | top-1 agreement | perplexity (float32: 64.9) |
|---|---|---|---|
| JAX bf16 | 5.7e-4 | 0.990 | 64.7 |
| mlx-lm bf16 | 7.5e-4 | 0.984 | 64.8 |
| JAX int8 | 2.4e-3 | 0.967 | 65.2 |
| mlx-lm 8-bit | 3.4e-3 | 0.963 | 65.4 |
| JAX int4 | 0.24 | 0.762 | 75.9 |
| mlx-lm 4-bit | 0.24 | 0.770 | 75.3 |

Decode is bound by memory bandwidth: every token reads all the weights
(1.2 GB in bf16, 0.34 GB in int4). The weight products run at about the
M3's streaming bandwidth; what is left is ~570 small kernels per token
(~1 ms at int4). Prefill is bound by the GEMMs. Short quantized prompts
are the one place mlx-lm is clearly ahead: prefill dequantizes every
matrix to bf16 for its GEMMs, a fixed cost per call that grows with the
model (6-10 ms at 0.6B, ~175 ms at 4B). mlx-lm's GEMMs read the packed
weights directly; the plugin has no such kernel.

Memory (peak footprint of `generate.py`, 64 new tokens, max_len 1024):
Qwen3-0.6B 1.9 GB in bf16, 1.1 GB in int4; Qwen3-1.7B 4.5 GB in bf16,
2.1 GB in int4. On an 8 GB Mac, run nothing else heavy (a Bazel build of
the plugin takes ~4.5 GB) next to 1.7B in bf16.

Known gaps:

- **Batched decode** (B sequences at once) was slow in bf16 when measured
  here: 13.4 -> 26.8 ms per step at B = 2, because the plugin's GEMMs with
  2-64 rows ran at ~23 GB/s. The plugin has since fixed few-row GEMMs
  (CHANGELOG.md; `skinny_jax.py` / `skinny_mlx.py` here reproduce the
  kernel numbers in docs/performance.md): bf16 decode at B = 2 is now
  13.9 ms per step (B = 1: 13.3), per the plugin's measurement; the other
  numbers on this page were not re-measured. int8/int4 up to 4 sequences
  use reductions over the stored weights (`ROWS_MAX` in `qwen3.py`):
  int4 did 234 tokens/s at B = 2.
- One prompt at a time: no continuous batching, no paged cache, no
  prompt caching across calls. The KV cache is allocated at `--max-len`
  up front (attention reads only a window of it).
- Compile time is ~1-2 s per shape; each prompt-length bucket (powers of
  two from 16, prompts over 512 tokens in 512-token chunks) and each
  attention window (4 per doubling of the context) compiles once per
  process, so a long generation pauses briefly when it crosses into a
  new window unless `Engine.warmup` compiled it first (the CLI does).

## What made it fast

The first version worked on the first try and decoded at 35 tokens/s; the
final one is several times faster. None of the changes needed anything
beyond ordinary JAX, and most of them apply to any XLA backend.

Decode time per token, bf16 unless noted, M3 (10-core GPU, 8 GB):

| step | ms/token |
|---|---|
| first version: `lax.scan` over stacked layers, cache `[layer, batch, pos, head, dim]`, context 128, max_len 1024 | 28.5 |
| the same with max_len 4096 | 65 |
| unrolled layers, head-major per-layer caches (max_len 1024 / 4096) | 21.7 / 39.8 |
| buffer donation for `mtl` (a plugin fix this example prompted; max_len 4096) | 27.2 |
| fused q/k/v and gate/up products, checkpoint `[out, in]` layout, attention window (any max_len) | 13.3 |
| int8 weights | 7.6 |
| int4 weights, dequantize-then-dot | 14.4 |
| int4 weights, `_matvec_int4` | 5.3 |
| int4, values cached as `[.., dim, pos]` | 4.8 |

1. **Don't `scan` over layers when they carry a KV cache.** With the cache
   as the scan's input and output, XLA slices each layer's cache in,
   writes it back out whole and zero-fills a new output cache every call,
   so the step cost grew steeply with max_len. A Python loop over a list
   of per-layer weights and caches (unrolled at trace time; compile stays
   ~1 s) writes one row per layer per token.
2. **Donate the cache.** Without donation every decode step copies the
   whole cache. JAX keeps a hard-coded list of platforms that support
   donation, and `mtl` was not on it; the plugin now adds itself.
3. **Keep decode state on the device and dispatch ahead.** The token, the
   cache, the position and the RNG key are all inputs and outputs of one
   jitted step, so a token costs exactly one dispatch. The loop dispatches
   step i + 1 before reading token i back (`generate.Engine.generate`),
   which hides the ~2.5-3 ms host round trip (`decode_sync` vs `decode` in
   `bench.py`). A fully fused `lax.while_loop` is no faster than that.
4. **For one token, a matmul is a reduction.** XLA rewrites a
   vector-matrix product into a multiply-reduce kernel rather than calling
   the GEMM library. All of a step's products together read the weights
   at ~90 GB/s of GPU time, about what a plain streaming sum reaches, but
   small products read slower than large ones, so q/k/v and gate/up are
   concatenated into one product each.
5. **Attend over a window, not the whole cache.** Decode compiles one
   step per window size (256, 512, 768, 1024, 1280, ... max_len: an
   eighth of a power of two apart) and reads only that much of the cache;
   written as multiply-reduce (not einsum), the reductions read the
   window in place. max_len 4096 then costs the same as 256 for a short
   context. Long prompts are prefilled in 512-token chunks, which keeps
   the attention scores small (one 4096-token chunk: 1 GB per layer).
6. **Put the reduced axis last.** Keys are cached `[.., pos, dim]` and
   values `[.., dim, pos]`: both attention reductions then run along the
   minor axis. With values stored like keys, XLA transposed every layer's
   value window on every token, 10% of an int4 step.
7. **Weight-only quantization is almost free, if the dequantization fuses
   into the reduction.** For int8 (a scale per 64 weights) it does as
   written. For int4 (two weights per byte, affine per 64) the obvious
   unpack-concatenate-dequantize version fuses but is compute-bound and no
   faster than bf16; `_matvec_int4` instead splits the input by half-group
   to match the packing and applies the scale once per group, which reads
   the packed weights at the bandwidth limit.
8. **Measure the quantization, not just the speed.** `check.py` compares
   the next-token distributions with float32 on CPU. int8 costs little;
   int4 costs a lot on a model this small (perplexity +17%), as it does in
   mlx-lm. Where the 4-bit grid sits matters beyond its RMS error: putting
   zero on the grid, as MLX does, took the KL divergence from 0.33 to 0.24.

Tried and dropped: folding the RMSNorm weights into the following matrix
(fewer kernels in principle; 20% slower at int4) and normalizing the
softmax after the value product (no measurable change).

## What it found in the plugin

The model ran correctly on the first attempt: its bf16 logits are as close
to float32 as mlx-lm's (KL 5.7e-4 vs 7.5e-4 on 512 tokens) and about as
close as the same code in bf16 on the CPU, with no crashes or GPU errors
across a night of benchmarking. Along the way:

- **Buffer donation was off for `mtl`**, silently: JAX's hard-coded list
  of donation platforms did not include it, so every donated argument was
  copied. Fixed in the plugin (CHANGELOG.md, "Unreleased": "Buffer
  donation works on mtl"), which adds `mtl` at load time;
  a decode step at max_len 4096 went from 39.8 to 27.2 ms.
- **Decode is bandwidth-bound, and at Qwen3-0.6B's shapes the plugin
  reaches the bandwidth.** The weight products are XLA's MLIR reduction
  kernels (not the BlasLt GEMM path) and together read at about the rate
  of a plain streaming sum. Some shapes read slower: products with few
  output rows (1024 x 3072 int4: ~24 GB/s) and mid-size int4 products
  (60-80 GB/s), where MLX's hand-written kernels get ~90. Tuning the
  reduction emitter for these would speed up every quantized model.
- **GEMMs with 2-64 rows ran at ~23 GB/s** regardless of the row count
  (MLX: ~90 up to 8 rows), which made batched bf16 decode slow. Reported
  with a standalone repro (`skinny_jax.py`, `skinny_mlx.py`); fixed in
  the plugin since (see Known gaps).
- **What remains is kernel count.** An int4 decode step is ~570 kernels
  (20 per layer), and everything but the weight products (~3.8 ms) costs
  ~1 ms: norms, rope, the cache update, attention's reductions. A cheaper
  dispatch or fewer, larger fusions would show up directly here
  (`bench.py --quant int4 --max-len 256 --cases decode_fused`).
