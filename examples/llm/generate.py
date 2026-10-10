# Copyright 2026 The jax-graft Authors
# SPDX-License-Identifier: Apache-2.0
"""Generate text with Qwen3 in pure JAX.

  .venv/bin/python examples/llm/generate.py "Why is the sky blue?"

The prompt is wrapped in Qwen3's chat template (thinking off unless
--think). Decoding runs one jitted step per token from Python, which lets
it stream; --fused runs the whole decode loop as one jitted while loop.
"""
import argparse, functools, os, sys, time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qwen3  # noqa: E402

EOS = (151645, 151643)  # <|im_end|>, <|endoftext|>
MIN_WINDOW = 256
PREFILL_CHUNK = 512


def chat_prompt(text, think=False):
    p = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
    return p if think else p + "<think>\n\n</think>\n\n"


def bucket(n, minimum=16):
    """Prompt lengths are padded to a power of two to bound recompiles."""
    return max(minimum, 1 << (n - 1).bit_length())


class Engine:
    def __init__(self, repo="Qwen/Qwen3-0.6B", *, max_len=1024, batch=1,
                 dtype=jnp.bfloat16, quant=None, sampling=None):
        path = qwen3.model_dir(repo)
        self.cfg = qwen3.Config.from_json(os.path.join(path, "config.json"))
        self.params = qwen3.load_params(path, self.cfg, dtype, quant=quant)
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(os.path.join(path, "tokenizer.json"))
        self.max_len, self.batch, self.dtype = max_len, batch, dtype
        self.sampling = dict(sampling or {})
        cfg, smp = self.cfg, self.sampling

        # All decode state (token, cache, position, RNG key) stays on the
        # device and is threaded through the jitted functions, so a decode
        # step is exactly one dispatch with no host work in between. Only the
        # cache is donated: the loop in `generate` still reads the previous
        # token after dispatching the next step.
        @jax.jit
        def prefill(params, tokens, n, key):
            cache = qwen3.init_cache(cfg, tokens.shape[0], max_len, dtype)
            # The prompt starts at 0, so it only attends within its own length.
            logits, cache = qwen3.forward(params, tokens, cache, 0, cfg, last=n - 1,
                                          window=tokens.shape[1])
            key, sub = jax.random.split(key)
            return qwen3.sample(logits, sub, **smp), cache, n, key

        # Decode attends over a static `window` of the cache (see
        # `window`), so one step compiles per window size in use.
        def decode(params, tok, cache, pos, key, window):
            key, sub = jax.random.split(key)
            logits, cache = qwen3.forward(params, tok[:, None], cache, pos, cfg,
                                          window=window)
            return qwen3.sample(logits, sub, **smp), cache, pos + 1, key

        step = jax.jit(decode, static_argnums=(5,), donate_argnums=(2,))

        @functools.partial(jax.jit, static_argnums=(5, 6, 7), donate_argnums=(2,))
        def fused(params, tok, cache, pos, key, steps, window, stop=True):
            """`steps` decode steps in one program; stops early at EOS if `stop`.
            `window` must cover every position the steps write."""
            out = jnp.zeros((tok.shape[0], steps), jnp.int32)

            def cond(c):
                i, tok, *_ = c
                done = jnp.all((tok == EOS[0]) | (tok == EOS[1]))
                return (i < steps) & ~(done & stop)

            def body(c):
                i, tok, cache, pos, key, out = c
                tok, cache, pos, key = decode(params, tok, cache, pos, key, window)
                return i + 1, tok, cache, pos, key, out.at[:, i].set(tok)

            n, tok, cache, pos, key, out = jax.lax.while_loop(
                cond, body, (jnp.int32(0), tok, cache, pos, key, out))
            return out, n, (tok, cache, pos, key)

        # Prompts longer than PREFILL_CHUNK go through the cache in chunks:
        # attention materializes [heads, T, window] scores, 1 GB per layer
        # for a single 4096-token chunk.
        @functools.partial(jax.jit, static_argnums=(6,), donate_argnums=(2,))
        def chunk(params, tokens, cache, pos, last, key, window):
            logits, cache = qwen3.forward(params, tokens, cache, pos, cfg, last=last,
                                          window=window)
            key, sub = jax.random.split(key)
            return qwen3.sample(logits, sub, **smp), cache, pos + last + 1, key

        new_cache = jax.jit(lambda: qwen3.init_cache(cfg, batch, max_len, dtype))

        self._prefill, self._step, self._fused = prefill, step, fused
        self._chunk, self._new_cache = chunk, new_cache

    def encode(self, text, think=False):
        return self.tok.encode(chat_prompt(text, think)).ids

    def _prompt(self, ids):
        n = len(ids)
        if n > self.max_len:
            raise ValueError(f"prompt of {n} tokens exceeds max_len {self.max_len}")
        T = min(bucket(n), self.max_len)
        tokens = np.zeros((self.batch, T), np.int32)
        tokens[:, :n] = ids
        return tokens, np.int32(n)

    def prefill(self, ids, key):
        """Decode state (tok, cache, pos, key) after the prompt `ids`."""
        if len(ids) <= PREFILL_CHUNK:
            return self._prefill(self.params, *self._prompt(ids), key)
        if len(ids) > self.max_len:
            raise ValueError(f"prompt of {len(ids)} tokens exceeds max_len {self.max_len}")
        cache, pos = self._new_cache(), 0
        while pos < len(ids):
            part = ids[pos:pos + PREFILL_CHUNK]
            T = min(bucket(len(part)), self.max_len - pos)
            tokens = np.zeros((self.batch, T), np.int32)
            tokens[:, :len(part)] = part
            tok, cache, _, key = self._chunk(self.params, tokens, cache, np.int32(pos),
                                             np.int32(len(part) - 1), key,
                                             self.window(pos + T))
            pos += len(part)
        return tok, cache, jnp.int32(pos), key

    def window(self, n):
        """Cache slots attention reads when the context is n tokens: n
        rounded up to a multiple of an eighth of the next power of two (at
        least MIN_WINDOW), capped at max_len. Reading all of a long cache
        for a short context would cost more than the weights; plain powers
        of two over-read by up to 2x (context 1025 read 2048 slots: 7.5 ms
        per int4 token against mlx-lm's 6.0). Each window compiles once."""
        step = max(MIN_WINDOW, 1 << max((n - 1).bit_length() - 3, 0))
        return min(self.max_len, -(-n // step) * step)

    def step(self, state, pos):
        """One decode step from `state` whose token sits at position `pos`."""
        return self._step(self.params, *state, self.window(pos + 1))

    def fused(self, state, pos, steps, stop=True):
        return self._fused(self.params, *state, steps, self.window(pos + steps), stop)

    def warmup(self, ids, max_new, fused=False):
        """Compile (by running) the prefill for len(ids) and the decode steps
        a reply of up to max_new tokens uses; returns the seconds taken, so
        generate()'s timings exclude compile."""
        t0 = time.perf_counter()
        n = len(ids)
        state = self.prefill(ids, jax.random.key(0))
        if fused:
            jax.block_until_ready(self.fused(state, n, max_new - 1))
        else:
            for w in sorted({self.window(p + 1) for p in range(n, n + max_new)}):
                state = self.prefill(ids, jax.random.key(0))
                jax.block_until_ready(self._step(self.params, *state, w))
        return time.perf_counter() - t0

    def generate(self, ids, max_new=128, *, seed=0, stream=None, fused=False):
        """Token ids of the reply (batch row 0). Returns (ids, timings)."""
        if len(ids) >= self.max_len:
            raise ValueError(f"prompt of {len(ids)} tokens leaves no room to reply "
                             f"within max_len {self.max_len}")
        max_new = min(max_new, self.max_len - len(ids))
        t0 = time.perf_counter()
        state = self.prefill(ids, jax.random.key(seed))
        out = [int(np.asarray(state[0])[0])]
        t_first = time.perf_counter()
        if stream:
            stream(out[0])
        if fused:
            if out[0] not in EOS and max_new > 1:
                toks, n, _ = self.fused(state, len(ids), max_new - 1)
                out += np.asarray(toks)[0, : int(n)].tolist()
        elif out[0] not in EOS:
            # Dispatch step i + 1 before reading token i back to the host, so
            # the GPU never waits for Python (at the cost of one wasted step
            # at EOS).
            n = len(ids)       # the position of token out[i] is n + i
            nxt = self.step(state, n) if max_new > 1 else None
            while nxt is not None:
                state, nxt = nxt, None
                if len(out) + 1 < max_new:
                    nxt = self.step(state, n + len(out))
                out.append(int(np.asarray(state[0])[0]))
                if stream:
                    stream(out[-1])
                if out[-1] in EOS:
                    break
        t_end = time.perf_counter()
        if out[-1] in EOS:
            out.pop()
        return out, {"prompt_tokens": len(ids), "new_tokens": len(out),
                     "ttft_s": t_first - t0, "decode_s": t_end - t_first}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--fused", action="store_true")
    ap.add_argument("--quant", choices=["int8", "int4"], default=None,
                    help="weight-only quantization (default: bfloat16 weights)")
    args = ap.parse_args()

    eng = Engine(args.model, max_len=args.max_len, quant=args.quant, sampling=dict(
        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p))
    ids = eng.encode(args.prompt, args.think)
    if len(ids) >= args.max_len:
        ap.error(f"the prompt is {len(ids)} tokens; --max-len {args.max_len} leaves no room")
    max_new = min(args.max_new, args.max_len - len(ids))
    t_compile = eng.warmup(ids, max_new, args.fused)

    # Stream by re-decoding the reply so far; print only the new suffix.
    shown, reply = "", []

    def stream(t):
        nonlocal shown
        if t in EOS:
            return
        reply.append(t)
        text = eng.tok.decode(reply)
        if not text.endswith("�"):   # wait out partial UTF-8 sequences
            print(text[len(shown):], end="", flush=True)
            shown = text

    out, t = eng.generate(ids, args.max_new, seed=args.seed, fused=args.fused,
                          stream=None if args.fused else stream)
    if args.fused:
        print(eng.tok.decode(out), end="", flush=True)
    n = max(t["new_tokens"] - 1, 1)
    print(f"\n\n[{jax.devices()[0].platform}] prompt {t['prompt_tokens']} tokens, "
          f"compile {t_compile:.1f} s, first token {t['ttft_s'] * 1e3:.0f} ms, "
          f"{t['new_tokens']} new tokens, decode {n / t['decode_s']:.1f} tok/s",
          file=sys.stderr)


if __name__ == "__main__":
    main()
