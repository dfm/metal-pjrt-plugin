"""Check GEMM epilogue fusion (bias / ReLU / GELU, with and without aux
output) on the Metal backend against CPU, and that XLA really fused them into
the `__cublas$lt$matmul` custom call.

  JAX_PLATFORMS=metal,cpu python scripts/epilogue_check.py

Each case is compiled with XLA_FLAGS=--xla_dump_to=<tmp> (set below, before
JAX initializes) and the optimized HLO is searched for the expected
`"epilogue":"..."` in the custom call's backend config.
"""
import glob, os, re, sys, tempfile

DUMP = tempfile.mkdtemp(prefix="epilogue_dump_")
os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") +
                           f" --xla_dump_to={DUMP} --xla_dump_hlo_as_text").strip()

import numpy as np
import jax, jax.numpy as jnp

cpu = jax.devices("cpu")[0]
metal = jax.devices("metal")[0]

def gelu(x):  # the tanh approximation XLA's GemmRewriter matches
    return jax.nn.gelu(x, approximate=True)

CASES = [
    # name, fn(x, w, b), expected epilogue in the dumped HLO
    ("bias", lambda x, w, b: x @ w + b, "BIAS"),
    ("relu", lambda x, w, b: jax.nn.relu(x @ w), "RELU"),
    ("bias+relu", lambda x, w, b: jax.nn.relu(x @ w + b), "BIAS_RELU"),
    ("gelu", lambda x, w, b: gelu(x @ w), "GELU"),
    ("bias+gelu", lambda x, w, b: gelu(x @ w + b), "BIAS_GELU"),
    # The pre-activation value is also returned: GELU_AUX / BIAS_GELU_AUX.
    ("gelu+aux", lambda x, w, b: (gelu(x @ w), (x @ w) * 3.0), "GELU_AUX"),
    ("bias+gelu+aux",
     lambda x, w, b: (gelu(x @ w + b), (x @ w + b) * 3.0), "BIAS_GELU_AUX"),
    # Transposed result (n x m): the bias runs along m.
    ("transposed bias+gelu",
     lambda x, w, b: gelu(jnp.einsum("mk,kn->nm", x, w) + b[: x.shape[0]]),
     "BIAS_GELU"),
    # Batched (3-D) matmul with vector bias.
    ("batched bias+gelu",
     lambda x, w, b: gelu(jnp.einsum("bik,kj->bij", x.reshape(2, -1, x.shape[1]), w) + b),
     "BIAS_GELU"),
]

def epilogues_in_dump(tag):
    found = set()
    for path in glob.glob(os.path.join(DUMP, f"*{tag}*after_optimizations.txt")):
        text = open(path).read()
        found |= set(re.findall(r'"epilogue":"([A-Z_]+)"', text))
    return found

def run(i, name, fn, want, dtype):
    rng = np.random.default_rng(i)
    m, k, n = 64, 48, 96
    x = rng.standard_normal((m, k)).astype(np.float32)
    w = (rng.standard_normal((k, n)) / np.sqrt(k)).astype(np.float32)
    b = rng.standard_normal((n,)).astype(np.float32)
    args = [jnp.asarray(a, dtype) for a in (x, w, b)]
    # Name the jitted module so its dump files are recognizable.
    tag = f"epi{i}_{jnp.dtype(dtype).name}"
    fn_named = lambda *a: fn(*a)
    fn_named.__name__ = tag
    fn_named.__qualname__ = tag
    got = jax.jit(fn_named)(*[jax.device_put(a, metal) for a in args])
    expect = jax.jit(fn_named)(*[jax.device_put(a, cpu) for a in args])
    got, expect = jax.tree.leaves(got), jax.tree.leaves(expect)
    tol = {jnp.float32: 1e-4, jnp.float16: 1e-2, jnp.bfloat16: 5e-2}[dtype]
    for g, e in zip(got, expect):
        np.testing.assert_allclose(np.asarray(g, np.float32),
                                   np.asarray(e, np.float32), rtol=tol, atol=tol)
    fused = epilogues_in_dump(tag)
    if want not in fused:
        raise AssertionError(f"expected epilogue {want} in optimized HLO, found "
                             f"{sorted(fused) or 'none'}")
    return fused

def main():
    print("backend:", jax.default_backend(), "dump:", DUMP)
    failures = 0
    i = 0
    for dtype in (jnp.float32, jnp.float16, jnp.bfloat16):
        for name, fn, want in CASES:
            i += 1
            label = f"{name} [{jnp.dtype(dtype).name}]"
            try:
                fused = run(i, name, fn, want, dtype)
                print(f"PASS  {label:32s} epilogue={','.join(sorted(fused))}")
            except Exception as e:  # noqa: BLE001
                failures += 1
                msg = str(e).strip().splitlines()
                print(f"FAIL  {label:32s} {type(e).__name__}: "
                      f"{' | '.join(msg[:4])}")
    print(f"{i - failures}/{i} passed")
    sys.exit(1 if failures else 0)

if __name__ == "__main__":
    main()
