import numpy as np, jax, jax.numpy as jnp
cpu = jax.devices("cpu")[0]; gpu = jax.devices("metal")[0]
rng = np.random.default_rng(0)
bad = 0
def check(name, fn, *args, tol):
    global bad
    ref = np.asarray(jax.jit(fn, device=cpu)(*[jax.device_put(a, cpu) for a in args]), np.float32)
    got = np.asarray(jax.jit(fn, device=gpu)(*[jax.device_put(a, gpu) for a in args]), np.float32)
    err = np.max(np.abs(got - ref) / (np.abs(ref) + 1.0))
    ok = err < tol
    bad += not ok
    print(("ok  " if ok else "BAD ") + f"{name}: rel err {err:.2e}")
for dt, tol in ((jnp.bfloat16, 5e-2), (jnp.float16, 1e-2)):
    for (m, k, n) in ((64, 64, 64), (100, 37, 50), (1, 300, 7), (257, 129, 65), (512, 1024, 256)):
        a = jnp.asarray(rng.standard_normal((m, k)), dt); b = jnp.asarray(rng.standard_normal((k, n)), dt)
        check(f"{dt.__name__} {m}x{k}x{n}", lambda x, y: x @ y, a, b, tol=tol)
        check(f"{dt.__name__} {m}x{k}x{n} A.T B.T", lambda x, y: y.T @ x.T, a, b, tol=tol)
        check(f"{dt.__name__} {m}x{k}x{n} f32 out", lambda x, y: jnp.matmul(x, y, preferred_element_type=jnp.float32), a, b, tol=tol / 4)
        bias = jnp.asarray(rng.standard_normal((n,)), dt)
        check(f"{dt.__name__} {m}x{k}x{n} +bias relu", lambda x, y, c: jax.nn.relu(x @ y + c), a, b, bias, tol=tol)
    x = jnp.asarray(rng.standard_normal((6, 33, 70)), dt); y = jnp.asarray(rng.standard_normal((6, 70, 45)), dt)
    check(f"{dt.__name__} batched", lambda x, y: jnp.einsum("bij,bjk->bik", x, y), x, y, tol=tol)
    check(f"{dt.__name__} batched bcast", lambda x, y: jnp.einsum("bij,jk->bik", x, y[0]), x, y, tol=tol)
    check(f"{dt.__name__} grad", lambda x, y: jax.grad(lambda u, v: jnp.sum((u @ v).astype(jnp.float32) ** 2))(x[0], y[0]), x, y, tol=tol * 2)
print("FAILURES:", bad)
