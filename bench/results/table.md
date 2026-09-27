> Note: these results predate the rename of the JAX platform to "openmetal"
> (36cc906); `JAX_PLATFORMS: "metal"` below is what that run used. The
> table is not regenerated for the rename.

- mlx: no commit/version metadata (a run from before 2026-09-27)
- cpu commit: 64488ff-dirty
- cpu jax: 0.11.2
- cpu knobs: {"JAX_PLATFORMS": "cpu"}
- cpu platform_version: cpu
- metal commit: 64488ff-dirty
- metal jax: 0.11.2
- metal knobs: {"JAX_PLATFORMS": "metal"}
- metal platform_version: oneapi 1.3970105006.2990104082
- metal-gpu commit: 64488ff-dirty
- metal-gpu jax: 0.11.2
- metal-gpu knobs: {"JAX_PLATFORMS": "metal", "METAL_PJRT_TRACE": "1"}
- metal-gpu platform_version: oneapi 1.3970105006.2990104082
- medians of 3 round(s); wall ms unless noted

| case | metal | cpu | mlx | metal GPU ms | metal TFLOPS |
|---|---|---|---|---|---|
| elementwise chain 16M | 1.88 | 30.64 | 3.15 | 1.53 | - |
| reduce rows 4096x4096 | 1.18 | 0.64 | 2.57 | 0.80 | - |
| reduce cols 4096x4096 | 1.21 | 1.50 | 2.59 | 0.96 | - |
| reduce all 16M | 1.01 | 0.65 | 0.98 | 0.96 | - |
| softmax 8192x1024 | 1.91 | 1.33 | 0.97 | 1.53 | - |
| layernorm fwd 8192x1024 | 1.92 | 0.82 | 3.42 | 1.56 | - |
| layernorm fwd+bwd 8192x1024 | 2.62 | 1.51 | 10.62 | 2.15 | - |
| transpose 4096x4096 | 1.97 | 31.04 | 1.74 | 1.61 | - |
| cumsum 4096x4096 rows | 1.79 | 11.63 | 1.76 | 1.66 | - |
| adam 50x1M params | 15.71 | 27.82 | 15.97 | 15.29 | - |
| matmul f32 1024 | 1.29 | 3.86 | 1.06 | 1.12 | 1.78 |
| matmul f32 2048 | 6.32 | 35.26 | 6.40 | 5.97 | 2.78 |
| matmul f32 4096 | 48.65 | 315.11 | 52.24 | 46.60 | 2.91 |
| matmul bf16 2048 | 6.08 | 89.39 | 5.86 | 5.63 | 2.92 |
| batched matmul 16x512 | 2.04 | 10.80 | 1.82 | 2.37 | 2.41 |
| attention fwd 8x8x512x64 | 5.37 | 13.79 | 5.19 | 4.95 | 0.81 |
| attention fwd+bwd | 11.94 | 43.85 | 15.39 | 11.50 | 1.08 |
| cholesky 128 | 0.86 | 0.03 | - | 0.03 | 0.00 |
| solve 128x16 | 0.84 | 0.05 | - | 0.03 | 0.00 |
| qr 128 | 0.76 | 0.15 | - | 0.04 | 0.01 |
| eigh 128 | 0.91 | 0.55 | - | 0.03 | - |
| cholesky 512 | 0.99 | 0.23 | - | 0.09 | 0.05 |
| solve 512x16 | 1.24 | 0.56 | - | 0.09 | 0.08 |
| qr 512 | 4.66 | 3.66 | - | 0.19 | 0.08 |
| eigh 512 | 11.94 | 11.63 | - | 0.08 | - |
| cholesky 2048 | 6.76 | 8.59 | - | 1.05 | 0.43 |
| solve 2048x16 | 13.01 | 13.43 | - | 0.84 | 0.45 |
| qr 2048 | 111.40 | 111.94 | - | 1.82 | 0.21 |
| nanoGPT fwd (loss) | 59.73 | 408.18 | 68.50 | 57.92 | - |
| nanoGPT train step | 187.35 | 1257.89 | 231.84 | 178.60 | - |
| cnn fwd 32x32x32 | 2.66 | 0.92 | 0.69 | 2.52 | - |
| cnn fwd+bwd | 6.15 | 3.79 | 2.25 | 5.86 | - |
