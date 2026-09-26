| case | metal | jax-mps | mlx | cpu |
|---|---|---|---|---|
| elementwise chain 16M | 1.88 | 1.85 | 1.77 | 34.12 |
| reduce rows 4096x4096 | 0.98 | 2.69 | 2.64 | 0.77 |
| reduce cols 4096x4096 | 1.07 | 2.77 | 2.69 | 1.74 |
| reduce all 16M | 0.98 | 1.04 | 1.09 | 0.88 |
| softmax 8192x1024 | 1.09 | 1.02 | 1.01 | 1.60 |
| layernorm fwd 8192x1024 | 1.83 | 3.31 | 3.40 | 1.03 |
| layernorm fwd+bwd 8192x1024 | 2.58 | 11.00 | 10.99 | 1.97 |
| transpose 4096x4096 | 1.91 | 1.93 | 1.78 | 35.41 |
| cumsum 4096x4096 rows | 1.89 | 2.08 | 1.80 | 17.77 |
| adam 50x1M params | ERR | 13.44 | 17.83 | 37.53 |
| matmul f32 1024 | ERR | 1.04 | 1.14 | 4.42 |
| matmul f32 2048 | ERR | 6.63 | 6.80 | 41.76 |
| matmul f32 4096 | ERR | 70.20 | 73.24 | 401.45 |
| matmul bf16 2048 | ERR | 6.15 | 6.50 | 86.76 |
| batched matmul 16x512 | ERR | 1.82 | 1.91 | 10.97 |
| attention fwd 8x8x512x64 | ERR | 5.75 | 5.92 | 14.50 |
| attention fwd+bwd | ERR | 19.24 | 15.32 | 43.54 |
| nanoGPT fwd (loss) | ERR | 72.08 | 67.80 | 327.37 |
| nanoGPT train step | ERR | 262.60 | 238.07 | 1958.69 |
| cnn fwd 32x32x32 | ERR | 1.18 | 0.61 | 1.12 |
| cnn fwd+bwd | ERR | 2.27 | 1.85 | 4.94 |
