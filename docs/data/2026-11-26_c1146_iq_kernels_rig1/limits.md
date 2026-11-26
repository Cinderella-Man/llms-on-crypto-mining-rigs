# What limits the routed gate/up kernels (request d0p2)

ms a layer from the group profile; instructions: static SASS per 32 weights at one member; per instruction: ms a layer / instructions, against Q2_0's (1.00 = Q2_0's rate); GB/s: est. bytes a layer at 38.84 distinct experts over the time, against the card's ~173 GB/s measured.

| Type | bits a weight | est. MB a layer | setting | ms a layer | instructions | per instruction, Q2_0 = 1 | est. GB/s | of 173 |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| IQ2_XXS | 2.0625 | 32.81 | old (unset) | 0.938 | 131.44 | 1.17 | 35.0 | 20 % |
| IQ2_XXS | 2.0625 | 32.81 | CUDA C (1) | 0.557 | 76.94 | 1.19 | 58.9 | 34 % |
| IQ2_XXS | 2.0625 | 32.81 | PTX (2) | 0.464 | 72.50 | 1.05 | 70.7 | 41 % |
| IQ2_S | 2.5625 | 40.77 | old (unset) | 0.984 | 136.31 | 1.18 | 41.4 | 24 % |
| IQ2_S | 2.5625 | 40.77 | CUDA C (1) | 0.535 | 70.25 | 1.25 | 76.2 | 44 % |
| IQ2_S | 2.5625 | 40.77 | PTX (2) | 0.434 | 62.94 | 1.13 | 94.0 | 54 % |
| IQ2_XS | 2.3125 | 36.79 | old (unset) | 1.038 | 142.69 | 1.19 | 35.4 | 20 % |
| IQ2_XS | 2.3125 | 36.79 | CUDA C (1) | 0.598 | 81.62 | 1.20 | 61.6 | 36 % |
| IQ2_XS | 2.3125 | 36.79 | PTX (2) | 0.496 | 73.00 | 1.11 | 74.2 | 43 % |
| IQ1_M | 1.75 | 27.84 | old (unset) | 0.549 | 76.69 | 1.17 | 50.7 | 29 % |
| IQ1_M | 1.75 | 27.84 | CUDA C (1) | 0.331 | 52.50 | 1.03 | 84.0 | 49 % |
| IQ1_M | 1.75 | 27.84 | PTX (2) | 0.268 | 45.69 | 0.96 | 103.8 | 60 % |
| Q2_0 | 2.25 | 35.79 | old (unset) | 0.262 | 42.88 | 1.00 | 136.7 | 79 % |

## Over Q2_0's time, with the PTX kernels

| Type | layers | ms a layer over Q2_0's | ms a pass over Q2_0's |
|---|---:|---:|---:|
| IQ2_XXS | 11 | +0.202 | +2.225 |
| IQ2_S | 7 | +0.172 | +1.203 |
| IQ2_XS | 4 | +0.234 | +0.935 |
| IQ1_M | 2 | +0.007 | +0.013 |
| **the four types** | 24 | | **+4.377** |
