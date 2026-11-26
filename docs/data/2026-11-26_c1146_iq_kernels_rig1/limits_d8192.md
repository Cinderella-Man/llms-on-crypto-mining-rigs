# What limits the routed gate/up kernels (request d8192p2)

ms a layer from the group profile; instructions: static SASS per 32 weights at one member; per instruction: ms a layer / instructions, against Q2_0's (1.00 = Q2_0's rate); GB/s: est. bytes a layer at 38.84 distinct experts over the time, against the card's ~173 GB/s measured.

| Type | bits a weight | est. MB a layer | setting | ms a layer | instructions | per instruction, Q2_0 = 1 | est. GB/s | of 173 |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| IQ2_XXS | 2.0625 | 32.81 | old (unset) | 0.936 | 131.44 | 1.14 | 35.1 | 20 % |
| IQ2_XXS | 2.0625 | 32.81 | CUDA C (1) | 0.558 | 76.94 | 1.16 | 58.8 | 34 % |
| IQ2_XXS | 2.0625 | 32.81 | PTX (2) | 0.463 | 72.50 | 1.02 | 70.9 | 41 % |
| IQ2_S | 2.5625 | 40.77 | old (unset) | 0.967 | 136.31 | 1.14 | 42.2 | 24 % |
| IQ2_S | 2.5625 | 40.77 | CUDA C (1) | 0.526 | 70.25 | 1.20 | 77.4 | 45 % |
| IQ2_S | 2.5625 | 40.77 | PTX (2) | 0.428 | 62.94 | 1.09 | 95.2 | 55 % |
| IQ2_XS | 2.3125 | 36.79 | old (unset) | 1.050 | 142.69 | 1.18 | 35.0 | 20 % |
| IQ2_XS | 2.3125 | 36.79 | CUDA C (1) | 0.603 | 81.62 | 1.19 | 61.0 | 35 % |
| IQ2_XS | 2.3125 | 36.79 | PTX (2) | 0.500 | 73.00 | 1.10 | 73.6 | 43 % |
| IQ1_M | 1.75 | 27.84 | old (unset) | 0.552 | 76.69 | 1.15 | 50.5 | 29 % |
| IQ1_M | 1.75 | 27.84 | CUDA C (1) | 0.331 | 52.50 | 1.01 | 84.1 | 49 % |
| IQ1_M | 1.75 | 27.84 | PTX (2) | 0.271 | 45.69 | 0.95 | 102.6 | 59 % |
| Q2_0 | 2.25 | 35.79 | old (unset) | 0.267 | 42.88 | 1.00 | 133.9 | 77 % |

## Over Q2_0's time, with the PTX kernels

| Type | layers | ms a layer over Q2_0's | ms a pass over Q2_0's |
|---|---:|---:|---:|
| IQ2_XXS | 11 | +0.196 | +2.153 |
| IQ2_S | 7 | +0.161 | +1.126 |
| IQ2_XS | 4 | +0.233 | +0.931 |
| IQ1_M | 2 | +0.004 | +0.008 |
| **the four types** | 24 | | **+4.220** |
