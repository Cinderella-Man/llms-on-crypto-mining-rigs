# Registers and spills of the new kernels for compute 6.1 (`cuobjdump -res-usage -arch sm_61`)

128 threads per block; a P106 SM holds 65,536 registers and 96 KB of shared memory, so ~255 registers per thread allow
2 blocks per SM, ~200 allow 2, ~168 allow 3 and 128 allow 4. Ranges cover the three staging widths (16-, 8- and 4-byte
loads).

| Kernel | Build A (`6b880e344`) | Build B (`baf2e8ded`) |
|---|---|---|
| one token, plain or with bias, IQ2_S | 121–126 registers, 20.8 KB shared | the same |
| one token, plain or with bias, IQ3_XXS | 101–111 registers, 14.7 KB | the same |
| one token, fused gate/up, IQ2_S | 253–255 registers, 31.3 KB | 194–228 registers, 20.8 KB |
| one token, fused gate/up, IQ3_XXS | 247–255 registers, 27.2 KB | 201–251 registers, 14.7 KB |
| 2–4 tokens, fused gate/up, IQ2_S | 255 registers + 432–472 bytes of stack (spills) | 255 + 0–8 bytes |
| 2–4 tokens, fused gate/up, IQ3_XXS | 255 registers + 328–400 bytes of stack | 255 + 0 bytes |

Build B stages a fused gate/up's two matrices one after the other in the same buffer (as the Q6_K kernels do) instead
of both at once.
