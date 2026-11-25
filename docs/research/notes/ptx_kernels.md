# Hand-written PTX kernels in Miner Llama (started 2026-11-25)

A working note for the kernels Miner Llama writes in PTX by hand, one grid type at a time (QUEUES code rows 28r1–28r4,
registry [Q1871](../../../quants/experiments.md)). Each session that writes the next kernel reads this note first and
adds what it learned. The first kernel is IQ2_XXS's routed gate/up product at 2–8 tokens, the product that made the
better Flash-Next file (`gmix-q2`) slow ([R125](../../data/2026-11-25_r125_gmix_step/README.md)); its data and checks
are in [docs/data/2026-11-25_c1138_iq2xxs_mtok/](../../data/2026-11-25_c1138_iq2xxs_mtok/README.md).

The method, as the owner set it: write a fast version in any language, check it on the build machine, then rewrite the
whole kernel in PTX (not only the inner loop) and check it again; one kernel type per session.

## How a PTX kernel is built and loaded

- **Source.** `ggml/src/ggml-cuda/ptx/<kernel>.ptx`, written by `ptx/gen_<kernel>.py` and committed. The script holds
  every instruction; it only repeats the unrolled parts (blocks × matrices × groups, the member counts, the entry
  points) and fills in tables (the IQ2_XXS grid as codes, read from `ggml-common.h`). Edit the script, run it with
  `python3 -I`, commit both.
- **Build.** `ggml/src/ggml-cuda/CMakeLists.txt` runs `nvcc -fatbin -arch=compute_61 -code=sm_61,compute_61` on each
  `.ptx`: ptxas makes the compute 6.1 SASS at build time, and the PTX goes into the same fatbin so that a newer card's
  driver compiles it when it loads (the build machine's RTX 3090s run it that way). `ptx/embed.cmake` turns the fatbin
  into a byte array in a generated `.cpp` file of the `ggml-cuda` library. Add a kernel's name to the `foreach` there.
- **Loading.** The kernel's launcher (`mmvq-pascal-mtok-iq2_xxs-ptx.cu` for the first one) loads the fatbin once per
  device with `cuModuleLoadData`, finds its entry points with `cuModuleGetFunction` and launches with `cuLaunchKernel`
  on ggml's stream, so CUDA graphs capture it like any other launch. If the module does not load, the launcher says
  so once and returns false, and the switch falls back to the CUDA C version.
- **Parameters.** The PTX entry takes the CUDA C kernel's two structs by value (`.param .align 8 .b8 P[168]`,
  `F[80]`) and reads fields at fixed byte offsets. The launcher `static_assert`s every offset the PTX reads, so a change
  to a struct stops the build instead of reading the wrong field.
- **No driver library, no PTX.** The driver API needs `libcuda`, which ggml links unless `GGML_CUDA_NO_VMM`; such a
  build defines `GGML_CUDA_NO_PTX_KERNELS` and runs the CUDA C version.
- **Where the SASS is.** The PTX kernels are not in the binary's own fatbin sections (they are data), so
  `cuobjdump -sass <binary>` does not list them; dump the build tree's `ggml/src/ggml-cuda/ptx/<kernel>.fatbin`
  instead (`run/sass_all.sh` of the data folder takes it as `EXTRA_FATBIN`).
- **Bit for bit.** The C kernel is compiled with `-use_fast_math`, so its float operations are the `.ftz` forms, and
  SiLU is `mul.ftz` by −log2 e, `ex2.approx.ftz`, `add.ftz` 1, `div.approx.ftz`. Copy these forms from the C version's
  own PTX (`cuobjdump -xptx all <binary>`, then the kernel's `.entry`), or the PTX kernel's results drift by an ulp.

## IQ2_XXS (C1138): what the fast version does

Today's kernel (`mul_mat_vec_iq_pascal_mtok`, Q713) spent 131.4 SASS instructions per 32 weights of one matrix at one
member, Q2_0's 42.9 (static counts of one pass of the inner loop, `run/sass_loops.py`). Most of it: 43 XMAD (the
32-bit multiplies `(s4 * 0x00204081) & 0x01010101` and `* 0xFF` that turn sign bits into byte masks; Pascal has no
32-bit integer multiply and spends three XMAD on each), 23 logic and 19 shift instructions around them, 4 random
8-byte grid loads from shared memory, and every member's activations read twice (once for up, once for gate). The fast
version, bit for bit the old one:

1. **Gate and up in one pass.** Each member's 32 activations of a block are read from shared memory once for both
   matrices.
2. **Grid value and sign in one PRMT per 4 weights.** A weight's int8 value is byte `code + 4·sign` of the register
   pair {8, 25, 43, 0, −8, −25, −43, 0}, `code` 0–2 its grid value. A shared-memory table holds each grid entry's
   eight codes as nibbles (one word per entry, 8 copies interleaved by word so that lanes 8 apart use other banks).
   The sign bits go to bit 2 of the nibbles with a second PRMT over a 4-byte table: `x = (s7 | s7 << 6) & 0x3333`
   puts the sign bits in pairs into the selector's nibbles, and `prmt(0x44400400, 0, x)` spreads each pair to two
   nibbles. The eighth sign (the parity of the seven) goes in with `popc` and `bfi`. Per group of 8 weights: 12
   integer instructions and one shared load, no multiply.
3. **The weights in our own order.** Nibble n holds weight (0, 1, 4, 5, 2, 3, 6, 7)[n] of its group, so the PRMTs
   come out in that order; the activations are staged in the same order (the order is free: IQ2_XXS has one scale
   per 32 weights, so the 8 DP4A of a block are one integer sum).
4. **The integer sum's float in one FFMA.** The DP4A chain starts at 0x4B400000 (the float 1.5·2^23), so its result
   read as a float is 1.5·2^23 + sum exactly (|sum| < 2^18); `fma(ls, F, −ls·1.5·2^23)` is then `ls·sum` exactly,
   because every value stays below 2^24: the old kernel's `(float)(ls * sum)` without the integer multiply.
5. **Row segments straight into registers.** Each lane loads the five aligned 16-byte units around its super-block
   (no staging through shared memory). A warp takes rows 4 apart, so with 4-byte-aligned rows and 16-byte-aligned
   matrices its lanes' super-blocks start at the same word of their units (`kappa`); a warp-uniform branch picks the
   words, and one PRMT per word moves lane 0's super-block (which starts 2 bytes earlier in its word) into place.

Result (static, per 32 weights of one matrix, one pass): 76.9 instructions in CUDA C, 72.5 in PTX at one member
(old 131.4, Q2_0 42.9); 112.9 and 108.1 at four members (old 183.2, Q2_0 79.6). Each further member costs 12.0
(old 17.3, Q2_0 12.2). Shared-memory instructions per 32 weights at one member: 5.1 (4 random 4-byte loads, 8-copy
table) against the old 10.5 (4 random 8-byte loads, staging stores and loads, activations twice).

## What ptxas did with the PTX

- It kept the instructions as written: `bfe` → `BFE.U32`, `lop3` → `LOP3.LUT`, `bfi` → `BFI`, `popc` → `POPC`,
  `prmt` → `PRMT`, `dp4a` → `IDP.4A.S8.S8`, the `.ftz` float forms one to one.
- It fused `shl` + `add` of the table address into one `ISCADD`, and put the constant table registers' values in
  constant bank 2 or immediates as it liked.
- A multiply by 272 (a super-block's staged bytes) became two XMAD; a shift-and-add would be one instruction less
  per iteration (not per block: negligible).
- Registers: 165, 168 and 203 for groups of up to 2, 4 and 8 members, no spills; `.minnctapersm 3` (2 for 8 members)
  holds 3 blocks per SM, as the C kernel's `__launch_bounds__`.
- Against nvcc's SASS of the same algorithm in C, the PTX saves 4.4 instructions per 32 weights: nvcc built the table
  index with `BFE` + `BFI` + shift where the PTX's `bfe` + `shl` + `add` became `BFE` + `ISCADD`, and it issued the
  second matrix's five loads only after picking the first matrix's words (the PTX issues all ten loads first).

## What worked, what to watch

- **Count one path.** A loop with a switch inside (the `kappa` choice) holds all four cases in its static
  instructions; `run/sass_loops.py` follows one path (unconditional branches taken, conditional ones not), which is
  what a warp executes.
- **CUDA's `__byte_perm` masks the selector to 3 bits a nibble** (it hides PRMT's sign-replicate mode), so with a
  runtime selector the compiler adds an AND. Inline `prmt.b32` (or PTX) avoids it when the selector's bit 3 is clear.
- **A test fault proves the path ran.** `GGML_CUDA_MTOK_IQ_FAST_FAULT=1` swaps up and gate for the fast kernels; the
  old-against-new check must then differ in exactly the fused cases of 2–8 tokens, which shows the fast kernel took
  them (byte-identical results alone could mean the switch did nothing).
- **Shapes.** The fast kernel needs rows of 2 lanes (256·n weights with n = 2 mod 4: Flash-Next's 2,560), matrices
  16-byte aligned and ending on a 16-byte boundary. The PTX covers only the fused gate/up launch with SWIGLU at 2,560;
  everything else falls back (plain MUL_MAT_ID and the shared expert to the C fast kernel, other shapes to the old
  kernel).

## For the next kernels (IQ2_S, IQ2_XS, IQ1_M)

- **IQ2_XS** (74-byte super-blocks: 32 entries of a 9-bit grid index and a 7-bit sign pattern, two 4-bit scales per
  32 weights). The same PRMT decode; the sign path is the same (7 bits plus parity) but reads `bfe(entry, 9, 7)`.
  The grid has 512 entries: the code table is 512 words, so 8 copies take 16 KB of shared memory (4 copies, 8 KB, with
  more bank conflicts, is the other choice). Two scales a block mean two DP4A chains (weights 0–15 and 16–31) and
  `(float)(ls0·i0 + ls1·i1)`: each product is exact in float and so is their sum (below 2^24), so
  `fma(ls0, float(i0), ls1 · float(i1))` with each `float(i)` from the 0x4B400000 trick plus one `FADD`. Keep the
  activation order within each 16-weight half (the halves have different scales). Its rows: 74 × 10 = 740 bytes,
  which is 4 mod 16, the same `kappa` scheme; super-blocks start at 2 mod 4 bytes on every other lane, as IQ2_XXS.
- **IQ2_S** (82-byte super-blocks: an 8-bit index per 8 weights plus 2 high bits from `qh`, a full sign byte per 8
  weights, two 4-bit scales per 32). Full sign bytes need no parity: the sign path is `bfe`, `shl`, `lop3`, `prmt`
  (4 instructions, not 6). The grid has 1,024 entries: 4 KB a copy, so at most 2–4 copies fit; weigh bank conflicts
  against shared memory. Rows: 820 bytes, 4 mod 16 again.
- **IQ1_M** (56-byte super-blocks, 11-bit indices into a 2,048-entry grid of {−1, 0, 1}, a delta sign per group): the
  old kernel already decodes with PRMT over a nibble grid (`iq1s_grid_gpu`); look first at its index assembly and the
  8 KB grid's bank conflicts. Rows: 560 bytes, 0 mod 16, so every lane's super-block starts at a fixed alignment (no
  `kappa` branch).
- In every case: start from the C fast kernel's structure (`mmvq-pascal-mtok-iq2_xxs-fast.cu`), check old against
  new with `bench/mmvq_pascal/verify/mtok_iq_fast_check.cpp` (its type argument), add the type's entry points to the
  generator or a new `gen_*.py`, and count with `run/sass_loops.py`.
- **Not measured yet:** whether the P106 is limited by instruction issue or by the shared-memory pipe in these
  kernels. The sign path costs 6 integer instructions a group; a 128-entry sign table in shared memory (4 KB with 8
  copies) would cost 2 and one more random shared load. Rig 1's timing of the IQ2_XXS kernel (QUEUES) is the
  first number to decide it.
