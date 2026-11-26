# Hand-written PTX kernels in Miner Llama (started 2026-11-25)

A working note for the kernels Miner Llama writes in PTX by hand, one grid type at a time (QUEUES code rows 28r1–28r4,
registry [Q1871](../../../quants/experiments.md)). Each session that writes the next kernel reads this note first and
adds what it learned. The first kernel is IQ2_XXS's routed gate/up product at 2–8 tokens, the product that made the
better Flash-Next file (`gmix-q2`) slow ([R125](../../data/2026-11-25_r125_gmix_step/README.md)); its data and checks
are in [docs/data/2026-11-25_c1138_iq2xxs_mtok/](../../data/2026-11-25_c1138_iq2xxs_mtok/README.md). The second is
IQ2_S's ([docs/data/2026-11-26_c1138_iq2s_mtok/](../../data/2026-11-26_c1138_iq2s_mtok/README.md)), the third IQ2_XS's
([docs/data/2026-11-26_c1138_iq2xs_mtok/](../../data/2026-11-26_c1138_iq2xs_mtok/README.md)), the fourth and last
IQ1_M's ([docs/data/2026-11-26_c1138_iq1m_mtok/](../../data/2026-11-26_c1138_iq1m_mtok/README.md)), which also holds the
check of all four together on the real file. The [summary](#the-four-kernels-summary) at the end compares the four.

The method, as the owner set it: write a fast version in any language, check it on the build machine, then rewrite the
whole kernel in PTX (not only the inner loop) and check it again; one kernel type per session.

## How a PTX kernel is built and loaded

- **Source.** `ggml/src/ggml-cuda/ptx/<kernel>.ptx`, written by `ptx/gen_<kernel>.py` and committed. The script holds
  every instruction; it only repeats the unrolled parts (blocks × matrices × groups, the member counts, the entry
  points) and fills in tables (the grid as codes, read from `ggml-common.h`). Edit the script, run it with
  `python3 -I`, commit both.
- **One generator frame for the multi-token kernels.** `gen_mtok_iq2_xxs.py` writes the parts every type shares (the
  parameters, the block's group, the activations' staging, the skipped slots, the SiLU epilogue) in `entry(ntm, kt)`,
  and a body's frame in three functions every type's body calls: `row_start(k, nt, unit)` (the row-block loop, the
  lane pair's rows, `kappa`, the accumulators, the iteration's activation pointers), `segments(k, nt, sb, unit,
  nunits, unit_off, nwords)` (both matrices' row segments in 16- or 8-byte units, the warp-uniform choice of words, 4
  or 2 cases) and `row_end(k, nt)` (each super-block's `d`, the iteration loop, the lanes' sum, SiLU, the stores). A
  type brings a class with what differs (`prefix`, `copies`, `tab_words`, `unit_off`, and the functions `table`,
  `consts`, `body`). `gen_mtok_iq2_s.py` imports it (`sys.path` holds its folder, since `python3 -I` leaves the
  script's folder out) and also offers the parts of two-scale types (`scale_bytes`, `scale_floats`, `sums`, `values`),
  which `gen_mtok_iq2_xs.py` imports in turn. `gen_mtok_iq1_m.py` uses the frame's options for a type unlike the IQ2
  ones: the class attributes `stage_sel` and `stage_doc` (the activations' order within each group of 8), `lut` (the
  value registers) and `word_pick = False` (no choice of words), `row_start(..., kappa=False)`, `segments(...,
  direct=True)` (each lane's super-block loaded straight into `w<m>_k`) and `row_end(..., dscale=f)` (the type's own
  `d`). A new type: a new `gen_mtok_<type>.py` with its class, calling these functions; check that the older types'
  PTX stays the same bytes (the generators' outputs against the committed `.ptx` files).
- **Build.** `ggml/src/ggml-cuda/CMakeLists.txt` runs `nvcc -fatbin -arch=compute_61 -code=sm_61,compute_61` on each
  `.ptx`: ptxas makes the compute 6.1 SASS at build time, and the PTX goes into the same fatbin so that a newer card's
  driver compiles it when it loads (the build machine's RTX 3090s run it that way). `ptx/embed.cmake` turns the fatbin
  into a byte array in a generated `.cpp` file of the `ggml-cuda` library. Add a kernel's name to the `foreach` there.
- **Loading.** One launcher for every type, `mmvq-pascal-mtok-iq-ptx.cu`: a table of each type's fatbin and entry-point
  names (add a row for a new type); it loads a type's fatbin once per device with `cuModuleLoadData`, finds its entry
  points with `cuModuleGetFunction` and launches with `cuLaunchKernel` on ggml's stream, so CUDA graphs capture it like
  any other launch. `ggml_cuda_mtok_iq_ptx_launch(type, ...)` returns false where no PTX takes the product; if a module
  does not load, the launcher says so once and returns false, and the switch falls back to the CUDA C version.
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
- **Which tests reach the kernel.** `test-backend-ops -o MUL_MAT_ID` matches that op name exactly: the fused gate/up
  cases are the ops `MUL_MAT_VEC_FUSION` and `MUL_MAT_ID_FUSION`, the routed + shared launch `MUL_MAT_ID_SHARED`
  (pass `-o 'MUL_MAT_VEC_FUSION|MUL_MAT_ID_FUSION'`, a regex once it holds a `|`). For IQ2_XXS at 2,560 inputs the
  fused cases are the PTX kernel's; the plain op only ever runs the C version.
- **Proof that the PTX ran.** Byte-identical results do not show it (the C fallback gives the same bytes). The PTX
  kernels are launched with `cuLaunchKernel`, so `run/launch_count.c` (an `LD_PRELOAD` wrapper of that one symbol)
  counts them by name at exit. A server must then exit through `exit()`: send SIGINT to `llama-server` itself, since
  the container's `bash` does not pass `docker stop`'s SIGTERM on.
- **Run the processor-heavy checks one or two at a time.** `test-backend-ops` spends most of its time in the
  processor's reference products, whose threads spin while they wait: six arms side by side on a 24-thread machine
  (load 78) ran slower in total than two at a time, and slowed a model on the same machine to under one prompt token a
  second.
- **Comparing every kernel of two builds.** Kernel names repeat (one copy per source file that instantiates them), and
  a kernel that ends a cubin picks up the next cubin's header lines in `cuobjdump`'s text: compare each name's copies as
  a set of instruction hashes with those headers and the blank lines between cubins dropped (IQ2_S's `run/sass_all.sh`;
  IQ2_XXS's version kept the blank lines, so two unchanged kernels read as changed once IQ2_S's new source files moved
  which kernel ends a cubin). Its `BASE_EXTRA` and `HEAD_EXTRA` add the PTX fatbins of both builds.
- **Shapes.** The fast kernels (all four types) need rows of 2 lanes (256·n weights with n = 2 mod 4:
  Flash-Next's 2,560), matrices 16-byte aligned and ending on a 16-byte boundary. The PTX covers only the fused gate/up launch with SWIGLU at 2,560;
  everything else falls back (plain MUL_MAT_ID and the shared expert to the C fast kernel, other shapes to the old
  kernel).

## IQ2_S (C1138, code row 28r2): what changed from IQ2_XXS

Today's kernel spent 136.3 SASS instructions per 32 weights of one matrix at one member (IQ2_XXS's 131.4), 19.3 more
per further member: 46 XMAD (the sign masks as for IQ2_XXS, and per member the two scale products `ls0 * i0 + ls1 * i1`,
6 XMAD a member), 43 logic and shift instructions, 4 random 8-byte lookups in a 1,024-entry grid, 820-byte rows staged
through shared memory with 4-byte loads, activations read twice. The fast version keeps IQ2_XXS's frame (gate and up in
one pass, the PRMT decode from a nibble code table, the integer sums' floats without conversions, row segments straight
into registers) and changes what IQ2_S's format needs:

1. **The 10-bit index in one PRMT.** A group's 8 low index bits are byte l of the block's `qs` word, its 2 high bits
   bits 2l and 2l + 1 of the block's byte of the `qh` word. With copies of the `qh` word shifted right by 2, 4 and 6
   (3 shifts per 4 blocks), `prmt(qs, qh >> 2l, l | (4 + b) << 4)` puts both side by side and the index is its low 10
   bits. In CUDA C an AND and the table address (shift and add) follow; in PTX one `bfi` of those 10 bits into the
   lane's copy offset gives the word address, and ptxas folds the table's base into the load: 2 instructions a group
   in PTX, 3 in C (IQ2_XXS: `bfe` + `ISCADD`).
2. **The code table: 1,024 entries in 4 copies** (16 KB, one 16-byte store per entry at the block's start). 8 copies
   (32 KB) would not fit beside 8 members' activations (24 KB) under the 48 KB a block may have; with 4 copies the
   blocks per SM stay the old kernel's (3 for groups of up to 4 members, 2 for 8). A copy is chosen by `lane & 3`, so
   8 lanes share each copy's 8 banks per entry slot (IQ2_XXS: 4 lanes, 4 banks).
3. **8 explicit sign bits per 8 weights.** `z = sg & 0x33333333` and `z2 = (sg >> 2) & 0x33333333` hold the sign
   pairs (0, 1), (4, 5) and (2, 3), (6, 7) of all four groups of a block in their nibbles; `prmt(z, z2, l | (4 + l) << 4)`
   picks group l's two bytes as the selector of the PRMT that spreads the signs to bit 2 of the code nibbles. 2.75
   instructions per 8 weights (IQ2_XXS's 7 bits and parity: 6). The nibble order within a group is IQ2_XXS's
   (0, 1, 4, 5, 2, 3, 6, 7), so the activations are staged as for IQ2_XXS.
4. **Two scales per 32 weights.** Two DP4A chains (weights 0–15 and 16–31), each started at 0x4B400000, so each result
   read as a float is 1.5·2^23 + its sum. Then `p1 = fma(ls1, F1, -ls1·1.5·2^23)` is `ls1·i1` and
   `fma(ls0, F0 - 1.5·2^23, p1)` is `ls0·i0 + ls1·i1`, each exact (the sums below 2^18, the products and their sum below
   2^23): the old kernel's `(float) (ls0 * i0 + ls1 * i1)`, 4 float operations a member and matrix (IQ2_XXS: 2), no
   multiply, no conversion. The free weight order (IQ2_XXS's step 3) holds within each 16-weight half; keeping it within
   each group of 8 keeps the halves apart without thought.
5. **The scales' floats.** The 8 blocks' scales are 2 words (a byte a block: the low nibble for weights 0–15, the high
   for 16–31); `((w << 1) & 0x1E1E1E1E) | 0x01010101` and `((w >> 3) & 0x1E1E1E1E) | 0x01010101` give every block's
   `2 s + 1` as bytes (2 instructions a word). In CUDA C a scale's float is a PRMT of its byte under the exponent of
   2^23 and an FADD of −2^23; in PTX `bfe` + `cvt.rn.f32.u32`, which ptxas turns into one I2F.F32.U8 with a byte select.
6. **Rows of 820 bytes** (4 mod 16, as IQ2_XXS's 660): each lane loads the 6 aligned 16-byte units that hold its
   82-byte super-block, lane 1 from 80 bytes after lane 0's first unit (82 = 80 + 2, so both lanes' super-blocks start
   at the same word of their units, lane 1 two bytes later, and IQ2_XXS's warp-uniform `kappa` branch holds).

Result (static, per 32 weights of one matrix, one pass): 70.25 instructions in CUDA C, 62.94 in PTX at one member
(old 136.31, IQ2_XXS's PTX 72.50); 111.56 and 104.38 at four members (old 194.12). Each further member costs 13.8
(old 19.3): 8 DP4A, 4 float operations, and its share of 2 16-byte activation loads. The PTX saves 7.3 over nvcc's C:
4.1 from the `bfi` address, 2.1 from the I2F scales, 1.3 from the index arithmetic and the loop's frame (fewer moves,
branches and XMAD than nvcc's).

## What ptxas did with IQ2_S's PTX

- `bfi.b32 ea, px, coff, 4, 10` then `add.u32 ea, ea, tab` became one `BFI` and `LDS.U.32 R, [R]`: the module's only
  static shared array sits at address 0 on sm_61, so the add vanished (elsewhere it would be the load's immediate);
  the address needs no alignment of the table either way.
- `bfe.u32 t, e, 8b, 8` then `cvt.rn.f32.u32 f, t` became one `I2F.F32.U8 f, e.Bb`. I2F runs at a quarter of the FP32
  rate on Pascal; 2 a block and matrix stay well under that pipe's limit by count, but the PRMT-and-FADD path is kept in
  the generator (`I2F_SCALES = False`) for a timing.
- It merged each super-block's 8 activation-scale loads into 2 `LDS.128` (nvcc did the same for the C version).
- Registers: 163, 168 and 244 for groups of up to 2, 4 and 8 members, no spills. nvcc's C needed 247 for 8 members and
  its one-member loop there grew to 77.69 instructions; the PTX's loops are the same in every entry point.

## IQ2_XS (C1138, code row 28r3): what changed from IQ2_S

Today's kernel spent 142.69 SASS instructions per 32 weights of one matrix at one member (IQ2_S's 136.31), 19.3 more
per further member: 46 XMAD (the sign masks, and 6 a member for the two scale products), 49 logic and shift
instructions, 4 random 8-byte lookups in a 512-entry grid, 740-byte rows staged through shared memory with 4-byte
loads, activations read twice. The fast version is IQ2_S's with IQ2_XS's format:

1. **Rows in 8-byte units.** A super-block is 74 bytes, so lane 1's starts 2 words further into a 16-byte unit than
   lane 0's (74 = 64 + 8 + 2), and with 16-byte loads the two lanes would need different words. Each lane loads the 10
   aligned 8-byte units from `(segment & ~7) + 72 sl`: both lanes' super-blocks then start at word `kappa` of their
   units (`kappa` = bit 2 of the segment's address, one word further each iteration of 148 bytes), lane 1's two bytes
   later, and the warp-uniform branch has 2 cases. 10 `LDG.64` a lane and matrix (1.25 per 32 weights; IQ2_S's 6
   `LDG.128`, 0.75), no shared-memory staging. Every 8-byte unit holds a byte of the matrix, so no load leaves it.
2. **The index in one BFI.** Each 8 weights have a 16-bit entry: a 9-bit grid index, 7 sign bits on top. For an entry
   in a word's low half, `bfi(word, coff, 5, 9)` puts the index at bits 5–13 of the table's byte address beside the
   lane's copy (`coff` = 4 · (lane & 7)); an entry in the high half is shifted down first (`shr` 16). 1.5 instructions
   a group (nvcc's C: `BFE`, `BFI` of the index beside the copy, `SHL` to bytes: 3). 512 entries in 8 copies: 16 KB,
   as IQ2_S's 1,024 in 4; 4 lanes per copy, as IQ2_XXS's.
3. **Signs as IQ2_XXS's:** `bfe(word, 9 or 25, 7)`, the pairs into the selector's nibbles (`shl`, `lop3`), the parity
   (`popc`, `bfi`), one PRMT: 6 a group.
4. **Scales and sums as IQ2_S's:** a scale byte per 32 weights (low nibble weights 0–15) at the same place in the
   last two words, so the bytes `2 s + 1`, their I2F floats, the two DP4A chains and the exact FFMAs are IQ2_S's
   generator code, called, not copied.

Result (static, per 32 weights of one matrix, one pass): 81.62 instructions in CUDA C, 73.00 in PTX at one member
(old 142.69, IQ2_S's PTX 62.94, IQ2_XXS's 72.50); 123.56 and 114.62 at four members (old 200.56). Each further member
costs 13.9 (old 19.3). The PTX saves 8.6 over nvcc's C: 6.0 from the table address, 2.0 from the I2F scales (C: a
PRMT and an FADD each), 0.6 from the loop's frame. Against IQ2_S's PTX it spends 10.1 more, nearly all in the signs
(6 instructions a group against IQ2_S's 2.75: 13 per 32 weights), less the index (IQ2_S's PRMT and BFI against one
BFI, or two for a high-half entry: 2 fewer).

## What ptxas did with IQ2_XS's PTX

- Kept the instructions one to one again: `bfi` into the copy offset and the `add` of the table base became one `BFI`
  and `LDS.U.32 R, [R]` (the table at shared address 0, as IQ2_S's); `ld.global.nc.v2.u32` became `LDG.E.CI.64`.
- Registers: 165, 168 and 215 for groups of up to 2, 4 and 8 members, no spills; the 8-member entry point's loops
  are the 4-member one's within 0.06 instructions per 32 weights (73.06 at one member).
- nvcc's C for the same algorithm built each table address with `BFE` of the 9 bits, `BFI` of them beside the lane's
  copy (a word index) and `SHL` by 2, where the PTX inserts the bits straight into the byte offset.


## IQ1_M (C1138, code row 28r4): what changed from IQ2_S

Today's kernel spent 76.69 SASS instructions per 32 weights of one matrix at one member (the IQ2 types' 131–143),
19.3 more per further member. IQ1_M has no signs, so its XMAD are only the per-member scale products (6 a member);
the rest is 17.5 logic and 17.8 shift instructions for the 11-bit index, the delta's sign and the 3-bit scales, 8 PRMT,
4 random grid lookups in a 2,048-entry table, the 560-byte rows staged through shared memory, activations read twice.
The fast version keeps the other three's frame and changes what IQ1_M's format needs:

1. **The delta's sign in the table.** The grid `iq1s_grid_gpu` holds grid + 1 (0, 1, 2) as nibbles, so bit 2 of every
   nibble is free. The code table has 4,096 words: the 2,048 entries, then the same with bit 2 of every nibble set
   (16 KB, one copy). A group's `qs` byte and its `qh` nibble side by side (one PRMT; odd groups take a copy of the
   `qh` word shifted right by 4, one shift per two blocks) make a 12-bit index, the 11-bit grid index with the delta's
   sign at bit 11, and one `bfi(px, 0, 2, 12)` puts it into the table's byte address (ptxas folds the table's base, at
   shared address 0, into the load). The word read is already the value PRMTs' selector with the sign in it:
   `prmt(t_pos, t_neg, c)`, `shr`, `prmt` over the register pair {−7, 1, 9, 0, −9, −1, 7, 0}, 8 (grid ± 1/8) by code.
   5 integer instructions and one shared load per 8 weights (IQ2_S 7.75, IQ2_XS 12.5). The plan written here before
   (2 copies of the 2,048 entries, the sign added by `bfe.s32` and `lop3`) would have spent 7; the second copy buys
   little, since 32 random words in 32 banks and 16 in 16 have about the same worst bank.
2. **The weight order is the grid's own.** A group's nibbles read weights (0, 4, 1, 5, 2, 6, 3, 7), the order the old
   kernel already stages the activations in (`act_perm_iq1`); the shared staging code takes the order as a parameter.
3. **Two 3-bit scales per 32 weights.** IQ2_S's two DP4A chains and exact FFMAs (values within ±9, so the sums stay far
   below 2^22). The scales of four blocks (a word of two 16-bit scale words, ls0 and ls1 at bits 0–2, 3–5 and 6–8,
   9–11 of each) become the bytes 2 s + 1 of two words with 7 instructions: `((w << 1) & 0x000E000E) | ((w << 3) &
   0x0E000E00) | 0x01010101` for ls0, `((w >> 2) & 0x000E000E) | (w & 0x0E000E00) | 0x01010101` for ls1 (3 shifts, 4
   `lop3`); then IQ2_S's `scale_floats`, one I2F each.
4. **d in four instructions a super-block and matrix.** d's nibbles are the high nibbles of bytes 1 and 3 of both scale
   words: `prmt(w12, w13, 0x7531)` gathers those four bytes, a `lop3` with a copy shifted left by 4 (high nibbles from
   the bytes, low nibbles from the copy) makes bytes 1 and 3 d's two bytes, and a `prmt` moves them to the low half;
   then `cvt.f32.f16` and the multiply by 0.125 of the old kernel's `sblock_scale`.
5. **Rows straight into registers, no choice of words.** 560-byte rows (0 mod 16) and 56-byte super-blocks with
   nothing in front: lane 0's super-block starts on a 16-byte boundary, lane 1's on an 8-byte one, and each lane loads
   its 7 aligned 8-byte units from `segment + 56 sl`, whose words are the super-block's words. No `kappa`, no branch,
   no word PRMTs (IQ2_XS spends 2.25 instructions per 32 weights choosing words).

Result (static, per 32 weights of one matrix, one pass): 52.50 instructions in CUDA C, 45.69 in PTX at one member
(old 76.69, Q2_0 42.88); 94.06 and 87.00 at four members (old 134.62, Q2_0 79.62). Each further member costs 13.8
(old 19.3). The PTX saves 6.8 over nvcc's C: 4.0 from the table address (nvcc: a shift and an AND with 0x3FFC per
group), 2.0 from the I2F scales (C: a PRMT and an FADD each), 0.8 from d and the loop's frame.

## What ptxas did with IQ1_M's PTX

- `bfi.b32 ea, px, ZERO, 2, 12` and the add of the table's base became `BFI R, R, 0xc02, RZ` and `LDS.U.32 R, [R]`;
  `bfe` + `cvt.rn.f32.u32` became `I2F.F32.U8` with a byte select, `cvt.f32.f16` of a half `F2F.F32.F16`; the rest one
  to one.
- The iteration's multiplies by 112 (a lane pair's bytes of a row) and 272 (a super-block's staged activations) became
  two XMAD each, 0.25 per 32 weights; pointers stepped by the loop would remove them
  ([Q1899](../../../quants/experiments.md)).
- Registers: 124, 155 and 185 for groups of up to 2, 4 and 8 members, no spills (nvcc's C: 135, 158, 204); the
  8-member entry point's one-member loop is 45.56. The build's fatbin gave exactly the SASS of `ptxas -v` on the
  generated file alone.

## The four kernels: summary

Static SASS per 32 weights of one matrix, one pass of the inner loop, at one member (four members in parentheses),
the entry points for groups of up to 4 members; `gmix-q2`'s routed gate/up layers per type and R125's time a layer at
four tokens on a P106 with the old kernel:

| Type (layers in `gmix-q2`, R125 ms a layer) | old | CUDA C | PTX | PTX against C | what the PTX bought over nvcc |
|---|---:|---:|---:|---:|---|
| IQ2_XXS (11, 0.937) | 131.44 (183.19) | 76.94 (112.88) | 72.50 (108.06) | −4.44 | the table address (`bfe` + `ISCADD`, nvcc `BFE` + `BFI` + shift); all ten row loads issued first |
| IQ2_S (7, 0.978) | 136.31 (194.12) | 70.25 (111.56) | 62.94 (104.38) | −7.31 | the table address as one `bfi` into the copy offset (4.1), I2F scales (2.1), index and frame (1.3) |
| IQ2_XS (4, 1.042) | 142.69 (200.56) | 81.62 (123.56) | 73.00 (114.62) | −8.62 | the table address (6.0), I2F scales (2.0), frame (0.6) |
| IQ1_M (2, 0.547) | 76.69 (134.62) | 52.50 (94.06) | 45.69 (87.00) | −6.81 | the table address (4.0), I2F scales (2.0), d and frame (0.8) |
| Q2_0, for comparison (0.271) | 42.88 (79.62) | | | | |

- **What every rewrite removed:** the 32-bit multiplies (sign masks and scale products, 43–46 XMAD per 32 weights in
  the IQ2 types; Pascal builds each from three XMAD), the staging of the rows through shared memory, the second read of
  every member's activations (gate and up in one pass), and the integer-to-float conversions (the DP4A chains start at
  the float 1.5·2^23, so their results are floats). What is left per 8 weights is a few PRMTs and bit-field
  instructions around one random 4-byte shared lookup.
- **What the PTX bought, every time, is the same two things.** nvcc never inserts an index straight into a byte address
  with one `BFI` (it builds a word index and shifts, or masks a shifted value: 1–1.5 instructions a group more), and it
  keeps a PRMT and an FADD for a scale's float where `bfe` + `cvt` folds into one `I2F` with a byte select. The decode's
  PRMTs, the DP4A chains and the exact FFMAs it emitted one to one. Registers: IQ1_M's C took 135, 158 and 204 against
  the PTX's 124, 155 and 185; IQ2_S's C 247 at 8 members against 244, with a slower one-member loop there.
- **The order the counts give:** IQ1_M's PTX (45.7) is within 3 instructions of Q2_0's kernel; IQ2_S (62.9) next;
  IQ2_XXS (72.5) and IQ2_XS (73.0) spend 24 of theirs on the computed sign bits (6 a group), which a sign table would
  cut to 8 ([Q1879](../../../quants/experiments.md)).
- **What rig 1's timing should look at first** (QUEUES rig 1 row 6b, one tarball for all four types):
  1. **Whether a kernel's time follows its instruction count or its shared loads.** All four make 4 random 4-byte
     lookups per 32 weights, and Q2_0 makes none. IQ1_M is the test: its count is near Q2_0's, so if its ms a layer
     comes near Q2_0's 0.271 the P106 is issue-bound there, and fewer instructions pay (Q1879's sign tables for IQ2_XXS
     and IQ2_XS, [Q1898](../../../quants/experiments.md)'s selectors without the shift); if IQ1_M stays well above Q2_0,
     the lookups' bank conflicts are the limit (IQ1_M's 32 lanes share one table copy, IQ2_S's 8 lanes a copy, IQ2_XXS's
     and IQ2_XS's 4), and more table copies or fewer lookups matter more than instructions.
  2. **`=1` against `=2` per type:** the PTX's 4.4–8.6 fewer instructions (6–11 %) show only if issue is the limit.
  3. **Then the step:** est. up to 15.91 ms of `gmix-q2`'s 18.15 ms verify-pass gap if all four reach Q2_0's rate
     (R125's 15.36 for the IQ2 types, IQ1_M's 0.55).
- **Not measured yet:** the kernels' time. IQ2_S's table has 8 lanes per copy and IQ1_M's 32, so their lookups conflict more
  than IQ2_XXS's and IQ2_XS's 4; IQ2_XS and IQ1_M issue 1.25 and 0.88 global load instructions per 32 weights (8-byte
  units) where IQ2_XXS and IQ2_S issue 0.62 and 0.75.
- **Ideas parked until that timing:** the signs from a table (Q1879, est. −16 per 32 weights for IQ2_XXS and IQ2_XS);
  IQ2_XS's high-half entries without their shift ([Q1894](../../../quants/experiments.md), est. −2.0) and its rows in
  16-byte loads ([Q1895](../../../quants/experiments.md)); one lane per row for all types
  ([Q1889](../../../quants/experiments.md)); IQ1_M's second selector without the shift, from 8-byte table entries (32 KB:
  groups of up to 4 members only, 2 blocks per SM) or from a 2-byte load of the entry's upper half (one more shared load
  a group), est. 45.69 → 41.69 ([Q1898](../../../quants/experiments.md)); IQ2_XXS's first-group address by one `bfi`
  ([Q1888](../../../quants/experiments.md), est. −1.0).
