# The 10,240-wide BF16 products at 3–4 tokens in the one-token order (C750, Q1325, 2026-11-11)

What this is: a new kernel and its checks on a build machine's CPU, with no GPU. In a verify pass of 3 or 4 tokens,
Miner Llama's Pascal BF16 kernel runs Qwen3.8-Flash-Next's 10,240-wide hyper-connection products (`hc_attn_down`,
`hc_ffn_down`, `hc_head_down`, 10,240 → 320) with 32 lanes per row group, where one token runs 128. A row's sum depends
on the lane count, so these products' sums differ from one-token decoding's
([C730](../2026-11-11_c730_bf16_hc/README.md): 2,333 of 2,560 emulated outputs). The existing switch
`GGML_CUDA_MMVF_PASCAL_RULE=1` makes them exact at 128 lanes' speed (63.384 and 62.432 µs a call at 4 tokens in two
P106 benches, against 46.528 and 44.728 at 32 lanes). This change gives the one-token sums at 32 lanes' loads.
Registry row [Q1325](../../../quants/experiments.md); fork branch `pascal-p106-v4-b12246-c750` (`69dfccc81`).

**The answer:**
- **`GGML_CUDA_MMVF_PASCAL_EXACT=1`** (off by default) adds every BF16 product of 2–4 tokens in the one-token order.
  On a P106-090 (5 SMs) that changes one thing: the 10,240-wide rows at 3–4 tokens run a new 32-lane kernel,
  `mul_mat_vec_bf16_pascal_x4p`, whose sums are the 128-lane kernel's bit for bit.
- **How:** the 128-lane kernel's lane `j + 32v` (its warp v) adds the chunks `j + 32v`, `j + 32v + 128`, … in one
  chain. Lane `j` of the new kernel runs those four chains one after another. Pass v adds its chain with the same FMAs
  and ends in the same butterfly, which gives warp v's sum. The running sum then takes it: `s = S0`, then `s += S1`,
  `s += S2`, `s += S3`, as the 128-lane kernel's warp 0 adds warps 1, 2 and 3.
- **The same loads as 32 lanes:** each weight is loaded once, by the same lane as in the 32-lane kernel, in four
  passes of stride 128 instead of one of stride 32. The loop has the 32-lane kernel's 2R + 4T loads and 16RT FMAs a
  pass. What it adds is three more butterflies a row (320 shuffles a warp at 4 tokens, against 80), est. at most
  about 2 µs a call, against the 16.856 and 17.704 µs that 128 lanes add.
- **Checked on the CPU:** an emulation of the kernel's code gives the 128-lane sums on every output (0 of 2,560 at
  Flash-Next's shape, 0 of 16,384 over every row width of 1 to 2,048 chunks). Every kernel of the base build is the
  same in the new build. ctest and `test-llama-archs` give the base's results. The rig's replay set is recorded, with a
  fault run that must fail.
- **Not yet on a P106:** bit identity on the card, the speed, and the served verify pass are R97's.

## The change

`ggml/src/ggml-cuda/mmvf-pascal-impl.cuh` (two kernels), `mmvf-pascal.cu` (the switch and where it applies),
`mmvf-pascal.cuh` (the switch's line), `tests/test-backend-ops.cpp` (three row widths).

- **Where it applies.** With the switch on, a product of 2–4 tokens takes the launch shape the rule gives
  (`mmvfp_pick_shape`, [C730's shapes.tsv](../2026-11-11_c730_bf16_hc/shapes.tsv)), and the one-token shape beside it:
  - if their lane counts are the same, nothing changes (the same lanes give the same sums at any row count);
  - if one token takes 128 lanes and 3–4 tokens 32 (rows of more than 512 chunks with enough rows for 20 blocks of 16:
    on 5 SMs the hyper-connections' `hc_*_down` and `hc_head_down`), the new kernel takes it, at the rule's 4 rows a
    lane group;
  - otherwise (on a card of 10 SMs the indexer's 2,560 → 512 takes 128 lanes at 3–4 tokens and 32 at one), the
    product takes its one-token shape.

  A forced shape (`GGML_CUDA_MMVF_PASCAL_SHAPE`) is left as it is; 5–8 tokens are unchanged.
- **`GGML_CUDA_MMVF_PASCAL_EXACT=2`** runs the same order through a second kernel, `mul_mat_vec_bf16_pascal_x4`. It
  keeps the four chains interleaved in one pass: lane j takes the 32-lane kernel's chunks j, j + 32, j + 64, … in
  order and adds the n-th to accumulator n mod 4. That is the row's first design, with 4TR accumulators a lane. At the
  128 registers its launch bound allows it keeps some of them on the stack inside its loop (below), so the passes
  form is the default. R97 times both.
- **`MMVF_PASCAL_X4_MIN_BLOCKS`** (build define, default 4): both kernels are built for 4 blocks a SM, at most 128
  registers. Then a 10,240 × 320 product's 20 blocks of 128 threads fit on 5 SMs in one wave, as the 32-lane kernel's
  do.
- **`GGML_CUDA_MMVF_PASCAL_EXACT_FAULT=1`** (checks only) leaves the fourth chain's sum out of every row of either
  kernel, a quarter of the terms, which every check of these products must catch. Without the switch it reaches
  nothing.
- **Tests:** three row widths in `test-backend-ops` take 32 lanes at 3–4 tokens and 128 at one on 5 SMs:
  K = 4,104, 4,392 and 4,736 (513, 549 and 592 chunks, 320, 320 and 333 rows). In the interleaved kernel they end a
  lane's chunks 1, 2 or 3 after its last group of 4, and 333 rows leave the last block part-filled.
- With the switch at 0 the dispatch is the base's line for line; the new code runs only behind it.

## The sums

[run/order_emul.c](run/order_emul.c) extends C730's emulation with both kernels, step by step as their code runs
(loops, tails, butterflies, the order of the final additions). It sums BF16 weights (normal, standard deviation 0.02)
against FP32 activations (standard normal), with `fmaf` and denormals flushed as the kernels' FTZ instructions do, and
compares every output's bits with the 128-lane kernel's ([order_emul.tsv](order_emul.tsv)):

| Rows | Width K, rows (columns) | Against the 128-lane order | Outputs | Differ | Median difference of those, ULP | Largest difference / sum of \|terms\| |
|---|---|---|---:|---:|---:|---:|
| `hc_*_down` | 10,240, 320 (8) | x4p (`EXACT=1`) | 2,560 | **0** | – | 0 |
| | | x4i (`EXACT=2`) | 2,560 | **0** | – | 0 |
| | | 32 lanes (the rule today, the control) | 2,560 | 2,323 | 5 | 2.04e-08 |
| | | x4p with the fault (the control) | 2,560 | 2,560 | 7,011,236 | 3.13e-02 |
| test width, 513 chunks | 4,104, 320 (4) | x4p; x4i | 1,280; 1,280 | **0; 0** | – | 0 |
| test width, 549 chunks | 4,392, 320 (4) | x4p; x4i | 1,280; 1,280 | **0; 0** | – | 0 |
| test width, 592 chunks | 4,736, 333 (4) | x4p; x4i | 1,332; 1,332 | **0; 0** | – | 0 |
| every width of 1 to 2,048 chunks | 8 to 16,384, 4 each (2) | x4p; x4i | 16,384; 16,384 | **0; 0** | – | 0 |

- **Both kernels repeat the one-token sums on every output,** at Flash-Next's shape, at the three test widths and at
  every row width of 1 to 2,048 chunks (4 rows and 2 columns each: every tail, and lanes with no chunk).
- **The controls differ:** the 32-lane order on 2,323 of 2,560 outputs (C730 counted 2,333 with its own draw of
  weights), the fault on every output.

## The machine code

[sass_x4.tsv](sass_x4.tsv) is cuobjdump's sm_61 code of the build's `mmvf-pascal.cu.o` (md5 `fd4a846492fca6d8d89d6c0f12f1504b`), read by
[run/sass_x4.sh](run/sass_x4.sh), beside the base kernel's 32- and 128-lane shapes at 4 rows. A "pass" is two chunks of
8 weights of each of the lane's 4 rows (four in the interleaved kernel):

| Kernel | Tokens | Registers | Stack, bytes | Stack accesses: in the kernel / in a loop | Loop LDG (expected) | Loop FFMA (expected) | Loop conversions | Other loop instructions | SHFL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 lanes (the rule today) | 3 | 80 | 0 | 0 / 0 | 20 (20) | 192 (192) | 64 | 40 | 60 |
| | 4 | 96 | 0 | 0 / 0 | 24 (24) | 256 (256) | 64 | 46 | 80 |
| 128 lanes (one token's order) | 3 | 80 | 0 | 0 / 0 | 20 (20) | 192 (192) | 64 | 29 | 60 |
| | 4 | 96 | 0 | 0 / 0 | 24 (24) | 256 (256) | 64 | 48 | 80 |
| **x4p** (`EXACT=1`) | 3 | 128 | 8 | 2 / **0** | 20 (20) | 192 (192) | 64 | 31 | 240 |
| | 4 | 128 | 16 | 12 / **0** | 24 (24) | 256 (256) | 64 | 34 | 320 |
| x4i (`EXACT=2`; a pass is 4 chunks) | 3 | 128 | 40 | 26 / 8 | 40 (40) | 384 (384) | 130 | 57 | 240 |
| | 4 | 128 | 96 | 85 / 36 | 48 (48) | 512 (512) | 131 | 104 | 320 |

- **The passes kernel loads once per row and chunk:** its loop is the 32-lane kernel's, 2R + 4T loads and 16RT FMAs a
  pass, each weight converted once (16R conversions a pass), with fewer other instructions (34 against 46 at 4
  tokens).
- **It touches the stack only between passes:** 2 accesses at 3 tokens, 12 at 4 (running sums parked across a pass,
  stored before its loop and loaded after it), none in a loop.
- **The interleaved kernel spills in its loop:** 36 stack accesses inside it at 4 tokens, 8 at 3. It needs 4TR = 64
  accumulators at 4 tokens, against the passes kernel's 2TR = 32 sums.
- **The cost is the butterflies:** 5 × 4 × T × R shuffles a warp (320 at 4 tokens), against 80. At one warp
  instruction a clock per SM, the 240 extra of 16 warps a SM take est. 2.0 µs at the P106-090's 1,911 MHz. Part of
  it overlaps other warps' loads, since each pass's butterfly comes in the middle of the kernel.

## Checks on the build machine

[run/chain.sh](run/chain.sh), CPU only, every step in a container without GPUs at a low CPU share; results in
[summary.tsv](summary.tsv):

| Check | Result |
|---|---|
| emulation | both kernels' orders give the 128-lane sums on every output (the table above) |
| rig tarball | `llama.cpp-20261111-69dfccc81-pascal-p106-v4-b12246-c750.tar.gz`, 658,057,828 bytes, md5 `df7fc616b618b8a9f970cd9062030a51`; `TESTS=1`, sm_61, 0 AVX instructions in `llama-bench`, GLIBC_2.38 at most |
| the new kernels' code | the table above: x4p loads once per row and chunk and touches the stack in no loop |
| every other kernel | `cuobjdump -sass` and `-ptx` of `llama-server` and `test-backend-ops` against the base's tarball (`5b329df16`), function by function: every base function the same in all four listings; new only the 8 instantiations of `mul_mat_vec_bf16_pascal_x4p` and `_x4` (R = 4; T = 3, 4; fault off, on) |
| ctest | the branch's static CPU build (every target): 70 of 70 pass (the 7 tests that need a download, the network or torch not run) |
| `test-llama-archs` | the base's static CPU build writes 146 architectures' logits, the branch's compares: 0 differ; 151 of 151 test cases pass in both |
| `multitoken_bitid` | mode `bf16`, linked against the rig build's tree: md5 `8a425cb9637e4506a07f3a0cd08ed433` (it runs on the rig) |
| replay set `bf16x` | 152 cases (BF16's 2-D `MUL_MAT` at 1–8 tokens), 9 runs ([replay_runs.tsv](replay_runs.tsv)), recorded on the CPU with the rig build's `test-backend-ops`; tar 1,200,547,840 bytes, md5 `f00104d99ff44a68aff262bb0ec144fe`. The fault runs select the 8 cases the 32-lane kernels take |
| CPU replay | the set's 152 cases replayed on the CPU with the rig build: 152 of 152 pass (the set's files are whole) |

The branch changes only CUDA files and `test-backend-ops`' list of cases, so the CPU builds compile the base's code:
ctest and `test-llama-archs` show the CPU side untouched. The switch off on a card rests on the machine code: every
kernel of the base is the same, and with the switch at 0 the dispatch runs the base's lines.

## What the rig run must show (R97)

[run/rig_c750.sh](run/rig_c750.sh) on a P106-090 rig, the rig idle, with the branch's tarball, the set and the
harness:

1. **replay:** the set `bf16x` through `scripts/replay_checks.sh`; every run ends `ok`, and `bf16x-fault` and
   `bf16x-fault2` fail every one of their cases (the products the 32-lane kernels take, 8 cases).
2. **bitid:** `multitoken_bitid bf16` with the switch at 1, at 2 and at 1 as for 10 SMs gives `0 with differences`;
   off (the control) the 3–4-token products of the widths whose lanes change differ (K = 1,000, 4,104, 4,392, 4,736
   and 10,240), and only those.
3. **perf:** `test-backend-ops perf` at Flash-Next's BF16 shapes, 1–4 tokens, arms `off`, `on`, `on2`, `rule1`, two
   passes. The verdict per kernel, each pass by value: gain, negligible or loss for `hc_*_down` at 3 and 4 tokens
   against `off` (the 32-lane order, not exact) and against `rule1` (exact at 128 lanes). The other shapes and 1–2
   tokens run the same kernel in every arm and give the spread.
4. **served:** R79's profile of a served verify pass (R49's
   [verify_prof.sh](../2026-11-10_served_verify_parts/run/verify_prof.sh)) on the Flash-Next preset with the MTP
   draft, `off on on off`: the hyper-connections' time per verify pass and the step time of the warm loads.

## Files

| File | What |
|---|---|
| [order_emul.tsv](order_emul.tsv) | the emulation: outputs compared and differing, per shape and pair of orders |
| [sass_x4.tsv](sass_x4.tsv) | the new kernels' and the base kernel's registers, stack, loop counts, shuffles and stack accesses |
| [summary.tsv](summary.tsv) | the build machine's checks, one line each |
| [replay_runs.tsv](replay_runs.tsv) | the set's runs: switches, filter, expected verdict (`runs-ptx.tsv`'s are the same with `GGML_CUDA_MMVF_PASCAL=2 GGML_CUDA_MMVF_PASCAL_NSM=5` first) |
| [replay_cases.tsv](replay_cases.tsv) | the set's cases: id and case string |
| [run/order_emul.c](run/order_emul.c) | the emulation |
| [run/sass_x4.sh](run/sass_x4.sh) | reads cuobjdump's output into `sass_x4.tsv` |
| [run/chain.sh](run/chain.sh) | the build machine's checks |
| [run/rig_c750.sh](run/rig_c750.sh) | the rig run (R97) |

Also changed for this row: `scripts/record_checks.sh` (suite `bf16x`), `bench/mmvq_pascal/verify/multitoken_bitid.cpp`
(mode `bf16`), [docs/switches.md](../../switches.md).

## Commands

On an x86-64 build machine with Docker and the `nvidia/cuda:12.9.2-devel-ubuntu24.04` image, no GPU (`BASE` is the
base build's rig tarball, of `pascal-p106-v4-b12246-c748` `5b329df16`; `SRC` holds a llama.cpp clone with the fork as
remote `fork`):

```bash
WORK=$WORK PUB=$PWD BASE=$BASE BASE_REF=5b329df16 LOCK=$WORK/../lock JOBS=8 DOCKER_ARGS='--label pascal-quants=1' \
    SEED_SRC=$SRC nice -n 19 bash docs/data/2026-11-11_c750_bf16_exact/run/chain.sh
# the emulation alone, on any x86-64 machine:
gcc -O2 -ffp-contract=off -o order_emul docs/data/2026-11-11_c750_bf16_exact/run/order_emul.c -lm && ./order_emul
```

On the rig (the branch's tarball unpacked, the set and the harness copied; `rig_c750.sh`'s header names every
variable):

```bash
BIN=<tarball>/bin OUT=<out> SET=<set>/bf16x REPLAY=scripts/replay_checks.sh BITID=<multitoken_bitid> \
    MODEL=<Flash-Next GGUF> DRAFT=<MTP draft GGUF> PRESET=release/flash-next-v3/settings.env \
    RIG_SERVER=bench/agent_check/rig_server.sh REQ=docs/data/2026-11-04_draft_one_card/run/greedy_request.json \
    VERIFY_PROF=docs/data/2026-11-10_served_verify_parts/run/verify_prof.sh \
    setsid nohup bash docs/data/2026-11-11_c750_bf16_exact/run/rig_c750.sh > console.txt 2>&1 < /dev/null &
```

## Provenance and licence

Code: Miner Llama (`github.com/Cinderella-Man/miner-llama`, MIT, llama.cpp's licence), branch
`pascal-p106-v4-b12246-c750` at `69dfccc81` (two commits on `pascal-p106-v4-b12246-c748` `5b329df16`). The emulation's
and the tests' inputs are generated (fixed seeds), not model weights. Everything here is this repository's licence.
