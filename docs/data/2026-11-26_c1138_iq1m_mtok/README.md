# IQ1_M's multi-token expert kernel, rewritten: CUDA C, then hand-written PTX; all four types checked together (C1138, Q1871, 2026-11-26)

What this is: the fourth and last kernel of code row 28r (registry [Q1871](../../../quants/experiments.md)), after
[IQ2_XXS's](../2026-11-25_c1138_iq2xxs_mtok/README.md), [IQ2_S's](../2026-11-26_c1138_iq2s_mtok/README.md) and
[IQ2_XS's](../2026-11-26_c1138_iq2xs_mtok/README.md), and the check of all four rewrites together. R125 found that
Qwen3.8-Flash-Next's better-quality file `gmix-q2` is 18.15 ms a verify step slower than the release file, almost all
of it in the routed experts' gate/up products in its grid types; at four tokens IQ1_M's took 0.547 ms a layer against
Q2_0's 0.271 ([R125](../2026-11-25_r125_gmix_step/README.md)). IQ1_M holds 2 of `gmix-q2`'s layers (est. 0.55 ms a
verify pass over Q2_0). This folder holds why the old kernel was slow (static instruction counts), a fast version in
CUDA C, the whole kernel in hand-written PTX, the checks that both give bit for bit the old kernel's results, and the
real file run with the routed gate/up of all four types' layers on the card at once. What IQ1_M taught and the summary
of all four kernels are in the note [ptx_kernels.md](../../research/notes/ptx_kernels.md).

**No speed was measured here.** The build machine (three RTX 3090s) takes no speed measurements and the P106 rigs were
off; every count below is a static count of compiled instructions (what the compiler emitted for one pass of a loop),
not a time. The kernels' time on a P106 is rig 1's row (QUEUES), with the tarball below, for all four types at once.

## Result

**Both rewrites give the old kernel's results bit for bit, and by static count need 32 % (CUDA C) and 40 % (PTX) fewer
instructions at one member; the PTX is within 3 instructions of Q2_0's kernel.** One pass of the inner loop, per 32
weights of one matrix (Flash-Next's 2,560-input gate/up rows), `run/sass_loops.py` of the IQ2_XXS folder; the other
types' columns are their folders':

| Members of a group | Q2_0, old | IQ2_XXS, PTX | IQ2_S, PTX | IQ2_XS, PTX | IQ1_M, old | IQ1_M fast, CUDA C | IQ1_M fast, PTX |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 42.88 | 72.50 | 62.94 | 73.00 | 76.69 | 52.50 | 45.69 |
| 2 | 55.25 | 84.25 | 76.69 | 86.94 | 96.12 | 66.44 | 59.25 |
| 4 | 79.62 | 108.06 | 104.38 | 114.62 | 134.62 | 94.06 | 87.00 |
| each further member | 12.2 | 11.9 | 13.8 | 13.9 | 19.3 | 13.9 | 13.8 |

At four tokens a group usually has one or two members (512 experts, 10 used a token). The counts are those of the
entry points for groups of up to 4 members; in the entry point for up to 8 the one-member loops are 76.50 (old), 52.44
(CUDA C) and 45.56 (PTX). By kind, at one member ([sass_counts.tsv](sass_counts.tsv) has every kind and member count):

| Kind | IQ1_M old | fast C | fast PTX |
|---|---:|---:|---:|
| XMAD (16-bit multiplies; three make a 32-bit one) | 6.12 | 0 | 0.25 |
| logic | 17.50 | 6.00 | 1.12 |
| shifts and bit fields (incl. BFE, BFI) | 17.81 | 9.56 | 9.19 |
| PRMT | 8.00 | 14.00 | 12.25 |
| integer add, compare, select | 8.25 | 0.94 | 0.69 |
| DP4A | 8.00 | 8.00 | 8.00 |
| float | 1.25 | 7.25 | 5.25 |
| conversions | 1.12 | 0.12 | 2.12 |
| shared loads: 4-byte / 8-byte / 16-byte, shared stores | 4.00 / 0.88 / 2.25 / 0.50 | 4.00 / 0 / 1.12 / 0 | 4.00 / 0 / 1.12 / 0 |
| global loads | 0.50 | 0.88 | 0.88 |
| control, moves | 0.50 | 0.62 | 0.81 |

**Where the old kernel's time went:** IQ1_M has no sign bits, so the old kernel was already cheaper than the IQ2
types' (76.7 against their 131–143); its cost: 17.5 logic and 17.8 shift instructions extracting each group's 11-bit
grid index from `qs` and `qh`, the delta's sign and the 3-bit scales; per member the two scale products
`ls0 * i0 + ls1 * i1` and their conversion (6 XMAD and 1 I2F a member, most of the 19.3 a further member costs); 4
random 4-byte grid lookups in shared memory (2,048 entries); the 560-byte rows staged through shared memory; every
member's activations read twice, once for up and once for gate.

**What the rewrite does** (details in [ptx_kernels.md](../../research/notes/ptx_kernels.md)): the method of the other
three types (gate and up in one pass; each 4 weights' values in one PRMT from a nibble code; the rows straight into
registers; IQ2_S's two DP4A chains per 32 weights turned into the old kernel's exact `(float) (ls0 * i0 + ls1 * i1)`
with two FFMA and one FADD), and for IQ1_M:
- **The delta's sign in the table.** ggml's `iq1s_grid_gpu` holds each entry's grid + 1 (0, 1, 2) as nibbles, so bit 2
  of every nibble is free: the code table has 4,096 words, the 2,048 entries and the same with bit 2 of every nibble
  set (16 KB, one copy). A group's `qs` byte with its `qh` nibble above it (one PRMT) is a 12-bit index, the 11-bit grid
  index with the delta's sign on top, and the word it addresses is already the selector of the PRMTs that pick
  8 (grid ± 1/8) from the register pair {−7, 1, 9, 0, −9, −1, 7, 0}: no sign arithmetic. Per 8 weights 6 integer
  instructions in CUDA C (PRMT, shift and AND for the address, PRMT, shift, PRMT) and 5 in the PTX (one BFI puts the 12
  bits straight into the table's byte address), and one shared load.
- **The weight order is the grid's own** (0, 4, 1, 5, 2, 6, 3, 7 within each group of 8), as the old kernel stages the
  activations.
- **Two 3-bit scales per 32 weights:** four blocks' scales become the bytes 2 s + 1 of two words with 7 logic and shift
  instructions; each byte's float is a PRMT under the exponent of 2^23 and an FADD in CUDA C, one I2F.F32.U8 in the PTX.
- **d** from the scale words' top nibbles once per super-block and matrix (4 instructions in the PTX, then the
  conversion and the multiply by 0.125 of the old kernel's `sblock_scale`).
- **The rows in 8-byte units with no choice of words:** 560-byte rows and 56-byte super-blocks with nothing in front,
  so lane 0's super-block starts on a 16-byte boundary and lane 1's on an 8-byte one; each lane loads its own 7 aligned
  8-byte units, whose words are the super-block's words (IQ2_XS's rows need a branch and a PRMT a word for that).
- **The PTX's own choices** (what it saves over nvcc's C, 6.8 instructions per 32 weights): the table address (4.0),
  each scale's float as one I2F (2.0), d and the loop's frame (0.8).

### Checks (LLMRig, one RTX 3090, the fork's Pascal kernels forced)

Every run on GPU 2 of LLMRig (an RTX 3090) with `scripts/pascal_forced.sh`'s settings, which force the fork's Pascal
kernels and a P106's launch shapes on a newer card (otherwise llama.cpp's own kernels run there); the rig build's
compute 6.1 code runs through the driver's PTX compiler, the hand-written kernels through their own PTX in the fatbins.

| Check | Base or arm | Result |
|---|---|---|
| Builds (`TESTS=1`, every check tool, web page b12674) | the C commit `f43006396`; the final `883f21f2c` | rc 0 both; the same warning lines as IQ2_XS's final build `605eda167` |
| The PTX alone, before the build (`ptxas -arch=sm_61 -v` on the generated file) | — | 124, 155 and 185 registers for groups of up to 2, 4 and 8 members, no spills; the build's fatbin gave exactly the same SASS |
| Old against new, `run/bitid.sh` with `TYPE=iq1_m` (104 cases: the fused gate/up, the plain product and the fused pair with a shared expert added in the same graph, 1–8 tokens, five routings including repeated and skipped ids; 11,980,800 bytes of output, every value finite) | C build: switch 1 against 0 | **byte-identical**; the test fault changes exactly the 56 fused cases of 2–8 tokens (with and without the shared expert) and nothing else |
| the same | final build: switch 1 and 2 against 0 | **byte-identical**, both; every case hash the same as in the C build; the fault at 2 changes exactly the same 56 cases |
| the same, with the IQ2_XXS folder's `run/launch_count.c` preloaded | final build: switch 2 | byte-identical; the PTX kernels launched 56 times (8 `ntm2`, 16 `ntm4`, 32 `ntm8`): every fused case of 2–8 tokens ran the hand-written kernel |
| The harness for IQ1_M | — | random blocks now carry their random FP16 d in IQ1_M's place (the top nibbles of the four scale words) instead of bytes 0–1; the other types' cases draw the same values as before, and their outputs hash the same as in the IQ2_XS folder |
| IQ2_XXS's, IQ2_S's and IQ2_XS's cases (`TYPE=iq2_xxs`, `iq2_s`, `iq2_xs`), unchanged by this row | C build: 1, fault at 1; final build: 1 and 2, fault at 2 | byte-identical in every arm, every case's hash the same as in the IQ2_XS folder; the faults 56 of 56; the three types' PTX fatbins the same bytes as IQ2_XS's final build's (md5 `8b8d0488…`, `9c277a2a…`, `f676b0ab…`), and their generators write their committed `.ptx` files byte for byte after this row's changes to the shared frame |
| Every kernel's SASS, `run/sass_all.sh` (`test-backend-ops` plus each build's PTX fatbins) | IQ2_XS's final build `605eda167` against the C build and against the final build | all 9,535 kernel names of the base (10,102 copies, the three IQ2 types' PTX entry points among them) the same bytes in both; new: IQ1_M's 6 CUDA C fast kernels (C build), and they and IQ1_M's 3 PTX entry points (final build); none gone. The 24 CUDA C fast kernels (four types) are the same bytes in the C commit's build and the final one |
| `test-backend-ops -o 'MUL_MAT_VEC_FUSION\|MUL_MAT_ID_FUSION'` (the fused gate/up cases), the four types' cases (`-p` on `type=iq1_m,`, `type=iq2_xs,`, `type=iq2_s,` and `type=iq2_xxs,`; `run/gpu_checks.sh`'s `ARM_FILTER`), two arms at a time | final build: off, 1, 2 | 349 of 349 pass in each, every verdict the same; at 2 the PTX kernels launched 42 times for IQ1_M (6 `ntm2`, 12 `ntm4`, 24 `ntm8`) and 32 times for each IQ2 type. The fault (IQ1_M's 58 cases) fails 49: the fused gate/up at 2,560 inputs, 2–8 tokens, 256 and 640 rows, with and without skipped slots and biases (biases run the C version), and IQ1_M's PTX kernels launched the same 42 times; the 9 cases the fast kernels do not take pass (1 and 17 tokens; 1,024 inputs, whose rows have 4 lanes) |
| `test-backend-ops -o 'MUL_MAT_ID_FUSION\|MUL_MAT_ID_SHARED'`, the same cases | final build: off, 1, 2 | 32 of 32 pass in each, every verdict the same; the fault fails all 7 IQ1_M routed + shared cases of 2–8 tokens, whose shared expert runs in the fast C kernel, and passes the one of 1 token |
| `test-backend-ops -o MUL_MAT_ID` (the plain op, llama.cpp's repeated-id cases left out), the same cases | final build: off, 1, 2 | 402 of 402 pass in each, every verdict the same (the plain product has no gate, so the fault cannot show: its 71 IQ1_M cases pass) |
| Why only the IQ cases | — | The switch changes the dispatch of IQ2_XXS, IQ2_S, IQ2_XS and IQ1_M only, and every other kernel is the same bytes as in IQ2_XS's final build (above), which is IQ2_XXS's final build plus the other types' kernels; IQ2_XXS's final build passed the full sets (2,389, 145 and 2,338 cases) |
| Real file: `gmix-q2` on one card (`run/gpu_checks.sh` step `model`): the IQ1_M layers' routed gate/up on the card (`IQL`, layers 8 and 13), every other expert in host memory, the MTP draft, prompts in micro-batches of 8, Flash-Next's v5 preset forced as a P106 runs it; two prompts, each answered with and without the top-8 probabilities, 192 greedy tokens | final build: 1 and 2 against 0 | **every reply the same token ids and text; the top-8 probabilities the same to the last printed digit at all 192 steps of both prompts**; the draft's proposals and acceptances the same counts (150 and 140, 214 and 119, 208 and 121), so the verify passes of 2–4 tokens computed the same logits. With `launch_count.c` preloaded the arm at 2 launched the PTX kernels 38 times, all IQ1_M's (6 `ntm2`, 4 `ntm4`, 28 `ntm8`: verify passes and prompt micro-batches); the arms at 0 and 1 none. One run |
| **The branch's combined check:** the real file with the routed gate/up of **all four types' layers on the card at once** (`IQL` = layers 0–15 and 24–31: IQ2_XXS's 11, IQ2_S's 7, IQ2_XS's 4, IQ1_M's 2; layers 16–23 are Q2_0 and stay in host memory; 17,761 MiB on the card), otherwise as the step above | final build: 1 and 2 against 0 | **every reply the same token ids and text; the top-8 probabilities the same to the last printed digit at all 192 steps of both prompts**; the draft's counts the same (150 and 140, 220 and 117, 214 and 119). The arm at 2 launched every type's PTX kernels: IQ2_XXS 201 times, IQ2_S 129, IQ2_XS 76, IQ1_M 38 (Flash-Next's preset forces CUDA graphs, whose replays run captured kernels without a driver-API call, so these counts show which kernels ran, not how often per layer); the arms at 0 and 1 none. Every shared expert in these layers is of another type (Q2_0, IQ4_XS, IQ3_S, Q4_K, Q5_K, Q6_K), so the file exercises the routed products; the fast kernels' routed + shared launch is covered by the harness and `test-backend-ops`. One run |
| Pipelined against plain decoding on the card, `bench/pipespec/gpu_pipe_check.sh` (default arms: F32 models on llama.cpp's kernels, their Q8_0 copies and the release types on the fork's, one card and three virtual cards; five architectures, six checks each) | IQ2_XS's final build `605eda167` against the final build, switch unset | 125 OK and 5 INFO in each build; all 130 checks give the base's plain-decoding token and logits hashes (and the same as in the IQ2_XS folder) |

## Files

| File | What it is |
|---|---|
| [sass_counts.tsv](sass_counts.tsv) | Static SASS counts by kind of the inner loops (one pass, one path), per 32 weights of one matrix: IQ1_M's old multi-token kernel, the fast CUDA C kernel and the PTX kernel, at 1–4 members (`sass_loops.py --tsv`) |
| [checks/bitid_runs.txt](checks/bitid_runs.txt), [checks/bitid_cases.txt](checks/bitid_cases.txt) | The old-against-new runs' verdicts and launch counts (all four types); IQ1_M's 104 cases with each output's FNV-1a hash (the same in every arm) |
| [checks/bops_arms.tsv](checks/bops_arms.tsv), [checks/bops_fault_cases.txt](checks/bops_fault_cases.txt) | `test-backend-ops` by op set and arm with the PTX launches; the cases the test fault fails |
| [checks/model_compare.txt](checks/model_compare.txt), [checks/model_launch_count.txt](checks/model_launch_count.txt) | The real file with IQ1_M's layers on the card: replies arm against arm; the PTX launches of the arm at 2 |
| [checks/combined_compare.txt](checks/combined_compare.txt), [checks/combined_launch_count.txt](checks/combined_launch_count.txt) | The real file with all four types' layers on the card (the branch's combined check): replies arm against arm; the PTX launches of the arm at 2 |
| [checks/sass_all.txt](checks/sass_all.txt) | The whole-build SASS comparison and the new kernels |
| [checks/pipe_summary.tsv](checks/pipe_summary.tsv) | `gpu_pipe_check.sh`'s verdicts and plain-decoding hashes, base and branch |
| [run/bitid.sh](run/bitid.sh) | Old against new on one card: `bench/mmvq_pascal/verify/mtok_iq_fast_check` for a type (`TYPE`, default `iq1_m`) with the switch at 0, 1, 2 and with the test fault, outputs compared byte for byte |
| [run/gpu_checks.sh](run/gpu_checks.sh) | `test-backend-ops` by op set with the switch off, at 1 and 2 and with the fault, at most `PAR` arms at a time; the real `gmix-q2` file on one card with the layers `IQL` lists on it (default its IQ1_M layers; the header lists every type's), greedy replies with the MTP draft, switch off against on |
| [run/sass_all.sh](run/sass_all.sh) | Every kernel's SASS in the base build's and the branch's `test-backend-ops` and their PTX fatbins, kernel name by kernel name (`NEW`, default IQ1_M's fast kernels) |
| [../2026-11-26_c1138_iq2s_mtok/run/split_sass.py](../2026-11-26_c1138_iq2s_mtok/run/split_sass.py) | Splits `cuobjdump -sass` output into one file per kernel (the input of `sass_loops.py`), used here unchanged |
| [../2026-11-25_c1138_iq2xxs_mtok/run/](../2026-11-25_c1138_iq2xxs_mtok/README.md#files) | The counter `sass_loops.py` and the launch counter `launch_count.c`, used here unchanged |
| [../../../bench/mmvq_pascal/verify/mtok_iq_fast_check.cpp](../../../bench/mmvq_pascal/verify/mtok_iq_fast_check.cpp) | The old-against-new harness (its second argument is the type) and its build script beside it |

## Exact commands

On the build machine, from a checkout of this repository; `W` is a work folder, `LOCK` a build lock file, `REPO` a
folder with `scripts/pascal_forced.sh`'s and `release/flash-next-v5/settings.env`'s copies, `MODELS` the folder with
`gmix-q2`'s shards, `DRAFTS` the one with the MTP draft, `BASE` IQ2_XS's final build's `bin/` (`605eda167`) and
`BASE_FATBINS` its three PTX fatbins, `OLD` the v5 build the branch is cut from (`2d839da1d`); every folder mounted at
the same path in the containers (`MOUNTS`, which must also hold the folder `gmix-q2`'s third shard links to), every
container with `--cpu-shares=256` and a label (`DOCKER_ARGS`). `X` is the IQ2_XXS folder's `run/`, `S` IQ2_S's, `R`
this folder's.

```bash
# the builds (twice: the C commit f43006396, the final 883f21f2c)
flock "$LOCK" env WORK=$W/rig TESTS=1 JOBS=8 DOCKER_ARGS="--cpu-shares=256 -e HF_UI_VERSION=b12674" \
    bash scripts/build_llamacpp_docker.sh fork/pascal-p106-v5-iq-mtok-fast
T=$W/rig/llama.cpp-20261126-883f21f2c-pascal-p106-v5-iq-mtok-fast
# the harness against each build's tree (before the next build cleans it), the PTX fatbins of each tree
bash bench/mmvq_pascal/verify/build_mtok_iq_fast_check.sh $W/rig/src $W/rig/src/build $W/bin2
cp $W/rig/src/build/ggml/src/ggml-cuda/ptx/*.fatbin $W/fatbin/
gcc -O2 -shared -fPIC -I/usr/local/cuda/include $X/launch_count.c -o $W/launch_count.so -ldl   # in the CUDA container
# static counts: a kernel's SASS, one file per kernel, then its inner loops (16 blocks of 32 weights a pass: a
# super-block of gate and of up); the old kernel from the v5 build, the C kernel from the C build, the PTX from its fatbin
cuobjdump -sass -arch sm_61 $OLD/test-backend-ops | python3 -I $S/split_sass.py /dev/stdin old 'mul_mat_vec_iq_pascal_mtokIL9ggml_type29E'
cuobjdump -sass -arch sm_61 $W/rig/llama.cpp-20261126-f43006396-pascal-p106-v5-iq-mtok-fast/bin/test-backend-ops > c.sass
python3 -I $S/split_sass.py c.sass c 'iq1_m_pascal_mtok_fast'
cuobjdump -sass $W/fatbin/mtok-iq1_m.fatbin > ptx.sass
python3 -I $S/split_sass.py ptx.sass ptx 'mtok_iq1m_fast_ptx_ntm'
python3 -I $X/sass_loops.py <kernel>.sass --units 16 [--tsv <name>]
# the PTX alone, before a build: registers, spills and SASS in a minute (the build's fatbin gave the same SASS)
python3 -I gen_mtok_iq1_m.py > mtok-iq1_m.ptx && ptxas -arch=sm_61 -v mtok-iq1_m.ptx -o m.cubin && cuobjdump -sass m.cubin
# old against new on GPU 2: the four types (C build: ARMS=1 FAULT_ARM=1), then the launch counter
for ty in iq1_m iq2_xxs iq2_s iq2_xs; do
    env WORK=$W/bitid-$ty CHECK=$W/bin2/mtok_iq_fast_check REPO=$REPO GPU=2 ARMS="1 2" FAULT_ARM=2 TYPE=$ty \
        bash $R/bitid.sh
done
env WORK=$W/bitid-lc CHECK=$W/bin2/mtok_iq_fast_check REPO=$REPO GPU=2 ARMS=2 FAULT_ARM= TYPE=iq1_m \
    PRELOAD=$W/launch_count.so bash $R/bitid.sh
# every kernel's SASS against IQ2_XS's final build, the PTX fatbins included
env WORK=$W/sass BASE=$BASE HEAD=$T/bin BASE_EXTRA="$BASE_FATBINS" \
    HEAD_EXTRA="$W/fatbin/mtok-iq2_xxs.fatbin $W/fatbin/mtok-iq2_s.fatbin $W/fatbin/mtok-iq2_xs.fatbin $W/fatbin/mtok-iq1_m.fatbin" \
    bash $R/sass_all.sh
# test-backend-ops, three op sets, the four types' cases (the fault on IQ1_M's), two arms at a time
i=0
for ops in MUL_MAT_ID 'MUL_MAT_ID_FUSION|MUL_MAT_ID_SHARED' 'MUL_MAT_VEC_FUSION|MUL_MAT_ID_FUSION'; do
    i=$((i + 1))
    env WORK=$W/bops$i BIN=$T/bin REPO=$REPO GPU=2 MODES="1 2" PAR=2 STEPS=bops OPS="$ops" PRELOAD=$W/launch_count.so \
        ARM_FILTER='.*type(_a)?=(iq2_(s|xs|xxs)|iq1_m),' FAULT_FILTER='.*type(_a)?=iq1_m,' bash $R/gpu_checks.sh
done
# the real file: gmix-q2's IQ1_M layers (8|13, the default IQL) on the card, then the layers of all four types
M=$MODELS/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-pmix-fast-gmix-q2-00001-of-00003.gguf
D=$DRAFTS/mtp-Swift-1.5-Qwen3.8-Flash-Next-draft.gguf
env WORK=$W/model BIN=$T/bin REPO=$REPO GPU=2 MODES="1 2" STEPS=model PRELOAD=$W/launch_count.so MODEL=$M DRAFT=$D \
    bash $R/gpu_checks.sh
env WORK=$W/combined BIN=$T/bin REPO=$REPO GPU=2 MODES="1 2" STEPS=model PRELOAD=$W/launch_count.so MODEL=$M DRAFT=$D \
    IQL='0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|24|25|26|27|28|29|30|31' bash $R/gpu_checks.sh
# pipelined against plain decoding on the card, IQ2_XS's final build against this one
env WORK=$W/pipe BUILDS="base=$BASE head=$T/bin" GPUS=2 MODEL_SEED=1 bash bench/pipespec/gpu_pipe_check.sh
```

The PTX sources are regenerated in the fork with `python3 -I ggml/src/ggml-cuda/ptx/gen_mtok_iq1_m.py >
ggml/src/ggml-cuda/ptx/mtok-iq1_m.ptx` (and the same for `iq2_xs`, `iq2_s` and `iq2_xxs`; run from that folder; they
read the grids from `ggml/src/ggml-common.h`); the committed files are the scripts' output byte for byte.
`gen_mtok_iq1_m.py` imports the frame of `gen_mtok_iq2_xxs.py` and the two-scale parts of `gen_mtok_iq2_s.py`.

## The rig tarball

`llama.cpp-20261126-883f21f2c-pascal-p106-v5-iq-mtok-fast.tar.gz`, 1,695,190,933 bytes, md5
`e8b91355f12c344048736ee53576b7bf` (`llama-server` `dae09aa55e6fe7e875a0d57bbaf0b796`, `test-backend-ops`
`7f189af1205e17706d52cd90f9ca6add`). It holds the four rewrites (IQ2_XXS's, IQ2_S's, IQ2_XS's and IQ1_M's) behind the
same switch. Its time on a P106 is the next step: the four types' routed gate/up per layer at four tokens with the
switch unset, at 1 and at 2 (R125: IQ2_XXS 0.937 ms unset, IQ2_S 0.978, IQ2_XS 1.042, IQ1_M 0.547, Q2_0 0.271), then
Flash-Next's step with `gmix-q2`. What to look at first is in the [note's summary](../../research/notes/ptx_kernels.md#the-four-kernels-summary).

## Provenance and licence

The fork's branch `pascal-p106-v5-iq-mtok-fast` of [Miner Llama](https://github.com/Cinderella-Man/miner-llama), on
IQ2_XS's `605eda167` (off `pascal-p106-v5` at `2d839da1d`); llama.cpp's MIT licence. The weights in the harness are
random or quantized from random values; the model file is `gmix-q2`, built in this project from UkisAI's Swift-1.5
GSQ-RCO Q2_0 of Qwen3.8-Flash-Next ([its pick](../2026-11-04_fmt_mixes/README.md#the-v31-pick-2026-11-11)). Every
table here is this project's own output, public domain.
