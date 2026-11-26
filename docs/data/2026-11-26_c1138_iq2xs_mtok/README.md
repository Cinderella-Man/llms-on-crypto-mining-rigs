# IQ2_XS's multi-token expert kernel, rewritten: CUDA C, then hand-written PTX (C1138, Q1871, 2026-11-26)

What this is: the third kernel of code row 28r (registry [Q1871](../../../quants/experiments.md)), after
[IQ2_XXS's](../2026-11-25_c1138_iq2xxs_mtok/README.md) and [IQ2_S's](../2026-11-26_c1138_iq2s_mtok/README.md). R125
found that Qwen3.8-Flash-Next's better-quality file `gmix-q2` is 18.15 ms a verify step slower than the release file,
almost all of it in the routed experts' gate/up products in the IQ2 grid types; at four tokens IQ2_XS's took 1.042 ms a
layer against Q2_0's 0.271, the slowest of the three ([R125](../2026-11-25_r125_gmix_step/README.md)). This folder holds
the IQ2_XS kernel's rewrite (4 of `gmix-q2`'s layers, est. 3.08 ms a verify pass over Q2_0): why the old kernel was
slow (static instruction counts), a fast version in CUDA C, the whole kernel in hand-written PTX, and the checks that
both give bit for bit the old kernel's results. What IQ2_XS taught, and what IQ1_M's kernel (the last of the four)
should do, is in the note [ptx_kernels.md](../../research/notes/ptx_kernels.md).

**No speed was measured here.** The build machine (three RTX 3090s) takes no speed measurements and the P106 rigs were
off; every count below is a static count of compiled instructions (what the compiler emitted for one pass of a loop),
not a time. The kernel's time on a P106 is rig 1's row (QUEUES), with the tarball below; the same run times IQ2_XXS's
and IQ2_S's.

## Result

**Both rewrites give the old kernel's results bit for bit, and by static count need 43 % (CUDA C) and 49 % (PTX) fewer
instructions at one member.** One pass of the inner loop, per 32 weights of one matrix (Flash-Next's 2,560-input
gate/up rows), `run/sass_loops.py` of the IQ2_XXS folder; the other types' columns are their folders':

| Members of a group | Q2_0, old | IQ2_XXS, PTX | IQ2_S, PTX | IQ2_XS, old | IQ2_XS fast, CUDA C | IQ2_XS fast, PTX |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 42.88 | 72.50 | 62.94 | 142.69 | 81.62 | 73.00 |
| 2 | 55.25 | 84.25 | 76.69 | 161.94 | 95.94 | 86.94 |
| 4 | 79.62 | 108.06 | 104.38 | 200.56 | 123.56 | 114.62 |
| each further member | 12.2 | 11.9 | 13.8 | 19.3 | 14.0 | 13.9 |

At four tokens a group usually has one or two members (512 experts, 10 used a token). The counts are those of the
entry points for groups of up to 4 members; in the entry point for up to 8 the one-member loops are 142.69 (old), 81.50
(CUDA C) and 73.06 (PTX). By kind, at one member ([sass_counts.tsv](sass_counts.tsv) has every kind and member count):

| Kind | IQ2_XS old | fast C | fast PTX |
|---|---:|---:|---:|
| XMAD (16-bit multiplies; three make a 32-bit one) | 46.12 | 0 | 0.25 |
| logic | 27.50 | 9.25 | 8.69 |
| shifts and bit fields (incl. BFE, BFI) | 21.69 | 28.44 | 22.38 |
| PRMT | 4.62 | 16.25 | 14.25 |
| POPC | 4.00 | 4.00 | 4.00 |
| integer add, compare, select | 13.75 | 1.25 | 1.12 |
| DP4A | 8.00 | 8.00 | 8.00 |
| float | 1.25 | 7.25 | 5.25 |
| conversions | 1.12 | 0.12 | 2.12 |
| shared loads: 4-byte / 8-byte / 16-byte, shared stores | 2.38 / 4.00 / 2.25 / 2.38 | 4.00 / 0 / 1.12 / 0 | 4.00 / 0 / 1.12 / 0 |
| global loads | 2.38 | 1.25 | 1.25 |
| control, moves | 1.25 | 0.69 | 0.56 |

**Where the old kernel's time went:** 46 XMAD per 32 weights: the 32-bit multiplies that turn each 4 sign bits into
byte masks (`(s4 * 0x00204081) & 0x01010101`, then `* 0xFF`; Pascal builds each from three XMAD) and, per member, the
two scale products `ls0 * i0 + ls1 * i1` (6 XMAD a member, most of the 19.3 a further member costs); 49 logic and shift
instructions around them; 4 random 8-byte grid lookups in shared memory (512 entries of 8 bytes); the row segments
staged through shared memory with 4-byte loads (740-byte rows allow no wider); every member's activations read twice,
once for up and once for gate.

**What the rewrite does** (details in [ptx_kernels.md](../../research/notes/ptx_kernels.md)): the method of the other
two types (gate and up in one pass; each 4 weights' grid value and sign in one PRMT from a nibble code table; no
multiplies; the rows straight into registers; IQ2_S's two DP4A chains per 32 weights turned into the old kernel's exact
`(float) (ls0 * i0 + ls1 * i1)` with two FFMA and one FADD, and its scale floats), and for IQ2_XS:
- **The rows in 8-byte units.** A super-block is 74 bytes, so lane 1's (74 bytes after lane 0's) sits 2 words further
  into a 16-byte unit than lane 0's, and 16-byte loads would give the two lanes different words to pick. Each lane loads
  the 10 aligned 8-byte units from `(segment & ~7) + 72 sl`; both lanes' super-blocks then start at the same word of
  their units (lane 1's two bytes later) and a warp-uniform branch of 2 cases picks the words. 1.25 global loads per 32
  weights against IQ2_S's 0.75.
- **The index:** each 8 weights' 16-bit entry holds a 9-bit grid index and 7 sign bits. The code table has 512 words
  (2 KB) and is kept in 8 copies interleaved by word (16 KB). In CUDA C the index is one BFE (nvcc then builds the
  address with a BFI beside the lane's copy and a shift: 3 instructions); in the PTX one BFI puts the 9 bits straight
  into the table's byte address, after a shift for an entry in a word's high half (1.5 instructions a group).
- **The signs** as IQ2_XXS's: 7 bits and their parity spread by a BFE, a shift, a LOP3, a POPC, a BFI and one PRMT
  per 8 weights.
- **The PTX's own choices** (what it saves over nvcc's C, 8.6 instructions per 32 weights): the table address (6.0),
  each scale's float as one I2F.F32.U8 (2.0; C: a PRMT under the exponent of 2^23 and an FADD), the loop's frame (0.6).

### Checks (LLMRig, one RTX 3090, the fork's Pascal kernels forced)

Every run on GPU 2 of LLMRig (an RTX 3090) with `scripts/pascal_forced.sh`'s settings, which force the fork's Pascal
kernels and a P106's launch shapes on a newer card (otherwise llama.cpp's own kernels run there); the rig build's
compute 6.1 code runs through the driver's PTX compiler, the hand-written kernels through their own PTX in the fatbins.

| Check | Base or arm | Result |
|---|---|---|
| Builds (`TESTS=1`, every check tool, web page b12674) | the C commit `acc6ba8ea`; the final `605eda167` | rc 0 both; the same 36 warning lines as IQ2_S's final build `886724824` |
| Old against new, `run/bitid.sh` with `TYPE=iq2_xs` (104 cases: the fused gate/up, the plain product and the fused pair with a shared expert added in the same graph, 1–8 tokens, five routings including repeated and skipped ids; 11,980,800 bytes of output) | C build: switch 1 against 0 | **byte-identical**; the test fault changes exactly the 56 fused cases of 2–8 tokens (with and without the shared expert) and nothing else |
| the same | final build: switch 1 and 2 against 0 | **byte-identical**, both; every case hash the same as in the C build; the fault at 2 changes exactly the same 56 cases |
| the same, with the IQ2_XXS folder's `run/launch_count.c` preloaded | final build: switch 2 | byte-identical; the PTX kernels launched 56 times (8 `ntm2`, 16 `ntm4`, 32 `ntm8`): every fused case of 2–8 tokens ran the hand-written kernel |
| IQ2_XXS's and IQ2_S's cases (`TYPE=iq2_xxs`, `iq2_s`), unchanged by this row | C build: 1; final build: 1 and 2, fault at 2 | byte-identical in every arm, every case's hash the same as in IQ2_S's folder; the faults 56 of 56; both types' PTX fatbins the same bytes as IQ2_S's final build's (md5 `8b8d0488…` and `9c277a2a…`), and the generators of both write their committed `.ptx` files byte for byte after this row's refactoring |
| Every kernel's SASS, `run/sass_all.sh` (`test-backend-ops` plus each build's PTX fatbins) | IQ2_S's final build `886724824` against the final build | all 9,526 kernel names of the base (10,093 copies, the IQ2_XXS and IQ2_S PTX entry points among them) the same bytes; 9 new: IQ2_XS's 6 CUDA C fast kernels and 3 PTX entry points; none gone. The 18 CUDA C fast kernels (three types) are the same bytes in the C commit's build and the final one |
| `test-backend-ops -o 'MUL_MAT_VEC_FUSION\|MUL_MAT_ID_FUSION'` (the fused gate/up cases), the three types' cases (`-p` on `type=iq2_xs,`, `type=iq2_s,` and `type=iq2_xxs,`; `run/gpu_checks.sh`'s `ARM_FILTER`), two arms at a time | final build: off, 1, 2 | 291 of 291 pass in each, every verdict the same; at 2 the PTX kernels launched 32 times for each type (4 `ntm2`, 8 `ntm4`, 20 `ntm8`). The fault (IQ2_XS's 52 cases) fails 36: the fused gate/up at 2,560 inputs, 2–8 tokens, 256 and 640 rows, with and without skipped slots and biases (biases run the C version), and IQ2_XS's PTX kernels launched the same 32 times; the 16 cases left to the old kernel pass (1 and 17 tokens, 1,024 inputs, 641 rows) |
| `test-backend-ops -o 'MUL_MAT_ID_FUSION\|MUL_MAT_ID_SHARED'`, the same cases | final build: off, 1, 2 | 24 of 24 pass in each, every verdict the same; the fault fails all 7 IQ2_XS routed + shared cases (2–8 tokens), whose shared expert runs in the fast C kernel |
| `test-backend-ops -o MUL_MAT_ID` (the plain op, llama.cpp's repeated-id cases left out), the same cases | final build: off, 1, 2 | 331 of 331 pass in each, every verdict the same (the plain product has no gate, so the fault cannot show: its 86 IQ2_XS cases pass) |
| Why only the IQ2 cases | — | The switch changes the dispatch of IQ2_XXS, IQ2_S and IQ2_XS only, and every other kernel is the same bytes as in IQ2_S's final build (above), which is IQ2_XXS's final build plus IQ2_S's kernels; IQ2_XXS's final build passed the full sets (2,389, 145 and 2,338 cases) |
| Real file: `gmix-q2` on one card (`run/gpu_checks.sh` step `model`): the IQ2_XS layers' routed gate/up on the card (`IQL`, layers 0, 2, 4, 5), every other expert in host memory, the MTP draft, prompts in micro-batches of 8, Flash-Next's v5 preset forced as a P106 runs it; two prompts, each answered with and without the top-8 probabilities, 192 greedy tokens | final build: 1 and 2 against 0 | **every reply the same token ids and text; the top-8 probabilities the same to the last printed digit at all 192 steps of both prompts**; the draft's proposals and acceptances the same counts (150 and 140, 216 and 119, 207 and 121), so the verify passes of 2–4 tokens computed the same logits. With `launch_count.c` preloaded the arm at 2 launched the PTX kernels 68 times, all IQ2_XS's (4 `ntm2`, 8 `ntm4`, 56 `ntm8`: verify passes and prompt micro-batches); the arms at 0 and 1 none. One run |
| Pipelined against plain decoding on the card, `bench/pipespec/gpu_pipe_check.sh` (default arms: F32 models on llama.cpp's kernels, their Q8_0 copies and the release types on the fork's, one card and three virtual cards; five architectures, six checks each) | IQ2_S's final build `886724824` against the final build, switch unset | 125 OK and 5 INFO in each build; all 130 checks give the base's plain-decoding token and logits hashes (and the same as in IQ2_S's folder) |

## Files

| File | What it is |
|---|---|
| [sass_counts.tsv](sass_counts.tsv) | Static SASS counts by kind of the inner loops (one pass, one path), per 32 weights of one matrix: IQ2_XS's old multi-token kernel, the fast CUDA C kernel and the PTX kernel, at 1–4 members (`sass_loops.py --tsv`) |
| [checks/bitid_runs.txt](checks/bitid_runs.txt), [checks/bitid_cases.txt](checks/bitid_cases.txt) | The old-against-new runs' verdicts and launch counts; IQ2_XS's 104 cases with each output's FNV-1a hash (the same in every arm) |
| [checks/bops_arms.tsv](checks/bops_arms.tsv), [checks/bops_fault_cases.txt](checks/bops_fault_cases.txt) | `test-backend-ops` by op set and arm with the PTX launches; the cases the test fault fails |
| [checks/model_compare.txt](checks/model_compare.txt), [checks/model_launch_count.txt](checks/model_launch_count.txt) | The real file's replies, arm against arm; the PTX launches of the arm at 2 |
| [checks/sass_all.txt](checks/sass_all.txt) | The whole-build SASS comparison and the new kernels |
| [checks/pipe_summary.tsv](checks/pipe_summary.tsv) | `gpu_pipe_check.sh`'s verdicts and plain-decoding hashes, base and branch |
| [run/bitid.sh](run/bitid.sh) | Old against new on one card: `bench/mmvq_pascal/verify/mtok_iq_fast_check` for a type (`TYPE`, default `iq2_xs`) with the switch at 0, 1, 2 and with the test fault, outputs compared byte for byte |
| [run/gpu_checks.sh](run/gpu_checks.sh) | `test-backend-ops` by op set with the switch off, at 1 and 2 and with the fault, at most `PAR` arms at a time; the real `gmix-q2` file on one card with the layers `IQL` lists on it (default its IQ2_XS layers), greedy replies with the MTP draft, switch off against on |
| [run/sass_all.sh](run/sass_all.sh) | Every kernel's SASS in the base build's and the branch's `test-backend-ops` and their PTX fatbins, kernel name by kernel name (`NEW`, default IQ2_XS's fast kernels) |
| [../2026-11-26_c1138_iq2s_mtok/run/split_sass.py](../2026-11-26_c1138_iq2s_mtok/run/split_sass.py) | Splits `cuobjdump -sass` output into one file per kernel (the input of `sass_loops.py`), used here unchanged |
| [../2026-11-25_c1138_iq2xxs_mtok/run/](../2026-11-25_c1138_iq2xxs_mtok/README.md#files) | The counter `sass_loops.py` and the launch counter `launch_count.c`, used here unchanged |
| [../../../bench/mmvq_pascal/verify/mtok_iq_fast_check.cpp](../../../bench/mmvq_pascal/verify/mtok_iq_fast_check.cpp) | The old-against-new harness (its second argument is the type) and its build script beside it |

## Exact commands

On the build machine, from a checkout of this repository; `W` is a work folder, `LOCK` a build lock file, `REPO` a
folder with `scripts/pascal_forced.sh`'s and `release/flash-next-v5/settings.env`'s copies, `MODELS` the folder with
`gmix-q2`'s shards, `DRAFTS` the one with the MTP draft, `BASE` IQ2_S's final build's `bin/` (`886724824`) and
`BASE_FATBINS` its two PTX fatbins, `OLD` the v5 build the branch is cut from (`2d839da1d`); every folder mounted at the
same path in the containers (`MOUNTS`), every container with `--cpu-shares=256` and a label (`DOCKER_ARGS`). `X` is the
IQ2_XXS folder's `run/`, `S` IQ2_S's, `R` this folder's.

```bash
# the builds (twice: the C commit acc6ba8ea, the final 605eda167)
flock "$LOCK" env WORK=$W/rig TESTS=1 JOBS=8 DOCKER_ARGS="--cpu-shares=256 -e HF_UI_VERSION=b12674" \
    bash scripts/build_llamacpp_docker.sh fork/pascal-p106-v5-iq-mtok-fast
T=$W/rig/llama.cpp-20261126-605eda167-pascal-p106-v5-iq-mtok-fast
# the harness against each build's tree (before the next build cleans it), the PTX fatbins of the final tree
bash bench/mmvq_pascal/verify/build_mtok_iq_fast_check.sh $W/rig/src $W/rig/src/build $W/bin2
cp $W/rig/src/build/ggml/src/ggml-cuda/ptx/*.fatbin $W/fatbin/
gcc -O2 -shared -fPIC -I/usr/local/cuda/include $X/launch_count.c -o $W/launch_count.so -ldl   # in the CUDA container
# static counts: a kernel's SASS, one file per kernel, then its inner loops (16 blocks of 32 weights a pass: a
# super-block of gate and of up); the old kernel from the v5 build, the C kernel from the C build, the PTX from its fatbin
cuobjdump -sass -arch sm_61 $OLD/test-backend-ops | python3 -I $S/split_sass.py /dev/stdin old 'mul_mat_vec_iq_pascal_mtokIL9ggml_type17E'
cuobjdump -sass -arch sm_61 $W/rig/llama.cpp-20261126-acc6ba8ea-pascal-p106-v5-iq-mtok-fast/bin/test-backend-ops > c.sass
python3 -I $S/split_sass.py c.sass c 'pascal_mtok_fast'
cuobjdump -sass $W/fatbin/mtok-iq2_xs.fatbin > ptx.sass
python3 -I $S/split_sass.py ptx.sass ptx 'mtok_iq2xs_fast_ptx_ntm'
python3 -I $X/sass_loops.py <kernel>.sass --units 16 [--tsv <name>]
# the PTX alone, before a build: registers, spills and SASS in a minute (the build's fatbin gave the same SASS)
python3 -I gen_mtok_iq2_xs.py > mtok-iq2_xs.ptx && ptxas -arch=sm_61 -v mtok-iq2_xs.ptx -o xs.cubin && cuobjdump -sass xs.cubin
# old against new on GPU 2: IQ2_XS (C build: ARMS=1 FAULT_ARM=1), then the other two types' cases, then the launch counter
for ty in iq2_xs iq2_xxs iq2_s; do
    env WORK=$W/bitid-$ty CHECK=$W/bin2/mtok_iq_fast_check REPO=$REPO GPU=2 ARMS="1 2" FAULT_ARM=2 TYPE=$ty \
        bash $R/bitid.sh
done
env WORK=$W/bitid-lc CHECK=$W/bin2/mtok_iq_fast_check REPO=$REPO GPU=2 ARMS=2 FAULT_ARM= TYPE=iq2_xs \
    PRELOAD=$W/launch_count.so bash $R/bitid.sh
# every kernel's SASS against IQ2_S's final build, the PTX fatbins included
env WORK=$W/sass BASE=$BASE HEAD=$T/bin BASE_EXTRA="$BASE_FATBINS" \
    HEAD_EXTRA="$W/fatbin/mtok-iq2_xxs.fatbin $W/fatbin/mtok-iq2_s.fatbin $W/fatbin/mtok-iq2_xs.fatbin" bash $R/sass_all.sh
# test-backend-ops, three op sets, the three types' cases (the fault on IQ2_XS's), two arms at a time
i=0
for ops in MUL_MAT_ID 'MUL_MAT_ID_FUSION|MUL_MAT_ID_SHARED' 'MUL_MAT_VEC_FUSION|MUL_MAT_ID_FUSION'; do
    i=$((i + 1))
    env WORK=$W/bops$i BIN=$T/bin REPO=$REPO GPU=2 MODES="1 2" PAR=2 STEPS=bops OPS="$ops" PRELOAD=$W/launch_count.so \
        ARM_FILTER='.*type(_a)?=iq2_(s|xs|xxs),' FAULT_FILTER='.*type(_a)?=iq2_xs,' bash $R/gpu_checks.sh
done
# the real file: gmix-q2's IQ2_XS layers (0|2|4|5, the default IQL) on the card; MOUNTS must also hold the folder its
# third shard links to
env WORK=$W/model BIN=$T/bin REPO=$REPO GPU=2 MODES="1 2" STEPS=model PRELOAD=$W/launch_count.so \
    MODEL=$MODELS/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-pmix-fast-gmix-q2-00001-of-00003.gguf \
    DRAFT=$DRAFTS/mtp-Swift-1.5-Qwen3.8-Flash-Next-draft.gguf bash $R/gpu_checks.sh
# pipelined against plain decoding on the card, IQ2_S's final build against this one
env WORK=$W/pipe BUILDS="base=$BASE head=$T/bin" GPUS=2 MODEL_SEED=1 bash bench/pipespec/gpu_pipe_check.sh
```

The PTX sources are regenerated in the fork with `python3 -I ggml/src/ggml-cuda/ptx/gen_mtok_iq2_xs.py >
ggml/src/ggml-cuda/ptx/mtok-iq2_xs.ptx` (and the same for `iq2_s` and `iq2_xxs`; run from that folder; they read the
grids from `ggml/src/ggml-common.h`); the committed files are the scripts' output byte for byte. `gen_mtok_iq2_xs.py`
imports the frame of `gen_mtok_iq2_xxs.py` and the two-scale parts of `gen_mtok_iq2_s.py`.

## The rig tarball

`llama.cpp-20261126-605eda167-pascal-p106-v5-iq-mtok-fast.tar.gz`, 1,680,089,508 bytes, md5
`00d761facd544811e5de76a5e15c601c` (`llama-server` `41d0fb7bd8a9c3396f02bf650c33052f`, `test-backend-ops`
`39a59c1b894fe25590f1a676ecf54f07`). It holds the three rewrites (IQ2_XXS's, IQ2_S's and IQ2_XS's) behind the same
switch. Its time on a P106 is the next step: the IQ2_XXS, IQ2_S and IQ2_XS routed gate/up per layer at four tokens with
the switch unset, at 1 and at 2 (R125: IQ2_XXS 0.937 ms unset, IQ2_S 0.978, IQ2_XS 1.042, Q2_0 0.271), then
Flash-Next's step with `gmix-q2`.

## Provenance and licence

The fork's branch `pascal-p106-v5-iq-mtok-fast` of [Miner Llama](https://github.com/Cinderella-Man/miner-llama), on
IQ2_S's `886724824` (off `pascal-p106-v5` at `2d839da1d`); llama.cpp's MIT licence. The weights in the harness are
random or quantized from random values; the model file is `gmix-q2`, built in this project from UkisAI's Swift-1.5
GSQ-RCO Q2_0 of Qwen3.8-Flash-Next ([its pick](../2026-11-04_fmt_mixes/README.md#the-v31-pick-2026-11-11)). Every
table here is this project's own output, public domain.
