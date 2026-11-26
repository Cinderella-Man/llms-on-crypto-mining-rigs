# IQ2_S's multi-token expert kernel, rewritten: CUDA C, then hand-written PTX (C1138, Q1871, 2026-11-26)

What this is: the second kernel of code row 28r (registry [Q1871](../../../quants/experiments.md)), after
[IQ2_XXS's](../2026-11-25_c1138_iq2xxs_mtok/README.md). R125 found that Qwen3.8-Flash-Next's better-quality file
`gmix-q2` is 18.15 ms a verify step slower than the release file, almost all of it in the routed experts' gate/up
products in the IQ2 grid types; at four tokens IQ2_S's took 0.978 ms a layer against Q2_0's 0.271
([R125](../2026-11-25_r125_gmix_step/README.md)). This folder holds the IQ2_S kernel's rewrite (7 of `gmix-q2`'s
layers, est. 4.95 ms a verify pass over Q2_0): why the old kernel was slow (static instruction counts), a fast version
in CUDA C, the whole kernel in hand-written PTX, and the checks that both give bit for bit the old kernel's results.
What IQ2_S taught for the next types is in the note [ptx_kernels.md](../../research/notes/ptx_kernels.md).

**No speed was measured here.** The build machine (three RTX 3090s) takes no speed measurements and the P106 rigs were
off; every count below is a static count of compiled instructions (what the compiler emitted for one pass of a loop),
not a time. The kernel's time on a P106 is rig 1's row (QUEUES), with the tarball below; the same run times IQ2_XXS's.

## Result

**Both rewrites give the old kernel's results bit for bit, and by static count need 48 % (CUDA C) and 54 % (PTX) fewer
instructions at one member.** One pass of the inner loop, per 32 weights of one matrix (Flash-Next's 2,560-input
gate/up rows), `run/sass_loops.py` of the IQ2_XXS folder; IQ2_XXS's and Q2_0's columns are that folder's:

| Members of a group | Q2_0, old | IQ2_XXS, PTX | IQ2_S, old | IQ2_S fast, CUDA C | IQ2_S fast, PTX |
|---:|---:|---:|---:|---:|---:|
| 1 | 42.88 | 72.50 | 136.31 | 70.25 | 62.94 |
| 2 | 55.25 | 84.25 | 155.56 | 84.06 | 76.69 |
| 4 | 79.62 | 108.06 | 194.12 | 111.56 | 104.38 |
| each further member | 12.2 | 11.9 | 19.3 | 13.8 | 13.8 |

At four tokens a group usually has one or two members (512 experts, 10 used a token). The counts are those of the
entry points for groups of up to 4 members; in the entry point for up to 8 the PTX kernel's loops are the same
(62.94 at one member), the CUDA C kernel's one-member loop grows to 77.69 (247 registers) and the old kernel's stays
136.25. By kind, at one member ([sass_counts.tsv](sass_counts.tsv) has every kind and member count):

| Kind | IQ2_S old | fast C | fast PTX |
|---|---:|---:|---:|
| XMAD (16-bit multiplies; three make a 32-bit one) | 46.12 | 0.38 | 0.25 |
| logic | 21.50 | 7.25 | 6.69 |
| shifts and bit fields (incl. BFE, BFI) | 21.44 | 14.25 | 10.12 |
| PRMT | 6.12 | 24.50 | 22.50 |
| integer add, compare, select | 15.25 | 1.50 | 1.31 |
| DP4A | 8.00 | 8.00 | 8.00 |
| float | 1.25 | 7.25 | 5.25 |
| conversions | 1.12 | 0.12 | 2.12 |
| shared loads: 4-byte / 8-byte / 16-byte, shared stores | 2.62 / 4.00 / 2.25 / 2.62 | 4.00 / 0 / 1.12 / 0 | 4.00 / 0 / 1.12 / 0 |
| global loads | 2.62 | 0.75 | 0.75 |
| control, moves | 1.38 | 1.12 | 0.81 |

**Where the old kernel's time went:** 46 XMAD per 32 weights: the 32-bit multiplies that turn each 4 sign bits into
byte masks (`(s4 * 0x00204081) & 0x01010101`, then `* 0xFF`; Pascal builds each from three XMAD) and, per member, the
two scale products `ls0 * i0 + ls1 * i1` (6 XMAD a member, which is most of the 19.3 a further member costs); 43 logic
and shift instructions around them; 4 random 8-byte grid lookups in shared memory (1,024 entries of 8 bytes); the row
segments staged through shared memory with 4-byte loads (820-byte rows allow no wider); every member's activations read
twice, once for up and once for gate.

**What the rewrite does** (details in [ptx_kernels.md](../../research/notes/ptx_kernels.md)): IQ2_XXS's method (gate
and up in one pass; each 4 weights' grid value and sign in one PRMT from a nibble code table; no multiplies; 16-byte
loads straight into registers, 6 a lane and matrix for IQ2_S's 82-byte super-blocks), and for IQ2_S:
- **The 10-bit index** in one PRMT, which puts the group's `qs` byte beside its block's `qh` byte taken from a copy of
  the `qh` word shifted right by 2 l, then 10 bits kept. The code table has 1,024 words (4 KB) and is kept in 4 copies
  interleaved by word (16 KB): 8 copies would not fit beside 8 members' staged activations.
- **8 explicit sign bits** (no parity): the sign word masked with `0x33333333`, and shifted by 2 and masked, holds every
  group's sign pairs in nibbles; one PRMT picks a group's two bytes of them as the selector of the PRMT that spreads the
  signs. 2.75 instructions per 8 weights against IQ2_XXS's 6.
- **Two scales per 32 weights:** two DP4A chains (weights 0–15 and 16–31), each started at the float 1.5·2^23; then
  `fma(ls1, F1, -ls1·1.5·2^23)` is `ls1·i1` and `fma(ls0, F0 - 1.5·2^23, ls1·i1)` is `ls0·i0 + ls1·i1`, both exact (every
  value below 2^24): the old kernel's `(float) (ls0 * i0 + ls1 * i1)` with 4 float operations a member and matrix and no
  multiply or conversion. The weight order chosen within each group of 8 keeps both halves' weights apart.
- **The scales** of 4 blocks as bytes `2 s + 1` of two words (two instructions a word); in CUDA C each becomes a float
  by a PRMT under the exponent of 2^23 and an FADD, in the PTX by one I2F of its byte.
- **The PTX's own choices** (what it saves over nvcc's C, 7.3 instructions per 32 weights): the table address as one BFI
  of the PRMT's 10 index bits into the lane's copy offset, the table's base folded by ptxas into the load (C: AND,
  shift, add); each scale's float as one I2F.F32.U8 with the byte picked by ptxas (C: PRMT and FADD); and the same
  loops in every entry point.

### Checks (LLMRig, one RTX 3090, the fork's Pascal kernels forced)

Every run on GPU 2 of LLMRig (an RTX 3090) with `scripts/pascal_forced.sh`'s settings, which force the fork's Pascal
kernels and a P106's launch shapes on a newer card (otherwise llama.cpp's own kernels run there); the rig build's
compute 6.1 code runs through the driver's PTX compiler, the hand-written kernels through their own PTX in the fatbins.

| Check | Base or arm | Result |
|---|---|---|
| Builds (`TESTS=1`, every check tool, web page b12674) | the C commit `9ca44b706`; the final `886724824` | rc 0 both; 4 warning lines each, the same as IQ2_XXS's final build `6aa2a8df0`'s |
| Old against new, `run/bitid.sh` with `TYPE=iq2_s` (104 cases: the fused gate/up, the plain product and the fused pair with a shared expert added in the same graph, 1–8 tokens, five routings including repeated and skipped ids; 11,980,800 bytes of output) | C build: switch 1 against 0 | **byte-identical**; the test fault changes exactly the 56 fused cases of 2–8 tokens (with and without the shared expert) and nothing else |
| the same | final build: switch 1 and 2 against 0 | **byte-identical**, both; the fault at 2 changes exactly the same 56 cases |
| the same, with the IQ2_XXS folder's `run/launch_count.c` preloaded | final build: switch 2 | byte-identical; the PTX kernels launched 56 times (8 `ntm2`, 16 `ntm4`, 32 `ntm8`): every fused case of 2–8 tokens ran the hand-written kernel |
| IQ2_XXS's cases (`TYPE=iq2_xxs`), unchanged by this row | C build: 1; final build: 1 and 2, fault at 2 | byte-identical in every arm, every case's hash the same as in the IQ2_XXS folder; the fault 56 of 56; IQ2_XXS's PTX fatbin is the same bytes as that build's (md5 `8b8d0488…`) |
| `test-backend-ops -o 'MUL_MAT_VEC_FUSION\|MUL_MAT_ID_FUSION'` (the fused gate/up cases), IQ2_XXS's and IQ2_S's cases (`-p` on `type=iq2_s,` and `type=iq2_xxs,`; `run/gpu_checks.sh`'s `ARM_FILTER`), two arms at a time | final build: off, 1, 2 | 239 of 239 pass in each, every verdict the same; at 2 the PTX kernels launched 32 times for each type (IQ2_S 4 `ntm2`, 8 `ntm4`, 20 `ntm8`; IQ2_XXS the same). The fault (IQ2_S's 68 cases) fails 36: the fused gate/up at 2,560 inputs, 2–8 tokens, 256 and 640 rows, with and without skipped slots and biases (biases run the C version), and the PTX kernels launched the same 32 times; the cases left to the old kernel pass (1,024 and 3,072 inputs, 641 rows, 1 and 17 tokens) |
| `test-backend-ops -o 'MUL_MAT_ID_FUSION\|MUL_MAT_ID_SHARED'`, the same cases | final build: off, 1, 2 | 17 of 17 pass in each, every verdict the same; the fault fails all 10 IQ2_S routed + shared cases (2–8 tokens), whose shared expert runs in the fast C kernel |
| `test-backend-ops -o MUL_MAT_ID` (the plain op, llama.cpp's repeated-id cases left out), the same cases | final build: off, 1, 2 | 245 of 245 pass in each, every verdict the same (the plain product has no gate, so the fault cannot show: its 99 IQ2_S cases pass) |
| Why only the IQ2 cases | — | The switch changes the dispatch of IQ2_XXS and IQ2_S only, and every other kernel is the same bytes as in IQ2_XXS's final build (below), whose full sets passed (2,389, 145 and 2,338 cases). The full `MUL_MAT_ID` set at two arms at a time ran 787 cases in 42 minutes on this machine (est. 8 h for the three sets and four arms), so it was stopped at 787 of 2,389 cases in the off arm and at 1, every case passing until then |
| Real file: `gmix-q2` on one card (`run/gpu_checks.sh` step `model`): the IQ2_S layers' routed gate/up on the card (`IQL`, layers 12, 15, 24, 25, 26, 28, 31), every other expert in host memory, the MTP draft, prompts in micro-batches of 8, Flash-Next's v5 preset forced as a P106 runs it; two prompts, each answered with and without the top-8 probabilities, 192 greedy tokens | final build: 1 and 2 against 0 | **every reply the same token ids and text; the top-8 probabilities the same to the last printed digit at all 192 steps of both prompts**; the draft's proposals and acceptances the same counts (150 and 140, 228 and 114, 225 and 115), so the verify passes of 2–4 tokens computed the same logits. With `launch_count.c` preloaded the arm at 2 launched the PTX kernels 143 times (21 `ntm2`, 28 `ntm4`, 94 `ntm8`: verify passes and prompt micro-batches); the arms at 0 and 1 none. One run |
| Pipelined against plain decoding on the card, `bench/pipespec/gpu_pipe_check.sh` (default arms: F32 models on llama.cpp's kernels, their Q8_0 copies and the release types on the fork's, one card and three virtual cards; five architectures, six checks each) | IQ2_XXS's final build `6aa2a8df0` against the final build, switch unset | 125 OK and 5 INFO in each build; all 130 checks give the base's plain-decoding token and logits hashes (and the same as in IQ2_XXS's folder) |
| Every kernel's SASS, `run/sass_all.sh` (`test-backend-ops` plus each build's PTX fatbins) | IQ2_XXS's final build `6aa2a8df0` against the final build | all 9,517 kernel names of the base (10,084 copies, IQ2_XXS's three PTX entry points among them) the same bytes; 9 new: IQ2_S's 6 CUDA C fast kernels and 3 PTX entry points. The 12 CUDA C fast kernels (both types) are the same bytes in the C commit's build and the final one. The hashes leave out the section headers and blank lines between cubins: the kernel that ends a cubin carries the next cubin's header in `cuobjdump`'s text, and which kernel that is moves when a build adds source files |

## Files

| File | What it is |
|---|---|
| [sass_counts.tsv](sass_counts.tsv) | Static SASS counts by kind of the inner loops (one pass, one path), per 32 weights of one matrix: IQ2_S's old multi-token kernel, the fast CUDA C kernel and the PTX kernel, at 1–4 members (`sass_loops.py --tsv`) |
| [checks/bitid_runs.txt](checks/bitid_runs.txt), [checks/bitid_cases.txt](checks/bitid_cases.txt) | The old-against-new runs' verdicts and launch counts; IQ2_S's 104 cases with each output's FNV-1a hash (the same in every arm) |
| [checks/bops_arms.tsv](checks/bops_arms.tsv), [checks/bops_fault_cases.txt](checks/bops_fault_cases.txt) | `test-backend-ops` by op set and arm with the PTX launches; the cases the test fault fails |
| [checks/model_compare.txt](checks/model_compare.txt), [checks/model_launch_count.txt](checks/model_launch_count.txt) | The real file's replies, arm against arm; the PTX launches of the arm at 2 |
| [checks/sass_all.txt](checks/sass_all.txt) | The whole-build SASS comparison and the new kernels |
| [checks/pipe_summary.tsv](checks/pipe_summary.tsv) | `gpu_pipe_check.sh`'s verdicts and plain-decoding hashes, base and branch |
| [run/bitid.sh](run/bitid.sh) | Old against new on one card: `bench/mmvq_pascal/verify/mtok_iq_fast_check` for a type (`TYPE`) with the switch at 0, 1, 2 and with the test fault, outputs compared byte for byte |
| [run/gpu_checks.sh](run/gpu_checks.sh) | `test-backend-ops` by op set with the switch off, at 1 and 2 and with the fault, at most `PAR` arms at a time; the real `gmix-q2` file on one card with the layers `IQL` lists on it, greedy replies with the MTP draft, switch off against on |
| [run/sass_all.sh](run/sass_all.sh) | Every kernel's SASS in the base build's and the branch's `test-backend-ops` and their PTX fatbins, kernel name by kernel name |
| [run/split_sass.py](run/split_sass.py) | Splits `cuobjdump -sass` output into one file per kernel (the input of `sass_loops.py`) |
| [../2026-11-25_c1138_iq2xxs_mtok/run/](../2026-11-25_c1138_iq2xxs_mtok/README.md#files) | The counter `sass_loops.py` and the launch counter `launch_count.c`, used here unchanged |
| [../../../bench/mmvq_pascal/verify/mtok_iq_fast_check.cpp](../../../bench/mmvq_pascal/verify/mtok_iq_fast_check.cpp) | The old-against-new harness (its second argument is the type) and its build script beside it |

## Exact commands

On the build machine, from a checkout of this repository; `W` is a work folder, `LOCK` a build lock file, `REPO` a
folder with `scripts/pascal_forced.sh`'s and `release/flash-next-v5/settings.env`'s copies, `MODELS` the folder with
`gmix-q2`'s shards, `DRAFTS` the one with the MTP draft, `BASE` IQ2_XXS's final build's `bin/` (`6aa2a8df0`) and
`BASE_FATBIN` its `mtok-iq2_xxs.fatbin`, `OLD` the v5 build the branch is cut from (`2d839da1d`); every folder mounted
at the same path in the containers (`MOUNTS`), every container with `--cpu-shares=256` and a label (`DOCKER_ARGS`).
`X` is the IQ2_XXS folder's `run/`.

```bash
# the builds (twice: the C commit 9ca44b706, the final 886724824)
flock "$LOCK" env WORK=$W/rig TESTS=1 JOBS=8 DOCKER_ARGS="--cpu-shares=256 -e HF_UI_VERSION=b12674" \
    bash scripts/build_llamacpp_docker.sh fork/pascal-p106-v5-iq-mtok-fast
T=$W/rig/llama.cpp-20261126-886724824-pascal-p106-v5-iq-mtok-fast
# the harness against the build's tree (before the next build cleans it), the launch counter, the PTX fatbins
bash bench/mmvq_pascal/verify/build_mtok_iq_fast_check.sh $W/rig/src $W/rig/src/build $W/bin
gcc -O2 -shared -fPIC -I/usr/local/cuda/include $X/launch_count.c -o $W/bin/launch_count.so -ldl   # in the CUDA container
cp $W/rig/src/build/ggml/src/ggml-cuda/ptx/*.fatbin $W/fatbin/
# static counts: a kernel's SASS, one file per kernel, then its inner loops (16 blocks of 32 weights a pass: a
# super-block of gate and of up); the old kernel from the v5 build, the C kernel from the C build, the PTX from its fatbin
cuobjdump -sass -arch sm_61 $OLD/test-backend-ops > old.sass
python3 -I docs/data/2026-11-26_c1138_iq2s_mtok/run/split_sass.py old.sass old 'mul_mat_vec_iq_pascal_mtokIL9ggml_type22ELb1ELi4ELi[248]E'
cuobjdump -sass -arch sm_61 $W/rig/llama.cpp-20261126-9ca44b706-pascal-p106-v5-iq-mtok-fast/bin/test-backend-ops > c.sass
python3 -I docs/data/2026-11-26_c1138_iq2s_mtok/run/split_sass.py c.sass c 'mul_mat_vec_iq2_s_pascal_mtok_fastILb1ELi[48]E'
cuobjdump -sass $W/fatbin/mtok-iq2_s.fatbin > ptx.sass
python3 -I docs/data/2026-11-26_c1138_iq2s_mtok/run/split_sass.py ptx.sass ptx 'mtok_iq2s_fast_ptx_ntm[248]'
python3 -I $X/sass_loops.py <kernel>.sass --units 16 [--tsv <name>]
# old against new on GPU 2: IQ2_S, then IQ2_XXS's cases, then IQ2_S with the launch counter
for ty in iq2_s iq2_xxs; do
    env WORK=$W/bitid-$ty CHECK=$W/bin/mtok_iq_fast_check REPO=$REPO GPU=2 ARMS="1 2" FAULT_ARM=2 TYPE=$ty \
        bash docs/data/2026-11-26_c1138_iq2s_mtok/run/bitid.sh
done
env WORK=$W/bitid-lc CHECK=$W/bin/mtok_iq_fast_check REPO=$REPO GPU=2 ARMS=2 FAULT_ARM= TYPE=iq2_s \
    PRELOAD=$W/bin/launch_count.so bash docs/data/2026-11-26_c1138_iq2s_mtok/run/bitid.sh
# every kernel's SASS against IQ2_XXS's final build, the PTX fatbins included
env WORK=$W/sass BASE=$BASE HEAD=$T/bin BASE_EXTRA="$BASE_FATBIN" \
    HEAD_EXTRA="$W/fatbin/mtok-iq2_xxs.fatbin $W/fatbin/mtok-iq2_s.fatbin" bash docs/data/2026-11-26_c1138_iq2s_mtok/run/sass_all.sh
# test-backend-ops, three op sets, IQ2_XXS's and IQ2_S's cases (the fault on IQ2_S's), two arms at a time
i=0
for ops in MUL_MAT_ID 'MUL_MAT_ID_FUSION|MUL_MAT_ID_SHARED' 'MUL_MAT_VEC_FUSION|MUL_MAT_ID_FUSION'; do
    i=$((i + 1))
    env WORK=$W/bops$i BIN=$T/bin REPO=$REPO GPU=2 MODES="1 2" PAR=2 STEPS=bops OPS="$ops" PRELOAD=$W/bin/launch_count.so \
        ARM_FILTER='.*type(_a)?=iq2_(s|xxs),' FAULT_FILTER='.*type(_a)?=iq2_s,' \
        bash docs/data/2026-11-26_c1138_iq2s_mtok/run/gpu_checks.sh
done
# the real file: gmix-q2's IQ2_S layers (12|15|24|25|26|28|31, the default IQL) on the card; MOUNTS must also hold
# the folder its third shard links to
env WORK=$W/model BIN=$T/bin REPO=$REPO GPU=2 MODES="1 2" STEPS=model PRELOAD=$W/bin/launch_count.so \
    MODEL=$MODELS/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-pmix-fast-gmix-q2-00001-of-00003.gguf \
    DRAFT=$DRAFTS/mtp-Swift-1.5-Qwen3.8-Flash-Next-draft.gguf bash docs/data/2026-11-26_c1138_iq2s_mtok/run/gpu_checks.sh
# pipelined against plain decoding on the card, IQ2_XXS's final build against this one
env WORK=$W/pipe BUILDS="base=$BASE head=$T/bin" GPUS=2 MODEL_SEED=1 bash bench/pipespec/gpu_pipe_check.sh
```

The PTX sources are regenerated in the fork with `python3 -I ggml/src/ggml-cuda/ptx/gen_mtok_iq2_s.py >
ggml/src/ggml-cuda/ptx/mtok-iq2_s.ptx` (and `gen_mtok_iq2_xxs.py > mtok-iq2_xxs.ptx`; run from that folder; they read
the grids from `ggml/src/ggml-common.h`); the committed files are the scripts' output byte for byte. `gen_mtok_iq2_s.py`
imports `gen_mtok_iq2_xxs.py`'s shared parts.

## The rig tarball

`llama.cpp-20261126-886724824-pascal-p106-v5-iq-mtok-fast.tar.gz`, 1,662,273,161 bytes, md5
`d405e3c5e5edb1ddae895a3f7f564547` (`llama-server` `f9fe848b2594d66ea322aa3fc9d9537b`, `test-backend-ops`
`60af5bf56f8ab78875df34ed327188b2`). It holds both rewrites (IQ2_XXS's and IQ2_S's) behind the same switch. Its time on
a P106 is the next step: the IQ2_S and IQ2_XXS routed gate/up per layer at four tokens with the switch unset, at 1 and
at 2 (R125: IQ2_S 0.978 ms unset, IQ2_XXS 0.937, Q2_0 0.271), then Flash-Next's step with `gmix-q2`.

## Provenance and licence

The fork's branch `pascal-p106-v5-iq-mtok-fast` of [Miner Llama](https://github.com/Cinderella-Man/miner-llama), on
IQ2_XXS's `6aa2a8df0` (off `pascal-p106-v5` at `2d839da1d`); llama.cpp's MIT licence. The weights in the harness are
random or quantized from random values; the model file is `gmix-q2`, built in this project from UkisAI's Swift-1.5
GSQ-RCO Q2_0 of Qwen3.8-Flash-Next ([its pick](../2026-11-04_fmt_mixes/README.md#the-v31-pick-2026-11-11)). Every
table here is this project's own output, public domain.
