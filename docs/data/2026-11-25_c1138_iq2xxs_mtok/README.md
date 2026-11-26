# IQ2_XXS's multi-token expert kernel, rewritten: CUDA C, then hand-written PTX (C1138, Q1871, 2026-11-25)

What this is: the first kernel of code row 28r (registry [Q1871](../../../quants/experiments.md)). R125 found that
Qwen3.8-Flash-Next's better-quality file `gmix-q2` is 18.15 ms a verify step slower than the release file, almost all
of it in the routed experts' gate/up products in the IQ2 grid types, which at four tokens ran 3.5 times slower than
Q2_0 for the same bytes ([R125](../2026-11-25_r125_gmix_step/README.md)). This folder holds the IQ2_XXS kernel's
rewrite (11 of `gmix-q2`'s layers): why the old kernel was slow (static instruction counts), a fast version in CUDA C,
the same kernel in hand-written PTX, and the checks that both give bit for bit the old kernel's results. The method
and what was learned for the next types are in the note [ptx_kernels.md](../../research/notes/ptx_kernels.md).

**No speed was measured here.** The build machine (three RTX 3090s) takes no speed measurements and the P106 rigs were
off; every count below is a static count of compiled instructions (what the compiler emitted for one pass of a loop),
not a time. The kernel's time on a P106 is rig 1's row (QUEUES), with the tarball below.

## Result

**Both rewrites give the old kernel's results bit for bit, and by static count need 41 % (CUDA C) and 45 % (PTX) fewer instructions at one member.** One
pass of the inner loop, per 32 weights of one matrix (Flash-Next's 2,560-input gate/up rows), `run/sass_loops.py`:

| Members of a group | Q2_0, old (`-mtok`) | IQ2_XXS, old | IQ2_XXS fast, CUDA C | IQ2_XXS fast, PTX |
|---:|---:|---:|---:|---:|
| 1 | 42.88 | 131.44 | 76.94 | 72.50 |
| 2 | 55.25 | 148.62 | 89.12 | 84.25 |
| 4 | 79.62 | 183.19 | 112.88 | 108.06 |
| each further member | 12.2 | 17.3 | 12.0 | 11.9 |

At four tokens a group usually has one or two members (512 experts, 10 used a token). By kind, at one member
([sass_counts.tsv](sass_counts.tsv) has every kind and member count):

| Kind | Q2_0 | IQ2_XXS old | fast C | fast PTX |
|---|---:|---:|---:|---:|
| XMAD (16-bit multiplies; three make a 32-bit one) | 0 | 43.12 | 0 | 0.25 |
| logic | 8.00 | 23.00 | 9.38 | 9.19 |
| shifts and bit fields (incl. BFE, BFI) | 6.38 | 19.19 | 29.19 | 21.12 |
| PRMT | 9.00 | 4.12 | 14.00 | 14.00 |
| POPC | 0 | 4.00 | 4.00 | 4.00 |
| integer add, compare, select | 2.75 | 13.75 | 1.25 | 5.06 |
| DP4A | 8.00 | 8.00 | 8.00 | 8.00 |
| float | 1.50 | 1.25 | 4.25 | 4.25 |
| conversions | 1.50 | 1.12 | 0.12 | 0.12 |
| shared loads: 4-byte / 8-byte / 16-byte, shared stores | 2.25 / 0.25 / 1.00 / 0.75 | 2.12 / 4.00 / 2.25 / 2.12 | 4.00 / 0 / 1.12 / 0 | 4.00 / 0 / 1.12 / 0 |
| global loads | 0.75 | 2.12 | 0.62 | 0.62 |
| control, moves | 0.75 | 1.25 | 1.00 | 0.75 |

**Where the old kernel's time went:** 43 XMAD per 32 weights from the 32-bit multiplies that turn each 4 sign bits
into byte masks (`(s4 * 0x00204081) & 0x01010101`, then `* 0xFF`; Pascal builds each 32-bit multiply from three
XMAD) and the 42 logic and shift instructions around them; 4 random 8-byte grid lookups in shared memory (bank
conflicts); the row segments staged through shared memory with 4-byte loads (660-byte rows allow no wider); every
member's activations read twice, once for up and once for gate. **What the rewrite does** (details in
[ptx_kernels.md](../../research/notes/ptx_kernels.md)): gate and up in one pass; each 4 weights' grid value and sign
in one PRMT from a nibble code table (8 copies) and sign bits spread by a second PRMT, no multiplies; the 8 DP4A of a
block one integer sum, turned into `(float) (scale * sum)` by one FFMA (the chain starts at the float 1.5·2^23); row
segments loaded straight into registers, 16 bytes a load. The PTX saves 4.4 more over nvcc's C (`BFE` + `ISCADD` for
the table address where nvcc used `BFE` + `BFI` + a shift; all ten global loads issued at once).

### Checks (LLMRig, one RTX 3090, the fork's Pascal kernels forced)

Every run on GPU 2 of LLMRig (an RTX 3090) with `scripts/pascal_forced.sh`'s settings, which force the fork's Pascal
kernels and a P106's launch shapes on a newer card (otherwise llama.cpp's own kernels run there); the rig build's
compute 6.1 code runs through the driver's PTX compiler, the hand-written kernel through its own PTX in the fatbin.

| Check | Base or arm | Result |
|---|---|---|
| Builds (`TESTS=1`, every check tool, web page b12674) | the C commit `675a62eed`; the final `6aa2a8df0` | rc 0 both; 32 warning lines as the base `2d839da1d`, none new (an anonymous-namespace warning of `bee5c63af` fixed in `6aa2a8df0`) |
| Old against new, `run/bitid.sh` (104 cases: the fused gate/up, the plain product and the fused pair with a shared expert added in the same graph, 1–8 tokens, five routings including repeated and skipped ids; 11,980,800 bytes of output) | C build: switch 1 against 0 | **byte-identical**; the test fault changes exactly the 56 fused cases of 2–8 tokens (with and without the shared expert) and nothing else |
| the same | final build: switch 1 and 2 against 0 | **byte-identical**, both; the fault at 2 changes exactly the same 56 cases |
| the same, with `run/launch_count.c` preloaded | final build: switch 2 | byte-identical; the PTX kernels launched 56 times (8 `ntm2`, 16 `ntm4`, 32 `ntm8`): every fused case of 2–8 tokens ran the hand-written kernel |
| `test-backend-ops -o 'MUL_MAT_ID_FUSION\|MUL_MAT_ID_SHARED'` | final build: off, 1, 2 | 145 of 145 pass in each, every verdict the same; the fault fails the 7 IQ2_XXS routed + shared cases (2–8 tokens), whose shared expert runs in the fast kernel too (switch 2 sends those launches to the C version) |
| Real file: `gmix-q2` on one card (`run/gpu_checks.sh` step `model`): the IQ2_XXS layers' routed gate/up on the card, every other expert in host memory, the MTP draft, prompts in micro-batches of 8, Flash-Next's v5 preset forced as a P106 runs it; two prompts, each answered with and without the top-8 probabilities, 192 greedy tokens; run twice | final build: 1 and 2 against 0 | **every reply the same token ids and text in both runs; the top-8 probabilities the same to the last printed digit at all 192 steps of both prompts**; the draft's proposals and acceptances the same counts (153 and 140, 223 and 116, 217 and 118), so the verify passes of 2–4 tokens computed the same logits. With `run/launch_count.c` preloaded (second run) the arm at 2 launched the PTX kernels 201 times (33 `ntm2`, 22 `ntm4`, 146 `ntm8`: verify passes and prompt micro-batches); the arms at 0 and 1 none |
| `test-backend-ops -o MUL_MAT_ID` (the plain op, llama.cpp's repeated-id cases left out) | final build: off, 1, 2 | 2,389 of 2,389 pass in each, every verdict the same (the plain product has no gate, so the fault cannot show here: its 146 IQ2_XXS cases pass) |
| Pipelined against plain decoding on the card, `bench/pipespec/gpu_pipe_check.sh` (default arms: F32 models on llama.cpp's kernels, their Q8_0 copies and the release types on the fork's, one card and three virtual cards; five architectures, six checks each) | base `2d839da1d` against the final build, switch unset | 125 OK and 5 INFO in each build; all 130 checks give the base's plain-decoding token and logits hashes |
| `test-backend-ops -o 'MUL_MAT_VEC_FUSION\|MUL_MAT_ID_FUSION'` (the fused gate/up cases, among them IQ2_XXS at 2,560 inputs and 256, 640 and 641 rows, 2–8 tokens) | final build: off, 1, 2 | 2,338 of 2,338 pass in each, every verdict the same; at 2 the PTX kernels launched 32 times (4, 8, 20). The fault fails 36 of the 171 IQ2_XXS cases: the fused gate/up at 2,560 inputs, 2–8 tokens, 256 and 640 rows; the others are shapes the fast kernel leaves to the old one (1,024 inputs: 4 lanes a row; 641 rows: the matrix does not end at a 16-byte boundary; 1 and 17 tokens: other kernels) |
| Every kernel's SASS, `run/sass_all.sh` (`test-backend-ops` plus the PTX fatbin) | base `2d839da1d` against the final build | all 9,508 kernel names of the base (10,075 copies) the same bytes; 9 new: the 6 CUDA C fast kernels and the 3 PTX entry points. The 6 CUDA C kernels are the same bytes in the C commit's build and the final one |
| `test-backend-ops -o MUL_MAT_ID` on the C commit's build | off | stopped after 45 minutes (the card's other runs needed the processor): the final build carries the same C kernel bytes, checked there at 1 |

## Files

| File | What it is |
|---|---|
| [sass_counts.tsv](sass_counts.tsv) | Static SASS counts by kind of the inner loops (one pass, one path), per 32 weights of one matrix: Q2_0's and IQ2_XXS's old multi-token kernels, the fast CUDA C kernel and the PTX kernel, at 1–4 members (`run/sass_loops.py --tsv`) |
| [checks/bitid_runs.txt](checks/bitid_runs.txt), [checks/bitid_cases.txt](checks/bitid_cases.txt) | The old-against-new runs' verdicts and launch counts; the 104 cases with each output's FNV-1a hash (the same in every arm) |
| [checks/bops_arms.tsv](checks/bops_arms.tsv), [checks/bops_fault_cases.txt](checks/bops_fault_cases.txt) | `test-backend-ops` by op set and arm; the cases the test fault fails and the PTX launches of those arms |
| [checks/model_compare.txt](checks/model_compare.txt), [checks/model_launch_count.txt](checks/model_launch_count.txt) | The real file's replies, arm against arm (second run); the PTX launches of the arm at 2 |
| [checks/sass_all.txt](checks/sass_all.txt) | The whole-build SASS comparison and the new kernels |
| [checks/pipe_summary.tsv](checks/pipe_summary.tsv) | `gpu_pipe_check.sh`'s verdicts and plain-decoding hashes, base and branch |
| [run/sass_loops.py](run/sass_loops.py) | The counter: inner loops of a kernel's `cuobjdump -sass`, instructions by kind along one path, divided by the 32-weight blocks a pass handles |
| [run/bitid.sh](run/bitid.sh) | Old against new on one card: `bench/mmvq_pascal/verify/mtok_iq_fast_check` with the switch at 0, 1, 2 and with the test fault, outputs compared byte for byte |
| [run/gpu_checks.sh](run/gpu_checks.sh) | `test-backend-ops` by op set with the switch off, at 1 and 2 and with the fault; the real `gmix-q2` file on one card, greedy replies with the MTP draft, switch off against on |
| [run/sass_all.sh](run/sass_all.sh) | Every kernel's SASS in the base build's and the branch's `test-backend-ops` (and a fatbin), kernel name by kernel name |
| [run/launch_count.c](run/launch_count.c) | A preloadable counter of a process's driver-API kernel launches by name (shows the PTX kernels ran) |
| [../../../bench/mmvq_pascal/verify/mtok_iq_fast_check.cpp](../../../bench/mmvq_pascal/verify/mtok_iq_fast_check.cpp) | The old-against-new harness (and its build script beside it) |

## Exact commands

On the build machine, from a checkout of this repository; `W` is a work folder, `LOCK` a build lock file, `REPO` a
folder with `scripts/pascal_forced.sh`'s and `release/flash-next-v5/settings.env`'s copies, `MODELS` the folder with
`gmix-q2`'s shards, `DRAFTS` the one with the MTP draft, `BASE` the base build's `bin/`; every folder mounted at the
same path in the containers (`MOUNTS`), every container with `--cpu-shares=256` and a label (`DOCKER_ARGS`).

```bash
# the builds (three times: the C commit 675a62eed, the PTX commit bee5c63af, the final 6aa2a8df0)
flock "$LOCK" env WORK=$W/rig TESTS=1 JOBS=8 DOCKER_ARGS="--cpu-shares=256 -e HF_UI_VERSION=b12674" \
    bash scripts/build_llamacpp_docker.sh fork/pascal-p106-v5-iq-mtok-fast
T=$W/rig/llama.cpp-20261125-6aa2a8df0-pascal-p106-v5-iq-mtok-fast
# the harness against the build's tree (before the next build cleans it), the launch counter, the PTX fatbin
bash bench/mmvq_pascal/verify/build_mtok_iq_fast_check.sh $W/rig/src $W/rig/src/build $W/bin
gcc -O2 -shared -fPIC -I/usr/local/cuda/include docs/data/2026-11-25_c1138_iq2xxs_mtok/run/launch_count.c \
    -o $W/bin/launch_count.so -ldl   # in the CUDA container
cp $W/rig/src/build/ggml/src/ggml-cuda/ptx/mtok-iq2_xxs.fatbin $W/
# static counts: a kernel's SASS (split at the "Function :" lines), then its inner loops; 16 blocks of 32 weights a
# pass for the IQ2_XXS kernels (a super-block of gate and of up), 8 for Q2_0's (two blocks of both)
cuobjdump -sass -arch sm_61 $T/bin/test-backend-ops > all.sass
cuobjdump -sass $W/mtok-iq2_xxs.fatbin > ptx.sass
python3 -I docs/data/2026-11-25_c1138_iq2xxs_mtok/run/sass_loops.py <kernel>.sass --units 16
# old against new on GPU 2
env WORK=$W/bitid CHECK=$W/bin/mtok_iq_fast_check REPO=$REPO GPU=2 ARMS="1 2" FAULT_ARM=2 \
    bash docs/data/2026-11-25_c1138_iq2xxs_mtok/run/bitid.sh
env WORK=$W/bitid-lc CHECK=$W/bin/mtok_iq_fast_check REPO=$REPO GPU=2 ARMS=2 FAULT_ARM= PRELOAD=$W/bin/launch_count.so \
    bash docs/data/2026-11-25_c1138_iq2xxs_mtok/run/bitid.sh
# test-backend-ops, three op sets
i=0
for ops in MUL_MAT_ID 'MUL_MAT_ID_FUSION|MUL_MAT_ID_SHARED' 'MUL_MAT_VEC_FUSION|MUL_MAT_ID_FUSION'; do
    i=$((i + 1))
    env WORK=$W/bops$i BIN=$T/bin REPO=$REPO GPU=2 MODES="1 2" PAR=1 STEPS=bops OPS="$ops" PRELOAD=$W/bin/launch_count.so \
        bash docs/data/2026-11-25_c1138_iq2xxs_mtok/run/gpu_checks.sh
done
# the real file
env WORK=$W/model BIN=$T/bin REPO=$REPO GPU=2 MODES="1 2" STEPS=model PRELOAD=$W/bin/launch_count.so \
    MODEL=$MODELS/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-pmix-fast-gmix-q2-00001-of-00003.gguf \
    DRAFT=$DRAFTS/mtp-Swift-1.5-Qwen3.8-Flash-Next-draft.gguf bash docs/data/2026-11-25_c1138_iq2xxs_mtok/run/gpu_checks.sh
# every kernel's SASS against the base; the pipe step on the card
env WORK=$W/sass BASE=$BASE HEAD=$T/bin EXTRA_FATBIN=$W/mtok-iq2_xxs.fatbin \
    bash docs/data/2026-11-25_c1138_iq2xxs_mtok/run/sass_all.sh
env WORK=$W/pipe BUILDS="base=$BASE head=$T/bin" GPUS=2 MODEL_SEED=1 bash bench/pipespec/gpu_pipe_check.sh
```

The PTX source is regenerated in the fork with `python3 -I ggml/src/ggml-cuda/ptx/gen_mtok_iq2_xxs.py >
ggml/src/ggml-cuda/ptx/mtok-iq2_xxs.ptx` (run from that folder; it reads the IQ2_XXS grid from `ggml/src/ggml-common.h`);
the committed file is the script's output byte for byte, and the fatbin's SASS is that of `ptxas -arch=sm_61` on it.

## The rig tarball

`llama.cpp-20261125-6aa2a8df0-pascal-p106-v5-iq-mtok-fast.tar.gz`, 1,644,639,278 bytes, md5
`e270460354988f46d72cfc0b35909b34` (`llama-server` `9442fc389b6e7a58b555eefb34120291`, `test-backend-ops`
`f6f765575dc1d98916dd4ed058e9695a`). Its time on a P106 is the next step: the IQ2_XXS routed gate/up per layer at four
tokens with the switch unset, at 1 and at 2 (R125: 0.937 ms unset, Q2_0 0.271), then Flash-Next's step with `gmix-q2`.

## Provenance and licence

The fork's branch `pascal-p106-v5-iq-mtok-fast` of [Miner Llama](https://github.com/Cinderella-Man/miner-llama), off
`pascal-p106-v5` at `2d839da1d`; llama.cpp's MIT licence. The weights in the harness are random or quantized from
random values; the model file is `gmix-q2`, built in this project from UkisAI's Swift-1.5 GSQ-RCO Q2_0 of
Qwen3.8-Flash-Next ([its pick](../2026-11-04_fmt_mixes/README.md#the-v31-pick-2026-11-11)). Every table here is this
project's own output, public domain.
