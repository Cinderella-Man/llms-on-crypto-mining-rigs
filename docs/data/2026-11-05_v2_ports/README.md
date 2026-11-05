# The v2-era branches carried onto v3 (2026-11-05)

Every feature Miner Llama ever built is to sit in the cumulative v3 build behind its own switch (registry Q1025,
Q1026). This folder covers the 21 experiment branches of the v2 line (2026-10-22 to 10-26) and the Q4_1 group-64
type, which until today existed only on one laptop: what each holds, whether v3 already has it, and the four ports of
what it lacks onto `pascal-p106-v3-b12076-ladder4` (`e3b30303d`), each off by default. Results: registry rows Q16, Q97,
Q108, Q360 and Q1035–Q1037 in [quants/experiments.md](../../../quants/experiments.md).

## The branches

All 22 are on the fork (github.com/Cinderella-Man/llama.cpp) under their own names. "In `-ladder4`" compares each
commit a branch adds over `miner-llama-v2.0`'s history with `-ladder4` by patch (`git cherry`), then by subject for the
commits the b12076 rebase rewrote; the subject matches were read line by line (they differ only where upstream moved
the code around them).

| v2-era branch | Tip | What it does | In `-ladder4` `e3b30303d` | Result (registry) | Switch on v3 (default) |
|---|---|---|---|---|---|
| `pascal-p106-v2-agentcache` | `cace93ac7` | agent sessions stop re-reading their prompts: `--prefix-cache DIR`, whole sliding-window ring in prefix files and checkpoints, restore right after the checkpoint (Q111, Q132) | yes: 5 of 5 commits (4 by patch, 1 by subject) | Q111: CPU bit-exact; in v2.0, off | `--prefix-cache` (off) |
| `pascal-p106-v2-combined-split` | `e11b88846` | the combined v2 test build (Q117) with PR #31047's MoE tile list for prompt batches only | 25 of 26; the split `511943869` was not | Q117: rig done; the split reversed on 2026-10-24 12:25 (decode the same with the tile list on and off) | ported: `-b12076-prefill-split`, `GGML_CUDA_MMQ_MOE_LEGACY=2` (off) |
| `pascal-p106-v2-drafter` | `e93bac721` | the staircase drafter as a guesser of the pipelined executor, windows from the layer split, the pending-commit flush fix (Q20/D3) | yes: 4 of 4 | Q20: served on the rig (rc4) at 50.4 tok/s with the store | `--pipe-guess staircase`, `--pipe-draft` (none) |
| `pascal-p106-v2-f16router` | `2dc8fee87` | a Pascal F16 matrix-vector kernel for one token (MoE routers kept in F16), on `-kquant-next` | 3 of 4; the kernel `2dc8fee87` was not | Q97b: no timing (its rig microbenchmark never ran) | ported: `-b12076-f16router`, `GGML_CUDA_MMVQ_PASCAL_F16=1` (off) |
| `pascal-p106-v2-f16router-persistent` | `0972b2768` | the same, and the persistent MoE kernel sums an F16 router in that kernel's order | 3 of 5; `2dc8fee87`, `0972b2768` were not | as above | ported with it (one switch for both) |
| `pascal-p106-v2-hybrid` | `b7af2c4e2` | pipelined decoding for the delta-net hybrids, their device-built mask (Q162) | yes: 6 of 6 (3 by patch, 3 by subject; as `-b12076-hybrid`, merged in `-ladder4`) | Q162 → Q550, R29a: depth 3 decodes 1.6× slower, not for the release | `--pipe-depth` (0), `LLAMA_GPU_KQ_MASK` (off) |
| `pascal-p106-v2-kquant-next` | `8ff4431ce` | Q4_K and Q6_K in the Pascal matrix-vector kernels, `GGML_CUDA_MMVQ_PASCAL_SKIP` (Q149) | yes: 3 of 3 | Q149: correct; in v2.0 | on; `GGML_CUDA_MMVQ_PASCAL_SKIP=q4_K,q6_K` off |
| `pascal-p106-v2-next-c7` | `99d61a35e` | pipe chunk slots on streams of their own, `--pipe-chunks`, the continuation store `--pipe-store` (Q161), fed tokens refused for M-RoPE/PLE models (C7) | yes: 7 of 7 | Q161: +1.4 % on the agent check | `LLAMA_PIPE_CHUNK_STREAM` (off), `--pipe-store` (off) |
| `pascal-p106-v2-pipeserve` | `2650855b2` | pipelined speculation in llama-server (`--pipe-depth`, `--pipe-guess`, `--pipe-copy-min`), the KV step `LLAMA_KV_PAD` (Q153), `--pipe-floor` (Q154) | yes: 20 of 20 | Q153: in v2.0; Q154: first data | `--pipe-depth` (0), `--pipe-floor` (0) |
| `pascal-p106-v2-pipeserve-tools` | `96fdd9cb1` | `-pipeserve` with Q67's tool-request sampling | yes: 24 of 24 | Q67: tool requests +13–15 % on the rig | `LLAMA_GRAMMAR_HOST_SAMPLING` (0: on the device) |
| `pascal-p106-v2-pipespec2` | `2dfbe403e` | llama-pipespec streams: packed asynchronous uploads (Q120), device-fed next token (Q121), `--pipe-floor` | yes: 18 of 18 | Q120: +21 % at 8 conversations; Q121: a regression in streams mode | llama-pipespec flags (a tool) |
| `pascal-p106-v2-prefill-attn` | `79613a54e` | the compute-6.1 prompt attention kernel (Q163) | yes: 5 of 5 | Q163: in v2.0, on | `GGML_CUDA_FATTN_TILE_PASCAL` (1: on) |
| `pascal-p106-v2-prefill-split` | `511943869` | PR #31047's tile list for prompt batches only; steps of up to 8 tokens keep the per-expert tiles | 4 of 5; the split `511943869` was not | reversed on 2026-10-24 12:25 (Q117) | ported: `-b12076-prefill-split`, `GGML_CUDA_MMQ_MOE_LEGACY=2` (off) |
| `pascal-p106-v2-q151` | `8d30766c2` | a Pascal decode attention kernel for one token (Q151) | yes: 1 of 1 | Q151: in v2.0, off (slower on the rig) | `GGML_CUDA_FATTN_PASCAL` (0) |
| `pascal-p106-v2-q155` | `99c3a7c00` | llama-pipespec waits for the prompt before the generation clock starts, with the pipeline work below it (Q155) | yes: 28 of 28 | Q155: steady state confirmed | llama-pipespec (a tool) |
| `pascal-p106-v2-q161` | `327656fb5` | the continuation store for pipelined speculation (Q161), fed tokens refused for M-RoPE/PLE models | yes: 31 of 31 | Q161: +1.4 % on the agent check | `--pipe-store` (off) |
| `pascal-p106-v2-rc-agentcache` | `c0b4a995d` | `-agentcache` on the release candidate (Q111) | yes: 5 of 5 | as Q111 | `--prefix-cache` (off) |
| `pascal-p106-v2-rc3` | `9cf56b155` | ggml-alloc plans again when a leaf no node reads needs memory its plan entry lacks (the rc3 fix) | yes: 1 of 1 | in v2.0 (rc4) | none (a fix) |
| `pascal-p106-v2-rescale` | `bdac267bb` | the MoE down rescale off at one token on compute 6.1 | yes: 1 of 1 (by subject) | in v2.0 | `LLAMA_NO_MOE_DOWN_RESCALE` (unset: off at one token on 6.1; `0` keeps it) |
| `pascal-p106-v2-toolsample` | `f11ac1589` | backend sampling kept for tool requests, the host checks the device's token against the grammar (Q67); `LLAMA_GRAMMAR_TEST_REJECT` | yes: 3 of 3 | Q67: tool requests +13–15 % on the rig | `LLAMA_GRAMMAR_HOST_SAMPLING` (0), `LLAMA_GRAMMAR_TEST_REJECT` (0) |
| `pascal-p106-v2-toolsample-screen` | `b6ffd7e6b` | `-toolsample` with the screened output head, kept with a grammar until the grammar can reject a token | 6 of 7; `45d243e80` was not | Q360: planned (est. +1.0–1.6 % per generated token on the agent check) | ported: `-b12076-toolsample-screen`, `LLAMA_SCREEN_HEAD_GRAMMAR=1` (off) |
| `wip/q41g64-2026-10-23` | `43de1acd2` | the Q4_1 group-64 type `Q4_1_64` (Q16) | in v2.0 (patch-identical), retired on the b12076 base (`230506b65`: upstream gave its id 43 to F8_E4M3) | Q16, Q28: the makers' weights exactly, same KLD; below the quality bar, never shipped | ported: `-b12076-q41g64`, type 504 (files decide) |

So 16 branches are in v3 already, through v2.0 and the b12076 rebase, and five commits of six branches were not: they
are the four port branches below. None is superseded by a v3 version of its own.

## The ports

Each is one fork branch on `-ladder4` `e3b30303d`, to be merged into `-ladder5` (Q1026).

| Fork branch | Commits | What it adds | Switch (default) |
|---|---|---|---|
| `pascal-p106-v3-b12076-toolsample-screen` | `87cd55d01` (v2's `45d243e80`), `21c36929c` | the screened output head stays on for a request with a grammar until the grammar can reject a token (a lazy tool grammar before its trigger, the reasoning), and a verify step stops at the row where it becomes active (Q360) | `LLAMA_SCREEN_HEAD_GRAMMAR=1` (off: any grammar turns the screen off, as before) |
| `pascal-p106-v3-b12076-prefill-split` | `3be460d5a` (v2's `511943869`) | MMQ's `MUL_MAT_ID` keeps the per-expert tiles from before PR #31047's tile list for steps of up to 8 tokens, and takes the tile list for prompt batches (Q108's split) | `GGML_CUDA_MMQ_MOE_LEGACY=2` (0: the tile list at every size; 1: the old layout at every size, as before) |
| `pascal-p106-v3-b12076-f16router` | `073353260`, `709e10330` (v2's `2dc8fee87`, `0972b2768`), `9a0e4c8cc` | a Pascal F16 matrix-vector kernel for one token of one F16 matrix (MoE routers kept in F16: a lane group per row, 128-bit loads, FP32 sums), and the persistent MoE kernel's router rows summed in its order so the two paths stay bit-identical (Q97b) | `GGML_CUDA_MMVQ_PASCAL_F16=1` (off: `mul_mat_vec_f`, and the persistent kernel's rows in its order) |
| `pascal-p106-v3-b12076-q41g64` | `59ac8ce59` (reverts `230506b65`) | the Q4_1 group-64 type `Q4_1_64` (Q16): one scale and offset per 64 weights, the CPU and CUDA paths (matrix-vector, dequantize, get_rows), quantize and gguf-py, at type id 504 and file type 500, after N2P/N3P's 500–503 and away from upstream's numbering (upstream gave its old id 43 to F8_E4M3) | none: files decide. Files written with the old ids 43 or 44 do not load as Q4_1_64 (Q1037) |

## Checks (CPU only)

Each branch was built for the rig and checked against `-ladder4`'s own rig build (`run/checks.sh`, [summary.tsv](summary.tsv)):

| Branch | Rig build | Identity (`test-llama-archs -b CPU`) | SASS (sm_61, `-ladder4`'s 8,541 kernels) | Its own check |
|---|---|---|---|---|
| `-toolsample-screen` | OK | 143 of 143 logits comparisons equal | all identical | the screen check below: OK |
| `-prefill-split` | OK | equal | all identical (host code only) | GPU only (Q1035) |
| `-f16router` | OK | equal | 8,529 identical; the 12 that differ are all `k_moe_layer` (the persistent MoE kernel) instantiations, which gained the router branch the switch selects; 1 new, `mul_mat_vec_f16_pascal<2>` ([list](sass_f16router.txt)) | GPU only (Q1035) |
| `-q41g64` | OK | equal | all identical; 20 new, all Q4_1_64's ([list](sass_q41g64.txt)) | `test-quantize-fns`: Q4_1_64's quantization error 0.001147, reference error 0, dot-product error 0.008952, all within the test's limits; 0 tests failed |

Identity: `-ladder4`'s build writes the logits of every architecture's random model, the branch's build compares
(136 files, 143 comparisons); the same 7 of 148 tests fail in `-ladder4`'s run and in each branch's. The SASS check
hashes every kernel's sm_61 instructions in the two builds' `test-backend-ops` (linked statically, so every ggml-cuda
kernel), offsets stripped.

The screen check ([screen_compare.txt](screen_compare.txt)): Flint G3 with its screened head (Q92) on a native CPU build
of `-toolsample-screen`, one BFCL task per category at T 0 and T 1 with backend sampling, three server runs:

| Run | Environment | Against the full head | The screen turned on |
|---|---|---|---|
| screen | `LLAMA_SCREEN_HEAD=1 LLAMA_SCREEN_HEAD_GRAMMAR=1` | 16 of 16 requests identical (8 at T 0, 8 at T 1: reasoning, content, calls) | 14 times |
| default | `LLAMA_SCREEN_HEAD=1` | 16 of 16 identical | never |
| full | `LLAMA_SCREEN_HEAD=0` | — | — |

What the checks do not cover: the ports' CUDA code with the switches on (`test-backend-ops` on a GPU, Q1035), and the
speculative verify path of the screen with a grammar (no draft in the screen check).

## Files

| File | What |
|---|---|
| `commits.tsv` | the commit-level inventory: branch, commit, in v2.0, in `-ladder4`, subject |
| `summary.tsv` | every check's verdict and detail, per branch |
| `sass_<branch>.txt` | the SASS comparison's kernels that differ, are new or are gone, demangled |
| `screen_compare.txt` | `grammar_sampling_check.py compare` for the screen check, and the screen's on/off counts from the server logs |
| `run/checks.sh` | the checks, run detached on the build machine |

## How it was made

The inventory ([commits.tsv](commits.tsv): every commit a branch adds over `miner-llama-v2.0`'s history; `in_v2.0` and
`in_ladder4`: `patch` when `git cherry` finds the same patch, `subj` when only the subject matches, `no`), in a clone
of the fork, git only:

```bash
git log --format=%s 20a51f24a..e3b30303d > ladder4_subjects.txt    # -ladder4's own commits over upstream b12076
git log --format=%s 032b130f4..miner-llama-v2.0 > v2_subjects.txt   # v2.0's line since the oldest branch base
for b in $BRANCHES; do
  git cherry miner-llama-v2.0 "$b" | while read -r s c; do
    sub=$(git log -1 --format=%s "$c")
    l4=$(git cherry e3b30303d "$c" "$c~1" | awk '{print $1}')     # "-": the same patch is in -ladder4
    inl4=no; grep -qxF -- "$sub" ladder4_subjects.txt && inl4=subj; [ "$l4" = - ] && inl4=patch
    intag=no; [ "$s" = - ] && intag=patch
    [ $intag = no ] && grep -qxF -- "$sub" v2_subjects.txt && intag=subj
    printf '%s\t%s\t%s\t%s\t%s\n' "$b" "$(git rev-parse --short=9 "$c")" "$intag" "$inl4" "$sub"
  done
done > commits.tsv
```

The checks ran on the build machine, CPU only, in containers of `nvidia/cuda:12.9.2-devel-ubuntu24.04` at a low CPU
share (`--cpu-shares=256`, `nice -n 19`), one build at a time; the screen step ran beside the later builds, as a second
run with `BRANCHES=toolsample-screen STEPS=screen`:

```bash
ROOT=<work dir> REPO=<this repository> BASE_TREE=<-ladder4's unpacked rig tarball with its extra tests in bin/> \
BASE_SRC=<a llama.cpp clone holding -ladder4> BASE_CCACHE=<-ladder4's ccache> \
MODEL=<flint-q73-G3-screen.gguf: Flint G3 with the screened head of Q92> TASKS=<bench/bfcl_flint/prepare.py's tasks.jsonl> \
JOBS=8 THREADS=8 LOCK=<build lock> DOCKER_ARGS='<labels>' bash docs/data/2026-11-05_v2_ports/run/checks.sh
```

The BFCL tasks are from the Berkeley Function-Calling Leaderboard (Apache-2.0); the model outputs are Flint's
(Apache-2.0). No speeds are kept.
