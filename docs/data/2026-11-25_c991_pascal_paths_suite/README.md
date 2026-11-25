# The fork's other Pascal-only card paths in the card suite, on three newer cards (C991, Q1724, 2026-11-25)

What this is: registry row [Q1724](../../../quants/experiments.md), point 20 of the GPU checks of the night's branches
(registry [Q1687](../../../quants/experiments.md)). Besides the matrix-vector kernels, three of the fork's card paths
turn themselves on only on compute 6.1 cards, the P106: the attention block (`GGML_CUDA_ATTN_BLOCK`), the persistent MoE
kernel (`GGML_CUDA_PERSISTENT_MOE`) and the Pascal flash-attention tile kernel (`GGML_CUDA_FATTN_TILE_PASCAL`). Each
takes `=2` to run on any NVIDIA card. So [C947's card suite](../2026-11-21_c947_v5_merged_gpu/README.md), run on a
build machine's RTX 3090s without them, checked llama.cpp's kernels in their place in its `rest` and fused shares. This
runs those shares again with every one of the fork's compute-6.1-only paths forced on (the matrix-vector kernels
too, as [C968](../2026-11-23_c968_pascal_mmvq_suite/README.md) forced them) and finds which cases reach each path.
Correctness only: nothing here is timed.

## Result

**In plain words:** the fork's Pascal attention kernels and its persistent MoE kernel compute correctly on these
cards: every case that reaches them passes, and a deliberate fault in each one makes exactly those cases fail, so the
suite really checks them. The only failures are three copies into small number formats that round differently by
chance, as they did before with or without the fork's kernels. One path, the attention block, is reached by no case in
the suite.

| Share (card) | Operators | Cases | Passed | Failed |
|---|---:|---:|---:|---|
| `FLASH_ATTN_EXT` (0) | 1 | 4,679 | **4,679** | — |
| the rest of the support listing but `MUL_MAT` and `MUL_MAT_ID` (1) | 124 | 9,707 | **9,704** | 3 `CPY` F32 → Q5_1, Q4_1, Q4_0 at `[2560,4,512,1]` (the random rounding tie) |
| the fused cases test mode adds (2) | 44 | 5,357 | **5,357** | — |
| **all three** | 169 | **19,743** | **19,740** | |

([bops.tsv](bops.tsv), [coverage.tsv](coverage.tsv))

**Which cases reach each path** ([reach.tsv](reach.tsv); kernel launches from an Nsight Systems trace of each live
pass, [kernels.tsv](kernels.tsv)):

| Path (switch) | Cases on it | How it shows | Verdict |
|---|---|---|---|
| Pascal flash-attention tile kernel (`GGML_CUDA_FATTN_TILE_PASCAL`) | **173 of 4,679** `FLASH_ATTN_EXT` cases | 173 launches of `flash_attn_tile_pascal` (head 128 and 256, F16 and Q8_0 cache); its test fault (output × 1.05) fails exactly 173 cases | all 173 pass |
| the same with Ornith-1.5-122B's v5 attention switches added (`GGML_CUDA_FATTN_TILE_PASCAL_MIN=1 GGML_CUDA_FATTN_DEC_PASCAL=4`) | **227 of 4,679**: 212 launches of the tile kernel, 15 of pull request 9's decode kernel `flash_attn_dec_pascal` | the fault fails exactly 227 cases | 4,679 of 4,679 pass |
| persistent MoE kernel (`GGML_CUDA_PERSISTENT_MOE`) | **all 17** `MOE_LAYER` cases, nothing else | 17 launches of `k_moe_layer`; its fault 3 (the shared expert left out) fails 17 of 17 | 17 of 17 pass; its self-check: every layer's output the per-op path's bit for bit |
| attention block (`GGML_CUDA_ATTN_BLOCK`) | **none** | neither of its kernels (`mul_mat_vec_q_pascal_qkv`, `attn_epilogue_f32`) ran in any share | not checked by the suite (below) |
| matrix-vector kernels (`GGML_CUDA_MMVQ_PASCAL`, `GGML_CUDA_MMVF_PASCAL`; C968's) | fused products, and a few products in the rest share | 3,572 launches in the fused share, 15 in the rest | pass |

- **The Pascal tile kernel takes the attention calls a P106 sends it.** The rig build holds only compute 6.1 code, so
  on these cards the attention dispatch has no tensor-core kernel to choose and makes a P106's choices; with the
  switch at `2` the 173 cases are those a P106 runs on this kernel: head 128 or 256, K and V both F16 or both Q8_0, at
  least 9 query tokens with F16 and 3 with Q8_0, a group of 8 or 16 query heads per KV head, a KV length that is a
  multiple of 256, a single-plane F16 mask, no ALiBi and no logit softcap; 85 of them with attention sinks
  ([fa-reach.tsv](fa-reach.tsv), per head size, cache type and token count). The other cases of those head sizes and
  types take llama.cpp's tile or vector kernels on a P106 as here.
- **Ornith's switches add the one- and two-token calls.** With `GGML_CUDA_FATTN_TILE_PASCAL_MIN=1` calls of any token
  count go to the Pascal kernels, and with `GGML_CUDA_FATTN_DEC_PASCAL=4` those with a Q8_0 cache at head 256 and up
  to 4 tokens go to the decode kernel: 54 more cases on the Pascal kernels, all passing. That is the path Ornith-1.5-122B
  Pascal N2P GGUF decodes on; the suite's attention cases had not been run with it on this build.
- **The persistent MoE kernel's self-check** (`GGML_CUDA_PERSISTENT_MOE_CHECK=1`, [moe.tsv](moe.tsv)) compares each
  stage with the per-op path word for word: in all 17 layers the expert choice, the weights, every product and the
  output are the same bit for bit. With a BF16 router (9 of the 17) one router logit differs in its last bits, which
  changes neither the chosen experts nor their weights.
- **The attention block is reached by no suite case.** It runs one token's whole attention block (the Q, K and V
  products of one row, the norms, multiplies and RoPE of Q and K, both cache writes: 16 graph nodes) as two launches,
  and only when a graph holds that whole block. The suite has its parts (`RMS_NORM_MUL_ROPE`, `ROPE_SET_ROWS`, the
  products) but no case that builds all 16 nodes, so only whole-model runs on a P106 check it. A suite case for it is
  registry row Q1834.
- **The three failures are a random rounding tie, not a fork path.** Converting a float to Q4_0, Q4_1, Q5_0, Q5_1 or
  Q8_0 can land exactly on a rounding boundary, where the card and the processor round apart; at the hop cast's shape
  `[2560,4,512,1]` the test's bound is tight enough that one such value fails a case. None of the forced switches reaches
  llama.cpp's copy kernels. Repeated five times with the fork's paths forced and five times with nothing set
  ([cpy.tsv](cpy.tsv), [run/cpy_reps.sh](run/cpy_reps.sh)), every run fails some of the 9 F32 copies at that shape, a
  different set each time: forced 2, 1, 3, 3 and 2 failures, unset 4, 3, 3, 3 and 3. This is
  [C947's](../2026-11-21_c947_v5_merged_gpu/README.md) `CPY` tie and
  [ladder6's](../2026-11-10_ladder6/README.md) Q1229.
- **Every case of the test file ran.** The dry listing (`--record-dry`: every case test mode would run, nothing
  computed) names 27,293 cases; the three shares ran all 19,743 outside `MUL_MAT` (5,161) and `MUL_MAT_ID` (2,389),
  which C968 checked forced. The test file has grown since C947 with upstream b12674 (4,679 attention cases against
  4,659, 9,707 rest against 9,495, 5,357 fused against 5,274), and the fused cases have a 44th operator,
  `ADD_RMS_NORM_MUL` (28 cases) ([coverage.tsv](coverage.tsv), with C947's counts).

**Paths the forced set does not reach**, each off by default on a P106 as well:

- pull request 9's decode kernel, without `GGML_CUDA_FATTN_DEC_PASCAL` (Ornith's preset sets it: the second pass above);
- the older Pascal decode attention kernel (`GGML_CUDA_FATTN_PASCAL`, no preset sets it);
- FP32 scales in the K-quant prompt products (`GGML_CUDA_MMQ_KQ_FP32SCALE`, no preset sets it);
- F32 products of 4 to 8 tokens on `mul_mat_vec_f` (`GGML_CUDA_MMVF_F32_MT`, Flash-Next's preset sets 1): checked with
  `=2` in C968;
- sparse decode attention over a QSA layer's kept cells (`GGML_CUDA_FATTN_SPARSE_PASCAL`, Flash-Next's preset sets 1):
  it needs a QSA model's graph, which no suite case builds.

**A model load on these cards needs one setting more.** When every card is compute 6.1, llama.cpp drops the MoE
down-projection rescale at one token by itself (the Pascal kernels sum in FP32), and the persistent MoE kernel runs a
layer only without that rescale. On a newer card the rescale stays, so a whole model's MoE layers keep the per-op path
even with `GGML_CUDA_PERSISTENT_MOE=2`, unless `LLAMA_NO_MOE_DOWN_RESCALE=tg` is set as well. The card suite builds its
own graphs and does not read it; the forced-set file below holds it for the drivers that load models.

## The forced set as one file

[scripts/pascal_forced.sh](../../../scripts/pascal_forced.sh) holds the set, for every GPU-check driver to source:

| Name | What it holds |
|---|---|
| `PASCAL_FORCED_MMV` | `GGML_CUDA_MMVQ_PASCAL=2 GGML_CUDA_MMVF_PASCAL=2` and a P106's launch shapes (`_NSM=5` for both, `GGML_CUDA_MMVQ_PASCAL_KCOL_NSM=5 GGML_CUDA_MMVQ_PASCAL_KCOL_SMSM=98304`): the set C953 and C968 used |
| `PASCAL_FORCED_PATHS` | `GGML_CUDA_ATTN_BLOCK=2 GGML_CUDA_PERSISTENT_MOE=2 GGML_CUDA_FATTN_TILE_PASCAL=2` |
| `PASCAL_FORCED_GRAPH` | `LLAMA_NO_MOE_DOWN_RESCALE=tg`: the MoE rescale dropped at one token, as llama.cpp does by itself when every card is compute 6.1 (read by model loads only) |
| `PASCAL_FORCED` | all three: the cards run what a P106 runs with no preset switch. This run's card switches are its first two parts |
| `PASCAL_GATES` | every switch whose `1` leaves newer cards out and `2` takes any card, with the four a preset turns on (`GGML_CUDA_MMVF_F32_MT`, `GGML_CUDA_MMQ_KQ_FP32SCALE`, `GGML_CUDA_FATTN_PASCAL`, `GGML_CUDA_FATTN_SPARSE_PASCAL`) |
| `pascal_raise_gates "<list>"` | each gate a preset's list sets to `1`, at `2` (`GGML_CUDA_MMVF_F32_MT=2 GGML_CUDA_FATTN_SPARSE_PASCAL=2` for Flash-Next's) |
| `pascal_force_preset "<list>"` | `PASCAL_FORCED` without the names the list sets, then `pascal_raise_gates`: `env <list> $(pascal_force_preset "<list>")` runs a preset as a P106 runs it, the preset's own values standing except a gate at `1` |

[C953's](../2026-11-23_c953_ornith_kcol_gpu/run/run.sh) and [C968's](../2026-11-23_c968_pascal_mmvq_suite/run/run.sh)
drivers source it. By default they now force `PASCAL_FORCED`; `FORCED=mmv` gives `PASCAL_FORCED_MMV`, the set their
runs used. C968's `fn-f32` setting takes its `GGML_CUDA_MMVF_F32_MT=2` from `pascal_raise_gates`.

## What ran where

| | |
|---|---|
| Build | `pascal-p106-v5` at its tip `eee615cdb` (v5 on upstream b12674); rig tarball md5 `575ef3c702f69f14d7d3022c1287c38f` (`TESTS=1`), `test-backend-ops` md5 `7e6731c4d42a98cfd6a90cf4bdeb0cc5` |
| Cards | three RTX 3090 24 GB on a build machine, running the rig build's compute 6.1 PTX through the driver's JIT, one share per card; a CUDA preflight passed on all three first |
| Setting | `PASCAL_FORCED`'s card switches: `GGML_CUDA_MMVQ_PASCAL=2 GGML_CUDA_MMVF_PASCAL=2 GGML_CUDA_MMVQ_PASCAL_NSM=5 GGML_CUDA_MMVF_PASCAL_NSM=5 GGML_CUDA_MMVQ_PASCAL_KCOL_NSM=5 GGML_CUDA_MMVQ_PASCAL_KCOL_SMSM=98304 GGML_CUDA_ATTN_BLOCK=2 GGML_CUDA_PERSISTENT_MOE=2 GGML_CUDA_FATTN_TILE_PASCAL=2`, every other switch at its default; probes add one switch each ([bops.tsv](bops.tsv), `extra_env`) |
| Shares | C947's split with `FLASH_ATTN_EXT` given a card of its own: the support listing's 127 operators but `MUL_MAT`, `MUL_MAT_ID` and `FLASH_ATTN_EXT` (124); the 44 operators of the dry listing the support listing lacks; every case but the repeated-expert-id ones (`-p "^(?!.*dup=1)"`) |
| Times | listing 11:08–11:09; `FLASH_ATTN_EXT` 11:09–11:14, its fault 11:14–11:18; rest 11:09–11:14; fused 11:09–11:32, the MoE check and fault 11:32–11:36; the copy repeats on card 1 at 11:18; Ornith's switches and their fault 11:38–11:45 on card 0 |

The `not_supported` column of [bops.tsv](bops.tsv) counts cases the test file holds that the CUDA backend does not
support (printed and skipped, as in C947's runs).

## Files

| File | What it holds |
|---|---|
| [bops.tsv](bops.tsv) | every `test-backend-ops` run: operators, extra environment, cases, passed, failed, not supported, exit code, failing cases |
| [coverage.tsv](coverage.tsv) | per operator: share, cases in the dry listing, cases run, passed, failed, C947's cases |
| [reach.tsv](reach.tsv) | per fork path: switch, kernels, launches per share and with Ornith's switches, the probe's count |
| [fa-reach.tsv](fa-reach.tsv) | `FLASH_ATTN_EXT` by head sizes, K/V types and query tokens: cases, passed, on the Pascal kernels with the forced set and with Ornith's switches |
| [moe.tsv](moe.tsv) | each `MOE_LAYER` case live, with the self-check and with the fault; the self-check's line per layer |
| [kernels.tsv](kernels.tsv) | every kernel each live pass ran, with its launch count (no times) |
| [cpy.tsv](cpy.tsv) | the copy repeats: setting, repeat, cases, failures with their error and bound |
| [run/run.sh](run/run.sh) | the driver: the listings, the three shares live and traced, the probes, the pass with Ornith's switches |
| [run/cpy_reps.sh](run/cpy_reps.sh) | the hop-cast copies repeated, forced and unset |
| [run/tables.py](run/tables.py) | the tables from a copy of the driver's work folder (standard library, `python3 -I`) |

## Commands

On the build machine, with the rig tarball unpacked, a copy of this repository's `scripts/pascal_forced.sh` and
`release/ornith-v5/settings.env` in `$REPO`, `run/` copied there, and the three cards free:

```bash
E="WORK=<work folder> NEW=<eee615cdb build>/bin REPO=<repository copy> MOUNTS='<folders holding the build>' \
DOCKER_ARGS='--cpu-shares=256' PREFLIGHT=<CUDA check, run as \$PREFLIGHT -g 0,1,2> \
NSYS=<nsight-systems>/bin/nsys GPUS=0,1,2"
# the listings, the three shares live on cards 0-2, then each share's probes on its card
env $E setsid nohup bash run/run.sh > <work folder>/driver.log 2>&1 < /dev/null &
# once card 1 was free: the hop-cast copies five times forced and unset
WORK=<work folder> NEW=<eee615cdb build>/bin REPO=<repository copy> CARD=1 N=5 MOUNTS='...' \
    setsid nohup bash run/cpy_reps.sh > <work folder>/cpy.log 2>&1 < /dev/null &
# after the first invocation wrote DONE: FLASH_ATTN_EXT with Ornith's attention switches, on card 0
env $E STEPS=ornith setsid nohup bash run/run.sh > <work folder>/driver.log 2>&1 < /dev/null &
# the tables, from a copy of the work folder
python3 -I run/tables.py <copy of the work folder> . ../2026-11-21_c947_v5_merged_gpu/bops-coverage.tsv
```

Every container runs `nvidia/cuda:12.9.2-devel-ubuntu24.04` with the PTX JIT's cache in `<work folder>/jitcache`.
`run.sh` reads `-o` operator lists without parentheses, since `test-backend-ops` reads an `-o` entry holding one as a
whole test case. Both invocations sourced an earlier draft of `scripts/pascal_forced.sh` whose `PASCAL_FORCED` held
the same card switches; its graph setting and the helpers' present rules came after, and `test-backend-ops` reads
neither.
