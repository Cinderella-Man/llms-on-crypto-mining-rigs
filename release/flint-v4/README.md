# Flint on Miner Llama v4

The v4 preset for Flint: the one command that starts it, the settings it adds to v3.0, and the switches still to be
tried on it. The file is my Flint Pascal Q8 GGUF (`p065`), the headline Flint file, on nine P106-090 cards at the full
262,144-token context. Which files to download and every v3.0 setting explained are in
[release/flint-v3](../flint-v3/README.md); the model's page is [models/flint](../../models/flint/README.md).

## What v4 changes for Flint

Little. v4's engine work was done on Flash-Next, whose speed comes from checking three drafted tokens in one pass;
Flint decodes one token a step (its pipelined guesses are one-token steps too), so v4's new kernels for several
tokens never run for it. Flint gets the newer upstream base (b12360) and three small host-side switches that give the
same tokens: a dropped guess frees its cache cells without sweeping all 262,144 of them, the server's reader thread
is woken only when a read waits, and the card's sampler finds the top 20 tokens in one pass with one copy less. On
`v4-rc2` the file loads with the same memory on every card and gives the same greedy replies as before the upstream
merge ([data](../../docs/data/2026-11-13_c831_upstream/README.md)). Whether v4 is faster than v3.0 on Flint is what
the release test below measures; v3.0 on this file and line: 47.4 tok/s, 12 of 12
([data](../../docs/data/2026-11-05_release_regression/README.md)).

## Run it

From a clone of this repository, with the v4 engine (the fork's tag `v4-rc2`) built or unpacked beside it as
`llama-p106-v4/`, and the file downloaded as in [release/flint-v3](../flint-v3/README.md#the-files):

```bash
MODELS=/mnt/models; . llms-on-crypto-mining-rigs/release/flint-v4/settings.env && env $PRESET_ENV llama-p106-v4/llama-server -m "$MODELS/$PRESET_MODEL" $PRESET_ARGS --host 127.0.0.1 --port 8080
```

Unsloth's Q8_0 runs on the same line (`PRESET_MODEL=flint-Q8_0.gguf` after sourcing the preset); Flint Pascal Q5 GGUF
runs it without `-ts` (layers by count), as in v3.0.

## The preset: [settings.env](settings.env)

v3.0's command line, unchanged, and three switches:

| Switch | What it does on Flint | Proved by |
|---|---|---|
| `LLAMA_KV_SEQ_RM_BOUND=1` | when pipelined speculation drops a wrong guess, its cells are freed by visiting only the cells the guess can hold, not every cell of the 262,144-token cache; the same cells and head as without it | C740 (Q1309); on nine P106 cards with Flash-Next, −0.33 ms a step (R122) |
| `LLAMA_SERVER_YIELD=2` | a step's hand-over to the server's reader thread wakes it only when a read waits | C741 (Q1310); R122, with Flash-Next |
| `LLAMA_SAMPLING_TOP_K_ROWS=1` | the card's sampling chain (`-bs`, top-k 20) runs one top-k over its output rows and reads the logits without their padded copy; the same tokens | C691 (Q1223), its graph fixed by C802 (Q1406); R83, with Flash-Next |

Each acts in host code or the sampling graph, the same for every model, and was checked on nine P106 cards.

## What is left out, and why

| Switches | Why not on Flint |
|---|---|
| The sparse QSA attention and its tables (`GGML_CUDA_FATTN_SPARSE_PASCAL*`, `LLAMA_QSA_*`) | only Flash-Next's QSA layers have them |
| The MTP draft's switches (`LLAMA_MTP_*`, `LLAMA_DRAFT_*`, `LLAMA_VERIFY_PREPARE`, `LLAMA_SPEC_REJECT_STOP`, `--spec-*`) and the n-gram table's prefetch | Flint runs no draft model and has no n-gram table |
| `LLAMA_HOP_CAST=q8_0` | Flint's graph does not call it (Q1125) |
| The kernels of 2–8-token passes: `GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM`, `_KCOL_TABLE`, `_KCOL_EXACT` (takes Q8_0), `_KCOL_SKIP=q4_0,q5_0` (takes Q5_0), `_Q3K_MT`, `_Q5K_MT`, `_Q6K_MT`, `_DMT`, `_GLU_MT`, `GGML_CUDA_MMVF_F32_MT` | all exact, but they act only on products of 2–8 tokens, which Flint's line runs at most in a prompt's last micro-batch; the file holds Q8_0 and Q5_0 weights, F16 routers and F32 norms, so of their types only Q8_0 (`_KCOL_EXACT`) and Q5_0 (`_KCOL_SKIP`) occur |
| `-b 4096` | Flint's `-b 512 -ub 128` keeps the 262,144-token context within the 4 GB host: a larger batch pins about 1 GiB of host memory at this context |

## Candidates

Switches that may help Flint but were never run on it. Each is an arm of the release test, the preset plus that one
candidate, never a default until it measures a gain or negligible there:

| Candidate | What it could buy | Why it is not in the preset |
|---|---|---|
| `GGML_CUDA_KEEPALIVE=decode GGML_CUDA_KEEPALIVE_EAGER=1` | the cards held at their boost clock while they decode, without the ~1 s wait at a kernel's first launch: Ornith gained 16.3 against 14.9 tok/s from the keep-alive on v3.0 | on Flint the keep-alive ran only together with the re-read fix, 44.7 against 45.9 tok/s, one run each, within v2.0's own two runs (45.9 and 44.6); never alone, never with the eager stop |
| `GGML_CUDA_HOP_STAGED=1` | each hop between cards through a pinned host ring; exact on Flint's F32 hops; part of Flash-Next's v3.0 preset | not run on Flint in v3.0 |
| `GGML_CUDA_GRAPHS_FORCE=1` | prompt micro-batches captured as CUDA graphs without the two-call warm-up that pipeline parallelism never lets a prompt complete | proved on Flash-Next only together with two draft switches (R123a); graphs take card memory, and this file fills two cards to 5,923 MiB |
| `--ctx-checkpoints-vram 4 --ctx-checkpoints-incall` | Flint keeps most of its attention in a 513-token window, so the server takes context checkpoints; on the cards and inside a prompt's call they took 18.99 % off Flash-Next's prompt time (R85) | the checkpoints take card memory beside the 5,923 MiB two cards already hold; the fit is checked first |
| `--ubatch-taper short=600:64` | a prompt of at most 600 tokens in micro-batches of 64 (Flash-Next, R60) | Flint already reads prompts in micro-batches of 128 |
| `LLAMA_SCHED_PER_SIZE_RESERVE=2` | after a re-plan of the shared scheduler, the plan for the largest batch over the whole memory again | it acts only when the scheduler re-plans: an arm only if the release test's report counts graph reallocations above 0 |

## The release test

On one rig, with this file, both arms on the same line:

- **The agent check** ([bench/agent_check](../../bench/agent_check/README.md), 12 bugfix tasks, a 12-minute cap a
  task): `v3` (the v3.0 release build `1551f5fe1`, tag `miner-llama-v3.0`, with [release/flint-v3](../flint-v3/settings.env)'s
  switches) against `v4` (`v4-rc2` with this preset), two runs each, alternated, each run its own load.
- **The deep lane** ([bench/deep_lane](../../bench/deep_lane/README.md)): the same two arms, two loads each after a
  warm-up load, prompts of 0, 4,096, 8,192, 16,384, 32,768 and 49,152 tokens, 256 tokens written at each, with the
  sampler pass.
- Then the candidates above, each as an arm against `v4` in the agent check, two runs each.

## The agent check: v4 against v3.0

Measured on 2026-11-14 on nine P106-090 cards with this file and line, Flint Pascal Q8 GGUF at 262,144 tokens, 12
bugfix tasks, two runs each, alternated, each run its own load
([data](../../docs/data/2026-11-14_r159_flint_release_agent/README.md)). Every value is one run; generation and
prompts at the agent check's report rounding.

**v4 is negligible against v3.0**: every speed of its two runs lies within the spread of v3.0's two runs.

| | v4, run 1 | v4, run 2 | v3.0, run 1 | v3.0, run 2 | Class |
|---|---|---|---|---|---|
| passed | 12 / 12 | 12 / 12 | 11 / 12 | 12 / 12 | |
| generation tok/s | 47.4 | 48.2 | 48.5 | 48.0 | negligible (−0.9 %) |
| prompt tok/s | 177 | 179 | 179 | 181 | negligible (−1.1 %) |
| first token median s | 2.85 | 2.56 | 2.65 | 2.61 | negligible (+2.9 %) |
| first token p90 s | 7.84 | 6.76 | 7.20 | 7.97 | negligible (−3.8 %) |
| minutes a task at equal work | 1.85 | not tabled | 1.81 | not tabled | |
| minutes, all 12 tasks | 31.3 | 29.2 | 24.7 | 19.7 | different work |
| graph reallocations | 323 in 71 of 73 requests | 308 in 79 of 80 | 188 in 64 of 65 | 237 in 66 of 68 | |
| lowest MemAvailable | 885 MiB | 873 MiB | 948 MiB | 947 MiB | |

v4's runs took longer in all because their attempts wrote and read more (decode 120 and 112 s a task against 49 and
73, 73 and 80 turns against 66 and 68), not because the engine was slower; registry row Q1488 checks whether a
switch changes what Flint samples. v3.0's first run stopped one task at the 12-minute cap.

## The deep lane: v4 against v3.0 at depth

Measured on 2026-11-14 on nine P106-090 cards with this file and line, two loads each after a warm-up, 256 tokens
written twice at each depth ([data](../../docs/data/2026-11-14_r161_flint_release_deep/README.md)). Every value is
one load; classes by the deep lane's rule (a gain or a loss beyond the larger of the two arms' spreads between their
loads, else negligible).

**v4 decodes 40.25 and 40.36 tok/s over six depths from 0 to 49,152 tokens against v3.0's 40.02 and 40.04 (+0.7 %,
a gain), and reads prompts at 318.9 and 321.3 tok/s against 309.9 and 311.0 (+3.1 %, a gain)**, with the same greedy
reply as v3.0 at every depth:

| Depth (tokens) | Decode tok/s, v3.0 | v4 | Class | Prompt tok/s, v3.0 | v4 | Class |
|---:|---|---|---|---|---|---|
| 0 | 49.94, 49.96 | 49.78, 50.79 | negligible (+0.7 %) | 85.1, 85.2 | 84.5, 84.0 | loss (−1.1 %) |
| 4,096 | 76.51, 76.54 | 77.31, 77.33 | gain (+1.0 %) | 477.5, 478.2 | 503.3, 515.7 | gain (+6.6 %) |
| 8,192 | 48.37, 48.37 | 48.77, 48.62 | gain (+0.7 %) | 442.6, 446.7 | 443.6, 476.3 | negligible (+3.4 %) |
| 16,384 | 66.56, 66.74 | 66.91, 67.20 | gain (+0.6 %) | 472.6, 475.4 | 488.5, 490.8 | gain (+3.3 %) |
| 32,768 | 26.18, 26.16 | 26.35, 26.36 | gain (+0.7 %) | 321.9, 322.3 | 332.7, 332.2 | gain (+3.2 %) |
| 49,152 | 23.29, 23.31 | 23.44, 23.44 | gain (+0.6 %) | 212.7, 213.6 | 218.0, 219.2 | gain (+2.6 %) |
| all | 40.02, 40.04 | 40.25, 40.36 | gain (+0.7 %) | 309.9, 311.0 | 318.9, 321.3 | gain (+3.1 %) |

Prompts at depth 0 are the 64-token tail on an empty cache; at the other depths the chunk read on top of the previous
depth's cache (the row "all" is the five chunks, 53,200 tokens).

**The candidates stacked lose at depth.** The same run held a third arm, this preset plus every candidate above but
`LLAMA_SCHED_PER_SIZE_RESERVE=2` (no load re-planned): `GGML_CUDA_KEEPALIVE=decode GGML_CUDA_KEEPALIVE_EAGER=1
GGML_CUDA_HOP_STAGED=1 GGML_CUDA_GRAPHS_FORCE=1 --ctx-checkpoints-vram 4 --ctx-checkpoints-incall
--ubatch-taper short=600:64`. It fits (at most 5,971 MiB on a card, 1,099 MiB of host memory left at the lowest) and
writes the same replies. Against this preset it gains at depth 0 (decode +3.0 %, the 64-token prompt +31.3 %) and on
the 4,096-token prompt (+6.1 %), and loses from 4,096 tokens on: decode −1.2 %, −1.1 %, −1.4 %, −2.0 % and −3.1 % at
4,096, 8,192, 16,384, 32,768 and 49,152 tokens, prompts −5.2 %, −20.6 % and −17.3 % at 16,384, 32,768 and 49,152.
Over all depths it decodes 39.67 and 39.79 tok/s against 40.25 and 40.36 (−1.4 %) and reads 271.7 and 273.5 tok/s
against 318.9 and 321.3 (−14.8 %). The stack is split to find the candidate that costs it. On Flint the server turns
`--ctx-checkpoints-incall` off at load (`not with sliding-window layers`), so of that candidate only
`--ctx-checkpoints-vram 4` acts.
