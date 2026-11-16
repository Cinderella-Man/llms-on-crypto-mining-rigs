# Flint on Miner Llama v4

The v4 preset for Flint: the one command that starts it and the settings it adds to v3.0. The file is my Flint Pascal
Q8 GGUF (`p065`), the headline Flint file, on nine P106-090 cards at the full 262,144-token context. Which files to
download and every v3.0 setting explained are in [release/flint-v3](../flint-v3/README.md); the model's page is
[models/flint](../../models/flint/README.md).

## What v4 changes for Flint

Prompts. Flint decodes one token a step (its pipelined guesses are one-token steps too), so v4's kernels for several
tokens never run for it, and generation stays where v3.0 left it. What v4 brings is a faster way to read prompts:

- prompt micro-batches captured as CUDA graphs;
- the scheduler's plan kept for the largest batch after a re-plan;
- the context checkpoints of Flint's sliding-window layers kept on the cards, so a restore takes 2 ms instead of 292;
- short prompt calls read as a pipeline across the nine cards (the prompt taper).

Each gives the same tokens, and on Flint the taper gives the same numbers to the digit. On a coding agent's twelve
tasks this preset reads prompts at 219.7 and 204.1 tok/s against v3.0's 179.2 and 181.3 (+17.6 %), and the first token
of a turn comes at a median of 1.74 and 1.82 s against 2.65 and 2.61 s (−32.3 %).

## Run it

From a clone of this repository, with the v4 engine (the fork's tag `v4-rc2`) built or unpacked beside it as
`llama-p106-v4/`, and the file downloaded as in [release/flint-v3](../flint-v3/README.md#the-files):

```bash
MODELS=/mnt/models; . llms-on-crypto-mining-rigs/release/flint-v4/settings.env && env $PRESET_ENV llama-p106-v4/llama-server -m "$MODELS/$PRESET_MODEL" $PRESET_ARGS --host 127.0.0.1 --port 8080
```

Unsloth's Q8_0 runs on the same line (`PRESET_MODEL=flint-Q8_0.gguf` after sourcing the preset); Flint Pascal Q5 GGUF
runs it without `-ts` (layers by count), as in v3.0. With the checkpoints on the cards two cards peak at 5,985 MiB of
6,144, and the host's lowest free memory in the agent check was 1,084 and 1,100 MiB.

## The preset: [settings.env](settings.env)

v3.0's command line and these:

| Switch | What it does on Flint | Proved by |
|---|---|---|
| `LLAMA_KV_SEQ_RM_BOUND=1` | when pipelined speculation drops a wrong guess, its cells are freed by visiting only the cells the guess can hold, not every cell of the 262,144-token cache; the same cells and head as without it | C740 (Q1309); on nine P106 cards with Flash-Next, −0.33 ms a step (R122) |
| `LLAMA_SERVER_YIELD=2` | a step's hand-over to the server's reader thread wakes it only when a read waits | C741 (Q1310); R122, with Flash-Next |
| `LLAMA_SAMPLING_TOP_K_ROWS=1` | the card's sampling chain (`-bs`, top-k 20) runs one top-k over its output rows and reads the logits without their padded copy; the same tokens | C691 (Q1223), its graph fixed by C802 (Q1406); R83, with Flash-Next; on Flint the same sampled tokens as v3.0 with the same seed ([C843](../../docs/data/2026-11-16_c843_flint_sampling/README.md)) |
| `GGML_CUDA_GRAPHS_FORCE=1` | prompt micro-batches captured as CUDA graphs without the two-call warm-up that pipeline parallelism never lets a prompt complete | C702; at depth slower at no depth, the same replies ([R165a](../../docs/data/2026-11-16_r165_flint_stack_split/README.md)) |
| `LLAMA_SCHED_PER_SIZE_RESERVE=2` | after a re-plan of the shared scheduler, the plan for the largest batch over the whole memory again; Flint re-plans in agent use (hundreds of graph reallocations a run) | C818 (Q1447); R143 |
| `--ctx-checkpoints-vram 4` | a slot's newest four checkpoints of the sliding-window state on the cards instead of in host memory: a restore takes 2.0 to 2.2 ms instead of 292 ms, and the host keeps about 200 MiB more free | Q440, Q548; at depth slower at no depth, the same replies ([R165b](../../docs/data/2026-11-16_r165_flint_stack_split/README.md)) |
| `--ubatch-taper short=600:64` | a prompt call of at most 600 tokens read in micro-batches of 64 instead of one of 128, which the nine cards work on as a pipeline; every agent turn ends with such a call | Q771; on Flint the same numbers to the digit: KL divergence 0, perplexity 0.000 %, a repeat exact ([FN96](../../docs/data/2026-11-16_fn96_flint_taper_kld/README.md)); at depth slower at no depth (R165b) |

## What is left out, and why

| Switches | Why not on Flint |
|---|---|
| `GGML_CUDA_KEEPALIVE=decode GGML_CUDA_KEEPALIVE_EAGER=1` | at depth a loss against the host switches alone: prompts −16.5 % over all five chunks (−20.1 % at 32,768 tokens), decode −3.0 % ([R165b](../../docs/data/2026-11-16_r165_flint_stack_split/README.md)) |
| `GGML_CUDA_HOP_STAGED=1` | decode +1.5 % at depth, but a loss on the 8,192-token prompt chunk (−7.7 %, [R165a](../../docs/data/2026-11-16_r165_flint_stack_split/README.md)); left out |
| `--ctx-checkpoints-incall` | the server turns it off on Flint (sliding-window layers) |
| The sparse QSA attention and its tables (`GGML_CUDA_FATTN_SPARSE_PASCAL*`, `LLAMA_QSA_*`) | only Flash-Next's QSA layers have them |
| The MTP draft's switches (`LLAMA_MTP_*`, `LLAMA_DRAFT_*`, `LLAMA_VERIFY_PREPARE`, `LLAMA_SPEC_REJECT_STOP`, `--spec-*`) and the n-gram table's prefetch | Flint runs no draft model and has no n-gram table |
| `LLAMA_HOP_CAST=q8_0` | Flint's graph does not call it (Q1125) |
| The kernels of 2–8-token passes: `GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM`, `_KCOL_TABLE`, `_KCOL_EXACT` (takes Q8_0), `_KCOL_SKIP=q4_0,q5_0` (takes Q5_0), `_Q3K_MT`, `_Q5K_MT`, `_Q6K_MT`, `_DMT`, `_GLU_MT`, `GGML_CUDA_MMVF_F32_MT` | all exact, but they act only on products of 2–8 tokens, which Flint's line runs at most in a prompt's last micro-batch; the file holds Q8_0 and Q5_0 weights, F16 routers and F32 norms, so of their types only Q8_0 (`_KCOL_EXACT`) and Q5_0 (`_KCOL_SKIP`) occur |
| `-b 4096` | Flint's `-b 512 -ub 128` keeps the 262,144-token context within the 4 GB host: a larger batch pins about 1 GiB of host memory at this context |

## The agent check: v4 against v3.0

Measured on 2026-11-14 and 2026-11-16 on nine P106-090 cards with this file, Flint Pascal Q8 GGUF at 262,144 tokens,
12 bugfix tasks ([bench/agent_check](../../bench/agent_check/README.md), a 12-minute cap a task). Three arms, two runs
each, each run its own load: v3.0 (the release build `1551f5fe1` with [release/flint-v3](../flint-v3/settings.env)),
`v4-rc2` with the three host switches alone, and this preset
([data](../../docs/data/2026-11-14_r159_flint_release_agent/README.md)). Every value is one run.

**This preset reads prompts 17.6 % faster than v3.0 and brings the first token 32.3 % sooner;** a task's minutes at
equal work and generation are level.

| Arm, run | Passed | Minutes a task, equal work | Generation tok/s | Prompt tok/s | First token, median s | First token, p90 s | A task's first turn, s | Graph reallocations | Lowest MemAvailable |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| v3.0, run 1 | 11 / 12 | 1.81 | 48.55 | 179.2 | 2.65 | 7.20 | 4.17 | 188 in 64 of 65 requests | 948 MiB |
| v3.0, run 2 | 12 / 12 | 1.82 | 48.02 | 181.3 | 2.61 | 7.97 | 4.18 | 237 in 66 of 68 | 947 MiB |
| v4, host switches only, run 1 | 12 / 12 | 1.85 | 47.45 | 176.9 | 2.85 | 7.84 | 4.29 | 323 in 71 of 73 | 885 MiB |
| v4, host switches only, run 2 | 12 / 12 | 1.82 | 48.19 | 178.9 | 2.56 | 6.76 | 4.16 | 308 in 79 of 80 | 873 MiB |
| **this preset, run 1** | **11 / 12** | **1.72** | **49.32** | **219.7** | **1.74** | **6.61** | **3.24** | 250 in 71 of 72 | **1,084 MiB** |
| **this preset, run 2** | **12 / 12** | **1.81** | **46.99** | **204.1** | **1.82** | **6.35** | **3.18** | 290 in 70 of 71 | **1,100 MiB** |

| Class (means of two runs) | This preset against v3.0 | This preset against the host switches alone |
|---|---|---|
| minutes a task at equal work | negligible (−2.8 %) | negligible (−3.8 %) |
| generation tok/s | negligible (−0.3 %) | negligible (+0.7 %) |
| prompt tok/s | gain (+17.6 %) | gain (+19.1 %) |
| first token median | gain (−32.3 %) | gain (−34.2 %) |
| first token p90 | gain (−14.6 %) | negligible (−11.2 %) |
| a task's first turn | gain (−23.1 %) | gain (−24.0 %) |

A class compares the means of an arm's two runs: a gain or a loss beyond the larger of the two arms' spreads,
negligible within it. The prompt gain sits in the short calls the taper reads as a pipeline: prompts under 512 tokens
at 172.7 and 183.8 tok/s against 123.5 and 121.5 with the host switches alone and 124.5 and 107.9 on v3.0
([prompt_by_size.tsv](../../docs/data/2026-11-14_r159_flint_release_agent/prompt_by_size.tsv)).

The run that missed a task stopped it at the 12-minute cap, Python `scale-generator` both times (v3.0's first run and
this preset's first): one reply was still being written after more than 650 s. The agent sends no seed, so each run
draws its own reply lengths, and with the same seed v4 writes v3.0's tokens exactly (26 replies to recorded agent
requests, 16,290 tokens each; [data](../../docs/data/2026-11-16_c843_flint_sampling/README.md)). The same is why the
minutes for all twelve tasks (v3.0 24.7 and 19.7, the host switches 31.3 and 29.2, this preset 28.8 and 23.3) are
different work in each run and are not compared.

## The deep lane: v4 against v3.0 at depth

Measured on 2026-11-16 on nine P106-090 cards with this file and this preset as it is, two loads each after a
warm-up, 256 tokens written twice at each depth
([data](../../docs/data/2026-11-16_r168_flint_f4t_deep/README.md)). Every value is one load; classes by the deep
lane's rule (a gain or a loss beyond the larger of the two arms' spreads between their loads, else negligible).

**This preset decodes 40.58 and 40.57 tok/s over six depths from 0 to 49,152 tokens against v3.0's 40.24 and 40.00
(+1.1 %, a gain), and reads prompts at 329.9 and 334.1 tok/s against 312.7 and 312.0 (+6.3 %, a gain)**, with the
same greedy reply as v3.0 at every depth:

| Depth (tokens) | Decode tok/s, v3.0 | v4 | Class | Prompt tok/s, v3.0 | v4 | Class |
|---:|---|---|---|---|---|---|
| 0 | 50.68, 48.26 | 50.75, 50.34 | negligible (+2.2 %) | 86.4, 83.8 | 91.1, 88.3 | gain (+5.4 %) |
| 4,096 | 76.66, 76.63 | 77.67, 77.61 | gain (+1.3 %) | 479.7, 479.9 | 549.3, 566.8 | gain (+16.3 %) |
| 8,192 | 48.44, 48.58 | 49.11, 49.43 | gain (+1.6 %) | 450.0, 432.1 | 524.7, 540.5 | gain (+20.8 %) |
| 16,384 | 66.88, 66.96 | 67.61, 67.61 | gain (+1.0 %) | 477.2, 475.7 | 520.4, 539.0 | gain (+11.2 %) |
| 32,768 | 26.28, 26.28 | 26.52, 26.51 | gain (+0.9 %) | 327.6, 326.0 | 338.6, 342.2 | gain (+4.2 %) |
| 49,152 | 23.43, 23.43 | 23.57, 23.58 | gain (+0.6 %) | 213.3, 214.1 | 221.2, 222.0 | gain (+3.7 %) |
| all | 40.24, 40.00 | 40.58, 40.57 | gain (+1.1 %) | 312.7, 312.0 | 329.9, 334.1 | gain (+6.3 %) |

Prompts at depth 0 are the 64-token tail on an empty cache; at the other depths the chunk read on top of the previous
depth's cache (the row "all" is the five chunks, 53,200 tokens). In each first load the four
52.5 MiB checkpoints are restored from the cards in 2.1, 2.1, 2.2 and 2.1 ms, where v3.0 reads them from host memory
in 291.9, 292.0, 294.8 and 291.9 ms, and the host keeps about 190 MiB more free at its lowest (1,149 and 1,130 MiB
against 940 and 952). Cards 1 and 4 peak at 5,987 MiB of 6,144 (v3.0: 5,953). No graph
reallocated in either arm.

**The earlier measurement, the three host switches alone.** On 2026-11-14, before this preset took its prompt
switches, the same lane timed `v4-rc2` with v3.0's line and only the three host switches against v3.0
([data](../../docs/data/2026-11-14_r161_flint_release_deep/README.md)). It decoded 40.25 and 40.36 tok/s over six
depths against v3.0's 40.02 and 40.04 (+0.7 %, a gain) and read prompts at 318.9 and 321.3 tok/s against 309.9 and
311.0 (+3.1 %, a gain), with the same greedy reply as v3.0 at every depth:

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

**The candidates stacked lose at depth.** The same run held a third arm, the three host switches plus every candidate then untried but
`LLAMA_SCHED_PER_SIZE_RESERVE=2` (no load re-planned): `GGML_CUDA_KEEPALIVE=decode GGML_CUDA_KEEPALIVE_EAGER=1
GGML_CUDA_HOP_STAGED=1 GGML_CUDA_GRAPHS_FORCE=1 --ctx-checkpoints-vram 4 --ctx-checkpoints-incall
--ubatch-taper short=600:64`. It fits (at most 5,971 MiB on a card, 1,099 MiB of host memory left at the lowest) and
writes the same replies. Against the host switches alone it gains at depth 0 (decode +3.0 %, the 64-token prompt +31.3 %) and on
the 4,096-token prompt (+6.1 %), and loses from 4,096 tokens on: decode −1.2 %, −1.1 %, −1.4 %, −2.0 % and −3.1 % at
4,096, 8,192, 16,384, 32,768 and 49,152 tokens, prompts −5.2 %, −20.6 % and −17.3 % at 16,384, 32,768 and 49,152.
Over all depths it decodes 39.67 and 39.79 tok/s against 40.25 and 40.36 (−1.4 %) and reads 271.7 and 273.5 tok/s
against 318.9 and 321.3 (−14.8 %). The stack is split to find the candidate that costs it. Forced graphs and staged
hops, each alone against the host switches, are not it
([data](../../docs/data/2026-11-16_r165_flint_stack_split/README.md)). At 32,768 and 49,152 tokens both read prompts
within the host switches' spread, and both decode as fast or faster. **The keep-alive is the stack's deep cost**
(`GGML_CUDA_KEEPALIVE=decode GGML_CUDA_KEEPALIVE_EAGER=1`, the same data). Alone against the host switches it reads the
32,768 and 49,152 chunks −20.1 % and −16.8 % slower, the stack's −20.6 % and −17.3 %. It decodes −3.0 % over all
depths and gains only on the 64-token prompt at depth 0. The checkpoints on the cards and the taper, each alone, are
slower at no depth. On Flint the server turns
`--ctx-checkpoints-incall` off at load (`not with sliding-window layers`), so of that candidate only
`--ctx-checkpoints-vram 4` acts.
