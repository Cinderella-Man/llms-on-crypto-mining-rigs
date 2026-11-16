# Ornith-1.5-122B on Miner Llama v4

The v4 preset for Ornith-1.5-122B in my Ornith-1.5-122B Pascal N2P GGUF (`valid1`) on nine P106-090 cards, 49,152
tokens of context: the one command that starts it, the settings it adds to v3.0, and one opt-in switch. The file,
where to get it and every v3.0 setting explained are in [release/ornith-v3](../ornith-v3/README.md); the model's page
is [models/ornith-1.5-122b](../../models/ornith-1.5-122b/README.md).

## What v4 changes for Ornith

Prompts. Ornith runs without a draft (its own MTP layer as the draft was a loss on v3.0), so it decodes one token a
step, and v4's kernels for several tokens never run for it: generation stays where v3.0 left it. What v4 brings is a
faster way to read prompts. Prompt calls of 4,096 tokens, prompt micro-batches as CUDA graphs, staged hops between
the cards, and the checkpoints of Ornith's 36 Gated DeltaNet layers kept on the cards and taken inside a call. Each
gives the same greedy replies on nine P106 cards. On a coding agent's twelve tasks this preset reads prompts at 103.0
and 105.9 tok/s against v3.0's 61.1 and 58.4 (+74.8 %), and a task takes 2.11 and 2.13 minutes at equal work against
2.52 and 2.54 (−16.2 %).

An opt-in switch, the micro-batch taper, makes short turns start about three times sooner. It changes Ornith's
numbers, so it is not in the preset ([below](#opt-in-the-micro-batch-taper)).

## Run it

From a clone of this repository, with the v4 engine (the fork's tag `v4-rc2`) built or unpacked beside it as
`llama-p106-v4/`, and the file downloaded as in [release/ornith-v3](../ornith-v3/README.md#the-file):

```bash
MODELS=/mnt/models; . llms-on-crypto-mining-rigs/release/ornith-v4/settings.env && env $PRESET_ENV llama-p106-v4/llama-server -m "$MODELS/$PRESET_MODEL" $PRESET_ARGS --host 127.0.0.1 --port 8080
```

Keep the host quiet while it works. With the checkpoints on the cards its lowest free memory in the agent check was
1,225 and 1,228 MiB, against 731 and 799 MiB on v3.0.

## The preset: [settings.env](settings.env)

v3.0's command line with prompt calls of 4,096 tokens, v3.0's keep-alive (`GGML_CUDA_KEEPALIVE=decode`, 16.3 against
14.9 tok/s), and these:

| Switch | What it does on Ornith | Proved by |
|---|---|---|
| `GGML_CUDA_KEEPALIVE_EAGER=1` | under CUDA 12's lazy loading a kernel's first launch on a card waits until the keep-alive block on that card times out, about 1 s, at the first call of every new graph shape; with it the block is ended first. The same tokens | C811 and C817 (Q1417, Q1426); on nine P106 cards with Flash-Next a 2,760-token prompt took 24,712.63 and 24,744.13 ms against 26,637.95 and 26,065.24 without it (R140) |
| `LLAMA_SERVER_YIELD=2` | a step's hand-over to the server's reader thread wakes it only when a read waits | C741 (Q1310); R122, with Flash-Next |
| `LLAMA_SAMPLING_TOP_K_ROWS=1` | the card's sampling chain (`-bs`, top-k 20) runs one top-k over its output rows and reads the logits without their padded copy; the same tokens | C691 (Q1223), its graph fixed by C802 (Q1406); R83, with Flash-Next |
| `GGML_CUDA_HOP_STAGED=1` | each hop between cards through a pinned host ring | the same greedy replies at every depth checked ([FN93b](../../docs/data/2026-11-16_fn93b_ornith_greedy_rig/README.md)) |
| `GGML_CUDA_GRAPHS_FORCE=1` | prompt micro-batches captured as CUDA graphs without the two-call warm-up that pipeline parallelism never lets a prompt complete | the same (FN93b) |
| `-b 4096` | prompt calls of 4,096 tokens instead of 2,048 | the same (FN93b) |
| `--ctx-checkpoints-vram 4` | a slot's newest four checkpoints of the Gated DeltaNet state on the cards instead of in host memory | the same (FN93b); it fits, at most 5,773 MiB on a card ([R162's fit load](../../docs/data/2026-11-14_r162_ornith_release_deep/README.md)) |
| `--ctx-checkpoints-incall` | checkpoints taken inside a prompt's call instead of cutting the call | the same (FN93b; it needs `--ctx-checkpoints-vram`) |

The first three act in host code, the same for every model. The last five are the prompt stack: on the agent check
together they are a gain, with or without the taper ([below](#the-agent-check-v4-against-v30)).

## Opt-in: the micro-batch taper

`--ubatch-taper short=600:64` reads a prompt call of at most 600 tokens in micro-batches of 64 instead of one of 512.
The nine cards hold Ornith's layers in turn, so a single micro-batch keeps one card busy at a time. Smaller ones flow
through the cards together, the way an assembly line does. Every agent turn ends with such a short call, so short
turns start in about a second instead of three. With the taper the first token's median is 3.05 and 3.27 s against
6.69 and 6.06 s without it, and a task takes 2.4 % less time at equal work.

It is not in the preset because it changes Ornith's numbers. On a 512-token call the 64-token micro-batches move the
next-token distributions by a mean KL divergence of 0.047 (93.9 % the same top token), where a repeat of the same
settings moves them by 0.00006 (100 %). That is close to the distance between this file and the maker's Q8_0 on
agent text, 0.052 (95.0 %). The perplexity is the same, 4.0750 against 4.0730 (+0.05 %), and every task passed with it
in both runs. So it is another rounding of the same arithmetic, of about that size, not a measured loss. Whether it
lies closer to or further from the model is open (registry Q1487), and so is why Ornith moves this far (Q1490)
([data](../../docs/data/2026-11-16_fn93c_ornith_taper_kld/README.md)). To turn it on, add it to the line:

```bash
PRESET_ARGS="$PRESET_ARGS --ubatch-taper short=600:64"
```

## What is left out, and why

| Switches | Why not on Ornith |
|---|---|
| The sparse QSA attention and its tables (`GGML_CUDA_FATTN_SPARSE_PASCAL*`, `LLAMA_QSA_*`) | only Flash-Next's QSA layers have them |
| The MTP draft's switches (`LLAMA_MTP_*`, `LLAMA_DRAFT_*`, `LLAMA_VERIFY_PREPARE`, `LLAMA_SPEC_REJECT_STOP`, `--spec-*`) and the n-gram table's prefetch | Ornith's preset runs no draft and the file has no n-gram table |
| `LLAMA_HOP_CAST=q8_0` | Ornith's graph does not call it (Q1125) |
| The kernels of 2–8-token passes: `GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM`, `_KCOL_TABLE`, `_KCOL_EXACT` (takes Q8_0), `_KCOL_SKIP=q4_0,q5_0` (takes Q5_0), `_Q3K_MT`, `_Q5K_MT`, `_Q6K_MT`, `_DMT`, `_GLU_MT`, `GGML_CUDA_MMVF_F32_MT` (takes the F32 shared-expert gate, one row) | all exact, but they act only on products of 2–8 tokens, which Ornith's line runs at most in a prompt's last micro-batch; the file holds N2P, N3P, Q5_0 and Q8_0 weights and F32 routers, so of their types only Q8_0, Q5_0 and the F32 gate occur |
| `LLAMA_KV_SEQ_RM_BOUND=1` | exact, but without speculation it acts once a request, when the server trims the cache to the reused prefix: nothing to gain |
| `LLAMA_SCHED_PER_SIZE_RESERVE=2` | it acts only when the scheduler re-plans, and no run on Ornith re-planned (0 graph reallocations in every agent run and deep-lane load) |

The prompt kernels for N2P and N3P are still untuned (Q351 in [quants/experiments.md](../../quants/experiments.md));
that is code, not a setting, and is not part of v4.

## The agent check: v4 against v3.0

Measured on 2026-11-14 and 2026-11-16 on nine P106-090 cards with this file, Ornith-1.5-122B Pascal N2P GGUF (valid1)
at 49,152 tokens, 12 bugfix tasks ([bench/agent_check](../../bench/agent_check/README.md), a 12-minute cap a task, the
makers' sampling at temperature 0.6). Four arms, two runs each, alternated, each run its own load: v3.0 (the release
build `1551f5fe1` with [release/ornith-v3](../ornith-v3/settings.env)), `v4-rc2` with the three host switches alone,
this preset, and this preset with the taper ([data](../../docs/data/2026-11-14_r160_ornith_release_agent/README.md),
[this preset's runs](../../docs/data/2026-11-16_r166_ornith_stack_no_taper/README.md)). Every value is one run.

**This preset is a gain against v3.0:** a task takes 16.2 % fewer minutes at equal work (the same work at each run's
speeds), prompts are read 74.8 % faster, and the first token comes sooner. Generation is unchanged.

| Arm, run | Passed | Minutes a task, equal work | Generation tok/s | Prompt tok/s | First token, median s | First token, p90 s | A task's first turn, s | Lowest MemAvailable |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| v3.0, run 1 | 12 / 12 | 2.52 | 16.22 | 61.1 | 7.26 | 19.56 | 16.34 | 731 MiB |
| v3.0, run 2 | 12 / 12 | 2.54 | 16.49 | 58.4 | 8.80 | 19.25 | 16.52 | 799 MiB |
| v4, host switches only, run 1 | 12 / 12 | 2.57 | 16.53 | 56.6 | 8.65 | 19.22 | 16.32 | 803 MiB |
| v4, host switches only, run 2 | 12 / 12 | 2.45 | 16.37 | 65.4 | 9.58 | 20.29 | 16.26 | 718 MiB |
| **this preset, run 1** | **12 / 12** | **2.11** | **16.52** | **103.0** | **6.69** | **11.29** | **8.43** | **1,225 MiB** |
| **this preset, run 2** | **12 / 12** | **2.13** | **16.13** | **105.9** | **6.06** | **11.32** | **8.45** | **1,228 MiB** |
| this preset + the taper, run 1 | 12 / 12 | 2.08 | 16.30 | 114.5 | 3.05 | 10.97 | 8.45 | 1,252 MiB |
| this preset + the taper, run 2 | 12 / 12 | 2.06 | 16.52 | 113.7 | 3.27 | 11.28 | 8.44 | 1,260 MiB |

| Class (means of two runs) | This preset against v3.0 | The taper added, against this preset |
|---|---|---|
| minutes a task at equal work | gain (−16.2 %) | gain (−2.4 %) |
| generation tok/s | negligible (−0.2 %) | negligible (+0.5 %) |
| prompt tok/s | gain (+74.8 %) | gain (+9.2 %) |
| first token median | gain (−20.6 %) | gain (−50.4 %) |
| first token p90 | gain (−41.7 %) | negligible (−1.6 %) |
| a task's first turn | gain (−48.6 %) | negligible (+0.1 %) |

A class compares the means of an arm's two runs: a gain or a loss beyond the larger of the two arms' spreads,
negligible within it. The host switches alone are negligible against v3.0 on every measure (minutes a task −0.8 %,
generation +0.6 %, prompts +2.1 %, first token median +13.5 %, within v3.0's own spread). No run counted a graph
reallocation. The minutes for all twelve tasks (v3.0 35.8 and 26.6, this preset 29.1 and 36.8, with the taper 27.6
and 25.9) are different work in each run and are not compared.

## The deep lane: v4 against v3.0 at depth

Measured on 2026-11-14 on nine P106-090 cards with this file and line, two loads each after a warm-up, 256 tokens
written twice at each depth ([data](../../docs/data/2026-11-14_r162_ornith_release_deep/README.md)). Every value is
one load; classes by the deep lane's rule (a gain or a loss beyond the larger of the two arms' spreads between their
loads, else negligible).

**v4 with the host switches alone is level with v3.0 at depth: it decodes 12.92 and 12.92 tok/s over six depths
from 0 to 48,640 tokens against v3.0's 12.88 and 12.88 (+0.3 %, a gain), and reads prompts at 145.2 and 145.0 tok/s
against 146.8 and 146.7 (−1.1 %, a loss)**, with the same greedy reply as v3.0 at every depth:

| Depth (tokens) | Decode tok/s, v3.0 | v4 | Class | Prompt tok/s, v3.0 | v4 | Class |
|---:|---|---|---|---|---|---|
| 0 | 18.34, 18.32 | 18.38, 18.39 | gain (+0.3 %) | 24.6, 24.3 | 24.6, 24.6 | negligible (+0.5 %) |
| 4,096 | 16.82, 16.80 | 16.87, 16.86 | gain (+0.4 %) | 132.4, 132.2 | 133.8, 133.9 | gain (+1.1 %) |
| 8,192 | 15.46, 15.45 | 15.54, 15.54 | gain (+0.6 %) | 144.8, 144.5 | 143.2, 143.6 | loss (−0.9 %) |
| 16,384 | 13.34, 13.33 | 13.39, 13.38 | gain (+0.4 %) | 193.8, 193.4 | 191.9, 191.7 | loss (−0.9 %) |
| 32,768 | 10.43, 10.42 | 10.46, 10.44 | gain (+0.2 %) | 155.5, 155.5 | 152.3, 152.3 | loss (−2.1 %) |
| 48,640 | 8.61, 8.61 | 8.63, 8.62 | gain (+0.1 %) | 120.7, 120.5 | 119.7, 119.2 | loss (−0.9 %) |
| all | 12.88, 12.88 | 12.92, 12.92 | gain (+0.3 %) | 146.8, 146.7 | 145.2, 145.0 | loss (−1.1 %) |

Prompts at depth 0 are the 64-token tail on an empty cache; at the other depths the chunk read on top of the previous
depth's cache (the row "all" is the five chunks, 54,736 tokens).

**The stack reads prompts 15.1 % faster.** The same run held a third arm, the stack: the host switches plus
this preset's five prompt switches and the taper, `GGML_CUDA_HOP_STAGED=1 GGML_CUDA_GRAPHS_FORCE=1
--ctx-checkpoints-vram 4 --ctx-checkpoints-incall --ubatch-taper short=600:64 -b 4096`. It fits (at most 5,773 MiB on
a card, 1,227 MiB of host memory left at the lowest, in a fit load at 48,640 tokens). Against the host switches alone
it reads prompts faster at every depth: +65.1 % on the 64-token prompt, +26.4 %, +27.1 %, +14.5 %, +10.8 % and +14.3 % at
4,096, 8,192, 16,384, 32,768 and 48,640 tokens, all five chunks at 167.2 and 166.9 tok/s against 145.2 and 145.0
(+15.1 %). Decode is level: 12.94 and 12.93 tok/s against 12.92 and 12.92 over all depths (+0.1 %, a gain), a gain at
0 and 4,096 tokens, negligible at 8,192, 32,768 and 48,640, a loss of 0.1 % at 16,384. With the checkpoints on the
cards the host's lowest MemAvailable is 1,238 and 1,254 MiB against 651 and 655. The stack writes another greedy
reply than the host switches alone at 8,192, 16,384 and 32,768 tokens (the same at 0, 4,096 and 48,640). The taper
alone does that ([data](../../docs/data/2026-11-16_fn93b_ornith_greedy_rig/README.md)); the other five give the same
replies, which is why they are in the preset and the taper is opt-in.

**This preset at depth, with and without the keep-alive** (R167a, [data](../../docs/data/2026-11-16_r167_keepalive_depth/README.md#r167a-ornith)): this file as it is decodes 12.97 and 12.98 tok/s over the six depths and reads all five prompt chunks at 159.9 and 159.8 tok/s (at 48,640 tokens 8.66 and 8.67, and 137.2 and 137.2); without `GGML_CUDA_KEEPALIVE=decode` and `GGML_CUDA_KEEPALIVE_EAGER=1` it reads prompts 2.6 % faster but decodes 3.4 % slower; without the eager stop alone it reads prompts 2.4 % faster and decodes 0.2 % faster, with the same replies (the agent check decides, Q1496).
