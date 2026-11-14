# Ornith-1.5-122B on Miner Llama v4

The v4 preset for Ornith-1.5-122B in my Ornith-1.5-122B Pascal N2P GGUF (`valid1`) on nine P106-090 cards, 49,152
tokens of context: the one command that starts it, the settings it adds to v3.0, and the switches still to be tried
on it. The file, where to get it and every v3.0 setting explained are in [release/ornith-v3](../ornith-v3/README.md);
the model's page is [models/ornith-1.5-122b](../../models/ornith-1.5-122b/README.md).

## What v4 changes for Ornith

Little. v4's engine work was done on Flash-Next, whose speed comes from checking three drafted tokens in one pass;
Ornith runs without a draft (its own MTP layer as the draft was a loss on v3.0), so it decodes one token a step and
v4's new kernels for several tokens never run for it. Ornith gets the newer upstream base (b12360) and three small
host-side switches beside v3.0's keep-alive, all giving the same tokens: the keep-alive now stops before a kernel's
first launch on a card, so that launch no longer waits about a second for it, the server's reader thread is woken
only when a read waits, and the card's sampler finds the top 20 tokens in one pass with one copy less. On `v4-rc2` the
file loads with the same memory on every card and gives the same greedy replies as before the upstream merge
([data](../../docs/data/2026-11-13_c831_upstream/README.md)). Whether v4 is faster than v3.0 on Ornith is what the
release test below measures; v3.0 on this file and line: 16.3 tok/s, 12 of 12, and 16.42 tok/s, 12 of 12, on a
second rig ([data](../../docs/data/2026-11-05_release_regression/README.md),
[second rig](../../docs/data/2026-11-06_ornith_mtp_rig2/README.md)).

## Run it

From a clone of this repository, with the v4 engine (the fork's tag `v4-rc2`) built or unpacked beside it as
`llama-p106-v4/`, and the file downloaded as in [release/ornith-v3](../ornith-v3/README.md#the-file):

```bash
MODELS=/mnt/models; . llms-on-crypto-mining-rigs/release/ornith-v4/settings.env && env $PRESET_ENV llama-p106-v4/llama-server -m "$MODELS/$PRESET_MODEL" $PRESET_ARGS --host 127.0.0.1 --port 8080
```

Keep the host quiet while it works, as on v3.0.

## The preset: [settings.env](settings.env)

v3.0's command line, unchanged, v3.0's keep-alive (`GGML_CUDA_KEEPALIVE=decode`, 16.3 against 14.9 tok/s), and three
switches:

| Switch | What it does on Ornith | Proved by |
|---|---|---|
| `GGML_CUDA_KEEPALIVE_EAGER=1` | under CUDA 12's lazy loading a kernel's first launch on a card waits until the keep-alive block on that card times out, about 1 s, at the first call of every new graph shape; with it the block is ended first. The same tokens | C811 and C817 (Q1417, Q1426); on nine P106 cards with Flash-Next a 2,760-token prompt took 24,712.63 and 24,744.13 ms against 26,637.95 and 26,065.24 without it (R140) |
| `LLAMA_SERVER_YIELD=2` | a step's hand-over to the server's reader thread wakes it only when a read waits | C741 (Q1310); R122, with Flash-Next |
| `LLAMA_SAMPLING_TOP_K_ROWS=1` | the card's sampling chain (`-bs`, top-k 20) runs one top-k over its output rows and reads the logits without their padded copy; the same tokens | C691 (Q1223), its graph fixed by C802 (Q1406); R83, with Flash-Next |

Each acts in host code, the keep-alive or the sampling graph, the same for every model, and was checked on nine P106
cards.

## What is left out, and why

| Switches | Why not on Ornith |
|---|---|
| The sparse QSA attention and its tables (`GGML_CUDA_FATTN_SPARSE_PASCAL*`, `LLAMA_QSA_*`) | only Flash-Next's QSA layers have them |
| The MTP draft's switches (`LLAMA_MTP_*`, `LLAMA_DRAFT_*`, `LLAMA_VERIFY_PREPARE`, `LLAMA_SPEC_REJECT_STOP`, `--spec-*`) and the n-gram table's prefetch | Ornith's preset runs no draft and the file has no n-gram table |
| `LLAMA_HOP_CAST=q8_0` | Ornith's graph does not call it (Q1125) |
| The kernels of 2–8-token passes: `GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM`, `_KCOL_TABLE`, `_KCOL_EXACT` (takes Q8_0), `_KCOL_SKIP=q4_0,q5_0` (takes Q5_0), `_Q3K_MT`, `_Q5K_MT`, `_Q6K_MT`, `_DMT`, `_GLU_MT`, `GGML_CUDA_MMVF_F32_MT` (takes the F32 shared-expert gate, one row) | all exact, but they act only on products of 2–8 tokens, which Ornith's line runs at most in a prompt's last micro-batch; the file holds N2P, N3P, Q5_0 and Q8_0 weights and F32 routers, so of their types only Q8_0, Q5_0 and the F32 gate occur |
| `LLAMA_KV_SEQ_RM_BOUND=1` | exact, but without speculation it acts once a request, when the server trims the cache to the reused prefix: nothing to gain |

## Candidates

Switches that may help Ornith but were never run on it. Each is an arm of the release test, the preset plus that one
candidate, never a default until it measures a gain or negligible there:

| Candidate | What it could buy | Why it is not in the preset |
|---|---|---|
| `GGML_CUDA_HOP_STAGED=1` | each hop between cards through a pinned host ring; exact on Ornith's F32 hops; part of Flash-Next's v3.0 preset | not run on Ornith in v3.0 |
| `GGML_CUDA_GRAPHS_FORCE=1` | prompt micro-batches captured as CUDA graphs without the two-call warm-up that pipeline parallelism never lets a prompt complete; Ornith's first token is its weak spot (8.39 s median on v3.0) | proved on Flash-Next only together with two draft switches (R123a); graphs take card memory, and this file fills two cards to 5,635 MiB |
| `--ctx-checkpoints-vram 4 --ctx-checkpoints-incall` | Ornith is a hybrid, so the server takes context checkpoints of its 36 Gated DeltaNet layers; on the cards and inside a prompt's call they took 18.99 % off Flash-Next's prompt time (R85) | the checkpoints take card memory beside the 5,635 MiB two cards already hold; the fit is checked first |
| `--ubatch-taper short=600:64` | a prompt of at most 600 tokens in micro-batches of 64 (Flash-Next, R60) | not run on Ornith |
| `-b 4096` | prompt calls of 4,096 tokens instead of 2,048 (Flash-Next, R123b) | larger host buffers on a host that left 563 MB free at its lowest in Ornith's agent check |
| `LLAMA_SCHED_PER_SIZE_RESERVE=2` | after a re-plan of the shared scheduler, the plan for the largest batch over the whole memory again | it acts only when the scheduler re-plans: an arm only if the release test's report counts graph reallocations above 0 |

The prompt kernels for N2P and N3P are still untuned (Q351 in [quants/experiments.md](../../quants/experiments.md));
that is code, not a setting, and is not part of v4.

## The release test

On one rig, with this file, both arms on the same line:

- **The agent check** ([bench/agent_check](../../bench/agent_check/README.md), 12 bugfix tasks, a 12-minute cap a
  task, the makers' sampling at temperature 0.6): `v3` (the v3.0 release build `1551f5fe1`, tag `miner-llama-v3.0`,
  with [release/ornith-v3](../ornith-v3/settings.env)'s keep-alive) against `v4` (`v4-rc2` with this preset), two runs
  each, alternated, each run its own load.
- **The deep lane** ([bench/deep_lane](../../bench/deep_lane/README.md)): the same two arms, two loads each after a
  warm-up load, prompts of 0, 4,096, 8,192, 16,384, 32,768 and 48,640 tokens (the deepest that leaves room for the 256
  tokens written in the 49,152-token context), with the sampler pass.
- Then the candidates above, each as an arm against `v4` in the agent check, two runs each.

## The deep lane: v4 against v3.0 at depth

Measured on 2026-11-14 on nine P106-090 cards with this file and line, two loads each after a warm-up, 256 tokens
written twice at each depth ([data](../../docs/data/2026-11-14_r162_ornith_release_deep/README.md)). Every value is
one load; classes by the deep lane's rule (a gain or a loss beyond the larger of the two arms' spreads between their
loads, else negligible).

**v4 is level with v3.0 at depth: it decodes 12.92 and 12.92 tok/s over six depths from 0 to 48,640 tokens against
v3.0's 12.88 and 12.88 (+0.3 %, a gain), and reads prompts at 145.2 and 145.0 tok/s against 146.8 and 146.7 (−1.1 %, a
loss)**, with the same greedy reply as v3.0 at every depth:

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

**The candidates stacked read prompts 15.1 % faster.** The same run held a third arm, this preset plus every candidate
above but `LLAMA_SCHED_PER_SIZE_RESERVE=2` (no load re-planned): `GGML_CUDA_HOP_STAGED=1 GGML_CUDA_GRAPHS_FORCE=1
--ctx-checkpoints-vram 4 --ctx-checkpoints-incall --ubatch-taper short=600:64 -b 4096`. It fits (at most 5,773 MiB on
a card, 1,227 MiB of host memory left at the lowest, in a fit load at 48,640 tokens). Against this preset it reads
prompts faster at every depth: +65.1 % on the 64-token prompt, +26.4 %, +27.1 %, +14.5 %, +10.8 % and +14.3 % at
4,096, 8,192, 16,384, 32,768 and 48,640 tokens, all five chunks at 167.2 and 166.9 tok/s against 145.2 and 145.0
(+15.1 %). Decode is level: 12.94 and 12.93 tok/s against 12.92 and 12.92 over all depths (+0.1 %, a gain), a gain at
0 and 4,096 tokens, negligible at 8,192, 32,768 and 48,640, a loss of 0.1 % at 16,384. With the checkpoints on the
cards the host's lowest MemAvailable is 1,238 and 1,254 MiB against 651 and 655. The stack writes another greedy
reply than this preset at 8,192, 16,384 and 32,768 tokens (the same at 0, 4,096 and 48,640); which candidate changes
it is registry row Q1487. Whether the stack also wins the agent check decides the preset.
