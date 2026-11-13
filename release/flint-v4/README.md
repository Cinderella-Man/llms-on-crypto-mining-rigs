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
