# Qwen3.8-Flash-Next on Miner Llama v4

The v4 preset for Flash-Next, UkisAI's Swift-1.5 GSQ-RCO Q2_0 with its MTP draft on nine P106-090 cards: the one
command that starts it and the settings it was measured with. Which files to download, and every v3.0 setting
explained, are in [release/flash-next-v3](../flash-next-v3/README.md); the model's page is
[models/qwen3.8-flash-next](../../models/qwen3.8-flash-next/README.md).

## What v4 changes for Flash-Next

Almost all of v4's work was done on Flash-Next, so it gains the most: 33 switches on top of v3.0's six, each proved
on nine P106 cards and each giving the same tokens or the same reply distribution. Most of them make a speculative
step cheaper: the three drafted tokens are checked in kernels that read each weight once for all of them, the draft
and the host do less between steps, and the sparse attention skips most of a long context. A few read prompts
faster: bigger prompt calls, prompt micro-batches as CUDA graphs, checkpoints kept on the cards. The coding agent
gets 27.61 and 26.67 tok/s against v3.0's 20.66, and deep in a context v4 reads prompts at more than twice v3.0's speed.

## Speed

A coding agent fixing twelve small bugs, with real tool calls and test runs, 65,536 tokens of context
([bench/agent_check](../../bench/agent_check/README.md)), on nine P106-090 cards
([data](../../docs/data/2026-11-13_r156_agent_v4r/README.md)):

| Engine and line | Tasks passed | Minutes a task, equal work | Generation tok/s | Prompt tok/s | First token per turn, s (median, p90) | Draft tokens a step |
|---|---|---:|---:|---:|---|---:|
| v3.0 (`1551f5fe1`), its preset | 11 of 12 | 3.08 | 20.66 | 65.5 | 6.52, 16.17 | 3.09 |
| **v4 (`v4-rc2`), this preset, run 1 (a second rig)** | **12 of 12** | **2.43** | **26.67** | 133.9 | 2.42, 9.00 | 3.25 |
| **v4 (`v4-rc2`), this preset, run 2** | **12 of 12** | **2.37** | **27.61** | 144.6 | 2.48, 8.65 | 3.28 |

Deep in a context (the [deep lane](../../bench/deep_lane/README.md): one prompt of each depth, 256 tokens written,
two loads per arm; [data](../../docs/data/2026-11-13_r157_release_deep/README.md)):

| Measure | v3.0, two loads | v4, two loads | v4 against v3.0 |
|---|---|---|---:|
| Prompts, five chunks to 55,248 tokens, tok/s | 91.2, 91.4 | 218.5, 218.8 | +139.4 % |
| Decode over six depths (0 to 49,152 tokens), tok/s | 16.41, 16.61 | 26.15, 26.14 | +58.4 % |
| Prompt at 49,152 tokens, tok/s | 70.3, 70.6 | 180.7, 180.8 | +156.5 % |
| Decode at 49,152 tokens, tok/s | 10.66, 10.98 | 24.26, 24.14 | +123.6 % |

Greedy decode at depth 0 is the one loss (23.50 and 23.49 against 26.71 and 26.77 tok/s): v4 writes another greedy
reply to that 64-token prompt and its draft is accepted less often on it, while its step is 20.1 % shorter.

## Run it

Set up the files as in [release/flash-next-v3](../flash-next-v3/README.md#run-it), steps 1 to 3 (`MODELS` holds
shards 1 and 2, the link to shard 3 and the MTP draft). Then, from a clone of this repository, with the v4 engine
(the fork's tag `v4-rc2`) built or unpacked beside it as `llama-p106-v4/`:

```bash
. llms-on-crypto-mining-rigs/release/flash-next-v4/settings.env && env $PRESET_ENV llama-p106-v4/llama-server -m "$MODELS/$PRESET_MODEL" -md "$MODELS/$PRESET_DRAFT" $PRESET_ARGS --host 127.0.0.1 --port 8080
```

Keep the host quiet while it works: in the two agent runs its lowest free memory was 312 and 419 MiB.

## The preset: [settings.env](settings.env)

Each switch and flag in the file carries a comment with what it does and the rows that proved it. Against v3.0's
preset:

| Group | What it adds | Proved by |
|---|---|---|
| v3.0's preset | the k-column kernels' budget, the Q8_0 hop, the staged hop, the draft's vocabulary shortlist, the keep-alive | v3.0's release check |
| R70's candidate | the sparse Pascal attention, probabilistic draft verification, four checkpoints on the cards, the prompt taper, the draft's micro-batch of 512 | [2026-11-10_v4_candidate](../../docs/data/2026-11-10_v4_candidate/README.md), [2026-11-11_r90_v4_vs_v3](../../docs/data/2026-11-11_r90_v4_vs_v3/README.md) |
| The switches of 2026-11-11 | read-once kernels for IQ4_XS, Q4_K and Q5_K, one top-k over the output rows, the QSA indexer and k-pool on the cards, checkpoints inside a prompt's call | the same |
| R72's `v4c2` | the QSA staging and mask on the cards, eight host switches of the speculative step, prompt graphs, the draft's catch-up overlapped, read-once kernels for Q3_K and Q6_K and the fused shared expert, the end-of-turn stop, the draft's token table on its card, `-b 4096` | [2026-11-12_r72_agent_v4c2](../../docs/data/2026-11-12_r72_agent_v4c2/README.md) |
| R142's `v4c3` and the rows of 2026-11-13 | the draft's schedulers planned once, the F32 gate on `mul_mat_vec_f`, the eager keep-alive stop, the lazy hidden-row half, exact k-columns, the Q4_0/Q5_0 skip at 2-4 tokens | [2026-11-13_r142_agent_v4c3](../../docs/data/2026-11-13_r142_agent_v4c3/README.md), [2026-11-13_r156_agent_v4r](../../docs/data/2026-11-13_r156_agent_v4r/README.md) |

Left out after their rig rows: C812's prompt split (`LLAMA_PP_MULTI_SPLIT=1`, a loss), the union gather (a loss for
the decode step at every depth with context), the adaptive draft length (below `v4c3` in its first run).

## Candidates

None: every v4 switch Flash-Next's path runs was decided on the rig, and the final line `v4r` measured within the
spread of `v4c3`'s two runs (2.37 and 2.43 against 2.35 and 2.44 minutes a task).
