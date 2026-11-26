# What a preset changes in the replies

A preset (`release/<model>-<version>/settings.env`) sets dozens of switches and flags. Most of them only move work,
memory, clocks or timing and give the same replies. A few change what the model computes, and those are the ones to
keep count of, because small changes can add up. Every `PRESET_ENV` and `PRESET_ARGS` line of a v5 preset starts its
comment with a tag saying which kind it is, and [scripts/preset_effects.py](../scripts/preset_effects.py) counts the
tags and fails on a line without one:

```bash
python3 -I scripts/preset_effects.py           # a table per v5 preset and every line that is not [same]
python3 -I scripts/preset_effects.py --check   # exit 1 if a line has no tag or an unknown one
```

A new line in a preset gets its tag in the same change, from what was measured: "the same replies" in a check is
`[same]`; a kernel or micro-batch change is `[order]` unless the replies were shown to be the same bytes.

## The tags

| Tag | What it means | Can it add up? |
|---|---|---|
| `[same]` | The same replies as without it: checked to give the same tokens, or it only moves work, memory, clocks or timing. | No. |
| `[draft]` | Only the draft's guesses, or how they are checked, change. Greedy replies stay the same; sampled replies come from the same distribution with other random draws. | No. |
| `[order]` | The same arithmetic in another order or another kernel. The logits move by rounding, so a reply can take another token where two are nearly tied. Not a loss on average. | Not as an error: several give one other rounding, not a growing one (est.). On these files one rounding change alone is measurable, so the whole stack gets its own check (below). |
| `[input]` | The model reads other tokens for the same text. | Yes, in principle; one line per preset. |
| `[lossy]` | Numbers rounded to fewer bits on purpose. Each is measured on its own and passed inside noise. | **Yes.** These are the lines whose errors can add up. |
| `[sampling]` | The sampling values (the makers' values in Miner Llama's own presets). | By design. |
| `[mode]` | A mode the replies depend on by design, for example reasoning off. | By design. |

The model file itself matters far more than any line: a 2-bit file moves the replies much more than the lossy lines
do. It is the preset's `PRESET_MODEL`, not a tagged line.

## The v5 presets

| Preset | `[lossy]` | `[input]` | `[order]` | `[draft]` | `[same]` | `[sampling]` / `[mode]` | Lines |
|---|---:|---:|---:|---:|---:|---|---:|
| [Flash-Next](../release/flash-next-v5/settings.env) | 1 | 1 | 11 | 5 | 32 | 1 sampling | 51 |
| [Ornith-1.5-122B](../release/ornith-v5/settings.env) | 2 | 1 | 2 | 0 | 9 | 1 sampling | 15 |
| [Flint](../release/flint-v5/settings.env) | 0 | 1 | 0 | 0 | 7 | 1 sampling | 9 |
| [Qwen3.6-35B-A3B](../release/qwen3.6-v5/settings.env) (a contributor's) | 0 | 0 | 2 | 4 | 7 | 1 mode | 14 |

### Flash-Next

- **`[lossy]`, 1:** `LLAMA_HOP_CAST=q8_0`. The residual crosses each of the eight card boundaries in 8 bits. Against
  F32 hops it gave no measurable KL divergence (C200k); every cast below 8 bits failed its KLD gate (Q963).
- **`[input]`, 1:** `LLAMA_SERVER_TEXT_REUSE=1`. When a reply comes back in the next request, the model reads the tokens
  it generated, where without it the server reads the re-tokenized text (Q260, Q1604).
- **`[order]`, 11:**
  - which kernel takes the 2–8-token products: `_KCOL_SMEM`, `_KCOL_TABLE` and `_KCOL_EXACT`; with `_KCOL_EXACT` a
    verify pass gives each token the sums a one-token pass gives it;
  - the attention over the same kept cells on another kernel or split count: `GGML_CUDA_FATTN_SPARSE_PASCAL=1`,
    `_MIN_RATIO=0.5`, `_NSM=10` and `_PROMPT=1`;
  - the run line's `-fa on` and `-bs`, `-b 4096 -ub 512` and `--ctx-checkpoints-incall`;
  - `--ubatch-taper short=1400:64,cuts=1`: reading a prompt in micro-batches of 64 instead of 512 moves the next-token
    distributions by a mean KLD of 0.0127, with the same top token 96.9 % of the time (FN95, Q1493); `cuts=1`, which
    plans each stretch between a call's checkpoint cuts on its own, moves them by 0.004828 more against the taper
    without it, the same top token at 98.061 % of the positions it can change (C981, Q1714).
- **The file:** Swift GSQ-RCO Q2_0, the makers' 2-bit file.

### Ornith-1.5-122B

- **`[lossy]`, 2.**
  - **The 8-bit attention cache** (`-ctk q8_0 -ctv q8_0`): within noise of the F16 cache up to 49,152 tokens. Past
    8,192 tokens a perplexity rise of a few tenths of a percent is not ruled out (C976). The same check at 65,536
    and 98,304 tokens is pending.
  - **Its int8 key scoring** (`GGML_CUDA_FATTN_TILE_PASCAL_KDP=1`, since 2026-11-26): the prompt kernel rounds the
    queries to 8 bits to score the 8-bit keys with `dp4a`. KLD 0.005951 against the preset without it, inside both
    rounding yardsticks (0.006160 and 0.005794); perplexity +0.06 %, and +0.20 % at 49,152 tokens (C1103).

  The two were measured one after the other: the cache against F16, then the key scoring against the cache. Both
  together against the F16 cache has not been measured.
- **`[input]`, 1:** text reuse, as on Flash-Next.
- **`[order]`, 2:** `GGML_CUDA_FATTN_TILE_PASCAL_MIN=1` and `GGML_CUDA_FATTN_DEC_PASCAL=4`, the decode attention on
  pull request 9's kernels. `--ctx-checkpoints-incall` gave the same replies on Ornith (FN93b), so it is `[same]` here.
- **Not in the preset:** the micro-batch taper. On Ornith it moves the next-token distributions by a mean KLD of 0.047,
  about the file's own distance from the makers' Q8_0 on agent text (0.052), with the same perplexity (Q1493), so it
  stays opt-in.
- **The file:** Ornith-1.5-122B Pascal N2P GGUF (`valid1`), 2-bit experts.

### Flint

- **No `[lossy]` line.** `[input]`: text reuse. The taper is `[same]` on Flint: the same numbers to the digit (KLD 0,
  perplexity 0.000 %; FN96).
- **The file:** Flint Pascal Q8 GGUF (`p065`).

### Qwen3.6-35B-A3B (a contributor's preset)

- **`[mode]`, 1:** the run line holds `--reasoning off`, so the model answers without thinking first.
- **No sampling values are set,** so llama-server's defaults apply unless the client sends its own.
- **`[order]`, 2:** `-ub 128` and `--ctx-checkpoints-incall`. **`[draft]`, 4:** the MTP head as the guesser, its
  shortlist, its cache in Q8_0 and the pipe under a grammar; the model checks every guess.

## What can add up, and the check that measures it

- **Lossy lines:** Flash-Next 1, Ornith-1.5-122B 2, Flint 0, Qwen3.6 0. Each one passed its own quality check. On
  Ornith the two were measured as a chain, not their sum against F16.
- **Order lines** are rounding, not error. Rounding the same math differently has no direction, so stacking them gives
  another rounding of the same size (est.). The files here are sensitive to it, though: one micro-batch change alone
  moves Flash-Next's distributions by KLD 0.0127 and Ornith's by 0.047.
- **The check (Q1916):** each preset's full line, and the same file with every `[lossy]`, `[order]` and `[input]` line
  set back, both scored against the reference logits on agent text at depth (KLD, same top token, perplexity). If the
  full line is no further from the reference than the plain one, the stack costs nothing measurable. If it is, the
  difference is what the stack costs, and the lines are taken back one at a time to find it.

## The task that runs long

`bugfix-javascript-variable-length-quantity` is the agent check's longest task for every reasoning model here, and it
ran long well before any v5 switch.
- **In the reasoning census** it has the most reasoning per attempt: Flint 6,852 tokens an attempt, 90.49 % of them
  reasoning; Flash-Next on the v3.0 release 7,090, 94.09 % reasoning
  ([data](data/2026-11-09_reasoning_census/README.md)).
- **It reached the 12-minute cap** on v3.0 and on the v4 candidate on 2026-11-11
  ([R57](data/2026-11-11_r57_v4night_p106/README.md), [R90](data/2026-11-11_r90_v4_vs_v3/README.md)). On 2026-11-26 one
  reply of about 21,550 tokens ran into the cap, on `pmix-q2-fast` at the v5 preset
  ([C1173](data/2026-11-26_c1173_gmix_agent_rig1/README.md)).
- **The loop census** of 4,713 replies (2026-10-26 to 2026-11-06) found the replies at the cap to be long reasoning,
  not loops. A repetition rule trips most often on this task, 11 of the 36 cases in the v3 release and Flash-Next
  arms, because its reasoning lists the same values again and again; 73 of the rule's 82 cases ended on their own
  ([data](data/2026-11-09_loop_census/README.md)).
- **Open:** a reply cut at the cap is not kept, because the agent check's proxy records requests and a cut reply never
  completes (Q1917 keeps its text). The census has not been run over v5's agent runs yet (Q1918).
