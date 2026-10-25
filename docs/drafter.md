# Building a drafter for pipelined speculation

> **How to update this document.** This page is the one canonical description of how the drafter is built and how far
> it has got. It changes as we learn.
> - **The training session** updates it after every scored run: a row in [Results](#6-results-and-learning-curve), a
>   changelog line, and a new method section whenever the recipe changes (data layout, windows, architecture, loss,
>   sampler).
> - **The engine session** updates it after every engine or rig result: steps 7–9 of [the process](#2-the-process-step-by-step),
>   a results row if acceptance was measured on the rig, a changelog line.
> - **Never remove a superseded method.** Mark it "superseded by Mn (date)" and leave its numbers.
> - Every result states its date, its card count and its coupling. Link the research notes and registry rows instead
>   of copying them; keep this page to tables and short bullets.

## 1. What a drafter is, and where we stand

On a layer split only one card works at a time. **Pipelined speculation** ([note](research/notes/pipelined_speculation.md),
Q87) feeds a guessed next token into card 1 while the later cards still finish the current one; the last card samples
every token and keeps a guess only if it equals the sample, so the output is exactly plain decoding's. Idle cards are
worth as much as the guesses are right.

The **drafter** (draft head) is the guesser: a small transformer that predicts Flint's next token from the hidden states
Flint has *already* computed for the tokens in flight. The newest token has only passed card 1, the one before it
card 2 or 3, older ones are complete: a "staircase" of depths. The design follows SPD (arXiv 2605.30852); why this
design and not EAGLE-3, Medusa or MTP is in [pipeline drafter](research/notes/pipeline_drafter.md). It sits behind the
copy lookup, which guesses first inside copied text.

| Where we stand (2026-10-25 morning) | |
|---|---|
| Best held-out score | 0.502 coupled acceptance on Q51 fresh text, 0.562 overall (M1, run R1, 13.5 M tokens, six cards); after a miss 0.541 / 0.603 |
| Best validation score | 0.580 staircase (M1, R1, 180 min) |
| Simulated speed | R1 on six cards: 1.90× plain decoding (70.9 tok/s), against 1.34× for the copy lookup with the n-gram cache |
| Gate to the rig | ≥ 0.55 on Q51 fresh text and ≥ 0.6 overall, coupled. Not a kill line (decision 2026-10-24 21:25) |
| Next | M2 round of 2026-10-25 06:00: four depth-general heads (3–12 cards, 6 + 9, 9, 6) on the same data |
| Engine | guesser built (`--pipe-guess staircase`, fork branch `pascal-p106-v2-drafter`); not yet timed on the rig |

Terms used below:

| Term | Meaning |
|---|---|
| Qnn | a row of the [experiment registry](../quants/experiments.md) |
| Ln | the input of Flint layer n (Flint has 49 layers, hidden size 2,048); "full" is the final hidden state |
| Anchor | one Flint hidden state fed to the head for one token |
| Staircase | the head on card 1 right after card 1's stage: newest token at card 1's output, older tokens deeper |
| Pre-step | the head on a spare card, before the cycle: the newest token has only its embedding |
| After-miss | the first guess after a wrong one: newest token at card 1's output, every older token complete |
| Acceptance | probability that the guess equals Flint's sample (temperature 1, top-k 20, top-p 0.95), given a correct prefix |
| α | the head's distribution is sharpened to q^α before the draw; chosen on validation (3 for every head so far) |
| Q51 | the held-out test set: 38 transcripts of hand-written prompts, 65,520 positions, 72.3 % fresh text |
| Fresh text | positions outside a ≥ 8-token copy from the prompt or earlier output (the copy lookup covers the rest) |
| Val | 3 % of the training answers held out by `train.py` |

## 2. The process, step by step

Prompts → answers → dump → train → score → simulate → export → engine → rig. Steps 1–6 run on the 7-GPU quant machine
(3 × RTX 3090, 2 × V100 32 GB, 2 × RTX 3080), never on the rig; steps 7–9 end on the rig. Commands and file formats:
[bench/staircase_drafter](../bench/staircase_drafter/README.md) (steps 1–6) and its [RIG.md](../bench/staircase_drafter/RIG.md)
(steps 7–9).

### Step 1: prompts

- **In → out:** public datasets → `prompts*.jsonl`.
- **Tools:** `prompts.py` (7,150 first turns: ShareGPT, UltraChat, Magicoder, Hermes function calling, NuminaMath,
  OpenR1-Math; 1,100 whole-file edits and 1,100 coding-agent turns built from public source files; `--extend` adds
  more, deduplicated), `tool_prompts.py` (ToolACE 10,581 and glaive-function-calling-v2 4,749 prompts, both Apache-2.0,
  their tool lists passed as `tools` so that the chat template writes Flint's own tool format).
- **Pitfalls:**
  - 44 % of the tool prompts failed to render until the datasets' `"int"`/`"float"` schema types were mapped to
    JSON-schema types (`integer`, `number`).
  - Hold out everything Q51 draws from: its prompts and their source files. The continuation store (Q161) uses the
    stricter rule "no 12-word span shared with a Q51 prompt".
  - Agent turns are 1,100 of the first 7,150 prompts but 2.5 % of the generated tokens (short answers). More of them
    costs nothing in training.

### Step 2: generation (answers)

- **In → out:** prompts → answers with exact token ids (`gen-*.jsonl`).
- **Tools:** `generate.py` against llama-server (`--jinja -no-kvu -bs --cache-ram 0`), the makers' sampling
  (temperature 1.0, top-k 20, top-p 0.95, min-p 0), reasoning kept, ≤ 2,048 tokens, seed 20000 + prompt number. Several
  server copies share one prompt file; each prompt is claimed by an exclusive file create.
  On RTX 3090s (and other cards of compute 7.5–8.9) the fastest writer is vLLM: `generate.py --api vllm` against
  vLLM nightly `c50aca14` with the makers' NVFP4 GGUF (W4A16 on Marlin), pipeline-parallel over two cards,
  `--attention-backend TRITON_ATTN` and the attention patch ([vllm/](../bench/staircase_drafter/vllm/), one command
  `vllm/run_generation.sh`).
- **Measured (2026-10-25 night):** G3 on two RTX 3090s at 96 slots: ~350 answer tokens/s (a decode step every
  ~270 ms); one UD-Q4_K_XL copy on a V100 at 48 slots: ~107. In all ~0.46k answer tokens/s, ~1.7 M tokens an hour.
  Tool prompts run at ~210 answer tokens/s (long tool lists, short answers).
  vLLM NVFP4 on the same two RTX 3090s with 160 requests in flight: ~2.4–2.7k answer tokens/s in vLLM's log (2.25k
  averaged over the first 10 minutes), ~7× the G3 copy, ~8–9 M tokens an hour; KV cache 50–75 % full, no preemption.
  Its answers have G3's shape on the same prompt family: 63 % end at an end token, mean 1,182 tokens (G3: 63 %,
  1,217).
- **Why G3 on two cards is slow:** all 96 slots stay busy, and the server's main thread at 100 % CPU is the CUDA driver
  spin-waiting, not CPU starvation. A 96-token decode batch is one ubatch at `-ub 512`, so the two cards of the layer
  split take turns and each is ~50 % busy. `-ub 48`, which splits each decode step into two ubatches, made a step
  slower (~455 ms, cards 31–38 % busy): the two halves ran one after the other, not pipelined, and each read the
  weights. More throughput needs a server that overlaps the cards, or a model that fits one card per copy.
- **Pitfalls:**
  - The generator only has to write Flint-like text: the dump re-labels every target with the Q8_0 teacher. A writer
    other than Q8_0 costs est. 1–3 % of acceptance by the literature ([data generation](research/notes/drafter_data_generation.md)),
    against ~6 % gained per doubling of data.
  - Give every slot its own KV cells (`-no-kvu`); with the unified cache 64–72 slots crawled at ~300 tok/s together.
  - 38 % of answers stop at the 2,048-token cap.
  - vLLM's Flint model asserts FlashInfer attention, which serves attention sinks only on compute ≥ 9.0. Below that
    `vllm/patch_flint_attention.py` routes attention through the standard layer, where `TRITON_ATTN` serves the
    sinks. V100s can't run vLLM's CUDA 13 images at all; they stay on llama.cpp.
  - The Hugging Face chat template gives llama-server's prompt ids only once string tool-call arguments are parsed
    into objects (`vllm/check_prompts.py`: 900 of 900 rows).
  - A client that marks a prompt failed when the server is unreachable burns through the whole prompt pool in
    minutes after a crash; `generate.py` keeps the claim and retries for 30 minutes.

### Step 3: dump (teacher targets and hidden states), the bottleneck

- **In → out:** answers → per position: the layer inputs of a fixed layer grid and the final hidden state (FP8 E4M3,
  one scale per row) plus Flint Q8_0's top-20 next-token log-probabilities.
- **Tools:** `feats.py seqs`, then `llama-staircase-dump` (built inside a llama.cpp tree) with Flint Q8_0;
  `check_dump.py` (Flint's output head on the dumped final state gives the dumped top-1 at 98–99 % of positions).
  `DUMP_ALLPOS` keeps card 1's layers for every position, the rest for the answer only; `DUMP_OUT2` or `feats.py pack`
  writes a smaller subset for a fast disk.
- **Measured:**

  | Card set | Layer grid | Tokens/s (prompt + answer) | Answer tokens/s |
  |---|---|---|---|
  | last card on PCIe x8 | 17 layers | ~1,650 | ~1.1–1.2k |
  | last card on PCIe x8 (two 3080s + a 3090) | 26 layers | 1,230 | ~1.0k |
  | three cards, last on x4 | 17 layers | ~850 | ~0.7k |
  | V100s on x1 links | 17 layers | ~280 | ~0.2k |

- **Storage:** 37 KB per answer token with the 17-layer grid, 58 KB with the 26-layer grid (M2), ~12 KB for a
  six-card pack. 70 M tokens at 37 KB is ~2.6 TB.
- **Pitfalls:**
  - Without help the dump copies all 150,144 logits per position to the host (600 KB). `DUMP_GPU_TOPK=1` (with `-b`
    equal to `-ub`; v2 commit `d402386`) takes the top-20 on the card and copies ~160 B. On Q51 it gives the same top-1
    everywhere and the same top-20 as the host path, but with the last card on x8 it ran at 0.89× the host path
    (1,098 against 1,230 tokens/s; `-b 512` costs more than the logits copy saves), so it pays only on narrow links.
    Otherwise put the card holding the output head on the widest link.
  - The layer grid fixes which splits a dump can serve. Record every boundary you may need (see M2's grid).
  - The dump replays every prompt token but trains only on answer tokens. Tool-call answers are ~60 % prompt (442
    prompt against 283 answer tokens each); chat and code answers ~10 %. When the dump is the bottleneck, dump general
    answers first and tool calls later: on 2026-10-25 the d4 queue was 87 % tool answers (the night's generation ran
    the tool prompt file first) and half of all its tokens were prompt.
  - Several dump instances share one pool through `feed_split.py`; with generation at ~2.5k answer tokens/s (vLLM) one
    three-card instance (~0.5–1k) falls behind.
  - Scanning a big feature file's record headers on a busy HDD takes hours (each header read pulls ~2.4 MB of
    read-ahead); take the dumped sequence ids from the dump's log instead.

### Step 4: pack and train

- **In → out:** dump files + vocabulary + Flint's embedding table and output head → `heads/<X>/head.pt`.
- **Tools:** `vocab.py` (40,000 most frequent answer ids plus Flint's control tokens; 32,000 gave 94 % code coverage on
  Q51), `flint_weights.py`, `train.py` (model in `drafter.py`): BF16 on RTX cards, FP16 with loss scaling on V100s,
  checkpoints every 30 min. `--resume` continues a run; `--init-from` warm-starts a changed model (weights only, new
  parameters at zero, fresh optimizer). M1 options: `--window-sample`, `--fuse`, `--loss lk --eta 3`, `--noise`,
  `--dropout`, `--val-seqs`; M2 adds `--general` and `--cards`. On a CPU (checks) training and scoring run in FP32:
  BF16 autocast on a CPU without BF16 instructions runs nearly single-threaded.
- **Measured:** M0's head A 23–25 steps/min on an RTX 3090; M1 44 steps/min. One epoch of 13.9 M tokens is ~1,650 steps
  of ~8k targets, ~38 min under M1.
- **Pitfalls:**
  - A 10 GB RTX 3080 can run out of memory with M1's settings (R0 did twice, at step 1,080 and ~30 min into its rerun).
  - Repeats overfit: U's validation peaked at 0.615 after ~5.6 epochs (180 min) and fell to 0.609 by 240 min with its
    KL rising from 1.43 to 1.77; R1 was flat after 140 min, with train KL 0.67 against validation 1.69.
  - Several trainers reading feature files from one HDD can starve. Spread the files over disks, and let a sampled
    window read only its own layers.
  - The validation split is every 33rd record of the concatenated file list: append new files at the end, or the split
    changes and validation leaks into training.
  - A disk guard that moves packs between disks must leave a symlink at the old path; a container sees that symlink
    only if it mounts the target disk at the same path.
  - Budget ≤ ~4 epochs per data size; beyond that repeats add little.

### Step 5: score

- **In → out:** head + Q51 dumped the same way → per-position acceptance (`scores-<head>.npz`, `summary.json`).
- **Tools:** `score.py --alpha` (α from validation, stored as `heads/<X>/alpha.txt`); 95 % intervals by resampling
  whole transcripts; `report.py` for the gate table.
- **What to report:** coupled acceptance on fresh text and overall; both couplings (Gumbel and inverse-CDF,
  [section 5](#5-coupling-with-the-sampler)); from M2 on, per card count (3, 4, 5, 6, 7, 8, 9, 10, 12; nine first).
- **Pitfalls:**
  - Validation runs 0.06–0.08 above Q51 fresh text (A at 8.6 M: val 0.527, Q51 fresh 0.463; R1: val 0.580, Q51 fresh
    0.502) and close to Q51 overall (R1 0.562), so only Q51 decides.
  - Compare heads paired, per position, not by their single intervals (±0.024 each).
  - All windows' logits at once (40,001 classes × positions × windows) run out of memory on 10–12 GB cards;
    `--window-group` scores a few windows per pass.
  - A head scores only on the layers it was trained on: six-card head A gets ~0.05 on 7–9-card windows that read
    layers it has no trained input for.

### Step 6: simulate

- **In → out:** per-position acceptance → tokens per second per workload.
- **Tools:** `sim.py` on top of [pipeline_sim.py](../bench/ngram_replay/pipeline_sim.py) (event model: per-card stage
  times, 0.3 ms per hop, 1.5 ms from sample to launch; copy lookup first, the head after it; no drafter time charged).
- **The rig's own oracle curve** (Flint chat, K2h-r16, nine cards, depth 3; an oracle replays a reference run with a
  chosen share of right guesses):

  | Guesses right | 0 % | 55 % | 70 % | 100 % |
  |---|---|---|---|---|
  | tok/s | 43.9 | ~78 (est., interpolated) | 97.0 | 144.8 |

### Step 7: export

- **In → out:** `head.pt` + vocabulary + α → one GGUF side file (architecture `staircase-drafter`).
- **Tools:** `export_gguf.py`, `check_export.py`, `llama-pipe-drafter-check`.
- **Side-file format:** the head's matrices with every RMS norm folded into the next projection, the draft-id → Flint-id
  table, and per-depth metadata, so the engine needs no change for a new head shape:

  | Key | Six-anchor heads (M0, M1) | Depth-general heads (M2) |
  |---|---|---|
  | `depth_proj` (depth 0–49 → projection) | L9, L17, L25, L33, L41 and the final state: one each | `drafter.BUCKET`: five buckets plus the final state |
  | `depth_emb_row` (depth → embedding row) | the 7-row anchor embedding | the 50-row depth embedding |
  | `recorded` (depths trained on) | 9, 17, 25, 33, 41 | the dumps' common layers (e.g. 4, 6, 8, 9, 10, …, 47) |
  | `timing` | `split_depths`' stage model: 0.435 ms per layer at six cards, +0.024 per extra card (min 0.3), head 2.5 ms, hop 0.3 ms | same |
  | others | sizes, `fuse` (+ `fuse_depths`), `committed`, `alpha`, `general`, `target_layers` = 49 | same |

  `export_gguf.py` reads `general` from the run's config and `recorded` from its `--feats` files (or `--recorded`).
- **Size:** Q8_0 matrices with an F16 output head (the default since v2 `6362f5d`): 0.29 GB for a two-layer fused head
  with 40,000 ids, 0.27 GB unfused. All-Q8_0 is 0.22 GB but moves the logits by ~0.16 rms (most likely token unchanged,
  its probability by up to 0.07); the F16 head by ~0.035.

### Step 8: engine

- **Where:** fork branch `pascal-p106-v2-drafter`, included in the combined release candidate `pascal-p106-v2-rc`.
  `llama-pipespec` and `llama-server` take `--pipe-guess staircase --pipe-draft FILE`.
- **How it runs** ([RIG.md](../bench/staircase_drafter/RIG.md)): every one-token pipe step copies the inputs of the
  window's layers and its final state (8 KB each) to pinned host memory on the card that computes them, one per card;
  the head runs on card 1 in its own CUDA stream (or `--pipe-draft-dev N`), waits for the hidden states it reads, and
  returns its top 16. A late hidden state makes a guess late, never wrong. The executor still samples every token, so
  the output is plain decoding's.
- **Window shape, from the run's split** (since fork `pascal-p106-v2-drafter` at 2026-10-25 03:00): the engine reads
  the layers per card from the actual device assignment, runs `split_depths` with the file's timing (kind `stair` on a
  card that holds layers, `pre` on a spare card; after a miss the one-row window falls out), and maps each depth to the
  deepest recorded depth at or below it. Tokens in flight older than the window read the last boundary before the
  output card. Six-anchor heads go through the same rule with `recorded` = their five anchors, which reproduces their
  six-card staircase L9, L17, L33, L41. Examples with M2's recorded layers:

  | Split | Window (newest first) | Older tokens |
  |---|---|---|
  | 17/16/16 | L17, L33 | L33 |
  | 13/12/12/12 | L12, L25, L37 | L37 |
  | 9/8/8/8/8/8 | L9, L17, L33, L41 | L41 |
  | 9/8/8/8/8/8, spare card | emb, L9, L25, L33, L41 | L41 |
  | 6/6/6/6/5/5/5/5/5 | L6, L12, L24, L34, L39, L44 | L44 |
  | 6/6/6/6/6/6/6/5/2 | L6, L12, L18, L24, L30, L36, L42, L47 | L47 |
  | 5/4/…/4 (12 cards) | L4, L9, L17, L25, L33, L41, L44 | L44 |

  `--pipe-draft-stair` overrides the window, `--pipe-draft-kind stair|pre` the start.
- **Coupling:** inverse-CDF with the served sampler's own uniform ([section 5](#5-coupling-with-the-sampler)).
- **Limits:** one sequence (no streams, races or fed tokens); the embedding table on the head's card or in host memory.

### Step 9: rig

1. Cost of one guess on one P106 (`llama-pipe-drafter-check --bench 300 --split 9,8,8,8,8,8 --ctx 2048`, and the
   nine-card split; the window length follows the split). Est. 1.6–2.6 ms; card 1 has ~2 ms of slack per cycle on six
   cards, ~0.5 ms on the nine-card 6/…/5/2 split.
2. Pipelined decoding next to plain and the `cont` guesser (`rig_run.sh` with `DRAFT=…`), sampling on the last card at
   temperature 1, identity checked; six and nine cards, and the head on a spare card.
3. Served chat and tool requests against today's guesser (copy lookup + continuation counts + store).

The kit (tarball, random heads of the real shape, R1's head, `rig_d3.sh`) is ready; nothing has run on the rig yet.

## 3. Methods

| Method | Date | State | What it changed |
|---|---|---|---|
| [M0](#m0-the-pilot-2026-10-24) | 2026-10-24 | superseded by M1 (2026-10-24 22:00) | first heads: fixed six-card windows, all windows trained per sequence |
| [M1](#m1-sampled-windows-fused-anchors-warm-starts-2026-10-24-2200) | 2026-10-24 22:00 | current | one sampled window per sequence, fused anchors, warm starts; 1.8× the steps |
| [M2](#m2-depth-general-for-312-cards-planned-2026-10-25-0600) | planned 2026-10-25 06:00 | planned | layer-indexed anchors, one head for any 3–12-card split; specialists on a few card counts for comparison |

### M0: the pilot (2026-10-24)

**Superseded by M1 (2026-10-24 22:00).**

- **Head A:** 2 transformer layers at Flint's width, a 32k–40k output vocabulary, on card 1 with fixed six-card anchors:
  newest token at L9, the older ones at L17, L33, L41, then full depth. Guesses Gumbel-coupled, sharpened to q^α with
  α = 3 chosen on validation.
- **Other heads, same data:** B (spare card, pre-step), U (newest at L9, every older token complete: the after-miss
  inputs), C (card 1's output only), T (tokens only), and a 4-layer A as a size test.
- **Data:** Flint's own answers, written by Q8_0, then by the G3 quant: 3.08 M → 8.6 M → 13.94 M tokens. Targets: Q8_0's
  top-20 log-probabilities and hidden states from the dump.
- **Training:** time-capped at ~120 min per head; every window trained for every sequence; 23–25 steps/min (A).
- **Results** (Q51, coupled, α 3):

  | Head | Data | Fresh | Overall |
  |---|---|---|---|
  | A | 3.08 M | 0.422 | 0.440 |
  | A | 8.6 M | 0.463 (CI 0.439–0.486) | 0.505 |
  | A, after-miss inputs | 8.6 M | 0.504 | 0.553 |
  | B (pre-step) | 8.6 M | 0.380 | 0.405 |
  | T | 3.08 M | ~0.30 | – |

- **Learned:** ~+0.03 per doubling of data; validation ~0.06 above Q51 fresh; A scores only 0.06 on window shapes it
  never saw (pre-step, nine-card); an EAGLE-style feature regression (`--reg`) gave nothing. Analysis:
  [drafter acceptance](research/notes/drafter_acceptance.md).

### M1: sampled windows, fused anchors, warm starts (2026-10-24 22:00)

**Current.** Code: v2 commit `f7d15de`.

- **What changed:**
  - one window sampled per sequence (nominal staircase, after-miss, or a random staircase) instead of all of them;
  - fully masked attention keys skipped;
  - warm start from a trained head (`--init-from`);
  - every anchor a token has reached fused into its row (SPD's design), not only the deepest;
  - options: the LK hybrid loss (KL → coupled match), feature noise, dropout.
- **Speed:** 44 steps/min against 23–25 for M0.
- **Runs (13.5 M training tokens of the 13.94 M; the rest is validation and answers over 8,192 tokens):**
  - R1 (fused, warm-started from U), 180 min:
    - validation staircase 0.538 (20 min), 0.568 (120 min), 0.578 (140 min), 0.580 (180 min);
    - Q51 at α 3: staircase 0.502 (CI 0.477–0.526) fresh, 0.562 overall; after a miss 0.541 / 0.603;
    - α 1 is ~0.03 lower.
  - R0 (unfused control, from U): validation 0.545 at 20 min, against R1's 0.538. It ran out of memory on a 10 GB RTX
    3080 twice, so there is no longer run.
- **Learned:**
  - One epoch of 13.5 M tokens is ~1,650 steps (~38 min), so runs on fixed data converge in 2–3 h and then overfit.
    Data, not run length, is the limit.
  - The new recipe added ~+0.04 fresh over the M0 learning curve: 0.502 at 13.5 M, where M0's +0.03 per doubling
    projects ~0.48.

### M2: depth-general for 3–12 cards (planned 2026-10-25 06:00)

**Planned.** Owner requirement of 2026-10-25 01:10: the drafter must serve any split of 3–12 cards.

- **Anchors are layer-indexed:** any layer input 1–48 or the final state. Projections are shared within five depth
  buckets (layer inputs 1–12, 13–20, 21–28, 29–36, 37–48; the final state its own), plus a learned per-layer depth
  embedding. Warm-started from R1 with its fused-anchor terms dropped (M2 has no fusion), so it starts as R1 without
  fusion on six-card depths. Checked: a general head loaded from R1 reproduces R1 without fusion exactly (max difference
  0). After six smoke-test steps it already scores ~0.40 at 7–9 cards on two Q51 transcripts, against 0.05 for head A.
- **Card count drawn per sequence:**

  | Cards | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | Weight | 8 % | 8 % | 8 % | 25 % | 6 % | 6 % | 25 % | 5 % | 4 % | 5 % |

- **Splits:** near-even with jitter (up to two single layers moved between cards), 30 % with a light last card (1–3
  layers next to the output head).
- **Window types:** card-1 staircase 60 %, spare-card pre-step 15 %, after-miss 25 %, each built from that split's stage
  times, at most 12 rows. Each depth maps to the deepest recorded layer at or below it.
- **Data:**
  - d1–d3: a 17-layer superset, every 6- and 9-card boundary: 6, 9, 12, 17, 18, 24, 25, 29, 30, 33, 34, 36, 39, 41,
    42, 44, 47 + final.
  - d4 onward: a 26-layer grid: 4, 6, 8, 9, 10, 12, 14, 16, 17, 18, 20, 22, 24, 25, 27, 29, 30, 33, 34, 36, 37, 39, 41,
    42, 44, 47 + final (4, 6 and 9 for every position); 58 KB per answer token. Without L4, 10–12-card windows fall back
    to the embedding for the newest token.
  - Q51 is dumped with the 26-layer grid as well (`q51-grid.bin`), so every card count is scored on its own layers.
- **Specialists (`--cards`):** the same architecture, warm start and data, but trained only on the standard splits of a
  few card counts (the even split; for nine cards also 6/6/6/6/6/6/6/5/2). They show what specialization buys.
- **Validation and checkpoint:** the six-card windows plus the staircases of the head's card counts (6 and 9 for the
  general head). The best checkpoint maximizes the mean over those staircases.
- **Round of 2026-10-25 06:00–08:15** (the owner's plan, 01:45). Generation and the dump stop at 06:00; data is d1–d4,
  the same for all four heads:

  | Run | Trained on | GPU |
  |---|---|---|
  | R2-gen | 3–12 cards, weights above | RTX 3090 |
  | R2-69 | the six-card split and both nine-card splits | RTX 3090 |
  | R2-9 | both nine-card splits | RTX 3090 |
  | R2-6 | the six-card split | V100 (FP16 with loss scaling) |

  All four are scored at 6 and 9 cards (both nine-card splits, staircase and after-miss); R2-gen also at 3, 4, 5, 7, 8,
  10 and 12. If a specialist beats the general head by more than ~0.01 at nine cards, per-card-count heads (~0.29 GB
  each) ship next to the general one.
- **Scoring:** per card count, 3, 4, 5, 6, 7, 8, 9, 10, 12; nine cards reported first (the reference rig runs nine).
- **Checks before a long run:** `check_general.py` (valid windows, the card-count mix, the specialists' windows, and a
  general head loaded from a six-card head reproducing it on six-card windows).

## 4. Why the card count matters

The split decides what the head can see and how many guesses can pay. A head trained on one split fails on others (M0:
0.06 on unseen window shapes), so **the window shape must come from the run's own split**, in training and in the
engine.

| Cards | Split (layers per card) | Card 1's output: newest token's depth | Guesses in flight | Est. effect |
|---|---|---|---|---|
| 3 | 17/16/16 | ~L16–17 | ~2 | good inputs, but the cycle is ~40 % of a ~24 ms token: pipelined speculation caps at ~1.4–1.6× even at p 0.6 |
| 4 | 13/12/12/12 | ~L12–13 | ~3 | |
| 6 | 9/8/8/8/8/8 | L9 | 3–4 (S_eff 4.2) | M0 and M1 windows |
| 9 | 6/6/6/6/6/6/6/5/2 (`-ts 6,6,6,6,6,6,6,5,2`) | ~L6 | 5–8 (S_eff 5.9 even split, 8.5 light last card) | the reference rig |
| 12 | 4–5 each | ~L4 | up to ~11 | est. −0.03 to −0.06 acceptance against six cards, but the most guesses per cycle |

- **Depth of the newest token:** set by card 1's share of the 49 layers.
- **Tokens in flight:** about cards − 1, fewer when the last card (it holds the output head) is the slowest stage; S_eff
  is the sum of stage times over the slowest one ([pipelined speculation](research/notes/pipelined_speculation.md)).
- **Older tokens:** which later boundaries each has reached when the guess is made, again set by the split.
- **Which owners run which count:** Flint at ~4–4.5 bits (25–29 GB) needs 5–6 × 6 GB cards (P106-090/100), or 3–4 ×
  8–11 GB cards (P104-100, GTX 1070/1080, P102-100, GTX 1080 Ti). The hardware of the reference rig:
  [mining_rig.md](mining_rig.md).

## 5. Coupling with the sampler

The head's guess and Flint's sample are drawn from the same random numbers, so that they agree as often as possible.
All acceptance numbers so far are Gumbel-coupled; the sampler that llama.cpp serves today couples differently.

| | Gumbel coupling (offline scores) | Inverse-CDF coupling (served device sampler) |
|---|---|---|
| Flint's draw | argmax over kept tokens of log p + G, G keyed by (seed, position, token) | one uniform u per position, keyed by (seed, position); the first token, in descending order, whose cumulative probability exceeds u |
| Head's draw | argmax of α·log q + the same G | the same u over the head's own descending q^α order (`llama_sampler_pos_uniform`) |
| Match probability | Σ_i 1 / Σ_j max(p_j/p_i, q_j/q_i) | the share of u on which both pick the same token |
| Flint p = (0.6, 0.25, 0.1, 0.05), head with the top two swapped | 0.61 (from the formula) | 0.15 (synthetic check, `check_coupling.py`) |

- Inverse-CDF is fragile: any disagreement in the order of the top tokens moves every boundary after it.
- **Both couplings are scored** (`score.py`, since v2 `521737d`). The rule: if inverse-CDF loses more than ~0.02
  against Gumbel on Q51, the fork switches to a position-keyed Gumbel-max sampler.
- **Measured on Q51** (2026-10-25, α 3), inverse-CDF against Gumbel: R1 staircase 0.493 against 0.502 fresh and 0.558
  against 0.562 overall; R1 after a miss 0.531 against 0.541; U 0.530 against 0.537. The loss is 0.007–0.010, under
  the rule, so the served sampler stays as it is. A trained head mostly agrees with Flint on the order of the top tokens.
- At temperature 0, and with sampling on the host, the guess is the head's most likely token and the coupling doesn't
  matter.

## 6. Results and learning curve

Coupled acceptance with α 3; "–" is not measured. Card count is the split the windows stand for; validation is on the
row's own window.

| Date | Method, run | Data | Cards | Val | Q51 fresh | Q51 overall | Coupling |
|---|---|---|---|---|---|---|---|
| 2026-10-24 | M0, A | 3.08 M | 6 (card 1) | 0.428 at α 1 | 0.422 | 0.440 | Gumbel |
| 2026-10-24 | M0, A | 8.6 M | 6 (card 1) | 0.527 | 0.463 (0.439–0.486) | 0.505 | Gumbel |
| 2026-10-24 | M0, A, after-miss inputs | 8.6 M | 6 (card 1) | 0.565 | 0.504 | 0.553 | Gumbel |
| 2026-10-24 | M0, A, window shapes not trained (pre-step, nine-card) | 8.6 M | 6 → others | – | 0.06 | – | Gumbel |
| 2026-10-24 | M0, B (pre-step) | 8.6 M | 6 (spare card) | 0.436 | 0.380 | 0.405 | Gumbel |
| 2026-10-24 | M0, T (tokens only) | 3.08 M | any | 0.296 at α 1 | ~0.30 | – | Gumbel |
| 2026-10-24 | M1, R1 (fused, from U), 20 min | 13.5 M | 6 (card 1) | 0.538 | – | – | Gumbel |
| 2026-10-24 | M1, R1, 120 min | 13.5 M | 6 (card 1) | 0.568 | – | – | Gumbel |
| 2026-10-25 | M1, R1, 180 min | 13.5 M | 6 (card 1) | 0.580 | 0.502 (0.477–0.526) | 0.562 (0.526–0.595) | Gumbel |
| 2026-10-25 | M1, R1, 180 min | 13.5 M | 6 (card 1) | – | 0.493 | 0.558 | inverse-CDF |
| 2026-10-25 | M1, R1, after-miss inputs | 13.5 M | 6 (card 1) | 0.612 | 0.541 | 0.603 | Gumbel |
| 2026-10-25 | M1, R0 (unfused, from U), 20 min | 13.5 M | 6 (card 1) | 0.545 | – | – | Gumbel |
| 2026-10-25 | M0, U (after-miss inputs), 240 min | 13.5 M | 6 (card 1) | 0.615 (peak, 180 min) | 0.537 | 0.593 | Gumbel |
| 2026-10-25 | M0, C (card 1's output only) | 13.5 M | 6 (card 1) | – | 0.437 | 0.517 | Gumbel |
| 2026-10-25 | M0, B (pre-step / late / after miss) | 13.5 M | 6 (spare card) | – | 0.300 / 0.316 / 0.331 | 0.303 / 0.323 / 0.339 | Gumbel |
| 2026-10-25 | M0, 4-layer A, stopped at 75 of 300 min (900 steps, 0.6 epochs) | 13.5 M | 6 (card 1) | 0.385 (60 min) | 0.355 | 0.364 | Gumbel |

- **Learning curve (head A, Q51 fresh):** 0.422 (3.08 M) → 0.463 (8.6 M) with M0, ~+0.03 per doubling of data; 0.502
  (13.5 M) with M1. At +0.03 per doubling, 24 M tokens project to ~0.53, so reaching 0.55 also needs recipe gains (M2,
  regularization, a bigger head). With M0's recipe the gate projected to ~60–70 M tokens
  ([drafter acceptance](research/notes/drafter_acceptance.md) 2.4).
- **Simulated speed (2026-10-25, R1, six cards, `sim.py`):** 1.90× plain decoding (70.9 tok/s) with the copy lookup in
  front, against 1.34× (50.1) for the copy lookup with the n-gram cache, 1.44× (53.7) with the continuation table, and
  1.85× for an oracle right 50 % of the time.
- **Gate:** ≥ 0.55 on Q51 fresh text and ≥ 0.6 overall. 0.55 gives about 2× in simulation on nine cards and sits near
  the top of what partial-depth inputs give (published early-exit drafters at 6–25 % depth sit near 0.5).
- **Not a kill line** (the owner, 2026-10-24 21:25: "do not give up on this - we need to make it work"). Missing it means
  more data, a bigger head, and a retrain.
- **In front of the drafter:** the continuation store (Q161, `--pipe-store`) adds ~5 % at no card cost.

## 7. Lessons and pitfalls

| Lesson | Evidence |
|---|---|
| Data, not run length, limits the head | +0.03 per doubling; an M1 epoch of 13.5 M takes ~38 min; U overfit after ~5.6 epochs |
| Train on the windows the rig will produce, for every split it may run | M0's A: 0.06 on unseen shapes |
| Report Q51, not validation | validation ~0.06 above Q51 fresh |
| Most misses are on tokens Flint is sure of | 55.5 % of Q51's fresh positions have Flint ≥ 0.99 sure, and A gets 0.587 of them: knowledge, not calibration |
| α sharpening is worth taking, once | +0.06 on val at α 3; α 2 → 3 adds only +0.004 |
| The writer of the training text matters little; the teacher matters | the dump re-labels targets; est. 1–3 % for a non-Q8_0 writer |
| With the dump on a wide link, generation sets the data rate | dump ~1.0k answer tokens/s on three cards; generation ~0.46k on three |
| Keep a slot's KV private in generation | `-kvu` with 64–72 slots crawled |
| Tool-schema types must be JSON-schema types | 44 % of tool prompts failed before the fix |
| Keep the exported output head in F16 | all-Q8_0 moves logits 0.16 rms, F16 head 0.035 |
| The 40k vocabulary is enough for now | it holds 0.971 of Flint's kept probability on fresh text |
| Inverse-CDF coupling punishes order swaps, but trained heads rarely swap | 0.15 against Gumbel's 0.61 on a swapped top pair; on Q51 only 0.007–0.010 below Gumbel |

## 8. Open questions and next steps

1. **M2 round and per-card-count scores** (2026-10-25 06:00–08:30): does the 3–12 head lose against the 6 + 9, 9 and
   6 specialists at six and nine cards?
2. ~~Score R1 on Q51 with both couplings; decide on the Gumbel-max sampler~~: done 2026-10-25; inverse-CDF is
   0.007–0.010 below Gumbel, so the served sampler stays.
3. **The engine reads the window shape from the run's split** (`-ts` and stage times) instead of the file, and taps
   any recorded layer, for M2's layer-indexed anchors.
4. **Hidden states from the release file:** the head trains on Q8_0's states and reads the release file's on the rig;
   the size of that shift is unmeasured.
5. **Cost per guess on a P106** and whether the head fits card 1's slack on nine cards (~0.5 ms on the 6/…/5/2 split),
   or needs one layer fewer on card 1.
6. **Order of the store and the head** behind the copy lookup.
7. **More data:** generation and dump continue on every free GPU; each new size is a results row.
8. **Bigger head:** the 4-layer A (1.85× the training cost per step, est. ~1.6× the guess time) against more data.
   The night's 4-layer run is inconclusive. It and the 2-layer control were both stopped at 75 of 300 minutes, and at
   equal steps the 4-layer head was not ahead (val 0.385 at 900 steps against 0.404 at ~920). The clean test is a warm
   start: `--layers 4 --init-from` a trained 2-layer head. The new blocks' output projections start at zero, so it
   starts exactly as that head, and training shows what the extra layers add.
9. **M1's options one by one:** the LK loss, noise and dropout, against a control on the same data.
10. **The 13.5 M M0 heads:** U, C, B and the 4-layer A are scored (section 6). A and R0 ran out of memory while scoring
    on 10–12 GB cards and are being rescored.
11. **Overfitting:** with every new data size, dropout or noise against a control, once fresh data stops arriving.
12. **Generation throughput:** the two-card G3 copy is GPU-bound with its cards taking turns; `-ub 48` did not help.
    A different server (vLLM) is a candidate.

## 9. Changelog

- 2026-10-23: Q20 dropped as a batched-verify drafter (Q19: a 3-token verify costs 2.2× a plain token).
- 2026-10-24 02:18: [pipeline drafter](research/notes/pipeline_drafter.md) note; Q20 revived as a staircase head for
  pipelined speculation.
- 2026-10-24 ~09:45: the pilot starts on all seven GPUs of the quant machine.
- 2026-10-24 14:30: the pilot scales to ~24 M tokens, the extension written by G3.
- 2026-10-24: M0 results: A 0.422 (3.08 M) → 0.463 (8.6 M) on Q51 fresh text.
- 2026-10-24 21:25: the gate stops being a kill line (the owner).
- 2026-10-24 21:33: tool-call prompts from ToolACE and glaive-function-calling-v2.
- 2026-10-24 22:00: M1 (v2 `f7d15de`); R1 reaches val 0.568 at 120 min.
- 2026-10-24 22:48: top-20 on the card in the dump (`DUMP_GPU_TOPK`, v2 `d402386`); JSON-schema types for tool prompts.
- 2026-10-25 00:06: the guesser in the pipelined executor: export, checks, rig kit (v2 `97f7e97`).
- 2026-10-25 00:27: `score.py` scores the inverse-CDF coupling next to Gumbel (v2 `521737d`).
- 2026-10-25 00:30: export keeps the output head in F16 by default (v2 `6362f5d`).
- 2026-10-25 01:06: card-1 windows for 5–9-card splits in `drafter.staircase` (v2 `55a9d5c`).
- 2026-10-25 01:10: the owner requires 3–12 cards; M2 planned for 06:00.
- 2026-10-25: this document.
- 2026-10-25 01:14: first Q51 scores of M1: R1 0.502 fresh / 0.562 overall (staircase), 0.541 / 0.603 after a miss;
  inverse-CDF 0.007–0.010 below Gumbel, so the served sampler stays.
- 2026-10-25 01:35: the GPU top-k dump gives the same top-20 as the host path but runs at 0.89× with the last card on
  x8, so it is not used there. The continuous dump d4 starts with the 26-layer grid.
- 2026-10-25 01:40: simulation: R1 on six cards gives 1.90× plain decoding (70.9 tok/s).
- 2026-10-25 01:45: the 06:00 round becomes four depth-general heads (3–12, 6 + 9, 9, 6 cards) on the same data
  (the owner); `--cards`; validation on the head's own card counts; `score.py --window-group`; FP32 on CPUs.
- 2026-10-25 01:52–02:04: the two-card G3 generation copy tried with `-ub 48` (its cards take turns at ~50 % each):
  slower (~455 against ~270 ms per decode step), back to `-ub 512`.
- 2026-10-25 02:47 (CEST): vLLM writes drafter text on two RTX 3090s, the makers' NVFP4 GGUF pipeline-parallel with
  the attention patch: ~2.4–2.7k answer tokens/s, ~7× the G3 llama.cpp copy on the same cards (v2 `118a5fc`).
- 2026-10-25 02:20 (CET): the dump is now the bottleneck (~540 answer tokens/s on three cards). 9,100 undumped tool
  answers (2.54 M tokens, ~61 % prompt) held back from the d4 queue until the day, so that the night's dump spends its
  time on general answers; `feed_split.py` for several dump instances in the day.
