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
| Best held-out score | 0.463 coupled acceptance on Q51 fresh text, 0.505 overall (M0, head A, 8.6 M tokens, six cards) |
| Best validation score | 0.568 staircase (M1, run R1, 13.9 M tokens, 120 min); not yet scored on Q51 |
| Gate to the rig | ≥ 0.55 on Q51 fresh text and ≥ 0.6 overall, coupled. Not a kill line (decision 2026-10-24 21:25) |
| Next | M2: one head for any split of 3–12 cards, planned 2026-10-25 06:00 |
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
- **Measured (2026-10-24/25):** G3 on two RTX 3090s at 96 slots, one UD-Q4_K_XL copy per V100 at 48 slots; ~0.75k
  answer tokens/s in all, ~2.7 M tokens an hour. Tool prompts run at ~210 answer tokens/s (long tool lists, short
  answers).
- **Pitfalls:**
  - The generator only has to write Flint-like text: the dump re-labels every target with the Q8_0 teacher. A writer
    other than Q8_0 costs est. 1–3 % of acceptance by the literature ([data generation](research/notes/drafter_data_generation.md)),
    against ~6 % gained per doubling of data.
  - Give every slot its own KV cells (`-no-kvu`); with the unified cache 64–72 slots crawled at ~300 tok/s together.
  - 38 % of answers stop at the 2,048-token cap.

### Step 3: dump (teacher targets and hidden states), the bottleneck

- **In → out:** answers → per position: the layer inputs of a fixed layer grid and the final hidden state (FP8 E4M3,
  one scale per row) plus Flint Q8_0's top-20 next-token log-probabilities.
- **Tools:** `feats.py seqs`, then `llama-staircase-dump` (built inside a llama.cpp tree) with Flint Q8_0;
  `check_dump.py` (Flint's output head on the dumped final state gives the dumped top-1 at 98–99 % of positions).
  `DUMP_ALLPOS` keeps card 1's layers for every position, the rest for the answer only; `DUMP_OUT2` or `feats.py pack`
  writes a smaller subset for a fast disk.
- **Measured:**

  | Card set | Tokens/s (prompt + answer) | Answer tokens/s |
  |---|---|---|
  | last card on PCIe x8 | ~1,650 | ~1.1–1.2k |
  | three cards, last on x4 | ~850 | ~0.7k |
  | V100s on x1 links | ~280 | ~0.2k |

- **Storage:** 37 KB per answer token with the 17-layer grid, ~54 KB with the 26-layer grid (M2), ~12 KB for a
  six-card pack. 70 M tokens at 37 KB is ~2.6 TB.
- **Pitfalls:**
  - Without help the dump copies all 150,144 logits per position to the host (600 KB). `DUMP_GPU_TOPK=1` (with `-b`
    equal to `-ub`; v2 commit `d402386`) takes the top-20 on the card and copies ~160 B; it agrees with the host path to
    2e-6 in log-probability. Otherwise put the card holding the output head on the widest link.
  - The layer grid fixes which splits a dump can serve. Record every boundary you may need (see M2's grid).

### Step 4: pack and train

- **In → out:** dump files + vocabulary + Flint's embedding table and output head → `heads/<X>/head.pt`.
- **Tools:** `vocab.py` (40,000 most frequent answer ids plus Flint's control tokens; 32,000 gave 94 % code coverage on
  Q51), `flint_weights.py`, `train.py` (model in `drafter.py`): BF16 on RTX cards, FP16 with loss scaling on V100s,
  checkpoints every 30 min. `--resume` continues a run; `--init-from` warm-starts a changed model (weights only, new
  parameters at zero, fresh optimizer). M1 options: `--window-sample`, `--fuse`, `--loss lk --eta 3`, `--noise`,
  `--dropout`, `--val-seqs`; M2 adds `--general`.
- **Measured:** M0's head A 23–25 steps/min on an RTX 3090; M1 44 steps/min. One epoch of 13.9 M tokens is ~1,650 steps
  of ~8k targets, ~38 min under M1.
- **Pitfalls:**
  - A 10 GB RTX 3080 can run out of memory with M1's settings (R0 did, at step 1,080).
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
- **Pitfalls:** validation runs ~0.06 above Q51 fresh text (A at 8.6 M: val 0.527, Q51 fresh 0.463), so only Q51
  decides. Compare heads paired, per position, not by their single intervals (±0.024 each).

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
- **Tools:** `export_gguf.py` (RMS norms folded into the next projection; draft-id → Flint-id table; metadata: anchor
  layers, fused or not, committed anchor, staircase, α), `check_export.py`, `llama-pipe-drafter-check`.
- **Size:** Q8_0 matrices with an F16 output head (the default since v2 `6362f5d`): 0.29 GB for a two-layer fused head
  with 40,000 ids. All-Q8_0 is 0.22 GB but moves the logits by ~0.16 rms (most likely token unchanged, its probability
  by up to 0.07); the F16 head by ~0.035.
- **Pitfall:** today's file format names six-card anchors (L9, L17, L25, L33, L41, final). M2's layer-indexed anchors
  need the format and the engine extended.

### Step 8: engine

- **Where:** fork branch `pascal-p106-v2-drafter`, included in the combined release candidate `pascal-p106-v2-rc`.
  `llama-pipespec` and `llama-server` take `--pipe-guess staircase --pipe-draft FILE`.
- **How it runs** ([RIG.md](../bench/staircase_drafter/RIG.md)): every one-token pipe step copies its anchor layers'
  inputs and final state (8 KB each) to pinned host memory on the card that computes them; the head runs on card 1 in
  its own CUDA stream (or `--pipe-draft-dev N`), waits for the anchors it reads, and returns its top 16. A late anchor
  makes a guess late, never wrong. The executor still samples every token, so the output is plain decoding's.
- **Window shape:** from the file, or `--pipe-draft-stair` (nine cards today: `emb,L9,L17,L33,L41`). It must come from
  the run's own split ([section 4](#4-why-the-card-count-matters)).
- **Limits:** one sequence (no streams, races or fed tokens); the embedding table on the head's card or in host memory.

### Step 9: rig

1. Cost of one guess on one P106 (`llama-pipe-drafter-check --bench 300 --window W --ctx 2048`). Est. 1.6–2.6 ms; card
   1 has ~2 ms of slack per cycle on six cards, ~0.5 ms on the nine-card 6/…/5/2 split.
2. Pipelined decoding next to plain and the `cont` guesser (`rig_run.sh` with `DRAFT=…`), sampling on the last card at
   temperature 1, identity checked.
3. Served chat and tool requests against today's guesser (copy lookup + continuation counts + store).

## 3. Methods

| Method | Date | State | What it changed |
|---|---|---|---|
| [M0](#m0-the-pilot-2026-10-24) | 2026-10-24 | superseded by M1 (2026-10-24 22:00) | first heads: fixed six-card windows, all windows trained per sequence |
| [M1](#m1-sampled-windows-fused-anchors-warm-starts-2026-10-24-2200) | 2026-10-24 22:00 | current | one sampled window per sequence, fused anchors, warm starts; 1.8× the steps |
| [M2](#m2-depth-general-for-312-cards-planned-2026-10-25-0600) | planned 2026-10-25 06:00 | planned | layer-indexed anchors, one head for any 3–12-card split |

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
- **Runs (13.9 M tokens):**
  - R1 (fused, warm-started from U): validation staircase 0.538 at 20 min, 0.568 at 120 min, already above M0's best
    (0.527).
  - R0 (unfused control) ran out of memory on a 10 GB RTX 3080 at step 1,080.
- **Learned:** one epoch of 13.9 M tokens is ~1,650 steps (~38 min), so runs on fixed data converge in 2–3 h. Data, not
  run length, is the limit.

### M2: depth-general for 3–12 cards (planned 2026-10-25 06:00)

**Planned.** Owner requirement of 2026-10-25 01:10: the drafter must serve any split of 3–12 cards.

- **Anchors are layer-indexed:** any layer input 1–48 or the final state. Projections are shared within five depth
  buckets (layer inputs 1–12, 13–20, 21–28, 29–36, 37–48; the final state its own), plus a learned per-layer depth
  embedding. Warm-started from R1, so it starts as the six-card head on six-card depths.
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
    42, 44, 47 + final; ~54 KB per answer token.
- **Scoring:** per card count, 3, 4, 5, 6, 7, 8, 9, 10, 12; nine cards reported first (the reference rig runs nine).
- **Checks before a long run:** `check_general.py` (valid windows, the card-count mix, and a general head loaded from a
  six-card head reproducing it on six-card windows).

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
- **Both couplings are scored** (`score.py`, since v2 `521737d`). If inverse-CDF loses more than ~0.02 against Gumbel on
  Q51, the fork switches to a position-keyed Gumbel-max sampler. Plain and pipelined outputs then stay identical to
  each other; only the mapping from a seed to its answer changes.
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
| 2026-10-24 | M1, R1 (fused, from U), 20 min | 13.9 M | 6 (card 1) | 0.538 | – | – | Gumbel |
| 2026-10-24 | M1, R1, 120 min | 13.9 M | 6 (card 1) | 0.568 | – | – | Gumbel |

- **Learning curve:** ~+0.03 fresh per doubling of data (0.422 → 0.463 from 3.08 to 8.6 M). With M0's recipe the
  gate projects to ~60–70 M tokens; with M1's changes est. ~30–40 M ([drafter acceptance](research/notes/drafter_acceptance.md) 2.4).
- **Gate:** ≥ 0.55 on Q51 fresh text and ≥ 0.6 overall. 0.55 gives about 2× in simulation on nine cards and sits near
  the top of what partial-depth inputs give (published early-exit drafters at 6–25 % depth sit near 0.5).
- **Not a kill line** (the owner, 2026-10-24 21:25: "do not give up on this - we need to make it work"). Missing it means
  more data, a bigger head, and a retrain.
- **In front of the drafter:** the continuation store (Q161, `--pipe-store`) adds ~5 % at no card cost.

## 7. Lessons and pitfalls

| Lesson | Evidence |
|---|---|
| Data, not run length, limits the head | +0.03 per doubling; an M1 epoch of 13.9 M takes ~38 min |
| Train on the windows the rig will produce, for every split it may run | M0's A: 0.06 on unseen shapes |
| Report Q51, not validation | validation ~0.06 above Q51 fresh |
| Most misses are on tokens Flint is sure of | 55.5 % of Q51's fresh positions have Flint ≥ 0.99 sure, and A gets 0.587 of them: knowledge, not calibration |
| α sharpening is worth taking, once | +0.06 on val at α 3; α 2 → 3 adds only +0.004 |
| The writer of the training text matters little; the teacher matters | the dump re-labels targets; est. 1–3 % for a non-Q8_0 writer |
| The dump, not generation, sets the data rate | 600 KB of logits per position until `DUMP_GPU_TOPK` |
| Keep a slot's KV private in generation | `-kvu` with 64–72 slots crawled |
| Tool-schema types must be JSON-schema types | 44 % of tool prompts failed before the fix |
| Keep the exported output head in F16 | all-Q8_0 moves logits 0.16 rms, F16 head 0.035 |
| The 40k vocabulary is enough for now | it holds 0.971 of Flint's kept probability on fresh text |
| Inverse-CDF coupling punishes order swaps | 0.15 against Gumbel's 0.61 on a swapped top pair |

## 8. Open questions and next steps

1. **M2 run and per-card-count scores** (2026-10-25 06:00): does one head lose against six-card-only heads on six and
   nine cards?
2. **Score R1 on Q51** with both couplings; decide on the Gumbel-max sampler (the ~0.02 rule).
3. **The engine reads the window shape from the run's split** (`-ts` and stage times) instead of the file, and taps
   any recorded layer, for M2's layer-indexed anchors.
4. **Hidden states from the release file:** the head trains on Q8_0's states and reads the release file's on the rig;
   the size of that shift is unmeasured.
5. **Cost per guess on a P106** and whether the head fits card 1's slack on nine cards (~0.5 ms on the 6/…/5/2 split),
   or needs one layer fewer on card 1.
6. **Order of the store and the head** behind the copy lookup.
7. **More data:** generation and dump continue on every free GPU; each new size is a results row.
8. **Bigger head:** the 4-layer A (1.85× the training cost per step, est. ~1.6× the guess time) against more data.
9. **M1's options one by one:** the LK loss, noise and dropout, against a control on the same data.
10. **The 13.94 M M0 heads** (A, 4-layer A, U, C, B, 300-minute budgets): their final Q51 scores belong in section 6.

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
