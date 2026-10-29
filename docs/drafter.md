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

| Where we stand (2026-10-25, drafter work stopped) | |
|---|---|
| Status | **Stopped for the v2.0 release** (the owner: the release is the llama.cpp fork's improvements and our model, nothing else). No drafter or store work runs, and nothing was generated, dumped or trained for v2.1 |
| Data (the owner, 2026-10-25 15:00) | **All drafter data so far is excluded**: every prompt file, answer, dump, pack, vocabulary, head, checkpoint and store, including the "clean" M3 sets, which were cut from mixed-source files. It is the owner's to delete. Every head so far (M0's A, B, C, U, T and the 4-layer A; R0, R1; M2's R2-gen, R2-69, R2-9; R3) and the continuation store were trained or built partly on ShareGPT, UltraChat and Magicoder prompts; their numbers stand as measurements of the method only |
| v2.1 plan (M4) | start from scratch on approved sources only: [`sources.json`](../bench/staircase_drafter/sources.json) lists them, and the gate [`check_sources.py`](../bench/staircase_drafter/check_sources.py) checks every stage (prompts, answers, dumps, then vocabulary, packs, heads and the store) with a chain of hashed gate records. Both are drafts: written, never run |
| Served on the rig | rc4, K2h-r16, llama-server at depth 3, nine cards 6,…,5,2: **50.4 tok/s** (median of 8 requests) with the store and R2-9, against 48.6 with the store alone and 33.4 with no guessing: the head adds 3.7 % (chat and code at T 0 8.6 %). Six cards with R2-gen: 49.5 / 49.0 / 34.3. Every answer identical in every configuration (section 6) |
| Answer store (2026-10-27) | rebuilt from the 1,177 gated sessions of agentic data run 1 (15.4 M tokens of Flint's own output, 124 MB, publishable): **+1.4 % per generated token (95 % interval 0.4–1.9 %) on the agent check**, Flint Q8_0 on the release line, measured by replaying the check's own requests to both setups on identical tokens (2026-10-28); the live arms (46.9 against 46.5 and 45.0 tok/s) could not separate it, and the model's +4.3 % on held-out sessions was ~3× too high. Ships with v2.1. No guess costs card time, so it pays where the head (M4's first heads: 29.7 / 30.7) does not yet ([answer store](benchmarks/2026-10-27-answer-store.md)) |
| Cost per guess | 4.7 ms on one P106 at the server's ring with v2.0 (5.3–7.3 ms at 8,192 positions), on the card that also runs Flint's first stage; **2.93 ms** with the cheap-guess options (fork branch `pascal-p106-v3-cheapguess`), under the simulated break-even of ~3 ms; in the served pipeline it costs more: on the agent check's paired replay (2026-10-28) the overnight head, right 24.8 % of the time, makes the release line 3.2 % slower (section 6, "A cheaper guess") |
| Release split | nine cards with a light last card, `-ts 6,6,6,6,6,6,6,5,2` (g9light): the most tokens in flight |
| Best on the release split | M2 run R2-9 (nine-card specialist, 27.6 M tokens): 0.476 coupled acceptance on Q51 fresh text, 0.538 overall; even nine-card split 0.492 / 0.553 |
| Best on six cards | M2 runs R2-69 / R2-gen: 0.525 / 0.584 and 0.523 / 0.583 (R1: 0.502 / 0.562) |
| One head for 3–12 cards | R2-gen: from 0.556 / 0.624 at three cards to 0.449 / 0.508 at twelve (section 6) |
| Simulated speed | R1 on six cards: 1.90× plain decoding (70.9 tok/s), against 1.34× for the copy lookup with the n-gram cache |
| Gate | the release needs a rig-measured win on the release split (the owner, 2026-10-25 08:15). The offline bar was ≥ 0.55 on Q51 fresh text and ≥ 0.6 overall on six cards |
| Next | for v2.1: M4 (section 3), then a cheaper guess (section 8). The M3 redo (R4 heads) never started |
| Engine | the drafter code is in the release engine, `37e717ff6` (rc4 with the release defaults; from `460160e93` on): `--pipe-guess staircase --pipe-draft FILE` (with `--pipe-store`, the store before the head), window from the run's split, six-anchor and depth-general heads, exact against plain decoding on the rig. Without a head file it is off: no taps, no memory, the default guesser `cont` unchanged |
| Release (the owner, 2026-10-25 15:20) | ships without the answer store and without a drafter head: the heads and store so far were trained or built on excluded data. The served release numbers use the built-in guesser (`cont`: copy lookup and the request's own continuations, no data), measured by `results/2026-10-25_release_rig/served_262k.sh` in the private companion repo (no guessing against `--pipe-depth 3`, Q8_0, K2h-r16 and UD-Q4_K_XL at 262,144 tokens). The drafter returns with heads trained on the strict data list (M3) |

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
- **Tools:** `clean_prompts.py` (code first, clean by construction: implement-this-function stubs, whole-file edits
  and coding-agent turns on the standard library, llama.cpp's own code and permissive packages; drops prompts that
  share 12 words with Q51), `prompts.py` (task texts; OpenR1-Math problems, pending the owner on maths; the first
  runs' sets, now excluded, listed as `EXCLUDED_SETS`), `tool_prompts.py` (ToolACE, glaive v2; waits for a decision).
- **Sources:** only what can be published (M3). The first runs' ShareGPT, UltraChat, Magicoder, unfiltered NuminaMath
  and Hermes prompts are excluded.
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
  The cache holds 204,344 tokens (9.94 GiB per card next to 11.5 / 12.1 GB of weights, layers 25/24).
  Its answers have G3's shape on the same prompt family: 63 % end at an end token, mean 1,182 tokens (G3: 63 %,
  1,217). At T 0 its answers to six prompts match llama-server's UD-Q4_K_XL answers for their first 32, 17, 7, 103, 4
  and 6 tokens, then go their own way: coherent Flint text, not the same tokens.
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
  six-card pack. 70 M tokens at 37 KB is ~2.6 TB. Measured: the first dump (2,618 answers, 3.08 M answer tokens) took
  117 GB; a later one wrote 171 GB in 50 minutes (57 MB/s of features, 13 MB/s of pack); the first round's dumps and
  packs, with archive copies, came to 4.4 TB.
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
  - Memory-mapped feature files on an HDD with a large read-ahead (8 MB here) read megabytes for every small access.
    Every `train.py` start scans all record headers: on 2026-10-25 three trainers read ~71 GB each in 30 minutes
    before their first step. `feats.py` therefore maps with `MADV_RANDOM` and reads a batch's rows with exact reads
    (`Record.read_rows`, v2 `617eb3c`); a header scan then costs one page per record.

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
  | `recorded` (depths trained on) | 9, 17, 25, 33, 41 | the 26-layer grid: 4, 6, 8, 9, 10, 12, …, 44, 47 |
  | `timing` | `split_depths`' stage model: 0.435 ms per layer at six cards, +0.024 per extra card (min 0.3), head 2.5 ms, hop 0.3 ms | same |
  | others | sizes, `fuse` (+ `fuse_depths`), `committed`, `alpha`, `general`, `target_layers` = 49 | same |

  `export_gguf.py` reads `general` from the run's config and writes the grid as `recorded` (`--recorded` for another
  set). The grid, not the layers every dump shares: each training record's window reads its own dump's layers, and the
  grid dumps cover all 26 (`train.py`'s common set only builds the validation windows).
- **Size:** Q8_0 matrices with an F16 output head (the default since v2 `6362f5d`): 0.29 GB for a two-layer fused head
  with 40,000 ids, 0.27 GB unfused. All-Q8_0 is 0.22 GB but moves the logits by ~0.16 rms (most likely token unchanged,
  its probability by up to 0.07); the F16 head by ~0.035.

### Step 8: engine

- **Where:** fork branch `pascal-p106-v2-drafter` and the release candidate `pascal-p106-v2-rc4` (`460160e93`).
  `llama-pipespec` and `llama-server` take `--pipe-guess staircase --pipe-draft FILE`. In llama-server with a
  continuation store (`--pipe-store`) the order is the copy lookup, the request's continuations of 2-token or longer
  contexts, the store, then the head. The store's thresholds are compiled in: the longest context it holds, up to 32
  tokens, at least 2 tokens long, and its top continuation at least 30 % of the context's occurrences. On held-out
  agent sessions they tie for best of 25 threshold pairs, and the store before the request's 2-token contexts is
  worse ([answer store](benchmarks/2026-10-27-answer-store.md)).
- **How it runs** ([RIG.md](../bench/staircase_drafter/RIG.md)): every one-token pipe step copies the inputs of the
  window's layers and its final state (8 KB each) to pinned host memory on the card that computes them, one per card;
  the head runs on card 1 in its own CUDA stream (or `--pipe-draft-dev N`), waits for the hidden states it reads, and
  returns its top 16. A late hidden state makes a guess late, never wrong. The executor still samples every token, so
  the output is plain decoding's.
- **Window shape, from the run's split** (since fork `pascal-p106-v2-drafter` at 2026-10-25 03:00): the engine reads
  the layers per card from the actual device assignment, runs `split_depths` with the file's timing (kind `stair` on a
  card that holds layers, `pre` on a spare card; after a miss the one-row window falls out), and maps each depth to the
  deepest recorded depth at or below it. The window ends at the first token that has passed the last card, so every
  older token in flight is complete and reads like a committed row: its final state. Six-anchor heads go through the
  same rule with `recorded` = their five anchors, which reproduces their six-card staircase L9, L17, L33, L41. Examples
  with M2's grid:

  | Split | Window (newest first) | Older tokens |
  |---|---|---|
  | 17/16/16 | L17, L33 | final state |
  | 13/12/12/12 | L12, L25, L37 | final state |
  | 9/8/8/8/8/8 | L9, L17, L33, L41 | final state |
  | 9/8/8/8/8/8, spare card | emb, L9, L25, L33, L41 | final state |
  | 6/6/6/6/5/5/5/5/5 | L6, L12, L24, L34, L39, L44 | final state |
  | 6/6/6/6/6/6/6/5/2 | L6, L12, L18, L24, L30, L36, L42, L47 | final state |
  | 5/4/…/4 (12 cards) | L4, L9, L17, L25, L33, L41, L44 | final state |

  `--pipe-draft-stair` overrides the window, `--pipe-draft-kind stair|pre` the start.
- **Depth:** the window matches the tokens a busy pipeline holds (Q87 measured 4.3 on six cards, 5.9 and 8.2 on the two
  nine-card splits, against windows of 4, 6 and 8 rows), so up to a `--pipe-depth` of the window plus one every state a
  guess reads exists at launch. Every row reads its own token's state, never a stand-in; at a larger depth the guess
  waits for the oldest tokens to leave the last card, and the executor slows to the pipeline's rate.
- **Coupling:** inverse-CDF with the served sampler's own uniform ([section 5](#5-coupling-with-the-sampler)).
- **Guess order and cost options** (fork branch `pascal-p106-v3-cheapguess`, from `5462d458a`): with or without a store
  the head comes after the copy lookup and the request's continuations of 2-token or longer contexts, in place of
  those of a 1-token context. The last block runs the output row only (default). `LLAMA_DRAFT_HEAD=q8_0` converts the
  output layer at load, `GGML_CUDA_MMVQ_MAX_COLS=4` sends the 8-row products to MMQ; `LLAMA_DRAFT_VOCAB`,
  `LLAMA_DRAFT_SPLIT` and `LLAMA_DRAFT_ANC` are the options that measured slower or cost acceptance
  ([A cheaper guess](#a-cheaper-guess-2026-10-27)).
- **Limits:** one sequence (no streams, races or fed tokens); the embedding table on the head's card or in host memory.

### Step 9: rig

1. **Cost of one guess** on one P106 with nothing else on it: `llama-pipe-drafter-check drafter.gguf --bench 300
   --split 6,6,6,6,6,6,6,5,2 --commits 1 --ctx 2048` (and the other splits, `--ctx 8192`). `--ctx 2048` gives the
   server's default ring (4,352 cells), `GGML_CUDA_PROFILE=<file>` a per-kernel trace. Measured 5.3–7.3 ms (section 6)
   against an estimate of 1.6–2.6 ms; 4.73 ms on v2.0 at the server's ring, 2.93 ms with the cheap-guess options
   ([A cheaper guess](#a-cheaper-guess-2026-10-27)).
2. **Served:** [rig_server.sh](../bench/pipespec/rig_server.sh) per split with `DEPTHS='0 3s 3ds' STORE=… DRAFT=…`,
   then staircase alone (`DEPTHS=3d APPEND=1`) on the first split: no guessing, the copy lookup + continuations + store
   (`cont`), the same then the head, and the copy lookup then the head; chat, code and edit (512 tokens) and a
   coding-agent tool request (1,536 tokens) at T 0 and at the makers' sampling; identity per request. Nine cards
   6,…,5,2 and 6,6,6,6,5,5,5,5,5 with R2-9, six cards with R2-gen. Run on 2026-10-25 (section 6).
3. Pipelined decoding without the server (`rig_run.sh` with `DRAFT=…`) and the head on a spare card: not run; the
   served test answers the release question.

## 3. Methods

| Method | Date | State | What it changed |
|---|---|---|---|
| [M0](#m0-the-pilot-2026-10-24) | 2026-10-24 | superseded by M1 (2026-10-24 22:00) | first heads: fixed six-card windows, all windows trained per sequence |
| [M1](#m1-sampled-windows-fused-anchors-warm-starts-2026-10-24-2200) | 2026-10-24 22:00 | current | one sampled window per sequence, fused anchors, warm starts; 1.8× the steps |
| [M2](#m2-depth-general-for-312-cards-2026-10-25-0600) | 2026-10-25 06:00 | method current; its heads trained on excluded data, not for release | layer-indexed anchors, one head for any 3–12-card split; specialists on a few card counts for comparison |
| [M3](#m3-licence-clean-data-2026-10-25-1340) | 2026-10-25 13:40 | stopped, superseded by M4 (2026-10-25 15:00) | M2's method on licence-clean data only, from scratch (no weights from earlier heads), clean vocabulary |
| [M4](#m4-from-scratch-on-approved-sources-planned-for-v21) | 2026-10-26 | data running | M2's method from scratch on agent sessions from approved sources only, every stage gated; trained on generated spans, with the engine's attention window |

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

### M2: depth-general for 3–12 cards (2026-10-25 06:00)

**Current.** Owner requirement of 2026-10-25 01:10: the drafter must serve any split of 3–12 cards. Code: v2
`d8131c0`, `ba83692` (split tokens), `617eb3c` (feature-file reads).

- **Result of the first round** (27.6 M answer tokens, 134 min per head, warm-started from R1): one general head is
  within 0.008 of the nine-card specialist on the release split. Paired over Q51's positions, R2-9 − R2-gen is
  +0.0076 at g9light (CI +0.006 … +0.010) and +0.004 on the even nine-card split. The specialist loses 0.015 at six
  cards, where it wasn't trained. Twice R1's data gave +0.021 at six cards.

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

### M3: licence-clean data (2026-10-25 13:40)

**Stopped, superseded by M4.** The owner, 2026-10-25 13:40: only data that can be published; every earlier head is
not for release. At 15:00 the owner went further: all of this data is excluded too and the work starts from scratch
(M4). The R4 round below never started. The section stays as the record of how the sources were judged. Lists and
counts: [`count_strict.py`](data/2026-10-24_q20_drafter/run/count_strict.py) (one place for every rule below); the
audit's other scripts and results: [docs/data/2026-10-24_q20_drafter](data/2026-10-24_q20_drafter/README.md).

| Source | Status | Answers | Answer tokens |
|---|---|---|---|
| Our own edit and agent prompts on the Python standard library | clean by construction (core) | 2,267 | 2.19 M |
| … on llama.cpp's own code (not `vendor/` or build folders) | core | 7,061 | 6.34 M |
| … on permissive site-packages (MIT, BSD, Apache-2.0, PSF; the owner's ruling 14:05) | core | 5,894 | 5.61 M |
| OpenR1-Math-220k default (crawled competition problems) | clean as used; maths waits for the owner | 2,523 | 3.77 M |
| NuminaMath-CoT (~45 % of rows written by GPT-4) | out unless filtered to crawled sources and signed off | 3,202 | 3.75 M |
| ToolACE, glaive-function-calling-v2, Hermes function calling (LLM-written, generator not named) | pending, likely out | 19,319 | 5.31 M |
| Prompts sharing a 12-word span with a Q51 test prompt (the edit task texts are Q51's) | out (keeps Q51 honest) | 1,585 | 2.99 M |
| llama.cpp `vendor/` and build folders; tqdm (MPL), pip's vendored certifi (MPL) and resolvelib (ISC), two vendored Arrow headers (CC0, unclear) | out | 337 | 0.33 M |
| ShareGPT, UltraChat, both Magicoder sets (ShareGPT alone: 8,865 answers, 12.62 M) | out | 29,523 | 44.37 M |

- **Core:** 15,222 answers, 14.14 M answer tokens (edit 12.59 M, agent 1.55 M); no chat and no code-from-instructions
  yet. Answers to excluded prompts stay in the feature files; `train.py --only <ids>` uses only the listed ones.
- **No inheritance:** the heads start from Flint's own weights (logit-lens projections, Flint's output rows), not from
  R1 or R2; the vocabulary is counted on the core only (`vocab.py --only`: 39,496 ids, Q51 coverage 0.956).
- **Code first (the owner, 14:10):** heads are picked on Q51's code, edit and agent text on the release split, chat
  reported apart; once other data joins, edit, agent and code answers are drawn twice per epoch (`--rec-weights`).
- **New clean text:** `clean_prompts.py` (implement-this-function stubs from documented standard-library and
  permissive-package functions, whole-file edits and agent turns on the allowed trees, no maths, nothing sharing 12
  words with Q51). vLLM writes the first set on two RTX 3090s: 26,172 prompts (7,266 code, 4,955 edit, 13,951 agent),
  ~19 M answer tokens expected. Generated agent sessions use our own tool schema and system texts.
- **Runs (R4):** a nine-card specialist weighted to the release split, a 3–12-card head, a copy-weighted and an LK
  variant and a second seed, from scratch on the core for 4 hours. The new code answers join a retrain (R5) with a
  vocabulary recounted on core plus code.

### M4: from scratch on approved sources (planned for v2.1)

**Data running; training prepared.** The owner, 2026-10-25 15:05: new data for the guesser and the head, from
approved sources only, and every piece of data checked for whether it can be used and published. Drafter work stopped
for the v2.0 release; the agent-session data started on 2026-10-26, and the head trains on it when it ends.

- **Approved sources** ([`sources.json`](../bench/staircase_drafter/sources.json), one entry per source with its
  licence, a link to it, who wrote the texts, the filter and the owner's approval date):
  - our own task texts, system prompts and tool schema (MIT);
  - the Python 3.12 standard library (PSF-2.0);
  - llama.cpp's own code at one commit, without `vendor/` and build folders (MIT, a few files Apache-2.0);
  - the permissive site-packages from a pinned install ([`requirements-prompts.txt`](../bench/staircase_drafter/requirements-prompts.txt));
    of the vendored code, only the sub-packages judged one by one;
  - OpenR1-Math-220k default (the problems only);
  - OpenAssistant/oasst2 root prompter messages (not synthetic, not deleted, passed review).

  Writers: Flint only (the makers' NVFP4 GGUF on vLLM, Unsloth's Q8_0, our own quants), recorded per answer. The file
  also lists what is excluded and why. Code-generation datasets wait for a research note and the owner's decision.
- **The gate** ([`check_sources.py`](../bench/staircase_drafter/check_sources.py)) runs after every stage, and the next
  stage starts only on PASS. For each run it writes a manifest (one row per prompt, answer or dumped record) and a
  gate record holding the files' hashes, `sources.json`'s hash and the sha256 of each parent gate record.
  - Prompts: each row names its source, the file or data set row must exist and carry exactly the prompt's text, and
    no prompt shares 12 words with Q51.
  - Answers: the prompt passed, the writer is approved, the prompt tokens are the prompt's own, and the decoded answer
    shares no 12 words with Q51.
  - Dumps: each record equals a passed answer token for token.
  - Vocabulary, packs, heads and the store: every file the producing command reads must trace back to a passed
    parent, an approved Flint file or a test file named as such.
- **Agent sessions** (2026-10-26, [agentic data pipeline](research/notes/agentic_data_pipeline.md),
  [bench/agentic_data](../bench/agentic_data/README.md)): our own agent loop in network-less sandboxes on code bases
  scanned file by file, harness texts written by Flint (the earlier task texts were drafted by an AI assistant and
  shared Q51's edit instructions), exact token ids, one sequence per session with its generated spans for the dump,
  per-turn rows for the store, gated at every stage. A 72-minute pilot (Flint Q8_0 on two V100s) made 22 sessions,
  235k generated tokens, through every gate into a store and the dump's sequence file; not for training until the
  owner approves the texts.
- **Long run 1** (2026-10-26 21:28): three writers on one task list, Flint Q8_0 in llama-server on two V100s and the
  makers' NVFP4 GGUF in vLLM on two sets of RTX cards; the Python standard library and Elixir's library, stage 1 then
  stage 2 per writer.
- **Training on sessions** ([agentic_head.sh](../bench/staircase_drafter/agentic_head.sh)): one export of all writers'
  runs; every session of every 20th module held out for scoring; the dump replays each session once with the tool
  results, and the head trains only on Flint's generated spans (`--spans`). The head attends to the last 4,096
  committed positions, as the engine's ring does (`--ctx 4096`), and long sessions train in pieces of 4,096 target
  positions (`--chunk-targets`; pieces of 16,384 ran out of memory on 24 GB cards). Heads: a 3–12-card general head and a nine-card one, from scratch on the grid of M2.
- **Left to do:** `clean_prompts.py` writes `source_id` and `row_id` for the single-turn chat and maths parts.

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

Every head in this section was trained on excluded data (ShareGPT, UltraChat and Magicoder prompts among its sources):
not for release (section 1). The numbers measure the method.

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
| 2026-10-25 | M2, R2-9 (nine-card specialist, from R1, 134 min, RTX 3090 BF16) | 27.6 M | 9 light (6/…/5/2) | 0.571 | 0.476 (0.455–0.498) | 0.538 (0.503–0.571) | Gumbel |
| 2026-10-25 | M2, R2-9 | 27.6 M | 9 even | 0.585 | 0.492 | 0.553 | Gumbel |
| 2026-10-25 | M2, R2-9 | 27.6 M | 6 | 0.594 | 0.507 | 0.567 | Gumbel |
| 2026-10-25 | M2, R2-69 (6 + 9 cards, from R1, 134 min, V100 FP16) | 27.6 M | 9 light / 9 even / 6 | 0.565 / 0.581 / 0.615 | 0.472 / 0.490 / 0.525 | 0.532 / 0.550 / 0.584 | Gumbel |
| 2026-10-25 | M2, R2-gen (3–12 cards, from R1, 134 min, V100 FP16) | 27.6 M | 9 light / 9 even / 6 | 0.563 / 0.580 / 0.613 | 0.469 / 0.488 / 0.523 | 0.531 / 0.549 / 0.583 | Gumbel |
| 2026-10-25 | M2, R2-gen per card count: 3 / 4 / 5 / 7 / 8 / 10 / 12 | 27.6 M | 3–12 | – | 0.556 / 0.529 / 0.521 / 0.495 / 0.488 / 0.451 / 0.449 | 0.624 / 0.587 / 0.582 / 0.560 / 0.549 / 0.510 / 0.508 | Gumbel |
| 2026-10-25 | M2, all three, after a miss (6 and 9 cards) | 27.6 M | 6, 9 | 0.635–0.650 | 0.547–0.562 | 0.611–0.623 | Gumbel |
| 2026-10-25 | M2, R2-gen, release split | 27.6 M | 9 light | – | 0.458 | 0.525 | inverse-CDF |

- **Learning curve (head A, Q51 fresh):** 0.422 (3.08 M) → 0.463 (8.6 M) with M0, ~+0.03 per doubling of data; 0.502
  (13.5 M) with M1; 0.523–0.525 (27.6 M) with M2 on the same six-card staircase, +0.021 for one doubling. At that
  rate ~50 M tokens give ~0.54 and ~75 M ~0.55 on six cards; the release split sits ~0.05 lower (0.476 at 27.6 M).
  With M0's recipe the gate projected to ~60–70 M tokens ([drafter acceptance](research/notes/drafter_acceptance.md)
  2.4).
- **Simulated speed (2026-10-25, R1, six cards, `sim.py`):** 1.90× plain decoding (70.9 tok/s) with the copy lookup in
  front, against 1.34× (50.1) for the copy lookup with the n-gram cache, 1.44× (53.7) with the continuation table, and
  1.85× for an oracle right 50 % of the time.
- **Gate:** ≥ 0.55 on Q51 fresh text and ≥ 0.6 overall. 0.55 gives about 2× in simulation on nine cards and sits near
  the top of what partial-depth inputs give (published early-exit drafters at 6–25 % depth sit near 0.5).
- **Not a kill line** (the owner, 2026-10-24 21:25: "do not give up on this - we need to make it work"). Missing it means
  more data, a bigger head, and a retrain.
- **In front of the drafter:** the continuation store (Q161, `--pipe-store`) adds ~5 % at no card cost. The store
  rebuilt from agent sessions (2026-10-27) adds 1.4 % (0.4–1.9 %) on the agent check, measured on identical tokens
  (+4.3 % in the model on held-out agent sessions; [answer store](benchmarks/2026-10-27-answer-store.md)).

### Served on the rig (2026-10-25)

R2-9, R2-gen and the 13.8 M-token continuation store were trained or built on excluded data: not for release; the
numbers measure the method and the engine. Per-request tables, the exactness set and the rig scripts:
[docs/data/2026-10-24_q20_d3](data/2026-10-24_q20_d3/README.md).

rc4 `460160e93`, K2h-r16, llama-server `--pipe-depth 3 -c 8192 -bs`, one request at a time, prompt caching off; chat,
code and edit answers of 512 tokens and a coding-agent tool request of 1,536 tokens, at T 0 and at the makers' sampling
(T 1, top-k 20, top-p 0.95, min-p 0, seed 1). Generation tok/s from the server's timings; "store" = the copy lookup, the
request's continuations and the store (`cont` with `--pipe-store`); "head" = `--pipe-guess staircase --pipe-draft`.

| Cards (layers per card as assigned) | Guessers | chat T0 | code T0 | edit T0 | tool T0 | chat T1 | code T1 | edit T1 | tool T1 | Median |
|---|---|---|---|---|---|---|---|---|---|---|
| 9, `-ts 6,…,5,2` (7/6/6/6/6/6/6/5/1) | none | 33.3 | 33.6 | 27.2 | 33.6 | 33.8 | 33.9 | 27.3 | 32.6 | 33.4 |
| | store | 41.8 | 50.1 | 49.3 | 49.3 | 46.5 | 51.2 | 46.6 | 48.0 | 48.6 |
| | store, then R2-9 | 45.4 | 54.5 | 50.7 | 50.7 | 48.3 | 52.0 | 47.0 | 50.2 | **50.4** |
| | copy lookup, then R2-9 | 42.6 | 46.3 | 44.9 | 46.9 | 40.4 | 43.9 | 41.4 | 46.0 | 44.4 |
| 9, `-ts 6,6,6,6,5,5,5,5,5` (7/6/6/6/5/5/5/5/4) | none | 33.4 | 33.7 | 27.4 | 33.5 | 33.8 | 33.9 | 27.4 | 32.2 | 33.4 |
| | store | 40.9 | 49.2 | 49.2 | 49.2 | 46.9 | 51.3 | 46.7 | 48.0 | 48.6 |
| | store, then R2-9 | 44.6 | 54.0 | 50.7 | 50.6 | 47.8 | 51.8 | 46.5 | 50.1 | **50.3** |
| 6, llama's split (9/8/8/9/8/7) | none | 34.4 | 34.3 | 28.9 | 34.5 | 34.7 | 34.5 | 28.7 | 32.8 | 34.3 |
| | store | 43.9 | 50.8 | 49.5 | 50.0 | 47.9 | 52.1 | 46.9 | 48.5 | 49.0 |
| | store, then R2-gen | 44.9 | 53.0 | 50.2 | 49.4 | 47.4 | 50.5 | 46.4 | 49.6 | **49.5** |

- **The head on top of the store:** +3.7 % median on the release split (geometric mean over the eight requests
  +4.2 %): chat and code at T 0 +8.6 %, edit and tool at T 0 +2.7–2.9 %, T 1 +0.8–4.7 %. Nine even +3.9 % (geometric
  mean). Six cards +0.4 % (−3.1 % to +4.4 %).
- **Against no guessing:** copy lookup, continuations and store 1.45×; the same, then the head, 1.51×; the copy lookup,
  then the head, 1.33× on the release split. The last one has neither the continuations nor the store, so it is not
  the head's worth without the store; that comparison (copy lookup alone against copy lookup, then the head) is the
  guess-sources test ([`rig_d3_sources.sh`](data/2026-10-24_q20_d3/run/rig_d3_sources.sh)).
- **Acceptance as served** (R2-9, release split): behind the copy lookup alone 53.8 % of its guesses are right (T 0
  55.2 %, T 1 52.6 %), the same as its offline 0.538 on Q51 overall, though it was trained on Q8_0's hidden states and
  reads K2h-r16's. Behind the store it guesses the harder positions: 47.1 % (T 0 50.9 %, T 1 43.8 %).
- **Windows:** the engine took them from the layers llama actually assigned: `-ts 6,6,6,6,6,6,6,5,2` gives
  7/6/6/6/6/6/6/5/1 and a 7-row window (L6, L12, L18, L25, L30, L37, L47), the even split 6 rows (L6, L12, L18, L30,
  L39, L44), six cards 4 rows (L9, L17, L25, L42).
- **Cost of one guess** (one P106, nothing else on it, median of 300): 6.20 / 7.29 ms with an 8-row window at 2,048 /
  8,192 committed positions, 5.34 / 5.90 ms with 6 rows, 5.30 / 5.86 ms with 4 rows. Card 1's own share of a token is
  ~3.5 ms on nine cards (7 layers at ~0.5 ms); the head runs in its own stream on that card and competes with it. On
  six cards card 1 holds 9 layers, which fits the gain disappearing there.
- **Exactness:** every answer equal in every configuration (at T 0 to no guessing; at T 1 the pipelined configurations
  to each other, their draws keyed by position); the rig's identity step (check_identity, nine cards even and light,
  greedy and the makers' sampling) passed.

### M4 heads on the agent check (2026-10-27)

The first heads trained on publishable data: agentic run 1 (1,108 sessions, 14.3 M generated tokens), the 26-layer
grid, `--ctx 4096`, 240 minutes each (~0.9 epoch), a 3–12-card head (`gen`) and a nine-card head (`nine`), vocabulary
29,288 ids. Data, logs and scripts: [docs/data/2026-10-27_mtp_head](data/2026-10-27_mtp_head/README.md). Offline at
nine cards (coupled acceptance, α 1, all positions / fresh text):

| Head | Validation (sessions of training modules) | Held-out modules (40 sessions) | Q51 |
|---|---|---|---|
| nine | 0.365 | 0.277 / 0.235 | 0.156 / 0.141 |
| gen | 0.343 | 0.256 / 0.218 | 0.145 / 0.135 |

α 3 (the export's) raises validation by ~0.09 (nine 0.455, gen 0.437). Q51 is low because these heads saw only agent
sessions: their vocabulary misses 14 % of Q51's chat tokens and 5–7 % of the rest.

On the rig (the v2.0 engine, Flint Q8_0, nine cards, 262k, the release line `-b 512 -ub 128 -ts 6,6,5,5,6,5,5,6,6
--pipe-depth 4`, the agent check with a 25-minute budget per arm):

| Arm | Tasks passed | Minutes per task (median) | First token per turn (median) | Generation tok/s |
|---|---|---|---|---|
| release line | 12 / 12 | 1.2 | 2.30 s | **45.0** |
| + nine-card head | 10 / 10 | 2.1 | 2.19 s | 29.7 |
| + 3–12-card head | 8 / 9 (one session ran to its time cap) | 1.5 | 2.78 s | 30.7 |

One guess costs 5.3 ms (both heads, median of 300, one P106, the release split's 6-row window at 8,192 positions; 4.7
ms at the server's 4,096-position ring). Two things made the heads slow the agent (next section):
- **The cost:** at ~5 ms a guess runs past card 1's slack and holds up the next token; the simulator puts the cliff at
  ~3 ms.
- **The guess order:** without a store, `--pipe-guess staircase` guessed with the copy lookup and then the head, so the
  head took the positions where the release line guesses from the request's own continuations. As served, those
  continuations were right 31.1 % of the time; the head 20.1 % (nine-card head, 26,190 checked guesses) and 19.3 %
  (3–12-card head, 17,937), below their offline 0.26–0.28 on held-out sessions.

The overnight continuation of both heads stopped after an hour, so the 240-minute heads stand.

### A cheaper guess (2026-10-27)

Where one guess of the nine-card head goes on a P106, what a guess may cost before the head stops paying, and the
changes of fork branch `pascal-p106-v3-cheapguess`. Data, traces and scripts:
[docs/data/2026-10-27_cheap_guess](data/2026-10-27_cheap_guess/README.md).

**Profile** (`llama-pipe-drafter-check --bench`, one P106, the release split's window: 6 tokens in flight and 1 commit
in an 8-row graph, 4,352 cells = llama-server's default 4,096-position ring; the fork's per-kernel trace
`GGML_CUDA_PROFILE` and per-evaluation trace `GGML_CUDA_PROFILE_HOST`, since nvprof and nsys refuse the P106):

| Part of one guess (v2.0) | ms | |
|---|---|---|
| Output layer, F16, 29,289 × 2,048 (120 MB) | 1.46 | 82 GB/s: a third of the guess |
| Input and layer products, 8 rows each (Q8_0, llama.cpp's 8-column kernel) | ~1.75 | ~49 GB/s |
| Attention, 8 rows over 4,352 cells, both layers | 0.67 | 1.2 at 8,448 cells |
| Depth projections (one per row, `mul_mat_id` through MMQ) | 0.36 | |
| Norms, masks, ropes, softmax, top-16, and the trace's own events | ~0.5 | |
| GPU, one guess | 4.36 | the CUDA graph is in use: 0.06 ms of launching |
| Uploads (7 hidden-state rows of 8 KB over PCIe 1.0 x1), host work, read-back | ~0.4 | |
| Launch to guess | **4.73** | 5.32 at 8,448 cells; 4.46 after a miss (one token in flight) |

**The guess order.** Split by the length of the context that found them, the release line's continuations are right
38 % of the time from 2-token or longer contexts and 16 % from a 1-token context (the simulator below, on the agent
check's requests). Offline on the 40 held-out sessions ([source_split.py](../bench/staircase_drafter/source_split.py),
nine-card window, the served coupling):

| Position (what the lookups would guess) | Share | Head right | Lookup right |
|---|---|---|---|
| copy lookup | 0.521 | 0.310 | 0.815 |
| continuation of a 2-token or longer context | 0.246 | 0.266 | 0.434 |
| continuation of a 1-token context | 0.188 | 0.244 | 0.191 |
| none | 0.044 | 0.273 | – |

So the head goes after the copy lookup and the continuations of 2-token or longer contexts, in place of those of a
1-token context. At its new positions it is right ~3 % less often than at the old ones (est. ~0.19 served). The fork's
staircase order is now: the copy lookup, the request's continuations of 2-token or longer contexts, the store, the head.
A gate on the head's confidence adds little: the head where its top probability is at least 0.3, else the 1-token
continuation, is right 0.265 of the time against 0.244; in place of the 2-token continuations at most +0.006; the copy
lookup is never worth overriding. The head still has to run to know its confidence, so a gate saves no time.

**What a guess may cost** ([cost_sim.py](../bench/staircase_drafter/cost_sim.py): Q87's simulator with llama-server's
guess order; the head starts on card 1 when card 1 has finished the token before, holds the card for its time per
guess, and the server waits for it; on the agent check's own 204 requests, 129,890 generated tokens, rebuilt with
Flint's chat template by [agent_tokens.py](../bench/staircase_drafter/agent_tokens.py); stages from plain decoding's
34.1 ms per token on the release split, depth 4). Against the rig: no guessing 29.3 tok/s (the calibration), the release
line 42.7 (measured 44.6–46.5), the old order with a 5.3 ms guess right 20 % of the time 31.9 (measured 29.7 and 30.7).
Generation tok/s with the new order:

| ms per guess \ head right | 0.10 | 0.15 | 0.20 | 0.30 | 0.40 |
|---|---|---|---|---|---|
| 0–2 | 42.2 | 43.1 | 44.1 | 46.3 | 48.8 |
| 3 | 42.0 | 42.9 | 43.8 | 45.9 | 48.3 |
| 4 | 39.1 | 40.0 | 40.9 | 42.9 | 45.2 |
| 5.3 | 34.3 | 35.1 | 36.0 | 37.8 | 39.9 |
| release line (`cont`) | 42.7 | | | | |

- Up to ~3 ms a guess hides in card 1's slack (card 1's stage ~3.4 ms, the last card's ~5.3 ms with Flint's output
  head); above it every millisecond costs ~3–4 tok/s.
- At ≤ 3 ms the head beats the release line from ~0.13 right: +3 % at 0.20, +8 % at 0.30, +14 % at 0.40.
- A spare card (no Flint work, +0.5 ms for its inputs) helps only above 3 ms (5.3 ms at 0.20: 38.5 against 36.0).
- A shortlist of the output layer costs acceptance on agent text: the first 4,096 / 8,192 / 16,384 draft ids cover
  84.2 / 90.3 / 95.6 % of the held-out sessions' generated tokens, all 29,288 97.7 %
  ([shortlist_coverage.py](../bench/staircase_drafter/shortlist_coverage.py)).

**Changes** (fork branch `pascal-p106-v3-cheapguess` at `72552ee12`; every option checked against
`drafter.Drafter.forward` on a random-weight head of the real shape with the release split's windows, on the CPU and on
a P106). One guess at 4,352 cells, 6 tokens in flight and 1 commit, median of 300 on one P106:

| Option (environment) | What it does | ms per guess |
|---|---|---|
| none (v2.0, or `LLAMA_DRAFT_PRUNE=0`) | every row through both blocks | 4.77 |
| last block on the output row (default) | in the last block every row leaves its keys and values and only the output row goes on (exact) | 4.07 |
| + `LLAMA_DRAFT_HEAD=q8_0` | the output layer converted to Q8_0 at load: 64 MB at 138 GB/s; the logits move ~0.16 rms and the top token stays | 3.06 |
| + `GGML_CUDA_MMVQ_MAX_COLS=4` | the 8-row products through MMQ instead of llama.cpp's 8-column kernel (exact; process-wide, so Flint's 5–8-token products, only a prompt's tail, go there too) | **2.93** (2.57 after a miss, 3.31 at 8,448 cells) |
| instead: `LLAMA_DRAFT_SPLIT=4` | the 8-row products as two 4-row products for the Pascal multi-token kernels (exact) | 3.56 |
| instead: `LLAMA_DRAFT_ANC=dense` | all depth projections as one product (exact) | 3.23 (2.98 with MMQ) |
| + `LLAMA_DRAFT_VOCAB=8192` | the output layer's first 8,192 draft ids | 2.58 |

With the Q8_0 output layer and MMQ the 2.95 ms of the trace are: output layer 0.47, depth projections 0.36, attention
0.34 (8 rows) + 0.13 (the output row), the 8-row products ~0.9 (MMQ at ~60 GB/s), the last block's products on one row
~0.27, the rest ~0.5. At 2.9–3.1 ms per guess and the head right ~19 % of the time at its new positions, the
simulator gives 43.6–43.8 tok/s against the release line's 42.7: ~+2–3 %, about the spread of one agent-check run.

**On the agent check (2026-10-28), paired.** Live runs of the agent check differ by a few percent from run to run, so
the test is a replay: six recorded traces of the release line's agent check (36 requests, 20,544 generated tokens) sent
with their prompts, a fixed seed per request and their reply lengths to two servers on the cheap-guess build (Flint
Q8_0, nine cards, 262k, the release line). Pipelined decoding is exact, so both generate the same tokens (checked per
request) and the ratio of ms per generated token is the head's effect. The head is the nine-card one after 1,000
minutes of training (held-out sessions 0.366 / fresh 0.323 at α 1, was 0.277 / 0.235), with the Q8_0 output layer and
MMQ; offline it is right 0.314 of the time at the 1-token continuations' positions (was 0.244), est. ~0.25 served:

| Arm | ms per generated token | tok/s | ms per prompt token | Head guesses checked / right |
|---|---|---|---|---|
| release line (`cont`) | 22.85 | 43.8 | 5.13 | – |
| + the head | 23.59 | 42.4 | 5.46 | 7,104 / 1,761 (24.8 %) |
| ratio (95 % over traces) | **1.032** (1.028–1.047) | | 1.063 (1.030–1.152) | |

The head is right as often as predicted and adds 834 right guesses net over the 1-token continuations it replaces
(1,761 against 927), yet every generated token takes 3.2 % longer, and prompts 6 % longer. The simulator reproduces
−3.2 % only with a ~4 ms guess and ~0.75 ms more per token, or a guess of more than ~4.3 ms alone. **A guess costs
more in the server than its 2.93 ms on an idle card**: the hidden-state copies on each card's stream (8 KB per tap over
PCIe 1.0 x1, in the stream's order), the uploads of the other cards' states to card 1, and the head's kernels sharing
card 1 with Flint's are the candidates, none measured yet; the prompt side adds the all-positions final state and its
copy. The first start of the head's server ran out of host memory while converting the output layer (~424 MB at
once); the engine now converts in 8 MB pieces (`1d07352bf`).

**Where the served cost goes (Q194).** A third arm loads the head and follows the sequence, the taps of every token
and the prompt's final states, but never guesses (`LLAMA_PIPE_DRAFT_TAPS_ONLY=1`, `366478ef6`; its guesses are the
release line's, the same counts). Five of the traces, all three arms with the per-evaluation host trace
(`GGML_CUDA_PROFILE_HOST`), which on the rig's 2-core host makes every token ~2.2× slower (49.4 ms per generated token
for the release line), so the ratios below are larger than served:

| Arm against the release line | ms per generated token | ms per prompt token | GPU ms per generated token, all cards |
|---|---|---|---|
| following, no guesses | 1.036 (1.005–1.081) | 1.52 | 114.7 against 110.5 (+3.9 %) |
| the head | 1.079 (1.041–1.124) | 1.51 | 120.6 (+9.2 %) |

- **Following** costs about as much as guessing. Card 1's GPU time rises by 1.9 ms per generated token, from the
  prompt positions committed in bulk to the head (~930 extra graphs), and the last card's by 0.9 ms (the final state of
  every prompt position); the prompt side is where most of it lands.
- **Guessing:** card 1 ran 9,387 head launches of ~1.65 ms of GPU time each for 3,163 guesses that a token used
  (26.6 % right): two launches in three go to waste, because the server launches the head after every submission whose
  next position the lookups cannot guess, also when no pipe slot will take the guess. The other cards evaluate ~5 %
  more tokens (the wrong guesses).
- **What can go, in order of size:** the wasted launches (launch only when a slot will take the guess), the prompt-side
  work (only the head's 4,096 newest positions need committing; long prompts commit all), the taps' copies (on a copy
  stream instead of each card's compute stream). The simulator's ceiling with all of it gone, at the head's served
  acceptance (0.248): +5.2 % at 2.93 ms per guess, +5.6 % at 2.2 ms; at 0.30 +7.7 %.

**On three RTX 3090s (Q310, 2026-10-29).** The same paired replay on a fast layer split (Flint Q8_0 on three RTX 3090s,
`-ts 17,16,16 -c 262144 --pipe-depth 3`, the sm_86 build of the cheap-guess branch, a quiet machine): the release line
8.09 ms per generated token (124 tok/s); with the 3–12-card head 8.33 (**×1.031**, 1.020–1.038), right 29.9 % of the
time as served (1,884 of 6,309); with the nine-card head 9.04 (×1.118), right 9.6 % (its windows are nine cards'). The
same tokens in every arm. So the head slows the faster cards too: its following and guessing costs are not
P106-specific ([q310_replay.md](data/2026-10-27_cheap_guess/q310_replay.md)).

**The agent check's own text (Q384, 2026-10-29).** The six traces of the paired replay dumped like the training data
and scored offline. At the head's positions (where neither the copy lookup nor a 2-token continuation guesses) the served
head is right **0.309** at g9 and 0.287 at g9miss with the served sharpening (α 3), against **0.248** on the rig: 0.04–0.06
is lost in serving (Q195 looks for it). At α 1 the check's text scores 0.07 below the held-out sessions at the same kind of
position (0.243 against 0.314); not by language: the three JavaScript traces score highest (0.37–0.40), the Python trace
scale-generator (52 % of the head's positions) lowest (0.275). Reasoning is 87 % of the
head's positions and it is right on 0.286 of them; where Flint is nearly certain (top probability above 0.99, 43 % of
the positions) it is right 0.39–0.55; acceptance falls with context (0.348 under 4k tokens, 0.270 at 8–16k). The
1,480-minute head adds +0.005 here ([q384_agentcheck.md](data/2026-10-27_cheap_guess/q384_agentcheck.md)).

**Where the head is right (Q385, 2026-10-29).** The held-out sessions re-scored at α 1 and α 3 with Flint's top
probability and the overlap Σ min(p, q) per position. For the served head at its positions (g9): held-out 0.322 at α 1,
**0.384** at α 3; the agent check's text 0.309; served 0.248. Reasoning is 85–87 % of the head's positions in both texts;
the check's reasoning is harder (0.286 against 0.357), and it has few tool-call tokens (5 % against 32 %), where the head
does best (0.59). Half the head's positions have Flint above 0.99 sure, and the head is right on 0.52 of them; on copy
positions with Flint that sure the copy lookup is right 0.880, the head 0.448. The coupled match is 93 % of the overlap,
so the coupling leaves little: the head's distribution is what is off. Acceptance is flat across a session's requests
after the first; by context it falls from 0.62 under 4k tokens to 0.36–0.39 from 16k on. After a miss (g9miss) the head is
0.021 lower. The 1,480-minute head adds +0.009 at α 3 ([q385_heldout.md](data/2026-10-27_cheap_guess/q385_heldout.md)).

**Where this stopped (2026-10-29).**
- **Done:** the guess profiled (Q190), the break-even, guess order and confidence gate (Q191), the options (Q192:
  4.77 → 2.93 ms), the paired replay (Q193: the release line 3.2 % slower with the overnight head, 24.8 % right as
  served), where the served cost goes (Q194: half following, half guessing, two in three launches wasted), three RTX
  3090s (Q310: ×1.031), the agent check's own text offline (Q384: 0.04–0.06 lost in serving), where the head is right
  on the held-out sessions (Q385).
- **Prepared, not run:** Q371 + Q372 (`train.py --kinds`, `--source-labels` / `--source-weights`,
  `--max-steps`; labels from `source_labels.py`). The new training options were checked to compile, not run.
- **Next, in order:** Q371 + Q372 against a control (the same 1,480-minute nine-card head, the same
  `--max-steps`, kill criteria in their registry rows); the serving cuts of Q194 in the fork (launch the head only when a
  pipe slot will take its guess; commit only the head's 4,096 newest prompt positions; the taps' copies on a copy
  stream), then the paired replay again; Q195's engine gap; the kernel cuts of section 8, item 0.
- **Open:** where the 0.04–0.06 lost in serving goes (Q195); whether the serving cuts reach the simulator's ceiling (it
  was 8 points off before them).
- **Against the owner's rule** (the head continues only with a credible path to ≥ +5 % on the P106 release line within
  a week of engine work): the serving cuts are est. 3–4 days of fork work, and with all of them the simulator's ceiling
  at today's served acceptance is +5.2 %, at the line with no margin; at 0.30 served (the training changes of
  [mtp_head_research.md](research/notes/mtp_head_research.md), or Q195's engine gap closed) it is +7.7 %. On three
  RTX 3090s the head costs 3.1 % (Q310).

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
| The offline score predicts the served acceptance | R2-9 alone on the rig: 53.8 % right, against 0.538 on Q51 overall; Q8_0's hidden states in training, K2h-r16's on the rig |
| Measure the guess cost before tuning acceptance | 5.3–7.3 ms per guess on a P106 against an estimate of 1.6–2.6 ms; it runs on card 1 next to Flint's first stage and caps the gain at ~4 % on nine cards, ~0 on six |
| Take the window from the split the run gets, not the one asked for | `-ts 6,6,6,6,6,6,6,5,2` assigned 7/6/6/6/6/6/6/5/1: 7 rows, not 8 |
| Check the guesser behind every source that can sit in front of it | with the store, commits piled up between head launches and hit an out-of-bounds read (fixed in `460160e93`) that runs without a store never reached |
| A head competes with the guess it replaces, not with no guess | as served, the request's continuations were right 31 % of the time and M4's first head 20 %; split, 2-token continuations 38 %, 1-token ones 16 % |
| Charge the guess's time in the simulator | with it the simulator reproduces the rig (31.9 against 29.7–30.7 tok/s) and shows a cliff at ~3 ms per guess |
| Profile the guess before guessing at its cost | launches were 0.06 ms (the CUDA graph was in use); the F16 output layer at 82 GB/s was a third of it |
| A guess on an idle card is not a guess in the pipeline | 2.93 ms on an idle P106, yet the paired replay is 3.2 % slower with the head at the predicted 24.8 % acceptance; the simulator needs ≳ 4 ms to match |
| Test a few-percent effect paired, not with live runs | release-line live runs spread 44.6–46.5 tok/s; the paired replay's interval over six traces is 1.028–1.047 |
| Watch the host memory of load-time conversions on a 4 GB host | converting the output layer at once (~424 MB) ran the rig out of memory next to Flint's load; in 8 MB pieces it fits |

## 8. Open questions and next steps

0. **A cheaper guess** (the limit since the rig run of 2026-10-25): profiled and cut from 4.73 to 2.93 ms per guess
   on a P106 at the server's ring ([A cheaper guess](#a-cheaper-guess-2026-10-27)). The CUDA graph was already in
   use (0.06 ms of launching); the F16 output layer at 82 GB/s was a third of the guess. The simulator on the agent
   check's own requests puts the break-even at ~3 ms per guess on nine cards, and with the head after the request's
   2-token continuations expects +2–3 % at ~19 % acceptance (+8 % at 30 %). A spare card helps only above 3 ms. **On
   the agent check (paired replay, 2026-10-28) the overnight head is right 24.8 % of the time as predicted, but the
   release line is 3.2 % slower with it**: a guess costs more in the server than on an idle card. Q194 splits that
   cost about evenly between following the sequence (the prompt's bulk commits and final states, the taps) and the
   guesses, two thirds of whose launches go to waste. Next: launch only when a pipe slot will take the guess, commit
   only the head's 4,096 newest prompt positions, move the taps' copies to a copy stream; the simulator's ceiling with
   all of that gone is +5.2 % at the served acceptance of 0.248. Then the kernels, by size: an 8-column Pascal kernel for the 8-row products (~0.9 ms at ~60 GB/s), the depth
   projections as one product per projection over rows grouped by depth (0.36 ms), an attention kernel for a few rows
   (0.34 ms at 26 GB/s). More acceptance is worth more than any of these: at ≤ 3 ms, 0.30 instead of 0.20 adds ~5 %.
1. ~~M2 round and per-card-count scores~~: done 2026-10-25 09:17 (section 6); R2-9 for nine cards, R2-gen for six and
   other counts.
2. ~~Score R1 on Q51 with both couplings; decide on the Gumbel-max sampler~~: done 2026-10-25; inverse-CDF is
   0.007–0.010 below Gumbel, so the served sampler stays.
3. ~~The engine reads the window shape from the run's split and taps any recorded layer~~: done 2026-10-25 03:30
   (fork `daf310b66`); M2's export loads with no engine change ([step 8](#step-8-engine)).
4. ~~Hidden states from the release file~~: answered on the rig 2026-10-25: R2-9 alone is right 53.8 % of the time on
   K2h-r16's states, its offline 0.538 on Q8_0's.
5. ~~Cost per guess on a P106~~: 5.3–7.3 ms (section 6), far above card 1's slack; see item 0.
6. ~~Order of the store and the head~~: built as copy lookup, request continuations, store, then the head (rc4
   `5215256bd`); on the rig the head adds 3.7 % behind the store. What each source is worth on its own: item 13. On
   agent sessions (2026-10-27) the store sits right after the request's 2-token contexts: before them, 1.8 points fewer
   right guesses.
7. **More data:** stopped; v2.1 rebuilds all data from scratch through the gate (M4).
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
13. **What each guess source is worth** (owner-approved rig test, [`rig_d3_sources.sh`](data/2026-10-24_q20_d3/run/rig_d3_sources.sh)): the copy lookup alone, with
    the request's continuations, with the store; the copy lookup then the head; and all of them. It answers whether the
    head pays off without the store, and whether the store ships. On hold, with the Q8_0 served test and the placement
    test, until the head and the store are rebuilt on the strict data list (the owner, 2026-10-25 13:40); no testing or
    benchmarking of the guesser or the drafter until then. **Partly answered on the agent check (2026-10-27, Flint
    Q8_0, release line):** `cont` 45.0 tok/s (and 46.5 in the same slot as the store), `cont` with the agentic store 46.9,
    `cont` then M4's first heads 29.7 / 30.7. On identical tokens (paired replay, 2026-10-28) the store makes
    generation 1.4 % faster (0.4–1.9 %) and ships; the head behind the store waits for a cheaper guess (item 0). Offline per source on held-out agent sessions: the copy lookup guesses 52.9 % of positions
    (81.6 % right), the request's 2-token contexts 23.6 % (42.7 %), the store 13.5 % (34.9 %).

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
- 2026-10-25 02:55 (CET): engine checks on the quant machine, random heads of the real shape: the guesser in
  llama-pipespec equals plain decoding at depths 0–4 (8 runs: heads A, A fused, B; T 0 and T 1 on the device;
  embeddings on the host and on the device); llama-server `--pipe-guess staircase` at depths 3 and 5 equals the server
  without guesses on 7 of 7 requests; loading the drafter leaves the logits unchanged.
- 2026-10-25 03:30 (CET): the engine takes the window from the run's split and reads depth-indexed side files (fork
  `daf310b66`, export and checks in v2): six-anchor heads keep their six-card staircase, depth-general heads (M2) load
  with no engine change; the C++ head equals `Drafter.forward` on 3-, 4-, 6-, 9-, 9-light- and 12-card windows and a
  spare card. Rig tarball `llama.cpp-20261025-daf310b66-pascal-p106-v2-drafter`.
- 2026-10-25 03:50 (CET): two corrections from C9's review of the side-file fields: a depth-general head's `recorded`
  is the 26-layer grid (the export default), and tokens in flight older than the split's window read their final state
  like committed rows (fork `c3a324ba0`). The 14 reference cases pass again on the CPU backend; the one different top
  id, on B's first step, is a near-tie (top two logits 0.0005 apart, error 0.003).
- 2026-10-25 04:22 (CET): GPU checks of `c3a324ba0` on the quant machine: the 14 cases on the card, every top id equal;
  llama-pipespec with the guesser equals plain decoding (tokens, logits, KV state, probe) with the depth-general random
  head on three devices of one card at depths 1, 3 and 5 (tokens in flight beyond the window), and with the fused random
  head on one device (K2h-r16 alone on a 32 GB card) at depths 1 and 3, T 1 sampled on the card.
  Rig tarball `llama.cpp-20261025-c3a324ba0-pascal-p106-v2-drafter`.
- 2026-10-25 06:00–09:17 (CET): the first M2 round on 27.6 M answer tokens, three heads from R1 (R2-gen 3–12 cards,
  R2-69, R2-9), 134 minutes each; vLLM generated until 07:55. On the release split (nine cards, light last card) R2-9
  leads with 0.476 / 0.538, R2-gen 0.008 behind (paired), below the ~0.01 that would justify per-card-count heads;
  six cards 0.525 / 0.584. R2-9 and R2-gen go to the rig.
- 2026-10-25 08:00–08:15: the owner's cut-off (no more generation, one final round) and decision: the release ships
  the drafter only as a confirmed rig-measured win on the release split.
- 2026-10-25 08:20–09:56 (CET): the served guesser for the release, rc4. llama-server's staircase guesser now takes the
  store too (copy, the request's continuations, the store, then the head in place of cont's last guess; fork
  `5215256bd`). The server check found a crash in the drafter path: when the copy lookup or the store guessed a long
  stretch, the commits pending between head launches overflowed the flush loop (out-of-bounds read, segfault in
  `launch()`); fixed in `460160e93`. Then on three cards with the store and a random head: staircase with the store
  equals the no-guess server on 10 of 10 requests at depths 1 and 3 (the replay that crashed before) and on 13 of 13
  with 512-token chat, code, edit and rewrite answers (rewrites: 440–457 right copy and store guesses, 17–26 head
  launches); `rig_server.sh` with the four rig configurations identical. R2-9 and R2-gen exported (0.27 GB each).
  The rig test: `rig_server.sh` per split with no guessing, cont + store, staircase + store, staircase alone.
- 2026-10-25 11:09–12:17 (CET): first served rig result (rc4 `460160e93`, K2h-r16, depth 3). Release split: 50.4 tok/s
  median with the store and R2-9, 48.6 with the store alone, 44.4 with the copy lookup then the head, 33.4 with no
  guessing (+3.7 % for the head; chat and code at T 0 +8.6 %); nine even 50.3 / 48.6 / 33.4; six cards with R2-gen
  49.5 / 49.0 / 34.3. Every answer identical in every configuration. Behind the copy lookup alone the head is right
  53.8 % of the time, as offline. One guess costs 5.3–7.3 ms on a P106 (est. 1.6–2.6 ms), the limit now. rc4 later
  became `2d53179aa` (C7's allocator fix for a rig-only crash with packed uploads, off the drafter's path; these numbers
  stand).
- 2026-10-25 13:32 (CET): the guess-sources test prepared ([`rig_d3_sources.sh`](data/2026-10-24_q20_d3/run/rig_d3_sources.sh), owner-approved): copy lookup alone,
  with the request's continuations, with the store, the copy lookup then the head, and all of them, on the release
  split. Until it runs, the head's worth without the store is not measured.
- 2026-10-25 13:40: the owner's data rule: only publishable sources. Every head so far and the continuation store were
  trained or built partly on ShareGPT, UltraChat and Magicoder prompts: trained on excluded data, not for release.
  C9 retrains the heads and C7 rebuilds the store on the strict list; the guess-sources, Q8_0 served and placement
  rig tests wait for them, and nothing of the guesser or the drafter is tested or benchmarked until then.
- 2026-10-25 13:40–14:00 (CET): licence-clean data only (the owner). The 15:00 round on mixed data is cancelled; M3
  starts: 23.30 M clean answer tokens of 74.67 M (18.69 M dumped), a clean vocabulary, the remaining clean answers
  dumped first, R4 heads from scratch from ~15:20. `train.py`, `vocab.py`, `feed_split.py` take `--only <ids>`;
  `train.py --copy-weight` down-weights targets inside copies.
- 2026-10-25 14:05–14:15 (CET): site-packages ruling applied (+6.70 M clean, list in
  [strict-sitepkgs.md](data/2026-10-24_q161_datastore/strict/strict-sitepkgs.md)); generic tool sets held pending their audit; code first: `--rec-weights`, `clean_prompts.py`,
  vLLM generation of clean code-first prompts on GPUs 2 and 5.
- 2026-10-25 14:20–14:40 (CET): the data audit (NuminaMath ~45 % GPT-4-written; the tool sets LLM-written; OpenR1
  clean as used) and the Q51 rule: R4 trains on the core only (15,222 answers, 14.14 M: our own edit and agent
  prompts on clean trees, nothing sharing 12 words with Q51) with a vocabulary counted on it; `prompts.py` defaults to
  publishable sources, the first runs' sets kept as `EXCLUDED_SETS`; `clean_prompts.py` drops Q51-overlapping prompts.
- 2026-10-25 15:00–15:05: the owner: all drafter data so far is excluded, the work starts from scratch on approved
  sources with every stage checked (M4). All generation, dumps and training stopped; R4 never started.
  `sources.json`, `requirements-prompts.txt` and the stage gate `check_sources.py` drafted, not run.
- 2026-10-25, afternoon: the owner: the release is the llama.cpp fork's improvements and our model only. All drafter
  work stops, the M4 pipeline included; nothing was generated for v2.1.
- 2026-10-26: the data pipeline for v2.1's agentic sessions ([agentic data pipeline](research/notes/agentic_data_pipeline.md),
  `bench/agentic_data`): provenance per session, per-file licence scans, Flint-written harness texts, the pi agent's
  tool names as a profile, and a 72-minute pilot (22 sessions, every gate passed, store built, dump sequences read).
  `check_sources.py derived` takes `--tool` for programs such as the datastore. Not for training until the owner's
  approvals.
- 2026-10-27: training on agent sessions prepared while long run 1 generates: `train.py` and `score.py` read only
  the generated spans (`--spans`), attend to the engine's 4,096-position window (`--ctx`) and take long sessions in
  pieces (`--chunk-targets`); `export.py` exports several writers' runs as one; `heldout.py` holds out whole modules;
  `agentic_head.sh` runs export to GGUF. `check_ctx.py` passes on synthetic data (windowed attention 8e-7 from a dense
  reference, pieces equal to whole sessions), and a short CPU run trains through the whole loop.
- 2026-10-27: M4's first heads (agentic run 1, 240 minutes each) on the agent check: 29.7 and 30.7 tok/s against the
  release line's 45.0. One guess costs 5.3 ms on a P106, and ~25–30 % of guesses are right on agent text, so the head
  slows the agent; the cheaper guess (section 8, item 0) comes before more training pays. Heads continue overnight.
- 2026-10-27 (evening): the answer store rebuilt from agentic run 1's 1,177 gated sessions (15.4 M tokens, 124 MB;
  `chain.py` walks its 19 gate records; `heldout_check.py` finds no agent-check or Q51 text in its sessions). Offline
  on 40 held-out sessions with a store built without their modules (`store_agentic.py`, `datastore_corpus.py
  --sessions`): right guesses 56.9 → 59.6 %, +4.3 % in the pipeline model at the release split; the engine's
  thresholds tie for best; +0.4 % per doubling of the store. On the rig's agent check: 46.9 tok/s with the store
  against 45.0 (the earlier run) and 46.5 (the same slot) without; single live runs cannot separate a few percent
  ([answer store](benchmarks/2026-10-27-answer-store.md)).
- 2026-10-28: the answer store's paired replay: six of the agent check's recorded traces replayed to the release line
  with and without the store (`release_check.sh`'s replay mode; same seeds and reply lengths, identical tokens in both
  setups): 22.55 against 22.87 ms per generated token, **+1.4 % (95 % interval 0.4–1.9 %)**, prompt reading unchanged;
  5.3 % more right guesses. The store ships with v2.1; the model's +4.3 % was ~3× too high.
- 2026-10-27 (evening): a cheaper guess. The guess profiled on a P106 (4.73 ms at the server's ring; the F16 output
  layer 1.46 ms at 82 GB/s), the break-even simulated on the agent check's own requests (`cost_sim.py`: flat to ~3 ms
  per guess), the head moved after the request's 2-token continuations, and the cheap-guess options (fork branch
  `pascal-p106-v3-cheapguess`): 2.93 ms per guess ([docs/data/2026-10-27_cheap_guess](data/2026-10-27_cheap_guess/README.md)).
- 2026-10-28: the cheaper head on the agent check, as a paired replay (six recorded traces, the same tokens in both
  arms): with the overnight nine-card head (1,000 minutes) the release line is 3.2 % slower per generated token (95 %
  1.028–1.047) and 6 % slower on prompts, though the head is right 24.8 % of the time as served, as predicted. The
  guess costs more in the served pipeline than on an idle card; that cost is the next thing to measure. The head
  file is now converted at load in 8 MB pieces (`1d07352bf`): at once it ran the rig's host out of memory.
- 2026-10-29: Q310: on three RTX 3090s the 3–12-card head costs 3.1 % per generated token, the nine-card head 11.8 %.
  The heads' third round to 1,480 minutes: held-out 0.380 (nine-card), +0.027 per doubling of steps. Q384: on the
  agent check's own text the served head is right 0.29–0.31 offline at its positions as served (α 3), 0.248 on the rig;
  the check's text is harder than the held-out sessions (0.384 at the head's positions, α 3) because its reasoning is
  (0.286 against 0.357 at the head's positions), not its language. Q385: the head is right 0.52 where Flint is above 0.99
  sure (half its positions); its coupled match is 93 % of the overlap Σ min(p, q).
