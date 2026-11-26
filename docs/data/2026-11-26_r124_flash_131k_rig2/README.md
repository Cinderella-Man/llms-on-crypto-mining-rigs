# Flash-Next's v5 line at 131,072 tokens of context, and the MTP draft on against off at depth (R124, Q1403, 2026-11-26)

What this measured: whether Qwen3.8-Flash-Next's v5 preset holds a 131,072-token context on nine P106-090 cards and
the rig's 4 GB host, what it decodes and reads at 49,152, 65,536 and 120,000 tokens, and whether the MTP draft still
pays there (the draft on against off, two loads each). Registry row [Q1403](../../../quants/experiments.md), from
[Q1251](../2026-11-11_fn65_draft_yield_deep/README.md) and Q1399.

Why: the v5 preset runs at 65,536 tokens, and real coding attempts are longer. In the sister project's bracket 3 of the
14 passed real cases fit 65,536 tokens and 11 fit 131,072
([task-level review](../../research/notes/v5_review_task_level_2026-11.md)). v3.0's line held 131,072 tokens on
2026-11-10 ([Q1154, Q1218](../2026-11-10_deep_lane/README.md), folder `ctx131k/`), and with v3.0's step times the draft
was at its break-even past 64k ([Q1251](../2026-11-11_fn65_draft_yield_deep/README.md)); both had to be measured again
on v5's engine.

**Result: the line holds 131,072 tokens, and the draft is a gain in generation at every depth.**
- **It fits with one change besides the context: micro-batches of 256 instead of 512.** Card 8, which holds the
  output head and the draft, is the limit: 5,931 MiB in use after loading (137 left of the 6,068 a card offers) and
  5,999 at the highest (69 left), reached on the first requests at 49,152 tokens and unchanged up to 120,000. Every
  other card keeps 651 MiB or more. llama.cpp's own projection leaves card 8 13 MiB at the preset's `-ub 512`, so 512
  does not fit.
- **The host holds it, tightly.** With the draft the first long prompt pushes 734 and 845 MiB to swap and the host
  falls to 257 and 289 MiB available; the RAM watchdog never acted, and no load was stopped. Without the draft the
  host keeps 515 and 520 MiB and writes nothing to swap.
- **Speed with the draft (the preset at 131,072 tokens), at 49,152, 65,536 and 120,000 tokens:** 28.63 and 27.60,
  30.85 and 30.84, 23.98 and 23.99 tok/s generated (256 tokens at temperature 0, both passes); a step on the repeated
  request 112.38 and 112.54, 115.85 and 115.93, 126.16 and 126.23 ms; prompts read at 187.3 and 187.1, 128.8 and 129.1,
  94.5 and 94.3 tok/s. Reading the prompt from 65,212 to 119,936 tokens (54,724 tokens) took 579.3 and 580.4 s.
- **The draft against no draft:** generation +46.5 %, +62.7 % and +34.3 % (a gain at every depth, the plain line
  19.20 and 19.17, 18.98 and 18.95, 17.88 and 17.84 tok/s); prompt reading −19.7 %, −16.4 % and −20.1 % (a loss at
  every depth, the plain line 232.9 and 233.0, 154.1 and 154.3, 118.1 and 118.0 tok/s). A draft step costs as much as
  2.30, 2.22 and 2.27 plain tokens on the first request at a depth, and yielded 3.28, 3.61 and 3.05 greedy (3.37, 3.12
  and 3.71 with the preset's sampler, one draw a depth). In a whole turn the draft pays as long as the turn reads fewer
  than 14.98, 16.09 and 6.80 new prompt tokens per token it writes (est. 10.02, 10.15 and 4.14 at the sampler's yields
  measured off the rig, Q1251).

Against v3.0's 131,072-token line on the same prompts (S38 and R55, 2026-11-10, rig 2 at stock clocks; today rig 2 runs
R239's clock profile, so the engine and the clocks together): the draft's step on the repeat 112.38 and 112.54 ms
against 181.81 and 182.88 at 49,152, 115.85 and 115.93 against 197.12 and 197.73 at 65,536, 126.16 and 126.23 against
249.94 and 249.96 at 120,000; prompts with the draft 94.5 and 94.3 tok/s against 49.0 and 48.9 at 120,000. A draft
step on the repeat now costs 2.18, 2.21 and 2.29 plain tokens at the three depths, where it cost 2.65, 2.67 and 2.77 on
v3.0: that is why the draft pays at depth again.

What it does not settle: whether the 131,072-token line is as fast as the 65,536-token preset where agent turns mostly
are (shallow, prompts read in micro-batches of 256 instead of 512); that is the agent check of the two lines
(Q1921). The preset is unchanged.

## The line

The v5 preset ([release/flash-next-v5/settings.env](../../../release/flash-next-v5/settings.env) at `bf7982b1`:
UkisAI's Swift-1.5 GSQ-RCO Q2_0 of Qwen3.8-Flash-Next, its n-gram table cut into a third shard, the MTP draft, 40
switches) with `-c 131072` and `-ub 256`, nothing else changed (`--spec-draft-n-ubatch 512` kept). That is the preset
as it stood when the run copied it at 14:10, with `GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=5` and `_MIN_RATIO=0.75`; at 15:11
the preset took C1175's `_NSM=10 _MIN_RATIO=0.5` (`d296f5ce`, [C1175](../2026-11-26_c1175_nsm10_rig2/README.md)), which
shorten a draft step wherever the per-token sparse kernel runs (4.55 and 4.51 ms at 8,192 tokens there), so on today's
preset the draft's steps here would be est. about 4.5 ms shorter and its case a little stronger. The started lines of
every load (the first line of each server log) were checked against the preset at `bf7982b1` with
[run/line_check.sh](run/line_check.sh) (`PRESET=` that file): the environment byte for byte, the flags the preset's
with those two values, plus `-md` (draft) or without the draft's flags (no draft); the three draft loads started the
same line, and so did the two plain loads.

Engine: the `pascal-p106-v5` candidate tree `eee615cdb` (upstream b12674 merged), llama-server md5
`2727faf28b5b34dfe42399a9d431da67` (statically linked), the build C1175 ran the same day on the same rig.

Rig: rig 2, nine P106-090 cards on PCIe 1.0 x1, a 2-core Celeron with 4 GB of RAM, driver 580.178.04, every card with
R239's soaked clock profile and the overclocking kit's watchdog running (one boot, every load on it). Each load started
on a machine checked quiet (no llama process, at most 100 MiB on any card).

Proposed lines for a 131,072-token preset, with their reply-effects tags ([docs/reply_effects.md](../../reply_effects.md);
the owner decides):

```bash
PRESET_ARGS="$PRESET_ARGS -ngl 99 -fa on -c 131072 -np 1 -bs --ctx-checkpoints 4 --cache-ram 0 --jinja --reasoning-format deepseek"  # [order] the run line (the context 131,072 instead of 65,536; R124)
PRESET_ARGS="$PRESET_ARGS -b 4096 -ub 256"  # [order] the micro-batch size sets the arithmetic's order; micro-batches of 256 so that card 8 holds the 131,072-token context (R124)
```

## What ran

[run/r124.sh](run/r124.sh) on `bench/deep_lane`: a warm-up load of the draft line at depth 0 (the load check), then
`draft1`, `nodraft1`, `nodraft2`, `draft2` (the second round reversed), 14:10 to 15:36. Per load and depth: a prime
(the prompt read on top of the cache), 256 tokens at temperature 0 twice (pass 1 is the first request at a new depth,
pass 2 the same request again on a warm cache), then once with the preset's own sampler and a fixed seed. The prompts
are cut from this project's recorded agent sessions (the v3.0 release check, tags `t1swift` and `t1base`), the same
token ids as S38's and R55's at the same depths, so v3.0's line is the direct comparison. Loads took 227 s (warm-up),
225 and 224 s with the draft and 210 and 211 s without.

Before the run, llama.cpp's own projection of each card's memory (`scripts/fit_probe.sh`, nothing loaded) for four
lines: the draft at 131,072 tokens with `-ub 512` and `-ub 256`, no draft at 131,072 with `-ub 256`, and the preset as
it is at 65,536 ([fit_probe.tsv](fit_probe.tsv); rows `projected` of [card_fit.tsv](card_fit.tsv)).

## Memory

MiB in use per card (cards 0 to 8), as nvidia-smi counts it; a card offers 6,068 MiB (6,144 less the driver's 76):

| Line | When | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | Left on the fullest |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| draft, 131,072, `-ub 512` | projected | 5,223 | 5,559 | 5,265 | 5,575 | 4,531 | 5,282 | 4,849 | 5,293 | 6,010 | card 8: 58 |
| draft, 131,072, `-ub 256` | projected | 5,024 | 5,322 | 5,028 | 5,338 | 4,294 | 5,045 | 4,610 | 5,056 | 5,773 | card 8: 295 |
| draft, 131,072, `-ub 256` | after loading (all three loads) | 5,083 | 5,311 | 5,019 | 5,327 | 4,283 | 5,035 | 4,597 | 5,047 | 5,931 | card 8: 137 |
| draft, 131,072, `-ub 256` | highest in `draft1` and `draft2` | 5,195 | 5,401 | 5,117 | 5,417 | 4,371 | 5,133 | 4,671 | 5,145 | 5,999 | card 8: 69 |
| no draft, 131,072, `-ub 256` | after loading (both loads) | 5,037 | 5,277 | 4,973 | 5,293 | 4,247 | 4,989 | 4,571 | 5,001 | 2,467 | card 3: 775 |
| no draft, 131,072, `-ub 256` | highest in `nodraft1` and `nodraft2` | 5,145 | 5,365 | 5,069 | 5,381 | 4,335 | 5,085 | 4,641 | 5,095 | 2,513 | card 3: 687 |
| the preset, 65,536, `-ub 512` | projected | 4,947 | 5,138 | 4,989 | 5,154 | 4,254 | 5,005 | 4,417 | 5,016 | 5,456 | card 8: 612 |
| the preset, 65,536, `-ub 512` | after loading, C1175's loads (same rig and build) | 5,007 | 5,059 | 4,911 | 5,075 | 4,175 | 4,927 | 4,345 | 4,939 | 5,545 | card 8: 523 |

The projection's own reading (6,023 MiB free a card at the time) left card 8 13 MiB at `-ub 512` and 250 at `-ub 256`.
Card 8 held 158 MiB more after loading than projected, so at `-ub 512` it would need est. 6,168 MiB: it does not fit.
Card 8 reached its highest, 5,999 MiB, during the sampler request at 49,152 tokens and held it to the end of each load:
the growth after loading comes from the first requests, not from depth. Against the 65,536-token preset the doubled
context (with `-ub 256`) costs card 8 386 MiB and cards 1 and 3 252 each, after loading.

The host ([host_memory.tsv](host_memory.tsv), the RAM watchdog's samples, one a second):

| Load | Available after loading, MiB | Lowest available, MiB | Written to swap in the first prompt, MiB | Most swap in use, MiB | Highest pressure (some, full), % |
|---|---:|---:|---:|---:|---|
| warm (draft) | 816.5 | 252 | none (no long prompt) | 51 | 0.00, 0.00 |
| draft1 | 1,036.6 | 257 | 734 | 1,070 | 2.92, 1.23 |
| nodraft1 | 1,023.5 | 515 | 0 | 71 | 0.18, 0.18 |
| nodraft2 | 848.8 | 520 | 0 | 71 | 0.18, 0.00 |
| draft2 | 804.0 | 289 | 845 | 1,165 | 7.66, 3.30 |

The lowest point of each draft load is during the first prompt (49,028 tokens read on an empty cache); after it the
host keeps 797 MiB or more. The RAM watchdog (limits: 60 MiB available, 20 MiB usable for 2 s, 40 % pressure for
10 s, 600 MiB swapped out in 10 s) never acted, and the kernel logged no Xid on the boot.

## Speed

From [table.md](table.md) (control: no draft) and [draft_pair.md](draft_pair.md). Generation, tok/s, both passes per
load (the load's ms a token):

| Depth | draft1 | draft2 | nodraft1 | nodraft2 | The draft against none | Class |
|---:|---|---|---|---|---:|---|
| 49,152 | 28.63 (34.93) | 27.60 (36.24) | 19.20 (52.08) | 19.17 (52.16) | +46.5 % | gain |
| 65,536 | 30.85 (32.41) | 30.84 (32.42) | 18.98 (52.69) | 18.95 (52.77) | +62.7 % | gain |
| 120,000 | 23.98 (41.70) | 23.99 (41.69) | 17.88 (55.92) | 17.84 (56.06) | +34.3 % | gain |

The step (ms) and what it yields: a draft step on pass 1 and pass 2 against a plain token, and the tokens a step:

| Depth | Draft step, pass 1 | Draft step, pass 2 | Plain token, pass 1 | Plain token, pass 2 | Break-even, tokens a step (pass 1) | Yield greedy | Yield with the preset's sampler |
|---:|---|---|---|---|---:|---|---|
| 49,152 | 116.91, 125.33 | 112.38, 112.54 | 52.60, 52.69 | 51.57, 51.63 | 2.30 | 3.28, 3.28 | 3.37, 3.37 |
| 65,536 | 117.90, 117.87 | 115.85, 115.93 | 53.10, 53.18 | 52.29, 52.37 | 2.22 | 3.61, 3.61 | 3.12, 3.12 |
| 120,000 | 129.52, 129.36 | 126.16, 126.23 | 56.80, 57.05 | 55.05, 55.07 | 2.27 | 3.05, 3.05 (3.08 on pass 2) | 3.71, 3.71 |

`draft2`'s first request at 49,152 (125.33 ms a step against `draft1`'s 116.91) ran while the host was still writing
to swap after the first prompt (19 MiB in its 14 s); its repeat took 112.54 against 112.38. With the sampler the two
loads drew the same tokens (fixed seed), so each depth's sampler yield is one draw: at the yields Q1251 measured off the
rig with the preset's sampler over three tails (2.876, 2.93 and 2.69 tokens a step), the draft against none is est.
+25.0 %, +32.1 % and +18.3 % on pass 1.

Prompt reading, tok/s (the prime request; the load's ms a prompt token):

| Depth | Read, on cached | draft1 | draft2 | nodraft1 | nodraft2 | The draft against none | Class | The draft adds, ms a prompt token |
|---:|---|---|---|---|---|---:|---|---:|
| 49,152 | 49,028 on 60 | 187.3 (5.34) | 187.1 (5.35) | 232.9 (4.29) | 233.0 (4.29) | −19.7 % | loss | +1.05 |
| 65,536 | 16,644 on 48,828 | 128.8 (7.76) | 129.1 (7.75) | 154.1 (6.49) | 154.3 (6.48) | −16.4 % | loss | +1.27 |
| 120,000 | 54,724 on 65,212 | 94.5 (10.59) | 94.3 (10.61) | 118.1 (8.47) | 118.0 (8.47) | −20.1 % | loss | +2.13 |

Each prime restored the cache from a checkpoint (the model's recurrent layers keep no history), so the prompt at
65,536 read on 48,828 cached tokens and the one at 120,000 on 65,212. Getting to 120,000 tokens took 970.3 and 971.4 s
of prompt reading with the draft (the three primes, 120,396 tokens read: 261.8, 129.2 and 579.3 s in `draft1`) and
781.9 and 782.0 s without.

A whole turn ([draft_pair.md](draft_pair.md), last tables): the draft saves 15.74, 20.44 and 14.45 ms a generated
token at this run's yields and costs 1.05, 1.27 and 2.13 ms a prompt token, so it pays in a turn that reads fewer than
14.98, 16.09 and 6.80 new prompt tokens per token it writes; est. 10.02, 10.15 and 4.14 at Q1251's sampler yields.

## The replies

[replies.tsv](replies.tsv): each arm wrote the same greedy reply in both its loads and on both passes. The draft's
greedy replies part from plain decoding at token 18 (49,152: " test" where plain writes " file"), 15 (65,536: "?"
for " with") and 3 (120,000: " output" for " is"); after the parting both arms write a coherent turn about the same
files (reasoning on a truncated tool output, then a tool call reading or running the test file). Parting at a near
tie is what the preset's `[order]` lines allow; the margins at these three tokens were not measured here (Q1922). At
49,152 tokens v3.0's line wrote the same greedy reply as today's draft line in one of its two loads. The sampler pass
parts at the first token, as it must: with the draft the accepted tokens come from other random draws.

## Files

| File | What |
|---|---|
| [table.md](table.md), [classes.tsv](classes.tsv) | the deep lane's own tables and classes, control `nodraft` (the rig wrote the same tables with `draft` as control) |
| [draft_pair.md](draft_pair.md), `draft_pair_*.tsv` | the draft against no draft part by part: generation per pass, the step and its break-even, estimates at other yields, prompts, a whole turn, and v3.0's line beside it |
| [card_fit.tsv](card_fit.tsv), [fit_probe.tsv](fit_probe.tsv) | memory per card: projected, after loading, highest in each load; the projection's per-card lines |
| [host_memory.tsv](host_memory.tsv) | the host per load and per request: lowest available, swap, pressure |
| [replies.tsv](replies.tsv) | per request and load, the reply's token-id and text md5 and the first token where it differs from `nodraft1`'s |
| [requests.csv](requests.csv), [loads.tsv](loads.tsv) | the lane: one line per request (deep_client.py) and per load (its start, load time, full server line) |
| [run/r124.sh](run/r124.sh) | the run script (C1175's `run/c1175.sh` with the context, micro-batch, depths and arms of this row) |
| [run/line_check.sh](run/line_check.sh) | checks a started line against `settings.env` with `-c 131072 -ub 256` |
| [run/card_fit.py](run/card_fit.py) | the memory table (`docs/data/2026-11-10_deep_lane/run/card_fit.py` reading the newer card log) |
| [run/probe_table.py](run/probe_table.py) | the projection's per-card lines as `fit_probe.tsv`, without the server lines (they hold the rig's paths) |
| [run/tables.sh](run/tables.sh) | every table above from the collected run |

## Commands

```bash
# on a machine that runs only ssh and tar, from the repository's root; RIG, WORK and the paths are yours
git show bf7982b1:release/flash-next-v5/settings.env > preset_bf7982b1.env    # the preset as this run used it
C="-bs --jinja --reasoning-format deepseek --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 -b 4096 -ts 6,6,6,6,5,6,5,6,3 -lm dio --no-host --ctx-checkpoints-vram 4 --ctx-checkpoints-incall --ubatch-taper short=1400:64"
DR="--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-n-ubatch 512 --spec-draft-sampling probabilistic --spec-draft-p-min 0 -md <MTP draft on the rig>"
. ./preset_bf7982b1.env
RIG=rig MODEL=<first shard on the rig> BUILDS=<folder of builds on the rig> OUT=fit_probe.md KEEP=400 \
  CONFIGS="draft131k_ub512|<build>|131072|$C -ub 512 $DR|$PRESET_ENV;draft131k_ub256|<build>|131072|$C -ub 256 $DR|$PRESET_ENV;nodraft131k_ub256|<build>|131072|$C -ub 256|$PRESET_ENV;draft65k_ub512|<build>|65536|$C -ub 512 $DR|$PRESET_ENV" \
  bash scripts/fit_probe.sh
PRESET=preset_bf7982b1.env RIG=rig WORK=/tmp/r124 BIN=<llama-server on the rig> MODEL=<first shard on the rig> MODEL_DRAFT=<MTP draft on the rig> \
  TEXT_LOCAL=<proxy.jsonl.bodies.gz> STEP=dry bash docs/data/2026-11-26_r124_flash_131k_rig2/run/r124.sh
PRESET=preset_bf7982b1.env bash docs/data/2026-11-26_r124_flash_131k_rig2/run/line_check.sh <the draft arm's line> draft    # and: <nodraft line> nodraft
# the same r124.sh call with STEP=start, then STEP=wait, then STEP=collect OUT=<folder>
OUT=<folder> PROBE=fit_probe.md bash docs/data/2026-11-26_r124_flash_131k_rig2/run/tables.sh
```

## Provenance and licence

Measured by this project on its own rig; the tables and scripts are this repository's. The prompts are cut from this
project's own recorded agent sessions (the v3.0 release check), sent as token ids; their text is not reproduced here,
and the replies are quoted only by the few tokens where they part. The model is UkisAI's Swift-1.5 GSQ-RCO Q2_0 of
Qwen3.8-Flash-Next with its MTP draft, under the makers' licences.
