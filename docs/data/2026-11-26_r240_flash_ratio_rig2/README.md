# Flash-Next's four-token switch point below 0.5, with 5 splits (R240, Q1910, 2026-11-26)

What this measured: Qwen3.8-Flash-Next's decode step on nine P106-090 cards when its four-token verify passes move to
the per-token sparse attention kernel at a smaller cache than the v5 preset's, on the deep lane. Registry row
[Q1910](../../../quants/experiments.md), from C1175 ([Q1908](../2026-11-26_c1175_nsm10_rig2/README.md)).

C1175 put `GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=10` (a four-token verify in 5 splits) and `_MIN_RATIO=0.5` (on the sparse
kernel from 4,102 cells) into Flash-Next's v5 preset. At 4,096 tokens the lower ratio made a step 1.40 ms shorter, 5.4
times what the kernels timed alone predicted (C839: est. 0.26 ms), so the point where the sparse kernel starts to win
could lie lower. This row moved it to 3,077 cells (`_MIN_RATIO=0.375`) and to 2,051 (`0.25`).

**Result: a loss for both lower switch points; the preset's 0.5 stays.** Wherever a lower ratio sends four-token
verifies to the sparse kernel, the step is longer, and nowhere is it shorter. On the repeated request (pass 2),
against the preset's 95.85 and 95.89 ms a step at 2,048 tokens and 97.69 and 97.82 at 3,072:
- **`_MIN_RATIO=0.25`: 98.09 and 98.15 ms at 2,048 tokens (+2.25 ms, loss) and 98.01 and 98.05 at 3,072 (+0.27,
  loss)**, both on the same greedy reply as the preset (the same token ids).
- **`_MIN_RATIO=0.375`: 98.48 and 98.51 ms at 3,072 tokens (+0.74, loss)**, on another reply; at 2,048 it runs the
  preset's kernels and writes its reply.
- **At depth 0 and at 4,096 tokens the step is the same** (at 4,096 all three settings run the same kernels).

The sparse kernel's cost at these sizes is what C839's kernel timings predict, less about 0.4 to 0.7 ms a step: est.
+2.64 ms at 2,048 tokens against +2.25 measured, est. +0.98 at 3,072 against +0.27, and at 4,096 est. −0.68 against
C1175's −1.40 (the estimates take C839's per-token kernel at 5 splits, 590.2 µs a call at 4,096 cells and 625.8 at
8,192, and its dense kernels, 611.9 and 1,200.5, each as a line in the cells of the view, × 12 calls a pass). On that
line the four-token crossover lies near est. 3,500 to 3,700 cells, between 0.375's and 0.5's switch points as the
engine applies them (below). Row R245 ([Q1925](../../../quants/experiments.md)) tries the three points in between:
views of 3,584, 3,840 and 4,096 cells (`_MIN_RATIO` 0.42, 0.45 and 0.49; est. −0.04, −0.45 and −0.87 ms a step
there). The agent check did not run: it follows only an arm that gains on the lane.

## The switches

In the engine's sparse attention (`fattn-sparse-pascal.cu` and `fattn-sparse-pascal-impl.cuh` at
[`eee615cdb`](https://github.com/Cinderella-Man/miner-llama/blob/eee615cdb/ggml/src/ggml-cuda/fattn-sparse-pascal.cu)),
with `GGML_CUDA_FATTN_SPARSE_PASCAL=1` (Flash-Next's v5 preset), a decode or verify call of `n_q` query tokens (up to
8) goes to the per-token kernel when the cache view it reads holds at least `_MIN_RATIO` × the kept cells × `n_q`
cells (`fasp_rule_takes`). Two details of Flash-Next's cache set where that happens:
- **The cache view grows in steps of 256 cells** (the KV cache's padding), so a switch point acts from the next
  multiple of 256 at or above it. The depths 2,048, 3,072 and 4,096 are the first where each arm acts: a request there
  generates 256 tokens, so its view is 2,304, 3,328 and 4,352 cells.
- **The kept cells are 2,051 once a QSA layer selects** (512 pools of 4 cells and a tail of 3). In a smaller cache the
  layer keeps every cell, and the count the rule reads is the sequence's pools rounded up to 64 (256 cells) and the
  tail (the pools plus one, rounded up to a multiple of 64 pools, × 4 + 3): at least the view plus 3 cells. There a
  call goes to the per-token kernel only when `_MIN_RATIO` × `n_q` is under 1, then in most of each 256-cell step:
  one-token calls in every arm, two-token calls at 0.375 and 0.25, three-token calls at 0.25, four-token calls in
  none.

All three arms run `_NSM=10` (5 splits at 4 tokens, 20 at 1), the preset's value.

| Arm | `_MIN_RATIO` | Four-token verifies on the sparse kernel from | That is, from a view of | Depths where the arm runs other kernels than the control |
|---|---:|---:|---:|---|
| `c`, the v5 preset (control) | 0.5 | 4,102 cells | 4,352 cells | |
| `r375` | 0.375 | 3,077 cells | 3,328 cells | 3,072 (four-token calls); at 0 two-token calls |
| `r25` | 0.25 | 2,051 cells | 2,304 cells | 2,048 (three- and four-token calls), 3,072 (four-token calls); at 0 two- and three-token calls |

At 4,096 tokens every arm runs every call of 1 to 4 tokens on the per-token kernel. At 2,048 `r375` runs the
control's kernels (its three-token point, 2,307.4 cells, is above the 2,304-cell view).

## The deep lane

### Result

On the repeated request (pass 2, which meets no token new to the draft's shortlist), ms a step, each value one load:

| Depth (tokens) | `c` (the preset, 0.5) | `r375` | `r375` − `c`, class | `r25` | `r25` − `c`, class |
|---:|---|---|---|---|---|
| 0 | 90.42, 90.59 | 90.66, 90.59 | +0.12, negligible | 90.88, 90.37 | +0.12, negligible |
| 2,048 | 95.85, 95.89 | 95.86, 97.00 | +0.56, negligible (the same kernels; see below) | 98.09, 98.15 | **+2.25, loss** |
| 3,072 | 97.69, 97.82 | 98.48, 98.51 | **+0.74, loss** | 98.01, 98.05 | **+0.27, loss** |
| 4,096 | 98.73, 98.79 | 98.58, 98.71 | −0.12, negligible | 98.74, 98.77 | −0.00, negligible |

The class is the lane's rule: a gain or a loss only beyond the larger of the two arms' spreads (the difference between
an arm's two loads). [lane_steps.md](lane_steps.md) has pass 1, the sampled request, the steps and the draft counts of
every load, and both class rules.

- **`r25` at 2,048 tokens: +2.25 ms a step on the preset's own reply.** Its four- and three-token verifies read 2,051
  cells a token where the dense kernels read the 2,304-cell view once for all tokens. C839's timings give est. +220.2 µs
  a four-token call (dense 354.4 µs at 2,304 cells, per-token 574.6), est. +2.64 ms a verify pass of 12 calls. Its
  draft accepts more of the same reply (183 of 215 against 181 of 221), so it needs 73 steps instead of 75 and the
  request took 7,160.4 and 7,165.0 ms against 7,188.8 and 7,191.9: the draft's luck on a near-tie, not the kernel.
- **At 3,072 tokens both lower ratios run the same kernels** (four-token verifies on the per-token kernel, the control
  dense). `r25`, on the preset's reply, is 0.27 ms a step slower; `r375`, whose reply parts from it after 24 equal
  tokens, 0.74. C839's est. +0.98 ms (dense 501.5 µs a call at 3,328 cells, per-token 583.5).
- **At 4,096 tokens the step is the same in all three arms**, which run the same kernels there. The lower ratios write
  another reply than the control (below) that accepts more drafts (180 of 223 against 175 of 237), so they need 76
  steps instead of 81: generation 33.45 and 33.44 tok/s (`r375`) and 33.19 and 33.19 (`r25`) against 31.43 and 31.40
  in [table.md](table.md), the same step over fewer steps. For the same reason `r25` is a "gain" of +1.5 % in
  generation over all depths (31.58 and 31.59 against 31.15 and 31.10) and +1.1 % at 3,072, where its step is longer.
  The step is what the switch changes.
- **`r375` at 2,048** runs the control's kernels and writes its reply with its draft counts; `r375b`'s 97.00 ms is one
  load's slower request (`r375a` 95.86).
- **Pass 1** (the first request at a new depth): `r25` +2.32 ms at 2,048 (loss), +0.37 at 3,072 (negligible), +0.43 at
  4,096 (loss); `r375` −0.86 at 3,072 ("gain" against the control's 0.66 ms spread there: 101.18 and 101.84) and +0.34
  at 4,096 (loss). The step on pass 1 carries the shortlist's misses as well; at 3,072 the two lower ratios run the same
  kernels and are 1.24 ms apart (100.63 and 100.67 against 101.71 and 102.06).
- **The sampled request** (the preset's sampler, a fixed seed): `r25` +3.62 ms at depth 0, +1.20 at 3,072 (losses),
  +0.49 at 2,048 (negligible); `r375` −0.54 at 3,072 (gain), negligible elsewhere; at 4,096 `ca` ran one slow request
  (113.98 against `cb`'s 104.13), so nothing is classed there. `r25`'s sampled reply is other text than the control's
  at every depth, `r375`'s from 3,072.

### Checks

- **Replies** ([replies.tsv](replies.tsv), R69's [replies.py](../2026-11-20_r69_sparse_switch/run/replies.py)):
  every reply is the same token ids in both loads of each arm, and every prime the same in all six loads.
  - At depth 0 and 2,048 every greedy reply is the control's. `r25`'s sampled replies at 0 and 2,048 part from the
    control's after 20 and 5 equal tokens, and its warm-up reply too: its two- and three-token calls take the other
    kernel there.
  - At 3,072 `r375`'s greedy reply parts after 24 equal tokens, `r25`'s is the control's.
  - At 4,096, where all three arms run the same kernels, both lower ratios' greedy replies part from the control's
    after 10 equal tokens, the same reply in all four loads. The difference comes from earlier in the load: the
    draft's shortlist keeps the target tokens a load meets (at the end 32,969 ids in `c`, 32,967 in `r375`, 32,970 in
    `r25`), so after an earlier reply differs the draft proposes other tokens, the verify passes have other sizes and
    the sums run in another order.
  - The per-token and dense kernels, and the split counts, add the same values in a different order, so a near-tie
    can turn; R69 and C1175 saw the same when the switch point moved.
- **Prompt reading** ([table.md](table.md)): negligible at 0, 3,072 and 4,096; at 2,048 `r375` 153.7 and 157.9 tok/s
  against 161.8 and 159.7 (−3.1 %, a loss by the lane's rule) and `r25` 162.6 and 167.5 (negligible). A prime's calls
  hold 64 to 512 tokens and follow `_PROMPT_MIN_RATIO`, which no arm changes, so every arm runs them on the same
  kernels; the primes scatter between loads as in R69, R175 and C1175 (code row C1177).
- **Memory and health:** every load's lowest host MemAvailable 172 to 388 MiB, no RAM watchdog stop, 0 graph
  reallocations in all 96 timed requests; every prime drafted nothing. The card log ([cards.md](cards.md)): median busy
  SM clock 2,100 to 2,151 MHz, memory 4,455 to 4,657 MHz; the kit's watchdog never tripped and the boot logged no GPU
  error.

### What ran

- **Rig and settings:** rig 2, nine P106-090 cards with that rig's soaked clock profile set on the morning's boot
  (memory +1,300 +1,000 +1,000 +1,100 +1,100 +1,200 +900 +900 +1,300 MT/s and core +256 +256 +269 +231 +206 +206
  +282 +256 +231 MHz on cards 0 to 8, the 75 W default power limit, the kit's watchdog running:
  [2026-11-19_r209_oc_v5](../2026-11-19_r209_oc_v5/README.md), "The profile again on the boot of 2026-11-26"), the
  boot C1175 ran on. The offsets read the same before and after, with no watchdog trip and no GPU Xid.
- **Model and preset:** Qwen3.8-Flash-Next as [release/flash-next-v5/settings.env](../../../release/flash-next-v5/settings.env)
  runs it at `d296f5ce` (md5 `3616639032df7218d1a4b744abe5b60e`, copied at 15:56: `_NSM=10 _MIN_RATIO=0.5`,
  `--ubatch-taper short=1400:64`; the taper's `cuts=1` joined the preset at 16:08, `602ba0cf`, after this run had
  started): UkisAI's Swift-1.5 GSQ-RCO Q2_0 (`Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-0000{1,2,3}-of-00003.gguf`; the
  third shard is its n-gram table) and the MTP draft (`mtp-Qwen3.8-Flash-Next-draft.gguf`), on, three tokens a step.
  Context 65,536.
- **Build:** one for every arm, fork `pascal-p106-v5-cand-b12674` at `eee615cdb` (upstream b12674 merged), the build
  C839 and C1175 ran; `llama-server` md5 `2727faf28b5b34dfe42399a9d431da67`.
- **The arms:** [run/r240.sh](run/r240.sh) drops the preset's `_NSM` and `_MIN_RATIO` and each arm sets both, so the
  three server lines differ in `_MIN_RATIO` only; each server's switch line printed it.
- **The deep lane** ([bench/deep_lane](../../../bench/deep_lane/README.md)) ran 15:56:54–16:39:04.
  - **Loads:** a warm-up load of `c` (untimed), then `ca r375a r25a r25b r375b cb`, each a fresh server start; the
    second round runs the arms in reverse order (Q1253). Every load read the files in 225 to 227 s.
  - **Quiet checks:** the check before every load passed: no llama process, 0 MiB on every card.
  - **Requests:** at 0, 2,048, 3,072 and 4,096 tokens each load read the prompt (a `prime` request, from 2,048 on),
    ran 256 tokens at temperature 0 twice (pass 1 and pass 2), then once with the preset's own sampler and a fixed
    seed. The prompts are the same ids on every load, cut from this project's recorded agent text.

## The agent check

Not run. The row runs it only on an arm that gains on the lane beyond the spread and is slower on no figure; both
arms are slower where they act. [run/flash_ratio.sh](run/flash_ratio.sh) is ready for a later arm: C1175's
`run/flash_nsm.sh` for the preset since `d296f5ce`, with the control `ca|cb` and any `r<digits>a|b` arm
(`_MIN_RATIO=0.<digits>`), and the taper read from `settings.env`. With the preset at `602ba0cf` its dry line for `ca`
([run/dry_line.sh](run/dry_line.sh)) is 1,724 bytes, md5 `6f2c7e6c5d6569e2bf2533f5cc14142f`: C1175's `r05` command
with `,cuts=1` added to the taper and nothing else. It passes [run/preset_check.sh](run/preset_check.sh) against
`settings.env` (40 switches, 30 flags); so do its `r375a` and `r25a` lines with their ratios.

## Files

| File | What |
|---|---|
| [lane_steps.md](lane_steps.md), [lane_steps.tsv](lane_steps.tsv) | the verdict table: the step of every load per depth and pass type, its steps and draft counts, both class rules, per arm |
| [table.md](table.md), [classes.tsv](classes.tsv), [classes_pass1.tsv](classes_pass1.tsv), [classes_pass2.tsv](classes_pass2.tsv) | the lane's own tables and classes (on the load, pass 1 and pass 2); `table.md` is the rig's byte for byte |
| [replies.tsv](replies.tsv) | per lane request and load, the reply's token-id and text md5 and the first token where it differs from `ca`'s |
| [requests.csv](requests.csv), [loads.tsv](loads.tsv) | the lane: one line per request (deep_client.py) and per load (its start, load time, full server line) |
| [cards.md](cards.md) | the lane's card log summary (`scripts/card_table.py`) |
| [run/r240.sh](run/r240.sh) | the deep lane's run script (C1175's `run/c1175.sh` with the switch point as the arm and lower depths) |
| [run/flash_ratio.sh](run/flash_ratio.sh), [run/dry_line.sh](run/dry_line.sh), [run/preset_check.sh](run/preset_check.sh) | the agent check's script for a switch-point arm, the server line it would start, and its check against `settings.env` (not run here) |

## Commands

```bash
# on a machine that runs only ssh and tar, from the repository's root; RIG, WORK and the paths are yours;
# PRESET=<settings.env as at d296f5ce> runs the preset as this run did (the default is the preset as it stands)
RIG=rig WORK=/tmp/r240 BIN=<llama-server on the rig> MODEL=<first shard on the rig> MODEL_DRAFT=<MTP draft on the rig> \
  TEXT_LOCAL=<proxy.jsonl.bodies.gz> STEP=dry bash docs/data/2026-11-26_r240_flash_ratio_rig2/run/r240.sh
# the same with STEP=start, then STEP=wait, then STEP=collect OUT=<folder>
python3 -I bench/deep_lane/deep_table.py OUT --control c --out table.md --tsv classes.tsv
python3 -I bench/deep_lane/deep_table.py OUT --control c --class-on pass1 --tsv classes_pass1.tsv
python3 -I bench/deep_lane/deep_table.py OUT --control c --class-on pass2 --tsv classes_pass2.tsv
python3 -I docs/data/2026-11-20_r175_flash_shortlist/run/steps.py OUT --control c --arm r375 --tsv steps_r375.tsv > steps_r375.md
python3 -I docs/data/2026-11-20_r175_flash_shortlist/run/steps.py OUT --control c --arm r25 --tsv steps_r25.tsv > steps_r25.md
# lane_steps.md is steps_r375.md and steps_r25.md under a heading each; lane_steps.tsv their rows, each once
python3 -I docs/data/2026-11-20_r69_sparse_switch/run/replies.py OUT --control-load ca --loads ca,r375a,r25a,r25b,r375b,cb > replies.tsv
python3 -I scripts/card_table.py OUT/cards.csv > cards.md

# the agent check's server line for an arm, and its check (no run here)
ARM=ca WORK=<scratch> NEW=<build folder on the rig> bash docs/data/2026-11-26_r240_flash_ratio_rig2/run/dry_line.sh > line_ca.txt
bash docs/data/2026-11-26_r240_flash_ratio_rig2/run/preset_check.sh line_ca.txt            # r25a: ... line_r25a.txt 0.25
```

## Provenance and licence

Measured by this project on its own rig; the tables and scripts are this repository's. The prompts are cut from this
project's own recorded agent sessions (the v3.0 release check), sent as token ids; their text is not reproduced here.
The model is UkisAI's Swift-1.5 GSQ-RCO Q2_0 of Qwen3.8-Flash-Next with its MTP draft, under the makers' licences.
