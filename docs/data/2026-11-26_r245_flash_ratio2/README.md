# Flash-Next's four-token switch point between 0.42 and 0.49, with 5 splits (R245, Q1925, 2026-11-26)

What this measured: Qwen3.8-Flash-Next's decode step on nine P106-090 cards when its four-token verify passes move to
the per-token sparse attention kernel a little sooner than the v5 preset moves them, on the deep lane. Registry row
[Q1925](../../../quants/experiments.md), from R240 ([Q1910](../2026-11-26_r240_flash_ratio_rig2/README.md)).

The v5 preset sends four-token verifies to the sparse kernel from a cache view of 4,352 cells
(`GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=0.5`, with `_NSM=10`: C1175). R240 found 0.375 (a view of 3,328) and 0.25
(2,304) slower, and its numbers put the point where the sparse kernel starts to win near est. 3,500 to 3,700 cells.
This row tries the three switch points in between: 0.42, 0.45 and 0.49, which act from views of 3,584, 3,840 and
4,096 cells.

**Result: `_MIN_RATIO=0.45` gains, and it is the one arm that is slower on no step figure.** On the repeated request
(pass 2), ms a step, each value one load:
- **At 3,840 tokens (views of 4,096 cells), all three lower switch points run four-token verifies on the sparse kernel
  and the preset does not:** 0.42 98.98 and 98.81, 0.45 98.80 and 99.52, 0.49 98.77 and 99.04, against the preset's
  99.94 and 100.02 (−1.08, −0.82 and −1.07 ms, gains), every load on the preset's own greedy reply.
- **At 3,584 tokens (views of 3,840), 0.42 and 0.45 run the sparse kernel:** 0.45 98.11 and 98.31 (−1.30 ms), 0.42
  98.19 and 98.47 (−1.18), against 99.51 and 99.50; both write another reply than the preset.
- **At 3,328 tokens (views of 3,584), only 0.42 does:** 98.83 and 98.50 against 99.17 and 99.02 (−0.43 ms), while 0.45
  and 0.49, on the preset's kernels and reply, measure −0.26 and −0.06 there.
- **0.42 and 0.49 fail the row's rule** (faster beyond the control's spread and slower on no figure): 0.42's sampled
  request at 3,328 tokens is slower in both loads (101.03 and 114.38 ms against 100.30 and 99.81, on another reply),
  and 0.49 is slower on pass 2 at 3,584 (100.65 and 99.54 against 99.51 and 99.50), where it runs the preset's kernels.

**What follows:** the agent check on 0.45 (R240's prepared `run/flash_ratio.sh`, arm `r45a|b`) did not run today: four
runs take about 2 hours (C1175's took 1 h 58 min) and could not end by the rigs' 22:00 stop. It is queued as R247
([Q1935](../../../quants/experiments.md)). If it holds, the preset line becomes `_MIN_RATIO=0.45`.

## The switches

In the engine's sparse attention (`fattn-sparse-pascal.cu` and `fattn-sparse-pascal-impl.cuh` at
[`eee615cdb`](https://github.com/Cinderella-Man/miner-llama/blob/eee615cdb/ggml/src/ggml-cuda/fattn-sparse-pascal.cu)),
with `GGML_CUDA_FATTN_SPARSE_PASCAL=1` (Flash-Next's v5 preset), a decode or verify call of `n_q` query tokens (up to
8) goes to the per-token kernel when the cache view it reads holds at least `_MIN_RATIO` × 2,051 (the cells a QSA layer
keeps) × `n_q` cells. The view grows in steps of 256 cells, so a switch point acts from the next multiple of 256 at or
above it ([R240](../2026-11-26_r240_flash_ratio_rig2/README.md), "The switches", has the details, including the rule
below 2,051 cells). A request at depth D generates 256 tokens, so the depths 3,328, 3,584 and 3,840 read views of
3,584, 3,840 and 4,096 cells: the first view where each arm acts.

All four arms run `_NSM=10` (5 splits at 4 tokens), the preset's value.

| Arm | `_MIN_RATIO` | Four-token verifies on the sparse kernel from | That is, from a view of | Depths where the arm runs other kernels than the control |
|---|---:|---:|---:|---|
| `c`, the v5 preset (control) | 0.5 | 4,102 cells | 4,352 cells | |
| `r42` | 0.42 | 3,446 cells (3,445.7) | 3,584 cells | 3,328, 3,584, 3,840 (four-token calls); at 0 two-token calls |
| `r45` | 0.45 | 3,692 cells (3,691.8) | 3,840 cells | 3,584, 3,840 (four-token calls); at 0 two-token calls |
| `r49` | 0.49 | 4,020 cells (4,019.96) | 4,096 cells | 3,840 (four-token calls); at 0 two-token calls |

Smaller calls switch sooner in the same proportion. Above 3,328 cells every arm, the control included, already runs
one-, two- and three-token calls on the per-token kernel. Below 2,051 cells a call goes to the per-token kernel only
when `_MIN_RATIO` × `n_q` is under 1: there the three lower ratios send two-token calls (0.84, 0.90 and 0.98) and the
control does not (1.0).

## The deep lane

### Result

On the repeated request (pass 2, which meets no token new to the draft's shortlist), ms a step, each value one load:

| Depth (tokens) | `c` (the preset, 0.5) | `r42` | `r42` − `c`, class | `r45` | `r45` − `c`, class | `r49` | `r49` − `c`, class |
|---:|---|---|---|---|---|---|---|
| 0 | 90.43, 90.60 | 90.40, 90.48 | −0.07, negligible | 90.44, 90.45 | −0.07, negligible | 90.62, 90.42 | +0.01, negligible |
| 3,328 | 99.17, 99.02 | 98.83, 98.50 | **−0.43, gain** | 98.89, 98.78 | −0.26, gain (the control's kernels) | 99.14, 98.92 | −0.06, negligible |
| 3,584 | 99.51, 99.50 | 98.19, 98.47 | **−1.18, gain** (another reply) | 98.11, 98.31 | **−1.30, gain** (another reply) | 100.65, 99.54 | +0.59, negligible (the control's kernels) |
| 3,840 | 99.94, 100.02 | 98.98, 98.81 | **−1.08, gain** | 98.80, 99.52 | **−0.82, gain** | 98.77, 99.04 | **−1.07, gain** |
| all four | 96.66, 96.69 | 95.95, 95.93 | −0.73, gain | 95.97, 96.15 | −0.62, gain | 96.76, 96.42 | −0.09, negligible |

The class is the lane's rule: a gain or a loss only beyond the larger of the two arms' spreads (the difference between
an arm's two loads). [lane_steps.md](lane_steps.md) has pass 1, the sampled request, the steps and draft counts of
every load, and both class rules (against the larger spread, and against the control's own).

- **Where the three lower ratios run the same kernels, they measure the same.** At 3,840 tokens all three run the
  sparse four-token kernel on the preset's reply: −1.08, −0.82 and −1.07 ms. C839's kernel timings with the server's
  offset R240 measured had estimated −0.87 for a view of 4,096 cells.
- **At 3,584 tokens (views of 3,840)** 0.42 and 0.45 gain 1.18 and 1.30 ms, more than the est. −0.45, but both on
  other replies (they part from the preset's after 4 and 15 equal tokens) that accept more drafts (182 of 216 and 175
  of 237 against 169 of 257), so the step's mix differs from the control's as well as its kernels. On pass 1 they gain
  2.99 and 3.24 ms there, again on other replies.
- **At 3,328 tokens (views of 3,584)** 0.42 gains 0.43 ms on a reply that parts from the preset's after 139 equal
  tokens, with the same draft counts (181 of 221). 0.45 and 0.49 run the preset's kernels and write its reply there,
  and measure −0.26 and −0.06: the loads' own scatter at this depth is that large, so 0.42's own effect here is est.
  −0.2 to −0.3 ms, near C839's est. −0.04.
- **Generation tok/s** follows the replies: 0.42 +19.0 % and 0.45 +8.8 % at 3,584 tokens are their other replies'
  fewer steps (74 and 81 against 87), not the switch alone; at 3,840, on the same reply, +1.1 %, +0.8 % and +1.1 %.

**Why 0.45 is the arm** (the row's rule: faster beyond the control's spread on pass 2 and slower on no figure):
- **0.45:** faster on pass 2 at 3,584 and 3,840 tokens, on pass 1 there (−3.24 and −0.90) and on the sampled request
  there (−2.68 and −1.76); negligible elsewhere; slower on no step figure under either class rule.
- **0.42:** the sampled request at 3,328 tokens is slower in both loads, 101.03 and 114.38 ms against 100.30 and 99.81
  (`r42b`'s request is one slow run: 8,463.8 ms against `r42a`'s 7,476.3 for the same reply; `r42a` is still 0.97 ms
  a step above the control's mean, on another reply with 74 steps against 79). Its prompt reading at depth 0 is also
  a loss by the lane's rule (55.6 and 55.5 tok/s against 58.7 and 58.4, on a 64-token prompt whose calls no arm
  changes).
- **0.49:** pass 2 at 3,584 tokens 100.65 and 99.54 against 99.51 and 99.50 (both loads above both control loads,
  where 0.49 runs the control's kernels), and `r49b`'s sampled requests are slow at depth 0 and 3,328 (102.49 and
  105.24 ms against `r49a`'s 93.25 and 100.28). It gains only at 3,840, where 0.45 gains too.

### Checks

- **Replies** ([replies.tsv](replies.tsv), R69's [replies.py](../2026-11-20_r69_sparse_switch/run/replies.py)):
  - Every greedy reply is the same token ids in both loads of each arm, and every prime the same in all eight loads.
  - At depth 0 and 3,840 tokens every greedy reply is the preset's. At 3,328 0.42's parts after 139 equal tokens; at
    3,584 0.42's after 4 and 0.45's after 15. Each arm's reply differs only at depths where its kernels differ.
  - Sampled replies (the preset's sampler, a fixed seed): the preset's at depth 0 in every load. 0.42's differ from
    3,328 on and 0.45's from 3,584 (where their kernels differ). At 3,840 every lower ratio's differs (0.42 and 0.45
    write one reply, 0.49 another).
  - **One sampled reply did not repeat:** `r49a`'s at 3,328 tokens parts from `r49b`'s, which is the preset's, after
    19 equal tokens, although at that depth 0.49 runs the preset's kernels and both loads wrote the same replies
    before it. `r49a`'s sampled reply at 3,584 then differs too (after 139 tokens), the draft's shortlist carrying the
    difference. Every greedy reply and every prime repeated. The cause is not known: code row C1196
    ([Q1936](../../../quants/experiments.md)) looks for it.
- **Prompt reading** ([table.md](table.md)): negligible except two figures on calls that no arm changes (prompt calls
  hold 64 tokens or more and follow `_PROMPT_MIN_RATIO`): 0.42 at depth 0 (above) and 0.45's prime at 3,840 tokens,
  181.0 and 181.4 tok/s against 184.1 and 182.7 (−1.2 %), the primes' known scatter between loads (R69, R175, C1175,
  R240; code row C1177).
- **Memory and health:** every load's lowest host MemAvailable: `ca` 256, `r42a` 181, `r45a` 198, `r49a` 196, `r49b`
  329, `r45b` 201, `r42b` 258, `cb` 202 MiB (the warm-up 257); no RAM watchdog stop; 0 graph reallocations in all
  128 timed requests; every prime drafted nothing. The card log ([cards.md](cards.md)): median busy SM clock 2,088 to
  2,151 MHz across the nine cards, memory 4,455 to 4,657 MHz; the kit's watchdog never tripped and the boot logged no
  GPU error.

### What ran

- **Rig and settings:** rig 2, nine P106-090 cards with that rig's soaked clock profile set on the morning's boot
  (memory +1,300 +1,000 +1,000 +1,100 +1,100 +1,200 +900 +900 +1,300 MT/s and core +256 +256 +269 +231 +206 +206
  +282 +256 +231 MHz on cards 0 to 8, the 75 W default power limit, the kit's watchdog running:
  [2026-11-19_r209_oc_v5](../2026-11-19_r209_oc_v5/README.md), "The profile again on the boot of 2026-11-26"), the
  boot C1175 and R240 ran on. The offsets read the same before and after, with no watchdog trip and no GPU Xid.
- **Model and preset:** Qwen3.8-Flash-Next as [release/flash-next-v5/settings.env](../../../release/flash-next-v5/settings.env)
  runs it at `602ba0cf` (md5 `b5ae1f101f4258713402827cb3d494cc`: `_NSM=10 _MIN_RATIO=0.5`,
  `--ubatch-taper short=1400:64,cuts=1`): UkisAI's Swift-1.5 GSQ-RCO Q2_0
  (`Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-0000{1,2,3}-of-00003.gguf`; the third shard is its n-gram table) and the MTP
  draft (`mtp-Qwen3.8-Flash-Next-draft.gguf`), on, three tokens a step. Context 65,536.
- **Build:** one for every arm, fork `pascal-p106-v5-cand-b12674` at `eee615cdb` (upstream b12674 merged), the build
  C839, C1175 and R240 ran; `llama-server` md5 `2727faf28b5b34dfe42399a9d431da67`.
- **The arms:** [run/r245.sh](run/r245.sh) drops the preset's `_NSM` and `_MIN_RATIO` and each arm sets both, so the
  four server lines differ in `_MIN_RATIO` only (the dry run's four lines are byte for byte the same once that value is
  masked); each server's switch line printed it.
- **The deep lane** ([bench/deep_lane](../../../bench/deep_lane/README.md)) ran 19:16:28–20:11:23.
  - **Loads:** a warm-up load of `c` (untimed), then `ca r42a r45a r49a r49b r45b r42b cb`, each a fresh server
    start; the second round runs the arms in reverse order (Q1253).
  - **Quiet checks:** the check before every load passed: no llama process, 0 MiB on every card.
  - **Requests:** at 0, 3,328, 3,584 and 3,840 tokens each load read the prompt (a `prime` request, from 3,328 on),
    ran 256 tokens at temperature 0 twice (pass 1 and pass 2), then once with the preset's own sampler and a fixed
    seed. The prompts are the same ids on every load, cut from this project's recorded agent text.

## The agent check

Not run today; queued. 0.45 passed the lane's rule, but four alternated runs (`ca r45a cb r45b`) take about 2 hours
and the lane ended at 20:11, so they could not end by the rigs' 22:00 stop. R240's
[run/flash_ratio.sh](../2026-11-26_r240_flash_ratio_rig2/run/flash_ratio.sh) runs it as it is (any `r<digits>a|b` arm
sets `_MIN_RATIO=0.<digits>`; the taper comes from `settings.env`), with R240's
[run/dry_line.sh](../2026-11-26_r240_flash_ratio_rig2/run/dry_line.sh) and
[run/preset_check.sh](../2026-11-26_r240_flash_ratio_rig2/run/preset_check.sh) (`RATIO` 0.45) to check its line
first.

## Files

| File | What |
|---|---|
| [lane_steps.md](lane_steps.md), [lane_steps.tsv](lane_steps.tsv) | the verdict table: the step of every load per depth and pass type, its steps and draft counts, both class rules, per arm |
| [table.md](table.md), [classes.tsv](classes.tsv), [classes_pass1.tsv](classes_pass1.tsv), [classes_pass2.tsv](classes_pass2.tsv) | the lane's own tables and classes (on the load, pass 1 and pass 2); `table.md` is the rig's byte for byte |
| [replies.tsv](replies.tsv) | per lane request and load, the reply's token-id and text md5 and the first token where it differs from `ca`'s |
| [requests.csv](requests.csv), [loads.tsv](loads.tsv) | the lane: one line per request (deep_client.py) and per load (its start, load time, full server line) |
| [cards.md](cards.md) | the lane's card log summary (`scripts/card_table.py`) |
| [run/r245.sh](run/r245.sh) | the deep lane's run script (R240's `run/r240.sh` with the switch points as the arms and the depths around the crossover) |

## Commands

```bash
# on a machine that runs only ssh and tar, from the repository's root; RIG, WORK and the paths are yours;
# PRESET=<settings.env as at 602ba0cf> runs the preset as this run did (the default is the preset as it stands)
RIG=rig WORK=/tmp/r245 BIN=<llama-server on the rig> MODEL=<first shard on the rig> MODEL_DRAFT=<MTP draft on the rig> \
  TEXT_LOCAL=<proxy.jsonl.bodies.gz> STEP=dry bash docs/data/2026-11-26_r245_flash_ratio2/run/r245.sh
# the same with STEP=start, then STEP=wait, then STEP=collect OUT=<folder>
python3 -I bench/deep_lane/deep_table.py OUT --control c --out table.md --tsv classes.tsv
python3 -I bench/deep_lane/deep_table.py OUT --control c --class-on pass1 --tsv classes_pass1.tsv
python3 -I bench/deep_lane/deep_table.py OUT --control c --class-on pass2 --tsv classes_pass2.tsv
for a in r42 r45 r49; do
  python3 -I docs/data/2026-11-20_r175_flash_shortlist/run/steps.py OUT --control c --arm $a --tsv steps_$a.tsv > steps_$a.md
done
# lane_steps.md is steps_r42.md, steps_r45.md and steps_r49.md under a heading each; lane_steps.tsv their rows, each once
python3 -I docs/data/2026-11-20_r69_sparse_switch/run/replies.py OUT --control-load ca \
  --loads ca,r42a,r45a,r49a,r49b,r45b,r42b,cb > replies.tsv
python3 -I scripts/card_table.py OUT/cards.csv > cards.md
```

## Provenance and licence

Measured by this project on its own rig; the tables and scripts are this repository's. The prompts are cut from this
project's own recorded agent sessions (the v3.0 release check), sent as token ids; their text is not reproduced here.
The model is UkisAI's Swift-1.5 GSQ-RCO Q2_0 of Qwen3.8-Flash-Next with its MTP draft, under the makers' licences.
