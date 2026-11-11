# R82 (Q1297): C702's prompt graphs on a P106 (2026-11-11)

What this measured: whether capturing prompt micro-batches as CUDA graphs (`GGML_CUDA_GRAPHS_FORCE=1`, C702,
[the change](../2026-11-11_c702_prompt_graphs/README.md)) gives the same replies and verify steps on a P106 rig, and
what it does to a fresh 4,097-token prompt, whose third queued micro-batch of each call blocked the host in the launch
([R42](../2026-11-10_prompt_timeline/README.md): 8,851.0 of 29,983.0 ms; [R81](../2026-11-11_r81_ubatch_trace/README.md):
8,872.2 ms in `l`). The registry row is Q1297 in [quants/experiments.md](../../../quants/experiments.md).

The build is Miner Llama's v4 night merge, `pascal-p106-v4-b12246-night` at `c408dabcb`
([2026-11-11_v4_night_merge](../2026-11-11_v4_night_merge/README.md)), which holds C702's change. No other switch it adds
was set, so with this switch off it computes what `pascal-p106-v4` computes. The model and line are the v3 preset's:
UkisAI's Swift-1.5 GSQ-RCO Q2_0 of Qwen3.8-Flash-Next on nine P106-090 cards, `-b 2048 -ub 512`, the MTP draft at three
tokens with catch-up passes of 512 rows (the identity lane, R70's v4 candidate) or the preset's 128 (R42's driver).

## Result

- **Identity holds.** In the deep lane (the v4 candidate: R70's five switches, greedy, 16 tokens a request at depths 0
  and 32,768) the greedy reply is the same on every pass and load of the three loads at both depths, and the switch's
  load gives the same 40 verify steps as the two eager loads, every field the same (the drafted tokens, their
  candidates and probabilities, the main model's targets and probabilities, the tokens kept); the two eager loads
  agree with each other on all 40 ([lane/dump_compare.tsv](lane/dump_compare.tsv)).
- **The switch is a gain on the 4,097-token prompt: −2,798.3 ms (−10.2 %) on the two fresh prompts' window.** In R42's
  graph-profile load, which runs at the unprofiled pace, the window is 24,932.3 and 24,305.5 ms with the switch against
  27,783.8 and 27,050.6 without it (the two fresh prompts with the micro-batch taper off, requests 7 and 10). Each
  prompt is 2,851.5 and 2,745.1 ms faster, beyond both arms' spreads (626.8 and 733.2 ms). The unprofiled load's prompt
  times agree: 25,345.82 and 24,706.76 ms against 28,199.43 and 27,170.76 (−2,658.8 ms, −9.6 %). With the taper on
  (requests 8 and 9): 23,033.7 and 23,114.7 ms against 26,338.4 and 26,779.7 (unprofiled 23,076.1 and 23,199.4 against
  26,371.8 and 26,821.4).
- **The blocked splits are gone with the taper off.** The host waited more than 20 ms before 15 splits on cards 1 to 7
  in each fresh prompt, 9,128.5 ms in request 7 and 9,472.0 ms in request 10: the third micro-batch of calls 1 and 2 on
  each of those cards, and the fourth of call 1 on card 7, the same splits as R42's; with the switch none waits (0 ms in
  0 splits). Of request 7's 9,128.5 ms, 6,082.2 ms now come back as the host's wait
  for the cards after each call (13,133.6 → 19,215.8 ms) and 195.3 ms as the other splits' host time (376.1 → 571.4,
  est. the captures and executable updates); the rest of the window stays as it was (card 0 inside a call 1,101.3 →
  1,082.7 ms, the draft's catch-up 2,041.8 → 2,068.6, after the draft 2,002.5 → 1,993.9, checkpoint saves 1,275.0 →
  1,262.7). With the taper on, fewer splits block, in three micro-batches per prompt (4, 12 and 18 of request 8):
  15,367.2 ms in 54 splits → 8,446.4 in 34 (request 8), 14,648.0 in 49 → 7,599.2 in 29 (request 9).
- **The cards are busier:** the nine cards' busy shares add up to 1.711 and 1.759 cards in the two fresh prompts with
  the switch, against 1.537 and 1.580 without it (taper on: 2.358 and 2.335 against 2.068 and 2.018). Each card's GPU
  time is the same; the window is shorter.
- **The graphs, as designed ([counts_split.tsv](counts_split.tsv)):** with the switch every main-model prompt evaluation
  after each card's first that has more than 8 tokens is captured: 621 captures, 0 replays (every micro-batch's
  properties differ from the last one's, so each split is captured and its executable updated once). The other 90
  eager prompt evaluations after each card's first are the 4-token last micro-batch of each of the 10 requests on the 9
  cards ([counts_eager_on.tsv](counts_eager_on.tsv)), which the switch leaves to the warm-up by design (8 tokens or
  fewer). Generation's main-model evaluations are the same in both arms: 126 eager, 36 captures, 9 replays. The
  draft's catch-up passes in the prompt windows (128 rows, so the switch takes them): 154 captures, 146 replays, 28
  eager with the switch, against 26, 131 and 171 without it; the draft's evaluations outside them the same in both
  arms (27 captures, 100 replays, 45 eager). The server logs the switch's line at load:
  `GGML_CUDA_GRAPHS_FORCE: batched graphs are captured at every change (devices with VMM)`.
- **Where nothing queues, the switch costs nothing measurable:** the host-profile load waits for each card's previous
  evaluation before the next one, so the launch queue never fills; there the two fresh prompts take 33,521.3 and
  32,827.9 ms with the switch against 33,564.1 and 32,716.2 without it (taper on 34,662.5 and 35,227.8 against 34,627.3
  and 35,173.3).
- **The long prime in the identity lane (one load each, a diagnostic):** a 32,704-token prompt on an empty cache with
  the taper on took 225,220.60 ms with the switch against 287,635.46 and 287,465.68 eagerly (145.21 against 113.70 and
  113.77 tok/s). Generation is unchanged where it repeats (the second request at 32,768: 158.35 ms a step against
  160.93 and 158.05); the first request after the prime generated its 16 tokens in 2,587.137 ms against 2,477.414 and
  2,474.836 ms, in one load.
- **Memory:** the cards hold 4 MiB more after a load with the switch (cards 0 to 5 and 7; card 6 2 MiB, card 8 none).
  The host's lowest MemAvailable per load: 169, 153, 162 MiB with the switch (`warm4096`, `graph4096`, `host4096`)
  against 250, 168, 175 without; in the identity lane 409 MiB with it against 434 and 411.

The switch therefore goes into the v4 preset candidate (R70) and its agent check (R72). The remaining blocked splits
with the taper on are R98's to read (the micro-batch trace with the switch on).

## How it was measured

1. **Identity, untimed** (`bench/deep_lane`): three loads of the v4 candidate in the order eager, with the switch,
   eager again, each with `LLAMA_SPEC_DUMP` (every verify step), greedy, `N_PREDICT=16`, depths 0 and 32,768 (the
   32,768-token prime runs its micro-batches through captured graphs), the RAM watchdog at `MIN_AVAIL_MB=60`.
   `deep_table.py`'s checks compare the greedy replies; [run/dump_compare.py](run/dump_compare.py) pairs the three dumps
   step by step without the fields that name the process (`run`, `step`, `task`).
2. **R42's driver** ([r42_run.sh](../2026-11-10_prompt_timeline/run/r42_run.sh)) on the same build, three loads per arm
   (`warm4096` unprofiled, `graph4096` under `GGML_CUDA_PROFILE_MODE=graph`, `host4096` under
   `GGML_CUDA_PROFILE_HOST`), first with the switch off, then with `GGML_CUDA_GRAPHS_FORCE=1` exported. Each load is
   the replay of one agent-check task (its reset and five turns, eight tokens each) and two fresh 4,097-token prompts,
   each with the taper off and on: request 7 is the first prompt with the taper off, 8 and 9 the two with it on, 10 the
   second with it off (read for the second time, after request 9).
3. **The tables**, on a build machine's CPU: R42's [timeline.py](../2026-11-10_prompt_timeline/run/timeline.py) per arm
   (window from the graph profile, GPU time from the host profile), and [run/r82_counts.py](run/r82_counts.py), which
   splits the graph profile's evaluations into each card's first prompt evaluation, the prompt windows and the rest,
   keeping the `how` of evaluations whose GPU time was not ready (timeline.py counts those as "not ready").

A blocked split is timeline.py's: a split on a card other than 0 before which the host waited more than 20 ms since
the previous evaluation. The class follows the project's rule: a gain when the arms' difference is beyond the larger of
the two arms' spreads, here the two fresh prompts of each load.

## Files

| File | What |
|---|---|
| [lane/table.md](lane/table.md), [lane/classes.tsv](lane/classes.tsv), [lane/requests.csv](lane/requests.csv), [lane/loads.tsv](lane/loads.tsv) | the identity lane: `deep_table.py`'s tables and checks, every request, each load's line |
| [lane/dump_compare.tsv](lane/dump_compare.tsv) | the 40 verify steps: whether the eager loads agree and the switch's load is the same, and which fields differ |
| [tables-off/](tables-off/), [tables-on/](tables-on/) | timeline.py's tables per arm: `requests.tsv`, `waits.tsv` (the window's parts), `busy.tsv`, `splits.tsv`, `phases.tsv`, `counts.tsv`, `summary.txt` |
| [counts_split.tsv](counts_split.tsv) | evaluations per arm by kind (main, draft), part (each card's first prompt evaluation, prompt, other) and how |
| [counts_eager_on.tsv](counts_eager_on.tsv) | with the switch: every eager main prompt evaluation after a card's first (request, micro-batch, card, launch groups) |
| [run/r82_run.sh](run/r82_run.sh) | the run: the lane, its wait and collection, and R42's driver per arm |
| [run/dump_compare.py](run/dump_compare.py), [run/r82_counts.py](run/r82_counts.py) | the dump comparison and the split counts |

## Commands

From a machine that only runs ssh, at the repository's root, with `RIG`, `WORK`, `BUILDS`,
`BUILD=llama.cpp-20261111-c408dabcb-pascal-p106-v4-b12246-night`, `MODELS`, `R42`, `R43` and `RECORDED` set (the
variables are in the script's header):

```bash
STEP=lane bash docs/data/2026-11-11_r82_c702_p106/run/r82_run.sh
STEP=lanewait bash docs/data/2026-11-11_r82_c702_p106/run/r82_run.sh
STEP=lanecollect OUT=<here>/lane bash docs/data/2026-11-11_r82_c702_p106/run/r82_run.sh
STEP=r42 ARM=off bash docs/data/2026-11-11_r82_c702_p106/run/r82_run.sh   # ends with $WORK/r42-off/DONE
STEP=r42 ARM=on bash docs/data/2026-11-11_r82_c702_p106/run/r82_run.sh    # after r42-off/DONE
ssh $RIG "tar -C $WORK -czf - r42-off r42-on" > r42.tar.gz
```

Then on a build machine's CPU, in the folder that holds `lane/`, `r42-off/` and `r42-on/`:

```bash
python3 -I run/dump_compare.py --control lane/dump-eager.jsonl --control2 lane/dump-eagerb.jsonl \
  --arm lane/dump-graphs.jsonl --out lane/dump_compare.tsv
for arm in off on; do
  python3 -I timeline.py --load warm4096=r42-$arm/warm4096 --load graph4096=r42-$arm/graph4096 \
    --load host4096=r42-$arm/host4096 --pair graph4096=host4096 --out tables-$arm > tables-$arm/summary.txt
done
python3 -I run/r82_counts.py --load off=r42-off/graph4096 --load on=r42-on/graph4096 --out counts_split.tsv \
  --odd counts_odd.tsv   # counts_eager_on.tsv = its "on" rows
```

The lane ran 14:56:12 to 15:26:14 (about ten minutes a load), the driver 15:26:49 to 15:52:38 with the switch off and
15:52:47 to 16:17:55 with it on (about eight and a half minutes a load). Every load started on a quiet rig.

## Provenance and licence

The recorded agent text that the lane cuts its prompts from, the dumps (which hold token ids of that text) and the
server logs are not published; the tables hold counts, times and ids only. Tables and scripts are this repository's
licence; the build is the fork's (MIT, as llama.cpp).
