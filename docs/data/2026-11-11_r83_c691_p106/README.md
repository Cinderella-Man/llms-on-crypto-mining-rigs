# R83 (Q1300): C691's one top-k over the output rows on a P106 (2026-11-11)

What this measured: C691's change to the backend samplers
([2026-11-11_c691_sampling_launches](../2026-11-11_c691_sampling_launches/README.md), switch
`LLAMA_SAMPLING_TOP_K_ROWS`, [docs/switches.md](../../switches.md)) natively on P106-090 cards. With `-bs` the card
that holds the output head (card 8 of nine) samples every output row of a pass. Without the switch it first copies the
logits into a padded tensor (`PAD`), then runs one `TOP_K` per row. With the switch on it runs one `TOP_K` over all the
rows and no `PAD`. Three questions: does every row keep its candidates; how much of the verify pass's sampling chain
(R49: 0.68 ms, [2026-11-10_served_verify_parts](../2026-11-10_served_verify_parts/README.md)) is left; and what does
the switch do to the served step. The registry row is Q1300 in [quants/experiments.md](../../../quants/experiments.md).

The build is Miner Llama's v4 night merge, `pascal-p106-v4-b12246-night` at `c408dabcb`
([2026-11-11_v4_night_merge](../2026-11-11_v4_night_merge/README.md)), which holds C691's branch. Only this switch was
set; every other switch the merge adds was off, so `off` computes what `pascal-p106-v4` computes. The native replay is
not repeated here: the merge's replay on a P106 ran C691's two runs, `tk` (39 `TOP_K` cases at 248,320 columns against
a CPU) and `tk-rows` (75 `TOP_K_ROWS` cases: one op's rows against each row's own top-k on the card), and passed 39 of
39 and 75 of 75 ([2026-11-11_v4night_replay](../2026-11-11_v4night_replay/README.md)).

## Result

**Every row keeps its candidates, the sampling chain halves, and the step is a gain.**

- **The candidates:** on a P106 each row of one multi-row `TOP_K` is the row's own top-k in all 75 `tk-rows` cases of
  the night replay. In the served loads all 16 replies carry the same text (checksum `e460d038a1f6`, the text of R49
  on Miner Llama v3.0), with the same accepted drafts: 170 of 255 on request 1 and 172 of 247 on request 2, in every
  load.
- **The sampling chain:** with the switch the head card's chain takes 0.323, 0.314, 0.317 and 0.316 ms a verify pass
  (two loads, two requests each), against 0.650, 0.655, 0.651 and 0.651 without it. R49 measured 0.68 and 0.68 on
  v3.0, and C691 estimated 0.29 left. The chain has one `TOP_K` (0.2231, 0.2188, 0.2226 and 0.2192 ms) where it had four
  (0.2931, 0.2959, 0.2961 and 0.2968 together), and no `PAD` (0.2164, 0.2181, 0.2163 and 0.2177 ms without the switch).
- **The draft's passes gain too:** the switch is read per context, so each pass of the MTP draft (on the same card)
  also loses its `PAD`: 505 of them a load, 87.90 and 89.19 µs each on average (the two draft graph shapes, raw).
  Over a whole profiled load, card 8's `PAD`, `TOP_K` and top-k gathers take 0.508 and 0.509 ms per generation step
  with the switch, against 1.068 and 1.067 without.
- **The step (warm loads, the server's own timings), request 2:** 121.05 and 121.26 ms a step with the switch, against
  121.78 and 121.98 without (−0.72 ms, −0.6 %; both `on` loads below both `off` loads, spread 0.21): a **gain**. Request 1 (the load's first, with its one-off launches): 127.07 and 127.53 against 129.21 and 131.63 (−3.12
  ms; the `off` loads lie 2.42 apart): a gain too. C691 estimated −0.39 ms (0.3 %) from the verify pass alone.

The switch goes into the v4 line's stacked gains (R90).

### The operator times

[perf/](perf/): `test-backend-ops perf` on card 0, two passes; `TOP_K` at 248,320 columns with k = 20 (Flash-Next's
vocabulary, the presets' `--top-k 20`), and the `PAD` of four such rows to five.

| Op | Rows | Pass 1, µs | Pass 2, µs |
|---|---:|---:|---:|
| `TOP_K` | 1 | 109.15 | 105.55 |
| `TOP_K` | 2 | 142.41 | 145.14 |
| `TOP_K` | 4 | 227.05 | 227.38 |
| `PAD` | 4 | 213.59 | 214.96 |

- **What a verify pass's four chains launch** ([chain.tsv](perf/chain.tsv)): without the switch, four one-row `TOP_K`
  and the `PAD`, 650.19 and 637.16 µs; with it, one four-row `TOP_K`, 227.05 and 227.38 µs. The difference is 423.14
  and 409.78 µs.
- **Four rows cost 2.08 and 2.15 times one row.** C691 estimated 0.165 ms for the four-row launch, half of four
  one-row ones; it took 227.05 and 227.38 µs here, and 0.2231, 0.2188, 0.2226 and 0.2192 ms in the served passes, so
  the chain keeps about 0.03 ms more than estimated.
- The random rows of `test-backend-ops` take longer per row than the model's logits: in the served `off` passes a
  one-row `TOP_K` takes 0.0733, 0.0740, 0.0740 and 0.0742 ms on average (each pass's four, below, over four).

### The served verify pass

[served/](served/): R49's profile of a served verify pass
([2026-11-10_served_verify_parts](../2026-11-10_served_verify_parts/README.md), `run/verify_prof.sh`). The v3 preset
runs exactly: UkisAI's Swift-1.5 GSQ-RCO Q2_0 of Qwen3.8-Flash-Next on nine P106-090 cards, with the MTP draft at
three tokens. Four loads ran in the order `off`, `on`, `on`, `off`. Each load is a `warm` load (no profile) and then a
`prof` load (the fork's profiler in group mode), and each answers the greedy request twice. `on` added
`LLAMA_SAMPLING_TOP_K_ROWS=1` to the preset's switches.

**Card 8 per verify pass, ms** ([chain.tsv](served/chain.tsv), [after_ops.tsv](served/after_ops.tsv);
`verify_parts.py`'s cut, the profiler's 7.6 µs subtracted per launch group; requests 1 and 2 of each profiled load):

| Load | Head's product | Sampling chain after it | Launch groups after it | `TOP_K` in the chain | `PAD` | The pass, all cards (sum of the parts) | Median raw GPU time of a typical pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 `off` | 8.107, 8.141 | 0.650, 0.655 | 86, 86 | 4: 0.2931, 0.2959 | 0.2164, 0.2181 | 101.056, 101.127 | 118.90, 119.03 |
| 2 `on` | 8.064, 8.068 | 0.323, 0.314 | 79, 79 | 1: 0.2231, 0.2188 | none | 100.653, 100.659 | 118.49, 118.42 |
| 3 `on` | 8.069, 8.085 | 0.317, 0.316 | 79, 79 | 1: 0.2226, 0.2192 | none | 100.671, 100.727 | 118.52, 118.58 |
| 4 `off` | 8.109, 8.113 | 0.651, 0.651 | 86, 86 | 4: 0.2961, 0.2968 | 0.2163, 0.2177 | 101.050, 101.080 | 118.87, 118.94 |

- **The chain is 0.334 ms shorter** (the means of the four `on` and the four `off` values), and the pass's sum 0.401
  ms. The four `off` chains lie within 0.005 ms of each other and the four `on` within 0.009.
- **The other groups after the head** keep their number, but for the gathers: `GET_ROWS` 24 → 21 a pass (each chain's
  own gather of its top-k logits becomes one over the rows; 0.0552, 0.0467, 0.0501, 0.0511 → 0.0325, 0.0268, 0.0251,
  0.0276 ms). The top-p, temperature and argmax groups are the same per row.
- **The head's product** reads 8.064, 8.068, 8.069 and 8.085 ms with the switch against 8.107, 8.141, 8.109 and 8.113
  without; the switch does not change its kernel.

**Card 8 over each whole profiled load** ([card/](card/), `run/card_ops.py`: every launch group of the backend
samplers' `PAD`, `TOP_K` and its gather on card 8, in every graph, the 7.6 µs subtracted per group; 167 generation
steps a load):

| Load | `PAD` of the main model's passes | `PAD` of the draft's passes | `TOP_K` (top_k) | Its gathers | Groups | Net, ms | ms per generation step |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 `off` | 171: 37.218 | 505: 41.146 | 1,178: 96.554 | 1,178: 3.439 | 3,032 | 178.357 | 1.068 |
| 2 `on` | none | none | 676: 82.474 | 676: 2.359 | 1,352 | 84.833 | 0.508 |
| 3 `on` | none | none | 676: 82.600 | 676: 2.373 | 1,352 | 84.973 | 0.509 |
| 4 `off` | 171: 37.132 | 505: 41.106 | 1,178: 96.686 | 1,178: 3.286 | 3,032 | 178.209 | 1.067 |

- The draft's graphs on card 8 (the evaluations with the draft's `mtp_` groups) hold 505 of the `PAD`s, 3.02 per
  generation step (the cut counts four draft passes a step; the draft samples three tokens). With the switch, a one-row chain's top-k reads the logits unpadded too.
- The per-step column divides the whole load (its prompts included) by its generation steps.

**The replies and the step** ([replies.tsv](served/replies.tsv), [steps.tsv](served/steps.tsv); ms a step = the
reply's generation time over its steps, steps = generated tokens less accepted drafts):

| Load | Request 1: ms a step | tok/s | Request 2: ms a step | tok/s |
|---|---:|---:|---:|---:|
| 1 `off` | 129.21 | 22.95 | 121.78 | 24.93 |
| 2 `on` | 127.07 | 23.33 | 121.05 | 25.08 |
| 3 `on` | 127.53 | 23.25 | 121.26 | 25.03 |
| 4 `off` | 131.63 | 22.53 | 121.98 | 24.89 |

- **The class, per request:** the spread is the larger of the two `off` loads' and the two `on` loads' differences; a
  gain when both `on` values are below both `off` values and the means differ by more than the spread. Request 2:
  −0.72 ms, spread 0.21: gain. Request 1: −3.12 ms, spread 2.42: gain.
- The step falls by more than the verify pass's chain (0.334 ms): the draft's passes lose their `PAD` too (card 8's
  sampler groups 0.559 ms less per generation step, the table above), and, est., the host launches fewer kernels a
  step (a `TOP_K` launch group is ten kernels).
- **Every reply carries one text** (`e460d038a1f6`), and every load accepts the same drafts (170 of 255, 172 of 247).
  At temperature 0 the token is the argmax, so the text alone does not test the candidates; the replay's `tk-rows`
  does.
- The profiled loads' server timings are slower and not used for the class (the profiler records an event between
  every two launch groups and synchronizes every evaluation): request 2 154.09 and 156.10 ms a step `off`, 158.42 and
  153.48 `on`.
- These are single-request diagnostics. The switch's agentic speed is measured with the other gains of the night
  merge (R90).

**The switch's log line.** `rig_c691.sh` counts the engine's line ("one top-k over the output rows") in each load's
server log: 0 in every load. llama-server runs at its default log verbosity (3), which prints none of the library's
`llama_context` lines, in any load. The switch reached the `on` servers (it is in `rig_server.sh`'s start line of both
`on` loads, [switch.tsv](served/switch.tsv)), and the profiles show it acting: one `TOP_K` and no `PAD` in both `on`
loads, four and a `PAD` in both `off` loads.

## What ran

One detached script on one rig ([rig_c691.sh](../2026-11-11_c691_sampling_launches/run/rig_c691.sh), steps
`perf served`), nothing else on the rig; 12:47:32 to 13:35:48 on 2026-11-11:

| Step | Time | What |
|---|---|---|
| `perf` | 12:47:32–12:47:48 | `test-backend-ops perf -o TOP_K` (1, 2, 4 rows) and `-o PAD` (4 rows) at 248,320 columns, two passes, card 0 |
| `served` | 12:47:48–13:35:48 | four loads of `verify_prof.sh` (`warm`, `prof`), `off on on off` |

- **The host:** MemAvailable at least 3,133,232 kB before each step and each load; load average at most 1.22.
- **The loads** took 11 min 58 s, 11 min 58 s, 12 min 6 s and 11 min 58 s.
- **The tables ran on a build machine's CPU**, never on the rig: `run/perf_table.py`, R49's `verify_parts.py` on each
  profiled load, `run/served_table.py`, `run/card_ops.py`.

## Files

| File | What |
|---|---|
| [perf/us_run.tsv](perf/us_run.tsv) | every case's µs per launch: op, pass, case string, µs (no header) |
| [perf/perf.tsv](perf/perf.tsv) | the same per op and rows, one column per pass |
| [perf/chain.tsv](perf/chain.tsv) | per pass: four one-row `TOP_K` and the `PAD` against one four-row `TOP_K` |
| [perf/perf.md](perf/perf.md) | `perf_table.py`'s report |
| [served/chain.tsv](served/chain.tsv) | per load and request: the head's product, the sampling chain, the groups after the head, the `TOP_K` and `PAD` groups on card 8, the pass's sum, the median raw pass |
| [served/after_ops.tsv](served/after_ops.tsv) | per load and request: the launch groups after the head's product by op, groups and ms a pass |
| [served/replies.tsv](served/replies.tsv) | every reply: text checksum, drafts, accepted, steps, ms a step, tok/s |
| [served/steps.tsv](served/steps.tsv) | the warm loads' ms a step per request, the difference, the spread, the class |
| [served/switch.tsv](served/switch.tsv) | per load: the engine's switch lines in the server log, the switch in the start line |
| [served/served.md](served/served.md) | `served_table.py`'s report |
| [served/loads/](served/loads/) | per load: `verify_parts.py`'s report, its cut, its parts per card group, the buckets (`accounting.tsv`), per op, per card, the last card's groups in graph order |
| [card/card_ops.tsv](card/card_ops.tsv) | per load: card 8's `PAD`, `TOP_K` and gather groups over the whole load, by label |
| [card/card_ops_totals.tsv](card/card_ops_totals.tsv) | per load: their groups, net ms, generation steps, ms a generation step |
| [card/card_ops.md](card/card_ops.md) | `card_ops.py`'s report |
| [run/perf_table.py](run/perf_table.py) | the `perf` tables |
| [run/served_table.py](run/served_table.py) | the switch lines, replies, step classes and the sampling chain per verify pass |
| [run/card_ops.py](run/card_ops.py) | card 8's sampler groups over a whole load |

## Commands

`RIG` is the rig's ssh alias, `R` the run folder on the rig, `NIGHT` the night merge's unpacked tarball on the rig, `S`
a folder on the rig with this repository's files (the preset, `rig_server.sh`, the greedy request), `W` a folder on
the build machine.

```bash
# copy rig_c691.sh and verify_prof.sh to the rig; then on the rig, idle:
STEPS="perf served" BIN=$NIGHT/bin OUT=$R/out \
  MODEL=<Swift Q2_0 first shard> DRAFT=<the MTP draft> PRESET=$S/release/flash-next-v3/settings.env \
  RIG_SERVER=$S/bench/agent_check/rig_server.sh REQ=$S/docs/data/2026-11-04_draft_one_card/run/greedy_request.json \
  VERIFY_PROF=$R/verify_prof.sh setsid nohup bash $R/rig_c691.sh > $R/console.out 2>&1 < /dev/null &
# the tables, on the build machine (the rig's out/ copied to $W/out), from this repository's root
D=docs/data/2026-11-11_r83_c691_p106/run
nice -n 19 python3 -I $D/perf_table.py $W/out/perf/us_run.tsv --out $W/tables/perf > $W/tables/perf/perf.md
for d in 1-off 2-on 3-on 4-off; do
  nice -n 19 python3 -I docs/data/2026-11-10_served_verify_parts/run/verify_parts.py \
    --kpass-parts docs/data/2026-11-04_kpass_profile/run/kpass_parts.py --prof-summary scripts/prof_summary.py \
    --event-us 7.6 --out $W/tables/served/$d $W/out/served/$d/prof/prof-verify.csv > $W/tables/served/$d/report.md
done
nice -n 19 python3 -I $D/served_table.py $W/out/served --parts $W/tables/served --out $W/tables/served > $W/tables/served/served.md
nice -n 19 python3 -I $D/card_ops.py --device 8 --event-us 7.6 --parts $W/tables/served --out $W/tables/card \
  1-off=$W/out/served/1-off/prof/prof-verify.csv 2-on=$W/out/served/2-on/prof/prof-verify.csv \
  3-on=$W/out/served/3-on/prof/prof-verify.csv 4-off=$W/out/served/4-off/prof/prof-verify.csv > $W/tables/card/card_ops.md
```

## Provenance and licence

The model is UkisAI's Swift-1.5 GSQ-RCO Q2_0 of Qwen3.8-Flash-Next (its maker's licence) with its MTP draft layer. The
request is S37's greedy request
([greedy_request.json](../2026-11-04_draft_one_card/run/greedy_request.json)). The engine is Miner Llama (MIT, as
llama.cpp). The operator cases use random tensors generated by `test-backend-ops`. The tables are this project's
measurements.
