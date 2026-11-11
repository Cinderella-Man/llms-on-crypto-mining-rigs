# C704's in-place indexer on a P106: decode at depth with the gather gone (R77, Q1282; 2026-11-11)

What this measured: what `LLAMA_QSA_INDEXER_ROWS=1` does on nine P106-090 cards. The switch is C704's in-place
indexer ([2026-11-11_c704_indexer_gather](../2026-11-11_c704_indexer_gather/README.md), Q1244): the k-pool indexer
reads each pooled key where it lies in the cache, instead of gathering every pooled key into a new tensor before each
`LIGHTNING_INDEXER`. It is timed in generation and prompt reading at context depths 0, 16,384, 32,768 and 49,152 tokens,
and one profiled load cuts its step into parts beside R51's
([2026-11-10_step_parts_depth](../2026-11-10_step_parts_depth/README.md)). The line is the v4 preset candidate (R70's
five switches, [2026-11-10_v4_candidate](../2026-11-10_v4_candidate/README.md)) on the Miner Llama v3 release preset
([release/flash-next-v3/settings.env](../../../release/flash-next-v3/settings.env): Qwen3.8-Flash-Next, Swift Q2_0,
the MTP draft of three tokens, a 65,536-token context). The build is the v4 night merge
`pascal-p106-v4-b12246-night` `c408dabcb` ([2026-11-11_v4_night_merge](../2026-11-11_v4_night_merge/README.md)), which
holds C704 behind its switch with every other switch off. The run used
[bench/deep_lane](../../../bench/deep_lane/README.md). Registry row Q1282 in
[quants/experiments.md](../../../quants/experiments.md).

| Arm | What it is |
|---|---|
| `v4c` (control) | the release preset with R70's five switches: `GGML_CUDA_FATTN_SPARSE_PASCAL=1 GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=1.0 GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=5`, `--spec-draft-sampling probabilistic --spec-draft-p-min 0 --ctx-checkpoints-vram 4 --ubatch-taper on --spec-draft-n-ubatch 512` |
| `rows` | `v4c` plus `LLAMA_QSA_INDEXER_ROWS=1` |
| `prof` | `rows` plus `GGML_CUDA_PROFILE` in group mode (one load, for the parts; its times carry the profiler's cost) |

The CUDA kernels' check on a P106 is the night merge's replay: run `lid` passes all 257 of its cases reached on the
card ([2026-11-11_v4night_replay](../2026-11-11_v4night_replay/README.md)). These are single-request diagnostics at a
known depth. They do not replace the agent check, which decides what goes into a preset.

## Result

**The in-place indexer is a gain in decode from 16,384 tokens on and changes no token.** The step on the repeated
request takes 168.94 and 169.16 ms in its two loads against 179.58 and 180.15 ms at 49,152 tokens (−10.81 ms,
−6.01 %), and decode over the four depths runs at 20.87 and 20.87 tok/s against 20.12 and 20.12 (+3.74 %). At depth 0,
where there is nothing to gather, the step is negligible. Prompt reading is negligible at every depth. Every greedy
reply is the same text in both arms, on every pass and load, at all four depths, with the same draft counters.

The profiled load shows why. The gather (`GET_ROWS` before the indexer) is gone from the main pass and from the
draft's passes at every depth: 0.000 ms a step against R51's 4.974, 9.837 and 14.685 ms at 16,384, 32,768 and 49,152
tokens. The indexer's scores, which now read the keys in place, take longer: 2.664, 5.679 and 8.454 ms against R51's
1.980, 3.691 and 5.392 (+0.684, +1.988, +3.062). The step's GPU time falls by 4.81, 8.67 and 12.47 ms (R51's 134.12,
140.59, 148.10 to 129.31, 131.92, 135.63). The unprofiled loads of this run fall by 4.24, 7.11 and 10.81 ms, 88 %,
82 % and 87 % of that. R51 ran on another rig of the same make, with the release build and the sparse attention
alone, so its parts other than the indexer's sit within 1 ms of this run's (`against_r51.tsv`). The slower read of
the scores is a new row (Q1342): est. up to 3.06 ms a step at 49,152 to win back.

Memory: cards 0 to 7 hold the same MiB in both arms after loading (5,059, 5,095, 4,963, 5,111, 4,879, 4,979, 4,381,
4,991) and at their highest in the run (5,165, 5,193, 5,069, 5,209, 4,975, 5,085, 4,463, 5,097). Card 8 holds 4,785
against 4,793 MiB after loading and 4,879 against 4,887 at its highest (−8 MiB). Every `rows` load, and none of the
`v4c` loads, ends with llama.cpp's warning "CUDA8 compute buffer size of 195.1185 MiB, does not match expectation of
194.5345 MiB" as a context is freed at the end. The line does not name the context. It is est. the MTP draft's: the
main model's card-8 compute buffer is 280.37 MiB in [R57's](../2026-11-11_r57_v4night_p106/README.md) `-lv 4` load of
this build. A graph on the rows path needed 0.584 MiB more than the graph reserved at load, so the buffer grew during
the run. That is a new row (Q1343). The host's lowest MemAvailable is 442 and 441 MiB (`rows`) against 443 and 439
(`v4c`), with no swap growth and no memory pressure. The compute buffers per card were not read: llama-server's default verbosity hides them, and the untimed
`-lv 4` loads that read them were stopped before printing them (Files), left to R100.

Every value is one load. Run on rig 2 on 2026-11-11 from 10:33:30 to 12:05:17 (92 minutes): a warm-up load (6.7
minutes), the four timed loads `v4c1`, `rows1`, `v4c2`, `rows2` (15.0, 15.0, 15.1 and 15.0 minutes), each started
after a quiet check, then the profiled load (25.1 minutes).

### Decode and prompts by depth

Decode: 256 greedy tokens twice per depth (pass 1, the first request at a new depth; pass 2, the same request again).
A load's speed is both passes' tokens over their time. The step is generation time over steps, on pass 2. Prompts:
the prime's chunk, read on top of the previous depth's checkpoint (at depth 0, the 64-token tail on an empty cache).
Class: gain or loss when the arms' means differ by more than the larger of the two arms' spreads between their loads,
else negligible ([rows_classes.tsv](rows_classes.tsv)).

| Depth (tokens) | Decode tok/s, `v4c` | `rows` | Class | Step on pass 2, ms, `v4c` | `rows` | Class | Prompt tok/s, `v4c` | `rows` | Class |
|---:|---|---|---|---|---|---|---|---|---|
| 0 | 27.04, 27.06 | 27.07, 27.09 | gain (+0.12 %) | 121.58, 121.62 | 121.62, 121.56 | negligible (−0.01 %) | 49.0, 49.5 | 49.6, 49.6 | negligible (+0.68 %) |
| 16,384 | 24.00, 24.00 | 24.66, 24.73 | gain (+2.91 %) | 149.29, 149.46 | 145.34, 144.94 | gain (−2.84 %) | 135.4, 134.6 | 134.2, 134.5 | negligible (−0.52 %) |
| 32,768 | 15.26, 15.27 | 15.94, 15.92 | gain (+4.38 %) | 164.16, 164.28 | 156.79, 157.42 | gain (−4.33 %) | 100.2, 100.3 | 100.3, 100.2 | negligible (+0.03 %) |
| 49,152 | 18.31, 18.27 | 19.43, 19.43 | gain (+6.23 %) | 179.58, 180.15 | 168.94, 169.16 | gain (−6.01 %) | 77.4, 77.3 | 77.5, 77.4 | negligible (+0.09 %) |
| all | 20.12, 20.12 | 20.87, 20.87 | gain (+3.74 %) | 154.59, 154.81 | 148.93, 149.07 | gain (−3.68 %) | 98.7, 98.6 | 98.6, 98.6 | negligible (−0.07 %) |

The step on pass 1 (the first request at a new depth): 125.81 and 125.66 against 125.54 and 125.39 ms at 0; 151.23
and 151.07 against 147.11 and 146.70 at 16,384; 181.69 and 181.41 against 174.26 and 174.06 at 32,768; 183.62 and
183.73 against 173.22 and 173.10 at 49,152. The decode gain at depth 0 (+0.03 tok/s) is the arms' 0.02 tok/s spread
exceeded by 0.01; the step there is negligible on pass 2. The draft accepts the same tokens in both arms at every
depth (178 of 229 and 181 of 220 at 0; 185 of 210 at 16,384; 159 of 287 at 32,768; 179 of 227 at 49,152).

### The profiled load beside R51, ms a step, pass 2

Each bucket summed over the main pass on cards 0 to 7, the main pass on card 8 and the draft's four passes;
[depth_parts.md](depth_parts.md) has each group and card. R51 is the release build with the sparse attention alone
(rig 1, 2026-11-10); this run is the night build with `rows`.

| Bucket | 16,384: R51 | R77 | 32,768: R51 | R77 | 49,152: R51 | R77 |
|---|---:|---:|---:|---:|---:|---:|
| the gather of every pooled key (`GET_ROWS` before the indexer) | 4.974 | 0.000 | 9.837 | 0.000 | 14.685 | 0.000 |
| the indexer's scores (`LIGHTNING_INDEXER`) | 1.980 | 2.664 | 3.691 | 5.679 | 5.392 | 8.454 |
| the indexer's top-k and mask | 0.898 | 0.902 | 0.981 | 0.997 | 1.078 | 1.096 |
| attention (`FLASH_ATTN_EXT`) | 14.811 | 14.718 | 14.775 | 14.592 | 15.060 | 14.960 |
| everything else (with the output heads and the indexer's other groups) | 111.456 | 111.027 | 111.301 | 110.646 | 111.884 | 111.120 |
| **GPU time of the step** | 134.12 | 129.31 | 140.59 | 131.92 | 148.10 | 135.63 |

The scores' launch groups per card: on the main pass's cards 0 to 7, 6.76 ms at 49,152 against R51's 4.279; on card 8,
0.61 against 0.391; on the draft, 1.08 against 0.722. Between 16,384 and 49,152 tokens the step's GPU time now grows
by +6.32 ms (pass 2), of which the scores take +5.79; R51's grew by +13.98.

## Files

| File | What |
|---|---|
| [summary.md](summary.md) | per depth and load: decode, the step on both passes, the draft's counters, prompts; the classes; the host's memory per load ([run/v4c_table.py](../2026-11-10_v4_candidate/run/v4c_table.py) with `--control v4c --arm rows`) |
| [rows_classes.tsv](rows_classes.tsv), [per_load.tsv](per_load.tsv) | the same classes and per-load values as tables |
| [table.md](table.md), [classes.tsv](classes.tsv) | the driver's own tables (`deep_table.py`), with the greedy reply checks |
| [requests.csv](requests.csv), [loads.tsv](loads.tsv) | one line per request (timings, draft counters, a hash of the reply) and per load (its line and switches; the profile path written as `$OUT`) |
| [depth_parts.md](depth_parts.md) | the profiled load cut by `depth_parts.py` beside R49's shallow trace: the step, its buckets, its parts per group, the launch groups that grow most, per card |
| [buckets.tsv](buckets.tsv), [totals.tsv](totals.tsv), [parts.tsv](parts.tsv), [groups.tsv](groups.tsv), [per_card.tsv](per_card.tsv), [steps.tsv](steps.tsv), [requests.tsv](requests.tsv) | `depth_parts.py`'s tables |
| [against_r51.tsv](against_r51.tsv) | the buckets and step totals beside R51's, per request ([run/against_r51.py](run/against_r51.py)) |
| [card_peaks.tsv](card_peaks.tsv) | each card's highest MiB per load, from the driver's samples every 10 s ([run/card_peaks.py](run/card_peaks.py)) |
| [run/lv4_loads.sh](run/lv4_loads.sh) | one untimed `-lv 4` load per arm on the run's exact line, for the buffer lines per card; started at 12:05:51 and stopped at 12:08:04 before its first load printed its compute buffers, to leave the rig to the next row (R100 runs it) |

## Commands

From a machine that only runs ssh, at the repository's root. `RIG` is the rig's ssh alias, `WORK` a folder on the rig
with ~150 MB free (the trace is 116 MB), `BIN` the night build's `bin/` on the rig, `MODELS` the folder with Swift
Q2_0's shards and the MTP draft, `RECORDED` the release check's recording (`proxy.jsonl.bodies.gz`, the source of the
prompts' text), `OUT` a results folder here:

```bash
V4C_ENV="GGML_CUDA_FATTN_SPARSE_PASCAL=1 GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=1.0 GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=5"
V4C_ARGS="--spec-draft-sampling probabilistic --spec-draft-p-min 0 --ctx-checkpoints-vram 4 --ubatch-taper on --spec-draft-n-ubatch 512"
RIG=$RIG WORK=$WORK BIN=$BIN/llama-server \
  MODEL=$MODELS/swift_q2_0/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf \
  MODEL_DRAFT=$MODELS/mtp-Qwen3.8-Flash-Next-draft.gguf \
  TEXT_LOCAL=$RECORDED/proxy.jsonl.bodies.gz TAG='^t1swift/' WD_ENV=MIN_AVAIL_MB=60 PAUSE=4 \
  ARMS="v4c rows prof" \
  ARM_ENV_v4c="$V4C_ENV" ARM_ARGS_v4c="$V4C_ARGS" \
  ARM_ENV_rows="$V4C_ENV LLAMA_QSA_INDEXER_ROWS=1" ARM_ARGS_rows="$V4C_ARGS" \
  ARM_ENV_prof="$V4C_ENV LLAMA_QSA_INDEXER_ROWS=1 GGML_CUDA_PROFILE=$WORK/out/prof-rows.csv GGML_CUDA_PROFILE_MODE=group" \
  ARM_ARGS_prof="$V4C_ARGS" \
  LOADS="warm:v4c v4c1:v4c rows1:rows v4c2:v4c rows2:rows prof:prof" WARM_DEPTHS="16384" DEPTHS="0 16384 32768 49152" \
  bash bench/deep_lane/remote.sh start
# one waiting command (here a watch on the console that exits with the driver), then:
ssh $RIG "cd $WORK/out && nice -n 19 gzip -k -9 prof-rows.csv"
ssh $RIG "tar -C $WORK/out --exclude=prof-rows.csv -cf - ." | tar -C $OUT -xf -
# the buffer lines per card (R100): one more untimed load per arm at -lv 4, on the run's own lines
RIG=$RIG LANE=$OUT RIGDIR=$WORK/lv4 OUT=$OUT/lv4 LOADS="v4c1 rows1" bash docs/data/2026-11-11_r77_c704_p106/run/lv4_loads.sh
```

The tables, on a build machine's CPU (`nice -n 19`; 17 s for the cut), with `TRACE` the gunzipped `prof-rows.csv`,
`SHALLOW` R49's trace (`prof-verify.csv`, md5 `92a891f032d1aa9c6278f0e59acf34ce`) and `RES` a folder for the tables:

```bash
R=docs/data
nice -n 19 python3 -I $R/2026-11-10_step_parts_depth/run/depth_parts.py \
  --verify-parts $R/2026-11-10_served_verify_parts/run/verify_parts.py --kpass-parts $R/2026-11-04_kpass_profile/run/kpass_parts.py \
  --event-us 7.6 --shallow $SHALLOW --shallow-label R49 --shallow-depth "79 to 335" \
  --deep $TRACE --requests $OUT/requests.csv --load prof --ref-load rows1 --out $RES > $RES/depth_parts.md
nice -n 19 python3 -I $R/2026-11-10_v4_candidate/run/v4c_table.py $OUT --control v4c --arm rows \
  --md $RES/summary.md --tsv $RES/per_load.tsv --classes $RES/rows_classes.tsv
nice -n 19 python3 -I $R/2026-11-11_r77_c704_p106/run/against_r51.py $R/2026-11-10_step_parts_depth $RES > $RES/against_r51.tsv
nice -n 19 python3 -I $R/2026-11-11_r77_c704_p106/run/card_peaks.py $OUT > $RES/card_peaks.tsv
```

The trace as run: `prof-rows.csv` 116,103,475 B, md5 `f15ddaaa41d396cce843d87d2e67bf19` (`gzip -9`: 13,967,202 B, md5
`beb522d976aba66402b72a1a9d98dc72`); kept with the run's raw logs, not here (over 3 MB).

## Provenance

Code: the fork's branch `pascal-p106-v4-b12246-night` (MIT, as llama.cpp); the scripts here and in `bench/deep_lane`
(this repository's licence). The prompts are cut from the recorded text of this project's own release check (the
agent check's requests on its bugfix tasks), as in R70 and R51; no reply text is published, only a hash per reply.
