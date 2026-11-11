# v4's day build with every gain of 2026-11-11 against v3.0, decode and prompts at depth and the agent check (R90, Q1322; 2026-11-11)

What this measured: how much faster Miner Llama v4, as it stood on 2026-11-11, decodes and reads prompts than the v3.0
release, on the same model, preset and rig. Three arms, all on the v3 release preset
([release/flash-next-v3/settings.env](../../../release/flash-next-v3/settings.env): Qwen3.8-Flash-Next, Swift Q2_0,
the MTP draft of three tokens, a 65,536-token context) on nine P106-090 cards:

| Arm | Build | Settings |
|---|---|---|
| `v3` (control) | the v3.0 release, `pascal-p106-v3-b12076-ladder5` `1551f5fe1` | the preset as shipped |
| `off` | v4's day build, `pascal-p106-v4-b12246-day` `18d0fa309` ([2026-11-11_v4_day_merge](../2026-11-11_v4_day_merge/README.md)) | the preset as shipped, every switch at its default (off) |
| `gains` | the same day build | the preset plus the switches below |

| Switch | Its own result on a P106 |
|---|---|
| `GGML_CUDA_FATTN_SPARSE_PASCAL=1 GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=1.0 GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=5`, `--spec-draft-sampling probabilistic --spec-draft-p-min 0`, `--ctx-checkpoints-vram 4`, `--ubatch-taper on`, `--spec-draft-n-ubatch 512` | R70's five, the v4 preset candidate: decode +33.08 %, prompts +14.55 % over five depths on the v3.0 build ([2026-11-10_v4_candidate](../2026-11-10_v4_candidate/README.md), Q1254) |
| `GGML_CUDA_MMVQ_PASCAL_DMT=1` | C722's read-once IQ4_XS and Q4_K products of 2–4 tokens: the warm step 114.01 and 113.62 ms against 122.10 and 122.03 ([2026-11-11_r79_c722_p106](../2026-11-11_r79_c722_p106/README.md), Q1277) |
| `GGML_CUDA_MMVQ_PASCAL_Q5K_MT=1` | C690's four-row Q5_K head: the step 116.37 and 116.25 ms against 122.11 and 121.81 ([2026-11-11_r75_c690_p106](../2026-11-11_r75_c690_p106/README.md), Q1222) |
| `LLAMA_SAMPLING_TOP_K_ROWS=1` | C691's one top-k over the output rows: the step 121.05 and 121.26 ms against 121.78 and 121.98 ([2026-11-11_r83_c691_p106](../2026-11-11_r83_c691_p106/README.md), Q1300) |
| `LLAMA_QSA_INDEXER_ROWS=1` | C704's in-place indexer: the step at 49,152 tokens 168.94 and 169.16 ms against 179.58 and 180.15 ([2026-11-11_r77_c704_p106](../2026-11-11_r77_c704_p106/README.md), Q1282) |
| `LLAMA_QSA_KPOOL_DEV=1` | C100's k-pool tables on the cards: the step at 49,152 tokens 173.97 against 179.96 ms ([2026-11-11_r74_c100_p106](../2026-11-11_r74_c100_p106/README.md), Q1273) |
| `--ctx-checkpoints-incall` | C217's checkpoints inside a prompt's call: −18.99 % of prompt time with the taper off, −6.56 % with it on ([2026-11-11_r85_c217_p106](../2026-11-11_r85_c217_p106/README.md), Q1306) |

On this build `DMT=1` and `Q5K_MT=1` alone run the kernel settings R79 and R75 found fastest (R = 1, C758 and C766).
Registry row Q1322 in [quants/experiments.md](../../../quants/experiments.md).

## Result

**v4 with the day's gains is much faster than v3.0 in decode at every depth from 8,192 tokens on, and faster in prompt
reading at every depth.** Over all five depths the greedy decode runs 22.50 and 22.24 tok/s in `gains`' two loads
against v3.0's 15.87 and 14.77 (+45.97 %), and the four prompt chunks read at 97.0 and 97.6 tok/s against 87.7 and 88.0
(+10.74 %). At 49,152 tokens decode is 21.06 and 20.05 tok/s against 11.68 and 10.36 (+86.51 %), the step on the
repeated request 149.81 and 150.10 ms against 302.25 and 302.37 (−50.40 %). A load's 19 timed requests took 701.7 and
700.6 s against 864.1 and 878.4 s (−19.4 %).

**The day build alone (`off`) is negligible against v3.0 in decode and a gain in prompts:** decode over all depths 15.26
and 15.67 tok/s (+0.92 %, negligible), the step on the repeated request 201.05 and 201.25 ms against 204.19 and 205.44
(−1.79 %, a gain), prompts 91.5 and 91.5 tok/s (+4.17 %, a gain). **So nearly all of v4's decode gain is the switches:**
`gains` against `off` is +44.64 % in decode over all depths and +6.31 % in prompts.

**On the agent check (12 bug-fix tasks, one run an arm) `gains` is a gain over v3.0:** 12 of 12 passed against 11 of
12, 2.79 minutes a task at equal work against 3.08 (−9.4 %), and 1.96 against 2.27 (−13.7 %) on the 11 tasks v3.0 did
not run to the cap. v3.0's two earlier runs of the same check (rig 2) lie −1.6 % and −0.6 % from this one, so the gain
is beyond v3.0's own spread. Generation runs 24.85 tok/s against 20.66 (+20.28 %), prompts 73.9 against 65.5
(+12.82 %), the median first token per turn 4.55 s against 6.52. **Worse: long fresh prompts.** Turns of 1,024 or more
new prompt tokens read at 80.1 and 76.4 tok/s against v3.0's 90.5 and 93.3, so the first token's 90th percentile is
24.48 s against 16.17 (est. the micro-batch taper on long prompts, registry row Q1235).

Every value is one load or one run. Run on rig 1 on 2026-11-11: the deep lane 16:27:47–18:46:35 (a warm-up load, then
`v31 gains1 v32 gains2 off1 off2`; `off1` ran again after the RAM watchdog stopped its first start, below), then the
agent check, `gains` 18:46:55–19:25:36 and `v3` 19:25:36–20:08:31, the client on another machine at
`--cpu-shares 256`.

### Decode and prompts by depth

Decode: 256 greedy tokens twice per depth (pass 1, the first request at a new depth; pass 2, the same request again),
a load's speed being both passes' tokens over their time. The step: generation time over steps, on pass 2. Prompts:
the prime's chunk, read on top of the previous depth's checkpoint (at depth 0, the 64-token tail on an empty cache).
Classes against `v3`.

| Depth (tokens) | Decode tok/s, `v3` | `off` | Class | `gains` | Class | Step on pass 2, ms, `v3` | `off` | Class | `gains` | Class |
|---:|---|---|---|---|---|---|---|---|---|---|
| 0 | 26.92, 26.62 | 27.32, 26.99 | gain (+1.45 %) | 27.15, 27.10 | gain (+1.33 %) | 123.66, 123.70 | 121.64, 122.03 | gain (−1.49 %) | 108.42, 108.80 | gain (−12.19 %) |
| 8,192 | 19.43, 19.30 | 18.01, 22.48 | negligible (+4.53 %) | 20.86, 20.90 | gain (+7.81 %) | 158.48, 152.65 | 152.27, 152.73 | negligible (−1.97 %) | 127.22, 126.80 | gain (−18.35 %) |
| 16,384 | 18.36, 16.61 | 16.68, 15.98 | negligible (−6.63 %) | 25.47, 25.38 | gain (+45.39 %) | 184.07, 185.79 | 181.30, 181.18 | gain (−2.00 %) | 130.06, 130.33 | gain (−29.60 %) |
| 32,768 | 11.58, 10.83 | 11.46, 11.55 | negligible (+2.68 %) | 19.71, 19.67 | gain (+75.76 %) | 244.09, 244.11 | 240.89, 241.08 | gain (−1.28 %) | 140.05, 140.58 | gain (−42.52 %) |
| 49,152 | 11.68, 10.36 | 11.32, 11.31 | negligible (+2.67 %) | 21.06, 20.05 | gain (+86.51 %) | 302.25, 302.37 | 299.38, 299.53 | gain (−0.94 %) | 149.81, 150.10 | gain (−50.40 %) |
| all | 15.87, 14.77 | 15.26, 15.67 | negligible (+0.92 %) | 22.50, 22.24 | gain (+45.97 %) | 204.19, 205.44 | 201.05, 201.25 | gain (−1.79 %) | 130.85, 131.06 | gain (−36.06 %) |

| Depth (tokens) | Prompt tok/s, `v3` | `off` | Class | `gains` | Class | Sampler pass tok/s, `v3` | `off` | Class | `gains` | Class |
|---:|---|---|---|---|---|---|---|---|---|---|
| 0 | 32.8, 31.7 | 33.1, 32.8 | negligible (+2.17 %) | 53.6, 52.1 | gain (+64.03 %) | 17.03, 16.12 | 16.65, 16.65 | negligible (+0.48 %) | 19.08, 19.09 | gain (+15.14 %) |
| 8,192 | 124.6, 125.7 | 133.0, 137.7 | gain (+8.16 %) | 142.0, 142.0 | gain (+13.45 %) | 19.33, 17.11 | 16.10, 15.83 | loss (−12.38 %) | 24.66, 24.64 | gain (+35.27 %) |
| 16,384 | 108.5, 108.4 | 113.3, 113.9 | gain (+4.69 %) | 125.8, 125.0 | gain (+15.60 %) | 17.57, 16.10 | 17.86, 17.53 | negligible (+5.10 %) | 27.15, 27.18 | gain (+61.36 %) |
| 32,768 | 90.0, 90.1 | 93.8, 93.3 | gain (+3.86 %) | 98.5, 98.7 | gain (+9.45 %) | 10.42, 8.73 | 10.68, 10.67 | negligible (+11.50 %) | 20.39, 20.46 | gain (+113.29 %) |
| 49,152 | 69.3, 69.5 | 71.8, 71.5 | gain (+3.23 %) | 75.5, 76.5 | gain (+9.50 %) | 8.71, 10.57 | 10.91, 10.92 | negligible (+13.24 %) | 19.75, 19.97 | gain (+106.07 %) |
| all | 87.7, 88.0 | 91.5, 91.5 | gain (+4.17 %) | 97.0, 97.6 | gain (+10.74 %) | 13.22, 12.76 | 13.76, 13.68 | gain (+5.59 %) | 21.79, 21.86 | gain (+67.99 %) |

`gains` against `off` (the switches alone, on the same build): decode +44.64 % over all depths (negligible at 0 and
8,192, +55.71 % at 16,384, +71.17 % at 32,768, +81.66 % at 49,152), the step on pass 2 −34.90 % (a gain at every
depth, −49.92 % at 49,152), prompts +6.31 % (a gain at every depth), the sampler pass +59.10 %
([classes_off_gains.tsv](classes_off_gains.tsv)).

Ms a token (decode, per load): 37.15 and 37.57 (`v3`), 36.60 and 37.05 (`off`), 36.84 and 36.90 (`gains`) at depth 0;
85.60 and 96.54, 88.32 and 88.45, 47.49 and 49.88 at 49,152. The prompt row "all" is the four primes (50,124 tokens).

### The draft and the replies

- **`gains` writes other greedy replies than v3.0, and accepts a smaller share of the draft on them.** Over all depths
  its greedy share accepted is 0.7084 in both loads against 0.8260 and 0.7744 (−11.47 %, a loss by the rule), 3.084
  tokens a step against 3.427 and 3.274. At depth 0, where v3.0's two loads write one reply, `gains` writes another
  one, the same in both its loads, and `off` writes v3.0's. This is C722's and C690's known change: with them the
  verify pass's 2–4-token sums are one-token decoding's, where the preset's k-column kernel's were not, so a near-tie
  in the logits flips (R79 and R75 saw the same with each switch alone). The step is what speeds up: decode tok/s is
  a gain at every depth from 0 in spite of the share.
- **The sampled request** (the server line's sampler and a fixed seed, where probabilistic verification acts): `gains`
  accepts 0.6612 in both loads against 0.5990 and 0.5703 (+13.10 %, a gain), 2.956 tokens a step against 2.765 and
  2.683.
- **Replies per load** ([replies.md](replies.md)): `gains` writes one reply over its two loads on all 15 requests
  (greedy and sampled, five depths), `off` on all 15, `v3` on 3 of 15 (depth 0 only; from 8,192 tokens on each v3.0
  load writes its own, as on every earlier deep-lane run, registry row Q1175). `off` writes v3.0's reply on the three
  requests where v3.0 has one.

### The host and the cards

| Load | Lowest MemAvailable, MiB | Most swap in use, MiB | Highest psi full, % | Card 3 at its highest, MiB | Card 8 at its highest, MiB |
|---|---:|---:|---:|---:|---:|
| `v31` | 207 | 975 | 11.80 | 5,155 | 4,627 |
| `v32` | 80 | 1,151 | 17.22 | 5,155 | 4,625 |
| `gains1` | 304 | 40 | 0.20 | 5,211 | 4,881 |
| `gains2` | 313 | 474 | 1.42 | 5,211 | 4,881 |
| `off1` (the rerun) | 310 | 1,085 | 22.45 | 5,155 | 4,709 |
| `off2` | 81 | 207 | 2.91 | 5,155 | 4,709 |

- **The RAM watchdog stopped `off1`'s first start** at 18:00:37, during the 32,768-token depth: MemAvailable fell 146,
  86, 69, 54 MiB in four seconds, below its 60 MiB floor, with 101 MiB of swap in use. The run resumed at 18:01:51
  (the driver skips finished loads and repeats the untimed warm-up), `off1` ran through (lowest 310 MiB) and `off2`
  after it (81 MiB). v3.0 itself fell to 80 MiB in `v32`: the preset keeps its four checkpoints in host memory, and
  the 4 GB host's margin under it is thin on both builds. `gains` keeps them on the cards and never went below 304 MiB
  (registry row Q1366 retries such a load once, after the others).
- On the cards ([card_max.tsv](card_max.tsv)) `gains` holds 40 to 66 MiB more than `v3` on each of cards 0 to 7
  (est. the four checkpoints on the cards, as in R70) and 254 MiB more on card 8, 4,881 against 4,627. Of those 254,
  `off` holds 82 (4,709: upstream's `h_nextn` buffer for the MTP draft's hidden rows, 80.04 MiB, as R57 found) and the
  switches 172 (R70's five held card 8 180 MiB above the preset: the draft's micro-batch of 512 and the checkpoints).
  Card 3 is the fullest in every load, 5,211 of 6,144 MiB at most.
- No server log holds an error line.

### The agent check

The release check's agent arm as `t1swift` ran it ([2026-11-05_release_flashnext](../2026-11-05_release_flashnext/README.md),
`run/rig_t1.sh`): the 12 bugfix tasks of [bench/agent_check](../../../bench/agent_check/README.md), one attempt each,
a 12-minute cap a task, a 40-minute budget an arm (neither reached it), `gains` first, then `v3`. Beside them, as
v3.0's earlier runs of the same check on the same build, file and preset (rig 2): `v3rel` (2026-11-11, R57,
[2026-11-11_r57_v4night_p106](../2026-11-11_r57_v4night_p106/README.md)) and `t1swift` (2026-11-06, the release
run). The equal-work column times one fixed amount of work for every arm: `v3`'s prompt tokens at the arm's prompt
speed, plus `v3`'s generated tokens at the arm's generation speed, plus `v3`'s time outside the server, over `v3`'s
tasks ([agent_tables.py](../2026-11-05_keepalive/run/agent_tables.py)).

All 12 tasks ([agent.tsv](agent.tsv)):

| Arm | Passed | Turns | Prompt tok/s | Generation tok/s | First token per turn, median and p90, s | Turn 1, median s | Draft acceptance | Tokens a step | Minutes, all tasks | Equal work, min a task | Against `v3` |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| `v3` (rig 1) | 11/12 | 66 | 65.5 | 20.66 | 6.52, 16.17 | 13.54 | 0.696 | 3.09 | 37.0 | 3.08 | — |
| `gains` (rig 1) | 12/12 | 66 | 73.9 | 24.85 | 4.55, 24.48 | 10.68 | 0.757 | 3.27 | 28.7 | 2.79 | −9.4 % |
| `v3rel` (rig 2) | 11/12 | 60 | 68.7 | 20.97 | 7.50, 16.39 | 13.63 | 0.700 | 3.10 | 33.0 | 3.03 | −1.6 % |
| `t1swift` (rig 2) | 12/12 | 66 | 69.4 | 20.40 | 6.89, 15.89 | 13.61 | 0.718 | 3.15 | 32.2 | 3.06 | −0.6 % |

The 11 tasks without `bugfix-javascript-variable-length-quantity` ([agent_11.tsv](agent_11.tsv)): equal work 1.96
minutes a task (`gains`) against 2.27 (`v3`, −13.7 %), `v3rel` 2.21 (−2.6 %), `t1swift` 2.26 (−0.4 %); generation 24.85
against 20.63 tok/s, prompts 73.8 against 65.6.

- **The capped task:** `v3` ran `bugfix-javascript-variable-length-quantity` to the 12-minute cap and failed it after 2
  turns, as `v3rel` did in R57; `gains` passed it in 4.6 minutes. A reply that runs on, which the same build does on
  some runs and not on others (`t1swift` passed it): it is why minutes over all tasks (28.7 against 37.0) say less
  than equal work.
- **Draft acceptance** 0.757 against 0.696 (+8.76 %): the agent sends a temperature of 1.0, where probabilistic
  verification acts.
- **Prompts by the size of a turn's new prompt** ([prompt_buckets.tsv](prompt_buckets.tsv)): `gains` reads short turns
  much faster and long ones slower.

  | New prompt tokens | `v3`: turns, tok/s, median first token s | `gains` | `v3rel` | `t1swift` |
  |---|---|---|---|---|
  | 0–511 | 42, 39.4, 3.57 | 42, 64.4, 2.12 | 36, 39.7, 3.78 | 42, 37.2, 3.56 |
  | 512–1,023 | 11, 61.3, 11.45 | 9, 71.2, 10.78 | 10, 60.3, 12.44 | 9, 58.9, 13.49 |
  | 1,024–1,999 | 10, 90.5, 16.02 | 8, 80.1, 18.26 | 10, 92.0, 15.93 | 11, 91.5, 15.76 |
  | 2,000 and more | 2, 93.3, 25.73 | 7, 76.4, 27.67 | 3, 98.2, 19.94 | 4, 105.5, 23.71 |

  The first turn of `bugfix-python-affine-cipher`, 2,780 tokens on an empty cache, took 42.18 s to its first token
  against 34.53. The deep lane's fresh 8,128-token chunk gains instead (142.0 and 142.0 tok/s against 124.6 and
  125.7): est. the taper's small micro-batches cost most on prompts of one to three thousand tokens, where R85 saw
  the taper lose too (1,024–1,999 new tokens, registry row Q1235). `gains` also met more long turns (7 of 2,000 and
  more against 2), its replies being other ones.
- **Host memory:** `v3`'s lowest MemAvailable 66 MiB, swap 24 → 963 MiB; `gains`' 415 MiB, swap 24 → 24. The RAM
  watchdog acted in neither.
- **The report's lines** ([gonogo.md](gonogo.md), [compare.md](compare.md), [report_table.md](report_table.md)): 23 of
  24 attempts without an infrastructure failure, one re-run (`gains`' `bugfix-python-scale-generator`, scored on its
  second try); malformed tool calls 0 of 125; the cache hit on turns from the second a median of 95.6 %; 14 and 13
  checkpoint restores, no full re-processing, no NaN. Its two NO-GO lines are the capped request (2 proxy rows for 1
  reply) and memory (66 MB, `v3`'s arm).

## Method

- **Loads:** one untimed warm-up load of `v3`, then `v31 gains1 v32 gains2 off1 off2`, each a fresh server start, every
  one after a quiet check (no llama process, at most 100 MiB on any card) and under the RAM watchdog (stop below 60 MiB
  available). `v3` and `gains` came first so that a run cut short still answers v4 against v3.0. Nothing else ran on
  the rig during the loads.
- **Two builds in one run:** `bench/deep_lane` takes `ARM_BIN_<arm>`, an arm's own llama-server; `BIN` is the day
  build, `ARM_BIN_v3` the v3.0 build. Each load's binary is in `loads.tsv`, each build's `--version` in `build.txt` and
  `build-v3.txt`.
- **The lines:** `gains` is the preset with the variables added and the flags set through `ARM_ENV_gains` and
  `ARM_ARGS_gains`, which replace a flag the preset holds instead of adding it twice: `--spec-draft-n-ubatch 128`
  became 512, and `--ctx-checkpoints 4` stays once, with `--ctx-checkpoints-vram 4` and `--ctx-checkpoints-incall`
  beside it. A dry run printed all three lines first; `loads.tsv` holds them as they ran.
- **The requests** of a load, depths ascending, on the release check's recorded `t1swift` sessions (12), as R70: the
  prime (the prompt without its 64-token tail, read on top of the previous depth's checkpoint), then 256 tokens at
  temperature 0 twice (decode, pass 1 and pass 2), then 256 tokens with the server line's own sampler
  (`--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0`) and a fixed seed (the sampler pass). End of text is ignored.
- **Classes,** per depth and over all depths (the driver's rule): the spread is the larger of the two arms'
  differences between their two loads; an arm is a gain when its mean is better than the control's by more than the
  spread, a loss when worse by more than it, else negligible.

## Commands

From the repository root on a machine that only runs ssh; the machine values in the environment:

```bash
RIG=<rig alias> CLIENT=<client alias> MODEL=<Swift Q2_0's first shard on the rig> DRAFT=<the MTP draft on the rig> \
  NEW=llama.cpp-20261111-18d0fa309-pascal-p106-v4-b12246-day REF=llama.cpp-20261105-1551f5fe1-pascal-p106-v3-b12076-ladder5 \
  TEXT_LOCAL=<the release check's proxy.jsonl.bodies.gz> RIGDIR=<a folder on the rig> OUT=<a local folder> \
  RIG_PORT=<port> RELAY_PORT=<port> CLIENT_PORT=<port> PROXY_PORT=<port> CLIENT_DIR=<folder on the client> \
  CLIENT_OUT=<folder on the client> CPU_SHARES=256 \
  setsid nohup bash docs/data/2026-11-11_r90_v4_vs_v3/run/r90.sh > r90.log 2>&1 < /dev/null &
```

The run went through `STEPS="lane prep agent report"` twice: the first stopped at `off1`'s watchdog stop, and the
second resumed the lane (its four finished loads skipped) and ran the rest. The tables, on a build machine's CPU, from
the collected lane folder `out` and the agent check's folder `agent`:

```bash
for p in "v3 gains" "v3 off" "off gains"; do set -- $p
  python3 -I docs/data/2026-11-10_v4_candidate/run/v4c_table.py out --control $1 --arm $2 --md summary_$1_$2.md \
    --tsv per_load.tsv --classes classes_$1_$2.tsv
done
python3 -I docs/data/2026-11-11_r90_v4_vs_v3/run/replies.py out --control v3 --tsv replies.tsv --md replies.md
python3 -I docs/data/2026-11-10_deep_lane/run/host_memory.py out > host_memory.tsv
python3 -I docs/data/2026-11-11_r90_v4_vs_v3/run/card_max.py out > card_max.tsv
# the agent check: R57's collected folder (v3rel, t1swift beside it) merged in, then the tables
NOW=agent OLD=<R57's merged folder> OLD_ARM=v3rel DEST=m1 bash docs/data/2026-11-11_r57_v4night_p106/run/merge_arm.sh
NOW=m1 OLD=<R57's merged folder> OLD_ARM=t1swift DEST=merged bash docs/data/2026-11-11_r57_v4night_p106/run/merge_arm.sh
python3 -I docs/data/2026-11-05_keepalive/run/agent_tables.py merged v3 gains v3rel t1swift
SKIP_TASKS=bugfix-javascript-variable-length-quantity OUT_TAG=_11 \
  python3 -I docs/data/2026-11-05_keepalive/run/agent_tables.py merged v3 gains v3rel t1swift
python3 -I docs/data/2026-11-11_r90_v4_vs_v3/run/prompt_buckets.py merged/res/turns.csv v3 gains v3rel t1swift \
  > prompt_buckets.tsv
```

## Files

| File | Content |
|---|---|
| [lane_table.md](lane_table.md), [lane_classes.tsv](lane_classes.tsv) | the deep lane driver's own tables, every arm against `v3` |
| [summary_v3_gains.md](summary_v3_gains.md), [summary_v3_off.md](summary_v3_off.md), [summary_off_gains.md](summary_off_gains.md) | each pair per depth and load: decode, the step, the draft, the sampler pass, prompts, with classes, and the host's memory |
| [classes_v3_gains.tsv](classes_v3_gains.tsv), [classes_v3_off.tsv](classes_v3_off.tsv), [classes_off_gains.tsv](classes_off_gains.tsv) | the same classes as tables |
| [per_load_v3_gains.tsv](per_load_v3_gains.tsv), [per_load_v3_off.tsv](per_load_v3_off.tsv) | every metric per depth and load |
| [replies.md](replies.md), [replies.tsv](replies.tsv) | each request's reply (its ids' md5) per load |
| [host_memory.tsv](host_memory.tsv), [card_max.tsv](card_max.tsv) | the host's memory per load and request; each card's highest memory per load |
| [loads.tsv](loads.tsv), [requests.csv](requests.csv) | per load its line and binary as it ran (the warm-up and `off1` listed twice: the run resumed); per request its counts and times |
| [agent.tsv](agent.tsv), [agent_11.tsv](agent_11.tsv) | the agent check per arm, all 12 tasks and the 11 |
| [prompt_buckets.tsv](prompt_buckets.tsv) | the agent check's prompt reading by the size of a turn's new prompt |
| [compare.md](compare.md), [report_table.md](report_table.md), [gonogo.md](gonogo.md) | the agent check's own report |
| [run/r90.sh](run/r90.sh) | the run: the deep lane, then the agent check |
| [run/replies.py](run/replies.py), [run/card_max.py](run/card_max.py), [run/prompt_buckets.py](run/prompt_buckets.py) | the tables of this page beside `v4c_table.py`, `host_memory.py` and `agent_tables.py` |

Provenance: measured on a Manli M-P106L9-N6G (nine P106-090) with the builds and files named above. The recorded
prompts are the release check's agent sessions (bench/agent_check's bugfix set); no prompt text is in this folder. The
tables are this repository's measurements, under its licence.
