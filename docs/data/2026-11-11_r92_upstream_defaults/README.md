# Upstream b12246's two new defaults on a P106: the MTP draft's row buffer and the early fetch (R92, Q1367; 2026-11-11)

What this measured: what each of the two defaults that upstream llama.cpp b12246 brought to the rig does on nine
P106-090 cards, each turned off alone through C693's switches
([2026-11-11_c693_upstream_switches](../2026-11-11_c693_upstream_switches/README.md), Q1226):

- `LLAMA_MTP_H_DEV`: by default (`1`, upstream [#31927](https://github.com/ggml-org/llama.cpp/pull/31927)) the target
  keeps the hidden rows that the MTP draft reads in a buffer on its output card; `0` copies them to host memory after
  each decode and the draft uploads them again, as the Miner Llama v3 release does.
- `GGML_SCHED_PREFETCH`: by default (`1`, upstream [#30966](https://github.com/ggml-org/llama.cpp/pull/30966)) the
  scheduler starts copying to host memory the inputs of the next split that a card's split already holds, before that
  split runs; `0` copies each split's inputs when it runs.

The line is the Miner Llama v3 release preset
([release/flash-next-v3/settings.env](../../../release/flash-next-v3/settings.env): Qwen3.8-Flash-Next, Swift Q2_0,
the MTP draft of three tokens, a 65,536-token context, `-b 2048 -ub 512`), timed in generation and prompt reading at
context depths 0, 16,384 and 49,152 tokens with [bench/deep_lane](../../../bench/deep_lane/README.md). The build is
the fork's `pascal-p106-v4-b12246-c693` `8d329fa0e` (the v4 line with b12246, C748 and the two switches; with both
switches at their defaults it computes what `-c748` computes). Registry row Q1367 in
[quants/experiments.md](../../../quants/experiments.md).

| Arm | What it is |
|---|---|
| `base` (control) | the release preset, both switches at their defaults (the buffer on card 8, the early fetch on) |
| `host` | `LLAMA_MTP_H_DEV=0`: the draft's rows through host memory, as the v3 release |
| `nopf` | `GGML_SCHED_PREFETCH=0`: no early fetch |
| `trace` | `base` with `LLAMA_UBATCH_TRACE=1` and `-lv 4`: one load for its log (the buffer's line, every scheduler's splits and any early fetch); its times carry the trace's cost and are not used |

These are single-request diagnostics at a known depth. They do not replace the agent check, which decides what goes
into a preset.

## Result

**Upstream's buffer for the draft's rows is a gain over the release's host path, and the early fetch has nothing to
act on in Flash-Next's split.** With the rows through host memory (`host`), decode over the three depths runs at
15.61 and 15.59 tok/s against `base`'s 16.43 and 16.12 (−4.18 %, a loss beyond the spread), prompts at 91.3 and
91.0 against 94.3 and 94.4 tok/s (−3.39 %), and card 8 holds 82 MiB less. Without the early fetch (`nopf`) every
speed is within the spread of `base` (decode 16.43 and 16.42 tok/s), and the trace load logs no early fetch at all:
`GGML_SCHED_PREFETCH` is classed **no effect** on this line. Both defaults stay as they are.

- **The buffer on card 8:** the trace load logs `h_nextn_dev_alloc: CUDA8 h_nextn buffer size = 80.04 MiB` (2,049
  rows of 10,240 F32 values; est. 80.0 MiB from the code). After loading, card 8 holds 4,647 MiB in every `base`,
  `nopf` and `trace` load and 4,565 MiB in every `host` load; at its highest in the run 4,707 against 4,625 MiB.
  Cards 0 to 7 hold the same MiB in every load (after loading 5,059, 5,095, 4,963, 5,111, 4,875, 4,979, 4,381,
  4,991; at their highest 5,101, 5,137, 5,005, 5,153, 4,913, 5,021, 4,423, 5,033) ([card_peaks.tsv](card_peaks.tsv)).
- **The same replies:** at depth 0 the greedy reply of every timed load of `base`, `host`, `nopf` and `trace` has the
  same text and token ids (md5 `4eab60d1…` and `08569d4a…`) with the same draft counters (178 of 229 accepted on
  pass 1, 181 of 220 on pass 2). So do the replies at 16,384 (`b94556e0…`) and 49,152 (`d9b2532c…`)
  ([table.md](table.md), last section).
- **No early fetch:** the trace load writes 2,822 `sched-trace:` lines, one per computed graph, and no
  `sched-trace: prefetch:` line. Each of the main model's 574 traced computations has 10 splits in the order CPU,
  card 0, card 1, … card 8; each of the draft's 2,248 has two, card 4 (its token embeddings,
  `-otd token_embd=CUDA4`) then card 8. The early fetch acts only where a split's next split reads inputs through host
  memory (a CPU split after a card's split). Here the one CPU split comes first, so the fetch never has anything to do.
  (The log's one other line with "prefetch" is the n-gram table's page prefetch at load, which is not this switch.)

### Decode, the step and prompts by depth

Decode: 256 greedy tokens twice per depth (pass 1, the first request at a new depth, right after the prime; pass 2,
the same request again). A load's speed is both passes' tokens over their time. The step is generation time over
steps (generated tokens minus accepted draft tokens). Prompts: the prime's chunk, read on top of the previous depth's
checkpoint (at depth 0, the 64-token tail on an empty cache). Class: gain or loss when the arms' means differ by more
than the larger of the two arms' spreads between their loads, else negligible
([host_classes.tsv](host_classes.tsv), [nopf_classes.tsv](nopf_classes.tsv)). Every value is one load.

| Depth (tokens) | Metric | `base` | `host` | Class (`host`) | `nopf` | Class (`nopf`) |
|---:|---|---|---|---|---|---|
| 0 | decode tok/s | 27.14, 27.16 | 26.76, 26.79 | loss (−1.39 %) | 27.01, 27.13 | negligible (−0.30 %) |
|  | step on pass 1, ms | 125.25, 124.61 | 126.88, 126.86 | loss (+1.55 %) | 126.01, 125.37 | loss (+0.60 %) |
|  | step on pass 2, ms | 121.28, 121.71 | 123.15, 122.89 | loss (+1.25 %) | 121.71, 121.25 | negligible (−0.02 %) |
|  | prompt tok/s | 31.9, 32.8 | 32.9, 32.1 | negligible (+0.60 %) | 31.4, 32.6 | negligible (−1.02 %) |
| 16,384 | decode tok/s | 18.08, 18.05 | 16.64, 16.35 | loss (−8.66 %) | 18.08, 18.05 | negligible (−0.00 %) |
|  | step on pass 1, ms | 185.03, 185.11 | 214.65, 221.46 | loss (+17.82 %) | 184.69, 185.24 | negligible (−0.06 %) |
|  | step on pass 2, ms | 180.40, 180.85 | 182.06, 182.23 | loss (+0.84 %) | 180.76, 180.71 | negligible (+0.06 %) |
|  | prompt tok/s | 131.8, 132.1 | 126.3, 124.5 | loss (−5.01 %) | 132.1, 131.5 | negligible (−0.12 %) |
| 49,152 | decode tok/s | 11.06, 10.65 | 10.55, 10.64 | negligible (−2.41 %) | 11.08, 11.06 | negligible (+1.97 %) |
|  | step on pass 1, ms | 318.49, 343.01 | 346.83, 341.62 | negligible (+4.07 %) | 317.59, 319.13 | negligible (−3.75 %) |
|  | step on pass 2, ms | 298.57, 298.06 | 300.14, 300.19 | loss (+0.62 %) | 298.41, 298.30 | negligible (+0.01 %) |
|  | prompt tok/s | 82.8, 82.8 | 80.4, 80.4 | loss (−2.88 %) | 83.3, 82.6 | negligible (+0.18 %) |
| all | decode tok/s | 16.43, 16.12 | 15.61, 15.59 | loss (−4.18 %) | 16.43, 16.42 | negligible (+0.92 %) |
|  | step on pass 1, ms | 208.18, 215.95 | 227.93, 228.53 | loss (+7.62 %) | 208.03, 208.49 | negligible (−1.79 %) |
|  | step on pass 2, ms | 199.91, 200.04 | 201.61, 201.60 | loss (+0.81 %) | 200.12, 199.91 | negligible (+0.02 %) |
|  | prompt tok/s | 94.3, 94.4 | 91.3, 91.0 | loss (−3.39 %) | 94.8, 94.1 | negligible (+0.11 %) |

The host path costs most right after a long prompt: the first request at 16,384 tokens takes 214.65 and 221.46 ms a
step against 185.03 and 185.11, and the prime there reads 126.3 and 124.5 tok/s against 131.8 and 132.1 (129.2 and
131.1 s against 123.8 and 123.5 s for 16,320 tokens). Est. the cause: on the host path each 512-token micro-batch's
rows (20 MiB) cross card 8's PCIe 1.0 x1 link to the host and back to the draft, where the buffer keeps them on the
card. On the repeated request (pass 2) the host path's step is longer by +1.52, +1.52 and +1.84 ms (the means) at
0, 16,384 and 49,152 tokens. The draft accepts the same tokens in every arm.

`nopf`'s one metric beyond the spread is the step on pass 1 at depth 0 (+0.75 ms against a spread of 0.64). With
nothing to fetch, the switch only skips a search for inputs that finds none, so this is classed with the rest of the
arm: no effect.

### The host's memory per load

The rig has 4 GB of host memory. From the RAM watchdog's samples (one a second), each load's lowest MemAvailable and
most swap in use, in MiB:

| Load | `warm` | `base1` | `host1` | `nopf1` | `base2` | `host2` (1st, stopped) | `warm` (2nd) | `nopf2` | `trace` | `host2` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lowest MemAvailable | 385 | 97 | 80 | 752 | 222 | 57 | 230 | 117 | 223 | 113 |
| most swap in use | 824 | 45 | 841 | 848 | 788 | 177 | 923 | 44 | 780 | 970 |

**The first `host2` load was stopped by the RAM watchdog** at 15:11:27, 872 s into the load: MemAvailable fell to
57 MiB, below its 60 MiB floor, as the request at 49,152 tokens began, right after the prime there. Its replies to
that point match every other load ([host2_try1_requests.csv](host2_try1_requests.csv)). The run was resumed with
the same settings and `host2` moved to the end, after `trace`; the second `host2` ran through, at a lowest
MemAvailable of 113 MiB. The lows differ more between loads of one arm than between arms (`base` 97 and 222, `nopf`
752 and 117), following how much the kernel had swapped out before the deep requests, so the host's memory is not
classed by arm here. Of the seven timed loads that finished, the two `host` loads had the lowest (80 MiB) and the
third-lowest (113 MiB) lows.

### The trace load's buffers

The trace load (`-lv 4`) also prints the compute buffers per card for this line, on this build: the main model 344.13
MiB on card 0, 280.75 on cards 1, 3 and 6, 280.37 on cards 2, 4, 5, 7 and 8, 123.26 MiB of pinned host memory; the
MTP draft 1.25 MiB on card 4, 57.46 on card 8 and 9.96 of pinned host memory, reserved again with the same sizes once
the row buffer is in place. Five seconds later the main model is reserved again: the same
sizes, but card 8 at 280.38 MiB and the host at 123.28 ([load_lines.tsv](load_lines.tsv)).

## Files

| File | What |
|---|---|
| [host_summary.md](host_summary.md), [nopf_summary.md](nopf_summary.md) | per depth and load: decode, the step on both passes, the draft's counters, prompts; the classes; the host's memory per load ([run/v4c_table.py](../2026-11-10_v4_candidate/run/v4c_table.py) with `--control base --arm host` and `--arm nopf`) |
| [host_classes.tsv](host_classes.tsv), [nopf_classes.tsv](nopf_classes.tsv), [host_per_load.tsv](host_per_load.tsv), [nopf_per_load.tsv](nopf_per_load.tsv) | the same classes and per-load values as tables |
| [table.md](table.md), [classes.tsv](classes.tsv) | the driver's own tables (`deep_table.py`), with the greedy reply checks; its classes of `trace` are not used |
| [requests.csv](requests.csv), [loads.tsv](loads.tsv) | one line per request (timings, draft counters, a hash of the reply) and per load (its line and switches; `warm` and `host2` are listed twice, the second lines are the resumed run's) |
| [host2_try1_requests.csv](host2_try1_requests.csv) | the stopped first `host2` load's requests, to the prime at 49,152 |
| [load_lines.tsv](load_lines.tsv), [load_lines_try1.tsv](load_lines_try1.tsv) | per load, what its server log says about the switches: the buffer's line, the host-row lines, the early fetch's lines, the trace's line count, the compute buffers, buffer warnings ([run/load_lines.py](run/load_lines.py)) |
| [card_peaks.tsv](card_peaks.tsv) | each card's highest MiB per load, from the driver's samples every 10 s ([run/card_peaks.py](../2026-11-11_r77_c704_p106/run/card_peaks.py)) |
| [run/load_lines.py](run/load_lines.py) | the log reader above (standard library only) |

## Commands

From a machine that only runs ssh, at the repository's root. `RIG` is the rig's ssh alias, `WORK` a folder on the rig,
`BIN` the c693 tarball's `bin/` on the rig, `MODELS` the folder with Swift Q2_0's shards and the MTP draft,
`RECORDED` the release check's recording (`proxy.jsonl.bodies.gz`, the source of the prompts' text), `OUT` a results
folder here:

```bash
RIG=$RIG WORK=$WORK BIN=$BIN/llama-server \
  MODEL=$MODELS/swift_q2_0/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf \
  MODEL_DRAFT=$MODELS/mtp-Qwen3.8-Flash-Next-draft.gguf \
  TEXT_LOCAL=$RECORDED/proxy.jsonl.bodies.gz TAG='^t1swift/' WD_ENV=MIN_AVAIL_MB=60 \
  ARMS="base host nopf trace" \
  ARM_ENV_host="LLAMA_MTP_H_DEV=0" \
  ARM_ENV_nopf="GGML_SCHED_PREFETCH=0" \
  ARM_ENV_trace="LLAMA_UBATCH_TRACE=1" ARM_ARGS_trace="-lv 4" \
  LOADS="warm:base base1:base host1:host nopf1:nopf base2:base host2:host nopf2:nopf trace:trace" \
  DEPTHS="0 16384 49152" WARM_DEPTHS="16384 49152" PAUSE=4 \
  bash bench/deep_lane/remote.sh start
RIG=$RIG WORK=$WORK bash bench/deep_lane/remote.sh wait
# after the watchdog's stop of host2: its files set aside on the rig, the same command with host2 last
ssh $RIG "cd $WORK/out && mkdir -p try1 && mv server-host2.log ram_watchdog-host2.tsv requests-host2.csv host2.jsonl quiet_host2.txt fit_host2.txt try1/"
# ... the start command again with LOADS="warm:base base1:base host1:host nopf1:nopf base2:base nopf2:nopf trace:trace host2:host"
RIG=$RIG WORK=$WORK OUT=$OUT CLEAN=0 bash bench/deep_lane/remote.sh collect
```

The driver skips the finished loads on the resumed start and runs its warm-up load again first. The tables, on a
build machine's CPU (`nice -n 19`), with `RES` a folder for them:

```bash
R=docs/data
nice -n 19 python3 -I $R/2026-11-11_r92_upstream_defaults/run/load_lines.py $OUT > $RES/load_lines.tsv
nice -n 19 python3 -I $R/2026-11-11_r92_upstream_defaults/run/load_lines.py $OUT/try1 > $RES/load_lines_try1.tsv
nice -n 19 python3 -I $R/2026-11-10_v4_candidate/run/v4c_table.py $OUT --control base --arm host \
  --md $RES/host_summary.md --tsv $RES/host_per_load.tsv --classes $RES/host_classes.tsv
nice -n 19 python3 -I $R/2026-11-10_v4_candidate/run/v4c_table.py $OUT --control base --arm nopf \
  --md $RES/nopf_summary.md --tsv $RES/nopf_per_load.tsv --classes $RES/nopf_classes.tsv
nice -n 19 python3 -I $R/2026-11-11_r77_c704_p106/run/card_peaks.py $OUT > $RES/card_peaks.tsv
grep -a 'h_nextn buffer size\|extracting the h_nextn\|sched-trace: prefetch' $OUT/server-trace.log
```

Run on rig 2 on 2026-11-11: 13:41:08 to 15:11:27 (`warm`, `base1`, `host1`, `nopf1`, `base2`, each about 15
minutes, then the stopped `host2`), and 15:13:20 to 16:13:57 (`warm`, `nopf2`, `trace`, `host2`). Every load
started after the driver's quiet check (no llama process, at most 100 MiB on any card).

## Provenance

Code: the fork's branch `pascal-p106-v4-b12246-c693` (MIT, as llama.cpp); the scripts here and in `bench/deep_lane`
(this repository's licence). The prompts are cut from the recorded text of this project's own release check (the
agent check's requests on its bugfix tasks), as in R70, R51 and R77; no reply text is published, only a hash per
reply.
