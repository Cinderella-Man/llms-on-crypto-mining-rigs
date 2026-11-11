# R86 (Q1313): the step trace on a P106, where the time before each verify pass goes (2026-11-11)

What this measured: C405's step trace (`LLAMA_STEP_TRACE`, [2026-11-11_c405_server_gap](../2026-11-11_c405_server_gap/README.md))
on one P106 rig beside the fork's host profile (`GGML_CUDA_PROFILE_HOST`). The question was where the 7.84 ms goes that
[S37](../2026-11-10_served_step_split/README.md) measured between a served step's last drafting pass and the next
verify pass, the one host part of the step above 5 ms. The registry row is Q1313 in
[quants/experiments.md](../../../quants/experiments.md). The run also gives the v4 line's step split beside S37's v3.0
one, and ranks the six moves C405 read from the code (Q1307–Q1312).

The build is Miner Llama's v4 night merge, `pascal-p106-v4-b12246-night` at `c408dabcb`
([2026-11-11_v4_night_merge](../2026-11-11_v4_night_merge/README.md)), which holds C405's trace. Every other switch it
adds was off, so it computes what `pascal-p106-v4` computes. The model is the v3 preset exactly: UkisAI's Swift-1.5
GSQ-RCO Q2_0 of Qwen3.8-Flash-Next on nine P106-090 cards, with the MTP draft at three tokens. There were two untimed
loads, `trace` (the profile and the trace) and `prof` (the profile alone). Each answered S37's greedy request (79
prompt tokens, 256 generated) and then the agent's first-turn request (2,760 prompt tokens, 256 generated).

## Result

**Of the 7.51 ms between the last drafting pass and the verify pass, 5.61 ms passes after the verify pass's inputs are
set and before card 0's graph is launched.** That stage is the scheduler's copies of card 0's inputs. The host's own
stages from the draft's end to the inputs set take 1.48 ms, and the drafted id's way back from card 8 takes 0.34 ms.
C405's six moves act on those two parts. The largest, C738, saves an est. 0.89 ms, and up to 6.50 ms if it also
queues card 0's uploads during the draft. The 5.61 ms itself belongs to the rows that shrink the uploads: the QSA
k-pool tables kept on the cards (C100, C720, C719) and the mask built on the cards (Q261, Q430). The ubatch trace
(C701) splits that stage per input copy. One row ranks above all six: the draft context builds its graph again twice
a step on the rig (Q1104).

### The gap, stage by stage

From [gap_trace.md](gap_trace.md) (`step_gap.py`): 73 typical steps of both requests, three drafted tokens a step, the
first three steps after each prompt left out. The trace and the profile are on one clock (see "Aligning the profile"
below). Medians, ms; the mean is beside each where it differs by more than 0.05.

| Stage | Median | What runs there | Rows that act on it |
|---|---:|---|---|
| the last drafting pass's work end on card 8 → its wait's end | 0.336 | the drafted id and the h row back over PCIe, the host's wake | C742 (Q1311): the h row only |
| the wait's end → back in the server | 0.004 | the drafter's end | |
| `s.drf.out → s.drf.end` | 0.100 | the yield's end: the task queue's worker thread stopped | C741 (Q1310) |
| `s.drf.end → s.ckpt` | 0.393 | draft bookkeeping, the draft context's `seq_rm`, checkpoints | C739 (Q1308), C740 (Q1309) |
| `s.ckpt → s.dec.y` | 0.013 | the batch, `pre_decode`, LoRA, embeddings flag, taper, metrics, screened head | C743 (Q1312): 0.009 of it |
| `s.dec.y → v.dec` | 0.009 | the yield's start, into `llama_context::decode` | C741 |
| `v.dec → v.ub.in` | 0.893 | the batch allocator 0.017, the memory slot 0.041, the outputs 0.010, the ubatch applied 0.084 (mean 0.194), the graph reused 0.069, the inputs set on the host 0.672 | C738 (Q1307) |
| **the host: last drafting pass's wait → `v.ub.in`** | **1.483** (mean 1.502) | | |
| `v.ub.in` → card 0's `graph_compute` entry | **5.610** (mean 9.310) | the scheduler: card 0's input copies | C100, C720, C719 (R74, R80, R78); Q261, Q430; split by R81 (C701) |
| **the gap, as S37 counts it** | **7.512** (mean 11.150) | | |

The rows of the table are medians of each stage, so they do not add up exactly to the totals, which are medians of
whole steps. S37's step split of the same load gives the same gap within 0.23 ms: its last turn is 7.74 ms (median,
both requests, 170 typical steps).

**Where the means come from.** After the 2,760-token prompt the long request's first seven steps hold one long wait
each ([long_stages.md](long_stages.md)). In six of them it is in that last stage, before card 0's launch: 112.66,
111.00, 114.26, 115.06, 117.30 and 161.01 ms. In the other it is in setting the inputs (96.0 ms). Both loads swapped
then (below).

### C738–C743 ranked by what each can save

The verify pass's wait is about 100 ms of host idle time a step, and the host also waits 2.825 ms in the last
drafting pass's `synchronize` (`d.sync(6)#3`). Host work moved into either wait leaves the token's path. Host work
after the catch-up is queued is on the path only by the catch-up's turn: the host reaches the next drafting pass 0.28
ms after the catch-up's work ends on card 8 (`trace`, greedy, median; 0.31 in `prof`).

| Rank | Row | Its stage, median ms | Can save a step, est. |
|---:|---|---|---:|
| 1 | C738 (Q1307), the verify pass prepared during the last drafting pass | `v.dec → v.ub.in` 0.893, inside the 2.825 ms wait | 0.89; up to 6.50 if the preparation also queues card 0's uploads of every input but the token ids (card 0 is idle during the draft) |
| 2 | C740 (Q1309), `seq_rm` bounded by the used cells | its draft-context call is in `s.drf.end → s.ckpt` 0.393; the main context's call after the acceptance (`s.acc.spec → s.rm`) is 0.779, after the catch-up is queued | 0.39 + up to 0.28 = 0.67 |
| 3 | C739 (Q1308), the draft context's `seq_rm` after the verify pass's launch | in `s.drf.end → s.ckpt` 0.393 | up to 0.39 (the same call as C740's first part) |
| 4 | C741 (Q1310), fewer yields | 0.100 + 0.004 in the gap; the catch-up's pair (`s.cu.out → s.cu.end`) 0.124 after the catch-up | 0.10 + up to 0.12 |
| 5 | C742 (Q1311), no h row back from the last drafting pass | inside the 0.336 ms way back | 0.05 (10 KB at the card-to-host rate of ~213 MB/s, [mining_rig.md](../../mining_rig.md)) |
| 6 | C743 (Q1312), the per-step settings before the draft | 0.009 | 0.01 |

**Q1104 (C619) ranks above all six.** The draft context builds its graph again twice a step on the rig: for the
catch-up (`d.ub.apply → d.ub.build` 0.689 ms, between the verify pass's wait and the catch-up's launch, so on the
token's path) and for the first drafting pass (1.890 ms, after the catch-up is queued). Passes 2 and 3 and the verify
pass reuse their graphs (0.015, 0.013 and 0.069 ms). One draft shape (the V100 fork's `LLAMA_MTP_PAD=4` with
`LLAMA_REUSE_OUT_ORDER=1`) saves an est. 0.69 + up to 0.28 = 0.97 ms a step.

### Other sizes the trace gives on the rig

From the waterfall in [gap_trace.md](gap_trace.md), medians:
- **A drafting pass:** the context's host work up to its graphs queued is 2.891 ms in pass 1 (1.890 of it the graph
  build), 0.896 in pass 2 and 0.862 in pass 3 (sums of the stages' medians). Then the wait for the cards is 2.796,
  2.941 and 2.825 ms, and the sampled id read and the next row 0.025, 0.022 and 0.020 ms.
- **The verify pass:** the host queues the nine cards' graphs in 9.970 ms (`v.ub.in → v.ub.comp`, every card's
  `graph_compute` entry inside it) and then waits 102.617 ms for them.
- **After the verify pass, on the token's path until the catch-up is launched:** the server reads the outputs
  (0.136), the drafter starts the catch-up (`mtp.cu → mtp.cu.dec` 0.270), the context's stages, the graph build above.
- **After the catch-up is queued:** the drafter's end of the catch-up 0.230, the acceptance's 24 waits for the cards
  0.400 in all (C714/Q1266's waits that sweep all nine cards), the main context's `seq_rm` 0.779, the trace line's
  write and the server's task loop 0.257 (`s.upd.end` → the next line).
- **Spare drafting passes at a reply's end** (C744/Q1314): 2 in 179 generation steps.

### The trace's own cost

The greedy request draws nothing at random. The two loads gave the same reply and the same draft counts (255 drafted,
170 accepted), so step i of one load pairs with step i of the other ([arms.md](arms.md),
[arms_paired.tsv](arms_paired.tsv)). Over 80 paired typical steps, the trace load's step is longer by +1.26 ms on
average (standard error 0.15, median +1.05). All of it is in the last turn (+1.30), the gap this page splits: the
verify pass, the leads, the turns before the last and the draft's GPU time move by 0.04 ms or less. For scale, S37's
two loads of the preset, both without the trace, differed by −0.68 ms on the same paired measure, −0.55 of it in the
last turn. The long request draws at random and is not paired;
there the trace load's median step is 138.49 ms against 139.17. **So the trace may add up to ~1.3 ms to the gap it
measures.** The stage sizes above are with the trace on. The profile-only load's last turn is 7.25 ms (greedy,
median) against the trace load's 8.43.

### The v4 line's step beside v3.0

The greedy request, typical steps, medians in ms ([split_prof.md](split_prof.md), [split_trace.md](split_trace.md); S37's
`d1off` load of v3.0, `1551f5fe1`, on 2026-11-10, from its README):

| Part of the step | v3.0, `d1off` | v4 night, `prof` | v4 night, `trace` |
|---|---:|---:|---:|
| verify pass: GPU time of the nine cards | 101.91 | 101.69 | 101.65 |
| verify pass: the rest of its span | 4.97 | 4.25 | 4.26 |
| catch-up: crossing card 4 → card 8 | 1.47 | 0.60 | 0.61 |
| catch-up: GPU time on card 8 | 3.63 | 3.70 | 3.70 |
| catch-up: host turnaround | 0.28 | 0.31 | 0.28 |
| drafting pass 1: crossing, GPU time, turnaround | 0.58, 2.64, 0.51 | 0.59, 2.56, 0.52 | 0.59, 2.56, 0.51 |
| drafting pass 2: crossing, GPU time, turnaround | 0.57, 2.58, 0.48 | 0.57, 2.56, 0.49 | 0.58, 2.56, 0.49 |
| drafting pass 3: crossing, GPU time | 0.58, 2.59 | 0.57, 2.56 | 0.57, 2.55 |
| the server between the last drafting pass and the next verify pass | 7.84 | 7.25 | 8.43 |
| **the step** | **131.57** | **128.92** | **130.25** |

S37's second load of the preset, `d1off-2`, gave a 130.51 ms step. These are untimed diagnostic loads of a single
request. The v4 line's speed is R57's agent check.

### Host memory: the long request's first steps swap

[waits.tsv](waits.tsv), [host_mem.tsv](host_mem.tsv) (S37's `waits.py`: a host turn over 50 ms after a drafting pass):

| Load | Request | Steps | Steps with a turn over 50 ms | Their sum, ms | Median such turn, ms | MemAvailable at start, min (MiB) | Swapped out (MiB) |
|---|---|---:|---:|---:|---:|---|---:|
| `trace` | greedy | 84 | 0 | 0.0 | — | 917, 377 | 0 |
| `trace` | long | 93 | 7 (steps 0–6) | 854.9 | 117.1 | 385, 238 | 78 |
| `prof` | greedy | 84 | 0 | 0.0 | — | 884, 372 | 0 |
| `prof` | long | 92 | 23 (steps 0–26) | 2,992.6 | 117.7 | 576, 271 | 203 |

In both loads the long request's first steps wait about 117 ms after the last drafting pass. The trace puts each wait
before card 0's launch, after the inputs are set (above). The server's own timings give 157.38 ms a step (`trace`)
and 204.55 ms (`prof`) on the long request, against 133.98 and 131.65 on the greedy one. S37's v3.0 loads had no such
step (`d1off` long: 0 of 92, 4 MiB swapped out), but they ran the long request first after the load; here it ran
second, after the greedy request. The waits come with swapping, on
either build (Q1328): v3.0 waits the same way whenever its long request swaps
([2026-11-11_c752_long_request_stalls](../2026-11-11_c752_long_request_stalls/README.md)).

## Aligning the profile with the trace

`step_gap.py` moves the profile onto the trace's clock with the log line "its 0 ms is <t> ms on the steady clock". The
CUDA backend logs it at ggml's INFO level. llama-server at its default verbosity does not print ggml's INFO lines, so
the load's log does not have it (Q1329). [run/align_t0.py](run/align_t0.py) finds the offset from the two files
instead ([align.md](align.md)). Both time the same thread on the same clock, so the offset is one constant. Every
`graph_compute` the profile times runs between its ubatch's inputs set and its graphs queued in the trace. Card 0's
188 evaluations, matched to the main context's ubatches one place apart (the profile's first is the load's warm-up),
bound the offset to 630.913 … 635.237 ms. Inside that, an offset of 631.693 … 631.723 ms puts all 3,176 evaluations of
all nine cards after the trace's first mark inside a window of their context. That includes the draft passes' windows
of under 1 ms. The middle, 631.708 ms, is the value used (`--profile-t0`).

## Files

| File | What |
|---|---|
| [gap_trace.md](gap_trace.md) | `step_gap.py`: the waterfall of every stage of a typical step, the gap with the profile's two rows, the time between lines |
| [gap_steps.tsv](gap_steps.tsv) | per generation step: typical or not, drafted, accepted, the line's length, the gap's host part |
| [long_stages.md](long_stages.md) | `long_stages.py`: the steps with an interval over 20 ms outside the waits for the cards, and where it is |
| [align.md](align.md) | `align_t0.py`: the profile's offset to the trace and the check per card |
| [split_trace.md](split_trace.md), [split_prof.md](split_prof.md) | S37's `step_split.py` on each load: the step part by part, per request |
| [steps_trace.tsv](steps_trace.tsv), [steps_prof.tsv](steps_prof.tsv) | `step_split.py --tsv`: every generation step's parts |
| [arms.md](arms.md), [arms.tsv](arms.tsv), [arms_paired.tsv](arms_paired.tsv) | S37's `arms.py` with `prof` as the reference: the server's timings per reply, the parts per load, the greedy request's paired steps (its class table needs two reference loads and is left out) |
| [waits.tsv](waits.tsv), [host_mem.tsv](host_mem.tsv) | S37's `waits.py`: steps with a host turn over 50 ms; the host's memory per request |
| [trace_head.jsonl](trace_head.jsonl) | the trace file's header and first 11 lines (the greedy request's prompt and first steps) |
| [run/align_t0.py](run/align_t0.py) | the profile's 0 ms on the steady clock, from the trace |
| [run/long_stages.py](run/long_stages.py) | the long intervals of every step, with card 0's launch from the profile |

The rig's driver is C405's [run/rig_trace.sh](../2026-11-11_c405_server_gap/run/rig_trace.sh), and the table scripts
are C405's [run/step_gap.py](../2026-11-11_c405_server_gap/run/step_gap.py) and S37's
[run/step_split.py](../2026-11-10_served_step_split/run/step_split.py), [run/arms.py](../2026-11-10_served_step_split/run/arms.py)
and [run/waits.py](../2026-11-10_served_step_split/run/waits.py).

## Commands

On the rig, the rig idle (`BIN`: the night tarball's `bin/`, `MODEL`: the Swift Q2_0 first shard, `DRAFT`: the MTP
draft, `S`: a folder with this repository's files, `LONG`: the first-turn request built by
`BUILD_SSH=<build machine> bash bench/agent_check/first_turn_request.sh long_request.json`):

```bash
BIN=$BIN MODEL=$MODEL DRAFT=$DRAFT PRESET=$S/release/flash-next-v3/settings.env \
  RIG_SERVER=$S/bench/agent_check/rig_server.sh REQ_LONG=$LONG \
  REQ_GREEDY=$S/docs/data/2026-11-04_draft_one_card/run/greedy_request.json OUT=$OUT \
  setsid nohup bash $S/docs/data/2026-11-11_c405_server_gap/run/rig_trace.sh > console.out 2>&1 < /dev/null &
```

On a build machine, CPU only (`OUT` copied there; `RES` a new folder; `C=docs/data/2026-11-11_c405_server_gap/run`,
`P=docs/data/2026-11-10_served_step_split/run`, `A=docs/data/2026-11-11_r86_step_trace/run`):

```bash
nice -n 19 python3 -I $A/align_t0.py $OUT/trace/step_trace.jsonl $OUT/trace/host_profile.csv > $RES/align.md
T0=$(sed -n 's/^t0 = //p' $RES/align.md)
nice -n 19 python3 -I $C/step_gap.py $OUT/trace/step_trace.jsonl --profile $OUT/trace/host_profile.csv \
  --profile-t0 $T0 --tsv $RES/gap_steps.tsv > $RES/gap_trace.md
nice -n 19 python3 -I $C/step_gap.py $OUT/trace/step_trace.jsonl --check | tail -n 1     # check: OK (0 problems)
nice -n 19 python3 -I $A/long_stages.py $OUT/trace/step_trace.jsonl $OUT/trace/host_profile.csv \
  --profile-t0 $T0 > $RES/long_stages.md
for l in trace prof; do
  nice -n 19 python3 -I $P/step_split.py $OUT/$l/host_profile.csv --load $l --requests greedy,long \
    --tsv $RES/steps_$l.tsv > $RES/split_$l.md
  mkdir -p $RES/arms/$l && cp $RES/steps_$l.tsv $RES/arms/$l/steps.tsv && cp $OUT/$l/reply_*.json $RES/arms/$l/
done
nice -n 19 python3 -I $P/arms.py $RES/arms --loads prof,trace --ref prof,prof --requests greedy,long \
  --tsv $RES/arms.tsv > $RES/arms.md
nice -n 19 python3 -I $P/waits.py $RES/arms $OUT --loads trace,prof --requests greedy,long \
  --waits-tsv $RES/waits.tsv --mem-tsv $RES/host_mem.tsv
```

## Limits

- One load with the trace and one without, on one rig: the gap's stages are with the trace on, and the trace may add
  up to ~1.3 ms to the gap (above).
- The stages are at 79 to 3,015 tokens of context. An agent session runs deeper, where the scheduler's copies grow if
  they carry per-cell inputs.
- The trace stops at `v.ub.in`. It does not say which of card 0's inputs the 5.61 ms copies; C701's ubatch trace logs
  each split's wait, copy and launch (R81).

## Provenance and licence

The model is UkisAI's Swift-1.5 GSQ-RCO Q2_0 of Qwen3.8-Flash-Next (its maker's licence) with its MTP draft layer. The
requests are S37's greedy request and the agent check's first turn (the command above rebuilds it). The engine is
Miner Llama (MIT, as llama.cpp). The tables are this project's measurements.
