# The host profile without waiting, on a P106, with Flash-Next (C1003, Q1736; 2026-11-25)

**The no-wait profile costs the same as the waiting one on Flash-Next's decode step, and leaves the prompt's pipeline
alone, where the waiting one reads an 8,128-token prompt at 149.78 tok/s against 222.23 and 214.92 without the
profile.** `GGML_CUDA_PROFILE_HOST=<file>` writes one line
per graph evaluation and card (host entry, host exit, GPU time). By default it waits at each entry for the card's
earlier evaluations; `GGML_CUDA_PROFILE_HOST_NOWAIT=1` ([C877](../2026-11-24_c877_host_profile_nowait/README.md))
writes only the evaluations that have ended and never waits. Here both ran on nine P106-090 cards with
Qwen3.8-Flash-Next's v5 preset and its MTP draft, between two loads without the profile:
- **Decode:** both ways cost about the same against no profile. Over all requests a load generated 25.96 and
  25.86 tok/s without the profile, 25.70 with the waiting profile and 25.69 without waiting. On the repeated request at 8,192 tokens a step took 115.81 and 115.97 ms without the profile,
  115.75 waiting and 116.35 without waiting.
- **Prompt:** reading the 8,128-token prompt took 36,574.705 and 37,819.428 ms without the profile, 38,105.559 ms
  without waiting and 54,265.951 ms waiting. The prompt runs its micro-batches through the cards as a pipeline: under
  the waiting profile no card ever had more than one evaluation in flight, so the pipeline ran one card at a time.
  Without waiting, up to three of a card's evaluations were in flight: card 0's share of the prompt ran from its
  first entry to its last exit in 30,327.3 ms, against 50,239.9 ms waiting, with GPU time of 11,133.1 and
  11,213.3 ms.
- **The same work and the same tokens:** every request gave the same tokens in all four loads; the two profiles wrote
  the same evaluations (584 on each of cards 0 to 7, 2,817 on card 8, no id missing or written twice), and on the
  repeated request each card's GPU time for a verify pass is the same both ways to within 0.09 ms.

**The row's rule is not met as written, and the rule was the wrong test.** Q1736 made the no-wait way the profile's
default if its step stayed within the spread of the unprofiled step. It does not: the two unprofiled loads were
0.12 and 0.06 tok/s apart at depths 0 and 8,192, and the no-wait load was 0.18 and 0.16 tok/s below the slower of
them. But that gap is the profile's own cost (events and a line per evaluation), which the waiting way pays too:
against no profile, no way of profiling can pass this test. On the choice the rule was written for, waiting or not,
the no-wait way is as fast on the decode step (22.59 against 22.62 tok/s at depth 0, 29.77 against 29.75 at 8,192)
and 16.16 s faster on the long prompt. Making it the default is a one-line change in the engine, listed as its own
row (Q1873) for the owner's word.

**No pipelined decode step to map.** Flash-Next's v5 preset does not pipeline its decode passes: in both loads a
card had an earlier evaluation still running only at the first decode pass after a request's own prompt (one entry
per card on each first request), and at no other decode entry. The step map the no-wait way was made
for (Flint's `--pipe-depth 4` preset, R176) is not run here: from 2026-11-25 new work goes to Flash-Next. The only
pipelined part of this preset's run is the prompt, whose per-card counts are in [prompt_cards.tsv](prompt_cards.tsv).

Registry [Q1736](../../../quants/experiments.md) (this check), from C877 (Q1583). New rows from it: Q1873 (the no-wait
way as the default) and Q1874 (earlier results that took prompt times with the waiting profile on).

## What was run

| | |
|---|---|
| Rig | the second rig: nine P106-090 6 GB cards (PCIe 1.0 x1 each), a 4 GB host; every card at stock clocks |
| Build | Miner Llama `pascal-p106-v5-cand-b12674` at `eee615cdb` (it holds C877's switch from `pascal-p106-v5` `af00bb13b`, but not C1004's clock anchors, so the `gpu_start_ms`/`gpu_end_ms` columns and their idle times are not part of this run); `llama-server` md5 `2727faf28b5b34dfe42399a9d431da67` |
| Model | Swift GSQ-RCO Q2_0: UkisAI's Swift-1.5 file of Qwen3.8-Flash-Next, with its n-gram table as the third shard, and the MTP draft `mtp-Qwen3.8-Flash-Next-draft.gguf` |
| Settings | the v5 preset [release/flash-next-v5/settings.env](../../../release/flash-next-v5/settings.env) as it is on `v5` at `fceadd24` (md5 `b8e950f2de8c7906c156d582416731bf`), unchanged: 65,536-token context, the MTP draft three tokens a step, probabilistic verification |
| Driver | [run/c1003.sh](run/c1003.sh): the deep lane ([bench/deep_lane](../../../bench/deep_lane/README.md)) in R121's way ([2026-11-17_r121_flash_step_map](../2026-11-17_r121_flash_step_map/README.md)): prompts of exactly 0 and 8,192 tokens cut from recorded agent text, the 8,192-token depth primed, then the whole prompt twice at temperature 0 (pass 1 and its repeat, pass 2), 256 tokens, and once with the preset's sampler (`s`), four idle seconds before every request |

The loads, in order (20:37–21:06): an untimed warm-up of `u` at depth 0, then `u1`, `hw1`, `hn1`, `u2`.

| Arm | Added to the preset |
|---|---|
| `u` | nothing (loads `u1` and `u2`, before and after the profiled loads) |
| `hw` | `GGML_CUDA_PROFILE_HOST=<OUT>/hp-hw.csv LLAMA_STEP_TRACE=<OUT>/st-hw.jsonl`: the profile the default way, waiting |
| `hn` | the same plus `GGML_CUDA_PROFILE_HOST_NOWAIT=1` |

## Results

Each value is one load; a cell with two values is the two `u` loads, `u1` then `u2`.

### The decode step

The server's ms a step: a request's generation time over its steps (generated tokens less accepted draft tokens).
Every load gave the same replies, so the steps are the same in every load (109, 108 and 97 at depth 0, 73, 72 and 87
at 8,192). From [step.tsv](step.tsv):

| Depth | Request | No profile (`u1`, `u2`) | Waiting (`hw1`) | Without waiting (`hn1`) |
|---:|---|---|---:|---:|
| 0 | pass 1 | 104.74, 105.74 | 106.70 | 106.80 |
| 0 | pass 2 | 101.43, 101.48 | 101.93 | 102.05 |
| 0 | sampled | 102.85, 105.32 | 104.59 | 104.64 |
| 8,192 | pass 1 | 119.66, 119.95 | 121.56 | 120.81 |
| 8,192 | pass 2 | 115.81, 115.97 | 115.75 | 116.35 |
| 8,192 | sampled | 122.06, 122.11 | 124.54 | 122.78 |

The load's generation speed, both greedy passes' tokens over their time ([table.md](table.md)): 22.89 and 22.77 tok/s
without the profile at depth 0, 22.62 waiting, 22.59 without waiting; at 8,192: 29.99 and 29.93, 29.75, 29.77. Over all
requests: 25.96 and 25.86, 25.70, 25.69.

- **Both ways cost about the same.** Without waiting minus waiting: +0.10, +0.12 and +0.05 ms at depth 0 (pass 1,
  pass 2, sampled), −0.75, +0.60 and −1.76 ms at 8,192, a mixed sign.
- **Neither stays inside the unprofiled loads' spread** on the load's speed: without waiting 0.18 and 0.16 tok/s
  below the slower unprofiled load at depths 0 and 8,192, waiting 0.15 and 0.18. The two unprofiled loads differ by
  0.05 ms on pass 2 at depth 0 but by 1.00 ms on pass 1 and 2.47 ms on the sampled request.

### The verify pass from the profile, both ways

Medians over each request's typical steps by S37's split ([split.tsv](split.tsv); per step in
[split/](split/)), ms:

| Depth | Request | Way | Step | Verify pass: GPU time of the nine cards | Verify pass: the rest |
|---:|---|---|---:|---:|---:|
| 0 | pass 2 | waiting | 102.60 | 86.72 | 3.56 |
| 0 | pass 2 | without waiting | 102.72 | 86.63 | 3.57 |
| 8,192 | pass 2 | waiting | 117.40 | 99.99 | 3.65 |
| 8,192 | pass 2 | without waiting | 117.72 | 100.17 | 3.66 |

Each card's GPU time for a verify pass, pass 2, ms:

| Depth | Way | Card 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | waiting | 10.82 | 10.18 | 10.39 | 10.54 | 8.60 | 10.08 | 8.52 | 9.89 | 7.71 |
| 0 | without waiting | 10.85 | 10.16 | 10.39 | 10.50 | 8.59 | 10.03 | 8.53 | 9.89 | 7.71 |
| 8,192 | waiting | 12.01 | 12.47 | 11.50 | 12.66 | 9.67 | 11.25 | 10.70 | 10.96 | 8.77 |
| 8,192 | without waiting | 12.04 | 12.45 | 11.54 | 12.75 | 9.73 | 11.19 | 10.70 | 10.98 | 8.82 |

The largest difference between the two ways on one card is 0.09 ms (card 3 at 8,192). For context, R121 took the same
split the waiting way on 2026-11-17 with v4.0's engine and preset: a verify pass's GPU time of 84.02 and 84.03 ms at
depth 0 and 97.43 and 97.57 at 8,192 (two loads), and an unprofiled step of 98.62 and 114.84 ms; another engine, preset
and day, so not a comparison.

### `queued`: evaluations still running at an entry

The profile's last column counts the card's earlier evaluations still running when the host entered the next one;
waiting, a 1 means the host waited there. From [queued.tsv](queued.tsv), in the generation requests, both ways the
same: on each pass 1 one entry per card had one evaluation still running (the first decode pass after the request's
own prompt), and every other entry had none. The decode passes do not overlap on a card.

### The prompt

The prime at 8,192: 8,128 tokens read on an empty cache ([prompt.tsv](prompt.tsv)):

| Load | ms | tok/s |
|---|---:|---:|
| `u1` (no profile) | 36,574.705 | 222.23 |
| `hw1` (waiting) | 54,265.951 | 149.78 |
| `hn1` (without waiting) | 38,105.559 | 213.30 |
| `u2` (no profile) | 37,819.428 | 214.92 |

Per card, the profile's lines of that prime (17 evaluations on each of cards 0 to 7, 34 on card 8, which also runs
the draft; [prompt_cards.tsv](prompt_cards.tsv)):

| Way | Card 0: GPU time | Card 0: first entry to last exit | Most of a card's evaluations in flight at an entry |
|---|---:|---:|---|
| waiting | 11,213.3 ms | 50,239.9 ms | 1 on every card |
| without waiting | 11,133.1 ms | 30,327.3 ms | 1 on cards 0 and 1, 2 on card 2, 3 on cards 3 to 8 |

Every card's line is in the table. On cards 0 to 7 the GPU time differs by at most 1.0 % between the two ways
(card 7: 10,640.0 against 10,531.2 ms); card 8, which also runs the draft's catch-up, took 5,703.7 against 5,473.0 ms.
The waiting way holds every card to one micro-batch at a time, which adds 19,912.6 ms to card 0's span.

### The profile files

Both ways wrote every evaluation once ([lines.tsv](lines.tsv)): ids 0 to 583 on cards 0 to 7 and 0 to 2,816 on card 8,
none missing, none repeated, the same counts both ways. The profile's own log lines (which way it runs, and the counts
at exit) are not in the server logs at the deep lane's log level. That the no-wait load ran without waiting shows in
its file: at the first decode pass of the first request the nine cards' entries are 3.1 to 4.3 ms apart, where the
waiting load's are 117.4 to 144.1 ms apart, each waiting for its card's prompt work ([profiles/](profiles/)).

## Files

| File | What |
|---|---|
| [step.tsv](step.tsv) | per load, depth and request: tokens, drafted, accepted, steps, the server's ms a step, tokens a step, tok/s |
| [replies.tsv](replies.tsv) | per depth and request: the md5 of the token ids in every load, and whether all four are the same |
| [split.tsv](split.tsv) | per profiled load and request: the medians over the typical steps of the step, the verify pass's span, its GPU time (nine cards), its rest, and each card's GPU time |
| [queued.tsv](queued.tsv) | per profiled load, card and generation request: entries, entries with an earlier evaluation still running, the most at once, entries by that number |
| [prompt.tsv](prompt.tsv) | per load and prompt read: tokens read, ms, tok/s |
| [prompt_cards.tsv](prompt_cards.tsv) | per profiled load and card, the prime at 8,192: evaluations, GPU ms, entries with earlier ones running, the most at once, the host's ms inside those entries, the span |
| [lines.tsv](lines.tsv) | per profiled load and card: lines, first and last evaluation id, ids missing, ids repeated |
| [split/](split/) | S37's step split per profiled load and request (`<load>-<request>.md`) and every step (`<load>.tsv`) |
| [profiles/](profiles/) | the two host profiles (`hp-hw.csv.gz`, `hp-hn.csv.gz`) and step traces (`st-hw.jsonl.gz`, `st-hn.jsonl.gz`) |
| [table.md](table.md) | the deep lane's own tables: tok/s per load, the step per pass, prompt reading, the draft |
| [loads.tsv](loads.tsv), [requests.csv](requests.csv) | the deep lane's load lines (`<BIN>`, `<OUT>` for the rig's paths) and every request |
| [run/c1003.sh](run/c1003.sh) | the run: R121's driver cut to these loads |
| [run/tables.sh](run/tables.sh), [run/c1003_tables.py](run/c1003_tables.py) | the tables from the deep lane's results folder |

## Commands

From this repository's root on the operator's machine (`RIG` the rig's ssh alias; `BIN`, `MODEL` and `MODEL_DRAFT`
paths on the rig; `TEXT_LOCAL` the v3.0 release check's recorded agent text, `proxy.jsonl.bodies.gz`, here):

```bash
RIG=$RIG WORK=$WORK BIN=$BIN MODEL=$MODEL MODEL_DRAFT=$MODEL_DRAFT TEXT_LOCAL=$TEXT_LOCAL DRY=1 bash docs/data/2026-11-25_c1003_host_nowait_flash/run/c1003.sh
RIG=$RIG WORK=$WORK BIN=$BIN MODEL=$MODEL MODEL_DRAFT=$MODEL_DRAFT TEXT_LOCAL=$TEXT_LOCAL bash docs/data/2026-11-25_c1003_host_nowait_flash/run/c1003.sh
RIG=$RIG WORK=$WORK STEP=wait bash docs/data/2026-11-25_c1003_host_nowait_flash/run/c1003.sh
RIG=$RIG WORK=$WORK STEP=collect OUT=$OUT bash docs/data/2026-11-25_c1003_host_nowait_flash/run/c1003.sh
S=$PWD OUT=$OUT RES=$RES bash docs/data/2026-11-25_c1003_host_nowait_flash/run/tables.sh
```

The prompts are cut from this repository's recorded agent text; every reply is generated text. Licence: the
repository's.
