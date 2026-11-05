# The cards' clocks while the release line decodes (C532, 2026-11-05)

What this measured: the SM and memory clock, P-state, power and clock-event reasons of each of the nine P106-090 cards
every 100 ms, while Qwen3.8-Flash-Next decodes on the release line of the first 4 GB rig (S35's line:
[2026-11-04_s35_stack](../2026-11-04_s35_stack/README.md)). It answers whether the cards sit at the 1,354 MHz base clock
while a nine-card layer split decodes, as the stand-in of [2026-11-04_persistent_pipe](../2026-11-04_persistent_pipe/README.md#where-the-saving-comes-from-the-clock-not-the-hop)
did, and so whether a keep-alive kernel that holds the boost clock has anything to win. Registry row Q1021 in
[quants/experiments.md](../../../quants/experiments.md); Q154 ([2026-10-27_clock_share](../2026-10-27_clock_share/README.md))
had only 10 s samples of agent sessions.

## Result

**While the release line decodes, the cards sit at the 1,354 MHz base clock.** The share of samples per phase, pooled
over the nine cards ([clocks.md](clocks.md) has every card):

| Phase | Seconds | Base (< 1,400 MHz) | Mid | Boost (≥ 1,800 MHz) | Power a card, W (mean / max) | Speed |
|---|---:|---:|---:|---:|---|---|
| idle, nothing loaded | 15 | 100 % (P8, 139 MHz) | 0 | 0 | 6.0 / 9.0 | |
| **decode, 1 token a pass** (llama-bench tg128) | 79 | **100 %** | 0 | 0 | 28.0 / 38.4 | 16.16 tok/s, 61.9 ms a token |
| **decode, 4 tokens a pass** (llama-bench `-p 4`) | 13 | **99 %** | 1 | 0 | 26.8 / 40.3 | 128.7 ms a pass |
| served, loaded and idle | 10 | 89 % (7 cards in P8) | 0 | 11 | 11.2 / 32.7 | |
| served prompt (2,760 tokens) | 29 | 14 % (10 % in P8) | 8 | 77 | 33.1 / 77.3 | 94 tok/s |
| **served decode** (MTP draft, 307 tokens) | 20 | **55 %** | 11 | 34 | 31.2 / 60.2 | 15.4 tok/s, 2.5 tokens a pass |

- **Plain decoding at one or four tokens a pass leaves every card at base in every sample.** The cards are in P0 with
  the memory at its full 4,006 MHz and no clock-event reason set: the driver is not limiting power, heat or idleness;
  it picks the P0 base clock for a card that runs a kernel only ~1/9 of the time. Only the SM clock drops.
- **The served decode's boost is the prompt's, carried over** ([timeline.tsv](timeline.tsv), each card's median clock
  second by second). The prompt keeps every card busy enough for boost (1,797–1,911 MHz). 4–7 s into the decode, cards
  0–3 and 5–7 fall to 1,354 MHz and stay there for the rest of it (the last 13 of its 20 s). Card 4 (the draft's token
  embeddings, `-otd token_embd=CUDA4`) holds 1,797 MHz and card 8 (the draft head's card) 1,885 MHz throughout, and
  both stay in P0 while the server idles, when the other seven drop to P8. So on this line seven cards holding 41 of
  the 49 layer shares (`-ts 6,6,6,6,5,6,5,6,3`) decode at base.
- **Clock changes are slow.** After the prompt, boost lasts 4–7 s of light load before a card drops to base. From P8, a
  card reaches 1,650–1,860 MHz within about a second of the prompt reaching it, so each turn's first micro-batch runs
  partly on a card that is still ramping.
- **Power:** ~28–31 W a card decoding at base, 38.5 W on card 8 at boost, up to 77 W on prompt micro-batches (3 % of
  the prompt's samples carry the software power cap at the 75 W limit). No thermal reason in any sample.

**What holding the boost clock is worth (Q1021, C533): a gain, est. 5–9 ms a token on the served decode.** The memory
clock is already full, so only the SM clock is lost (1,354 against ~1,885 MHz). How much that costs depends on how
clock-bound each card's kernels are, which this log cannot see: the stand-in of
[2026-11-04_persistent_pipe](../2026-11-04_persistent_pipe/README.md#where-the-saving-comes-from-the-clock-not-the-hop)
took 16 % less time at boost at one token and 27 % less at four, and Flint's plain decoding 12 % less a token
([Q154](../2026-10-27_clock_share/README.md)). With ~80 % of a pass on the cards
([2026-11-04_kpass_profile](../2026-11-04_kpass_profile/README.md)):

| Path | Cards at base | Est. saving with every card at boost |
|---|---|---|
| plain decoding, 1 token a pass (58.6 ms a token in S35's bench) | all nine | ~7 ms a token (−12 %) |
| served MTP decode (64.9 ms a token here; T(4) 128.7 ms) | seven, 84 % of the layers | 14–23 ms a 4-token pass, 5.5–9 ms a token at 2.5 tokens a pass (−8 to −14 %) |
| prompt | few (77 % at boost) | small: the ramp out of P8 at a turn's start |

The cost: est. +10 W a card while decoding (~29 W at base against ~40 W for a busy card at boost in C417), ~+70 W for
the seven cards. C533 measures the real saving with a keep-alive kernel on the rig.

- **The log's own cost:** tg128 ran at 16.16 ± 0.01 tok/s against S35's 17.06 ± 0.04 with the same build, file and
  line (61.9 against 58.6 ms a token). A nine-card query took ~37 ms, so the log sampled every ~137 ms (median) and kept
  the host's two cores busier. The 100 ms log is for diagnosis only; the agent check's own monitor queries every 10 s.
- **Two new ideas from the log** (registry): Q1045, holding the cards out of P8 between agent turns, so that a turn's
  prompt does not start on cards that are still ramping; Q1046, finding what keeps cards 4 and 8 in P0 while the server
  idles, which may be a keep-alive the engine already has.

## Method

One script on the rig ([run/clocks_run.sh](run/clocks_run.sh)) runs, in order, with
`nvidia-smi --query-gpu=timestamp,index,pstate,clocks.sm,clocks.mem,power.draw,clocks_event_reasons.active
--format=csv,noheader,nounits -lms 100` beside it the whole time:

1. **idle:** 15 s with nothing loaded.
2. **bench:** `llama-bench` on S35's no-draft decode line, the line of S35's [bench.tsv](../2026-11-04_s35_stack/bench.tsv):
   Miner Llama's `-kshape` build (`39c3ab5f1`), the file **pmix-q2-fast**, `-ngl 99 -fa 1 -b 2048 -ub 512
   -ts 6/5/5/6/5/5/6/5/6 -lm dio`, environment `GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=32768 GGML_CUDA_MMVQ_PASCAL_KCOL_TABLE=0
   LLAMA_HOP_CAST=q8_0 GGML_CUDA_HOP_STAGED=1`. Ten instances of a 4-token pass (`-p 4`, the way T(k) is timed in
   [2026-11-04_kshape](../2026-11-04_kshape/README.md)) and `-n 128`, ten repetitions each after a warm-up
   (`-p 4,4,4,4,4,4,4,4,4,4 -n 128 -r 10 -o jsonl --progress`). Both output streams are time-stamped line by line, so
   each instance's timed repetitions form one window: **bench_k4** (100 passes of 4 tokens) and **bench_k1** (1,280
   tokens one at a time).
3. **served:** S35's llama-server line exactly (the agent check's settings `-ngl 99 -fa on -c 65536 -np 1 -bs
   --ctx-checkpoints 4 --cache-ram 0 --jinja --reasoning-format deepseek --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0`,
   then `-b 2048 -ub 512 -ts 6,6,6,6,5,6,5,6,3 -fit off -lm dio -md <MTP draft> --spec-type draft-mtp
   --spec-draft-n-max 3 -otd token_embd=CUDA4 --spec-draft-n-ubatch 128 --no-host`, and S35's environment above plus
   `LLAMA_DRAFT_SHORTLIST=32768,8192,2`) on the loopback; 10 s idle once healthy (**served_idle**), then one request
   sent from the rig itself: the agent check's first turn ([bench/agent_check/first_turn_request.sh](../../../bench/agent_check/first_turn_request.sh):
   the first bugfix task with the agent's system prompt and four tools, ~2,800 tokens, the reply capped at 1,024
   tokens). The reply's `timings` split it into **served_prompt** (from the send, `prompt_ms`) and **served_decode**
   (up to the reply, `predicted_ms`).

Both llama processes ran under [scripts/ram_watchdog.sh](../../../scripts/ram_watchdog.sh) with the 60 MiB floor, as
S35 ran them. [run/clock_phases.py](run/clock_phases.py) counts the samples that fall in each phase's windows, per card:
**base** is an SM clock below 1,400 MHz (the P106-090's base is 1,354 MHz; idle P8 is 139 MHz), **boost** 1,800 MHz or
more (the top bin is 1,911 MHz), **mid** in between. The clock-event reasons are the driver's bit mask (`0x1` GPU idle,
`0x4` software power cap, `0x20`/`0x40` thermal; none set means the card runs at the clock its boost table allows).

## Files

| File | What it is |
|---|---|
| [clocks.md](clocks.md) | per phase and card: samples, base / mid / boost share, SM clock p10 / median / p90, memory clock, P-states, event reasons, power |
| [clocks.tsv](clocks.tsv) | the same with every reason bit and the P-state shares as columns |
| [phases.tsv](phases.tsv) | every phase's windows (epoch seconds) |
| [bench.tsv](bench.tsv) | llama-bench per instance: tok/s and ms a pass |
| [served.tsv](served.tsv) | the served request's timings (prompt and decode tokens and ms, drafted and accepted tokens) |
| [timeline.tsv](timeline.tsv) | the served request second by second: each card's median SM clock and the phase |
| [run/clocks_run.sh](run/clocks_run.sh) | the run, on the rig |
| [run/clock_phases.py](run/clock_phases.py) | the tables, from the run's folder |
| [run/rig_drive.sh](run/rig_drive.sh) | drives both from a machine that only runs ssh: the request, the copy, the detached start, the wait, the tables, the copy back, the clean-up |

## Commands

From the repository root, with the rig's paths:

```bash
RIG=rig OUT=<local folder> BIN=<the -kshape build>/bin MODEL=<pmix-q2-fast first shard> DRAFT=<the MTP draft> \
  bash docs/data/2026-11-05_card_clocks/run/rig_drive.sh
```

The steps are `request send start wait tables collect clean` (`STEPS` picks some). The request is built on the rig
from the agent check's task files (`first_turn_request.sh` with `BUILD_SSH="ssh $RIG"` and `LONG_MAX_TOKENS=1024`);
the task text is not kept here.

## Provenance

Run on 2026-11-05, 11:26–11:39, on the first 4 GB rig (nine P106-090, driver 580, the 75 W limit, the host booted with
`kho=off`), idle and quiet before the start (no llama or `nvidia-smi` process, every card in P8, 3.2 GB available,
no memory pressure). Model loads took ~5 min each (llama-bench, then llama-server). The watchdog never fired: the
lowest usable host memory was 1,011 MiB under llama-bench and 91 MiB under llama-server. The served request was the
first bugfix task's first turn (`bugfix-python-affine-cipher`, rebuilt by `first_turn_request.sh`); the reply ended
in a tool call (`finish_reason` `tool_calls`), and the draft head proposed 369 tokens of which 183 were accepted. The
tables hold no prompt or reply text.
