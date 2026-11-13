# `LLAMA_PP_MULTI_SPLIT=1`'s prompt time in alternated loads (R141, Q1437, 2026-11-13)

What this measured: whether C812's switch `LLAMA_PP_MULTI_SPLIT=1` costs prompt time on nine P106-090 cards. It gives
the host 67 MiB more free memory ([R138](../2026-11-12_r138_swap_cause/README.md)). R138 found `pp1` 4.36 s a pass
slower than `embd` on one build. Then R135's `pp1`, the same line and build, had been 5.24 s faster than R138's, and
R138's `pp1` ran second. So this run loads the two rungs alternately, `embd`, `pp1`, `pp1`, `embd`, so that the load
order cancels out. Result page: registry [Q1437](../../../quants/experiments.md).

**By the ladder's rule the switch is a loss: +1.39 s a pass (+0.58 %), against the two `embd` loads' spread of
0.21 s.** The two `embd` loads came out 240.00 and 240.21 s a pass (means of two passes); the two `pp1` loads 239.74
and 243.25. The loss is not steady:
- **The first `pp1` load was as fast as `embd`:** −0.27 s a pass against the `embd` load before it.
- **The second lost 5.95 s in its first pass to one slow stretch.** In that stretch, turns 30 to 52 of 62, each of
  the nine turns with 775 new tokens or more took 306 ms or more longer than the same turn in `embd4` (968 at
  most). Shorter turns were not slower. Its second pass was as fast as `embd`.
- **Such a stretch is not the switch's alone.** R138's `pp1` had one in its second pass. R123b's `base2`, with the
  switch unset, had one in its second pass too. Of the 16 passes of the eight loads compared here, three have one:
  two with the switch, one without.

The other results:
- **The replies:** `base`'s greedy replies on all 62 turns, in both passes of all four loads.
- **The memory:** `pp1`'s lowest MemAvailable was 453 and 447 MiB, against `embd`'s 383 and 381. That is +70 and +66.
- **The swap:** 5 MiB in every 2 s sample of all four loads.

## What ran

Four loads of R43's replay on nine P106-090 cards, one after the other on one rig. R43's replay is the agent check's
62 prompts, two passes, with the taper `short=600:64` on every request. All four ran on the `-c812` tarball
(`llama.cpp-20261112-d679a6d9d-pascal-p106-v4-b12246-c812`, `bin/llama-server` md5 `b8365b25864cf1aab0b81b47ec2f40a7`)
with R135's launcher, [r135_load.sh](../2026-11-12_c812_draft_pinned/run/r135_load.sh), unchanged:

| Load | Rung | Start | Done | Server up after | Load average (1 min) at the start | Page cache at the start |
|---|---|---|---|---:|---:|---:|
| `embd1` | `embd`, the switch unset | 10:38:34 | 10:52:39 | 226 s | 0.09 | 596 MiB |
| `pp1_2` | `pp1`, `LLAMA_PP_MULTI_SPLIT=1` | 10:52:54 | 11:06:57 | 225 s | 0.92 | 533 MiB |
| `pp1_3` | `pp1` | 11:07:54 | 11:22:04 | 224 s | 0.45 | 583 MiB |
| `embd4` | `embd` | 11:22:15 | 11:36:19 | 224 s | 1.31 | 586 MiB |

The line is R123b's `embd` rung: R123a's kept line, `LLAMA_DRAFT_EMBD_DEV=layer`, and `r43_run.sh` without
`-otd token_embd=CUDA4`. `pp1` adds the switch. [run/r141_loads.sh](run/r141_loads.sh) ran the four loads in order.
It recorded the host before each load and renamed each load's folder after its `DONE`.
- The loads were 15, 57 and 11 s apart. The 57 s gap is where the driver itself was restarted; the load it was
  waiting for had already ended.
- Nothing else ran on the rig during a load, and the page cache was not dropped.
- R43's inputs and the SwapFree `rig_taper.sh` are R138's, with the same md5s: `r43_run.sh`
  `b79bb3d224ad5f7ace4bbf4e3505cb51`, `rig_taper.sh` `353ec109ef8b5806c38e7c14a1c23dfb`, `taper_replay.py`
  `329df66616395a0162065530a0f6108e`, `turns.csv` `a3c7dc91660a507cd415694d47bf7ced` and `settings.env`
  `0c94ef0531708c73a94350b9ad73d4ff`.

## The time

Seconds of the server's prompt time a pass over the 62 replayed turns, by new tokens ([pair.tsv](rig/pair.tsv)). A
rung's value is the mean of its two loads' means. The spread is the two `embd` loads' means apart. The class compares
`pp1` − `embd` with the spread: gain below −spread, loss above +spread, negligible between.

| New tokens | Turns | `embd1` | `pp1_2` | `pp1_3` | `embd4` | `pp1` − `embd` | Spread | Class |
|---|---:|---|---|---|---|---:|---:|---|
| 0–511 | 38 | 64.83, 63.53 | 64.79, 63.47 | 64.99, 63.49 | 64.92, 63.51 | −0.01 | 0.04 | negligible |
| 512–1,023 | 10 | 58.90, 53.34 | 59.09, 53.28 | 59.43, 53.71 | 58.79, 53.57 | +0.23 | 0.06 | loss |
| 1,024–1,999 | 11 | 85.55, 85.09 | 85.01, 85.15 | 90.34, 85.07 | 85.19, 85.47 | +1.07 | 0.01 | loss |
| 2,000+ | 3 | 34.37, 34.41 | 34.32, 34.37 | 34.54, 34.92 | 34.45, 34.52 | +0.10 | 0.10 | loss |
| all | 62 | 243.64, 236.37 | 243.21, 236.27 | 249.30, 237.20 | 243.35, 237.07 | +1.39 (+0.58 %) | 0.21 | loss |

The 2,000+ bucket is a loss at the third decimal (+0.100 against 0.095). Each `pp1` load against the `embd` load
next to it in the order ([order.tsv](rig/order.tsv)):

| New tokens | `pp1_2` − `embd1` | `pp1_3` − `embd4` |
|---|---:|---:|
| 0–511 | −0.05 | +0.03 |
| 512–1,023 | +0.06 | +0.39 |
| 1,024–1,999 | −0.24 | +2.38 |
| 2,000+ | −0.04 | +0.25 |
| all | −0.27 | +3.04 |

**The slow stretch.** `pp1_3`'s first pass was 5.95 s slower than `embd4`'s first pass, and 5.02 s of that came in
turns 30 to 52. Each of the nine turns there with 775 or more new tokens took longer than in `embd4` (and about as much
longer than in `pp1_2`): affine-cipher turns 1 and 2 +610 and +866 ms, list-ops turn 2 +968, phone-number turn 1
+467, pig-latin turns 1, 2 and 3 +393, +306 and +543, scale-generator turns 1 and 2 +481 and +320. The turns with
fewer new tokens were not slower. One earlier turn, grade-school turn 2 (1,984 new tokens), was 435 ms slower. The same kind of stretch, from the per-turn times in [turns.csv](rig/turns.csv) and R138's:
- R138's `pp1`, second pass: its 1,024–1,999 bucket took 90.36 s against 86.16 in its first.
- R123b's `base2` (`-kern`, the switch unset), second pass: affine-cipher turn 2 took 9,373 ms against 7,896 in its
  first pass.
- In the same buckets the two passes of R138's `embd`, R135's `pp1` and R141's other three loads are within 1.50 s of
  each other.

So with the switch two of eight passes on `-c812` (R135's, R138's and these two loads) have such a stretch, against
none of six for `embd`, but a load without the switch has had one too. Est.: something on the rig slows long prompts
for a minute or two at times (a card's clock, the host). The watchdog does not log clocks, so this run cannot tell.
New C832 (Q1464) logs them.

The wall time per request, median and p90 per pass ([first.tsv](rig/first.tsv)): the second passes' medians are
2.94 s in all four loads. `pp1_3`'s first pass has the one high p90, 9.17 s, against 8.61 (`embd1`), 8.47 (`pp1_2`) and
8.53 (`embd4`).

**R123a's ladder** with R123b's `base` loads as the reference ([ladder.tsv](rig/ladder.tsv),
[decision.md](rig/decision.md)) puts each load against the one before. Its replies table is the one quoted below.

## Memory and swap

From R123b's reader ([host.tsv](rig/host.tsv), [cards.tsv](rig/cards.tsv)), R135's (`r135_extra.sh`:
[reserves.tsv](rig/reserves.tsv), [swap.tsv](rig/swap.tsv), [memavail.tsv](rig/memavail.tsv)) and R138's
(`r138_swap.sh`: [swap_summary.tsv](rig/swap_summary.tsv), [swap_time.tsv](rig/swap_time.tsv)):

| Load | Host after loading: used, shared, available | Available after the replay | Lowest MemAvailable | Swap used | Card 8 after loading, after the replay |
|---|---|---:|---:|---|---|
| `embd1` | 2,971, 265, 865 MiB | 384 MiB | 383 MiB at 703 s | 5 MiB in all 417 samples | 5,545, 5,627 MiB |
| `pp1_2` | 2,922, 200, 914 MiB | 451 MiB | 453 MiB at 720 s | 5 MiB in all 416 samples | 5,461, 5,543 MiB |
| `pp1_3` | 2,924, 200, 912 MiB | 450 MiB | 447 MiB at 695 s | 5 MiB in all 419 samples | 5,461, 5,543 MiB |
| `embd4` | 2,993, 265, 843 MiB | 384 MiB | 381 MiB at 697 s | 5 MiB in all 416 samples | 5,545, 5,627 MiB |

- **MemAvailable, sample by sample, once the server is up (from 230 s, 301 samples):** `pp1_2` is +73.9 MiB above
  `embd1` on average, `pp1_3` +63.2 above `embd4`.
- **The draft's buffers are R135's:** its `CUDA_Host` compute buffer is 21.60 MiB with the switch against 86.41, and
  its card-8 buffer 195.49 against 278.39. Only `pp1`'s draft reserve prints the switch's line.
- **The rest is the same in all four loads:** 315 checkpoint lines, 150 taper lines, and 0 graph reallocations in
  146 requests ([realloc_marks.tsv](rig/realloc_marks.tsv)).

## Replies

[replies.tsv](rig/replies.tsv): in all four loads, the greedy reply (8 tokens) of each of the 62 turns is in R123b
`base`'s set in both passes. The two passes agree on all 62 turns.

## What it changes

- **`LLAMA_PP_MULTI_SPLIT=1` does not stay in `v4c3`'s release settings by the rule:** a loss of +1.39 s a pass.
  Parked with these numbers, behind its switch (off by default).
- **What would reopen it:** a load with the clocks logged (C832) that shows the slow stretch is a clock drop or the
  host, not the switch. Then the first pair, −0.27 s a pass, is the switch's time. The agent check decides release
  settings in the end.
- **The memory it gives back** (+70 and +66 MiB at the lowest) matters where the 4 GB host swaps. R123b's `queues` and
  `b4096` rungs swapped 887 and 949 MiB after the replay. None of these loads swapped.

## New ideas

- **C832 (Q1464): log the cards' clocks beside the watchdog.** `rig_taper.sh` also records each card's SM clock and
  the active throttle reasons every 2 s, from one `nvidia-smi --query-gpu=... -lms 2000` process for the whole load,
  so no new process starts each sample. Then a load with a slow stretch shows whether a card clocked down. Est. 15 min
  of code; read in the next load of R43's replay that runs anyway.

## Commands

On the rig (an ssh alias in `RIG`), from this repository, in a folder (`WORK`) holding R43's five inputs and the
SwapFree `rig_taper.sh`; `PRE` is a local folder for the host records:

```bash
RIG=<alias> WORK=<folder> BUILDS=/opt/llama-builds BUILD=llama.cpp-20261112-d679a6d9d-pascal-p106-v4-b12246-c812 \
  MODEL=<Swift Q2_0's first shard> DRAFT=<the MTP draft> PRE=<local folder> \
  bash docs/data/2026-11-13_r141_ppsplit_alt/run/r141_loads.sh
# after the driver stopped during load 2 (the load itself ran on): wait for it, rename it, go on
RIG=<alias> WORK=<folder> ... PRE=<local folder> ORDER="pp1 pp1 embd" N0=2 WAIT=1 \
  bash docs/data/2026-11-13_r141_ppsplit_alt/run/r141_loads.sh
# the load folders beside R123b's six (r123b_<rung>), R135's and R138's in data/
ssh "$RIG" "tar cf - -C $WORK r141_1_embd r141_1_embd.out r141_1_embd.launch r141_2_pp1 r141_2_pp1.out \
  r141_2_pp1.launch r141_3_pp1 r141_3_pp1.out r141_3_pp1.launch r141_4_embd r141_4_embd.out r141_4_embd.launch" \
  | tar xf - -C data/
```

The tables, on any machine with those folders in `data/`:

```bash
OUT=rig REPO=<this repository> bash docs/data/2026-11-12_r123b_prompt_ladder/run/r123b_logs.sh \
  embd1=data/r141_1_embd pp1_2=data/r141_2_pp1 pp1_3=data/r141_3_pp1 embd4=data/r141_4_embd
python3 -I docs/data/2026-11-12_r123a_prompt_ladder/run/r123a_ladder.py --load base=data/r123b_base/load-A \
  --load embd1=data/r141_1_embd/load-A --load pp1_2=data/r141_2_pp1/load-A --load pp1_3=data/r141_3_pp1/load-A \
  --load embd4=data/r141_4_embd/load-A --load base2=data/r123b_base2/load-A --base base --base2 base2 --out rig
python3 -I docs/data/2026-11-13_r141_ppsplit_alt/run/r141_pair.py --ref embd1=data/r141_1_embd/load-A \
  --test pp1_2=data/r141_2_pp1/load-A --test pp1_3=data/r141_3_pp1/load-A --ref embd4=data/r141_4_embd/load-A \
  --out pair && mv pair/decision.md rig/pair_decision.md && mv pair/*.tsv rig/
DATA=data OUT=rig bash docs/data/2026-11-12_c812_draft_pinned/run/r135_extra.sh embd1=r141_1_embd pp1_2=r141_2_pp1 \
  pp1_3=r141_3_pp1 embd4=r141_4_embd
DATA=data OUT=rig bash docs/data/2026-11-12_r138_swap_cause/run/r138_swap.sh embd1=r141_1_embd pp1_2=r141_2_pp1 \
  pp1_3=r141_3_pp1 embd4=r141_4_embd r138_embd=r138_embd r138_pp1=r138_pp1
```

The readers' md5s: `r123b_logs.sh` `99aa2735b470846bf0854ee65a066101`, `r123a_ladder.py`
`cd13ac0ade46bafb95f3f8cc22fd0503`, `r135_extra.sh` `30e6d200b7f0d358f31227d947299213`, `r138_swap.sh`
`c8917487c012cf1cd22bf8ac1c7831c1`, `r141_pair.py` `7cef422adbe453159425c8697af7d8fa`.

## Files

| File | What it is |
|---|---|
| [run/r141_loads.sh](run/r141_loads.sh) | the driver: R135's launcher once per rung in order, the host before each load, each folder renamed after its load |
| [run/r141_pair.py](run/r141_pair.py) | the pair reader: two reference loads against two test loads by bucket, the class against the reference's spread, and each test load against its neighbour |
| [rig/pair.tsv](rig/pair.tsv), [rig/order.tsv](rig/order.tsv), [rig/pair_decision.md](rig/pair_decision.md) | the pair reader's tables: seconds a pass by bucket and class, each `pp1` against its neighbouring `embd`, the classes in words |
| [rig/ladder.tsv](rig/ladder.tsv), [rig/first.tsv](rig/first.tsv), [rig/replies.tsv](rig/replies.tsv), [rig/memory.tsv](rig/memory.tsv), [rig/decision.md](rig/decision.md), [rig/turns.csv](rig/turns.csv) | R123a's ladder with R123b's `base` and `base2`: seconds a pass by bucket, wall and prompt time per pass, replies against `base`, the lowest MemAvailable, the classes in words, every replayed turn |
| [rig/loads.tsv](rig/loads.tsv), [rig/cards.tsv](rig/cards.tsv), [rig/host.tsv](rig/host.tsv), [rig/buffers.tsv](rig/buffers.tsv), [rig/switch_lines.txt](rig/switch_lines.txt), [rig/realloc_marks.tsv](rig/realloc_marks.tsv) | R123b's reader: times and counts per load, card and host memory, every buffer line, the switches, the reallocations |
| [rig/reserves.tsv](rig/reserves.tsv), [rig/swap.tsv](rig/swap.tsv), [rig/memavail.tsv](rig/memavail.tsv) | `r135_extra.sh`: the reserves with the switch's column, swap after loading and after the replay, `embd1`'s and `pp1_2`'s MemAvailable side by side |
| [rig/swap_summary.tsv](rig/swap_summary.tsv), [rig/swap_time.tsv](rig/swap_time.tsv) | `r138_swap.sh`: the host before each load, swap and the lowest MemAvailable per load (R138's two beside), every 2 s sample |

## Provenance and licence

The engine is llama.cpp's licence (MIT); the model is UkisAI's Swift-1.5 GSQ-RCO Q2_0 of Qwen3.8-Flash-Next and the
MTP draft as the v3 release lists them. The prompts' text is the Python standard library's source on the rig (PSF
licence), tokenized there and not stored; the request sizes are the agent check's, from
[turns.csv](../2026-11-03_retype_agent/agent/turns.csv). The tables are this run's own measurements.
