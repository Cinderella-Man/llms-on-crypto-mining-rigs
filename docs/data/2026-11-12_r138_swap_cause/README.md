# Why one load of R43's replay swapped (R138, Q1430, 2026-11-12)

What this measured: why the 4 GB host swapped 496 MiB in R135's load of R43's replay with C812's
`LLAMA_PP_MULTI_SPLIT=1` ([R135](../2026-11-12_c812_draft_pinned/README.md#the-rig-check-r135-q1421)). Three causes were
possible: the switch, the `-c812` build, or the host's page cache (R135 started 17 s after the build's tree was
unpacked). Result page: registry [Q1430](../../../quants/experiments.md).

**Neither the switch nor the `-c812` build makes the host swap.** Two loads ran on the `-c812` build, with the switch
unset (`embd`) and set (`pp1`). Both kept swap at 20 MiB from start to end, checked every 2 s. `embd` started with
more page cache than R135 had (1,817 MiB against 1,622) and did not swap either. So page cache at the start does not
explain R135's swap on its own. What only R135 had is a start 17 s after an unpack, at a load average of 3.05.

The other results:
- **The switch's memory gain:** `pp1`'s lowest MemAvailable is 446 MiB against `embd`'s 379, +67 MiB (est. +65).
- **The switch's time is not settled.** On one build `pp1` read the 62 prompts 4.36 s a pass slower than `embd`
  (+1.81 %). That is a loss by the ladder's rule, which takes its spread, 3.77 s, from R123b's two `base` loads. But
  R135's load of the same line on the same build was 5.24 s a pass faster than this one. R141 (Q1437) tells the switch
  from the load order.
- **The replies:** `base`'s greedy replies on all 62 turns, in both passes of both loads.

## What ran

Two loads of R43's replay on nine P106-090 cards, one after the other on one rig. R43's replay is the agent check's 62
prompts, two passes, with the taper `short=600:64` on every request. Both loads ran on the `-c812` tarball
(`llama.cpp-20261112-d679a6d9d-pascal-p106-v4-b12246-c812`, `bin/llama-server` md5 `b8365b25864cf1aab0b81b47ec2f40a7`).

The line is R123b's `embd` rung: R123a's kept line, `LLAMA_DRAFT_EMBD_DEV=layer`, and `r43_run.sh` without
`-otd token_embd=CUDA4`. R135's launcher, [r135_load.sh](../2026-11-12_c812_draft_pinned/run/r135_load.sh), started
each load unchanged:
- `RUNG=embd`, the switch unset: 20:59:28 to 21:13:34;
- then `RUNG=pp1`, with `LLAMA_PP_MULTI_SPLIT=1`: 21:14:11 to 21:28:26.

The server was up after 226 s in both. Nothing else ran on the rig during a load, and the page cache was not dropped.

The one change to the driver: the watchdog in
[rig_taper.sh](../2026-11-04_prompt_taper/run/rig_taper.sh) (md5 `353ec109ef8b5806c38e7c14a1c23dfb`) now writes
`SwapFree` beside MemAvailable every 2 s, as a third column of `load-A/memavail.txt`. R43's other inputs are R123b's,
with the same md5s: `r43_run.sh` `b79bb3d224ad5f7ace4bbf4e3505cb51`, `taper_replay.py`
`329df66616395a0162065530a0f6108e`, `turns.csv` `a3c7dc91660a507cd415694d47bf7ced` and the v3 preset's
`settings.env` `0c94ef0531708c73a94350b9ad73d4ff`. They sit in a fresh folder, so R135's own load folder was left
as it was.

## The host before each load

From each load's own `quiet_check.txt`, `uptime` and `free -m` just before its server started
([swap_summary.tsv](rig/swap_summary.tsv)). For R138's two loads, [start.tsv](rig/start.tsv) also has `/proc/meminfo`, read 1 s
before each start.

| Load | Build | What ran just before | Load average (1 min) | Page cache (buff/cache) | Available | Swap used |
|---|---|---|---:|---:|---:|---:|
| R138 `embd` | `-c812` | R134's `-c809` run, ended 20:52 (its tree unpacked before 20:39) | 0.00 | 1,817 MiB | 3,140 MiB | 20 MiB |
| R138 `pp1` | `-c812` | R138 `embd` | 0.61 | 531 MiB | 3,135 MiB | 20 MiB |
| R135 `pp1` | `-c812` | the `-c812` tree's unpack and `sync`, ended 17 s before | 3.05 | 1,622 MiB | 3,142 MiB | 22 MiB |
| R123b `embd` | `-kern` | R123b `base` | 0.67 | 687 MiB | 3,149 MiB | 20 MiB |

Of `embd`'s 1,817 MiB, 1,479,192 kB was inactive file cache and 271,308 kB active.

## Swap and MemAvailable

Swap used after loading and after the replay is from `free -m`; over time it is SwapTotal minus the watchdog's
SwapFree ([swap_time.tsv](rig/swap_time.tsv), every 2 s). The lowest MemAvailable is the watchdog's, in seconds from
its first sample ([memory.tsv](rig/memory.tsv)).

| Load | Swap after loading | Swap after the replay | Swap over time | Lowest MemAvailable |
|---|---:|---:|---|---:|
| R138 `embd` | 20 MiB | 20 MiB | 20 MiB in all 417 samples | 379 MiB at 750 s |
| R138 `pp1` | 20 MiB | 20 MiB | 20 MiB in all 422 samples | 446 MiB at 795 s |
| R135 `pp1` | 22 MiB | 518 MiB | not logged; MemAvailable climbs from about 289 s | 494 MiB at 281 s |
| R123b `embd` | 20 MiB | 20 MiB | not logged | 388 MiB at 801 s |

- **The switch:** `pp1` on `-c812` did not swap. Not the switch.
- **The build:** `embd` on `-c812` did not swap. Not the build.
- **Page cache at the start:** `embd` began with 195 MiB more page cache than R135 and did not swap. In `embd` the
  kernel dropped that cache while the model loaded: 871 MiB was left after loading, and 871 after the replay. So a
  full page cache at the start does not explain R135's swap on its own. What R135 alone had is the start 17 s after
  an unpack, at a load average of 3.05. Est.: the kernel treats page cache that was written seconds before as
  recently used and keeps it, so it swaps the server's memory out instead. These rigs' kernel uses the
  multi-generation LRU (`/sys/kernel/mm/lru_gen/enabled` is `0x0007`, `vm.swappiness` 60). Q1431 (C819) tests dropping
  the page cache before a load.
- **R123b's `queues` and `b4096` loads swapped too, for another reason:** 887 and 949 MiB after the replay, from 20 after
  loading, with the lowest MemAvailable at 171 and 183 MiB. Those two rungs add the doubled launch queue's 222 MiB of
  host memory (`b4096` on top of it). That swap is the server's own need. R123b's `base`, `embd`, `sparse` and `base2`
  stayed at 20 MiB.

**`pp1` against `embd`, MemAvailable** ([memavail.tsv](rig/memavail.tsv), paired by sample):

| What | `embd` | `pp1` | `pp1` − `embd` |
|---|---:|---:|---:|
| The lowest | 379 MiB at 750 s | 446 MiB at 795 s | +67 MiB (est. +65) |
| At 750 s, `embd`'s lowest | 379 MiB | 454 MiB | +75 MiB |
| Mean over the samples from 230 s on (302, the server up) | — | — | +60.3 MiB (lowest +49 at 559 s) |
| Mean over the samples up to 226 s (113, loading) | — | — | +1.7 MiB |
| After loading ([host.tsv](rig/host.tsv)) | 844 MiB | 921 MiB | +77 MiB |
| After the replay | 377 MiB | 450 MiB | +73 MiB |
| Shared memory after loading | 264 MiB | 199 MiB | −65 MiB |

Everything else is as in R135 against R123b's `embd` ([reserves.tsv](rig/reserves.tsv), [buffers.tsv](rig/buffers.tsv),
[cards.tsv](rig/cards.tsv)):
- the draft's reserves: `sched copies = 1` and the switch's line in `pp1`, against 4 in `embd`;
- the draft's `CUDA_Host` compute buffer: 21.60 MiB against 86.41;
- the draft's card-8 compute buffer: 195.49 MiB against 278.39;
- card 8: 5,461 MiB against 5,545 after loading, and 5,543 against 5,627 after the replay.

Both loads have 315 checkpoint lines, 150 taper lines and 0 graph reallocations in 146 requests
([realloc_marks.tsv](rig/realloc_marks.tsv)).

## The time on one build

R123a's ladder ([ladder.tsv](rig/ladder.tsv), [decision.md](rig/decision.md)) with R123b's two `base` loads (`-kern`)
as the reference and the spread. Seconds of the server's prompt time a pass, over the 62 replayed turns:

| New tokens | Turns | `embd` (`-c812`) | `pp1` (`-c812`) | `pp1` − `embd` | Spread | Class by the rule |
|---|---:|---|---|---:|---:|---|
| 0–511 | 38 | 64.97, 63.55 | 64.98, 63.58 | +0.02 | 0.18 | negligible |
| 512–1,023 | 10 | 58.93, 53.46 | 59.56, 53.56 | +0.36 | 0.07 | loss |
| 1,024–1,999 | 11 | 85.38, 85.30 | 86.16, 90.36 | +2.92 | 2.73 | loss |
| 2,000+ | 3 | 34.38, 34.50 | 35.38, 35.62 | +1.06 | 0.79 | loss |
| all | 62 | 243.67, 236.82 | 246.08, 243.12 | +4.36 (+1.81 %) | 3.77 | loss |

These are four loads of this line:

| Load | Build | Seconds a pass | Mean |
|---|---|---|---:|
| R123b `embd` | `-kern` | 246.53, 238.34 | 242.43 |
| R138 `embd` | `-c812` | 243.67, 236.82 | 240.24 |
| R135 `pp1` | `-c812` | 244.89, 233.83 | 239.36 |
| R138 `pp1` | `-c812` | 246.08, 243.12 | 244.60 |

- **By the ladder's rule `pp1` is a loss on one build**, +4.36 s a pass against a spread of 3.77. The gap grows with
  the prompt: +0.02 s under 512 new tokens, +2.92 s at 1,024 to 1,999 (where `pp1`'s second pass, 90.36 s, is 4.20 s
  above its first).
- **The same line and build, loaded twice, differ by more:** R135's `pp1` and this `pp1` are 5.24 s a pass apart. The
  spread of 3.77 comes from R123b's `base` loaded first and last on another build. This `pp1` started 15 minutes after
  `embd`. R123b's `base`, loaded again last, was 3.77 s a pass slower than its first load.
- **So the time is not settled.** R141 (Q1437) loads `embd` and `pp1` alternately on one build, `embd`, `pp1`,
  `pp1`, `embd`, so that the load order cancels out. The release line is chosen on agentic speed, so the agent check
  decides in the end.
- **Est. a cause, if R141 confirms it:** with one copy of each input, the draft's next micro-batch cannot be uploaded
  while the last one still computes. That would cost more on long prompts, where the draft's catch-up runs more
  micro-batches. Not measured.
- **`embd` against R123b's `base`**, the build difference included: 243.67 and 236.82 s a pass against 247.52 and
  240.27, −3.65 s (−1.50 %), negligible.

## New ideas

- **R141 (Q1437): `LLAMA_PP_MULTI_SPLIT=1`'s time in alternated loads on one build.** Four loads of R43's replay on
  the `-c812` tarball, in the order `embd`, `pp1`, `pp1`, `embd`, with this folder's driver. Read pass times, both
  `embd`s against both `pp1`s, by bucket. Est.: no difference beyond the two `embd` loads' own gap, or a cost only in
  the buckets of 1,024 new tokens and more.

## Commands

On the rig (an ssh alias in `RIG`), one load per call, each in a folder (`WORK`) that holds R43's five inputs and this
`rig_taper.sh`:

```bash
for RUNG in embd pp1; do   # one at a time: the second starts after the first's DONE
  RIG=<alias> WORK=<folder> BUILDS=/opt/llama-builds BUILD=llama.cpp-20261112-d679a6d9d-pascal-p106-v4-b12246-c812 \
    MODEL=<Swift Q2_0's first shard> DRAFT=<the MTP draft> RUNG=$RUNG \
    bash docs/data/2026-11-12_c812_draft_pinned/run/r135_load.sh
  ssh "$RIG" "while [ ! -e $WORK/r135_$RUNG/DONE ]; do sleep 30; done"
done
# the host just before each start (start.tsv)
ssh "$RIG" 'date -Is; uptime; free -m; grep -E "^(MemTotal|MemFree|MemAvailable|Buffers|Cached|SwapCached|Active\(file\)|Inactive\(file\)|Shmem|SwapTotal|SwapFree|CmaFree):" /proc/meminfo'
# the load folders, renamed r138_<rung>, beside R123b's six (r123b_<rung>) and R135's (r135_pp1) in data/
ssh "$RIG" "tar cf - -C $WORK --transform 's/^r135_/r138_/' r135_embd r135_embd.out r135_embd.launch r135_pp1 r135_pp1.out r135_pp1.launch" | tar xf - -C data/
```

The tables, on any machine with those folders in `data/`:

```bash
P=docs/data/2026-11-12_r123b_prompt_ladder/run
OUT=rig REPO=<this repository> bash $P/r123b_logs.sh embd=data/r138_embd pp1=data/r138_pp1
python3 -I docs/data/2026-11-12_r123a_prompt_ladder/run/r123a_ladder.py --load base=data/r123b_base/load-A \
  --load embd=data/r138_embd/load-A --load pp1=data/r138_pp1/load-A --load base2=data/r123b_base2/load-A \
  --base base --base2 base2 --out rig
DATA=data OUT=rig bash docs/data/2026-11-12_c812_draft_pinned/run/r135_extra.sh embd=r138_embd pp1=r138_pp1 \
  r123b_embd=r123b_embd r135_pp1=r135_pp1
DATA=data OUT=rig bash docs/data/2026-11-12_r138_swap_cause/run/r138_swap.sh embd=r138_embd pp1=r138_pp1 \
  r123b_base=r123b_base r123b_embd=r123b_embd r123b_sparse=r123b_sparse r123b_queues=r123b_queues \
  r123b_b4096=r123b_b4096 r123b_base2=r123b_base2 r135_pp1=r135_pp1
```

The readers' md5s: `r123b_logs.sh` `99aa2735b470846bf0854ee65a066101`, `r123a_ladder.py`
`cd13ac0ade46bafb95f3f8cc22fd0503`, `r135_extra.sh` `30e6d200b7f0d358f31227d947299213`.

## Files

| File | What it is |
|---|---|
| [run/r138_swap.sh](run/r138_swap.sh) | the swap reader: the host before each load, swap after loading, after the replay and over time, the lowest MemAvailable |
| [rig/start.tsv](rig/start.tsv) | the host 1 s before each R138 load: load average, `free -m`, and `/proc/meminfo`'s cache, file LRU and swap lines |
| [rig/swap_summary.tsv](rig/swap_summary.tsv) | per load (R138's two, R123b's six, R135's): the host before the start, swap after loading, after the replay, over time, the lowest MemAvailable |
| [rig/swap_time.tsv](rig/swap_time.tsv) | every watchdog sample: seconds, MemAvailable and swap used (empty where a load logged no SwapFree) |
| [rig/swap.tsv](rig/swap.tsv), [rig/memavail.tsv](rig/memavail.tsv), [rig/reserves.tsv](rig/reserves.tsv) | `r135_extra.sh`: swap after loading and the replay; `embd`'s and `pp1`'s MemAvailable side by side; the reserves with the switch's column |
| [rig/ladder.tsv](rig/ladder.tsv), [rig/first.tsv](rig/first.tsv), [rig/replies.tsv](rig/replies.tsv), [rig/memory.tsv](rig/memory.tsv), [rig/decision.md](rig/decision.md), [rig/turns.csv](rig/turns.csv) | R123a's ladder: seconds a pass by bucket and class, wall and prompt time per pass, replies against `base`, the lowest MemAvailable, the classes in words, every replayed turn |
| [rig/loads.tsv](rig/loads.tsv), [rig/cards.tsv](rig/cards.tsv), [rig/host.tsv](rig/host.tsv), [rig/buffers.tsv](rig/buffers.tsv), [rig/switch_lines.txt](rig/switch_lines.txt), [rig/realloc_marks.tsv](rig/realloc_marks.tsv) | R123b's reader: times and counts per load, card and host memory, every buffer line, the switches, the reallocations |

## Provenance and licence

The engine is llama.cpp's licence (MIT); the model is UkisAI's Swift-1.5 GSQ-RCO Q2_0 of Qwen3.8-Flash-Next and the
MTP draft as the v3 release lists them. The prompts' text is the Python standard library's source on the rig (PSF
licence), tokenized there and not stored; the request sizes are the agent check's, from
[turns.csv](../2026-11-03_retype_agent/agent/turns.csv). The tables are this run's own measurements.
