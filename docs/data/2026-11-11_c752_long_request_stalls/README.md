# C752 (Q1328): the long request's ~117 ms host waits come with swapping, on v3.0 and the v4 night build (2026-11-11)

What this answers: in R86's two loads of the v4 night build ([2026-11-11_r86_step_trace](../2026-11-11_r86_step_trace/README.md),
`pascal-p106-v4-b12246-night` `c408dabcb`) the agent's first-turn request (2,760 prompt tokens) held 7 and 23
generation steps that waited about 117 ms on the host before card 0's launch, while 78 and 203 MiB of the 4 GB host
went to swap. S37's loads of the v3.0 release build ([2026-11-10_served_step_split](../2026-11-10_served_step_split/README.md),
`pascal-p106-v3-b12076-ladder5` `1551f5fe1`) ran that request first after the load and had none. Is it the build or
the order? And what grows in host memory, and what is paged back in? The registry row is Q1328 in
[quants/experiments.md](../../../quants/experiments.md).

This is data only, with no new run. Three sources were read:
- R57's agent check of both builds on one rig the same morning ([2026-11-11_r57_v4night_p106](../2026-11-11_r57_v4night_p106/README.md),
  12 tasks an arm, with its server logs, the proxy's timings and the host monitors);
- R86's two loads of the night build;
- S37's six loads of v3.0, two of them with the draft's row cache.

The model is the v3 preset throughout: UkisAI's Swift-1.5 GSQ-RCO Q2_0 of Qwen3.8-Flash-Next on nine P106-090 cards
with the MTP draft.

## Result

**It is not the build. v3.0 waits the same way whenever the long request swaps, and the night build does not wait
more.** On both builds, every long request that swapped during its own run held such waits (medians 117.1, 117.7, 115.9 and
120.5 ms). The long requests whose swap stayed flat held none, or one. In the agent check the night build has fewer slow
requests than v3.0, not more. Its first request, the same 2,780-token first turn, did not swap and did not slow down.
v3.0's first request swapped 155 MiB and took 1.35 s longer than the fit, about 11.6 waits' worth. The order matters
only through the swap: in R86 the greedy request ran first and left the long request starting at 383 and 385 MiB of
MemAvailable, against 897 to 921 MiB in S37's loads.

**What pages back in is not settled by the data.** Two leads remain, and one rig row (R96, below) separates them:
- the server's main thread faulting on its own swapped-out pages;
- its reads of the n-gram table waiting behind the swap's I/O on the same SSD.

### The diagnostic loads: waits against swap, per build

[loads_vm.md](loads_vm.md) (`run/load_vm.py`): the long request of each of R86's and S37's loads. Swap used is read
from the RAM watchdog once a second. Swap-in, disk reads and iowait come from `vmstat` over the generation, weighted
by overlap (10-second lines). The waits are S37's measure, a host turn over 50 ms after a drafting pass, from each
folder's `waits.tsv`.

| Build | Load | MemAvailable at launch, lowest (MiB) | Swap used at launch, first token, end (MiB) | Steps with a wait | Median wait, ms | Swap-in in generation, MiB | Iowait in generation, % |
|---|---|---|---|---:|---:|---:|---:|
| night | `trace` (greedy first) | 385, 238 | 0, 39, 78 | 7 | 117.1 | 0.5 | 7.2 |
| night | `prof` (greedy first) | 383, 271 | 0, 203, 203 | 23 | 117.7 | 5.5 | 11.3 |
| v3.0 | `d1cache` (long first) | 920, 320 | 20, 269, 357 | 46 | 115.9 | 6.6 | 18.6 |
| v3.0 | `d1on` (long first) | 914, 88 | 27, 28, 91 | 16 | 120.5 | 0.1 | 11.3 |
| v3.0 | `d1carry` (long first) | 921, 96 | 24, 24, 27 | 1 | 73.3 | 0.0 | 3.0 |
| v3.0 | `d1off` (long first) | 921, 78 | 16, 20, 20 | 0 | | 0.0 | 5.7 |
| v3.0 | `d1off-2` (long first) | 916, 86 | 27, 27, 28 | 0 | | 0.0 | 3.5 |
| v3.0 | `warm` (long first, no profile) | 897, 75 | 0, 0, 29 | | | 0.3 | 5.7 |

- **The waits follow swap traffic, not MemAvailable's low point.** The four loads with waits swapped 63 to 337 MiB
  out during the long request, before its first token or during its generation. The loads without waits swapped 0 to
  4 MiB, even where MemAvailable fell to 75 to 96 MiB. With waits, iowait over the generation was 7.2 to 18.6 %;
  without them it was 3.0 to 5.7 %.
- **The wait is the same length on both builds:** medians 117.1, 117.7, 115.9 and 120.5 ms. S37 found its two
  (`d1cache`, `d1on`) with the draft's row cache on and parked the question as Q1209. The preset's own first long
  request now swaps as well, in R86 and in R57's v3.0 arm (below), which was Q1209's condition to reopen.
- **The memory pressure counter stays low.** In `prof` the host's `/proc/pressure/memory` "some" average over 10 s
  was at most 2.89 %, while 2.99 s of waits fell in est. 6 s of steps. Reclaim and refaults of recently active pages
  both count as memory stalls there, so most of each wait is neither of them. The two leads below say what it could
  be instead.

### The agent check: per request, per build

[r57_requests.md](r57_requests.md), [r57_requests.tsv](r57_requests.tsv) (`run/agent_requests.py`). R57's log keeps
no per-step times, so this is per request. A generation step is one verify pass (generated tokens minus accepted
drafts). One least-squares line for both arms fits the generation time:

generation ms = −54.2 + steps × (128.90 + 3.418 × depth / 1,000)

The line was refitted twice without the outliers, over 108 of 124 requests of 10 steps or more. A request's excess
is its time minus the line. A request is above the cut when its excess is more than 404.2 ms, that is three scaled
median absolute deviations (134.6 ms each) above the median residual. One R86 wait adds about 117 ms, so the cut
is about 3.5 waits.

| Build (arm) | Requests | Steps | Median ms a step | Requests above the cut | Their excess, s | A task, s | Share of a task (2.75 min at equal work) | Swap-out in the requests, MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| night (`v4night`) | 65 | 9,943 | 144.9 | 5 | 5.33 | 0.44 | 0.27 % | 864 |
| v3.0 (`v3rel`) | 59 | 4,944 | 145.7 | 10 | 9.63 | 0.80 | 0.49 % | 985 |

The arms' first request is the agent's first turn of `bugfix-python-affine-cipher`, 2,780 prompt tokens, the first
request after the load. It is R86's long request in S37's order:

| Build | Steps | ms a step | Excess, ms | Checkpoints saved, ms | MemAvailable at start, lowest (MiB) | Swap at start, end (MiB) |
|---|---:|---:|---:|---|---|---|
| night | 19 | 152.2 | 314 | 3, 1,895 | 919, 229 | 0, 1 |
| v3.0 | 19 | 206.9 | 1,353 (as 11.6 waits) | 3, 1,936 | 819, 173 | 10, 165 |

- **R86's pattern shows up on v3.0, not on the night build.** v3.0's arm started second, 100 MiB lower, after the
  night arm's server had stopped with 10 MiB left in swap. Its first request swapped out 117.3 MiB during the prompt.
- **Requests that swapped out 20 MiB or more during their prompt** (R86's waits followed such a swap-out): the night
  build's 8 have a median excess of −24 ms and one above the cut. v3.0's 10 have a median of +252 ms, five above the
  cut and 5.70 s in all (0.47 s a task). So in the agent check, a swap-out costs v3.0 more than the night build. One
  candidate is the night build's draft rows: they stay on card 8 (#31927) where v3.0 copied them into new host
  memory after each catch-up. One arm each cannot say more.
- **Swap in the agent check is the norm on both builds.** Swap use peaked at 914,968 and 1,028,584 kB (R57's report):
  the kernel kept the n-gram table's page cache and moved anonymous pages out. Even so, 109 of the 124 requests of 10
  steps or more sit within the cut.

### What grew in host memory

- **The checkpoints, the same on both builds.** Every load and arm saved two 112.6 MiB recurrent-state checkpoints
  for the greedy request and three for the long one, in host memory (`--ctx-checkpoints 4`). The save times match:
  1,257 to 1,284 ms for the greedy request's two. For the long request's three: 1,889 to 1,936 ms without swap,
  2,164 ms in `prof` and 4,385 ms in `d1cache`, where it swapped.
- **The night build does not hold more host memory.** On R57's rig, 20 s after the load, MemAvailable was 918,544 kB
  on the night build and 942,516 kB on v3.0, one reading each. In the agent arms the lowest MemAvailable was 206,680
  against 197,336 kB, and swap use peaked at 914,968 against 1,028,584 kB.
- **The first request after a load takes 500 to 850 MiB on either build.** In R86 the greedy request took MemAvailable
  from 917 to 377 MiB (`trace`) and from 885 to 372 (`prof`). The long request launched at 385 and 383 MiB; in `prof`
  it was 576 a second later, when the greedy request's checkpoints were dropped. On v3.0, S37's long requests took it
  from 914 to 921 MiB down to 75 to 96 without swapping. R57's first requests took it from 919 to 229 MiB (night, no
  swap) and from 819 to 173 (v3.0, 155 MiB swapped). Two checkpoints are 225 MiB and three are 338. The rest is not
  itemised by these monitors.
- **The night merge's code** (`git diff 1551f5fe1 c408dabcb`: input staging, pinned buffers, upstream #31927's
  `h_nextn`, the checkpoints, C100's tables, C720's stage):
  - #31927 keeps the target's hidden rows in an 80.04 MiB buffer on card 8. The draft no longer copies a catch-up's
    rows into a host vector, as v3.0 did (`verify_h`, up to rows × 10,240 floats).
  - The context's pinned output buffer still reserves room for those rows on the host. `has_embd_nextn` follows
    `embeddings_nextn`, which the MTP draft sets, so the space is reserved on both builds. The night build no longer
    reads it when the device buffer is on (`write_h_nextn`). If the target's rows are unmasked, that is
    n_embd × n_batch floats, est. 80 MiB at `-b 2048`. C759 (Q1336) drops it.
  - Every other new host allocation in those paths sits behind a switch that was off in R86 and R57:
    - C217's in-call checkpoint copies (`data_tgt`, `data_dft`);
    - the k-pool tables and checks of C100 (`LLAMA_QSA_KPOOL_DEV`, `_CHECK`);
    - C720's stage (`LLAMA_QSA_KPOOL_STAGE`, graph tensors in the compute buffers).

### What is paged back in: the two leads

The trace puts six of R86's seven waits between the verify pass's inputs set (`v.ub.in`) and card 0's
`graph_compute` entry, and one in setting the inputs. That stage holds the graph's CPU part and then the copies of
card 0's inputs. The CPU part includes the n-gram table's lookup, read from the mapped file (the 27,465.95 MiB
`CPU_Mapped` buffer). The 260.50 MiB `CPU` model buffer is pageable host memory, so it can be swapped out.
Two leads fit the numbers above:

1. **The main thread faults on its own pages that went to swap.** `prof` and `d1cache` swapped 5.5 and 6.6 MiB back
   in during the generation, and the one wait in setting the inputs (96.0 ms in `trace`) is a write to host memory.
   Against it: `d1on` swapped in 0.1 MiB and still held 16 waits.
2. **The table's reads queue behind the swap's I/O.** On the rig that ran R86 and S37, the swap file (`/swap.img`) and the
   table's shard are on the same LUKS-encrypted SATA SSD. The table's rows are read once per token, 16 rows a token (est.
   0.6 ms a token without contention: [the book's n-gram table section](../../book/05-nine-cards.md#the-n-gram-table-on-the-ssd)).
   `d1on` (63 MiB swapped out during the generation, 0.1 MiB in) and `trace` (39 MiB out, 0.5 MiB in) fit this
   better.

Neither lead explains why the wait is so constant (112.66, 111.00, 114.26, 115.06 and 117.30 ms in five of `trace`'s six; the sixth 161.01). The trace stops at
card 0's entry, the monitors sample once a second, and `vmstat` once every 10 s, so the data cannot split them.

## What decides it: the rig row R96 (Q1337)

[run/rig_swap.sh](run/rig_swap.sh): R86's driver with R86's order (the greedy request first; both of R86's loads
swapped), on the night build, untimed, est. 15 min with two loads:
- `swap`: the preset;
- `vram`: the preset with `--ctx-checkpoints-vram 4` (Q440: a slot's checkpoints on the cards, so 112.6 MiB a
  checkpoint stays off the host).

Each load runs four tools beside the preset:
- the host profile;
- the step trace;
- C701's per-split trace (`LLAMA_UBATCH_TRACE=1`: each split's wait, copy and launch, so the CPU split's table reads
  and the copies come apart);
- [run/proc_sampler.sh](run/proc_sampler.sh) every 0.05 s.

The sampler records the main thread's state and kernel wait channel, its major faults, the process's VmRSS and
VmSwap, the host's `pswpin`, `pswpout`, `pgmajfault` and workingset refaults, MemAvailable and SwapFree. It forks
nothing per sample: on a build machine it took 0.059 s a sample at 0.05 s.

What decides it:
- A wait whose samples show the thread in `D` state in a swap-in, with `majflt` and `pswpin` rising, is lead 1.
- A wait in a block-layer or filemap wait while `pswpout` rises and `pswpin` does not, inside the CPU split's span
  of C701's trace, is lead 2.
- `vram` swapping less and waiting less makes VRAM checkpoints the fix (Q440, already built). If it still swaps, the
  host demand is elsewhere, and the sampler's VmRSS steps name the moment it grows.

## Files

| File | What |
|---|---|
| [loads_vm.md](loads_vm.md) | `load_vm.py`: every request of R86's two and S37's six loads; MemAvailable, swap, waits, swap-in, disk reads and iowait over the generation |
| [r57_requests.md](r57_requests.md) | `agent_requests.py` on R57's two arms: the fit, the per-arm table, the requests above the cut, the first requests, excess against swap-in and against swap-out in the prompt |
| [r57_requests.tsv](r57_requests.tsv) | the same per request: steps, generation time, depth, fit, excess, checkpoints, MemAvailable, swap, vmstat sums, pressure |
| [run/agent_requests.py](run/agent_requests.py) | per request of an agent check against one fit of the arms' steps, with the host's swap around it |
| [run/load_vm.py](run/load_vm.py) | per request of a diagnostic load: the waits, swap and disk traffic while it ran |
| [run/rig_swap.sh](run/rig_swap.sh) | R96's driver (on the rig, detached) |
| [run/proc_sampler.sh](run/proc_sampler.sh) | R96's sampler of the server's main thread and the host's paging |

## Commands

On a build machine's CPU. `R57` is R57's collected agent-check folder (`proxy.jsonl`, `rig/` with the server logs,
`vmstat.log` and the RAM watchdog's files). `R86` and `S37` are the two rig runs' output folders, one folder a load.
The `waits.tsv` files are the published ones.

```bash
nice -n 19 python3 -I run/agent_requests.py $R57 --arms v4night,v3rel --tsv r57_requests.tsv > r57_requests.md
nice -n 19 python3 -I run/load_vm.py --load trace=$R86/trace --load prof=$R86/prof \
  $(for l in warm d1off d1cache d1carry d1on d1off-2; do echo --load $l=$S37/$l,long,greedy,smoke --build $l=v3.0; done) \
  --build trace=night --build prof=night \
  --waits ../2026-11-11_r86_step_trace/waits.tsv --waits ../2026-11-10_served_step_split/waits.tsv > loads_vm.md
```

R96, on the rig with the rig idle. The variables are as in R86's command (`BIN` is the night tarball's `bin/`; `S`
is a folder with this repository's files; `LONG` is the first-turn request built by `bench/agent_check/first_turn_request.sh`):

```bash
BIN=$BIN MODEL=$MODEL DRAFT=$DRAFT PRESET=$S/release/flash-next-v3/settings.env \
  RIG_SERVER=$S/bench/agent_check/rig_server.sh SAMPLER=$S/docs/data/2026-11-11_c752_long_request_stalls/run/proc_sampler.sh \
  REQ_LONG=$LONG REQ_GREEDY=$S/docs/data/2026-11-04_draft_one_card/run/greedy_request.json OUT=$OUT \
  setsid nohup bash $S/docs/data/2026-11-11_c752_long_request_stalls/run/rig_swap.sh > console.out 2>&1 < /dev/null &
```

## Limits

- R57's server log has no per-step times. So the agent check's count is per request, against a fit, and a wait
  hidden in a long reply's noise (scaled MAD 134.6 ms a request) is not counted.
- One arm a build, and the arms ran one after the other: v3.0's started with 100 MiB less MemAvailable and 10 MiB in
  swap.
- `vmstat`'s 10-second lines are wider than most of the waits' spans, and the watchdog's are 1 s.
- The rig that ran R57 is not the rig that ran R86 and S37. The table's location on R57's rig was not read.

## Provenance and licence

The model is UkisAI's Swift-1.5 GSQ-RCO Q2_0 of Qwen3.8-Flash-Next (its maker's licence) with its MTP draft layer. The
tasks are bench/agent_check's bugfix set. The engine is Miner Llama (MIT, as llama.cpp). The tables are this
project's measurements, under its licence.
