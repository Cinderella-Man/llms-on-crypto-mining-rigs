# The decode step's eight host switches as one ladder, on a P106 (R122, Q1400, 2026-11-12)

What this is: the rig run that classes the v4 line's eight host-side switches of a speculative decode step. Before
each verify pass the server spends 7.512 ms on the host ([R86's step trace](../2026-11-11_r86_step_trace/README.md),
Q1313), and these eight switches each shorten a part of it. Each was written and checked on a CPU on its own branch.
Since the kernel merge, Miner Llama's `pascal-p106-v4` at `d097d6d88` holds them all, so one run puts them on one at a
time, each rung on top of the rungs before it:

| Rung | Switch added | Change | Est. before the run, ms a step |
|---|---|---|---:|
| `a` | `LLAMA_DRAFT_PREFETCH=1` | C615: each drafted token's n-gram rows issued as the drafter has them ([data](../2026-11-11_c615_draft_prefetch/README.md)) | up to 4.3 |
| `b` | `LLAMA_VERIFY_PREPARE=2` | C738: the verify pass prepared during the last drafting pass, card 0's inputs uploaded ([data](../2026-11-11_c738_verify_prepare/README.md)) | 1.75 |
| `c` | `LLAMA_MTP_GRAPH_KEEP=1` | C619: the draft's graph kept between its passes ([data](../2026-11-11_c619_draft_graph_reuse/README.md)) | 0.95 |
| `d` | `LLAMA_KV_SEQ_RM_BOUND=1` | C740: `seq_rm` reads one sequence's cells from its index ([data](../2026-11-11_c740_seq_rm_bound/README.md)) | 0.67 |
| `e` | `LLAMA_SERVER_YIELD=2` | C741: one yield to the task queue a step, the worker not woken ([data](../2026-11-11_c741_server_yields/README.md)) | 0.37 |
| `f` | `LLAMA_MTP_EMBD_FIRST=1`, `LLAMA_MTP_DRAFT_CAP=1`, `LLAMA_MTP_SKIP_LAST_ROW=1` | C771: the draft's k-pool update in its layers' split ([data](../2026-11-11_c771_draft_kpool_split/README.md)); C744: the drafter stops at the slot's cap ([data](../2026-11-11_c744_draft_cap/README.md)); C742: no hidden row back from the last drafting pass ([data](../2026-11-11_c742_last_draft_row/README.md)) | C771 0.16 to 0.24, C742 0.19, C744 near 0 |

The line under them is the one the next preset runs: R90's `gains` line on the v4 build, with the micro-batch taper
at `short=600:64` ([R60](../2026-11-12_r60_taper_v4/README.md)), without `LLAMA_SAMPLING_TOP_K_ROWS=1` (it reallocates
the prompt graph, [FN84](../2026-11-12_fn84_graph_realloc/README.md)), and with C720's and C719's k-pool switches
([run/preset.env](run/preset.env)). Registry [Q1400](../../../quants/experiments.md); the rungs' own rows are Q1354,
Q1345, Q1334, Q1344, Q1350 and Q1361 (C744 and C742 had none).

## Result

**Every rung is a gain, the greedy reply is base's in all thirteen loads, and the stack takes 7.21 ms off the step:
111.94 ms against 119.15 (−6.05 %).** The rungs' estimates added up to about 8 ms. The nine cards' own work did not
change (verify_gpu 91.80 ms in `base`, 91.71 in `f`): all of the time came off the host.

Each load's value is the mean of its two requests' median typical step (S37's `arms.py`). A rung is classed against
the two loads of the rung before it: a gain when it is below both and their mean minus it is more than their spread.

| Rung | Its two loads, ms | The rung before, ms | Minus their mean | Their spread | Class | The greedy steps paired with the rung before's, ms |
|---|---|---|---:|---:|---|---|
| `a` (C615) | 114.76, 114.66 | `base` 119.22, 119.09 | **−4.44** | 0.13 | **gain** | −4.50 ± 0.20, −4.80 ± 0.17 |
| `b` (C738) | 113.85, 114.18 | `a` 114.76, 114.66 | **−0.70** | 0.10 | **gain** | −0.58 ± 0.22, −0.41 ± 0.15 |
| `c` (C619) | 112.89, 113.08 | `b` 113.85, 114.18 | **−1.03** | 0.33 | **gain** | −1.32 ± 0.19, −1.34 ± 0.20 |
| `d` (C740) | 112.65, 112.64 | `c` 112.89, 113.08 | **−0.33** | 0.19 | **gain** | −0.48 ± 0.08, −0.25 ± 0.15 |
| `e` (C741) | 112.44, 112.64 | `d` 112.65, 112.64 | **−0.11** | 0.01 | **gain** (at the rule's edge) | +0.03 ± 0.15, +0.11 ± 0.16 |
| `f` (C771, C744, C742) | 111.89, 111.99 | `e` 112.44, 112.64 | **−0.60** | 0.20 | **gain** | not paired (below) |
| the stack, `f` against `base` | 111.89, 111.99 | `base` 119.22, 119.09 | **−7.21** | 0.13 | **gain** | — |

Each load is also classed on its own value against the rung before ([ladder.md](ladder.md)), and every load of every
rung is a gain that way too. Paired: the greedy request's typical steps against the first load of the rung before,
step by step (mean difference ± its standard error, 81 steps).

- **The replies.** The greedy request (temperature 0) wrote `base`'s reply in all thirteen other loads: the same text,
  256 tokens generated, 254 drafted and 170 accepted. The long request samples at temperature 1.0, so its text and
  its draft counts differ from load to load, as they do between `base` and `base-2`. It is compared by its steps, not
  its text.
- **Rung `e` is a gain by the rule, but only just.** Its reference loads `d` and `d-2` differ by 0.010 ms, the smallest
  spread on the ladder (the other rungs': 0.129, 0.099, 0.330, 0.193 and 0.198). `e-2`'s 112.636 ms is 0.008 below
  `d-2`'s 112.644, and the greedy steps paired with `d`'s are +0.03 ± 0.15 and +0.11 ± 0.16 ms. Against any other
  rung's spread C741 would be negligible. Either class keeps it on the line, because the replies are the same.
- **Where each rung's time came from** (the median per part, the mean of the two requests, each load's two values;
  [tables/arms_all.md](tables/arms_all.md)):

  | Load | Step | The draft's span | The server's turn before the verify | verify_rest | verify_gpu |
  |---|---:|---:|---:|---:|---:|
  | `base` | 119.22, 119.09 | 22.74, 22.40 | 7.41, 7.08 | 4.68, 4.67 | 91.80, 91.78 |
  | `a` | 114.76, 114.66 | 17.99, 18.00 | 2.84, 2.82 | 4.65, 4.62 | 91.80, 91.63 |
  | `b` | 113.85, 114.18 | 17.37, 17.59 | 2.33, 2.35 | 4.63, 4.61 | 91.53, 91.76 |
  | `c` | 112.89, 113.08 | 17.19, 17.30 | 2.34, 2.37 | 3.68, 3.67 | 91.69, 91.74 |
  | `d` | 112.65, 112.64 | 16.93, 17.01 | 1.98, 1.98 | 3.68, 3.68 | 91.80, 91.72 |
  | `e` | 112.44, 112.64 | 16.87, 16.96 | 1.95, 2.03 | 3.59, 3.58 | 91.67, 91.71 |
  | `f` | 111.89, 111.99 | 16.43, 16.62 | 1.61, 1.81 | 3.57, 3.57 | 91.71, 91.64 |

  C615 shortens the server's turn before the verify pass, where the verify reads its n-gram rows: 7.41 and 7.08 ms, then
  2.84 and 2.82. C738 and C740 shorten the same turn again. C619's rung shows in verify_rest, the host's part of the
  verify pass (4.63 and 4.61 ms, then 3.68 and 3.67). The draft's span falls with every rung.
- **`f` is not paired, and only its step compares with `e`'s.** With C771 the draft's k-pool update runs in the same
  evaluation as its layers, so the profile counts 4 draft passes a step instead of 8. S37's `arms.py` pairs only steps
  of the same shape. For the same reason the per-pass parts in [tables/rung_f.md](tables/rung_f.md) (the leads, the
  turns before the last) are not comparable between the two rungs. The step and the draft's span are.
- **`f` also shortens a server's first request.** The greedy request is the first after the server starts. Its prompt
  took 7,522.2 and 7,545.3 ms with `f`, against `e`'s 8,516.7 and 8,518.5. Its generation time outside the profiled
  steps was 1,692.4 and 1,668.3 ms, against 3,622.2 and 3,634.4. The server's speed for that reply was 23.49 and 23.56
  tok/s, against 19.90 and 19.87. The second request shows no such change. There, every load spends either about
  1.25 s or about 2.26 s of generation outside the steps, whatever its rung (`f` 1,244.2, `f-2` 2,260.9; `base`
  2,270.2, `base-2` 1,275.4; [ladder.md](ladder.md), last table). So this saving comes once per server start. What the
  server does in the time outside the steps of every reply is a new row (Q1410).
- **The switches acted.** C615's closing line is in every load from `a` on: in `a`, for example, 708 calls issued the
  rows of 708 tokens, with 0 tokens over the cap. C738's line is in every load from `b` on, and every verify decode was
  prepared ahead of its last token ids (0 not prepared, 0 dropped). Each load's lines are in
  [switch_lines.tsv](switch_lines.tsv). The other six switches log no closing line, so they show only in the step.
- **No graph reallocations.** All 28 requests show 0 graph reallocations of either kind
  ([reallocs.tsv](reallocs.tsv)), so the line is free of FN84's prompt-graph reallocations.
- **The host's memory.** The lowest MemAvailable per load ([vm.tsv](vm.tsv)) was 499 and 515 MiB in `base`, 519 and
  510 in `f`, and lowest in `d-2`, at 478. Swap stood at 19 MiB from start to end of every load, and nothing was
  swapped out. `base` had 3,347 major faults as the run's first load, with the files still cold. Every other load had
  at most 15.
- **Est. for an agent task:** the row's own scale (8 ms of a step, up to 3 % of a task) puts the stack's 7.21 ms at
  est. 2.7 % of an agent task. R72's agent check measures it.

## What ran

- **The rig:** nine P106-090 cards (Manli M-P106L9-N6G, the stock 4 GB host), idle; no other work ran on it
  during the loads.
- **The build:** Miner Llama `pascal-p106-v4` at `d097d6d88` (the kernel merge's rig tarball, `TESTS=1`, sm_61).
- **The model:** Swift GSQ-RCO Q2_0 (UkisAI's Swift-1.5 file of Qwen3.8-Flash-Next, the maker's three shards) with the MTP
  draft; the n-gram table is read from the third shard.
- **The line:** [run/preset.env](run/preset.env), the same in every load; the rungs' switches went on top through
  `LOAD_ENV` ([run/r122_loads.sh](run/r122_loads.sh)).
- **Fourteen loads in one run**, 12:26:01 to 13:35:25: `base a b c d e f`, then the same seven again (`base-2` …
  `f-2`). Each load runs R108's driver ([rig_loads.sh](../2026-11-11_c615_draft_prefetch/run/rig_loads.sh)) as
  follows:
  - it drops the n-gram table's file from the page cache and checks that the host is quiet;
  - it starts the server through `bench/agent_check/rig_server.sh` with the host profile (`GGML_CUDA_PROFILE_HOST`)
    and its RAM watchdog;
  - it samples the host's memory and paging every 0.5 s
    ([vm_sampler.sh](../2026-11-11_c761_ngram_prefetch/run/vm_sampler.sh));
  - it sends S37's two requests: the greedy request (79 prompt tokens, temperature 0, 256 tokens), then the agent's
    first-turn request (2,760 prompt tokens, the preset's sampler, at most 256 tokens).

  Every request exited 0, and no watchdog fired.
- **The reading** ran on a build machine's CPU at `nice -n 19` ([run/tables.sh](run/tables.sh)). For each load, S37's
  `step_split.py` and the memory from `vm.tsv`. Then S37's `arms.py` three ways: every load against `base`, each rung
  against the one before it, and `f` against `base`. Last, [run/ladder.py](run/ladder.py)'s summary.

## Files

| File | What it holds |
|---|---|
| [ladder.md](ladder.md), [ladder.tsv](ladder.tsv) | each rung against the one before and the stack, per load and per rung; every load's server counts and memory; the server's time outside the profiled steps |
| [vm.tsv](vm.tsv) | per load: the lowest MemAvailable, swap at the start and at most, major faults, pages swapped in and out |
| [switch_lines.tsv](switch_lines.tsv) | each load's closing lines from the switches (C615, C738) |
| [reallocs.tsv](reallocs.tsv) | the graph reallocations per request (the server's own count) |
| [tables/arms_all.md](tables/arms_all.md) (`.tsv`, `_class.tsv`, `_paired.tsv`) | S37's `arms.py` over all fourteen loads, `base` and `base-2` as the reference: the server's timings, every part of the step, the greedy replies |
| [tables/rung_a.md](tables/rung_a.md) … [tables/rung_f.md](tables/rung_f.md) (with their `.tsv` files) | `arms.py` per rung, the rung before as the reference |
| [tables/stack.md](tables/stack.md) (with its `.tsv` files) | `arms.py`, `f` against `base` |
| `tables/split_<load>.md`, `tables/steps_<load>.tsv` | S37's `step_split.py` per load: the step's parts and every step's row |
| [run/preset.env](run/preset.env) | the line under the ladder |
| [run/r122_loads.sh](run/r122_loads.sh) | the rungs (`LOAD_ENV`) on R108's driver |
| [run/tables.sh](run/tables.sh), [run/ladder.py](run/ladder.py) | the reading |

## Commands

```bash
# on the rig, the rig idle (S: a folder holding this repository's files at their paths; BIN: the build's bin/;
# MODEL: the Swift Q2_0 first shard; DRAFT: the MTP draft; TABLE: the shard holding the n-gram table;
# LONG: the first-turn request built by BUILD_SSH=<build machine> bash bench/agent_check/first_turn_request.sh long_request.json)
cd $S && BIN=$BIN MODEL=$MODEL DRAFT=$DRAFT TABLE=$TABLE \
  PRESET=$S/docs/data/2026-11-12_r122_host_ladder/run/preset.env \
  RIG_SERVER=$S/bench/agent_check/rig_server.sh \
  VM_SAMPLER=$S/docs/data/2026-11-11_c761_ngram_prefetch/run/vm_sampler.sh \
  RIG_LOADS=$S/docs/data/2026-11-11_c615_draft_prefetch/run/rig_loads.sh \
  REQ_LONG=$LONG REQ_GREEDY=$S/docs/data/2026-11-04_draft_one_card/run/greedy_request.json OUT=$OUT \
  setsid nohup bash docs/data/2026-11-12_r122_host_ladder/run/r122_loads.sh > console.out 2>&1 < /dev/null &
# it ends with $OUT/DONE (about 70 minutes)

# on a build machine, CPU only ($OUT copied there; REPO a checkout of this repository)
REPO=$REPO nice -n 19 bash $REPO/docs/data/2026-11-12_r122_host_ladder/run/tables.sh $OUT tables
```

## Limits

- **One order.** A rung's class is what its switch adds on top of the rungs before it, in this order. It is not what
  the switch gives alone, nor in another order.
- **Two loads a rung.** The reference spread is the difference between the rung before's two loads. That spread was
  0.010 ms for `e` and 0.330 ms for `c`, so `e`'s class rests on the smallest spread of the run.
- **One depth.** The requests are the greedy request's 79 tokens and the first turn's 2,760. Deep contexts were not
  measured here: R113's planned depths of 16,384 and 49,152 tokens for C771 were not run.
- **Two replies a load, 256 tokens each, and not an agent session.** Agentic speed (`bench/agent_check`) is R72's.
- **The table's page cache.** This rig has no `fincore`, so the n-gram table's resident bytes before and after the drop
  are not recorded. The drop itself ran before every load.

## Provenance and licence

Measured on 2026-11-12 on one rig of nine P106-090 cards, with the Miner Llama fork's `pascal-p106-v4` (llama.cpp,
MIT). The tables are this project's own measurements. The models are UkisAI's Swift-1.5 GSQ-RCO Q2_0 of
Qwen3.8-Flash-Next and the MTP draft; neither is republished. The greedy request is
[greedy_request.json](../2026-11-04_draft_one_card/run/greedy_request.json). The first-turn request is built by
`bench/agent_check/first_turn_request.sh` from this project's recorded agent requests and is not in this folder. The
replies are not in this folder either; the tables hold their counts and whether they match `base`'s. Licence: the
repository's.
