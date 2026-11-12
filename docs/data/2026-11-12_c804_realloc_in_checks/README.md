# Graph reallocations counted in every switch's rig check (C804, Q1408, 2026-11-12)

What this changed: the drivers that run a switch's rig check, and the multi-card correctness runs on the quant
machine, now count the main model's graph reallocations per request and mark a switch whose loads reallocate where its
line without it does not. Registry row [Q1408](../../../quants/experiments.md).

Why: C691's switch (`LLAMA_SAMPLING_TOP_K_ROWS=1`) changed the graph's node count between a prompt's micro-batches, so
every request reallocated the graph, each time after a wait for all nine cards
([FN84](../2026-11-12_fn84_graph_realloc/README.md)). Its own rig check (R83,
[2026-11-11_r83_c691_p106](../2026-11-11_r83_c691_p106/README.md)) timed verify passes only, where every micro-batch has
outputs and the node count never changes, so the check passed. The cost, a 3,313-token turn read in 42.49 s against
21.14 s without the switch (FN84, one request each), showed up two runs later in R90's agent check and in R60's prompt
lane. With the count in every driver's summary, a switch that adds reallocations is marked in its own check, before it
joins a stacked line.

Nothing here is a measurement. Every check reads server logs that earlier rows wrote on P106 rigs, or runs a driver
against stand-ins on the quant machine's CPU.

## What a driver counts

llama-server prints a block per request (`print_timing`) at its default verbosity. On Miner Llama's v4 builds the block
ends with the main model's running count of graph reallocations since the load:

```
slot print_timing: id  0 | task 6 | prompt eval time =   21139.87 ms /  3313 tokens (...)
slot print_timing: id  0 | task 6 |   graph reallocs =          0    /     0        (same size) / (changed size)
```

A request's reallocations are the difference to the previous request's count: the first number counts those at an
unchanged graph size, the second those at a changed one. This is R60's
[`run/realloc_counts.py`](../2026-11-12_r60_taper_v4/run/realloc_counts.py), now
[`scripts/realloc_counts.py`](../../../scripts/realloc_counts.py) with its default table unchanged. Only the main
model's context is counted. The MTP draft's own graph reallocates 2 to 3 times a request on every line (C803,
[Q1407](../../../quants/experiments.md)) and has its own count, which this one leaves out.

A load is marked against its line without its switch:

| Mark | Meaning |
|---|---|
| `reallocates` | the load has requests with a reallocation, and none of its references has any: the switch adds them |
| `reallocates more` | both have some, and a larger share of the load's requests has one |
| `as its line` | both have some, and the load's share is no larger |
| `none` | no request of the load has one |
| `-` | no reference (the line itself): the count alone |

## The drivers

Each driver runs the same loads and requests as before; only its summary gains the count, and each one's header says
so.

| Driver | Used by | What it adds |
|---|---|---|
| [`2026-11-04_prompt_taper/run/rig_taper.sh`](../2026-11-04_prompt_taper/run/rig_taper.sh) | R43's prompt lane (`r43_run.sh`), R60, FN84, R123a, R123b | `load-<L>/reallocs.tsv` (one row per request) and a line in `run.log`: `load A: graph reallocations: W of N requests (S at the same size, D changed)` |
| [`2026-11-11_c615_draft_prefetch/run/rig_loads.sh`](../2026-11-11_c615_draft_prefetch/run/rig_loads.sh) | R108, R122 | `<load>/reallocs.tsv`, a count line per load in `status.txt`; at the end `realloc_loads.tsv`, `realloc_marks.tsv` and a `status.txt` line per load that reallocates where its line does not. A load's references are the loads whose switches (`LOAD_ENV`'s, and the step trace of a `trace` load) are the largest proper subset of its own: with R122's ladder the rung before (both of its loads), with R108's default loads `dp` against `prof` and `prof-2` |
| [`bench/deep_lane/deep_table.py`](../../../bench/deep_lane/deep_table.py) | `bench/deep_lane` | a line in the table's checks: the requests with a reallocation per load (from `server-<load>.log`), and an arm marked when its loads reallocate where the control's do not |
| [`2026-11-12_r123a_prompt_ladder/run/r123a_logs.sh`](../2026-11-12_r123a_prompt_ladder/run/r123a_logs.sh), [`2026-11-12_r123b_prompt_ladder/run/r123b_logs.sh`](../2026-11-12_r123b_prompt_ladder/run/r123b_logs.sh) | R123a, R123b | `realloc_marks.tsv`: `scripts/realloc_counts.py --marks ladder` over the loads, every rung against the rung before, a repeat (`base2`) in its rung |

`scripts/realloc_counts.py` marks any set of loads: `--marks ladder` (rungs in the order of the logs; a name plus `2`
or `-2` is a repeat of that rung), or explicit `load=ref[,ref]` pairs. The rig drivers count with a short `awk`
program of their own (`realloc_count`), so a rig folder needs no new file.

`bench/agent_check` is not changed yet: tonight's agent check (R72) runs from it. Its part, the same count per server
log in `report.py gonogo`'s server-log lines and an arm marked against the other in `report.py compare`, follows once
R72 is recorded.

## The correctness runs on the quant machine

ggml's scheduler reads `GGML_SCHED_DEBUG_REALLOC`: at 1 it aborts the process at a reallocation of an unchanged graph
size (`ggml/src/ggml-backend.cpp`, `ggml_backend_sched_alloc_splits`), at 2 at any reallocation. Such a reallocation
means the plan the scheduler made does not fit a micro-batch of the same graph, which is what C691's switch caused.
The variable reaches every scheduler in the process, the MTP draft's too, and the draft's graph reallocates at an
unchanged size on every line: C692's seven loads (v3.0 and v4, three newer cards) logged 44 to 152 such
reallocations of the draft's 204 to 220-node graph per load. So:

| Script | Its check | Default |
|---|---|---|
| [`2026-11-11_fn64_sparse_prompt_ptx/run/kld_depth.sh`](../2026-11-11_fn64_sparse_prompt_ptx/run/kld_depth.sh) | FN64 (C705), llama-perplexity on several cards, no draft | `GGML_SCHED_DEBUG_REALLOC=1` in every run (`SCHED_DEBUG_REALLOC=0` leaves it out); a run that aborts fails, and the console names it: `graph reallocation at an unchanged size in <arm> <depth>` |
| [`bench/draft_count/run_draft_count.sh`](../../../bench/draft_count/run_draft_count.sh) | FN80, FN83 (C722–C729's read-once kernels) | the main model's count per load (`reallocs_<label>.tsv`, a `LOAD <label>: graph reallocations: …` line); `SCHED_DEBUG_REALLOC=1` sets the variable, off by default since every load holds the draft |
| [`2026-11-10_prob_verify_reply_length/run/run_pv_replies.sh`](../2026-11-10_prob_verify_reply_length/run/run_pv_replies.sh) | FN66 (C675), FN69 (C711) | the same |
| [`2026-11-11_c692_device_rows_ptx/run/run_c692.sh`](../2026-11-11_c692_device_rows_ptx/run/run_c692.sh) | C692 | the same |

Once C803 keeps the draft's graph from reallocating at an unchanged size, `SCHED_DEBUG_REALLOC=1` can become the
default of the three draft drivers.

## The checks

[run/test_c804.sh](run/test_c804.sh) on the quant machine's CPU at `nice -n 19`, against logs earlier rows wrote:
R60's two loads and R85's two (`-lv 5`, nine P106 cards), FN84's load, R123a's six loads, R122's fourteen and C692's
seven. Every driver ran end to end against stand-ins: a server that prints a recorded log, a `rig_server.sh` that puts
one in the load's folder, a `docker` that writes llama-perplexity's lines. No llama process started and no request was
sent. Each driver before C804 (from `v4` at `353f5b9`) ran beside it on the same stand-ins.

**47 of 47 checks pass** ([summary.tsv](summary.tsv)):

| Step | What was checked | Result |
|---|---|---|
| syntax | `bash -n` on the eight changed shell scripts and this one; the two Python files parse | 11 of 11 |
| tool | `scripts/realloc_counts.py`'s default table against the published ones | R60's [reallocs.tsv](../2026-11-12_r60_taper_v4/reallocs.tsv) (load `a`: 373 of 373 requests, 802 at the same size, 745 changed), FN84's [reallocs.tsv](../2026-11-12_fn84_graph_realloc/reallocs.tsv) (0 of 8) and R123a's [reallocs.tsv](../2026-11-12_r123a_prompt_ladder/reallocs.tsv) (0 of 876 in six loads), byte for byte; and the same as R60's script on all 18 logs |
| awk | each of the five drivers' `realloc_count`, with mawk 1.3.4 (Ubuntu 24.04's, as on the rigs) and busybox awk, against the tool on 18 logs | the same summary line and the same 2,060 requests (number, prompt tokens, reallocations of each kind) in all ten; C692's loads 0 of 7 each, the draft's own reallocations left out |
| marks | `--marks ladder` on R122's 14 loads and R123a's 6; FN84's load as a line and R60's load `a` as the line with a switch | R122 and R123a: every rung `none` ([marks.tsv](marks.tsv)); the switch: `reallocates` (373 of 373 against 0 of 8 and 0 of 8), and `as its line` and `reallocates more` on a log that holds both |
| logs | `r123a_logs.sh` and `r123b_logs.sh` before and after, on R123a's six loads | every file they wrote before the same, R123a's `loads.tsv` as published, plus `realloc_marks.tsv` (`-` for `base` and `base2`, `none` for the rest) |
| kld | `kld_depth.sh` with a stand-in `docker` | `-e GGML_SCHED_DEBUG_REALLOC=1` in both runs, none with `SCHED_DEBUG_REALLOC=0`; a run that prints the scheduler's abort is named (`graph reallocation at an unchanged size in sw 8192`) and fails the run |
| taper | `rig_taper.sh` before and after, loads A and B on a stand-in server that prints FN84's log and R60's load `a` | `run.log` as before plus `load A: graph reallocations: 0 of 8 requests (0 at the same size, 0 changed)` and `load B: graph reallocations: 373 of 373 requests (802 at the same size, 745 changed)`; every other file of a load the same, plus `reallocs.tsv` |
| loads | `rig_loads.sh` before and after through R122's `r122_loads.sh` (14 loads), on R122's own logs, then with loads `a` and `a-2` printing R60's load `a` | `status.txt` as before plus 15 and 16 lines; R122's logs: every load `none`; the swap: `a` and `a-2` `reallocates` against `base` and `base-2` (0 of 2 each), `b` and `b-2` `none` against them, and the line `graph reallocations: a reallocates (373 of 373 requests; its line without the switch, base,base-2: 0 of 2,0 of 2)` |
| deep | deep_lane's own selftest with the new `deep_table.py`; the selftest's rig run tabled again with recorded server logs (the `fast` arm's loads R60's load `a`, the rest FN84's) | the selftest passes; the mock's logs give "no `graph reallocs` line"; the table marks **`fast` reallocates where `base` does not** (`fast1`: 373 of 373 requests); the table as before plus that one line |

R122's ladder and R123a's: no load of either reallocates the main model's graph (R122: 2 requests a load, 28 in
all; R123a: 146 a load, 876 in all), so none of their switches adds reallocations. Both lines start from the line
without `LLAMA_SAMPLING_TOP_K_ROWS=1`.

## Files

| File | Content |
|---|---|
| [summary.tsv](summary.tsv) | every check: step, check, verdict, detail |
| [marks.tsv](marks.tsv) | `realloc_counts.py --marks` on R122's ladder, R123a's ladder and the switch case |
| [run/test_c804.sh](run/test_c804.sh) | the checks |

## Commands

On the quant machine, from a checkout of this repository, with the earlier rows' load folders (their raw output is
private; any `-lv` server log of a v4 build serves):

```
REPO=<this repository> OUT=<results> ORIG=<the six drivers from v4 353f5b9> R60=<R60's loads> R85=<R85's loads> \
  FN84=<FN84's load> R123A=<R123a's loads> R122=<R122's rig out/> C692=<C692's work folder> \
  nice -n 19 bash docs/data/2026-11-12_c804_realloc_in_checks/run/test_c804.sh
```

In a driver's own results:

```
python3 -I scripts/realloc_counts.py --line base=<load>/server.log sw=<load>/server.log
python3 -I scripts/realloc_counts.py --marks ladder base=… a=… b=… base-2=… a-2=… b-2=… > realloc_marks.tsv
```

Provenance: the server logs are the rows' own (R60, R85, FN84, R123a, R122 on rigs 1 and 2; C692 on the quant
machine), not published here; the tables in this folder are counts from them. Licence: the repository's.
