# Graph reallocations in the agent check's report (C804 step (a), Q1408, 2026-11-13)

What this changed: [bench/agent_check](../../../bench/agent_check/README.md)'s `report.py` now counts, in every server
log, the requests that reallocated the main model's compute graph. A switch that makes the engine rebuild its graph on
every request is visible in its own agent run. This is the last driver of
[C804](../2026-11-12_c804_realloc_in_checks/README.md), which put the same count into every other switch check's
driver. Registry row [Q1408](../../../quants/experiments.md).

Why: C691's switch (`LLAMA_SAMPLING_TOP_K_ROWS=1`) made every request reallocate the graph. A 3,313-token turn then
took 42.49 s against 21.14 s without the switch (one request each,
[FN84](../2026-11-12_fn84_graph_realloc/README.md)). That cost was v4's loss on long prompts in R90's agent check
([2026-11-11_r90_v4_vs_v3](../2026-11-11_r90_v4_vs_v3/README.md)) and in R60's prompt lane, and no report counted it.
C802 later removed it ([R127](../2026-11-12_c802_topk_fixed_graph/README.md#the-rig-check-r127)).

Nothing here is a measurement. The checks read server logs that R90 and R72 wrote on a P106 rig, run the agent check's
selftest against its mock server, and rebuild the reports on the quant machine's CPU.

## What the reports say now

`gonogo.md`: every server-log line ends with the count. These are R72's lines, rebuilt from its own folder:

```
- server log server-gains.log: backend sampling off 0, …, watchdog kill 0, graph reallocations 69 of 69 requests (329 same size, 137 changed)
- server log server-v4c2.log: backend sampling off 0, …, watchdog kill 0, graph reallocations 0 of 91 requests (0 same size, 0 changed)
```

`W of N requests`: the requests with at least one reallocation, of all the log's requests. `S same size, D changed`:
the reallocations at an unchanged graph size and at a changed one. The count comes from each request's
`graph reallocs = S / D` line, the main model's running count since the load
([scripts/realloc_counts.py](../../../scripts/realloc_counts.py); the MTP draft's own graph has its own count, which
this leaves out). A log without that line (stock llama.cpp, the mock server) says
`graph reallocations not counted (no `graph reallocs` line)`.

`compare.md`: with the rig's server logs (`--server-log rig/server-*.log`, which `client_report.sh` now passes), a
block after the per-arm table gives each arm's count. It marks an arm `reallocates` where another arm's logs have none.
R72's four arms:

```
Graph reallocations of the target in each arm's server logs (C804): an arm is marked `reallocates` where another arm's logs have none.

- gains: graph reallocations 69 of 69 requests (329 same size, 137 changed): **reallocates** where v4c2, v4c22, v4c2g do not
- v4c2: graph reallocations 0 of 91 requests (0 same size, 0 changed)
- v4c22: graph reallocations 0 of 65 requests (0 same size, 0 changed)
- v4c2g: graph reallocations 0 of 65 requests (0 same size, 0 changed)
```

A server log belongs to the arm its name gives (`server-<arm>.log`, as `rig_server.sh` writes it) or to the one named
in an `ARM=PATH` argument. Every other line of both reports is unchanged, and `table.md`, `res/attempts.csv` and
`res/turns.csv` are byte for byte the same.

## The counts of R90's and R72's runs

[logs.tsv](logs.tsv): each agent server log, its build and whether C691's switch was set in its command:

| Run | Arm | Build | `LLAMA_SAMPLING_TOP_K_ROWS=1` | Requests | With a reallocation | Same size | Changed |
|---|---|---|---|---|---|---|---|
| R90 | `v3` | v3.0, `1551f5fe1` | no | 65 | 0 | 0 | 0 |
| R90 | `gains` | v4 day build, `18d0fa309` | yes | 70 | 70 | 352 | 139 |
| R72a | `gains` | v4 kernel merge, `d097d6d88` | yes | 69 | 69 | 329 | 137 |
| R72b | `v4c2` | C802 build, `8a55934f3` | yes | 91 | 0 | 0 | 0 |
| R72c | `v4c2g` | C802 build, `8a55934f3` | yes | 65 | 0 | 0 | 0 |
| R72d | `v4c22` | C802 build, `8a55934f3` | yes | 65 | 0 | 0 | 0 |

They agree with FN84 and R127. With C691's switch, every request of the `gains` line reallocates: on the day build
(R90) and on the kernel merge (R72a) alike, as in R60's prompt lane (373 of 373). On the C802 build, the same switch
reallocates in no request of the three runs, as in R127 (0 of 8). v3.0 reallocates in none.
[agent_reallocs.tsv](agent_reallocs.tsv) splits each log by prompt size (R90's buckets).

Rebuilt with this change, R90's own `compare.md` marks `gains` (`70 of 70 requests`) `reallocates` where `v3` (`0 of
65`) does not: the run where the loss first showed would have named its cause.

## Other changes

- **`realloc_counts.py` counts each load from its own start.** An agent-check arm that is resumed appends its new
  load to the same `server-<arm>.log` (`rig_server.sh` and the copy back both append). Before, the second load's first
  request was counted against the first load's last count, and the difference was negative. Now a server start
  (llama-server's `llama_server: initializing`, or `rig_server.sh`'s `=== ` line), or a count lower than the previous
  one, starts the count from zero. On every log C804 checked the output is unchanged.
- **The prep copies `realloc_counts.py` to the client.** `release_check.sh`, `race_rigs.sh` and
  [release_check_s12.sh](../2026-11-03_flash_next_mtp/run/release_check_s12.sh) (R72's driver) copy the bench files to
  the client, where `client_report.sh` runs in a container that mounts only that folder. The tar adds
  `scripts/realloc_counts.py` beside `task_results.py`, where `report.py` finds it. A client folder copied before this
  change has no counter. Its reports are still written, and the count lines say
  `not counted (scripts/realloc_counts.py not found)`.

## The checks

[run/test_c804a.sh](run/test_c804a.sh) on the quant machine's CPU at `nice -n 19`, with the repository before this
change (`v4` at `5e46c8f`) beside it. **44 of 44 checks pass** ([summary.tsv](summary.tsv)):

| Step | What was checked | Result |
|---|---|---|
| syntax | `bash -n` on the six changed shell scripts; the two Python files parse | 8 of 8 |
| selftest | `bench/agent_check/selftest.sh` with the docker sandbox: 12 mock tasks, resume, turn cap, replay, the token mark, and three new checks on stand-in server logs. One arm's log holds two loads, the second resumed; the other arm's log has no reallocation; a third log has no count line | the selftest passes, 34 of 34 (31 before this change, 3 new). gonogo: `2 of 3 requests (5 same size, 2 changed)` (counted from the start of each load), `0 of 2 requests`, `not counted`. compare: `mock` marked `reallocates` where `mock-cap` does not. Every other line of compare is the same as without the logs |
| reports | `client_report.sh` before and after, each in a copy of R90's and R72's client folders | before, `compare.md` and `gonogo.md` are the runs' own. After, `compare.md` has 0 lines changed and the block added (5 lines for R90's 2 arms, 7 for R72's 4), `gonogo.md` has its 2 and 4 server-log lines extended, and `table.md`, `attempts.csv` and `turns.csv` are identical |
| client | the prep's tar into a folder, and `report.py` run from it on R72's folder, with and without `realloc_counts.py` | the counter lands beside `task_results.py`, the copy writes the checkout's reports, and a copy without it writes them with `not counted` |
| counter | `realloc_counts.py` before and after on the 18 logs of C804's checks (R60, R85, FN84, R123a's six rungs, C692's seven), in every mode; the published tables; R72's `gains` log twice in one file | the same output in all four modes (2,060 requests). R60's, FN84's and R123a's `reallocs.tsv` byte for byte. Two loads: `138 of 138 requests (658 at the same size, 274 changed)`, twice one load's `69 of 69 (329, 137)`; the tool before this change gave `137 of 138` with a negative count for the second load's first request |

Two details of the checks. compare orders tasks that started in the same second by Python's set order, which changes
from one process to the next. The selftest's mock starts tasks within one second, so its comparison sorts the lines;
the real runs have no such ties. gonogo lists the server logs in the order of the shell's glob, so the test runs with
`LC_ALL=C`, the client container's order.

## Files

| File | Content |
|---|---|
| [summary.tsv](summary.tsv) | every check: step, check, verdict, detail |
| [logs.tsv](logs.tsv) | each agent server log of R90 and R72: build, C691's switch, requests, with a reallocation, same size, changed |
| [agent_reallocs.tsv](agent_reallocs.tsv) | `realloc_counts.py`'s table of the same six logs, by prompt size |
| [r90_compare.md](r90_compare.md), [r90_gonogo.md](r90_gonogo.md) | R90's reports rebuilt with this change (arms `v3`, `gains`) |
| [r72_compare.md](r72_compare.md), [r72_gonogo.md](r72_gonogo.md) | R72's reports rebuilt with this change (arms `gains`, `v4c2`, `v4c22`, `v4c2g`) |
| [run/test_c804a.sh](run/test_c804a.sh) | the checks |

## Commands

On the quant machine, from a checkout of this repository and one at `v4` `5e46c8f`, with the client folders of R90
and R72 and the folders C804's checks read (their raw output is private; any agent check's collected folder serves):

```
NEW=<this repository> OLD=<the repository at 5e46c8f> OUT=<results> R90=<R90's client folder> R72=<R72's client folder> \
  R60=<R60's loads> R85=<R85's loads> FN84=<FN84's load> R123A=<R123a's loads> C692=<C692's work folder> \
  nice -n 19 ionice -c3 bash docs/data/2026-11-13_c804a_agent_realloc/run/test_c804a.sh
```

In an agent check's collected folder:

```
python3 -I bench/agent_check/report.py gonogo res --log proxy.jsonl --server-log rig/server-*.log
python3 -I bench/agent_check/report.py compare res --log proxy.jsonl --arms gains,v4c2 --server-log rig/server-*.log
```

Provenance: the server logs and client folders are R90's (rig 1, 2026-11-11) and R72's (rig 1, 2026-11-12), not
published here; the tables in this folder are counts from them and the reports rebuilt from them. Licence: the
repository's.
