# A lighter kernel profiler for Pascal: per layer, sampled and per graph (2026-11-04)

The llama.cpp fork's kernel profiler, `GGML_CUDA_PROFILE`, records a CUDA event between every two launch groups and
synchronizes after every evaluation. On a P106 those events cost est. 7–9 µs a group, 17–23 ms of R13's 136 ms token,
so every small op profiles at about four times its real time ([megakernel census](../2026-11-04_megakernel_census/README.md)).
That inflation misled three estimates already: Q88's persistent kernels, the megakernel
([megakernel.md](../../research/notes/megakernel.md)) and the hops between cards
([card_hops_x1.md](../../research/notes/card_hops_x1.md): 1.7–2.2 ms a hop profiled, est. 0.5–0.9 ms unprofiled).
This folder holds three lighter modes in the fork, the summary script that reads them, the CPU checks, and the rig run
that checks them against the unprofiled token. Registry row Q681 in [quants/experiments.md](../../../quants/experiments.md).

## Result (tonight: code and CPU checks; the rig run is next)

- **Today's profiler keeps CUDA graphs on.** Under stream capture its events become external event-record nodes of
  the captured graphs (`cudaEventRecordWithFlags(..., cudaEventRecordExternal)`), so replays are timed as they run;
  nothing in it turns graphs off. What it adds: one event node between every two launch groups (~2,520 a token on
  Flash-Next), a `cudaStreamSynchronize` after each card's evaluation, and ~280 CSV lines written per evaluation on the
  host before the next card's work is queued.
- **What that costs on the wall**: in R13's profiled run a token took **156 ms** (median interval of card 0's decode
  evaluations, host clock, 93 intervals), against 112–120 ms for unprofiled tokens at comparable context (the census);
  R28's 140 ms. The summed spans were 136 ms (R13) and 121 ms (R28). So the default mode inflates both the small ops
  (the event nodes, on the GPU) and the hops (the synchronize and the CSV writing, on the host).
- **Three new modes** in fork branch `pascal-p106-v3-b12076-lightprof` (one commit on `-ladder2`), chosen with
  `GGML_CUDA_PROFILE_MODE`; the default stays `group`, byte for byte. All three keep CUDA graphs on and never
  synchronize: an evaluation's events are read when the same graph is evaluated again, a token later in decoding.

| Mode | Events per card and evaluation | Est. events per Flash-Next token (9 cards) | Reports |
|---|---|---:|---|
| `group` (default, unchanged) | one between every two launch groups, a sync after each evaluation | ~2,520 + 9 syncs | every launch group |
| `layer` | one where the layer of the groups changes, a pair around the evaluation | ~2 per layer + ~4 per card | each layer's time per card |
| `sample=<N>` | a pair around every N-th launch group, rotating per evaluation; a pair around the evaluation | ~2 × 2,520 / N + 18 (630 at N = 8) | every launch group, each once in N tokens |
| `graph` | a pair around the whole evaluation | 18 | each card's evaluation |

- **Sample mode under CUDA graphs** keeps one set of captured executables per phase (each phase's events are nodes of
  its own graphs), so it holds N times the graph executables and captures each phase once after every warm-up; a
  phase's set captured before the graph's latest warm-up has stale pointers and is captured again before it runs.
- **`scripts/prof_summary.py`** reads all four formats. It subtracts a per-span event cost (`--event-us`, from C140's
  `graph+events` minus `graph` on `bench/megakernel_p106`), keeps replays only by default (a warm-up's eager and
  capture runs go at another speed), and reports per launch group, op and weight type (group, sample), per layer
  (layer) or per card (graph), each with its share of the token; the sum of the timed spans per token; the whole
  evaluations' GPU time; the profiled run's wall time per token; and, given the unprofiled time per token
  (`--wall-ms`), the sum against it, whose difference is what no card's timeline holds (hops, host work). On today's
  format it prints the old report first, unchanged (identical on R13's and R28's traces).
- **Not measured yet**: nothing ran on a GPU. Whether the light modes sum to the unprofiled token is tomorrow's rig run
  (below); the event cost to subtract is C140's.

## How the modes work (fork `ggml/src/ggml-cuda/ggml-cuda.cu`, `profile-light.h`)

- **Where the events go.** `layer` and `sample` record their events inside the evaluation loop, before and after the
  launch groups they time, so under capture they become nodes of the graphs and every replay records them at the same
  places. The pair around the evaluation is recorded before the capture begins and after the last chunk is launched,
  so it is never a graph node.
- **A layer** is the number at the end of a node name's first word (llama.cpp names a layer's tensors `<name>-<layer>`),
  else the `blk.<n>.` of a weight it reads; `result*` names are `output`, `inp_*` names `input`, and a group whose names
  say nothing stays in the current segment.
- **Sampling.** Launch group k of an evaluation in phase p is timed when k mod N = p, and the phase is the
  evaluation's index among the evaluations of the same graph on that card, mod N: N evaluations in a row time every
  group once.
- **Reading without a sync.** At the start of an evaluation the profiler writes the previous evaluation of the same graph
  and phase, if its last event has completed (in decoding it completed a token ago); otherwise it writes one
  `(not ready)` line and nothing waits. A replay of graphs captured before the loop last ran eagerly is written as
  `(unmatched)`. The last evaluation of each graph is never written. The file is block-buffered and flushed once a
  second at most.
- **The file** (light modes): `device,eval,host_ms,group,label,us,mode,graph,phase,how,groups,eval_groups,eval_us`,
  one line per timed span: `host_ms` when the evaluation was enqueued; `group` the span's first launch group; `label`
  the group's label (sample), `layer <n>`/`input`/`output`/`other` (layer) or `graph`; `us` the span's GPU time; `graph`
  the graph's number on its card; `phase`; `how` = `replay` (captured graphs launched), `capture` or `eager`; `groups`
  the span's launch groups, `eval_groups` the evaluation's; `eval_us` the whole evaluation's GPU time. The `how`
  column also says, per card and token, whether decoding replays its graphs (C141's question).

## Checks (quant machine, CPU only, no GPU)

| Check | Result |
|---|---|
| The rig build of the branch (`TESTS=1`, CUDA 12.9, sm_61, every CUDA file recompiled since `common.cuh` changed) | built; md5 and details under "The build" |
| `tests/test-cuda-profile-light.cpp` (g++ 13.3, `-Wall -Wextra -Wshadow`, no warning): mode strings, layer names, layer segments (2,000 random sequences tile their groups, one event per boundary), sampling covers every group once in N, and a model of `ggml_backend_cuda_graph_compute`'s warm-up and capture logic with the phase sets (300 random runs with changing pointers: no replay of a stale set, nothing lost) | 49,267 checks, 0 failed |
| `prof_summary.py --self-test` (Python 3.12): synthetic traces in all four formats from a known three-card token, a warm-up's slow evaluations, an unread evaluation | 50 checks, 0 failed |
| `prof_summary.py` on R13's and R28's real traces (today's format), against the previous version | the previous report's 67 lines identical; new lines appended |
| `run/dryrun.sh`: `run/lightprof_rig.sh` unchanged against `run/fake_llama_server.py` (stand-ins for the server, curl and nvidia-smi), arms off, graph, layer, sample=8, group, off | pass |

Today's two traces through the new summary (raw, and with an illustrative 7 µs per span; C140 gives the real value):
[legacy_traces.tsv](legacy_traces.tsv).

| Trace | Decode steps | Spans summed, raw (ms) | With 7 µs per span (ms) | Profiled wall per token (ms) |
|---|---:|---:|---:|---:|
| R13 (v3, Swift Q2_0) | 96 | 135.96 | 118.27 | 156.44 |
| R28 (v3, hyper-connections in Q8_0) | 96 | 120.70 | 102.97 | 139.87 |

## The build

Fork branch `pascal-p106-v3-b12076-lightprof`, commit `973bcf09d` (one on `pascal-p106-v3-b12076-ladder2`
`a330672c1`). Rig tarball `llama.cpp-20261104-973bcf09d-pascal-p106-v3-b12076-lightprof.tar.gz`, 346,442,186 bytes,
md5 `c9342dfd221da9007f900994deb4ff24`, built with
`TESTS=1 scripts/build_llamacpp_docker.sh fork/pascal-p106-v3-b12076-lightprof` (image
`nvidia/cuda:12.9.2-devel-ubuntu24.04`): no AVX instruction, highest glibc symbol 2.38, 160 sm_61 images in
`llama-server` and no other architecture; `llama-server`, `llama-bench` and `test-backend-ops` carry the new mode
strings, and the compiler reports no warning on the changed files. With `GGML_CUDA_PROFILE` unset the new code does
nothing.

## Next: the rig run (C142r, ~35 min, the rig idle)

One short Flash-Next line per mode, R13's v3 settings (`-c 65536 -b 2048 -ub 512 -ts 6,5,5,6,5,5,6,5,6 -fit off -lm dio
-fa on`, one slot), on the `-lightprof` build: arms `off` (no profiler, the reference), `graph`, `layer`, `sample=8`,
`group`. Each loads the model (~5.5 min with direct I/O) and sends the same greedy requests (a 16-token warm-up, then
3 × 64 tokens, end-of-text ignored, no prompt cache), so every arm decodes the same tokens. Then each profile goes
through `prof_summary.py` against the off arm's ms per token.

```bash
# bundle: the run scripts and the two tools from scripts/
B=<a folder on a disk, not tmpfs: the group arm writes ~60 MB>; ssh $RIG "mkdir -p $B"
tar cf - -C docs/data/2026-11-04_light_profiler/run lightprof_rig.sh lightprof_requests.py \
  -C ../../../../scripts prof_summary.py ram_watchdog.sh | ssh $RIG "tar xf - -C $B"
# the -lightprof tarball unpacked on the rig (its folder's bin/ is BIN), MODEL the model's first shard
ssh $RIG "cd $B && BIN=<lightprof build>/bin MODEL=<first shard> OUT=$B/out EVENT_US=0 \
  setsid nohup bash lightprof_rig.sh > $B/console.log 2>&1 < /dev/null &"
ssh $RIG "test -e $B/out/DONE && cat $B/out/check.txt"
```

Read `out/check.txt` and `out/check.tsv`:
- **The check (C142):** the `layer` and `graph` sums within a few percent (`TOL_PCT`, 5) of the unprofiled token time.
  What they miss is the hops and the host's work between cards, so a sum a little below the unprofiled time is the
  expected result; the difference is the unprofiled hop budget per token.
- **Each mode's own cost** on the token: its server ms per token against the off arm's (`server_vs_off_pct`); the
  `group` arm shows today's cost.
- `sample8`: the coverage (1.000 = every launch group timed) and the per-op table in `summary-sample8.txt`, the first
  per-op profile of Flash-Next without the inflation (after `EVENT_US`).
- The `how` counts in each summary's first line: replays against captures and eager evaluations per card (C141).
- `same_output_as_off`: the profiler does not change the tokens.

When C140 has measured the event cost, rerun the summaries with it (no rig needed):
`python3 scripts/prof_summary.py out/prof-<arm>.csv --event-us <C> --wall-ms <off ms per token> --tol-pct 5`.

## Files

| File | What |
|---|---|
| [legacy_traces.tsv](legacy_traces.tsv) | R13's and R28's traces through the new summary: raw and at 7 µs per span |
| [cpu_checks.tsv](cpu_checks.tsv) | the CPU checks above with their counts |
| [run/lightprof_rig.sh](run/lightprof_rig.sh) | the rig run: one server line per mode, requests, summaries, the check table |
| [run/lightprof_requests.py](run/lightprof_requests.py) | the requests of one line (standard library) |
| [run/dryrun.sh](run/dryrun.sh), [run/fake_llama_server.py](run/fake_llama_server.py) | the rig script's dry run on any CPU |
| [../../../scripts/prof_summary.py](../../../scripts/prof_summary.py) | the summary, all four formats, with `--self-test` |

The fork's change: `ggml/src/ggml-cuda/ggml-cuda.cu` (the modes), `ggml/src/ggml-cuda/profile-light.h` (the parts that
need no GPU), `ggml/src/ggml-cuda/common.cuh` (the phase sets of a CUDA graph), `tests/test-cuda-profile-light.cpp`.

Commands of the checks: `g++ -std=c++17 -O1 -Wall -Wextra -Wshadow tests/test-cuda-profile-light.cpp && ./a.out` in
the fork's tree; `python3 scripts/prof_summary.py --self-test`; `bash docs/data/2026-11-04_light_profiler/run/dryrun.sh`
from this repository's root (both in `python:3.12-slim-bookworm`).

Provenance and licence: code and tables by this project (MIT, this repository; the fork's code under llama.cpp's MIT
licence); the two traces are R13's and R28's runs of the same project ([R13](../2026-11-03_flash_next_baseline/README.md)).
The request prompt is written for this run.
