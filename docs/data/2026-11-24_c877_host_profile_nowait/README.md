# The host profile without waiting (C877, Q1583; 2026-11-24)

What this is: the change that registry row Q1583 in [quants/experiments.md](../../../quants/experiments.md) asked for,
and its correctness check. `GGML_CUDA_PROFILE_HOST=<file>` writes one line per graph evaluation and card: the host's
entry to and exit from the card's evaluation and the card's GPU time for it. It is the profile behind the step maps
([R176](../2026-11-17_r176_flint_step_map/README.md)). Until now it waited, at each entry, for the card's previous
evaluation to end before it wrote that one. Flint's preset runs its cards as a pipeline (`--pipe-depth 4`), which queues
a card's next evaluation before its last one ends, so the profile held each card to one evaluation at a time: on nine
P106s Flint took 34.98 and 34.90 ms a token under it against 18.09 without it, and no pipelined step could be profiled.
The fix is a new switch, `GGML_CUDA_PROFILE_HOST_NOWAIT=1`, on the fork branch `pascal-p106-v5-host-profile-nowait`
(`7d182f9b0`, one commit on `pascal-p106-v5` at `8a9629942`). It was built and checked on a build machine's three
RTX 3090s, which run the rig build's compute 6.1 code through the driver's compiler. Correctness and counts only:
nothing here is timed. Its use on a P106 is row Q1736.

## Result in plain words

- **Without waiting, the pipeline keeps running under the profile.** With Flint Pascal Q8 GGUF's v5 preset on the
  three cards, the old way found the card's previous evaluation still running at almost every entry on the second and
  third cards (4,447 and 4,593 of 4,925 to 4,932 entries on the pipe's requests) and waited there each time, so never
  more than one evaluation per card was in flight. With the switch the host never waited, and up to 6, 7 and 8 earlier
  evaluations were still running on the three cards when it entered the next one, most often 4 on the second and
  third cards.
- **Every evaluation gets its line.** Each card's lines carry every evaluation id from the first to the last, once,
  and the count line the profile now writes when the server exits says all were written (5,319 of 5,319 per card
  without waiting, 5,326 of 5,326 waiting). The profile writes the same evaluations for each request whichever way it
  runs, and the same as the old build's: where two loads gave the same reply, their evaluations per request and card
  are the same number.
- **Profiling changes no token.** The four loads (profile off, waiting, without waiting, and the old build with its
  profile) gave the same replies to every request but one sampled request, which gives one of two known replies on
  every build ([C962](../2026-11-23_c962_sampled_reply_gpu/README.md); [C985](../2026-11-24_c985_topk_order/README.md)
  fixes it), and the request that continues its conversation. The pipe's own counts (guesses, right, wrong) follow the
  replies the same way.
- **With the profile unset the build is the base.** The card code is the base's, kernel for kernel; the logits of 148
  random models are the same bits; ctest is the same test by test; no warning is new.
- **What it does not show:** whether the GPU times are right on a P106, and what the profile now costs there. That is
  rig row Q1736, with the step map of a pipelined step it was made for.

## The change

All in `ggml/src/ggml-cuda/ggml-cuda.cu`. Each card keeps the evaluations it has entered but not yet written in a queue,
each with its pair of CUDA events, and reuses event pairs once written.

- **`GGML_CUDA_PROFILE_HOST_NOWAIT=1`:** at each entry the host writes, in order, the card's evaluations whose end
  event has passed (`cudaEventQuery`) and leaves the rest for a later entry. Nothing waits. Past 1,024 unwritten
  evaluations on one card the host waits for the oldest, so a card that never finishes cannot grow the queue without
  bound (counted; it never happened here).
- **Unset (the default), the profile waits as before:** at each entry it waits for the card's earlier evaluations and
  writes them.
- **A new last column, `queued`:** how many of this card's earlier evaluations were still running on the card when the
  host entered this one. Waiting, it is 0 or 1, and 1 means the host waited there. The first five columns are as
  before, and every script that reads the file finds its columns by name.
- **Nothing is lost at the end.** When a card's backend is freed (the server exits cleanly), the evaluations still
  unwritten are waited for and written, and one log line per card gives the counts: evaluations, written, entries with
  earlier ones still running, the most at once, entries at which the host waited. The old profile never wrote each
  card's last evaluation.
- The load line names the way: `..., without waiting (GGML_CUDA_PROFILE_HOST_NOWAIT=1)` or `..., waiting for each
  card's earlier evaluations`.

The lines are in the order they are written, which lags the evaluations by as many as are queued; sort by
`host_enter_ms` for a timeline. Both events sit on the card's stream around the same work as before, so what the GPU
time covers is unchanged; its values are not reported here.

## The checks

All on a build machine (Ryzen 9 5900X, three RTX 3090s, driver 595, CUDA 12.9 in the containers) with the rig
build, whose card code is compute 6.1 SASS and PTX: the 3090s run the PTX through the driver's compiler. Base
`8a9629942`, head `7d182f9b0`.

### With the profile unset: the base's build

[checks.tsv](checks.tsv) has these rows as a file.

| Check | Base `8a9629942` | Head `7d182f9b0` | Verdict |
|---|---|---|---|
| Rig build: warning lines | 32 | 32 | none new |
| CPU build of every target: warning lines | 2 | the same 2 (the CMake line names the work folder) | none new |
| Card code: SASS of `llama-server`, one SHA-1 per kernel copy | 9,518 | 9,518 | every copy identical |
| Card code: PTX of `llama-server`, one SHA-1 per function | 9,518 | 9,518 | every function identical, no module-level line differs |
| `test-llama-archs -b CPU -s 1` of the rig builds | writes 148 logits files, 153 of 153 pass | 153 of 153 pass, plain and comparing | 148 of 148 files bit for bit |
| ctest, 75 tests, no network (CPU build) | 71 pass; `test-kv-cache`, `test-jinja`, `test-jinja-py`, `test-kv-cache-scales` fail | the same | the same, test by test |

The base's warning lines, CPU build log and ctest table come from the
[C880 landing](../2026-11-24_c880_land/README.md), which built `8a9629942` with the same script and settings; the
logits files were written by its rig build in this run. ctest leaves out the seven tests every run of `cpu_tests.sh`
leaves out (C880's README lists them).

### Flint's v5 preset with the profile, on three cards

Flint Pascal Q8 GGUF with its v5 preset ([release/flint-v5/settings.env](../../../release/flint-v5/settings.env),
`--pipe-depth 4` kept; the log shows pipeline parallelism on and 10 pipe slots), the split changed to the three cards
(`-ts 1,1,1`; the file is 40.4 GB). Four loads, one at a time, each running [C950](../2026-11-21_c950_flint_slot_plan_gpu/README.md)'s
session twice: 14 requests a round, six plain-text replies that the pipe computes (R1, R8, R12 to R15) and eight tool
calls under a JSON schema, which decode one token a step with the pipe off.

| Load | Build | Profile |
|---|---|---|
| off | head | unset |
| wait | head | `GGML_CUDA_PROFILE_HOST=<file>` (waits: the default) |
| nowait | head | `GGML_CUDA_PROFILE_HOST=<file> GGML_CUDA_PROFILE_HOST_NOWAIT=1` |
| basewait | base | `GGML_CUDA_PROFILE_HOST=<file>` (the profile as it was) |

[run/session_marks.py](run/session_marks.py) runs C950's session unchanged and notes the steady clock's reading when
each request is sent and answered; the profile's log line gives that clock's reading at the file's 0 ms, so every
profile line is given to its request. The marks only place the lines.

**The pipe's work in flight** ([queued.tsv](queued.tsv), the pipe's requests, both rounds): at each entry, how many of
the card's earlier evaluations were still running.

| Load | Card | Entries | With earlier ones running | Most at once | 0 | 1 | 2 | 3 | 4 | 5 or more | Entries the host waited at |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| wait | CUDA0 | 4,932 | 69 | 1 | 4,863 | 69 | 0 | 0 | 0 | 0 | 69 |
| wait | CUDA1 | 4,932 | 4,447 | 1 | 485 | 4,447 | 0 | 0 | 0 | 0 | 4,447 |
| wait | CUDA2 | 4,932 | 4,593 | 1 | 339 | 4,593 | 0 | 0 | 0 | 0 | 4,593 |
| nowait | CUDA0 | 4,925 | 3,398 | 6 | 1,527 | 1,038 | 1,392 | 361 | 286 | 321 | 0 |
| nowait | CUDA1 | 4,925 | 4,575 | 7 | 350 | 217 | 97 | 471 | 3,109 | 681 | 0 |
| nowait | CUDA2 | 4,925 | 4,593 | 8 | 332 | 104 | 198 | 90 | 3,296 | 905 | 0 |

CUDA2 holds the output layer, so its count is the pipe's own: how many of the pipe's earlier tokens were still on the
cards when the next one reached the last card. Without waiting that was 4 at 3,296 entries and 5 to 8 at 905; the pipe
keeps a sampled token and up to 4 guesses in flight, and a dropped guess's evaluations still run while the next tokens
queue behind them. Waiting, the host found the previous token still running at the same 4,593 entries and stopped
until it ended. The tool calls (392 entries per card, 391 in the old build) never had an earlier evaluation running in
either way: each step waits for its token anyway, as R176 found on its speculative arms.

**Every evaluation written** ([profile.tsv](profile.tsv)):

| Load | Lines per card | Evaluation ids | The count line at exit (each card) |
|---|---|---|---|
| wait | 5,326 | 0 to 5,325, none missing or repeated | 5,326 evaluations, 5,326 written; the host waited at 69, 4,447 and 4,593 entries |
| nowait | 5,319 | 0 to 5,318, none missing or repeated | 5,319 evaluations, 5,319 written; the host waited at 0 |
| basewait | 5,318 | 0 to 5,317, none missing or repeated | (none in the old build; it never writes a card's last evaluation) |

**The same evaluations per request** ([evals.tsv](evals.tsv)): 90 rows (the load and 28 requests, three cards each).
The three profiled loads wrote the same number of evaluations in 75. The other 15 are R14 and R15 of both rounds, where
the loads' replies differ, and R11 of round 2, the last request, where the old build lacks its last evaluation per card
(26 against 27). R14's evaluations follow its reply: 297 per card with one list, 298 with the other, in every load; R15
after them 441 and 433.

**Tokens and the pipe's counts** ([tokens.tsv](tokens.tsv), [pipe.tsv](pipe.tsv)): 24 of 28 requests gave one token
list in all four loads. R14 (temperature 0.7, top-k 8, seed 7) gave C950's two lists, `fe36bf356f32` and
`de544dad17e0`, in each load: off and nowait one in each round, wait the first in both, basewait one in each round. R15
continues R14's conversation, so its two lists pair with them. The pipe's counts are the same in every load on every
other request (R1: 395 guesses, 267 right, 32 wrong; R8: 194, 70, 31; R12: 521, 325, 49; R13: 427, 367, 15) and
follow R14's list (281, 241, 10 or 284, 243, 11).

**A first pass** ([pass1/](pass1/)): the same four loads with one round each, 04:45 to 04:55. They gave the same
pattern (waiting at most 1 and the host waiting at 31, 2,232 and 2,290 of 2,459 entries on the pipe's requests; without
waiting at most 6, 7 and 8, and no wait), but their servers were killed rather than stopped: the container's shell,
its first process, did not pass the stop signal on, so the count lines were never written and each file lost its last
buffered lines. The driver now sends the signal to the server itself and waits for it to exit; the second pass, above,
is the result.

## Files

| File | What |
|---|---|
| [checks.tsv](checks.tsv) | every check with the profile unset: base, head, verdict |
| [queued.tsv](queued.tsv) | per profiled load, card and kind of request (the pipe's, the tool calls, the load): entries, entries with an earlier evaluation of the card still running, the most at once, the entries by that number |
| [profile.tsv](profile.tsv) | per profiled load and card: lines, evaluation ids (first, last, missing, repeated), whether the `queued` column is there, the count line at exit |
| [evals.tsv](evals.tsv) | per request and card: the evaluations each profiled load wrote for it, and whether they agree |
| [tokens.tsv](tokens.tsv) | per request: tokens and the md5 of the token ids in each load, and whether all agree |
| [pipe.tsv](pipe.tsv) | per request: the server's pipelined line (guesses, right, wrong) in each load, and whether all agree |
| [pass1/](pass1/) | the same five tables from the first pass |
| [run/run.sh](run/run.sh) | the driver: the steps below |
| [run/session_marks.py](run/session_marks.py) | C950's session with each request's marks on the profile's clock |
| [run/c877_tables.py](run/c877_tables.py) | the tables from the session's output |

The profile files themselves (host and GPU times from a build machine, not the rig) are not published.

## Commands

On a build machine with Docker and NVIDIA cards, from a copy of this repository at `REPO` (with this folder, C950's
`run/session.py`, `release/flint-v5/settings.env`, `scripts/build_llamacpp_docker.sh`,
[cpu_tests.sh](../2026-11-05_c565_ladder4fix/run/cpu_tests.sh),
[sass_compare.py](../2026-11-17_c861a_fp32_scale/run/sass_compare.py) and
[ptx_functions.py](../2026-11-24_c985_topk_order/run/ptx_functions.py)), detached:

```bash
W=<work dir> SEED=<llama.cpp clone holding 8a9629942> REPO=<that copy> \
  BASE_RIG=<8a9629942's rig build folder> BASE_RIG_LOG=<its build log> \
  BASE_CTEST=<8a9629942's ctest table> BASE_CPU_LOG=<its CPU build log> \
  FLINT=<Flint Pascal Q8 GGUF> MOUNTS='<model folders>' \
  RIG_CCACHE=<ccache dir> CPU_CCACHE=<ccache dir> PREFLIGHT=<cuda check> LOCK=<flock file> JOBS=8 \
  setsid nohup bash "$REPO/docs/data/2026-11-24_c877_host_profile_nowait/run/run.sh" > run.out 2>&1 < /dev/null &
```

Then the second pass, the first pass's `session/` moved aside: the same with `STEPS='session tables' ROUNDS=2`.

Steps (`STEPS`, default all, in this order): `src` (a clone, the branch fetched and checked against its commit),
`rig` (the rig tarball with `TESTS=1` and the extra targets `test-llama-archs test-cuda-keepalive`, its md5 and
warnings), `session` (the four loads above on three cards), `archs` (the rig builds' `test-llama-archs` logits),
`sass` (the card code), `cpu` (a CPU build of every target and ctest), `tables`. Images:
`nvidia/cuda:12.9.2-devel-ubuntu24.04` for the rig build, the loads, the logits and `cuobjdump`; `ubuntu:24.04` with
`git cmake build-essential ccache python3 python3-jinja2` for the CPU step. The first run took 19 minutes (04:44 to
05:03: the rig build 1 minute 15 seconds from a warm compiler cache, the loads 10, the card code 6), the second pass 3.

Rig tarball: `llama.cpp-20261124-7d182f9b0-pascal-p106-v5-host-profile-nowait.tar.gz`, 842,662,793 bytes, md5
`fb03c52763693e728786ed8a9825da49`; its `llama-server` md5 `b27739369730d4d3fa56269434ea2d44`.

Provenance: llama.cpp (MIT) with the fork's commits; Flint Pascal Q8 GGUF as the release's model table names it;
C950's session prompts are this repository's own text.
