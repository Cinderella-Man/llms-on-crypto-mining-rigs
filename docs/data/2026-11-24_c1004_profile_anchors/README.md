# Each evaluation's start and end on the card, on the host profile's clock (C1004, Q1737; 2026-11-24)

What this is: the change that registry row Q1737 in [quants/experiments.md](../../../quants/experiments.md) asked for,
and its correctness check. `GGML_CUDA_PROFILE_HOST=<file>` writes one line per graph evaluation and card: the host's
entry to and exit from the card's evaluation and the card's GPU time for it. Since
[C877](../2026-11-24_c877_host_profile_nowait/README.md) it can run without waiting for the cards
(`GGML_CUDA_PROFILE_HOST_NOWAIT=1`), so a pipelined preset keeps its pipeline under it. It still did not say *when* a
card ran each evaluation, so a pipelined step's map could not show where a card sits idle between evaluations. The new
switch `GGML_CUDA_PROFILE_HOST_ANCHORS` adds two columns, `gpu_start_ms` and `gpu_end_ms`: the evaluation's start and
end on the card, on the file's own clock. It is on by default with `GGML_CUDA_PROFILE_HOST_NOWAIT=1`. It lives on the
fork branch `pascal-p106-v5-profile-anchors` (`cad303eb9`, two commits on `pascal-p106-v5` at `af00bb13b`) and was
built and checked on a build machine's RTX 3090s, which run the rig build's compute 6.1 code through the driver's
compiler. Correctness only: nothing here is timed. Its use on a P106 is row Q1736.

## Result in plain words

- **Every evaluation on every card gets its place on the clock.** With Flint Pascal Q5 GGUF's v5 preset split over two
  cards as a pipeline, all 3,733 evaluations of each card had both columns, without waiting and waiting.
- **The places agree with the host's own readings.** No evaluation was placed starting before the host entered it by
  more than 6 µs, none started before the card's previous one ended, and where the host had waited for a card's
  previous evaluation, its next reading came after that evaluation's placed end every time (3,418 waits in the last
  pass, the closest 5 µs after it). The card's clock ran slower than the host's by some 13 ppm (each card and load in
  the table below); each card gets a new anchor every 100 ms of work, so the drift moves a placed time by at most
  3.8 µs (est.: 263 ms from the anchor at 14.41 ppm).
- **The first way did not reach the later cards.** The first commit recorded the anchor on the card's own stream when
  it was idle. The second card's stream is never idle when the host enters it in a pipeline: it always holds the next
  input's copy from the first card. That card got no anchor at all without waiting, in 4,014 entries. The second commit
  gives each card a stream of its own for the anchors, and every card got them.
- **The profile changes no token and the build is the base's where the profile is off.** With the top-k sort switch
  on, so that the one sampled request gives one reply ([C985](../2026-11-24_c985_topk_order/README.md)), the five
  loads (profile off, waiting, waiting with anchors, no-wait with anchors, no-wait without) gave the same tokens to all
  14 requests and ran the same 3,733 evaluations per card. The card code is the base's, kernel for kernel, and no
  warning is new.
- **What it does not show:** the map itself on P106s, and what the anchors cost there. That is rig row Q1736, which
  now runs on this build; [the P106 step](#the-p106-step-row-q1736) below says what changes in it.

## The change

All in `ggml/src/ggml-cuda/ggml-cuda.cu`, host code only.

- **`GGML_CUDA_PROFILE_HOST_ANCHORS`:** unset, it follows `GGML_CUDA_PROFILE_HOST_NOWAIT` (on without waiting, off when
  waiting); `0` turns it off, `1` turns it on in the waiting profile too. With it the file has two more columns,
  `gpu_start_ms` and `gpu_end_ms`, after C877's six. Without it the file is C877's, column for column.
- **An anchor:** each card gets a CUDA stream of its own that never holds work. At an entry where the card's newest
  anchor is older than 100 ms, the host records an event on that stream and reads its clock. The card reaches an event
  on an idle stream at once, so the event's time on the card's clock is the host's reading.
- **Placing an evaluation:** when the evaluation is written, its start and end events are placed from the newest anchor
  the card has reached: the anchor's host reading plus `cudaEventElapsedTime` from the anchor to the event. The
  elapsed time may be negative (an evaluation queued before the anchor): CUDA gives the reverse of two events as the
  exact negative time, with no error ([run/negtest.cu](run/negtest.cu), on one card and across two streams of one
  card). The anchor is never far from the event (at most 263 ms here), so the float keeps microseconds.
- **Nothing waits.** An anchor is one event record; checking whether the card has reached it is `cudaEventQuery`.
  Until a card reaches its first anchor its lines have the two columns empty (none were, in passes 2 and 3).
- **A check of every anchor:** each anchor the card reaches is compared with the one before it (the card's clock
  between the two against the host's), and at the backend's free one log line per card gives the anchors, the
  farthest an event was placed from its anchor, and the clock difference: the largest at one check, with the time
  between those two anchors, and the mean drift in ppm.
- The load line ends `..., with each evaluation's start and end on the card (GGML_CUDA_PROFILE_HOST_ANCHORS)` when it is
  on.

The lines stay in the order they are written; sort by `gpu_start_ms` within a card for its timeline. The idle time
before an evaluation on a card is its `gpu_start_ms` minus the card's previous `gpu_end_ms`, and the host profile's 0 ms
is on the same steady clock as `LLAMA_STEP_TRACE`'s marks.

## The checks

All on a build machine (Ryzen 9 5900X, RTX 3090s, driver 595, CUDA 12.9 in the containers) with the rig build, whose
card code is compute 6.1 SASS and PTX: the 3090s run the PTX through the driver's compiler. Base `af00bb13b`
(`pascal-p106-v5`), head `d49b92251` (pass 1) and `cad303eb9` (passes 2 and 3).

### With the profile off: the base's build

[checks.tsv](checks.tsv) has these rows.

| Check | Base `af00bb13b` | Head `cad303eb9` | Verdict |
|---|---|---|---|
| Rig build (`TESTS=1`, the base's extra targets): warning lines | 32 | 32 | none new |
| Card code: SASS of `llama-server`, one SHA-1 per kernel copy | 9,519 | 9,519 | every copy identical |
| Card code: PTX of `llama-server`, one SHA-1 per function | 9,519 | 9,519 | every function identical, no module-level line differs |

Pass 1's head (`d49b92251`) gave the same three rows. The base's build and its log are the
[landing's](../2026-11-24_land_v5_batch2/README.md). The change is host code inside the profile's own functions, which
run only with `GGML_CUDA_PROFILE_HOST` set, so no CPU build or logits test was run for it.

### Flint's v5 preset with the profile, on two cards

Flint Pascal Q5 GGUF ([the published file](https://huggingface.co/CinderellaMan/Flint-Pascal-Q5-GGUF), 29.4 GB) with its
v5 preset ([release/flint-v5/settings.env](../../../release/flint-v5/settings.env), `--pipe-depth 4` kept; the log
shows pipeline parallelism on and 10 pipe slots), the split changed to the two cards (`-ts 1,1`). Five loads, one at a
time, each running [C950](../2026-11-21_c950_flint_slot_plan_gpu/README.md)'s session once through C877's
[session_marks.py](../2026-11-24_c877_host_profile_nowait/run/session_marks.py): 14 requests, six plain-text replies
that the pipe computes (R1, R8, R12 to R15) and eight tool calls under a JSON schema.

| Load | Profile |
|---|---|
| off | unset |
| wait | `GGML_CUDA_PROFILE_HOST=<file>` (waits; C877's six columns) |
| waitanc | the same and `GGML_CUDA_PROFILE_HOST_ANCHORS=1`: the host waits for each card's previous evaluation, so its next reading checks the placed end |
| nowait | `GGML_CUDA_PROFILE_HOST=<file> GGML_CUDA_PROFILE_HOST_NOWAIT=1` (anchors on by default) |
| nowait0 | the same and `GGML_CUDA_PROFILE_HOST_ANCHORS=0` (C877's six columns) |

Three passes of the five loads:

| Pass | Build | Added to every load |
|---|---|---|
| 1 | `d49b92251`: the anchor on the card's own stream, recorded at an entry where the stream is idle | nothing |
| 2 | `cad303eb9`: the anchor on a stream of its own | nothing |
| 3 | `cad303eb9` | `GGML_CUDA_TOP_K_SORT=1` (the sampled request R14 then gives one reply) |

**Every line placed, in order** ([anchors.tsv](anchors.tsv)). Three checks per line, each against something the
anchors cannot fake: `span`, the placed end minus the placed start against the GPU time of the same two events
(`gpu_us`); `order`, an evaluation starts no earlier than the card's previous one ended (one stream); `enter`, an
evaluation starts no earlier than the host entered it (the host reads its clock, then records the start event). The
bound, fixed before passes 2 and 3 from pass 1's anchor-to-anchor jitter (18 µs), is 100 µs: a placed time that
contradicts a host reading by more is a failure, less is the anchor's jitter.

| Pass | Load | Card | Lines | Placed | Span: worst | Order: below 0 | Enter: below 0 | Enter: worst | Beyond 100 µs |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | waitanc | CUDA0 | 3,725 | 3,725 | 1.0 µs | 0 | 77 | −3.0 µs | 0 |
| 1 | waitanc | CUDA1 | 3,725 | 3,304 | 1.0 µs | 0 | 249 | −16.0 µs | 0 |
| 1 | nowait | CUDA0 | 4,014 | 4,014 | 1.0 µs | 0 | 61 | −4.0 µs | 0 |
| 1 | nowait | CUDA1 | 4,014 | **0** | | | | | |
| 2 | waitanc | CUDA0 | 3,717 | 3,717 | 1.0 µs | 0 | 81 | −4.0 µs | 0 |
| 2 | waitanc | CUDA1 | 3,717 | 3,717 | 1.0 µs | 0 | 18 | −6.0 µs | 0 |
| 2 | nowait | CUDA0 | 4,374 | 4,374 | 1.0 µs | 0 | 17 | −3.0 µs | 0 |
| 2 | nowait | CUDA1 | 4,374 | 4,374 | 1.0 µs | 0 | 0 | +372.0 µs | 0 |
| 3 | waitanc | CUDA0 | 3,733 | 3,733 | 1.0 µs | 0 | 51 | −3.0 µs | 0 |
| 3 | waitanc | CUDA1 | 3,733 | 3,733 | 1.0 µs | 0 | 0 | −1.0 µs | 0 |
| 3 | nowait | CUDA0 | 3,733 | 3,733 | 1.0 µs | 0 | 12 | −2.0 µs | 0 |
| 3 | nowait | CUDA1 | 3,733 | 3,733 | 1.0 µs | 0 | 0 | +234.0 µs | 0 |

"Below 0" counts lines beyond the file's rounding (1.5 µs); "worst" is the smallest value. The span differs from
`gpu_us` only by the file's rounding. In pass 1, CUDA1 got its first anchor in the waiting load only after 421
evaluations and none at all without waiting.

**The host's waits against the placed ends** ([waits.tsv](waits.tsv)): in the waiting load with anchors, at each entry
where the host waited for the card's previous evaluation, its entry reading after the wait minus that evaluation's
placed end. The host can only wake after the end, so a negative value would be the anchor's error.

| Pass | Card | Waits | Below 0 | Smallest | Median | Largest |
|---|---|---:|---:|---:|---:|---:|
| 1 | CUDA0 | 124 | 0 | 9.0 µs | 11.0 µs | 31.0 µs |
| 1 | CUDA1 | 2,889 | 0 | 6.0 µs | 9.0 µs | 58.0 µs |
| 2 | CUDA0 | 143 | 0 | 9.0 µs | 12.0 µs | 28.0 µs |
| 2 | CUDA1 | 3,244 | 0 | 5.0 µs | 8.0 µs | 102.0 µs |
| 3 | CUDA0 | 158 | 0 | 9.0 µs | 12.0 µs | 44.0 µs |
| 3 | CUDA1 | 3,260 | 0 | 5.0 µs | 7.0 µs | 115.0 µs |

**The anchors' own check** (the log line per card, in [profile.tsv](profile.tsv) for passes 2 and 3):

| Pass | Load | Card | Anchors | Farthest from its anchor | Checks | Largest clock difference (over) | Drift |
|---|---|---|---:|---:|---:|---:|---:|
| 2 | waitanc | CUDA0 | 173 | 222.695 ms | 172 | −69.8 µs (5,123.723 ms) | −13.99 ppm |
| 2 | waitanc | CUDA1 | 175 | 130.605 ms | 174 | −68.1 µs (5,338.977 ms) | −12.94 ppm |
| 2 | nowait | CUDA0 | 190 | 263.054 ms | 189 | −8.5 µs (585.613 ms) | −14.41 ppm |
| 2 | nowait | CUDA1 | 190 | 142.280 ms | 189 | −8.7 µs (105.922 ms) | −13.02 ppm |
| 3 | waitanc | CUDA0 | 175 | 227.238 ms | 174 | −65.5 µs (4,909.317 ms) | −13.14 ppm |
| 3 | waitanc | CUDA1 | 176 | 130.024 ms | 175 | −60.2 µs (5,127.923 ms) | −11.66 ppm |
| 3 | nowait | CUDA0 | 168 | 228.646 ms | 167 | −64.0 µs (4,925.214 ms) | −12.96 ppm |
| 3 | nowait | CUDA1 | 169 | 139.721 ms | 168 | −60.6 µs (5,140.849 ms) | −11.48 ppm |

The largest differences are over the idle seconds between two requests, where the drift adds up (13 ppm over 5 s is
65 µs); between anchors 100 ms apart it is about 1 µs. Pass 1's log line had another form: CUDA0 in the no-wait load
got 284 anchors with events placed up to 1,258.639 ms after the newest, CUDA1 none.

**Tokens and counts** ([tokens.tsv](tokens.tsv), [profile.tsv](profile.tsv)). Pass 3: all 14 requests gave one token
list in all five loads, and every profiled load wrote 3,733 evaluations per card, ids 0 to 3,732 once each, every one
written (the count line). Passes 1 and 2: 12 of 14; R14, sampled at temperature 0.7, gave a different list in most loads
(the same lists recur across loads with other profile settings: pass 1's wait and nowait gave one, pass 1's nowait0 and
pass 2's nowait another; the cause is the card's top-k order, [C962](../2026-11-23_c962_sampled_reply_gpu/README.md)),
and R15 continues R14's conversation. The evaluation counts per load differ in those passes with those two replies. The
`queued` counts are as in C877: the host waited at every entry with an earlier evaluation running when waiting (155 and
3,260 in pass 3) and never without waiting (up to 7 and 8 still running).

## The P106 step (row Q1736)

Row Q1736 (rig 2 row 22) runs Flint Pascal Q8 GGUF's v5 preset on nine P106s through the deep lane, R176's repeated
request at depths 0 and 8,192, in three loads: profile unset, `GGML_CUDA_PROFILE_HOST` (waiting), and the same with
`GGML_CUDA_PROFILE_HOST_NOWAIT=1`. On this build:

- **The build:** the rig tarball `llama.cpp-20261124-cad303eb9-pascal-p106-v5-profile-anchors.tar.gz` (1,385,344,730
  bytes, md5 `6fd1ee546ae2122b6ed445be66cbcfa0`; `llama-server` md5 `7a4521100e8387471486802298773aa8`) in place of
  `af00bb13b`'s. Where the profile is unset it is `af00bb13b`'s card code kernel for kernel, so the unset load is the
  base's.
- **The loads:** unchanged. The waiting load writes C877's six columns as before; the no-wait load gets the two columns
  by default, which is the map. In R176's script the arms' variables are `ARM_ENV_<arm>`, so the three loads are
  for example `ARM_ENV_off=` (empty), `ARM_ENV_h1="GGML_CUDA_PROFILE_HOST=$O/hp-h1.csv LLAMA_STEP_TRACE=$O/st-h1.jsonl"`
  and `ARM_ENV_n1="GGML_CUDA_PROFILE_HOST=$O/hp-n1.csv GGML_CUDA_PROFILE_HOST_NOWAIT=1 LLAMA_STEP_TRACE=$O/st-n1.jsonl"`;
  nothing else is added.
- **What to read from it, besides the row's own questions:** each card's anchors line in the server log (anchors, the
  farthest placement, the clock difference and the drift on a P106); every line of every card placed; then per
  pipelined step and card, the idle time before each evaluation (its `gpu_start_ms` minus the card's previous
  `gpu_end_ms`), next to `LLAMA_STEP_TRACE`'s marks on the same clock. That map is Q194's question.
- **If the no-wait step is slower than the unprofiled step beyond the spread,** one more load,
  `GGML_CUDA_PROFILE_HOST_NOWAIT=1 GGML_CUDA_PROFILE_HOST_ANCHORS=0`, tells the anchors' part (ten event records a
  second per card) from no-wait's.

## Files

| File | What |
|---|---|
| [checks.tsv](checks.tsv) | the build checks: warnings, binaries, card code per kernel copy and per PTX function, the tarballs, the negative-time test |
| [tokens.tsv](tokens.tsv) | per pass and request: kind, tokens and md5 of the token ids per load, and whether all loads agree |
| [profile.tsv](profile.tsv) | per pass, profiled load and card: the file's columns, lines, evaluation ids, the count line and (passes 2 and 3) the anchors line |
| [anchors.tsv](anchors.tsv) | per pass, load with the two columns, and card: lines placed, and the span, order and enter checks |
| [waits.tsv](waits.tsv) | per pass and card, the waiting load with anchors: the host's reading after each wait minus the placed end |
| [run/run.sh](run/run.sh) | the run: clone and fetch, rig build, card code, the five loads, the tables |
| [run/c1004_tables.py](run/c1004_tables.py) | one pass's tables from the loads' output |
| [run/combine.py](run/combine.py) | the three passes' tables into one file each |
| [run/negtest.cu](run/negtest.cu) | `cudaEventElapsedTime` with the second event first, on one stream and across two |

## Commands

On the build machine, with a llama.cpp clone that holds `af00bb13b` (`SEED`), the base's rig build and its log, a copy
of this repository (`REPO`), Flint Pascal Q5 GGUF (`MODEL`), and the cards free:

```bash
W=<work dir> SEED=<clone> BASE_RIG=<af00bb13b's rig build folder> BASE_RIG_LOG=<its build log> REPO=<repo copy> \
  MODEL=<flint-pascal-Q5.gguf> GPUS=0,1 TS=1,1 MOUNTS='<model and work folders>' RIG_CCACHE=<compiler cache> \
  PREFLIGHT=<cuda check> LOCK=<build lock> JOBS=8 bash run.sh
```

That is pass 2; pass 1 was the same with `REF` at `d49b92251` (`HEAD_SHA=d49b9225109d454f08d0f4ff50048415a424d547`),
and pass 3 added `STEPS="session tables" EXTRA_ENV=GGML_CUDA_TOP_K_SORT=1`. Then, over the three passes' `tables`
folders (pass 1's made again by this `c1004_tables.py`, whose log patterns read passes 2 and 3):

```bash
python3 -I run/combine.py . 1=<pass 1 tables> 2=<pass 2 tables> 3=<pass 3 tables>
```

The negative-time test: `nvcc -O2 -arch=sm_61 -o negtest negtest.cu && ./negtest` in `nvidia/cuda:12.9.2-devel-ubuntu24.04`
on one card.

## Provenance and licence

Fork branch `pascal-p106-v5-profile-anchors` of [Miner Llama](https://github.com/Cinderella-Man/miner-llama) (`d49b92251`,
`cad303eb9`; MIT, as llama.cpp). Flint Pascal Q5 GGUF is published by this project; C950's session prompts are this
repository's. The tables summarize the server logs and profile files, which are not published.
