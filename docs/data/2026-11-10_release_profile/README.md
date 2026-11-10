# The release line's pass, part by part, at one, four and five tokens (R41, Q1191)

Registry rows: [Q1191](../../../quants/experiments.md) (this profile), [Q681](../../../quants/experiments.md) (the
profiler), Q1192 and Q1193 (two things the run found about the method). The method is C120r's
([2026-11-04_moe_skip](../2026-11-04_moe_skip/README.md)) with C271's requests
([2026-11-04_kreal_p106](../2026-11-04_kreal_p106/README.md)); the line is Miner Llama v3.0's Flash-Next preset
([release/flash-next-v3](../../../release/flash-next-v3/README.md)).

Every time estimate for Flash-Next before this run was priced on an older build's token. This is the map of the build
that shipped: where a pass of k consecutive tokens spends its time on nine P106-090 cards, with the Q8_0 hop cast, the
staged hop and the keep-alive on.

## What ran

- **Rig and build:** nine P106-090, idle; fork build `pascal-p106-v3-b12076-ladder5` `1551f5fe1` (v3.0).
- **Model and line:** Swift Qwen3.8-Flash-Next GSQ-RCO Q2_0 with the preset's environment and arguments
  (`release/flash-next-v3/settings.env`), without the MTP draft and its arguments, plus `--no-warmup` and
  `-ot token_embd.weight=CPU`. The override changes nothing about where the embeddings are; it turns llama.cpp's
  pipelining of a call's micro-batches off, so that each micro-batch runs alone, as a decode or verify pass does.
- **Requests (C271's):** the first N tokens of held-out coding-agent text as one prompt whose micro-batches hold
  exactly k tokens (`"ubatch_taper": "short=1000000:k,min=1"`), one output token, the KV cache not reused.
- **Three loads of llama-server**, in this order ([run/release_prof.sh](run/release_prof.sh)):
  - `warm`: unprofiled, 1,680 tokens at k = 1, 4 and 5;
  - `prof`: under `GGML_CUDA_PROFILE_MODE=group`, the first 340 tokens at k = 1, 4 and 5;
  - `wall`: unprofiled, 1,680 tokens at each k, then the profile's own 340 tokens at each k.
- **The cards:** after each load's requests every card ran at 1,847 to 1,911 MHz (the keep-alive holds the boost
  clock) at 31 to 40 °C. Loaded, the nine cards held 4,885, 4,981, 4,837, 4,997, 4,113, 4,853, 4,275, 4,865 and
  2,333 MiB.

A first run of the same three loads had the MTP draft loaded, as the preset has. With the draft loaded the engine
keeps every planned micro-batch at five tokens or more, so both of its requests (asked for k = 1 and k = 4) ran in
five-token passes. Its trace is kept as the five-token pass of the draft line
([below](#five-tokens-with-the-draft-loaded)).

## Results

GPU time per pass by part, ms, summed over the nine cards, the profiler's 7.6 µs subtracted per span
([parts.tsv](parts.tsv), traces `release-k1`, `release-k4`, `release-k5`; the parts are
[C266](../2026-11-04_kpass_profile/README.md)'s). The passes are plain prompt passes: no output row. The unprofiled
pass is the 336-token call of the `wall` load's 340-token request divided by its passes
([pass_times.tsv](pass_times.tsv)).

| Part | k = 1 | k = 4 | k = 5 |
|---|---:|---:|---:|
| attention and recurrent projections | 15.81 | 34.13 | 39.00 |
| routed experts | 6.97 | 19.68 | 24.21 |
| hyper-connections (BF16) | 9.49 | 12.73 | 18.20 |
| other operations without weights | 4.72 | 6.10 | 6.23 |
| shared expert | 2.56 | 4.79 | 5.13 |
| Gated DeltaNet state and conv | 2.65 | 3.79 | 4.25 |
| router | 1.79 | 2.35 | 4.57 |
| attention (FLASH_ATTN_EXT) | 0.59 | 1.01 | 1.50 |
| other weights | 0.47 | 0.78 | 0.94 |
| **the cards' GPU time** | **45.05** | **85.36** | **104.03** |
| the unprofiled pass | 48.31 | 91.20 | 111.00 |
| **hops, host and gaps** (their difference) | **3.26** | **5.84** | **6.97** |

Passes kept: 334 at k = 1, 82 at k = 4, 65 at k = 5.

- **The cards' kernels are 93.3 %, 93.6 % and 93.7 % of a pass** at one, four and five tokens. Hops, host work and gaps are 3.26 ms of a one-token pass and
  5.84 ms of a four-token pass. That figure still holds the request's start (the time from the request's launch to its
  first pass, 112.7 ms in the profiled load's second request): at most 0.34 ms a pass at k = 1 and 1.34 ms at k = 4.
- **At four tokens the projections are 40 % of the GPU time** (34.13 ms), the routed experts 23 % (19.68 ms) and the
  hyper-connections 15 % (12.73 ms). The largest launch groups at k = 4: the routed experts' gate and up with SwiGLU
  10.88 ms, their down projection 7.04 ms, `attn_qkv` in Q3_K 5.42 ms, `attn_gate` in Q3_K 4.95 ms, `ssm_out` in
  IQ4_XS 3.88 ms ([summaries/release-k4.txt](summaries/release-k4.txt)).
- **From one token to four the pass grows 1.89 times** (45.05 to 85.36 ms of GPU time): the routed experts 2.82
  times, the projections 2.16, the hyper-connections 1.34.
- **Per card** ([per_card.tsv](per_card.tsv)), ms a pass at k = 1 and k = 4: card 0 5.85 and 11.56, card 1 5.77 and
  10.45, card 2 5.76 and 10.54, card 3 5.84 and 10.75, card 4 4.40 and 8.64, card 5 5.61 and 10.70, card 6 4.85 and
  9.12, card 7 5.54 and 10.72, card 8 (three layers) 1.44 and 2.88.

### The pass times, each load

Two ways to read a pass time from a request ([calls.tsv](calls.tsv), [pass_times.tsv](pass_times.tsv)):

| | k = 1 | k = 4 | k = 5 |
|---|---:|---:|---:|
| prompt time / (N / k), 1,680 tokens, load `warm` | 51.10 | 97.01 | 118.38 |
| the same, load `wall` | 50.75 | 97.01 | 118.35 |
| the same, 340 tokens, load `wall` | 50.24 | 98.92 | 121.22 |
| first call / its passes, 1,680 tokens (a call of 1,164), load `warm` | 49.60 | 93.99 | 114.63 |
| the same, load `wall` | 49.11 | 93.96 | 114.59 |
| first call / its passes, 340 tokens (a call of 336), load `wall` | 48.31 | 91.20 | 111.00 |
| second call / its passes, 1,680 tokens (512 tokens at depth 1,164 to 1,676, one checkpoint inside), load `warm` | 51.45 | 98.72 | 120.46 |
| the same, load `wall` | 51.41 | 98.80 | 120.47 |
| under the profiler: prompt time / (N / k), 340 tokens | 79.02 | 133.61 | 164.00 |
| under the profiler: from pass to pass | 72.07 | 116.06 | 135.84 |

- **Two loads agree within 0.35 ms** by the prompt time (51.10 and 50.75 at k = 1; 97.01 and 97.01 at k = 4; 118.38
  and 118.35 at k = 5).
- **The prompt time divided by the passes is not the time of a pass.** llama-server reads a prompt of N tokens in calls
  of N − 516, 512 and 4 tokens (N − 4 and 4 for 340). A request's prompt time also holds its start, a context
  checkpoint after each call but the last, and, in a load's first request at a size, one-off launches. In the profiled
  load ([prof_requests.tsv](prof_requests.tsv)): the pause between the first call and the last is 700.7, 787.4 and
  725.8 ms (one pass and one checkpoint); the first pass of a request at a new size takes 467.2, 726.1 and 511.6 ms
  against 72.10, 116.10 and 135.80 ms for the later ones; the first pass with an output row took 907.8 ms on the last
  card. So the table above takes the unprofiled pass from one call: the 336-token call of the 340-token requests,
  which run after their sizes have run in the same load.
- **A pass costs more deeper in the context (est.).** The 512-token call at depth 1,164 to 1,676 takes 26,324.2 ms at
  k = 1 and 12,646.5 ms at k = 4, one checkpoint inside. The 340-token requests' tails (a checkpoint and the last four
  tokens) take 848.1, 747.6 and 757.6 ms, which puts a checkpoint at est. 650 ms. Without it the deeper call is
  est. 50.1 ms a pass at k = 1 and est. 93.7 ms at k = 4, against 48.31 and 91.20 ms at depth 0 to 336. The profile
  covers depth 0 to 336 only.

### Against the earlier builds

The same requests on the same rig and text, prompt time / (N / k) at 1,680 tokens, two loads each:

| Build | T(1) | T(4) | T(5) |
|---|---:|---:|---:|
| `-moeskip` `f3072cbe8`, the switch off (C120r, 2026-11-05) | 67.49 and 65.28 | 139.63 and 139.90 | 175.23 and 171.33 |
| the release line `1551f5fe1` (this run) | 51.10 and 50.75 | 97.01 and 97.01 | 118.38 and 118.35 |
| change, the means of the two loads | −15.46 (−23.3 %) | −42.76 (−30.6 %) | −54.92 (−31.7 %) |

GPU time per pass, ms. The earlier tables took the mean over every evaluation of a request (its first pass and its
last call included); the release line's numbers are given both ways (traces `every-k*` and `release-k*`):

| Part | `-moeskip`, k = 1 | release, every evaluation | release, plain passes | `-moeskip`, k = 5 | release, every evaluation | release, plain passes |
|---|---:|---:|---:|---:|---:|---:|
| routed experts | 8.94 | 7.06 | 6.97 | 30.55 | 24.46 | 24.21 |
| attention and recurrent projections | 20.08 | 16.02 | 15.81 | 49.85 | 39.48 | 39.00 |
| hyper-connections (BF16) | 10.65 | 9.74 | 9.49 | 22.82 | 28.68 | 18.20 |
| shared expert | 3.33 | 2.69 | 2.56 | 6.41 | 5.73 | 5.13 |
| router | 2.88 | 1.82 | 1.79 | 6.06 | 4.60 | 4.57 |
| Gated DeltaNet state and conv | 3.52 | 2.68 | 2.65 | 5.45 | 4.33 | 4.25 |
| attention (FLASH_ATTN_EXT) | 0.83 | 0.61 | 0.59 | 1.97 | 1.57 | 1.50 |
| other operations without weights | 6.75 | 7.90 | 4.72 | 8.89 | 8.72 | 6.23 |
| other weights, output layer | 0.78 | 0.52 | 0.47 | 1.40 | 1.15 | 0.94 |
| the cards' GPU time | 57.75 | 49.04 | 45.05 | 133.41 | 118.71 | 104.03 |

At k = 4, `-moeskip`'s 116.20 ms (C556) against 93.23 ms over every evaluation and 85.36 ms over the plain passes.

- **The pass is 23.3 %, 30.6 % and 31.7 % shorter than on `-moeskip`** at one, four and five tokens. At four tokens
  it fell by 42.76 ms, of which the cards' GPU time is 22.97 ms (every evaluation, against C556's 116.20 ms) or
  30.84 ms (plain passes); the rest is hops, host and gaps. This run does not separate the causes (the hop cast, the
  staged hop, the keep-alive's clock, `-ladder5`'s kernels).
- **Means over every evaluation hold one-off launches.** Over every evaluation the release line's five-token
  hyper-connections read 28.68 ms against 18.20 ms over the plain passes: the first five-token pass of the load spent
  733.1 ms on card 0. How large that share was in the earlier tables is not known, so the like-for-like column is
  "every evaluation", and the plain passes are the figure to price work on.

### One output row

The last pass of each request carries one output row, which the server samples on the card (`-bs`). One such pass each
from the k = 4 and the k = 5 request, both of four tokens, against the 82 plain four-token passes
(traces `output-k4`, `output-k5`):

| Part | plain, 82 passes | the k = 4 request's last pass | the k = 5 request's last pass | first − plain |
|---|---:|---:|---:|---:|
| output layer | 0.00 | 2.86 | 2.86 | +2.86 |
| other operations without weights | 6.10 | 7.93 | 7.59 | +1.83 |
| shared expert | 4.79 | 5.13 | 5.12 | +0.34 |
| attention (FLASH_ATTN_EXT) | 1.01 | 1.33 | 1.34 | +0.32 |
| every other part | 73.46 | 73.32 | 73.08 | −0.14 |
| the cards' GPU time | 85.36 | 90.58 | 89.99 | +5.22 |

One output row adds 5.22 and 4.63 ms of GPU time in the two passes: 2.86 ms in the output layer both times, and 1.83
and 1.49 ms in operations without weights. Each is a single evaluation, at depth 336 where the plain passes sit
around depth 170.

### Five tokens with the draft loaded

The first run's `prof` load had the MTP draft loaded (the preset's whole line). Its two requests each ran 66
five-token passes, one of two tokens and two of four; 65 five-token passes of each are kept (traces `draft-k5-r1`,
`draft-k5-r2`; the draft's own graphs set aside):

| Part | no draft, k = 5 | draft loaded, request 1 | draft loaded, request 2 | request 2 − no draft |
|---|---:|---:|---:|---:|
| Gated DeltaNet state and conv | 4.25 | 9.27 | 9.16 | +4.91 |
| other operations without weights | 6.23 | 6.84 | 6.79 | +0.56 |
| routed experts | 24.21 | 24.91 | 24.68 | +0.47 |
| shared expert | 5.13 | 5.34 | 5.30 | +0.17 |
| hyper-connections (BF16) | 18.20 | 18.47 | 18.31 | +0.11 |
| router | 4.57 | 4.70 | 4.65 | +0.08 |
| attention and recurrent projections | 39.00 | 39.42 | 39.03 | +0.03 |
| attention (FLASH_ATTN_EXT) | 1.50 | 1.49 | 1.48 | −0.02 |
| other weights | 0.94 | 0.96 | 0.94 | 0.00 |
| the cards' GPU time | 104.03 | 111.39 | 110.34 | +6.31 |

- **The draft line's rollback snapshots cost 4.91 ms of GPU time in a five-token pass.** With a draft of three tokens
  the model's context keeps three snapshots of the Gated DeltaNet state (`n_rs_seq` = 3), and the
  `GATED_DELTA_NET` launch groups take 9.16 ms against 4.25 ms.
- **From pass to pass under the profiler** the draft line takes 142.73 and 142.47 ms against 135.84 ms without the
  draft. Unprofiled, its 336-token call took 9,425.4 and 8,110.5 ms in the two 340-token requests of its `wall` load
  ([calls_draft.tsv](calls_draft.tsv)) against 7,485.1 ms without the draft; the calls end in different short passes,
  so this is est. 9 ms more a pass.
- **The draft's own graphs** (its mirror of the prompt on cards 4 and 8) took 151.2 ms of GPU time in the second
  340-token request and 3,025.1 ms in the first, which holds their first launches.
- **Why five tokens:** a recurrent model's trailing `n_rs_seq` + 1 tokens must stay in one micro-batch, and the split
  that follows a planned taper raises every planned micro-batch to `n_rs_seq` + 2 tokens. So on a line with an MTP
  draft a request's `ubatch_taper` cannot give passes of one to four tokens (Q1192).

## How the passes are cut

A group-mode trace names no graph. [run/trace_census.py](run/trace_census.py) tells an evaluation's graph from its
launch groups ([census.tsv](census.tsv), [census_draft.tsv](census_draft.tsv)): per card the trace holds one
evaluation while the server loads, then 340, 85 and 69 evaluations for the three requests (340 tokens at k = 5 are 67
passes of five tokens, one of one token and the last call's four). [run/release_parts.py](run/release_parts.py) cuts
each card's evaluations by those counts, keeps the plain passes of each request (passes 3 to 336, 3 to 84 and 3 to
67: not the first two, not the last call) and calls C120r's estimator on them
(`kpass_parts.py`'s `parts_per_pass` and `prof_summary.py`). What was dropped is listed per card in
[cut.tsv](cut.tsv); every pass's host time and raw GPU time is in [passes.tsv](passes.tsv).

## Limits

- One profiled load. The unprofiled pass times come from two loads (`warm` and `wall`) by the prompt time and from
  one load (`wall`) by the call.
- The passes are prompt passes of real text at depth 0 to 336: no output row, no draft, no rollback snapshots. A
  served step of the preset adds those and the draft's own passes; the last two tables give one output row and the
  snapshots at five tokens, and the step itself is measured from generation, not from prompts (Q961).
- The request's start is inside the unprofiled pass (at most 0.34 ms at k = 1, 1.34 ms at k = 4, 1.68 ms at k = 5,
  taking the profiled load's 112.7 ms).

## Files

| File | What it holds |
|---|---|
| [requests.tsv](requests.tsv), [requests_draft.tsv](requests_draft.tsv) | every request of the run and of the first run with the draft loaded: load, k asked for, prompt tokens, prompt time, prompt time / (N / k). In `requests_draft.tsv` every pass held five tokens whatever k says |
| [parts.tsv](parts.tsv) | ms per pass by part: `release-k1/4/5` (plain passes), `every-k1/4/5` (every evaluation of the request), `output-k4/5` (the last pass, one output row), `draft-k5-r1/r2` (the draft loaded) |
| [per_card.tsv](per_card.tsv) | the same traces per card: ms per pass, passes, the largest kept evaluation over the median |
| [cut.tsv](cut.tsv) | per trace: the request's passes, which were kept, their host times, the time from pass to pass under the profiler, the evaluations set aside (the draft's), and the raw GPU time of every dropped pass per card |
| [passes.tsv](passes.tsv), [passes_draft.tsv](passes_draft.tsv) | every pass of every profiled request: when the first and the last card started it, the last card's and all cards' raw GPU time |
| [calls.tsv](calls.tsv), [calls_draft.tsv](calls_draft.tsv) | every request from the server's log: prompt time, the first call's tokens and time, the second call's, the tail |
| [pass_times.tsv](pass_times.tsv) | per load and request: prompt time / (N / k), first call / its passes, second call / its passes |
| [prof_requests.tsv](prof_requests.tsv) | the profiled requests: the trace's clock in the log's time, the request's start, its first pass, the time from pass to pass, the pause before the last call |
| [census.tsv](census.tsv), [census_draft.tsv](census_draft.tsv) | the graphs each card ran: evaluations, launch groups, median and summed GPU time, first and last host time |
| [summaries/](summaries/) | `prof_summary.py`'s report per trace: per card, the first ops, the weight types, the top launch groups |
| [run/release_prof.sh](run/release_prof.sh) | the rig run: three loads of the preset's line, the requests per k, the cards' memory, temperature and clock around each load |
| [run/trace_census.py](run/trace_census.py) | the graphs of a group-mode trace per card |
| [run/release_parts.py](run/release_parts.py) | the cut by request and the parts of the kept passes |
| [run/pass_times.py](run/pass_times.py) | the pass times from the requests and the server's log |
| [run/tables.py](run/tables.py) | the Markdown tables from `parts.tsv` and `pass_times.tsv` |

## Commands

```bash
# on the rig, idle, detached (about 22 minutes); TEXT is C271's file, the first 20,000 bytes of the agent text
# (md5 312fd205dec2fba4879d4bf421f1e80e); PRESET is release/flash-next-v3/settings.env
BIN=<bin/ of the release build> MODEL=<Swift Q2_0 first shard> PRESET=<settings.env> TEXT=<agent text> OUT=out2 \
  KS='1 4 5' setsid nohup bash run/release_prof.sh > console2.out 2>&1 < /dev/null &
# the first run, with the draft loaded (about 20 minutes): every pass held five tokens
BIN=<bin/ of the release build> MODEL=<Swift Q2_0 first shard> DRAFT=<the MTP draft> PRESET=<settings.env> \
  TEXT=<agent text> OUT=out setsid nohup bash run/release_prof.sh > console.out 2>&1 < /dev/null &

# off the rig (CPU only), the traces as prof-release.csv (out2) and prof-draft.csv (out)
K=../2026-11-04_kpass_profile/run/kpass_parts.py; S=../../../scripts/prof_summary.py
python3 -I run/trace_census.py prof-release.csv --out census.tsv
python3 -I run/trace_census.py prof-draft.csv --out census_draft.tsv
P="python3 -I run/release_parts.py --kpass-parts $K --prof-summary $S --event-us 7.6"
$P --tag release --segments x:1 1:340:3-336 4:85:3-84 5:69:3-67 --out res-release prof-release.csv
$P --tag every --segments x:1 1:340:1-340 4:85:1-85 5:69:1-69 --out res-every prof-release.csv
$P --tag output --segments x:1 1:340:340-340 4:85:85-85 5:69:69-69 --out res-output prof-release.csv
$P --tag draft --segments 5:69:2-66 5:69:2-66 --other-regex mtp_ --out res-draft prof-draft.csv
python3 -I run/pass_times.py --requests out2/requests.tsv --log warm=out2/server-warm.log \
  --log prof=out2/server-prof.log --log wall=out2/server-wall.log --passes res-release/passes.tsv --out res-times
python3 -I run/pass_times.py --requests out/requests.tsv --log warm=out/server-warm.log \
  --log prof=out/server-prof.log --log wall=out/server-wall.log --out res-times-draft
# parts.tsv, per_card.tsv and cut.tsv here are the four runs' files joined (one header; output-k1, the first pass
# with an output row and its one-off launches, left out of parts.tsv); then the tables
python3 -I run/tables.py --parts parts.tsv --pass-times pass_times.tsv
```

## Provenance and licence

Fork branch `pascal-p106-v3-b12076-ladder5` of llama.cpp (MIT), commit `1551f5fe1`. Swift-1.5 GSQ-RCO Q2_0 of
Qwen3.8-Flash-Next, the MTP draft and the agent text as in
[2026-11-04_kreal_p106](../2026-11-04_kreal_p106/README.md) and
[2026-11-05_release_flashnext](../2026-11-05_release_flashnext/README.md). The tables are this project's own
measurements.
