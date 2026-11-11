# The served step, part by part (S37, Q961, 2026-11-10)

What this measured: where the time of one served speculative step goes on the Miner Llama v3.0 release line
(`pascal-p106-v3-b12076-ladder5` `1551f5fe1`, the v3 preset exactly: UkisAI's Swift-1.5 GSQ-RCO Q2_0 of
Qwen3.8-Flash-Next on nine P106-090 cards, the MTP draft at three tokens), and what the draft's two one-card
switches change ([Q961](../../../quants/experiments.md)). A step is one verify pass of the main model over four
tokens plus the draft's four passes that prepare the next one. The passes themselves were profiled the same morning
([2026-11-10_release_profile](../2026-11-10_release_profile/README.md), R41): a plain four-token pass is 91.20 ms,
of which 85.36 ms is GPU time. This run measures the step around that pass, from generation, with the fork's host
profile (`GGML_CUDA_PROFILE_HOST`).

## Result

### The step of the preset

Load `d1off`, the greedy request (82 typical steps at 79 to 335 tokens of context, the depth of R41's passes). The
mean column adds up to the step; medians beside it.

| Part of the step | mean, ms | median, ms |
|---|---:|---:|
| verify pass: GPU time of the nine cards | 102.10 | 101.91 |
| verify pass: the rest of its span (the eight crossings, the sampled rows and hidden rows back, the acceptance test, the server before the draft) | 4.98 | 4.97 |
| catch-up pass of the draft: crossing card 4 → card 8 | 1.48 | 1.47 |
| catch-up pass: GPU time on card 8 | 3.66 | 3.63 |
| catch-up pass: host turnaround | 0.33 | 0.28 |
| draft pass 1: crossing, GPU time, turnaround | 0.58, 3.34, 0.52 | 0.58, 2.64, 0.51 |
| draft pass 2: crossing, GPU time, turnaround | 0.58, 3.43, 0.51 | 0.57, 2.58, 0.48 |
| draft pass 3: crossing, GPU time | 0.63, 3.23 | 0.58, 2.59 |
| the server between the last draft pass and the next verify pass | 7.82 | 7.84 |
| **the step** | **133.18** | **131.57** |

- **On a card's clock: 115.76 ms of the 133.18 (86.9 %).** The verify pass's 102.10 ms and the draft's 13.66 ms on
  card 8.
- **On no card's clock: 17.43 ms (13.1 %).** The verify pass's rest 4.98 ms, the draft's four crossings 3.27 ms,
  its turnarounds before the last pass 1.36 ms, and the server's 7.82 ms before the next verify pass.
- A drafting pass's GPU time is 2.58 to 2.64 ms with the shortlist head. In 14 of the 82 steps the three drafting
  passes take 6.3 ms each: the preset drafts with the full head for two steps after a target token outside the
  shortlist. That is why the means sit above the medians.

### Beside R41's plain four-token pass

GPU time per card, ms (medians; R41: 82 prompt passes without the draft, depth 0 to 336, no output row):

| Card | R41, plain pass of four tokens | the verify pass of a step (`d1off`, greedy) | difference |
|---|---:|---:|---:|
| 0 | 11.56 | 12.65 | +1.09 |
| 1 | 10.45 | 11.24 | +0.79 |
| 2 | 10.54 | 11.38 | +0.84 |
| 3 | 10.75 | 11.48 | +0.73 |
| 4 | 8.64 | 9.71 | +1.07 |
| 5 | 10.70 | 11.41 | +0.71 |
| 6 | 9.12 | 9.63 | +0.51 |
| 7 | 10.72 | 11.44 | +0.72 |
| 8 (the output head) | 2.88 | 12.95 | +10.07 |
| **all nine** | **85.36** | **101.89** | **+16.53** |
| the pass's span beyond its GPU time | 5.84 | 4.97 | −0.87 |
| **the pass** | **91.20** | **106.88** | **+15.68** |

- **A served verify pass is 15.68 ms longer than a plain four-token pass, and all of it is GPU time.**
- **Card 8 takes 10.07 ms more:** the verify pass computes four output rows and samples each on the card, where a
  prompt pass computes none. R41 measured +5.22 and +4.63 ms for one output row.
- **Cards 0 to 7 take 6.46 ms more together.** With a draft of three the model keeps three rollback snapshots of
  its recurrent state; R41 measured 4.91 ms for them in a five-token pass.
- **Outside the cards the verify pass costs no more than a plain pass:** 4.97 ms against 5.84 ms. The sampled rows'
  way back, the acceptance test and the server's work before the draft are inside those 4.97 ms.

### The step by request

Median typical step of `d1off`, ms (the second load `d1off-2` in brackets):

| Request | context, tokens | steps | step | verify GPU | verify rest | draft GPU | crossings | turnarounds | server before the verify |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| long (the agent's first turn) | 2,760 to 3,015 | 88 (15) | 140.46 (139.26) | 109.23 (108.13) | 5.19 (5.21) | 13.32 (12.93) | 3.53 (3.51) | 1.34 (1.35) | 7.74 (7.91) |
| greedy | 79 to 335 | 82 (82) | 131.57 (130.51) | 101.91 (101.91) | 4.97 (4.94) | 11.48 (11.40) | 3.22 (3.21) | 1.28 (1.29) | 7.84 (7.48) |
| smoke | 79 to 421 | 117 (45) | 129.91 (127.75) | 102.08 (101.53) | 4.96 (4.97) | 11.54 (11.42) | 3.21 (3.19) | 1.27 (1.28) | 6.84 (6.31) |

At 2,760 tokens of context the verify pass's GPU time is 7.32 ms above the greedy request's and the draft's 1.84 ms;
the parts outside the cards do not grow with depth. The release's agent arm on this build ran 150.4 ms a step over
whole sessions at 2,000 to 12,000 tokens of context
([flash_next_review_2026-11.md](../../research/notes/flash_next_review_2026-11.md)).

### Parts above 5 ms

Beyond R41's plain pass, in the greedy request's medians:

| Part | ms | Registry row it belongs to |
|---|---:|---|
| the draft's GPU time on card 8 (catch-up 3.63, three passes of 2.58 to 2.64) | 11.48 | — (the draft's own work) |
| the verify pass's four output rows and their sampling, card 8 | +10.07 | Q663 (a shortlist head for verify batches, parked), Q1211 (its parts, to measure) |
| the server between the last draft pass and the next verify pass | 7.84 | Q960 (host timestamps that split it) |
| the verify pass on cards 0 to 7 above a plain pass | +6.46 | Q1159 (replay instead of rollback snapshots, upstream) |
| the draft's crossings and its turnarounds before the last pass | 3.22 + 1.28 = 4.50 | Q1148, Q969: below their 5 ms line |
| the verify pass's rest | 4.97 | no more than a plain pass's 5.84 |

### The two switches

Median typical step per request, ms, every load by value (typical steps in brackets). `d1off` and `d1off-2` are two
loads of the preset; a switch arm's class compares its value with both (the rule is in `run/arms.py`).

| Load | long | greedy | smoke |
|---|---:|---:|---:|
| `d1off` | 140.46 (88) | 131.57 (82) | 129.91 (117) |
| `d1off-2` | 139.26 (15) | 130.51 (82) | 127.75 (45) |
| `d1cache` (`LLAMA_DRAFT_EMBD_CACHE=8192`) | 248.04 (87): **loss** | 130.74 (82): negligible | 127.75 (111): negligible |
| `d1carry` (`LLAMA_DRAFT_H_CARRY=1`) | 140.60 (13): negligible | 130.61 (83): negligible | 129.06 (103): negligible |
| `d1on` (both) | 246.04 (26): **loss** | 130.18 (82): negligible | 128.43 (169): negligible |

The greedy request draws nothing at random, so loads with the same reply and draft counts ran the same steps, and
step i of one load pairs with step i of another. Mean difference from `d1off` over the 82 paired steps:

| Load | step, ms | standard error | crossings | turnarounds before the last pass | verify rest | server before the verify |
|---|---:|---:|---:|---:|---:|---:|
| `d1off-2` (the same arm) | −0.68 | 0.13 | −0.06 | +0.03 | +0.03 | −0.55 |
| `d1cache` | −0.91 | 0.17 | −3.27 | +1.59 | +1.05 | −0.70 |
| `d1on` | −1.07 | 0.13 | −3.27 | +1.31 | +1.05 | −0.62 |

- **On the two short requests neither switch is beyond the spread of two loads of the preset.** The row cache
  removes the 3.27 ms counted as crossings, but 2.64 ms (`d1cache`) and 2.36 ms (`d1on`) of host time remain before
  the passes enter card 8: each pass's inputs still go up there, and with the cache that time is counted in the
  turnarounds and in the verify pass's rest. A second load of the preset is already 0.68 ms faster than the first.
- **The hidden-row carry alone moves time between passes** (`d1carry`, greedy, medians): crossings 2.82 ms against
  3.22, turnarounds before the last pass 1.51 against 1.28.
- **With the row cache the long request's steps wait on the host.** See the next section. That makes both cache
  arms a loss as measured.

### The row cache in the long request: host waits and swap

[waits.tsv](waits.tsv): generation steps that hold a host wait of more than 50 ms after a draft pass, and
[host_mem.tsv](host_mem.tsv): the host's memory during each request (4 GB of RAM, sampled once a second).

| Load | long request: steps with such a wait | median wait, ms | their sum, ms | swapped out during the long request, MiB | lowest MemAvailable, MiB | highest memory pressure (full, avg10) |
|---|---:|---:|---:|---:|---:|---:|
| `warm` | not profiled | | | 29 | 75 | 0.18 % |
| `d1off` | 0 of 92 | | | 4 | 70 | 0.18 % |
| `d1cache` | 46 of 90 (steps 0 to 55) | 115.9 | 5,620.8 | 337 | 320 | 17.80 % |
| `d1carry` | 1 of 16 (step 0) | 73.3 | 73.3 | 3 | 96 | 0.44 % |
| `d1on` | 16 of 29 (steps 1 to 17) | 120.5 | 2,012.2 | 64 | 88 | 0.87 % |
| `d1off-2` | 0 of 18 | | | 1 | 86 | 0.18 % |

- **The waits sit between draft passes, not in any card's work.** In `d1cache` 28 follow the catch-up pass and 18
  the last draft pass; in `d1on` all 16 follow the last draft pass. The passes' GPU times are unchanged.
- **No greedy or smoke request has such a wait, in any load.**
- **The two loads with the row cache are the two that swapped most** during the long request, the first request
  after the load and the one that takes the host closest to its limit: MemAvailable falls to 75, 70, 96 and 86 MiB
  in the four loads without the row cache.
- **The cause is not established.** One load an arm cannot tell whether the row cache pushes the host into swap or
  whether these two loads crossed the limit by chance, and the host profile does not show what the host does during
  a wait. Q1209 holds the question. The preset without the cache waits the same way when its long request swaps
  (R86, R57: [2026-11-11_c752_long_request_stalls](../2026-11-11_c752_long_request_stalls/README.md)).

### The drafts compared

`run/identity.py` on the greedy replies ([identity.txt](identity.txt)): `d1off-2`, `d1cache`, `d1on` and the
unprofiled `warm` load carry `d1off`'s text and draft counts (255 drafted, 170 accepted). **`d1carry` differs:** the
same reply text, but 258 tokens drafted and 169 accepted, so at least one drafted token was another and one more
step ran. Which token, and why the carry with the row cache (`d1on`) does not show it, is Q1210.

### The profile's own cost, and what the server reports

[arms.md](arms.md), first table. The greedy reply's generation took 11,262.9 ms in the unprofiled `warm` load and
11,334.4 and 11,279.0 ms in the two profiled loads of the preset (the same text, 86 steps by the server's count:
130.96, 131.80 and 131.15 ms a step). The host profile changes the step by less than two profiled loads differ.
It does slow prompt reading, where micro-batches overlap across the cards: the 2,760-token prompt took 30,112.6 ms
unprofiled and 34,591.9 and 33,817.5 ms profiled.

The server's generation time over its step count is above the typical step where a reply holds one-off costs. In
`d1off`'s long reply (180.66 ms a step by the server's counts, 140.46 ms the typical step) the draft's first
catch-up after the prompt took 1,004.7 ms of GPU time on card 8 and the reply's last verify pass 1,308.3 ms
(1,005.9 ms of it on card 4): est. the first use of a shape in a load (Q1193).

### Graph reuse (Q1104)

The server logs the main model's reused graphs per request. In `d1off`: 88 of the long request's 100 main passes,
83 of the greedy request's 88 and 119 of the smoke request's 121. So the main model rebuilds no graph per step. The
draft's context logs no count. Its turnaround between a step's drafting passes is 0.48 to 0.51 ms and its crossing
0.57 to 0.58 ms, which leaves no room for a rebuild per drafting pass; the catch-up pass, whose size changes from
step to step, is not settled by this.

## What ran

One detached script on the rig ([run/step_loads.sh](run/step_loads.sh)), six loads in this order, 41 minutes
(09:23 to 10:04), nothing else on the rig:

| Load | What | Profile |
|---|---|---|
| `warm` | the preset | none: warms the cards, gives the server's timings without the profile |
| `d1off` | the preset | `GGML_CUDA_PROFILE_HOST` |
| `d1cache` | + `LLAMA_DRAFT_EMBD_CACHE=8192` (the draft's token rows cached on card 8) | the same |
| `d1carry` | + `LLAMA_DRAFT_H_CARRY=1` (the draft's hidden rows kept on card 8 between a step's passes) | the same |
| `d1on` | both | the same |
| `d1off-2` | the preset again | the same |

- **The line:** `release/flash-next-v3/settings.env` exactly (`PRESET_ENV`, `PRESET_ARGS`, the MTP draft), started
  through `bench/agent_check/rig_server.sh` with its monitors and RAM watchdog (floor 60 MiB), as the release's
  agent arms ran. Both switches are in the release build, off by default
  ([2026-11-04_draft_one_card](../2026-11-04_draft_one_card/README.md) describes them).
- **Requests, three per load, four seconds apart:** `long`, the agent's first-turn request (2,760 prompt tokens,
  at most 256 generated, the preset's sampling, built by `bench/agent_check/first_turn_request.sh`); `greedy`
  ([greedy_request.json](../2026-11-04_draft_one_card/run/greedy_request.json), temperature 0, 256 tokens); `smoke`
  ([smoke_request.json](../2026-11-03_flash_next_mtp/run/smoke_request.json), the preset's sampling, at most 768
  tokens, the greedy request's prompt). `long` and `smoke` draw at random without a fixed seed, so their replies
  and lengths differ per load: in three loads the long reply ended early in a tool call (53, 109 and 60 tokens).
- **The step is read from generation.** With the draft loaded a prompt gives no pass under five tokens (Q1192).
- **The split ran on a build machine's CPU**, never on the rig.

## How a step is cut

`run/step_split.py` (its header has the definitions):
- The profile has one line per graph evaluation and card: the host's time at entry and exit, and the GPU time
  between two events around the work. The profile waits at entry for the card's previous evaluation, so an
  evaluation's work starts at its entry.
- A main pass is one evaluation on each of the nine cards. The host queues all nine within 3.79 ms (the greedy
  request's median) and the cards hand the residual on through staged copies, so the host's clock does not show
  when each card worked. The verify pass's span is therefore cut in two only: the cards' GPU time (they work one
  after another) and the rest.
- The draft's passes run one at a time, each on card 4 (its token row) and card 8 (its layer and head), or on
  card 8 alone with the row cache. Their crossings and turnarounds are read directly.
- A typical step has four draft passes and a verify pass within 12 % of the request's median GPU time; the first
  three steps of a request are left out. This drops a prompt's last small micro-batch and a reply's last verify
  pass where it pays one-off launches.
- With the row cache the catch-up pass has no crossing by construction: what it uploads before entering card 8 is
  counted in the verify pass's rest (+1.05 ms there). Compare arms on the step, or on sums across that boundary.

## Files

| File | What it is |
|---|---|
| [split_d1off.md](split_d1off.md), [split_d1off-2.md](split_d1off-2.md), [split_d1cache.md](split_d1cache.md), [split_d1carry.md](split_d1carry.md), [split_d1on.md](split_d1on.md) | per load: the trace's pieces, steps per request, and every part's mean, median, p10 and p90 over all requests and per request; the verify pass's GPU time per card |
| `steps_<load>.tsv` | per load: every generation step with its parts (step, verify span, GPU time per card, each draft pass's crossing, GPU time and turnaround) |
| [arms.md](arms.md), [arms.tsv](arms.tsv) | the server's own timings per reply (tokens, drafts, acceptance, generation and prompt time); each part's median and mean per load and request |
| [arms_class.tsv](arms_class.tsv) | each switch arm's step against the two loads of the preset, with its class |
| [arms_paired.tsv](arms_paired.tsv) | the greedy request's steps paired with `d1off`'s: mean difference per part |
| [identity.txt](identity.txt) | the greedy replies' text and draft counts against `d1off`'s |
| [waits.tsv](waits.tsv) | per load and request: steps with a host wait over 50 ms, after which pass, their sum |
| [host_mem.tsv](host_mem.tsv) | per load and request: MemAvailable at the start and its minimum, swap used, memory pressure |
| `passes_<load>.md`, `host_steps_<load>.md` | the 2026-11-04 scripts' output over each whole trace (`draft_passes.py`, `host_steps.py`), for comparison with [C338's split](../2026-11-04_draft_one_card/c338_passes.md) |
| [run/step_loads.sh](run/step_loads.sh) | the loads, on the rig |
| [run/step_split.py](run/step_split.py) | a trace cut into requests, steps and parts |
| [run/arms.py](run/arms.py) | the loads side by side, the classes, the paired steps |
| [run/waits.py](run/waits.py) | the host waits and the host's memory per request |

## Commands

```bash
# on the rig, the rig idle (BIN: the release build's bin/, MODEL: the Swift Q2_0 first shard, DRAFT: the MTP draft,
# S: a folder with this repository's files, LONG: the first-turn request built by
# BUILD_SSH=<build machine> bash bench/agent_check/first_turn_request.sh long_request.json)
BIN=$BIN MODEL=$MODEL DRAFT=$DRAFT PRESET=$S/release/flash-next-v3/settings.env \
  RIG_SERVER=$S/bench/agent_check/rig_server.sh REQ_LONG=$LONG \
  REQ_GREEDY=$S/docs/data/2026-11-04_draft_one_card/run/greedy_request.json \
  REQ_SMOKE=$S/docs/data/2026-11-03_flash_next_mtp/run/smoke_request.json OUT=$OUT \
  setsid nohup bash $S/docs/data/2026-11-10_served_step_split/run/step_loads.sh > console.out 2>&1 < /dev/null &

# on a build machine, CPU only (OUT copied there; RES a new folder; R=docs/data/2026-11-10_served_step_split/run)
for l in d1off d1cache d1carry d1on d1off-2; do
  mkdir -p $RES/$l && cp $OUT/$l/reply_*.json $RES/$l/
  nice -n 19 python3 -I $R/step_split.py $OUT/$l/host_profile.csv --load $l --tsv $RES/$l/steps.tsv > $RES/$l/split.md
  nice -n 19 python3 -I docs/data/2026-11-04_draft_one_card/run/draft_passes.py $OUT/$l/host_profile.csv > $RES/$l/passes.md
  nice -n 19 python3 -I docs/data/2026-11-04_verify_host/run/host_steps.py $OUT/$l/host_profile.csv > $RES/$l/steps.md
done
mkdir -p $RES/warm && cp $OUT/warm/reply_*.json $RES/warm/
nice -n 19 python3 -I $R/arms.py $RES --loads warm,d1off,d1cache,d1carry,d1on,d1off-2 --ref d1off,d1off-2 \
  --tsv $RES/arms.tsv > $RES/arms.md
nice -n 19 python3 -I $R/waits.py $RES $OUT --loads warm,d1off,d1cache,d1carry,d1on,d1off-2 \
  --waits-tsv $RES/waits.tsv --mem-tsv $RES/host_mem.tsv
(cd $RES && python3 -I ../docs/data/2026-11-04_draft_one_card/run/identity.py d1off/reply_greedy.json \
  d1off-2/reply_greedy.json d1cache/reply_greedy.json d1carry/reply_greedy.json d1on/reply_greedy.json \
  warm/reply_greedy.json) > $RES/identity.txt
```

## Limits

- One load per switch arm and two of the preset. The classes rest on that; the paired greedy steps are 82 a load.
- The steps are at 79 to 3,015 tokens of context. An agent session runs deeper, where the verify pass's GPU time
  grows (7.32 ms from the greedy request's depth to 2,760 tokens).
- The verify pass's rest (4.97 ms) is one number: the host profile cannot separate the eight crossings from the
  rows' way back and the server's work. The same holds for the 7.84 ms before the next verify pass.
- The long request ran once per load, as each load's first request. Whether the row cache's host waits return on a
  later long request, or in a load that does not swap, was not measured.
- No agent check ran: these are single requests, a diagnostic of the step, not agentic speed.

## Provenance and licence

Measured on 2026-11-10 on one rig of nine P106-090 cards with the release build of the Miner Llama fork. The tables
are this project's own measurements. The models are UkisAI's Swift-1.5 GSQ-RCO Q2_0 of Qwen3.8-Flash-Next and the
MTP draft, not republished. The first-turn request is built from the agent check's task files at run time and is not
part of this folder; neither are the replies.
