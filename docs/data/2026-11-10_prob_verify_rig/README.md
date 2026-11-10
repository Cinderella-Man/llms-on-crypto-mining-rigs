# Probabilistic verification on the MTP release line, timed on the rig (R38; Q731; 2026-11-10)

What this measured: whether probabilistic verification of the draft makes the Miner Llama v3 release preset
([release/flash-next-v3/settings.env](../../../release/flash-next-v3/settings.env): Qwen3.8-Flash-Next, UkisAI's
Swift-1.5 GSQ-RCO Q2_0 on nine P106-090 cards, an MTP draft of three tokens, build `pascal-p106-v3-b12076-ladder5`
`1551f5fe1`) generate faster on the rig. Two arms:

- **`pv0`**, the preset: the draft takes its first choice, and the main model keeps a drafted token only when its own
  sample is the same token;
- **`pv1`**, the preset plus `--spec-draft-sampling probabilistic --spec-draft-p-min 0`: the draft samples its
  tokens, and the main model keeps a drafted token with probability min(1, p/q).

Counted on newer cards the night before, `pv1` drew 7.74 % more tokens a step
([2026-11-10_prob_verify_acceptance](../2026-11-10_prob_verify_acceptance/README.md), Q1173). The open half was
time: by the source the accept test runs on the host, so a `pv1` step may cost more than a preset step. Here the 66
recorded requests of the release check's coding-agent session (arm `t1swift`, 12 bug-fix tasks) were re-sent with
[bench/agent_check/replay.py](../../../bench/agent_check/replay.py) to two loads of each arm on the rig, alternated
(`pv0 pv1 pv0 pv1`) after a warm-up load: the same prompts, the same seed per request and the recorded reply length as
the limit. Registry row Q731 in [quants/experiments.md](../../../quants/experiments.md).

## Result

**Probabilistic verification is a gain in generation speed on the rig: 21.31 and 21.22 tok/s in its two loads
against the preset's 20.60 and 20.70, +0.61 tok/s (+2.97 %; 95 % interval over requests +0.24 to +1.03), where two
loads of one arm are 0.09 and 0.10 tok/s apart. A step costs 1.10 % more with it (153.74 and 155.13 ms against
152.63 and 152.89) and yields 4.04 % more tokens (3.237 and 3.258 against 3.112 and 3.132). On the rig the gain in
tokens a step is about half of the +7.74 % counted off it.**

Definitions. Steps: generated tokens minus accepted draft tokens, the passes of the main model that yielded tokens.
Tokens a step: generated / steps. Ms a step: the generation's time over the verify steps it holds. The server
counts that time (`predicted_ms`) from the end of the prompt, and a request's first token comes from the prompt's
last pass, so the time holds steps − 1 verify steps. Generation tok/s: generated / generation time, as the agent
check's report counts it. Pooled means summed over the 66 requests before dividing.

### Every load

| Arm | Load | Generated | Accepted of drafted | Share accepted | Steps | Tokens a step | Generation, s | Ms a step | Generation tok/s | Prompt tok/s |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `pv0` | 1 | 19,589 | 13,294 of 18,631 | 0.7135 | 6,295 | 3.112 | 950.7 | 152.63 | 20.60 | 75.57 |
| `pv0` | 2 | 19,536 | 13,298 of 18,508 | 0.7185 | 6,238 | 3.132 | 943.7 | 152.89 | 20.70 | 75.90 |
| `pv1` | 1 | 17,764 | 12,277 of 16,354 | 0.7507 | 5,487 | 3.237 | 833.4 | 153.74 | 21.31 | 75.75 |
| `pv1` | 2 | 20,977 | 14,539 of 19,178 | 0.7581 | 6,438 | 3.258 | 988.5 | 155.13 | 21.22 | 75.89 |
| the recorded session (live, 2026-11-06) | — | 27,067 | 18,447 of 25,704 | 0.7177 | 8,620 | 3.140 | 1,326.8 | 155.10 | 20.40 | 69.42 |

- Loads ran in the order `pv0` 1, `pv1` 1, `pv0` 2, `pv1` 2. [loads.tsv](loads.tsv); every request in every load in
  [requests.tsv](requests.tsv).
- A replayed reply ends at the model's own stop or at the recorded reply's length, whichever comes first, so a load
  generates fewer tokens than the recorded session did (33 and 29 of `pv0`'s 66 replies reached the limit, 29 and 33
  of `pv1`'s).
- `pv1`'s two loads generated 17,764 and 20,977 tokens: two loads of one arm write different text (below). Their
  tok/s are 0.09 apart all the same.
- The recorded session's prompt speed is not comparable: a replayed request re-reads the recorded reply in front of
  it, so a replay load reads 63,823 or 63,887 prompt tokens where the live session read 39,863.

### `pv1` against the preset

An arm's value is the mean of its two loads' pooled values. Spread: the difference between an arm's two loads. Class:
gain when the arm is better than the preset by more than the spread (higher, or lower for ms a step), loss when
worse by more than it, negligible otherwise; the class is the same against the preset's spread alone and against the
larger of the two arms' spreads. The intervals are 95 % bootstrap intervals of the difference, over requests (each
request keeps its rows in all four loads, so the comparison stays paired) and over whole sessions (the 12 tasks).
[compare.tsv](compare.tsv).

| Metric | `pv1`, each load | `pv0`, each load | Difference | % | Spread of `pv1` | Spread of `pv0` | Class | 95 % (requests) | 95 % (sessions) |
|---|---|---|---:|---:|---:|---:|---|---|---|
| generation tok/s | 21.31, 21.22 | 20.60, 20.70 | +0.61 | +2.97 | 0.09 | 0.10 | **gain** | +0.24 to +1.03 | +0.37 to +0.81 |
| ms a step | 153.74, 155.13 | 152.63, 152.89 | +1.68 | +1.10 | 1.39 | 0.26 | **loss** (dearer) | +0.79 to +2.72 | +1.05 to +2.62 |
| tokens a step | 3.237, 3.258 | 3.112, 3.132 | +0.126 | +4.04 | 0.021 | 0.020 | **gain** | +0.069 to +0.186 | +0.088 to +0.161 |
| share accepted | 0.7507, 0.7581 | 0.7135, 0.7185 | +0.0384 | +5.36 | 0.0074 | 0.0050 | **gain** | +0.0190 to +0.0581 | +0.0261 to +0.0490 |
| prompt tok/s (the control) | 75.75, 75.89 | 75.57, 75.90 | +0.08 | +0.11 | 0.13 | 0.33 | negligible | −0.12 to +0.27 | −0.16 to +0.28 |

- **The speed gain is the two effects together:** 4.04 % more tokens a step at a step 1.10 % dearer is 2.97 % more
  tokens a second. Each `pv1` load is faster than each `pv0` load.
- **Paired by request** (each request's value averaged over its arm's two loads, then the mean difference over the
  66 requests, unweighted):
  - tokens a step +0.125 (95 % +0.060 to +0.191), higher in 47 requests, lower in 17, equal in 2;
  - ms a step +3.89 (+1.78 to +6.36), higher in 42, lower in 24;
  - generation tok/s +0.28 (−0.21 to +0.80), higher in 37, lower in 29.

  Unweighted, a short reply counts as much as a long one, and the short replies are where `pv1`'s step is dearest
  (next section). The pooled numbers weigh a request by its time, as a session does.
- **The control holds.** The prompts are the same in every load and neither switch touches prompt reading: 75.57 to
  75.90 tok/s in all four loads, `pv1` +0.11 % against `pv0`.

### Where the extra time is: in the grammar rejections, not in the step

The requests in two halves by their reply limit, which is the same in every load ([by_length.tsv](by_length.tsv)):

| Half | Reply limit, tokens | Metric | `pv1`, each load | `pv0`, each load | Difference | % |
|---|---|---|---|---|---:|---:|
| the 33 shorter | 43 to 172 | ms a step | 157.15, 159.32 | 151.37, 154.90 | +5.10 | +3.33 |
| | | tokens a step | 3.336, 3.320 | 3.134, 3.181 | +0.171 | +5.40 |
| | | generation tok/s | 22.02, 21.64 | 21.44, 21.28 | +0.47 | +2.20 |
| the 33 longer | 183 to 8,747 | ms a step | 153.08, 154.48 | 152.85, 152.55 | +1.08 | +0.71 |
| | | tokens a step | 3.218, 3.248 | 3.108, 3.123 | +0.118 | +3.78 |
| | | generation tok/s | 21.17, 21.15 | 20.46, 20.60 | +0.63 | +3.08 |

A least-squares split of each reply's generation time into a cost per reply and a cost per step that grows with
context depth, per arm over its 132 rows ([fit.tsv](fit.tsv); depth is the prompt's tokens plus half the generated
ones, 5,969 tokens on average over all steps; 95 % bootstrap intervals over requests):

| Term | `pv0` | `pv1` | `pv1` − `pv0` | 95 % of the difference |
|---|---:|---:|---:|---|
| ms a reply, whatever its steps | 223.38 | 301.47 | +78.09 | +19.13 to +171.03 |
| ms a step at depth 0 | 130.21 | 132.42 | +2.21 | −2.05 to +3.58 |
| ms a step per 1,000 tokens of context | 3.41 | 3.11 | −0.30 | −0.52 to +0.37 |
| ms a step at the mean depth | 150.54 | 150.97 | +0.44 | −0.85 to +0.74 |

The same with a cost per grammar rejection added (`pv1` only: the preset's replies had none, so its model stays the
first one):

| Term | `pv1` | 95 % | `pv1` − `pv0` | 95 % of the difference |
|---|---:|---|---:|---|
| ms a reply, whatever its steps | 218.38 | 73.11 to 313.19 | −5.00 | −124.26 to +109.92 |
| ms a step at the mean depth | 151.04 | 149.87 to 153.23 | +0.51 | −0.39 to +0.96 |
| ms a grammar rejection | 389.65 | 174.61 to 771.03 | — | — |

- **A `pv1` step is not measurably dearer than a preset step:** +0.44 ms at the mean depth (−0.85 to +0.74). The
  accept test on the host does not show in the step.
- **The extra time is in the grammar rejections.** Without them in the model a `pv1` reply carries 78.09 ms more
  than a preset reply. With a cost per rejection, a `pv1` reply costs what a preset reply costs (−5.00 ms), and each
  rejection costs 389.65 ms. That is why the pooled step is 1.10 % dearer and why the shorter replies show it most.
- **All 26 rejections (15 and 11 a load) are in replies that end in a tool call,** none in a reply cut at its limit.
  That is the missing stop at the end of a turn (registry Q1187): the rejection path goes on verifying the drafted
  tokens behind an accepted end of turn. The main sampler's token for such a row fails the tools' grammar and is
  drawn again from the whole vocabulary on the host. The preset's path stops there.
- **Est.: with that stop a `pv1` load saves 13 × 389.65 ms, about 5.1 s of its 911 s of generation,** and the gain
  in generation tok/s rises from +2.97 % to about +3.6 % (est.).
- The preset's terms agree with the same morning's profile of single requests
  ([2026-11-10_served_step_split](../2026-11-10_served_step_split/README.md)): a step of 131.57 ms at 79 to 335
  tokens of context and 140.46 ms at 2,760.

### Beside the count off the rig

| Where | `pv0`: share accepted | `pv0`: tokens a step | `pv1`: share accepted | `pv1`: tokens a step | `pv1` against `pv0`, tokens a step |
|---|---|---|---|---|---:|
| off the rig, three newer cards (Q1173) | 0.6892, 0.6993 | 3.043, 3.072 | 0.7663, 0.7731 | 3.283, 3.304 | +7.74 % |
| the rig, nine P106 (this run) | 0.7135, 0.7185 | 3.112, 3.132 | 0.7507, 0.7581 | 3.237, 3.258 | +4.04 % |
| the recorded session, live on the rig | 0.7177 | 3.140 | — | — | — |

- **On the rig the preset keeps more of its drafts than off it, and `pv1` fewer.** Each difference is two to three
  times the spread between two loads of one arm. The rig's preset loads agree with the live session.
- The requests, seeds and reply limits are the same in both runs. What differs: the card split (nine cards, so the
  residual crosses eight card boundaries in Q8_0 where three cards have two), the fork's Pascal kernels (the newer
  cards run other kernels at the preset's defaults), and the untimed request before each load here. Which of these
  moves acceptance is registry row Q1217.
- **A gain in tokens a step counted off the rig is an upper guide, not the rig's number.**

### How far two loads of one arm differ

- Generation tok/s: 0.10 between `pv0`'s loads (+0.48 %, 95 % −0.19 to +0.43), 0.09 between `pv1`'s. Ms a step:
  0.26 and 1.39. Tokens a step: 0.020 and 0.021.
- Prompt tok/s: `pv0`'s first load, the first timed load of the run, read 0.43 % slower than its second (75.57
  against 75.90; 95 % +0.04 to +0.60 for the difference). `pv1`'s loads are 0.17 % apart.
- With the same prompts and seeds, `pv0`'s two loads wrote the same reply in 30 of 66 requests and `pv1`'s in 18;
  a `pv1` load and a `pv0` load in 1 or 2 ([same_reply.tsv](same_reply.tsv)). Off the rig the counts were 37 and
  35.

### The server logs and the loads' conditions

| Load | Requests with a grammar | Their sampled tokens, drawn on the device | Rejected by the grammar after a device draw | Checkpoint restores at a prompt | Error lines | Watchdog kills |
|---|---:|---|---:|---:|---:|---:|
| `pv0` 1 | 66 | 19,594 of 19,594 | 0 | 64 | 0 | 0 |
| `pv0` 2 | 66 | 19,550 of 19,550 | 0 | 64 | 0 | 0 |
| `pv1` 1 | 66 | 17,809 of 17,809 | 15 | 65 | 0 | 0 |
| `pv1` 2 | 66 | 21,017 of 21,017 | 11 | 65 | 0 | 0 |

- **Grammar rejections:** 15 and 11 a load with probabilistic verification, none with the preset (off the rig: 14
  against 1). Each is one resample from the whole vocabulary on the host, 389.65 ms on the rig by the fit above.
- Sampling stays on the device in both arms. The accepted and drafted tokens of the server's request lines equal
  the replay's counters plus the warm-up request's ([log_check.tsv](log_check.tsv)).

| Load | Replay | Minutes | Cards, mean °C | Mean SM clock, MHz | Mean power a card, W | Lowest MemAvailable, MiB | Swapped out, in during the replay, MiB |
|---|---|---:|---:|---:|---:|---:|---|
| `pv0` 1 | 11:13:04 to 11:43:04 | 30.0 | 40.1 | 1,862 | 37.0 | 76 | 342.7, 98.9 |
| `pv1` 1 | 11:48:47 to 12:16:48 | 28.0 | 37.5 | 1,859 | 36.8 | 68 | 337.7, 103.2 |
| `pv0` 2 | 12:22:48 to 12:52:37 | 29.8 | 36.8 | 1,869 | 36.7 | 135 | 457.0, 134.9 |
| `pv1` 2 | 12:58:34 to 13:29:10 | 30.6 | 37.2 | 1,864 | 36.2 | 72 | 350.4, 132.7 |

- Every load held the same memory on the nine cards: 5,101, 5,137, 5,005, 5,153, 4,913, 5,021, 4,423, 5,033 and
  4,625 MiB. Each server was up 311 to 316 s after its start.
- **The 4 GB host swaps in every load, both arms alike:** 338 to 457 MiB written to swap during a replay, and
  MemAvailable down to 68 to 135 MiB (the watchdog's floor is 60). Nothing was killed.
  [conditions.tsv](conditions.tsv).

## What it decides

- **Q731, the speed half:** a gain on the rig, +2.97 % in generation tok/s over the session's 66 requests. In the
  recorded live session generation took 1,326.8 s of the server's 1,901.0 s (prompts 574.2 s), so that is est. 2.0 %
  less server time a session at the same replies.
- **Next for Q731:** the agent check on the rig, the preset against `pv1`, two runs an arm, read on minutes per
  task. This replay caps every reply at its recorded length, so it cannot see a reply that runs longer; the agent
  check can.
- **Q1187 (the missing stop at the end of a turn) has its size on the rig:** 389.65 ms a grammar rejection, 13 a
  load of 66 replies, est. +0.6 points on `pv1`'s gain.
- **Counts off the rig:** the acceptance gain did not carry over whole (Q1217).

## Limits

- **One session's requests:** 66 requests of 12 bug-fix tasks, prompts of 1,952 to 15,137 tokens. The requests of
  one task share their text; the interval over sessions allows for that.
- **Two loads an arm.** A spread is one difference per arm. The gain in tok/s is six times the larger spread.
- **The texts differ** between arms and between loads of one arm, so "paired by request" pairs the prompt, the seed
  and the reply limit, not the reply. `pv1`'s loads differ by 3,213 generated tokens.
- **Generation only.** Time to first token and prompt reading are the preset's in both arms.
- **The costs per reply, per step and per grammar rejection are a fit,** with wide intervals for the per-reply and
  per-rejection terms. The rejection's 389.65 ms was not timed directly: the loads ran without the step trace and
  without a profile.
- **Quality is not measured here.** Off the rig all 12 tasks passed in two runs of each arm
  ([2026-11-10_prob_verify_agent_check](../2026-11-10_prob_verify_agent_check/README.md)).
- One model file, one draft, one draft length, one rig.

## Method

- **The line:** the preset exactly (`PRESET_ENV`, `PRESET_ARGS`, the MTP draft), one llama-server per load on the
  rig's loopback, started through [bench/agent_check/rig_server.sh](../../../bench/agent_check/rig_server.sh) with
  its monitors and RAM watchdog (MemAvailable floor 60 MiB), as the release's agent arms ran. `pv1` adds its two
  flags. No step trace (`LLAMA_TRACE`) and no profile.
- **The loads:** `warm pv0 pv1 pv0 pv1`. The warm-up load is the preset replaying the first task's 9 requests; it
  warms the cards after the rig idled and is in no table. Before each load the rig was checked quiet: no llama
  process and 0 MiB on every card.
- **The requests:** the release check's recorded session `t1swift`
  ([2026-11-05_release_flashnext](../2026-11-05_release_flashnext/README.md)): 66 chat requests in 12 sessions, sent
  in their recorded order, one at a time, streamed as recorded, with their tools and sampling fields (temperature 1,
  top-p 0.95, top-k 20). `replay.py` sets `max_tokens` to the recorded reply's length and a seed fixed per session
  and request, the same in every load. The cache carries over from request to request as it did live.
- **One untimed request first:** after each server is up, one short chat request
  ([smoke_request.json](../2026-11-03_flash_next_mtp/run/smoke_request.json)) takes the first-use costs; it is in no
  table.
- **Three machines:** the rig runs llama-server; a second machine runs `replay.py` under `nice`, where the release
  check's client ran; a third only runs ssh, starts and stops everything and relays the requests through two ssh
  forwards. While a load's requests run, no command runs on the rig: the driver polls the client only.
- **The times** are the server's own per request (`timings`: `predicted_ms`, `prompt_ms`), which `replay.py` keeps
  with the counters `draft_n`, `draft_n_accepted`, `predicted_n` and an MD5 of the reply.
- **The tables:** [run/pv_rig_table.py](run/pv_rig_table.py) on the client machine; 10,000 bootstrap resamples (2,000
  for the fit), seed 1.

## Files

| File | Holds |
|---|---|
| [summary.md](summary.md) | every table of `pv_rig_table.py` as Markdown |
| [loads.tsv](loads.tsv) | per load: generated, drafted, accepted, steps, tokens a step, generation time, ms a step, generation and prompt tok/s, replies at the limit, grammar counters; the recorded session's row |
| [compare.tsv](compare.tsv) | `pv1` against `pv0` and each arm's second load against its first, per metric: every load by value, difference, spreads, class, intervals, per-request counts |
| [requests.tsv](requests.tsv) | per request and load: prompt tokens, reply limit, generated, drafted, accepted, steps, tokens a step, generation ms, ms a step, tok/s, prompt ms, grammar counters, finish reason, the first 12 characters of the reply's MD5 |
| [by_length.tsv](by_length.tsv) | the two halves by reply limit, per load |
| [fit.tsv](fit.tsv) | the least-squares split per arm, without and with a cost per grammar rejection: ms a reply, ms a step at depth 0, its growth per 1,000 tokens, the step at the mean depth, ms a rejection, intervals, the difference to `pv0` |
| [same_reply.tsv](same_reply.tsv) | per pair of loads: requests with the same reply and with the same counters |
| [log_check.tsv](log_check.tsv) | per load, from the server log: the switches on its line, sampling on the device, grammar rejections, accepted and drafted tokens three ways, checkpoint restores, error lines |
| [conditions.tsv](conditions.tsv) | per load: the replay's times, the cards' temperature, clock, power, utilization and memory, the host's free memory and swap |
| [run/run_pv_rig.sh](run/run_pv_rig.sh) | the chain: per load the quiet check, the server on the rig, the warm-up request, the replay on the client, the logs copied off |
| [run/pv_rig_table.py](run/pv_rig_table.py) | the tables |

The tasks are the bug-fix set of [bench/agent_check](../../../bench/agent_check/README.md) (`tasks/bugfix`); the
tables name them by task id and hold no prompt or reply text.

## Commands

On the machine that only runs ssh, from a checkout of this repository. `RIG` and `CLIENT` are ssh aliases; `BIN` is
the build's `llama-server` on the rig, `MODEL` the first shard of Swift Q2_0 there (the other two beside it),
`MODEL_DRAFT` the MTP draft, `RIG_LOGDIR` a folder on the rig; `CLIENT_DIR` and `CLIENT_WORK` are folders on the
client, `LOG` the recorded `proxy.jsonl` there with its `proxy.jsonl.bodies.gz` beside it; `OUT` is a local folder.

```bash
RIG=$RIG CLIENT=$CLIENT RIG_PORT=8138 RELAY_PORT=8238 CLIENT_PORT=18238 BIN=$BIN MODEL=$MODEL \
  MODEL_DRAFT=$MODEL_DRAFT RIG_LOGDIR=$RIG_LOGDIR CLIENT_DIR=$CLIENT_DIR CLIENT_WORK=$CLIENT_WORK LOG=$LOG OUT=$OUT \
  setsid nohup bash docs/data/2026-11-10_prob_verify_rig/run/run_pv_rig.sh > console.txt 2>&1 < /dev/null &
```

It is done when `$OUT/done` exists (`$OUT/failed` otherwise); the five loads took 2 h 32 min, 10:57 to 13:29. Then,
on the client:

```bash
nice -n 19 python3 -I pv_rig_table.py "$CLIENT_WORK" --out "$CLIENT_WORK/tables"
```

## Provenance

- The requests are the release check's own recording of this project's agent check; the tables hold counters, times
  and hashes only.
- The model is UkisAI's Swift-1.5 GSQ-RCO Q2_0 of Qwen3.8-Flash-Next with its MTP draft, as in the release preset.
- Raw output (server logs, the replay's rows, the monitors' logs) is not published.
