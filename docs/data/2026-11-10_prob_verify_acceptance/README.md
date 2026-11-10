# Probabilistic verification and kept draws on the MTP release line: tokens a step (Q1173; 2026-11-10)

What this is: how many tokens one step of the main model yields on the Miner Llama v3 release preset
([release/flash-next-v3/settings.env](../../../release/flash-next-v3/settings.env): Qwen3.8-Flash-Next, Swift Q2_0,
an MTP draft of three tokens) with two switches, each against the preset:

- **`pv1`, probabilistic verification** (`--spec-draft-sampling probabilistic --spec-draft-p-min 0`): the draft
  samples its tokens instead of taking its first choice, and the main model accepts a drafted token with probability
  min(1, p/q) instead of only when its own sample is the same token;
- **`kd1`, kept draws** (`LLAMA_SPEC_KEEP_DRAWS=1`): after a step that rolls back through a checkpoint, the random
  draws that step used stay used.

`pv0` is the preset. The 66 recorded requests of the release check's coding-agent session (arm `t1swift`, 12 bug-fix
tasks) were re-sent with [bench/agent_check/replay.py](../../../bench/agent_check/replay.py): the same prompts, the
same seed per request and the recorded reply length as the limit, to two loads of each arm, alternated
(`pv0 pv1 kd1 pv0 pv1 kd1`). Registry row Q1173 in [quants/experiments.md](../../../quants/experiments.md), the
acceptance half of Q731 and Q1160.

Why: on another fork's MTP line for this model, greedy drafts cost 5 points of acceptance and about 7 % of speed
against probabilistic ones ([note](../../research/notes/qwen_v100_llamacpp.md)). Acceptance and tokens a step depend
on the two models, the sampler and the text, not on the card, so they were counted on a machine with newer cards.
No time is read or reported here. What a step costs in time under each switch is a rig measurement (Q731).

## Result

**Probabilistic verification is a gain in tokens a step: 3.283 and 3.304 in its two loads against 3.043 and 3.072
for the preset, +0.237 (+7.74 %; 95 % interval +0.164 to +0.292), with 0.7663 and 0.7731 of the drafted tokens
accepted against 0.6892 and 0.6993. The preset's own two loads are 0.029 apart. Kept draws change nothing on this
line: 3.052 and 3.062, −0.000 against the preset, and no step of any load rolled back through a checkpoint, so the
switch's code never ran.**

Tokens a step is generated tokens over steps; a step is one pass of the main model that yielded tokens (generated
minus accepted draft tokens). Pooled means the counters are summed over the 66 requests before dividing.

### Every load

| Arm | Load | Generated | Accepted of drafted | Share accepted | Steps | Tokens a step (pooled) | Tokens a step (mean of the 66 requests) |
|---|---|---:|---|---:|---:|---:|---:|
| `pv0` | 1 | 17,410 | 11,688 of 16,958 | 0.6892 | 5,722 | 3.043 | 3.127 |
| `pv0` | 2 | 17,359 | 11,708 of 16,743 | 0.6993 | 5,651 | 3.072 | 3.143 |
| `pv1` | 1 | 18,788 | 13,066 of 17,050 | 0.7663 | 5,722 | 3.283 | 3.344 |
| `pv1` | 2 | 18,784 | 13,099 of 16,944 | 0.7731 | 5,685 | 3.304 | 3.369 |
| `kd1` | 1 | 16,762 | 11,270 of 16,266 | 0.6929 | 5,492 | 3.052 | 3.116 |
| `kd1` | 2 | 17,140 | 11,542 of 16,576 | 0.6963 | 5,598 | 3.062 | 3.134 |
| the recorded session (the rig, live) | — | 27,067 | 18,447 of 25,704 | 0.7177 | 8,620 | 3.140 | 3.191 |

A replayed reply ends at the model's own stop or at the recorded reply's length, whichever comes first, so a load
generates fewer tokens than the recorded session did (30 and 32 of `pv0`'s 66 replies reached the limit).
[loads.tsv](loads.tsv); every request in every load in [requests.tsv](requests.tsv).

### Each switch against the preset

An arm's value is the mean of its two loads. The spread is the difference between an arm's two loads. Class: gain
when the arm is above the preset by more than the spread, loss when below by more than it, negligible otherwise;
the class is the same against the preset's spread alone and against the larger of the two arms' spreads. The
intervals are 95 % bootstrap intervals of the difference: over requests (each request keeps its rows in all loads, so
the comparison stays paired) and over whole sessions (the 12 tasks). [compare.tsv](compare.tsv).

| Switch | Metric | Its loads | The preset's loads | Difference | % | Its spread | The preset's spread | Class | 95 % (requests) | 95 % (sessions) |
|---|---|---|---|---:|---:|---:|---:|---|---|---|
| `pv1` | tokens a step | 3.283, 3.304 | 3.043, 3.072 | +0.237 | +7.74 | 0.021 | 0.029 | **gain** | +0.164 to +0.292 | +0.160 to +0.288 |
| `pv1` | share accepted | 0.7663, 0.7731 | 0.6892, 0.6993 | +0.0755 | +10.87 | 0.0067 | 0.0100 | **gain** | +0.0512 to +0.0936 | +0.0502 to +0.0920 |
| `kd1` | tokens a step | 3.052, 3.062 | 3.043, 3.072 | −0.000 | −0.01 | 0.010 | 0.029 | **negligible** | −0.029 to +0.036 | −0.029 to +0.035 |
| `kd1` | share accepted | 0.6929, 0.6963 | 0.6892, 0.6993 | +0.0003 | +0.05 | 0.0035 | 0.0100 | **negligible** | −0.0096 to +0.0132 | −0.0096 to +0.0128 |

Per request (each request's tokens a step averaged over its arm's two loads, then the mean difference over the 66
requests): `pv1` +0.222 (95 % +0.176 to +0.267), higher in 58 requests, lower in 6, equal in 2; `kd1` −0.010 (−0.040
to +0.019), higher in 17, lower in 16, equal in 33.

### Where the gain is: steps by accepted tokens

Verify steps of three drafted tokens, by how many the main model kept ([steps.tsv](steps.tsv); the few shorter
drafts at a reply's end are in the file):

| Load | None | One | Two | All three |
|---|---:|---:|---:|---:|
| `pv0` 1 | 1,029 | 766 | 650 | 3,198 |
| `pv0` 2 | 963 | 749 | 648 | 3,212 |
| `pv1` 1 | 675 | 684 | 590 | 3,728 |
| `pv1` 2 | 656 | 657 | 562 | 3,767 |
| `kd1` 1 | 970 | 739 | 608 | 3,096 |
| `kd1` 2 | 950 | 797 | 587 | 3,180 |

With probabilistic verification about a third fewer steps keep nothing (675 and 656 against 1,029 and 963), and
more keep the whole draft (3,728 and 3,767 against 3,198 and 3,212).

### How far two loads of one arm differ

- **In the counters:** tokens a step differs by 0.029 between `pv0`'s loads (+0.96 %; 95 % interval of the
  difference −0.043 to +0.103), by 0.010 between `kd1`'s and by 0.021 between `pv1`'s; the share accepted by 0.0100,
  0.0035 and 0.0067.
- **In the text:** with the same prompts and seeds, `pv0`'s two loads wrote the same reply in 37 of 66 requests,
  `kd1`'s in 35 and `pv1`'s in 35 ([same_reply.tsv](same_reply.tsv)). A fixed seed does not give two loads the same
  text on this line, so a replay comparison needs repeated loads and their spread.
- **The differences go by session, not by request.** Between `pv0`'s loads 7 of the 12 sessions are the same in
  every request, 4 differ from their first request and 1 from its second; after a session's first differing reply
  none of its later 24 replies is the same again ([same_reply_sessions.tsv](same_reply_sessions.tsv); 1 of 26
  between `pv1`'s loads, 0 of 25 between `kd1`'s). A replayed request's prompt is the recorded history, not the
  replay's own earlier replies, so what carries the difference through a session is the server's state, not the
  prompt.
- **A load's very first request already differs.** It reads 2,780 prompt tokens into an empty cache: `pv0` 1,
  `kd1` 1 and `kd1` 2 wrote the same 73 tokens, `pv0` 2 wrote 49 others. Prompt depth does not sort the
  differences: the 6 requests with 8,192 or more prompt tokens are the same in both `pv0` loads.
- **`kd1`'s loads are two more loads of the preset.** A `kd1` load and a `pv0` load wrote the same reply in 46, 35,
  38 and 36 of 66 requests, which is what `pv0`'s own two loads do (37). `pv1` writes other text by design: 2 of 66
  replies equal a `pv0` load's.

Why two loads of one line write different text is registry row Q575; the release line's replies under greedy
decoding are in [2026-11-10_draft_line_exactness](../2026-11-10_draft_line_exactness/README.md).

### The server logs

Every arm ran with `LLAMA_TRACE=1`, which prints one line per verify step and changes no computation
([log_check.tsv](log_check.tsv)).

| Load | Requests with a grammar | Their sampled tokens, drawn on the device | Rejected by the grammar after a device draw | Verify steps | Steps that restored a checkpoint | Checkpoint restores between requests | Error lines |
|---|---:|---|---:|---:|---:|---:|---:|
| `pv0` 1 | 66 | 17,420 of 17,420 | 1 | 5,661 | 0 | 65 | 0 |
| `pv0` 2 | 66 | 17,368 of 17,368 | 1 | 5,590 | 0 | 65 | 0 |
| `pv1` 1 | 66 | 18,833 of 18,833 | 14 | 5,691 | 0 | 64 | 0 |
| `pv1` 2 | 66 | 18,828 of 18,828 | 14 | 5,655 | 0 | 64 | 0 |
| `kd1` 1 | 66 | 16,772 of 16,772 | 1 | 5,431 | 0 | 65 | 0 |
| `kd1` 2 | 66 | 17,151 of 17,151 | 1 | 5,536 | 0 | 65 | 0 |

- **Sampling stays on the device in every arm:** each of the 66 requests carries the tools' grammar, and every
  token the main model's sampler drew was drawn on the device, with probabilistic verification as without it.
- **No step restored a checkpoint.** The drafts are three tokens, within the target's rollback snapshots, so a
  partly accepted draft is rolled back without a checkpoint. `kd1` acts only in a step that restores one; its
  server line reports the switch on and its branch was never reached.
- The checkpoint restores between requests are the replay's own: a replayed reply differs from the recorded one, so
  the next request restores the checkpoint in front of it.
- The accepted and drafted tokens add up to the same numbers three ways: over the step lines, over the server's
  per-request lines and over the `timings` the replay keeps.
- With probabilistic verification the grammar rejected 14 device draws a load, against 1.

**What it decides.**
- **Q731 (probabilistic verification on this line):** the rig measurement keeps one pair, the preset against
  `pv1`, for the time of a step. If a step costs the same, +7.74 % tokens a step is the same gain in generation
  speed (est.). By the source, the accept test of rejection sampling runs on the host, on the candidate list the
  sampler leaves, so the time of a step is the open half.
- **Q1160 (kept draws):** no cost and no effect on the preset, and it leaves the rig run. It matters for drafts that
  outgrow the rollback snapshots (four or more tokens, lookup drafts), where a step restores a checkpoint.

**Limits.**
- **Counts, not time.** A step with probabilistic verification may cost more or less than a preset step on the rig.
- **One session's requests:** 66 requests of 12 bug-fix tasks, prompts of 1,952 to 15,137 tokens (median 4,643).
  The requests of one task share their text; the interval over sessions allows for that and is about as wide.
- **Two loads an arm.** The spread is one difference per arm. The gain is eight times the preset's spread, and each
  `pv1` load is above each `pv0` load.
- **The text differs between arms and loads** (see above), so "paired by request" pairs the prompt, the seed and the
  reply limit, not the reply.
- **Quality is not measured here.** Rejection sampling keeps the main model's distribution in exact arithmetic;
  whether the tasks still pass is the agent check's question.
- One model file, one draft, one draft length, the sm_61 build's compute 6.1 PTX on newer cards.

## Method

- **The line:** the release preset without its nine-card split (`-ts`), the draft's device override (`-otd`) and the
  keep-alive variable, on three RTX 3090 cards, llama-server choosing the split. The build
  (`pascal-p106-v3-b12076-ladder5`, `1551f5fe1`) holds only sm_61 code and compute 6.1 PTX, so the newer cards'
  driver compiles the PTX; `CUDA_FORCE_PTX_JIT` is not set. After loading, the cards held 13,938, 13,162 and
  15,042 MiB, the same in all six loads.
- **The arms:** `pv0` the line above; `pv1` adds `--spec-draft-sampling probabilistic --spec-draft-p-min 0`; `kd1`
  adds the variable `LLAMA_SPEC_KEEP_DRAWS=1`. Six loads in the order `pv0 pv1 kd1 pv0 pv1 kd1`, one llama-server
  process per load.
- **The requests:** the release check's recorded session `t1swift`
  ([2026-11-05_release_flashnext](../2026-11-05_release_flashnext/README.md)): 66 chat requests in 12 sessions, sent
  in their recorded order, one at a time, streamed as recorded, with their tools and sampling fields (temperature 1,
  top-p 0.95, top-k 20). `replay.py` sets `max_tokens` to the recorded reply's length and a seed fixed per session
  and request, the same in every load. The cache carries over from request to request as it did live; no slot is
  erased.
- **The counters:** `draft_n`, `draft_n_accepted` and `predicted_n` of the server's `timings` per request, which
  `replay.py` now keeps in each row together with an MD5 of the assembled reply.
- **The tables:** [run/pv_table.py](run/pv_table.py); 10,000 bootstrap resamples, seed 1.

## Files

| File | Holds |
|---|---|
| [summary.md](summary.md) | every table of `pv_table.py` as Markdown |
| [loads.tsv](loads.tsv) | per load: generated, drafted, accepted, share accepted, steps, tokens a step; the recorded session's row |
| [compare.tsv](compare.tsv) | each switch against the preset and each arm's second load against its first: every load by value, difference, spreads, class, intervals, per-request counts |
| [requests.tsv](requests.tsv) | per request and load: prompt tokens, reply limit, generated, drafted, accepted, steps, tokens a step, tokens drawn on the device, grammar counters, finish reason, the first 12 characters of the reply's MD5 |
| [same_reply.tsv](same_reply.tsv) | per pair of loads: requests with the same reply and with the same counters, below and from 8,192 prompt tokens |
| [same_reply_sessions.tsv](same_reply_sessions.tsv) | per pair of loads and session: requests with the same reply, the first that differs, how many after it are the same again |
| [steps.tsv](steps.tsv) | per load: verify steps by drafted and accepted tokens |
| [log_check.tsv](log_check.tsv) | per load, from the server log: sampling on the device, grammar rejections, steps, restores, error lines, and the three sums of accepted and drafted tokens |
| [run/run_prob_verify.sh](run/run_prob_verify.sh) | the six loads: one llama-server container per load, the replay, a check that each load is whole |
| [run/pv_table.py](run/pv_table.py) | the tables |

The tasks are the bug-fix set of [bench/agent_check](../../../bench/agent_check/README.md) (`tasks/bugfix`); the
tables name them by task id and hold no prompt or reply text.

## Commands

On the machine with the cards. `WORK` is a work folder, `REPO` a checkout of this repository, `BIN` the build's
`llama-server`, `MODEL` the first shard of Swift Q2_0 (the other two beside it), `MODEL_DRAFT` the MTP draft, `LOG`
the recorded `proxy.jsonl` with its `proxy.jsonl.bodies.gz` beside it, `GPUS` the three cards as Docker names them,
`MOUNTS` the folders the container reads.

```bash
cd "$WORK" && WORK=$WORK REPO=$REPO BIN=$BIN MODEL=$MODEL MODEL_DRAFT=$MODEL_DRAFT LOG=$LOG GPUS=$GPUS \
  MOUNTS="$MOUNTS" DOCKER_ARGS="--label pascal-quants=1 --cpu-shares=256" NAME=prob-verify PORT=18143 \
  setsid -f nohup nice -n 19 bash "$REPO/docs/data/2026-11-10_prob_verify_acceptance/run/run_prob_verify.sh" \
  > console.txt 2>&1 < /dev/null
```

It is done when `$WORK/done` exists (`$WORK/failed` otherwise); the six loads ended 43 minutes after the start.
Then:

```bash
nice -n 19 python3 -I "$REPO/docs/data/2026-11-10_prob_verify_acceptance/run/pv_table.py" "$WORK" --out "$WORK/tables"
```

## Provenance

- The requests are the release check's own recording of this project's agent check; the tables hold counters and
  hashes only.
- The model is UkisAI's Swift-1.5 GSQ-RCO Q2_0 of Qwen3.8-Flash-Next with its MTP draft, as in the release preset.
- Raw output (server logs, the replay's rows with their times, the cards' memory) is not published.
