# The MTP draft's passes on one card (Q961, 2026-11-04)

What this measured: what Flash-Next's Gyro MTP draft pays per pass for running on two cards and for sending its
hidden row through the host, and two switches that keep a step's drafting passes on one card without changing a
single drafted token. The question ([Q961](../../../quants/experiments.md)) came from C338's split of a speculative
step ([2026-11-04_verify_host](../2026-11-04_verify_host/README.md)). With S15's `-otd token_embd=CUDA4`, the draft's
token embedding table sits on card 4 and its layer and head sit on card 8. Every draft pass evaluates card 4 for one
row and sends that row to card 8 through the host, since the rig's cards have no peer-to-peer. It also copies its
hidden row to the host, where the next pass uploads it again, and it waits for its sampled token before the next pass
is queued.

## The cost, from C338's host profile

`run/draft_passes.py` splits the draft's evaluations of the C338 profile (S34's MTP line, pmix-q2-fast on `-ladder3`,
the draft's full head, n-max 3, one short request: 186 steps) into four passes per step: the catch-up pass over the
verified tokens (pass 0) and the three drafting passes. Medians in ms ([c338_passes.md](c338_passes.md)):

| Pass | crossing (card 4's exit to card 8's entry) | GPU time | host wait after the pass | host time after the pass's GPU work |
|---|---:|---:|---:|---:|
| 0, catch-up | 1.56 | 3.63 | 1.77 | overlapped |
| 1, first draft | 2.42 (~0.7 of it the crossing, the rest waits for pass 0 on card 8) | 7.06 | 7.48 | 0.55 |
| 2 | 0.60 | 7.06 | 7.49 | 0.51 |
| 3 | 0.59 | 7.06 | 14.50 (the server's work before the next verify pass) | 7.03 |

- **The draft's 22 ms host span in C338 is mostly card 8's GPU time** (24.8 ms summed over a step's passes). What
  stays on the host is ~0.6 ms of crossing and ~0.5 ms of turnaround per drafting pass.
- **Most of those bytes are the hidden row, not the token row.** Flash-Next's MTP layer reads the main model's
  residual stream as its h: 4 hyper-connection streams × 2,560 = 10,240 F32, 40 KB a row, against 10 KB for the token
  row. Each drafting pass downloads its h row and the next pass uploads it again: 200 KB a step over a PCIe 1.0 x1
  link that moves a 160 KB hop in ~1.2 ms ([2026-11-04_megakernel_rig](../2026-11-04_megakernel_rig/README.md)).
- **Est. ~2-2.5 ms of a 179 ms step for the two switches together** (~1.3 %): the crossing of passes 2 and 3 (their
  token row and h upload) and part of pass 1's, plus the h downloads after passes 1-3.
- **What chaining the passes on the device would add** (the next pass's token taken from the previous pass's
  device-side sample, no host round trip): the rest of the turnaround, under ~0.3 ms a pass once the h row stays on
  the card. Pass 3's 7 ms after its GPU work is the server's accept and the next verify batch (Q960).

## The two switches

Fork branch `pascal-p106-v3-b12076-draft1card` (two commits on `-kshape` `d27845b67`), both off by default.

**`LLAMA_DRAFT_EMBD_CACHE=<rows>[,<max tokens>]`: the token rows on the layer's card.**
- The draft context gets a cache of rows of its own token embedding table, in the table's type and bytes (Q8_0 for
  the Gyro draft: 2,720 bytes a row, 21 MiB for 8,192 rows), in a buffer of the card that holds the MTP layer.
- Every micro-batch of at most `max tokens` tokens (default 16: the drafting passes and the catch-up over a verify
  step's tokens) maps its tokens to slots before its graph is chosen. The MTP graph then reads `get_rows` over the
  cache, with the slots as its token input. Larger batches (the prompt) read the table on card 4 as before.
- A token the cache does not hold takes a free slot, or else the slot read longest ago (never one the same batch
  reads). The host reads its row from the table and writes it into the slot once the layer's card has finished the
  work queued on it. In agentic text a few thousand distinct tokens cover almost every pass: Q636 counted 805 in 8,626
  tokens of the agent check's assistant turns.
- `get_rows` over the same bytes on the same kind of card gives the same numbers, so the drafts are the same as
  without the cache. The card must have a `get_rows` for the table's type (the fork's CUDA backend reads every type the
  rig's files use); otherwise the switch logs that it is off. The draft logs the cache's counts every 1,000 rounds and
  at the end: `draft embedding row cache: N token rows read from it, M of them copied from the table first; K batches
  read the table`.

**`LLAMA_DRAFT_H_CARRY=1`: the hidden rows stay on the layer's card between a step's passes.**
- The draft context gets one row per sequence on the layer's card (F32, 40 KB). Every drafting pass writes its
  output's h there (`set_rows` after the layer), and the decode copies no h row to the host.
- Every drafting pass after a step's first reads its h from that row (`get_rows`) instead of an uploaded input. The
  first still takes the main model's row from the host, and the catch-up passes run as before.
- The carried row is the previous pass's output, copied on the device: the same F32 values that the host round trip
  delivered, so the drafts do not change. It is off for drafts with chained heads or on the main model's memory, which
  read the rows on the host.

With both switches, a drafting pass after the first sends the card only its slot index, position and mask, and reads
back only its sampled candidates (80 bytes).

The other ways out:
- **S31's shared embeddings** (`LLAMA_DRAFT_SHARE=embd,head`, [2026-11-04_draft_mem](../2026-11-04_draft_mem/README.md)):
  the draft reads the main model's token embeddings, which live in host memory. The row is looked up on the host and
  sent to card 8 with the pass's other inputs, so card 4 drops out here too, and card 4 holds 644 MiB less. But the
  draft computes with the main model's embedding (cosine 0.988 to its own), so its tokens can change. No code: S37
  measures it as an arm.
- **The passes chained on the device**: Q969, parked on the ceiling above.
- **The main model's hidden rows to the draft without the host** (Q970, planned): the verify pass copies its h rows
  (40 KB a token) to the host for the draft's catch-up, which uploads them again, and both live on card 8. Est.
  ~1.8 ms a step, and on a prompt 20 MB each way per 512-token micro-batch.

## CPU check

`run/cpu_check.sh` on the build machine (Docker, CPU backend, no GPU): the unit test, then the real files (Swift
Qwen3.8-Flash-Next GSQ-RCO Q2_0 and the Gyro MTP draft) on the CPU, one load per arm, the same three requests.

**Every switch arm drafts what the same build drafts without the switches:** 15 of 15 replies carry the same tokens
and draft counts. The arms are the cache, a 16-row cache that replaces rows all the time, the carry, both switches,
and both with the shortlist head, on two greedy requests and one sampled request
([cpu_check.md](cpu_check.md), [cpu_check.tsv](cpu_check.tsv)). The unit test passes 10,885 checks: the cache holds
the table's bytes, and `get_rows` over it gives the table's numbers exactly for F32, Q8_0 and Q3_K. A CPU check runs
both switches' code paths, but the rows stay in host memory there; the card-side copies are S37's, which compares a
greedy request's drafts with the switches off and on (`run/identity.py`).

On this build no drafting arm's greedy reply equals plain decoding's, `base` included. From token 20-24 on, the 4-token
verify batch and the single-token decodes part at a near-tie (Q971). That is why the check compares the switch arms
with `base`, not with plain decoding.

## The rig run (S37)

`run/rig_d1.sh` runs on S35's line ([2026-11-04_s35_stack](../2026-11-04_s35_stack/README.md): pmix-q2-fast, the
k-column setting, the Q8_0 hop cast, the MTP draft with the shortlist head, `-otd token_embd=CUDA4`). It uses this
build with the switches off (`d1off`), each switch alone (`d1cache`, `d1carry`), both (`d1on`) and S31's shared
embeddings (`d1share`). Each arm gets one profiled load with `GGML_CUDA_PROFILE_HOST`: the agent's first-turn
request, a greedy request (`run/greedy_request.json`) and S12's smoke request. The build machine then splits the profile with
`run/draft_passes.py` and C338's `host_steps.py`, and checks with `run/identity.py` that the switch arms' greedy replies
carry `d1off`'s text and draft counts. Then the agent check runs for `d1off` and `d1on`. Expected: `d1on`'s
drafting passes show no crossing, its step is ~2 ms shorter, and its acceptance is the same. `d1share` should lose the crossing too, with its acceptance within noise.

## Files

| File | What it is |
|---|---|
| [c338_passes.md](c338_passes.md) | the draft's passes per step in C338's profile: crossing, GPU time, wait and turnaround per pass; per step the draft's host span, GPU time and the crossings' sum |
| [run/draft_passes.py](run/draft_passes.py) | the split, from a `GGML_CUDA_PROFILE_HOST` file |
| [cpu_check.md](cpu_check.md) | the CPU check: verdicts and the replies per arm |
| [cpu_check.tsv](cpu_check.tsv) | per arm and request: tokens, drafted, accepted, md5 of the token ids |
| [run/cpu_check.sh](run/cpu_check.sh) | the CPU check: build, unit test, the real files with the switches off and on (`STEPS=verdicts` recomputes the verdicts from a run's outputs) |
| [run/rig_d1.sh](run/rig_d1.sh) | the rig run: the build to the rig, the profiled loads and their split, the drafts' identity on the cards, the agent check |
| [run/greedy_request.json](run/greedy_request.json) | the greedy chat request of each profiled load |
| [run/identity.py](run/identity.py) | whether replies to the greedy request carry the reference's text and draft counts |

## Commands

```bash
# the split of a host profile (any machine with python3)
python3 -I docs/data/2026-11-04_draft_one_card/run/draft_passes.py host_profile.csv > passes.md
# the CPU check (build machine; WORK/src holds the branch)
WORK=<dir> MODEL=<Swift Q2_0 first shard> DRAFT=<mtp-Qwen3.8-Flash-Next-draft.gguf> LOCK=<build lock> JOBS=8 THREADS=8 \
  bash docs/data/2026-11-04_draft_one_card/run/cpu_check.sh
# the rig run (machine values in a private file; see the script's header)
T=<build folder> MD5=<tarball md5> ... bash docs/data/2026-11-04_draft_one_card/run/rig_d1.sh <out>
```

## Provenance

The host profile is C338's (rig 2, 2026-11-04, `docs/data/2026-11-04_verify_host`); the split's tables summarize it.
The CPU check's requests are the two prompts of [2026-11-04_draft_mem](../2026-11-04_draft_mem/README.md); the models
are Swift's GSQ-RCO Q2_0 of Qwen3.8-Flash-Next and the Gyro MTP draft, not republished.
