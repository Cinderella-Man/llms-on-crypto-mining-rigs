# The taper's `cuts=1` on three RTX 3090s: Flash-Next's v5 preset (C956, Q1687 point 10; 2026-11-23)

What this is: the GPU half of the check of `cuts=1`, a new key in the prompt taper's setting
([C905](../2026-11-21_c905_taper_cuts/README.md), registry row Q1638 in
[quants/experiments.md](../../../quants/experiments.md); fork branch `pascal-p106-v5-taper-cuts` `1720f26d3`, two
commits on `pascal-p106-v5` `3dfad876f`). Flash-Next's v5 preset reads a prompt call of up to 1,400 tokens in
micro-batches of 64 (`--ubatch-taper short=1400:64`), and the server also cuts each call at its two in-call
checkpoints, 516 and 4 tokens before the call's end. Where the first cut lands a few tokens short of a planned
micro-batch end, a piece of 4 to 6 tokens is left, which runs the slower matrix-vector kernels. With `cuts=1` the
taper plans each stretch between the cuts on its own. C905 checked it on a processor; this run puts the real preset
file on a build machine's three RTX 3090s, which run the rig build's compute 6.1 PTX. It is one point of registry row
Q1687, the GPU half of the branches built on the night of 2026-11-20/21. Correctness only: nothing here is timed.

## Result in plain words

- **On the cards, `cuts=1` leaves no piece under 9 tokens beside a cut.** Without it the server cut the same five
  calls as on the rig (577, 583, 775, 896 and 1,086 tokens: a piece of 4 to 6 tokens). With it, none of 218 prompt
  calls had such a piece, and every call was cut exactly as C905's processor check planned.
- **The cards cut every call the way the processor check and the rig did.** Without `cuts=1` all 52 call sizes the
  rig logged under `short=1400:64` (R225) and the 20 it logged under `short=2048:64` (C222) came out piece for piece
  as the rig's. With or without it, 153 of 157 distinct setting-and-size pairs came out as the processor check's; the
  other four are one size that check did not run (a 601-token continuation) under each of the four settings.
- **Nothing changes unless `cuts=1` is set.** The branch gave the base's replies, top-3 log-probabilities, counters
  and pieces bit for bit on every request, on both settings and on the replay.
- **The replies stay the same wherever the pieces stay the same.** The replay of the agent check's 62 turns gave
  the same 8-token replies on all 73 requests with and without `cuts=1`, in two pairs of passes, including the 14
  requests whose pieces it changes. On the sized prompts, every request whose pieces `cuts=1` leaves alone gave the
  same 64-token reply and log-probabilities bit for bit. Setting it on the server and sending it per request gave
  the same result in every field but one draft count (the last point).
- **Where `cuts=1` changes the pieces, a 64-token greedy reply can part at a near-tie, as it does for any change of
  pieces.** Of 17 sized requests whose pieces changed, 16 replies parted from the control's, at tokens 0 to 56. The
  control already does the same when only the existing setting changes: `short=2048:64` against `short=1400:64`,
  neither with `cuts=1`, cuts the 1,534- and 1,985-token calls differently, and those replies part at tokens 5 and
  11, by the same amount. On a P106, reading a call in pieces of 64 instead of 512 changes Flash-Next's top token at
  3.1 % of positions (FN95, Q1493), so a 64-token reply is expected to move whenever the pieces move. This check's
  criterion, "the same replies", could therefore not hold for any change of pieces. It holds wherever the pieces are
  the same, and that is the rule applied below. Whether `cuts=1` changes quality at all is its own row, Q1714.
- **The fork's Pascal kernels ran.** On a 3090 they stay off unless forced, so they were forced on with a P106's
  launch shapes, as [C953](../2026-11-23_c953_ornith_kcol_gpu/README.md) found is needed: the matrix-vector kernels
  (`GGML_CUDA_MMVQ_PASCAL=2`, `GGML_CUDA_MMVF_PASCAL=2`; every server listed `MMVQ_PASCAL = 1` and `MMVF_PASCAL = 1`
  among its features) and the sparse attention (`GGML_CUDA_FATTN_SPARSE_PASCAL=2` in place of the preset's 1, which
  turns it off on cards from Turing on; it logs no line of its own). The tiny pieces `cuts=1` removes run those
  matrix-vector kernels on a P106.
- **Seen in passing:** each load's first request drafts differently from the same request later on a load (the
  30-token prompt: 66 drafted, 40 accepted, against 63 and 41), with the same reply, in all three loads (Q1715).

## What ran

Flash-Next's v5 preset ([release/flash-next-v5/settings.env](../../../release/flash-next-v5/settings.env)) as it is,
on Swift GSQ-RCO Q2_0 (the preset's file, its n-gram table in a third shard), with two changes: the layer split over
the three cards (`-ts 1,1,1`) and the log at verbosity 4 (`-lv 4`, where the server's `ubatch taper:` lines and
checkpoint lines show). The draft was Swift-1.5's own MTP head (the preset's draft file is on the rigs only), as in
[C951](../2026-11-21_c951_flash_head_gpu/README.md) and [C952](../2026-11-21_c952_kv_seq_rm_gpu/README.md): `-b 4096
-ub 512`, 4 checkpoints taken inside the calls, the MTP draft of three. Three loads, one at a time on GPUs 0–2
([run/run.sh](run/run.sh)):

| Load | Build | Server's taper | Sized-prompt passes | Replay passes |
|---|---|---|---|---|
| base | `3dfad876f` (the base) | `short=1400:64` | srv, s2048 | 0 (warm-up), 1: `short=1400:64` |
| head | `1720f26d3` (the branch) | `short=1400:64` | srv, v5c, s2048, s2048c, srv2 | 0 (warm-up), 1, 4: `short=1400:64`; 2, 3: `short=1400:64,cuts=1` |
| headc | `1720f26d3` | `short=1400:64,cuts=1` | srv | - |

Passes: srv and srv2 leave the taper to the server; v5c sends `short=1400:64,cuts=1` with each request; s2048 and
s2048c send `short=2048:64` without and with `cuts=1`, the setting under which C222 logged the 1,534- and
1,985-token calls' tiny pieces.

- **Sized prompts** ([run/client.py](run/client.py)): 24 requests a pass, greedy, 64 reply tokens with their top-3
  log-probabilities. Fresh prompts of 30, 302, 516, 520, 569, 577, 583, 599, 610, 775, 803, 815, 896, 936, 1,086,
  1,130, 1,300, 1,359, 1,400, 1,534, 1,985 and 3,313 tokens (C905's server check's sizes and the 14 call sizes
  `cuts=1` changes under `short=1400:64`), each a stretch of text tokens from the READMEs of C905, C222 and R225
  with a first token no other prompt has, so each is read whole in one call; a continuation that reads the 577-token
  prompt's reply and 600 new tokens on top of the cache; an edit at token 1,100 of the 1,534-token prompt, which
  restores the checkpoint the server took inside that call.
- **Replay** ([taper_replay.py](../2026-11-04_prompt_taper/run/taper_replay.py), as C222 and R225 ran it on a rig):
  the agent check's 12 tasks and 62 turns, each turn's new tokens on top of the task's history, 8 greedy tokens a
  turn, 73 requests a pass with the tasks' prefix requests. Pass 0 is a warm-up (its first turn goes out with nothing
  cached).
- **Comparison** ([run/compare.py](run/compare.py)): every `ubatch taper:` line of the server log is matched to its
  request; pairs of passes compared field by field; every call's pieces checked against the floor and against
  C905's processor plan check ([plans-rs3.tsv](../2026-11-21_c905_taper_cuts/plans-rs3.tsv)) and, without `cuts=1`,
  the rig's logged pieces ([plans.tsv](../2026-11-21_c905_taper_cuts/plans.tsv)).
  [run/divergence.py](run/divergence.py) finds where two passes' replies part and by how much.

## The pieces

[calls.tsv](calls.tsv) lists every distinct call the cards logged: setting, size, the pieces, the processor plan
check's and the rig's.

| Setting | Call sizes logged | The processor check's pieces | The rig's pieces | Calls with a piece under 9 beside a cut |
|---|---:|---:|---:|---|
| `short=1400:64` | 56 | 55 the same, 1 not run | 52 of 52 the same | 5: 577 (5@61), 583 (4@63), 775 (4@255), 896 (5@380), 1,086 (6@570) |
| `short=1400:64,cuts=1` | 55 | 54 the same, 1 not run | (not run there) | 0 |
| `short=2048:64` | 23 | 22 the same, 1 not run | 20 of 20 the same | 7: the five above, 1,534 (6@1018), 1,985 (5@1469) |
| `short=2048:64,cuts=1` | 23 | 22 the same, 1 not run | (not run there) | 0 |

The size neither check ran is the continuation's 601-token call: `64 21 43 64x7 21 4` without `cuts=1`, `64 21 64x8 4`
with it. The v5 preset's 14 calls change as C905 planned, each by one micro-batch fewer: for example 577 tokens from
`61 5 62 64x6 61 4` to `61 64x8 4`, 1,086 from `64x8 58 6 64x7 58 4` to `64x8 58 64x8 4`. The sized prompts add
520 (`4 60 64x6 68 4` to `4 64x8 4`) and 1,400 (`64x13 52 12 64x7 52 4` to `64x13 52 64x8 4`).

## The comparisons ([verdicts.tsv](verdicts.tsv))

| Pair | Rule | Result |
|---|---|---|
| sizes: branch against base, `short=1400:64` and `short=2048:64` | same | 24 of 24 the same in every field, both settings |
| sizes: srv2 against srv (one load) | same | 24 of 24 the same (the 30-token request's draft counts aside: the load's first request, Q1715) |
| sizes: `cuts=1` on the server against per request | same | 24 of 24 the same (the same aside) |
| sizes: `short=1400:64,cuts=1` against `short=1400:64` | split | counters 24 of 24 the same; pieces the same in 7, all 7 bit for bit; pieces changed in 17, replies parted in 16 |
| sizes: `short=2048:64,cuts=1` against `short=2048:64` | split | counters 24 of 24 the same; pieces the same in 4, all bit for bit; pieces changed in 20 (the 17 and the 1,534-token prompt, its edit, the 1,985-token one), replies parted in 17 |
| sizes, the control: `short=2048:64` against `short=1400:64`, no `cuts=1` | split | pieces the same in 21, all bit for bit; changed in 3 (1,534, its edit, 1,985), replies parted in 2 |
| replay: branch against base, passes 0 and 1 | same | 73 of 73 the same, both passes |
| replay: pass 4 against 1, pass 3 against 2 | same | 73 of 73 the same, both |
| replay: `cuts=1` (passes 2, 3) against `short=1400:64` (1, 4) | replies and counters | 73 of 73 the same replies and counters in both pairs; pieces changed in 14 requests |
| every pass with `cuts=1` (sizes v5c, s2048c, headc; replay 2, 3) | floor | no piece under 9 beside a cut; every call the processor check's plan |
| every pass without it | plans | every call the processor check's plan, and the rig's where the rig logged that size |

"Same" means every reply token, top-3 log-probability, counter (prompt tokens read, cached tokens used, checkpoints
made and restored) and piece. "Split": every counter the same; where a request's call, and the call that built the
cache it reads on, is cut into the same pieces in both passes, the reply and its log-probabilities bit for bit; where
the pieces differ, the reply is listed and sized, not judged. The drafted and accepted counts are listed, not judged.
Verdict: **PASS** on these rules.

**Where the replies part** ([divergence.tsv](divergence.tsv)), `short=1400:64,cuts=1` against `short=1400:64`: the
first differing token of the 16 is 6 (520), 40 (569), 35 (577), 22 (the continuation), 1 (583), 0 (599), 18 (610),
30 (803), 10 (815), 38 (896), 7 (936), 50 (1,086), 19 (1,130), 27 (1,300), 0 (1,359) and 56 (1,400); the 775-token
prompt's 64 tokens stayed the same. In the `short=1400:64` reply at that token, the two candidates were apart in
log-probability by 0.0018, 0.0022, 0.0082, 0.026, 0.026, 0.039, 0.047, 0.053, 0.064, 0.14, 0.20, 0.21, 0.28 and
0.31, and twice the other token was outside its top 3. Up to the parting token, the largest move of a top-3
log-probability in a request was between 0.12 and 2.43 (the 1,359-token reply parts at its first token, whose top 3
differ). The two settings without `cuts=1` move the same way: the 1,534-token reply parts at token 5 (0.026 apart)
and the 1,985-token one at 11 (0.20 apart), their first tokens' top-3 log-probabilities by up to 0.050 and 0.85; the
edit of the 1,534-token prompt keeps its 64 tokens with log-probabilities up to 0.43 apart.

## Files

| File | What |
|---|---|
| [verdicts.tsv](verdicts.tsv) | every check: the pairs, the floor and plans per pass, the verdict |
| [calls.tsv](calls.tsv) | every distinct call the cards logged, against the processor check's plan and the rig's pieces |
| [sizes.tsv](sizes.tsv) | every sized request: load, pass, setting, its reply's md5, counters, drafted and accepted, the call's pieces, tiny pieces, the plan check |
| [replies.tsv](replies.tsv) | every sized request's 64 reply token ids |
| [replay.tsv](replay.tsv) | every replay request: pass, setting, task and turn, tokens read and cached, the reply's md5, the pieces (no times) |
| [divergence.tsv](divergence.tsv) | per request of three pairs: whether the pieces are the same, the first differing token and the log-probability margins there |
| [run/run.sh](run/run.sh) | the build machine's driver: three loads, the sized client and the replay on each, then the comparison |
| [run/client.py](run/client.py), [run/compare.py](run/compare.py), [run/divergence.py](run/divergence.py) | the sized prompts, the comparison, where replies part |

## Commands

On the build machine, from a copy of this repository in `$WORK/repo`, with the branch's rig tarball
(`llama.cpp-20261121-1720f26d3-pascal-p106-v5-taper-cuts.tar.gz`, md5 `f4102df5e469492ea57fa7a0517461d8`,
`llama-server` md5 `0cdda3417d39cea7886bc9b0555346fd`) and the base's (`3dfad876f`, md5
`0b5b983e22d7296ff44fe363d55f61bb`, `llama-server` `5b330eccef892b7dd9f4faf1027319c3`) unpacked:

```bash
WORK=<work dir> NEW=<branch build>/bin REF=<base build>/bin REPO=$WORK/repo \
  FN=<folder>/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf \
  DRAFT=<folder>/mtp-Swift-1.5-Qwen3.8-Flash-Next-draft.gguf \
  MOUNTS='<folders the containers read>' DOCKER_ARGS='--label <label> --cpu-shares=256' PREFLIGHT=<CUDA check> \
  setsid nohup bash $WORK/repo/docs/data/2026-11-23_c956_taper_cuts_gpu/run/run.sh > $WORK/driver.log 2>&1 < /dev/null &
```

It ran 17:35–18:19 on 2026-11-23 (the base load 13 minutes, the branch's 27, the third 4). Then, from this folder,
on the copied `load/` folder:

```bash
python3 -I run/compare.py <WORK>/load ../2026-11-21_c905_taper_cuts/plans.tsv . > verdicts.tsv
python3 -I run/divergence.py <WORK>/load head/v5c head/srv head/s2048c head/s2048 head/s2048 head/srv > divergence.tsv
```

The driver's own comparison step ran an earlier `compare.py` that required the same reply on every sized request
and so failed on the 16; the published one applies the rules above, and the replay's time columns are never read.

## Provenance and licence

The engine is Miner Llama (llama.cpp's MIT licence). Swift GSQ-RCO Q2_0 and the MTP head are made from UkisAI's
Swift-1.5 of Qwen3.8-Flash-Next, under its makers' licences; only token ids and log-probabilities of their replies are
published.
The prompts are token stretches of this repository's own READMEs and the agent check's replay plan
([docs/data/2026-11-03_retype_agent](../2026-11-03_retype_agent/)), whose replay text is the build machine's Python
standard-library source tokenized at run time. Tables and scripts here: the repository's licence.
