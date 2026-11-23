# Why a sampled reply changes between loads with a fixed seed (C962, Q1692; 2026-11-23)

What this is: the diagnosis asked for by registry row Q1692 in [quants/experiments.md](../../../quants/experiments.md).
In [C950](../2026-11-21_c950_flint_slot_plan_gpu/README.md), Flint Pascal Q8 GGUF with its v5 preset (pipelined
speculation, `--pipe-depth 4`) on a build machine's three RTX 3090s gave one of two 256-token replies to the same
plain-text request at temperature 0.7, top-k 8, top-p 1, seed 7. That was R14 of C950's coding-agent session. The two
replies share their first 23 tokens. Every build gave both, the base too, while every greedy reply was the same. The
question was whether the model's scores (logits) at the 24th token differ between loads, so that a near tie flips, or
whether the random draws depend on more than the seed and the tokens. Correctness only: nothing here is timed.

## Result in plain words

- **Neither the scores nor the random draw change. What changes is the order in which the card hands its eight top
  candidates to the step that draws among them.** At the 24th token the candidates have fixed probabilities and the
  draw is a fixed number. But the draw picks the first candidate whose running total of probabilities passes it, and
  that running total is summed in the order the candidates arrive. ggml's CUDA top-k kernel writes the candidates in
  the order its thread blocks reach an atomic counter, and that order changes from call to call. So one seed can give
  two replies.
- **It is a reproducibility fault, not a quality fault.** Whatever the order, each candidate is drawn with its own
  probability, so the sampled text is as good and as random as it should be. Only a fixed seed no longer fixes the
  reply.
- **It reaches only requests that leave the top-k's order alone.** Here that means top-k on the card (backend
  sampling, `-bs`) with top-p 1 and no other step that sorts. With top-p 0.95, Flint's preset value and what the
  agent check sends, the top-p step sorts the candidates first: every such request here gave one reply (8 of 8 in the
  session's own context, 9 of 9 fresh). Without a top-k (top-k 0) the order is the vocabulary's: also one reply (8 of 8
  and 9 of 9).
- **The pipelined speculation works as designed.** It keys every draw by the seed and the token's position, and those
  draws are the server's exactly (checked token for token on 18 requests). The same kernel serves decoding without the
  pipe, and the last bits of its probabilities show its order changing there too. Here the pipe showed the effect more
  often, in the session's own context: R14 as C950 sent it gave C950's second reply in 8 of 15 rounds on 7 loads.
- **The fix is to hand the draw a fixed order:** sort the top-k's few candidates before the draw, or make the kernel
  write them in column order. It is queued as Q1718 and not written here.

## The two contexts

R14's prompt, 2,290 tokens, is built by C950's session from its earlier greedy turns. It reaches the server two ways,
and they are not the same input:

- **in the session's own context:** after R1 to R13 with the prompt cache, the server reads only R14's last 14 tokens on
  top of 2,276 cached ones. This is how C950 sent it ([run/session_r14.py](run/session_r14.py));
- **fresh:** the whole prompt read at once, with no cache ([run/requests.py](run/requests.py),
  [run/r14_prompt.txt](run/r14_prompt.txt)).

At the 24th token the two contexts give different probabilities, because the cache holds the turns as the server
generated and kept them, not as the text tokenizes again. The draw there is u = 0.3042, because the token at index 23
is drawn at position 2,312:

| Context | 14 | 65618 | 11844 | 69344 | four more | Picks 65618 when |
|---|---|---|---|---|---|---|
| session | 0.5470 | 0.2770 | 0.1534 | 0.0177 | 0.0049 | 11844 (or 69344 and the four small ones) comes before 65618, and 14 after it |
| fresh | 0.6850 | 0.2299 | 0.0550 | 0.0257 | 0.0044 | 11844 and 69344 both come before 65618, and 14 after it |

C950's first reply takes token 14 there, and it is the greedy reply all the way. The second takes 65618.

## Runs

All on the base build `3dfad876f` (C908's rig tarball, md5 `0b5b983e22d7296ff44fe363d55f61bb`, `llama-server`
`5b330ecc…`). Flint Pascal Q8 GGUF ran with its v5 preset ([release/flint-v5/settings.env](../../../release/flint-v5/settings.env),
md5 `685f0348…`, C950's), with the split changed to the three cards (`-ts 1,1,1`) and `-lv 4`. Each arm was one load in
its own container, 2026-11-23, times on the build machine's clock:

| Step | Loads | What each load ran |
|---|---|---|
| loads, 19:09–19:33 | `pipe-a`, `pipe-b`, `pipe-c` (preset, `--pipe-depth 4`); `nopipe-a`, `nopipe-b` (`--pipe-depth 0`); `p106k` (preset, with the fork's Pascal matrix-vector kernels forced on with a P106's launch shapes) | C950's session once (R14 in the session's own context, as C950 sent it); then fresh R14: `k8` ×5 (C950's settings), `k8probs` ×2 (with `n_probs` 8 and `post_sampling_probs`: the pipe stays off, with the same draws and the dist step's probabilities), `k8p95` ×3 (top-p 0.95), `k0` ×3 (top-k 0), `greedyprobs` ×1 (temperature 0, `n_probs` 8: the raw top 8 on the host) |
| session, 19:34–19:41 | `spipe-a`, `spipe-b`, `snopipe-a` | six rounds of C950's session from R1 to R14, R14 in turn as sent, with `n_probs` (×3), greedy with `n_probs`, as sent |
| probe, 19:45–19:53 | each card on its own | [run/topk_order_probe.cu](run/topk_order_probe.cu): the card's top-k on one fixed row, 2,000 calls per card, input and mode |
| session, 19:58–20:04 | `spipe-c`, `spipe-d` | twelve rounds, R14 in turn as sent, with top-p 0.95, with top-k 0 |

Kernels: the stock arms ran llama.cpp's own CUDA kernels, as C950 did. On a 3090 the fork's Pascal matrix-vector kernels
are off unless forced ([C953](../2026-11-23_c953_ornith_kcol_gpu/README.md)). `p106k` forced them on
(`GGML_CUDA_MMVQ_PASCAL=2 GGML_CUDA_MMVF_PASCAL=2`, `_NSM=5`, `_KCOL_NSM=5`, `_KCOL_SMSM=98304`) to show the effect with
the rig's own kernels. The top-k and the draw that cause it are the same kernels either way. The probe runs the top-k
kernel alone, copied from the fork.

## Findings

**1. C950's two replies recur, and only where the top-k's order reaches the draw.** ([lists.tsv](lists.tsv),
[replies.tsv](replies.tsv)).

| R14 | Pipe | Rounds | Lists |
|---|---|---|---|
| session's own context, top-p 1 (as C950 sent it) | yes | 15 on 7 loads | C950's second reply (`de544dad17e0`) 8, the first (`fe36bf356f32`, the greedy reply) 7; both kinds on 4 of 4 loads with several rounds |
| session's own context, `n_probs` (pipe off, same draws) | — | 6 on 2 loads | one list, the first (6) |
| session's own context, top-p 0.95 | yes | 8 on 2 loads | one list (8) |
| session's own context, top-k 0 | yes | 8 on 2 loads | one list (8) |
| session's own context, top-p 1 | no | 7 on 3 loads | one list (7; another one, because the draws without the pipe come one after another, not by position) |
| fresh, top-p 1 (`k8` pipelined, `k8probs` with the pipe off) | yes | 15 + 6 on 3 loads | one list (21) |
| fresh, top-p 0.95 / top-k 0 | yes | 9 / 9 on 3 loads | one list each |
| fresh, every kind | no | 2 loads | one list per kind |

**2. The scores do not change between loads.** ([logits.tsv](logits.tsv)). The greedy replies' raw top-8 probabilities,
computed on the host from the logits, are the same bits on all five stock loads (0 of 2,560 positions differ) and in the
session's own context on three loads (0 of 768). At the 24th token in the session's own context, the eight candidates and
their probabilities are the same in all nine `n_probs` rounds on three loads, apart from the last bit (14 at 0.5470037 or
0.5470038).

**3. The draw is the fork's position-keyed draw, exactly.** ([drawcheck.tsv](drawcheck.tsv)). With top-p 0.95 the top-p step
sorts the candidates, so each token follows from the draw and the probabilities alone. Draws computed as the fork computes
them predict all 18 such requests token for token, up to and including the token where each leaves greedy decoding:

- with the pipe, token index 176, where the draw is 0.8467 and token 1910 holds 0.8316, so the draw picks 1002;
- without the pipe, token index 23, where the draws come one after another and the draw is 0.7908.

With the pipe each draw is keyed by the seed and the position: splitmix64 of the seed and the position seeds a
std::mt19937, whose first `uniform_real_distribution<double>` draw is used. Without the pipe the draws come from
std::mt19937 seeded with the seed.

**4. So only the candidates' order can change the 24th token, and it does change.** With u = 0.3042 and the session
context's probabilities, 65618 comes out only when the card hands over 11844 before 65618 and 14 after it. No change in
the last bits of a score can do that, since 14's probability (0.547) is far from the draw. Three facts on the server show
the order moving:

- The dist step's probabilities are computed on the card, in the order the candidates arrive. They differ in the last
  bit between identical requests: 791 of 5,376 compared positions fresh, 625 of 4,608 in the session's own context,
  never by more than 2.4e-7. The scores under them are the same bits, so only the order in which they were summed
  differs.
- Where the order decides the token, the pipelined servers' token is the one the sorted order would give at only 1
  of 3 positions fresh and 1 of 2 in the session's own context ([draws.tsv](draws.tsv)). The card's order is not sorted.
- At every position of every `n_probs` reply, the server's token is one that some order of the eight candidates gives
  with the fork's draw (0 exceptions in 21 replies).

**5. The kernel's order changes from call to call on a quiet card.** ([probe.tsv](probe.tsv)). For one row of more than
4,096 columns, ggml's CUDA top-k takes the radix path when built with CUDA 12.9 (its CUB, 2.8.2, has no DeviceTopK). This
holds for every rig build. The radix gather kernel writes each candidate above the threshold at the slot an
`atomicAdd` hands out, so the order is the order in which the warps of up to 64 blocks reach that counter. The probe
runs that kernel copied verbatim from the fork, on one fixed row of 150,144 columns (Flint's vocabulary), 2,000 times per
card in three modes: alone; with the row rewritten before each call; with a spinning kernel on half the card's SMs. It
counts the orders and replays the draw on each:

| Input (candidates at the 24th token, u = 0.3042) | Orders in 2,000 calls (9 runs) | Tokens picked |
|---|---|---|
| session context | 3 to 16 | 65618 in 1,303 to 1,994 calls a run, 11844 in 5 to 355, 14 in 0 to 436 |
| fresh context | 6 to 39 | 14 in all 18,000 calls (no order seen puts 11844 and 69344 before 65618 and 14 after it) |
| `p106k` at token index 110 (all eight candidates in the first block) | 2 to 5 | 1002 in all 18,000 |

The probe shows that one row, one draw and the kernel alone on one card give different tokens. Its frequencies are not
the server's. Its row has a made-up background, and on the server the kernel runs inside a larger graph. There, the
`n_probs` rounds in the session's own context took 14 in 6 of 6.

**6. With the fork's Pascal kernels forced on (`p106k`), Flint's scores move slightly, and sampled replies move at near
ties:**

- The greedy reply is the stock one.
- At token index 110 the draw (0.0285) lies just under token 1002's probability (0.0305 forced, 0.0023 stock).
- At the 24th token, 14 has 0.5646 and 65618 has 0.3363, so 65618 comes out whenever it is handed over before 14: one
  of the five fresh pipelined requests did.

This is the near-tie kind of difference, between kernel settings, not between loads.

## Files

| File | What it holds |
|---|---|
| [replies.tsv](replies.tsv) | every reply: arm, pipe, request, kind, tokens, prompt tokens read, md5 of the token ids, its list, where it leaves list A; C950's two R14 lists named |
| [lists.tsv](lists.tsv) | the lists per kind of request, how many replies and which arms gave each, where each leaves list A |
| [logits.tsv](logits.tsv) | per kind and kernel setting: pairs of replies compared position by position along their shared tokens, positions whose top-8 probabilities differ, the largest difference |
| [drawcheck.tsv](drawcheck.tsv) | the top-p 0.95 requests predicted from the draws and the greedy reply's probabilities: tokens checked, mispredicted, the token where the reply leaves greedy decoding with its draw, kept candidates and prediction |
| [draws.tsv](draws.tsv) | per `n_probs` reply: positions where the order decides, tokens no order could give, tokens equal to the sorted order's pick |
| [partings.tsv](partings.tsv) | where two `n_probs` replies of the same draws and context part: candidates, draw, tokens (one pair, stock against forced kernels) |
| [probe.tsv](probe.tsv) | the probe: input, mode, card, the draw, calls, distinct orders, the most common orders with the token each picks, tokens picked |
| [arms.tsv](arms.tsv) | per load: pipe depth, kernels, load, the pipe's line, the one-top-k line, vocabulary, the pipe's guesses per request, MiB per card |
| [run/run.sh](run/run.sh) | the driver: `STEP=loads`, `STEP=session`, `STEP=probe` |
| [run/requests.py](run/requests.py) | the fresh R14 requests |
| [run/session_r14.py](run/session_r14.py) | C950's session up to R14, R14 in one of five variants |
| [run/r14_prompt.txt](run/r14_prompt.txt) | R14's prompt as C950's session builds it from its greedy turns |
| [run/topk_order_probe.cu](run/topk_order_probe.cu) | the probe (the radix top-k kernels and host function copied from the fork's `ggml/src/ggml-cuda/top-k.cu` at `3dfad876f`, MIT licence) |
| [run/probe/](run/probe/) | the probe's three inputs, written by `c962_tables.py --probe-inputs` |
| [run/c962_tables.py](run/c962_tables.py) | every table here from the run's output |

## Commands

On the build machine, from a copy of this repository at `$REPO` (this folder's `run/`, C950's `run/session.py`,
`release/flint-v5/settings.env`), detached:

```bash
COMMON="WORK=$WORK REF=$REF REPO=$REPO FLINT=$FLINT MOUNTS='<build> <model folders>' \
  DOCKER_ARGS='--label <policy label> --cpu-shares=256' PREFLIGHT=<cuda check> LOCK=<build lock>"
env $COMMON STEP=loads setsid nohup bash run.sh > $WORK/driver-loads.log 2>&1 < /dev/null &
env $COMMON STEP=session setsid nohup bash run.sh > $WORK/driver-session.log 2>&1 < /dev/null &
python3 -I run/c962_tables.py <this folder> $WORK/out --probe-inputs $WORK/probe   # the probe's inputs
env $COMMON STEP=probe setsid nohup bash run.sh > $WORK/driver-probe.log 2>&1 < /dev/null &
env $COMMON STEP=session ARMS="spipe-c spipe-d" VARIANTS="pipe p95 k0 pipe p95 k0 pipe p95 k0 pipe p95 k0" \
  setsid nohup bash run.sh > $WORK/driver-session.log 2>&1 < /dev/null &
python3 -I run/c962_tables.py <this folder> $WORK/out --probe $WORK/probe
```

`REF` is the rig build's `bin/` and `FLINT` is Flint Pascal Q8 GGUF. One step at a time, each ending with
`$WORK/DONE-<step>`.

## Provenance and licence

The model's replies are Flint Pascal Q8 GGUF's (released by this project), to C950's synthetic coding task. The probe
copies about 210 lines of llama.cpp's `ggml/src/ggml-cuda/top-k.cu` (MIT licence, Copyright (c) 2023-2026 The ggml
authors). The tables were made from the run's raw output with `run/c962_tables.py`.
