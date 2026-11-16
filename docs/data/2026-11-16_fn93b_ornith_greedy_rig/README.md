# Which of Ornith's stacked candidates changes the greedy reply on the P106 (FN93b, Q1487; 2026-11-16)

What this measured: which one of the six candidate switches stacked in R162's arm `v4s`
([docs/data/2026-11-14_r162_ornith_release_deep](../2026-11-14_r162_ornith_release_deep/README.md)) makes nine
P106-090 cards write another greedy reply than the v4 preset deep into a context. In R162 `v4s` wrote another reply
than `v4` and v3.0 at 8,192, 16,384 and 32,768 tokens, the same in both of its loads. Ornith-1.5-122B Pascal N2P GGUF
(`valid1`: N2P and N3P experts, a Q5_0 output layer) on `v4-rc2` (`pascal-p106-v4` `eb82916d3`; the rig build's
`llama-server` reports `eb312b7bc`, branch `pascal-p106-v4-b12360-c828`), a 49,152-token context, no draft. It reads
replies only, no speed. Registry row Q1487 in [quants/experiments.md](../../../quants/experiments.md).

| Arm | Settings: the v4 preset ([release/ornith-v4/settings.env](../../../release/ornith-v4/settings.env)) plus |
|---|---|
| `v4` (the control) | nothing |
| `b4096` | `-b 4096` in place of `-b 2048` |
| `taper` | `--ubatch-taper short=600:64` |
| `incall` | `--ctx-checkpoints-vram 4 --ctx-checkpoints-incall` |
| `ckvram` | `--ctx-checkpoints-vram 4` |
| `hop` | `GGML_CUDA_HOP_STAGED=1` |
| `graphs` | `GGML_CUDA_GRAPHS_FORCE=1` |

`incall` carries `--ctx-checkpoints-vram 4` as well, because without it the server turns the in-call checkpoints off.
`v4-rc2` then logs `--ctx-checkpoints-incall: it needs --ctx-checkpoints-vram - off`. So `incall` is read against
`ckvram`, which differs from it by that one switch. Its server log shows both switches on: "context checkpoints on the
devices: the newest 4 of each slot" and "context checkpoints inside a prompt's decode calls".

## Result

**The micro-batch taper (`--ubatch-taper short=600:64`) is the candidate that changes the reply.** On its own it
writes `v4s`'s reply at 8,192 and 16,384 tokens, md5 for md5:

- at 8,192 the reply first differs from `v4`'s at its 11th token;
- at 16,384 it first differs at its 6th token.

At 4,096 tokens it writes a third reply, which first differs at its 21st token from the one `v4` and `v4s` both wrote
there. The five other candidates write `v4`'s reply at every depth, and so does the control.

**Why.** The taper reads a short call of at most 600 tokens, one that starts on idle cards and ends in a wait, in
micro-batches of 64 tokens instead of one of 512. On Ornith, a hybrid of Gated DeltaNet and attention layers, that is
another arithmetic, and a near-tie in the logits goes the other way.

Each prime (the request that reads the next chunk of the context) is cut into calls at its checkpoints:

| Prime (depth, tokens read) | Calls of `v4` and `taper` | Calls of R162's `v4s` | Replies |
|---|---|---|---|
| 4,096 (4,032) | 2,048 + 1,468 + **512** + 4 | 4,032 | `taper` differs; `v4s` = `v4` |
| 8,192 (4,612) | 2,048 + 2,048 + **512** + 4 | 4,096 + **516** | `taper` = `v4s` ≠ `v4` |
| 16,384 (12,804) | 6 × 2,048 + **512** + 4 | 3 × 4,096 + **516** | `taper` = `v4s` ≠ `v4` |
| 32,768 (16,900), R162 only | 8 × 2,048 + **512** + 4 | 4 × 4,096 + **516** | `v4s` ≠ `v4` |
| 48,640 (16,388), R162 only | 7 × 2,048 + 1,536 + **512** + 4 | 4 × 4,096 + 4 | `v4s` = `v4` |

- Bold marks a short call, which the taper reads in 64-token micro-batches. With the v4 preset's calls there is one
  before the last 4 tokens of every prime; `v4` itself has no taper and reads it as one micro-batch of 512.
- In R162 the stack's `-b 4096` and in-call checkpoints joined the 4,096 prime into one call, so the taper had no
  short call to act on there. At 48,640 the stack's last call is 4 tokens, which the taper leaves as it is.
- At 8,192 and 16,384 the stack's short call of 516 tokens falls into the same micro-batches as `taper`'s 512 + 4. So
  the two write the same reply.

Without the taper, where the calls end does not change a reply. `-b 4096` reads in calls of 4,096, the forced graphs
do from the 8,192 prime on, and the in-call checkpoints join the calls of the 4,096 prime into 2,048 + 1,984. Each
moves the ends of the 512-token micro-batches, and each writes `v4`'s reply at every depth.

The switch's help says the taper leaves results unchanged; on Ornith it does not. Which of the two replies is better
is not what this run reads: whether the taper's arithmetic stays within the build's noise is Q1487's second step, a
KLD check against `v4` on a fixed text.

Every value is one load. Run on one rig on 2026-11-16, 11:11:00–12:46:57, one load an arm in the order of the table,
each load 538 or 539 s. No load was stopped, and no graph was reallocated in any of a load's 12 requests.

### Every reply

From [replies.md](replies.md). A reply is the 256 greedy tokens of decode pass 1, as the md5 of its token ids. Pass 2
wrote the same reply in every load and depth. "Differs at token N" means the first N − 1 tokens are the same. R162's
`v4` and `v4s` wrote one reply each per depth, in both of their loads.

| Depth (tokens) | `v4` | `b4096` | `taper` | `incall` | `ckvram` | `hop` | `graphs` |
|---:|---|---|---|---|---|---|---|
| 0 | = both | = both | = both | = both | = both | = both | = both |
| 4,096 | = both | = both | differs from both at token 21 | = both | = both | = both | = both |
| 8,192 | = `v4`; `v4s` at token 11 | = `v4` | **= `v4s`**; `v4` at token 11 | = `v4` | = `v4` | = `v4` | = `v4` |
| 16,384 | = `v4`; `v4s` at token 6 | = `v4` | **= `v4s`**; `v4` at token 6 | = `v4` | = `v4` | = `v4` | = `v4` |

"Both" means R162's `v4` and `v4s`, which wrote the same reply at 0 and 4,096 tokens.

### Checks

- The control `v4` wrote R162's `v4` reply at every depth.
- Every prompt and every prime's prompt was R162's, md5 for md5.
- Every prime read R162's chunk on the same cached tokens:
  - at 4,096, 4,032 tokens on an empty cache;
  - at 8,192, 4,612 tokens on 3,516 cached;
  - at 16,384, 12,804 tokens on 3,516 cached.
- Each arm's switch shows in its server log: the taper's "micro-batch taper of prompt calls: short=600:64", the
  checkpoints on the devices and inside the calls, and the staged hop and forced graphs among the switches set.

## Method

R162's deep lane ([bench/deep_lane](../../../bench/deep_lane/README.md)) on its prompts, with these changes:

- **One load an arm.** The control comes first.
- **Replies only.** `QUIET=0`: no quiet check, every request marked `valid=0`, so no time in the files is a
  measurement.
- **No warm-up load, and no sampler pass.**
  - The sampler pass only replaces decode pass 2's checkpoint at the same position, so the checkpoints a prime
    restores are R162's. The prime's `cache_n` confirms it (Checks).
  - The deepest depth is 16,384. Over these four depths the prompts are R162's first four: the client builds them as
    the first tokens of one stream of recorded sessions.
- **The token ids of every reply.** The lane's client now keeps them in its per-load log. That places the first
  differing token; R162's `v4s` reply is placed through the arm that wrote it here.

The recorded text is the release agent check's request log (the same `proxy.jsonl.bodies.gz`, tag `t1swift/`), as in
R162. The comparison ran on the rig after the last load ([run/replies.py](run/replies.py) against R162's
[requests.csv](../2026-11-14_r162_ornith_release_deep/requests.csv) and its per-load logs).

The first start, at 11:08:03, was stopped at 11:10:34 during its first load, before any request. At that time `incall`
held `--ctx-checkpoints-incall` alone, which the server would have turned off. The run started again at 11:11:00 with
the arms above, in a fresh folder.

## Commands

From the repository's root on a machine that runs only ssh and tar (paths on the rig in the environment):

```bash
. docs/data/2026-11-16_fn93b_ornith_greedy_rig/run/arms.env
export RIG=<rig alias> WORK=<run folder on the rig> BIN=<v4-rc2's llama-server on the rig> \
  MODEL=<Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf on the rig> TEXT_LOCAL=<proxy.jsonl.bodies.gz> OUT=<folder here>
F=docs/data/2026-11-16_fn93b_ornith_greedy_rig/run/fn93b.sh
STEP=start DRY=1 bash $F        # each arm's server line: the preset's plus its one change
STEP=start bash $F
STEP=wait bash $F               # one ssh call that returns when the seven loads have ended
REF_JSONL="<R162's v41.jsonl v42.jsonl s1.jsonl s2.jsonl>" STEP=replies bash $F   # replies.md and .tsv, on the rig
STEP=collect bash $F            # the results to OUT; the run folder on the rig removed
```

## Files

| File | Content |
|---|---|
| [replies.md](replies.md), [replies.tsv](replies.tsv) | every arm's reply per depth against R162's `v4` and `v4s`: the same md5, or the first differing token; the checks |
| [loads.tsv](loads.tsv) | per load its line and binary as it ran, and its load time |
| [requests.csv](requests.csv) | per request its counts and reply md5s (`valid=0`: the times are not measurements) |
| [run/fn93b.sh](run/fn93b.sh) | the run: start, wait, status, replies on the rig, collect |
| [run/arms.env](run/arms.env) | the seven arms |
| [run/replies.py](run/replies.py) | the replies table and its checks |

The server logs, consoles and per-load logs (which hold the replies' text and token ids) stay with the raw output and
are not published, since they hold text written on recorded agent sessions.

## Provenance and licence

Code: Miner Llama (MIT, like llama.cpp), `v4-rc2`. Model: Ornith-1.5-122B as Ornith-1.5-122B Pascal N2P GGUF
(`valid1`), described with its licence in [release/ornith-v3](../../../release/ornith-v3/README.md); the prompts are the
deep lane's recorded agent text ([bench/deep_lane](../../../bench/deep_lane/README.md)). The tables and
`requests.csv` hold counts and md5s, no text. Run on a Manli M-P106L9-N6G (nine P106-090).
