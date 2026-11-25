# R175 on a P106: one full-head drafting round after a shortlist miss instead of two (Q1507; 2026-11-20)

**One round is faster where the draft meets new rare tokens, and the same everywhere else.** Qwen3.8-Flash-Next's MTP
draft reads a shortlist of its output head: the ids below 32,768, the special tokens and the context's tokens. When
the main model's latest token is outside that list, the next drafting rounds read the full head, and the token joins
the list. The v5 preset runs two such rounds (`LLAMA_DRAFT_SHORTLIST=32768,8192,2`); this test runs one (`…,1`).
- **Where it acts, a gain.** The first request at 8,192 tokens writes 14 tokens new to the list. There a step took
  109.82 and 110.05 ms with one round, against 111.30 and 111.20 with two (−1.31 ms, −1.2 %), on the same 83 steps
  and the same accepted draft tokens. The request took 9,115.3 and 9,134.0 ms of generation against 9,237.6 and
  9,229.3 (−108.8 ms).
- **Where it cannot act, the same.** The repeat of a request (pass 2) writes the tokens of the first one, which have
  already joined the list, so it has no miss. There the arms lie +0.07, +0.01 and −0.05 ms a step apart at 0, 4,096
  and 8,192 tokens.
- **The replies are the same.** Every greedy reply is the same token ids in all four loads at every depth. The draft
  accepted the same tokens except at depth 0, where one round accepted one more (148 of 318 against 147 of 321).
  The sampled replies differ between the arms, as they must: the draft's tokens change the sampler's draws.
- **The server's own counts:** with two rounds a load drafted 52 rounds with the full head after 26 misses, with one
  round 31 after 31. On the greedy requests the second full-head round after a miss bought no accepted token.

An agent task is all new text, like the first requests. In the coding-agent check's control run of R227 the server
counted 158 misses and 309 full-head rounds over its 12 tasks, so one round would draft about 151 fewer rounds with the
full head: est. 1.2 s a run at about 8 ms a round (est. from the 8,192-token cell), est. 0.1 s a task. Registry
[Q1507](../../../quants/experiments.md) (this test), [Q1642](../../../quants/experiments.md) (its agent pair) and
[Q1643](../../../quants/experiments.md) (no full-head round at all). The shortlist itself is Q592 and Q635; the
estimate that put this test in the queue is in [2026-11-12_r122_host_ladder](../2026-11-12_r122_host_ladder/README.md).
No full-head round at all (`…,0`) against one is [2026-11-25_r230_flash_shortlist0](../2026-11-25_r230_flash_shortlist0/README.md)
(R230, this folder's run script with `VARIANT=r230`): a loss, because the draft can no longer guess the second piece of
a rare word.

## What was run

- **Rig and settings:** rig 2, nine P106-090 cards with that rig's soaked clock profile set on the morning's boot
  (memory +900 to +1,300 MT/s and core +206 to +282 MHz per card, the 75 W default power limit, the kit's watchdog
  running: [2026-11-19_r209_oc_v5](../2026-11-19_r209_oc_v5/README.md), "The profile again on the boot of
  2026-11-20"). The offsets read the same before and after the lane, with no watchdog trip and no GPU Xid.
- **Model and preset:** Qwen3.8-Flash-Next as [release/flash-next-v5/settings.env](../../../release/flash-next-v5/settings.env)
  ran it on 2026-11-20 at 19:00 (md5 `4cac44b57bd1c6465390dfaab6cd1847`: the sparse ratio 0.75, `--ubatch-taper
  short=600:64`): UkisAI's Swift-1.5 GSQ-RCO Q2_0 (`Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-0000{1,2,3}-of-00003.gguf`;
  the third shard is its n-gram table) and the MTP draft (`mtp-Qwen3.8-Flash-Next-draft.gguf`), on, three tokens a
  step. Context 65,536. The preset took `short=1400:64` later that evening; that setting reaches only prompt calls of
  up to 1,400 tokens, not the decode steps classed here.
- **Build:** one for both arms, fork `pascal-p106-v5` at `602691153` (the candidate `pascal-p106-v5-cand-b12483`,
  upstream b12483 merged); `llama-server` md5 `86607661d5a19d1a90d19ee4db10c90b`.
- **The setting** (`common/draft-shortlist.h` at `602691153`): `LLAMA_DRAFT_SHORTLIST=<k_low>,<m_cap>,<rounds>`. When
  the main model's latest token is outside the list, the next `<rounds>` drafting rounds read the full head, the
  round right after the miss included; 0 never falls back. The token joins the list at once either way. A drafting
  round is one step's three drafting passes, all on the same head.

| Arm | Setting | Loads |
|---|---|---|
| `c` (control) | `LLAMA_DRAFT_SHORTLIST=32768,8192,2`, the v5 preset's value | `ca`, `cb` |
| `fb1` | `LLAMA_DRAFT_SHORTLIST=32768,8192,1` | `fb1a`, `fb1b` |

The run script drops the preset's own value and each arm sets its one value, so the two server lines differ in that
value only; every server printed its rounds in its set-up line ("full head for 2 drafts", "for 1 drafts").

The deep lane ([bench/deep_lane](../../../bench/deep_lane/README.md)) ran 18:58:31–19:27:03.
- **Loads:** a warm-up load of `c` (untimed), then `ca fb1a fb1b cb`, each a fresh server start; the second round runs
  the arms in reverse order (Q1253). Every load read the files in 224 s (the warm-up 227 s).
- **Quiet checks:** the check before every load passed: no llama process, 0 MiB on every card.
- **Requests:** at 0, 4,096 and 8,192 tokens each load read the prompt (a `prime` request, from 4,096 on), ran 256
  tokens at temperature 0 twice (pass 1 and pass 2), then once with the preset's own sampler and a fixed seed. The
  prompts are the same ids on every load, cut from this project's recorded agent text.

## Result

### The step, ms ([steps.md](steps.md), [steps.tsv](steps.tsv))

A step is one pass of the main model: its generation time over its steps (the generated tokens less the accepted draft
tokens). Each value is one load: ms (steps; accepted of drafted). Two classes: `ctl` takes the control's two loads'
difference as the spread, `lane` the larger of the two arms' differences (the lane's own rule). Gain when the arm's
mean is below the control's by more than the spread, loss when above by more, else negligible.

| Pass | Depth | `c` | `fb1` | Arm − control | `ctl` spread, class | `lane` spread, class |
|---|---:|---|---|---:|---|---|
| greedy, pass 1 | 0 | 102.81 (109; 147/321), 94.53 (109; 147/321) | 94.17 (108; 148/318), 94.04 (108; 148/318) | −4.56 | 8.28, negligible | 8.28, negligible |
| greedy, pass 1 | 4,096 | 104.24 (79; 177/231), 103.81 (79; 177/231) | 103.61, 103.76 (79; 177/231) | −0.34 | 0.43, negligible | 0.43, negligible |
| greedy, pass 1 | 8,192 | 111.30, 111.20 (83; 173/245) | 109.82, 110.05 (83; 173/245) | **−1.31** | 0.10, **gain** | 0.22, **gain** |
| greedy, pass 2 | 0 | 90.13, 90.17 (108; 148/318) | 90.26, 90.18 (107; 149/315) | +0.07 | 0.04, loss | 0.09, negligible |
| greedy, pass 2 | 4,096 | 99.52, 99.58 (79; 177/231) | 99.44, 99.67 (79; 177/231) | +0.01 | 0.05, negligible | 0.23, negligible |
| greedy, pass 2 | 8,192 | 103.82, 103.72 (78; 178/230) | 103.75, 103.69 (78; 178/230) | −0.05 | 0.10, negligible | 0.10, negligible |
| sampled | 0 | 92.92, 92.89 (97; 159/285) | 94.62, 94.69 (111; 145/329) | other replies | — | — |
| sampled | 4,096 | 114.12, 102.41 (79; 177/232) | 100.65, 100.86 (77; 179/228) | other replies | — | — |
| sampled | 8,192 | 112.93, 104.77 (75; 181/222) | 105.95, 105.75 (75; 181/220) | other replies | — | — |

- **Pass 1 at 8,192 is the cell where the setting acts.** Its reply has 14 tokens of 32,768 or above that neither the
  prompt nor an earlier reply held (`bled`, `uncated`, ` summarize`, ` sentinel` …), and both arms wrote and accepted
  the same tokens, so the 1.31 ms is the second full-head round's cost alone. At 4,096 the first request's reply has
  one such token (` oddly`): −0.34 ms, inside the 0.43 ms spread. At depth 0 the first timed load `ca` ran slow first
  requests (102.81 ms against `cb`'s 94.53), Q1253's place-in-the-round effect, so that cell has no class; against
  `cb` alone the arm is 0.36 and 0.49 ms faster.
- **Pass 2 at depth 0 is "loss" by the control's spread only because the arm takes one step fewer.** The reply is the
  same, the arm accepted one more draft token, so it ran 107 steps instead of 108 and spread the request's fixed time
  over fewer steps. The request itself took 9,658.1 and 9,648.8 ms of generation against 9,734.1 and 9,738.4 (−82.8 ms).
  The control's 0.04 ms spread is narrower than the step's error; by the lane's rule the cell is negligible.
- **The sampled replies part from the control's at tokens 26, 163 and 30**, the same in each arm's two loads, so their
  steps compare other texts and are not classed. `ca` ran slow first requests here too (114.12 and 112.93 ms).

### Generation per request, ms (`requests.csv`; the greedy replies are the same, so the time compares directly)

| Pass | Depth | `c` | `fb1` | Arm − control | Spread (`ctl`, the arm's) |
|---|---:|---|---|---:|---|
| 1 | 0 | 11,206.0, 10,303.4 | 10,170.5, 10,156.2 | −591.3 | 902.6, 14.3 |
| 1 | 4,096 | 8,234.9, 8,201.2 | 8,185.0, 8,196.7 | −27.1 | 33.7, 11.7 |
| 1 | 8,192 | 9,237.6, 9,229.3 | 9,115.3, 9,134.0 | **−108.8** | 8.3, 18.7 |
| 2 | 0 | 9,734.1, 9,738.4 | 9,658.1, 9,648.8 | **−82.8** | 4.3, 9.3 |
| 2 | 4,096 | 7,862.4, 7,866.8 | 7,856.0, 7,874.1 | +0.5 | 4.3, 18.1 |
| 2 | 8,192 | 8,098.0, 8,090.3 | 8,092.7, 8,087.5 | −4.1 | 7.7, 5.2 |

The lane's own table ([table.md](table.md)) classes generation over all depths at 28.94 and 28.93 tok/s with one round
against 28.25 and 28.75 (+1.5 %, negligible: `ca`'s slow first requests widen the control's spread to 0.50), and the
8,192-token depth a gain (29.75 and 29.73 tok/s against 29.53 and 29.56, +0.7 %).

### The shortlist's counts (the server log's last line per load)

| Load | Full-head rounds after a miss | Drafts with the shortlist | With the full head | Target tokens outside it | Ids in it at the end |
|---|---:|---:|---:|---:|---:|
| `ca`, `cb` | 2 | 736 | 52 | 26 | 33,028 |
| `fb1a`, `fb1b` | 1 | 768 | 31 | 31 | 33,031 |

The arm's five extra misses come from its sampled replies, which hold more new tokens (9 and 3 against 5 and 2 at 0
and 4,096 tokens). The list started at 32,801 ids (the ids below 32,768 and 33 special tokens) and the context's
tokens never came near the 8,192 cap.

### Checks

- **Replies** ([replies.tsv](replies.tsv), made with R69's
  [replies.py](../2026-11-20_r69_sparse_switch/run/replies.py)): every greedy reply and every prime's token the same
  ids in all four loads; the sampled replies one group per arm.
- **Prompt reading** ([table.md](table.md)): the same at 0 and 4,096 tokens; at 8,192 the arm's primes took 21.41 and
  20.71 s against 19.50 and 19.67 (the lane classes it −7.0 %, a loss). A prime drafts nothing (0 drafted tokens in
  every load) and reads the same 4,612 tokens on the same 3,516 cached from the same checkpoint, and the setting only
  picks the head of a drafting round, so the difference is not read as the setting's; its cause is open, as with
  R69's prime scatter. The agent pair reads prompt time per task.
- **Memory and health:** every load's lowest host MemAvailable 224 to 319 MiB, no RAM watchdog stop, 0 graph
  reallocations in all 48 requests; the card log ([cards.md](cards.md)): median busy clock 2,100 to 2,164 MHz, power
  cap in 8 to 30 of each card's busy samples, highest single power sample 82.21 W (card 3), no watchdog trip.

## Files

| File | What |
|---|---|
| [steps.md](steps.md), [steps.tsv](steps.tsv) | the verdict table: the step of every load per depth and pass type, both classes, and the shortlist's counts per load |
| [table.md](table.md), [classes.tsv](classes.tsv), [classes_pass1.tsv](classes_pass1.tsv), [classes_pass2.tsv](classes_pass2.tsv) | the lane's own tables and classes (on the load, pass 1 and pass 2); `table.md` is the rig's byte for byte |
| [replies.tsv](replies.tsv) | per request and load, the reply's token-id and text md5 and the first token where it differs from `ca`'s |
| [requests.csv](requests.csv) | one line per request (deep_client.py) |
| [loads.tsv](loads.tsv) | one line per load: its start, load time and the full server line |
| [cards.md](cards.md) | the card log's summary (`scripts/card_table.py`) |
| [run/r175.sh](run/r175.sh) | the run script (R69's `run/r69.sh` with the shortlist's rounds as the arm) |
| [run/steps.py](run/steps.py) | the verdict table from `requests.csv` and the server logs |

## Commands

```bash
# on a machine that runs only ssh and tar, from the repository's root; RIG, WORK and the paths are yours
RIG=rig WORK=/tmp/r175 BIN=<llama-server on the rig> MODEL=<first shard on the rig> MODEL_DRAFT=<MTP draft on the rig> \
  TEXT_LOCAL=<proxy.jsonl.bodies.gz> STEP=dry bash docs/data/2026-11-20_r175_flash_shortlist/run/r175.sh
# the same with STEP=start, then STEP=wait, then STEP=collect OUT=<folder>
python3 -I bench/deep_lane/deep_table.py OUT --control c --out table.md --tsv classes.tsv
python3 -I bench/deep_lane/deep_table.py OUT --control c --class-on pass1 --tsv classes_pass1.tsv
python3 -I bench/deep_lane/deep_table.py OUT --control c --class-on pass2 --tsv classes_pass2.tsv
python3 -I docs/data/2026-11-20_r175_flash_shortlist/run/steps.py OUT --control c --arm fb1 --tsv steps.tsv > steps.md
python3 -I docs/data/2026-11-20_r69_sparse_switch/run/replies.py OUT --control-load ca --loads ca,fb1a,fb1b,cb > replies.tsv
python3 -I scripts/card_table.py OUT/cards.csv > cards.md
```

## Provenance and licence

Measured by this project on its own rig; the tables and scripts are this repository's. The prompts are cut from this
project's own recorded agent sessions (the v3.0 release check), sent as token ids; their text is not reproduced here.
The model is UkisAI's Swift-1.5 GSQ-RCO Q2_0 of Qwen3.8-Flash-Next with its MTP draft, under the makers' licences.
