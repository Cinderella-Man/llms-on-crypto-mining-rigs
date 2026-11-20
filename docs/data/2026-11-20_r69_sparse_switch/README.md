# R69 on a P106: where Flash-Next's sparse attention should take over from the dense kernel (Q1252; 2026-11-20)

**A ratio of 0.75 makes the steady decode step faster at 6,144 and 7,168 tokens of context, and changes nothing at
4,096 and 5,120. 0.625 gains as much at 7,168 but adds a small loss at 5,120.** Flash-Next's attention layers switch
from the dense kernel to the sparse one once the cache holds `GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO` × 2,051 kept
cells × the call's query tokens. The v5 preset's 1.0 makes a four-token verify pass switch at 8,204 cells; 0.75 at
6,153 and 0.625 at 5,128. On the steady second pass a step takes:
- at 6,144 tokens: 102.75 and 102.71 ms with 0.75, against 104.44 and 104.44 with 1.0 (−1.71 ms, −1.6 %, gain);
- at 7,168: 103.09 and 103.02 ms with 0.75 and 103.10 and 103.07 with 0.625, against 106.50 and 106.60 (−3.50 and
  −3.46 ms, −3.3 % and −3.2 %, gains);
- at 5,120: 102.85 and 102.83 ms with 0.625, against 102.53 and 102.54 (+0.31 ms, a loss; +0.88 ms on the first
  pass).

So for a four-token verify pass the two kernels cross between about 5,632 and 6,400 cells, and 0.75's switch point of
6,153 cells lies inside that band. 0.625 sends passes over 5,376 to 5,632 cells to the sparse kernel, which is still
slower there.

**The replies change from 6,144 tokens on, so the agent check decides.** The sparse kernel reads only the cells the
indexer kept, while the dense kernel reads the whole cache under a mask, and the two add up in another order. A greedy
reply therefore parts from the control's after 8 to 180 of its 256 tokens wherever one arm runs a call on the other
kernel. Each arm's two loads wrote the same greedy replies. The draft's acceptance follows the reply. At 8,192 tokens
every arm runs the same verify kernel, but the lower ratios' reply accepted 182 of 218 draft tokens on both passes,
against the control's 187 of 204 on pass 2. Generation over all five depths on pass 2 is therefore 34.23 and 34.27
tok/s with 0.75 and 34.19 and 34.50 with 0.625, against 34.71 and 34.69 (−1.3 % and −1.0 %, losses by the rule). Over
both passes it is 33.43 and 33.49 and 33.52 and 33.65, against 32.94 and 33.83 (negligible). The step is the
kernel's own measure; tok/s on different replies also measures how well the draft guesses each text. R50's run of the
same switch saw the acceptance move the other way (0.781 to 0.964 against the control's 0.666 and 0.738). Ratio 0.75
goes to the agent check against 1.0: planned row Q1635.

The switch's first deep lane (R50, ratios 1.5, 1.0, 0.5 and 0.25) is the
[eighth run in 2026-11-10_deep_lane](../2026-11-10_deep_lane/README.md#the-eighth-run-the-sparse-kernels-threshold-for-verify-passes-q1212).
The step map that priced this test is [2026-11-17_r121_flash_step_map](../2026-11-17_r121_flash_step_map/README.md).
The switch is listed in [docs/switches.md](../../switches.md). Registry rows [Q1252](../../../quants/experiments.md)
(this test), [Q1212](../../../quants/experiments.md) (R50) and [Q1635](../../../quants/experiments.md) (the agent
check that follows).

## What was run

- **Rig and settings:** rig 2, nine P106-090 cards with that rig's soaked clock profile set on the morning's boot
  (memory +900 to +1,300 MT/s and core +206 to +282 MHz per card, the 75 W default power limit, the kit's watchdog
  running: [2026-11-19_r209_oc_v5](../2026-11-19_r209_oc_v5/README.md), "The profile again on the boot of
  2026-11-20"). The offsets read the same before and after the lane, with no watchdog trip and no GPU Xid.
- **Model and preset:** Qwen3.8-Flash-Next as [release/flash-next-v5/settings.env](../../../release/flash-next-v5/settings.env)
  runs it: UkisAI's Swift-1.5 GSQ-RCO Q2_0 (`Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-0000{1,2,3}-of-00003.gguf`,
  10,998,979,680, 26,750,834,816 and 28,800,138,432 bytes; the third shard is its n-gram table) and the MTP draft
  (`mtp-Qwen3.8-Flash-Next-draft.gguf`, 2,697,439,648 bytes), on, three tokens a step. Context 65,536.
- **Build:** one for every arm, fork `pascal-p106-v5` at `602691153`: the candidate `pascal-p106-v5-cand-b12483`
  (upstream b12483 merged). Its `llama-server` md5 is `86607661d5a19d1a90d19ee4db10c90b`.
- **What the ratio reaches:** in mode 1 (`GGML_CUDA_FATTN_SPARSE_PASCAL=1`) a decode or verify call of up to 8 query
  tokens goes to the sparse kernel once the cache's cells reach the ratio × 2,051 × its query tokens. Prompt
  micro-batches of 9 to 128 tokens follow `_PROMPT_MIN_RATIO` (1.0) instead, and longer ones always run the dense
  kernels. The ratio counts cells per query token, so shorter verify passes switch earlier too: a three-token pass at
  4,615 cells with 0.75, against 6,153 with 1.0.
- **Which kernel each arm ran:** a request at depth D reads D cells in its prompt's last call. The server restores the
  checkpoint 4 tokens before the prompt's end and reads those 4 tokens again. Its verify passes then read from about
  D + 256 cells (the cache is padded to 256 cells) to D + 512.

| Depth (tokens) | `c10` (1.0, from 8,204 cells) | `r075` (0.75, from 6,153) | `r0625` (0.625, from 5,128) |
|---:|---|---|---|
| 4,096 | dense | dense | dense |
| 5,120 | dense | dense | prompt's last call dense, verify sparse |
| 6,144 | dense | prompt's last call dense, verify sparse | sparse |
| 7,168 | dense | sparse | sparse |
| 8,192 | prompt's last call dense, verify sparse | sparse | sparse |

| Arm | Setting | Loads |
|---|---|---|
| `c10` (control) | `GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=1.0`, the v5 preset's value | `c10a`, `c10b` |
| `r075` | `=0.75` | `r075a`, `r075b` |
| `r0625` | `=0.625` | `r0625a`, `r0625b` |

The run script drops the preset's own ratio and each arm sets its value, so every server line names one ratio. Every
load's server printed `switches: 40 set` with its arm's ratio.

The deep lane ([bench/deep_lane](../../../bench/deep_lane/README.md)) ran 10:10:04–11:02:19.
- **Loads:** a warm-up load of `c10` (untimed, depth 0), then the timed loads `c10a r075a r0625a r0625b r075b c10b`,
  each a fresh server start: the second round runs the arms in reverse order (Q1253). The loads read the files in
  226, 224, 225, 225, 224, 225 and 224 s (warm-up first).
- **Quiet checks:** the check before every load passed: no llama process, 0 MiB on every card.
- **Requests:** at 4,096, 5,120, 6,144, 7,168 and 8,192 tokens, each load read the prompt (a `prime` request), then
  ran 256 tokens at temperature 0 twice (pass 1 and pass 2), then once with the preset's own sampler and a fixed seed.
  The prompts are the same ids on every load, cut from this project's recorded agent text.
- **No kernel cases:** `test-backend-ops` was not run on the rig. The sparse kernel's own checks are in Q230 and
  [2026-11-11_c103_union_gather](../2026-11-11_c103_union_gather/README.md).

## Result

### The step, ms (`classes_pass2.tsv`, `classes_pass1.tsv`)

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts. Its time is a
request's generation time over its steps. Each value is one load. The class is the lane's rule: a gain or loss means
the difference of the means is larger than the larger of the two arms' spreads (the gap between an arm's two loads);
otherwise it is negligible. "Same kernels" marks cells where the arm runs exactly the control's verify kernels, so
any difference there is the load, not the switch.

**Pass 2, the same request again (the steady figure):**

| Depth (tokens) | `c10` | `r075` | Class | `r0625` | Class |
|---:|---|---|---|---|---|
| 4,096 | 99.88, 99.87 (78 steps) | 99.85, 99.83: same kernels | gain by the rule (−0.03 ms) | 99.90, 99.81: same kernels | negligible (−0.02 ms) |
| 5,120 | 102.53, 102.54 (70) | 102.89, 102.49: same kernels | negligible (+0.15 ms) | 102.85, 102.83 | **loss (+0.31 ms, +0.3 %)** |
| 6,144 | 104.44, 104.44 (69) | 102.75, 102.71 (71) | **gain (−1.71 ms, −1.6 %)** | 104.25, 103.34 (68) | negligible (−0.64 ms, −0.6 %; spread 0.91) |
| 7,168 | 106.50, 106.60 (71) | 103.09, 103.02 (72) | **gain (−3.50 ms, −3.3 %)** | 103.10, 103.07 (72) | **gain (−3.46 ms, −3.2 %)** |
| 8,192 | 103.54, 103.69 (69) | 103.85, 103.74 (74): same verify kernel | loss by the rule (+0.18 ms) | 107.27, 103.64 (74): same verify kernel | negligible (+1.84 ms; spread 3.63) |

**Pass 1, the first request at a new depth:**

| Depth (tokens) | `c10` | `r075` | Class | `r0625` | Class |
|---:|---|---|---|---|---|
| 4,096 | 104.22, 104.24 | 104.05, 103.97: same kernels | gain by the rule (−0.22 ms) | 103.84, 104.27: same kernels | negligible (−0.17 ms) |
| 5,120 | 106.55, 106.45 | 106.78, 106.65: same kernels | loss by the rule (+0.21 ms) | 107.25, 107.50 | **loss (+0.88 ms, +0.8 %)** |
| 6,144 | 121.16, 109.18 | 107.86, 107.36 | negligible (−7.56 ms; spread 11.98) | 107.99, 108.45 | negligible (−6.95 ms; spread 11.98) |
| 7,168 | 120.81, 112.81 | 107.50, 107.08 | **gain (−9.52 ms, −8.1 %)** | 107.49, 107.34 | **gain (−9.40 ms, −8.0 %)** |
| 8,192 | 115.22, 106.01 | 107.86, 107.76: same verify kernel | negligible (−2.80 ms) | 108.48, 108.19: same verify kernel | negligible (−2.28 ms) |

- **Where the arms run the control's kernels, the classes come from the loads.** At 4,096 and 5,120 those arms lie
  −0.22 to +0.21 ms from the control on either pass. At 8,192, on different replies, they lie +0.18 ms away on pass 2,
  or +1.84 ms with `r0625a`'s one slow load (107.27 ms). Those cells set the floor below which no class here means
  anything. The two marked "by the rule" are such cells.
- **The control's own two loads lie 0.00 to 0.15 ms apart on pass 2**, which is narrower than that floor. Classed
  against the control's spread alone, all five same-kernel cells on pass 2 would come out a gain or a loss: `r075`
  and `r0625` at 4,096, `r075` at 5,120, and both at 8,192. So the class used here is the lane's rule, the larger of
  the two arms' spreads. With the control's spread alone, `r0625` at 6,144 would also be a gain (−0.64 ms); every
  class where an arm runs other kernels than the control is otherwise the same either way.
- **Pass 1 at 6,144 to 8,192 carries the first timed load's slow first requests.** `c10a`, the load right after the
  warm-up, took 121.16, 120.81 and 115.22 ms a step there, against `c10b`'s 109.18, 112.81 and 106.01. That widens the
  control's spread to 8.00 to 11.98 ms, the place-in-the-round effect of Q1253. On pass 2 the two `c10` loads lie
  within 0.15 ms at every depth.

### Decode, tok/s (`classes_pass2.tsv`, `classes.tsv`)

| Depth (tokens) | `c10` | `r075` | Class | `r0625` | Class |
|---:|---|---|---|---|---|
| 4,096 | 32.86, 32.86 | 32.87, 32.88 | gain by the rule (+0.0 %) | 32.85, 32.88 | negligible |
| 5,120 | 35.67, 35.67 | 35.55, 35.68 | negligible | 35.56, 35.56 | loss (−0.3 %) |
| 6,144 | 35.52, 35.52 | 35.09, 35.10 | loss (−1.2 %) | 36.11, 36.43 | gain (+2.1 %) |
| 7,168 | 33.86, 33.82 | 34.49, 34.51 | gain (+2.0 %) | 34.49, 34.50 | gain (+1.9 %) |
| 8,192 | 35.83, 35.78 | 33.31, 33.35 | loss (−6.9 %) | 32.25, 33.38 | loss (−8.4 %) |
| all, pass 2 | 34.71, 34.69 | 34.23, 34.27 | loss (−1.3 %) | 34.19, 34.50 | loss (−1.0 %) |
| all, both passes | 32.94, 33.83 | 33.43, 33.49 | negligible (+0.2 %; spread 0.89) | 33.52, 33.65 | negligible (+0.6 %) |
| all, pass 1 | 31.35, 33.02 | 32.67, 32.74 | negligible (+1.6 %; spread 1.67) | 32.88, 32.83 | negligible (+2.1 %) |

Pass 2's tok/s per depth; "all" is every depth's tokens of that pass (or of both passes, the lane's `table.md`) over
their time. Pass 1 carries `c10a`'s slow first requests (below).
Where the replies differ (6,144 to 8,192), tok/s follows the draft's acceptance on each reply as much as the step:

| Depth (tokens) | Pass 2, accepted of drafted: `c10` | `r075` | `r0625` |
|---:|---|---|---|
| 6,144 | 187 of 201 (0.930) | 185 of 207 (0.894) | 188 of 200 (0.940) |
| 7,168 | 185 of 207 (0.894) | 184 of 210 (0.876) | 184 of 210 (0.876) |
| 8,192 | 187 of 204 (0.917) | 182 of 218 (0.835) | 182 of 218 (0.835) |

At 4,096 and 5,120 every arm accepted the same tokens (178 of 229 and 186 of 206 on pass 1).

### Prompt reading, tok/s (`table.md`)

| Depth (tokens) | Read | `c10` | `r075` | Class | `r0625` | Class |
|---:|---|---|---|---|---|---|
| 4,096 | 3,972 on 60 cached | 148.9, 161.0 | 168.5, 164.4 | negligible | 165.3, 168.5 | negligible |
| 5,120 | 4,996 on 60 | 241.4, 242.3 | 243.2, 247.1 | negligible | 245.5, 239.5 | negligible |
| 6,144 | 6,020 on 60 | 318.9, 263.6 | 299.4, 309.8 | negligible | 305.0, 307.3 | negligible |
| 7,168 | 7,044 on 60 | 341.5, 320.4 | 295.7, 329.2 | negligible | 267.1, 308.8 | loss (−13.0 %) |
| 8,192 | 8,068 on 60 | 328.7, 330.1 | 316.2, 314.3 | loss (−4.3 %) | 305.8, 285.6 | loss (−10.2 %) |

The prime requests read their prompt in micro-batches of 388 to 512 tokens, which run the dense kernels in every arm.
The ratio reaches at most a call of up to 8 tokens, and the 62 calls of 4 tokens in the server logs took 0.101 to
0.434 s each. At 8,192 the primes took 24.54 and 24.44 s (`c10`), 25.51 and 25.67 (`r075`) and 26.38 and 28.25
(`r0625`). That is 1.0 to 3.8 s more than the control, which the switch cannot account for.
- Prompt speeds scatter within one arm by up to 55.3 tok/s (`c10` at 6,144) and 41.7 (`r0625` at 7,168).
- No card hit its power cap during any prime at 6,144 to 8,192 tokens. The cards' median busy clock there was 1,999
  to 2,088 MHz in every load (the lane's card log), and the control's two primes at 8,192 ran at the lowest
  (1,999 and 1,999 MHz) while reading fastest.
- The 7,168 and 8,192 classes are therefore not read as the switch's, and their cause is open. The agent check counts
  prompt time per task, so it shows whether any of this reaches a real task.

### The sampler pass

One request a depth and load, with the preset's sampler and a fixed seed, so these are not classed:

| Depth (tokens) | `c10` | `r075` | `r0625` |
|---:|---|---|---|
| 4,096 | 30.66, 30.84 tok/s | 30.71, 30.72 | 30.69, 30.71 |
| 5,120 | 32.00, 31.87 | 31.90, 31.98 | 31.46, 31.46 |
| 6,144 | 34.57, 34.46 | 35.08, 35.02 | 36.21, 36.49 |
| 7,168 | 31.47, 31.47 | 32.15, 32.18 | 32.08, 32.13 |
| 8,192 | 31.81, 31.79 | 30.61, 30.69 | 30.36, 30.63 |

### Checks

- **Replies** (`replies.tsv`: per request and load, the token-id and text md5, and the first token where the reply
  differs from `c10a`'s with that token's text):
  - At 4,096 every greedy and sampled reply is the same bytes and token ids in all six loads. At 5,120 every greedy
    reply is the same. The sampled reply of `r0625`, the arm that verifies on the sparse kernel there, parts from
    token 5 on.
  - At 6,144 the greedy replies form three groups, one per arm, each the same in its two loads. `r075` parts from the
    control at token 8 and `r0625` at token 180.
  - At 7,168 and 8,192 `r075` and `r0625` write the same greedy replies as each other, parting from the control at
    token 14 and token 12. At 7,168 the prime's one generated token already differs (`c10` `' punctuation'`, the
    others `" '"`). There the prompt's last call is dense for `c10` and sparse for the others.
  - Pass 1 and pass 2 at temperature 0 gave the same reply on every load. Each arm's two loads gave the same sampled
    reply, except `r0625` at 6,144, whose two sampled replies part at token 26.
- **The same work:** the same prompt ids on every load (4,096 to 8,192 at the five depths), and 256 tokens out on
  every timed request.
- **Memory** (`memory.tsv`): every card held the same MiB in all seven loads (5,007, 5,059, 4,911, 5,075, 4,175,
  4,927, 4,345, 4,939 and 5,545 MiB on cards 0 to 8). The host's lowest MemAvailable during a load was 302 and 257 MiB
  for `c10`, 239 and 234 for `r075`, and 248 and 242 for `r0625`. The RAM watchdog did not stop a load.
- **Graph reallocations:** none in any load's 21 requests.
- **Heat and power** (`cards.md`, the lane's card log, every card every 0.5 s):
  - The hottest card reached 51 °C.
  - The highest single power sample per card was 81.96 to 87.75 W. The kit's watchdog stops a run when a card holds
    80 W for 5 s, and it did not trip.
  - "Power cap" was a clock reason in 67 to 162 of each card's 2,789 to 3,036 busy samples.
  - The median busy SM clock was 2,088 to 2,151 MHz.
  - The two 2-second stretches at 1,354 MHz (cards 0 and 2, 10:18:01) fall inside `c10a`'s file load, before any
    request.

## What it means, and next

- **Ratio 0.75 is the candidate.** It changes the kernel only for four-token verify passes over 6,153 to 8,204 cells,
  where the sparse kernel is measured faster (1.71 ms a step at 6,400 to 6,656 cells, 3.50 ms at 7,424 to 7,680).
  It keeps the dense kernel below that, where 0.625 loses. On the steady step it is a gain where it acts, and runs the
  control's kernels elsewhere.
- **The agent check decides** (planned row Q1635): Flash-Next's v5 preset with
  `GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=0.75` against the preset as it is, two runs each, on one rig, engine and
  boot. The replies differ from 6,144 tokens on, and tok/s on this lane moves with the draft's acceptance on each
  reply, so only whole tasks say whether the step's gain survives. The est. effect is up to 3.3 % of a step on turns
  whose cache holds 6,153 to 8,204 cells. Agent turns end at 2.8k to 10k tokens.
- **The crossover between about 5,632 and 6,400 cells** is higher than R121's estimate of 5,220 cells (ratio est.
  0.64), which came from the dense kernel's growth per 1,000 cells and the sparse kernel's time at 8,192.

## Files

| File | What |
|---|---|
| `table.md` | the lane's tables as `deep_table.py` prints them, with `c10` as the control and classes on the load (both passes) |
| `classes.tsv`, `classes_pass1.tsv`, `classes_pass2.tsv` | each arm against `c10`, classed on the load, on pass 1 and on pass 2: metric, depth, arm, each load's value, spread, delta, percent, class |
| `replies.tsv` | per request and load: the reply's token-id and text md5, the first token that differs from `c10a`'s reply (with both tokens' text), the draft's accepted of drafted; a `#` line per request groups the loads by reply |
| `requests.csv` | one line per request (`deep_client.py`): prompt and generated tokens, times, draft counters, the reply's text and token-id md5 |
| `loads.tsv` | one line per load: arm, start time, seconds to load, its variables and the server line |
| `memory.tsv` | per load: MiB on each card, MemAvailable when the model was up, the lowest MemAvailable during the load (C865b's [run/memory_table.py](../2026-11-18_c865b_ornith_glu_rig1/run/memory_table.py)) |
| `cards.md` | the card log over the whole lane ([scripts/card_table.py](../../../scripts/card_table.py)): per card the P-states, busy clocks, peak power and temperature, clock reasons |
| `run/r69.sh` | the run script: R205's deep-lane design with the ratio as the arm |
| `run/replies.py` | makes `replies.tsv` from the lane's `<load>.jsonl` files (kept with `KEEP_REPLY=1`) |

## Commands

From the repository's root. The variables:
- `RIG`: the rig's ssh alias.
- `WORK`: the run folder on the rig.
- `BIN`: the build's `llama-server` there.
- `MODEL`: the first shard of the Swift GSQ-RCO Q2_0 file on the rig, with the other two beside it.
- `MODEL_DRAFT`: the MTP draft on the rig.
- `TEXT_LOCAL`: the recorded agent text, the release check's `proxy.jsonl.bodies.gz`, as in R121, R162, C865 and R205.

```bash
export RIG=<rig> WORK=<folder on the rig> \
  BIN=/opt/llama-builds/llama.cpp-20261117-602691153-pascal-p106-v5-cand-b12483/bin/llama-server \
  MODEL=<models>/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf \
  MODEL_DRAFT=<models>/mtp-Qwen3.8-Flash-Next-draft.gguf TEXT_LOCAL=<proxy.jsonl.bodies.gz>
R=docs/data/2026-11-20_r69_sparse_switch/run/r69.sh
STEP=dry bash $R      # the three server lines, nothing started
STEP=start bash $R    # the lane, detached on the rig
STEP=wait bash $R     # returns when the lane has ended (here 52 min)
OUT=<folder> STEP=collect bash $R
# the tables, on any machine with Python 3 (here the machine that drove the run):
D=docs/data/2026-11-20_r69_sparse_switch
python3 -I bench/deep_lane/deep_table.py <folder> --control c10 --out $D/table.md --tsv $D/classes.tsv
python3 -I bench/deep_lane/deep_table.py <folder> --control c10 --class-on pass1 --tsv $D/classes_pass1.tsv
python3 -I bench/deep_lane/deep_table.py <folder> --control c10 --class-on pass2 --tsv $D/classes_pass2.tsv
python3 -I $D/run/replies.py <folder> > $D/replies.tsv
python3 -I docs/data/2026-11-18_c865b_ornith_glu_rig1/run/memory_table.py <folder> $D/memory.tsv
python3 -I scripts/card_table.py <folder>/cards.csv --out $D/cards.md
cp <folder>/requests.csv <folder>/loads.tsv $D/
```

**Provenance:**
- The model file is UkisAI's Swift-1.5 GSQ-RCO Q2_0 of Qwen3.8-Flash-Next with its n-gram table cut into a third
  shard, and the MTP draft, as [release/flash-next-v4](../../../release/flash-next-v4/README.md) describes them.
- The recorded agent text is this project's own release check.
- The tables carry the licence of this repository.
