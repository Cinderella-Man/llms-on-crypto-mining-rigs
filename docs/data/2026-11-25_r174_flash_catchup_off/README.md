# R174 on a P106: the first request with the draft's overlapped catch-up off (Q1506; 2026-11-25)

**The overlapped catch-up never reaches a first request, so it cannot be where the first request's extra time
goes.** A first request at a new depth generates more slowly than the same request repeated (R157 measured 444.11 ms
at 4,096 tokens). This test turned off the draft's overlapped catch-up (`LLAMA_MTP_CATCHUP_OVERLAP=0`; the v5 preset
sets `1`) to see whether that time moves into prompt time, which would mean the catch-up was being counted as
generation.
- **It does not move.** At 4,096 tokens, with the catch-up off, the first request still generated 343.25 and
  324.53 ms more than its repeat over the same 79 steps (with it on: 446.40 and 663.83). Its prompt time did not
  grow: 1,470.30 and 1,450.56 ms against 1,472.89 and 1,484.30.
- **Why:** the control's server counted one overlapped catch-up per load ("1 catch-ups ran after the next call was
  queued"). That one is in the 8,192-token prime, the only prompt in the lane cut into two calls (4,612 tokens at
  `-b 4096`). A first request's 64 new tokens are one call that ends in the reply's first token: the overlap only
  takes calls without an output row, so that catch-up runs the same way with the switch on or off. The first request's cost lies in the passes R191 and R192 found
  ([R191](../2026-11-17_r191_keepalive_stop/README.md), [R192](../2026-11-17_r192_first_request_graphs/README.md)).
- **Class of `off` against the preset: a small loss.** The only requests that run after the one prompt the switch
  reaches are those at 8,192 tokens. There a step took 120.58 and 120.71 ms on the first request with the catch-up off,
  against 120.47 and 120.22 with it on (+0.30 ms, spread 0.25), and 113.27 and 113.32 on the repeat against 113.20 and
  113.04 (+0.18 ms, spread 0.16). At 0 and 4,096 tokens the steps are negligible. The 8,192-token prime itself, where
  the catch-up overlaps, took 21,357.59 and 20,886.25 ms with it off against 20,809.37 and 21,687.65 (negligible:
  spread 878.29). The preset keeps `LLAMA_MTP_CATCHUP_OVERLAP=1`.
- **The replies are the same.** Every greedy reply is the same token ids on both passes of all four loads at every
  depth, with the same accepted draft tokens.

By the class rule the arm also counts as a gain on the first request's generation at 4,096 tokens (−221.23 ms against
a spread of 217.44) and over all depths (−385.38 ms against 88.70). The switch changes nothing in those requests: the
control had one slow first request at depth 0 (load `ca`, a step of 105.37 ms against 102.42 to 102.70 in the other
three loads) and one at 4,096 (load `cb`, 117.17 against 113.02 to 114.35), on code that is the same in both arms. The
first request's time varies between loads by that much. Row 5b on the same rig, build and preset that evening gave
346.30 and 321.30 ms at 4,096 in its control loads (R230, one shortlist setting apart).

Registry [Q1506](../../../quants/experiments.md) (this test). The switch is C703 (Q1242,
[design](../2026-11-11_c703_catchup_overlap/README.md)); the first request's cost is Q1133. The figure this row was
priced on is R121's trace ([2026-11-17_r121_flash_step_map](../2026-11-17_r121_flash_step_map/README.md)): the draft's
catch-up's share of the first request's cost −1.41 and −0.93 ms.

## What was run

- **Rig and settings:** the first rig, nine P106-090 cards, with that day's memory-only profile set on the afternoon's
  boot (memory +900, +1,000, +1,100, +1,200, +1,100, +1,300, 0, +1,100 and +900 MT/s on cards 0 to 8, every core at
  stock, the 75 W default power limit, the kit's watchdog running:
  [2026-11-20_r226_rig1_card_search](../2026-11-20_r226_rig1_card_search/README.md#the-memory-alone-as-the-daily-profile-r238),
  "The memory alone as the daily profile (R238)"). The offsets read the same before and after the lane, with no
  watchdog trip and no GPU Xid.
- **Model and preset:** Qwen3.8-Flash-Next as [release/flash-next-v5/settings.env](../../../release/flash-next-v5/settings.env)
  runs it (md5 `b8e950f2de8c7906c156d582416731bf`): UkisAI's Swift-1.5 GSQ-RCO Q2_0
  (`Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-0000{1,2,3}-of-00003.gguf`; the third shard is its n-gram table) and the MTP
  draft (`mtp-Qwen3.8-Flash-Next-draft.gguf`), on, three tokens a step. Context 65,536, `-b 4096 -ub 512`.
- **Build:** one for both arms, fork `pascal-p106-v5-cand-b12674` at `eee615cdb` (upstream b12674 merged; rig tarball
  `llama.cpp-20261124-eee615cdb-pascal-p106-v5-cand-b12674.tar.gz`, md5 `575ef3c702f69f14d7d3022c1287c38f`);
  `llama-server` md5 `2727faf28b5b34dfe42399a9d431da67`, version 1.0.0-dev (build 13218).
- **The setting** ([docs/switches.md](../../switches.md), `LLAMA_MTP_CATCHUP_OVERLAP`): with `1`, llama-server's MTP
  draft catches up on a prompt call that has no output row once the next call is queued on the main model, so the
  draft's passes run while that call starts through the cards. With `0` the draft catches up on each such call before
  the next one starts. `LLAMA_MTP_HROW_LAZY=1` stays set in both arms and changes nothing without the overlap.

| Arm | Setting | Loads |
|---|---|---|
| `c` (control) | `LLAMA_MTP_CATCHUP_OVERLAP=1`, the v5 preset as it is | `ca`, `cb` |
| `off` | `LLAMA_MTP_CATCHUP_OVERLAP=0` | `offa`, `offb` |

The run script drops the preset's own value and each arm sets its one value, so the two server lines differ in that
value only. Only the control's servers print the overlap's set-up line and count.

The deep lane ([bench/deep_lane](../../../bench/deep_lane/README.md)) ran 18:49:11–19:25:17.
- **Loads:** a warm-up load of `c` (untimed), then `ca offa offb cb`, each a fresh server start; the second round runs
  the arms in reverse order (Q1253). Every load read the files in 306 to 308 s.
- **Quiet checks:** the check before every load passed: no llama process, 0 MiB on every card.
- **Requests:** at 0, 4,096 and 8,192 tokens each load read the prompt (a `prime` request, from 4,096 on), ran 256
  tokens at temperature 0 twice (pass 1, the first request at that depth, and pass 2, the same request again), then
  once with the preset's own sampler and a fixed seed. Pass 1 reads 64 new tokens, pass 2 reads 4. The prompts are the
  same ids on every load, cut from this project's recorded agent text.
- **The overlap's count** (control servers, at exit): `ca` 1, `cb` 1, the warm-up load (depth 0 only) 0.

## Results

`first.md` has every figure by load; the rows below are its centre. Times in ms; excess = pass 1 minus pass 2.

| Depth | Load | Pass 1 prompt (64 tokens) | Pass 1 generation (steps) | Pass 2 generation (steps) | Excess: generation | Excess: prompt | Step pass 1 | Step pass 2 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | `ca` | 1,222.735 | 11,484.789 (109) | 10,592.418 (108) | 892.371 | 999.735 | 105.37 | 98.08 |
| 0 | `cb` | 1,232.335 | 11,193.776 (109) | 10,599.499 (108) | 594.277 | 1,124.901 | 102.70 | 98.14 |
| 0 | `offa` | 1,235.375 | 11,165.816 (109) | 10,595.690 (108) | 570.126 | 1,127.679 | 102.44 | 98.11 |
| 0 | `offb` | 1,240.751 | 11,163.465 (109) | 10,596.770 (108) | 566.695 | 1,126.944 | 102.42 | 98.12 |
| 4,096 | `ca` | 1,472.889 | 9,033.677 (79) | 8,587.279 (79) | 446.398 | 1,351.946 | 114.35 | 108.70 |
| 4,096 | `cb` | 1,484.295 | 9,256.210 (79) | 8,592.377 (79) | 663.833 | 1,363.018 | 117.17 | 108.76 |
| 4,096 | `offa` | 1,470.296 | 8,936.002 (79) | 8,592.752 (79) | 343.250 | 1,346.581 | 113.11 | 108.77 |
| 4,096 | `offb` | 1,450.561 | 8,928.596 (79) | 8,604.069 (79) | 324.527 | 1,326.335 | 113.02 | 108.91 |
| 8,192 | `ca` | 1,321.207 | 9,999.159 (83) | 8,829.364 (78) | 1,169.795 | 1,193.216 | 120.47 | 113.20 |
| 8,192 | `cb` | 1,348.065 | 9,978.594 (83) | 8,816.835 (78) | 1,161.759 | 1,219.315 | 120.22 | 113.04 |
| 8,192 | `offa` | 1,327.354 | 10,008.403 (83) | 8,835.195 (78) | 1,173.208 | 1,203.011 | 120.58 | 113.27 |
| 8,192 | `offb` | 1,347.504 | 10,018.642 (83) | 8,838.779 (78) | 1,179.863 | 1,222.236 | 120.71 | 113.32 |

At 8,192 tokens the first request takes 83 steps and its repeat 78 (173 of 245 draft tokens accepted against 178 of
230) in every load, so its generation excess there is partly five more steps; a step's excess is 7.28 and 7.19 ms with
the catch-up on, 7.31 and 7.39 with it off.

Classes of `off` against `c` (the spread is the larger of the two arms' differences between their two loads; a time
below the control's by more than the spread is a gain, above it a loss):

| Measure | 0 | 4,096 | 8,192 | All depths |
|---|---|---|---|---|
| First request's excess in generation | −174.91 (spread 298.09), negligible | −221.23 (217.44), gain | +10.76 (8.04), loss | −385.38 (88.70), gain |
| First request's excess in prompt time | +64.99 (125.17), negligible | −21.02 (20.25), gain | +6.36 (26.10), negligible | +50.33 (162.34), negligible |
| Step, pass 1 | −1.60 (2.67), negligible | −2.69 (2.82), negligible | +0.30 (0.25), loss | −1.34 (0.33), gain |
| Step, pass 2 | +0.00 (0.07), negligible | +0.11 (0.14), negligible | +0.18 (0.16), loss | +0.09 (0.06), loss |
| Prime's prompt time | — | −644.49 (1,240.58), negligible | −126.59 (878.29), negligible | −771.08 (685.63), gain |

The 4,096-token prime (4,032 tokens) is one call, as are both passes' prompts at every depth: the switch reaches none
of them, so their classes measure the spread between loads, not the switch. The rows it can reach are the 8,192-token
prime (negligible) and the requests after it (the losses at 8,192).

**The sampled request** (the preset's own sampler, not classed here): load `offa` took 123.62 and 124.97 ms a step at
4,096 and 8,192 tokens against 111.82 to 114.51 in the other three loads, with the same accepted draft tokens. During
those two requests its host had 259 and 403 MiB in swap and memory pressure up to 0.55 % (the RAM watchdog's log),
where `offb` and `cb` stayed at 80 to 81 MiB and `ca` reached 210 MiB. Swap varied by load, not by arm; the host's
memory under Flash-Next is Q549. The card clocks were the same in both `off` loads.

## Files

| File | What it holds |
|---|---|
| `first.md`, `first.tsv` | this row's table (`run/first.py`): per depth and load, pass 1 and pass 2 prompt and generation time, the first request's excess, the step of each pass and the prime's prompt time; the classes; the greedy replies' token ids |
| `table.md`, `table_pass1.md`, `table_pass2.md` | the deep lane's tables (`bench/deep_lane/deep_table.py`), classed on both passes, pass 1 and pass 2: tok/s, the step, prompt reading, draft counters, the sampled request, the checks |
| `classes.tsv`, `classes_pass1.tsv`, `classes_pass2.tsv` | the same classes as tables |
| `steps.md`, `steps.tsv` | the step of every load per depth and pass type, classed two ways, and the draft shortlist's counts (R175's `steps.py`) |
| `requests.csv` | one line per request of every load (deep_client.py) |
| `loads.tsv` | one line per load: arm, start, load time, settings and server line |
| `run/first.py` | the script behind `first.md` |

The run script is R175's, [2026-11-20_r175_flash_shortlist/run/r175.sh](../2026-11-20_r175_flash_shortlist/run/r175.sh),
with `VARIANT=r174`.

## Commands

From the repository's root on the machine that drives the rig (it runs only ssh and tar; everything timed runs
detached on the rig):

```bash
export VARIANT=r174 RIG=rig WORK=/var/tmp/r174 PRESET=release/flash-next-v5/settings.env \
  BIN=<llama-server of the eee615cdb build on the rig> \
  MODEL=<folder on the rig>/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf \
  MODEL_DRAFT=<folder on the rig>/mtp-Qwen3.8-Flash-Next-draft.gguf \
  TEXT_LOCAL=<the release check's proxy.jsonl.bodies.gz>
STEP=dry bash docs/data/2026-11-20_r175_flash_shortlist/run/r175.sh
STEP=start bash docs/data/2026-11-20_r175_flash_shortlist/run/r175.sh
STEP=wait bash docs/data/2026-11-20_r175_flash_shortlist/run/r175.sh
STEP=collect OUT=<results folder> bash docs/data/2026-11-20_r175_flash_shortlist/run/r175.sh
```

The tables, where `OUT` is:

```bash
python3 -I docs/data/2026-11-25_r174_flash_catchup_off/run/first.py OUT --control c --arm off --tsv first.tsv > first.md
python3 -I bench/deep_lane/deep_table.py OUT --control c --out table.md --tsv classes.tsv
python3 -I bench/deep_lane/deep_table.py OUT --control c --class-on pass1 --out table_pass1.md --tsv classes_pass1.tsv
python3 -I bench/deep_lane/deep_table.py OUT --control c --class-on pass2 --out table_pass2.md --tsv classes_pass2.tsv
python3 -I docs/data/2026-11-20_r175_flash_shortlist/run/steps.py OUT --control c --arm off --tsv steps.tsv > steps.md
```

## Provenance and licence

Measured on the first rig on 2026-11-25. The recorded agent text is this project's own (the release check's proxy log,
md5 `820be205246dd1113f4dff327c4b27d1`, as R69, R175 and R230 used). The model files are UkisAI's Swift-1.5 GSQ-RCO
Q2_0 and the MTP draft named in the preset's README. The tables and the script are this repository's, under its
licence.
