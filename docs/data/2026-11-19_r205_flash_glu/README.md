# R205 on a P106: Flash-Next with `GGML_CUDA_MMID_GLU_UNFUSE` unset, `1` and `2` (Q1594, its Flash-Next half; 2026-11-19)

**On Flash-Next both settings are a loss, with the same replies.** Ornith decodes 7.0 % faster with the switch. On
Qwen3.8-Flash-Next as its v4 preset runs it (UkisAI's Swift-1.5 GSQ-RCO Q2_0 with the MTP draft), every step gets
slower. On the steady second pass a step takes 100.88 and 100.85 ms with `1`, against 98.40 and 98.59 ms with the
switch unset, at depth 0. With `2` it takes 100.67 and 100.83 ms. At 8,192 tokens a step takes 115.17 and 114.91 ms
with `1` and 115.12 and 115.33 ms with `2`, against 112.97 and 112.98 ms. Decode on that pass over both depths is
27.01 and 27.04 tok/s with `1` and 27.05 and 27.00 with `2`, against 27.62 and 27.59: −2.1 % for each, a loss. Every
greedy reply and every sampled reply is the same bytes and the same token ids in all three arms, and the draft
accepts the same tokens. Each card holds the same MiB in every arm.

`2` against `1` is negligible on the second pass at both depths, so the cost is in what both settings share. Both
take apart the routed experts' fused gate, up and GLU launch. That launch is Q2_0 in all 48 of Flash-Next's layers.
On Ornith's N2P and N3P experts the fused launch is the slower one; on Q2_0 it is the faster one. Flash-Next's preset
therefore leaves the switch unset, and no agent check is needed for it. Ornith's v5 preset keeps `2`
([R204](../2026-11-18_r204_ornith_glu_agent/README.md)). The switch is set per preset, so each file gets its faster
setting.

The switch, its code and what it reaches are in [2026-11-17_c865_glu_unfuse](../2026-11-17_c865_glu_unfuse/README.md);
Ornith's deep lane with it is [2026-11-18_c865b_ornith_glu_rig1](../2026-11-18_c865b_ornith_glu_rig1/README.md). It is
listed in [docs/switches.md](../../switches.md). Registry row [Q1594](../../../quants/experiments.md) (this test; Flint
Pascal Q8 GGUF is its other half, still planned) and [Q1559](../../../quants/experiments.md) (the switch).
Flash-Next's step on v4.0, taken apart, is [2026-11-17_r121_flash_step_map](../2026-11-17_r121_flash_step_map/README.md).

## What was run

- **Rig and settings:** rig 1, nine P106-090 cards at stock clocks and their 75 W default power limit (no offsets, no
  clock lock), booted that morning, idle before the run.
- **Model and preset:** Qwen3.8-Flash-Next as [release/flash-next-v4/settings.env](../../../release/flash-next-v4/settings.env)
  runs it: UkisAI's Swift-1.5 GSQ-RCO Q2_0 (`Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-0000{1,2,3}-of-00003.gguf`,
  10,998,979,680, 26,750,834,816 and 28,800,138,432 bytes; the third shard is its n-gram table) and the MTP draft
  (`mtp-Qwen3.8-Flash-Next-draft.gguf`, 2,697,439,648 bytes), on, three tokens a step. Context 65,536.
- **Build:** one for every arm, fork `pascal-p106-v5` at `602691153`: the candidate `pascal-p106-v5-cand-b12483`
  (upstream b12483 merged). Its `llama-server` md5 is `86607661d5a19d1a90d19ee4db10c90b`, and the binary names
  `GGML_CUDA_MMID_GLU_UNFUSE`. With the switch unset this build runs v4.0's fusions.
- **What the switch reaches in this file** (the code at `602691153`, `ggml-cuda.cu`):
  - The routed experts' gate and up (`MUL_MAT_ID`) are Q2_0 in all 48 layers. Unset, they and their GLU are one
    fused `mul_mat_vec_q` launch at one token and in verify passes of up to 8 tokens. `1` and `2` both run them as two
    launches, then the GLU. Both also refuse the one launch for a routed and a shared expert together.
  - `2` also refuses every other fused `mul_mat_vec_q`: the shared expert's gate, up and GLU at one token, and a
    product with its bias or scale.
  - The preset's `GGML_CUDA_MMVQ_PASCAL_GLU_MT=1` is a separate kernel of the fork's own, and it runs in every arm.
    It is the shared expert's gate, up and SWIGLU as one launch in a verify pass of 2 to 8 tokens.
  - Prompt micro-batches (of 64, 448 and 512 tokens here) run other kernels, which the switch does not change.
- **No kernel cases:** `test-backend-ops` was not run on the rig. Its CPU reference takes hours on the rig's two-core
  host. The same-reply check below covers the whole model.

| Arm | Setting | Loads |
|---|---|---|
| `off` (control) | the switch unset: v4.0's fusions | `f1`, `f2` |
| `u1` | `GGML_CUDA_MMID_GLU_UNFUSE=1`: the routed gate and up products as two launches, then the GLU | `a1`, `a2` |
| `u2` | `GGML_CUDA_MMID_GLU_UNFUSE=2`: upstream's open #32239, no fused `mul_mat_vec_q` on Pascal | `b1`, `b2` |

The deep lane ([bench/deep_lane](../../../bench/deep_lane/README.md)) ran 09:42:32–10:30:52.
- **Loads:** a warm-up load of `off` (untimed), then the timed loads `f1 a1 b1 f2 a2 b2`, each a fresh server start.
  The loads read the files in 311, 311, 311, 310, 313, 314 and 310 s (warm-up first).
- **Quiet checks:** the check before every load passed: no llama process, 0 MiB on every card.
- **Requests:** at depth 0 and at 8,192 tokens each load ran 256 tokens at temperature 0 twice (pass 1 and pass 2),
  then once with the preset's own sampler and a fixed seed.
- **Switches:** every load's server line printed the switches it set (`common_params_print_info: switches: 39 set`
  for `off`, `40 set` with `GGML_CUDA_MMID_GLU_UNFUSE=1` or `=2` for `u1` and `u2`).

## Result

### The step, ms (`classes_pass2.tsv`, `classes_pass1.tsv`)

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts. Its time is a
request's generation time over its steps. The replies and the draft's acceptance are the same in every arm, so every
arm runs the same steps: 108 on pass 2 at depth 0 and 70 at 8,192. Each value is one load. The class is against `off`:
a gain or loss means the difference of the means is larger than the larger of the two arms' spreads (the gap between an
arm's two loads); otherwise it is negligible.

| Depth (tokens) | Pass | `off` | `u1` | Class | `u2` | Class |
|---:|---|---|---|---|---|---|
| 0 | 2 | 98.40, 98.59 | 100.88, 100.85 | loss (+2.37 ms, +2.4 %) | 100.67, 100.83 | loss (+2.25 ms, +2.3 %) |
| 8,192 | 2 | 112.97, 112.98 | 115.17, 114.91 | loss (+2.06 ms, +1.8 %) | 115.12, 115.33 | loss (+2.25 ms, +2.0 %) |
| 0 | 1 | 101.88, 102.67 | 105.06, 105.00 | loss (+2.76 ms, +2.7 %) | 104.96, 104.53 | loss (+2.47 ms, +2.4 %) |
| 8,192 | 1 | 139.35, 117.28 | 119.49, 128.76 | negligible (−3.3 %; spread 22.08) | 119.36, 119.24 | negligible (−7.0 %; spread 22.08) |

**Pass 2 is the steady figure.** Pass 1 at 8,192 is the first request at that depth after a load. `f1`'s took 139.35
ms a step and `a2`'s 128.76, against 117.28 (`f2`), 119.49 (`a1`), 119.36 (`b1`) and 119.24 (`b2`). `f1`'s value sets
the control's spread there (22.08 ms), so neither setting can be classed on that pass. On pass 2 the two `off` loads
lie 0.19 ms apart at depth 0 and 0.01 ms at 8,192.

### Decode, tok/s (`classes.tsv`, `classes_pass1.tsv`, `classes_pass2.tsv`)

| Depth (tokens) | Passes | `off` | `u1` | Class | `u2` | Class |
|---:|---|---|---|---|---|---|
| 0 | 2 | 24.09, 24.04 | 23.50, 23.50 | loss (−2.3 %) | 23.55, 23.51 | loss (−2.2 %) |
| 8,192 | 2 | 32.37, 32.37 | 31.75, 31.83 | loss (−1.8 %) | 31.77, 31.71 | loss (−1.9 %) |
| all | 2 | 27.62, 27.59 | 27.01, 27.04 | loss (−2.1 %) | 27.05, 27.00 | loss (−2.1 %) |
| 0 | both | 23.56, 23.44 | 22.91, 22.92 | loss (−2.5 %) | 22.95, 22.98 | loss (−2.3 %) |
| 8,192 | both | 28.76, 31.54 | 30.95, 29.79 | negligible (+0.7 %; spread 2.78) | 30.97, 30.96 | negligible (+2.7 %; spread 2.78) |
| all | both | 25.90, 26.89 | 26.33, 25.91 | negligible (−1.1 %; spread 0.99) | 26.36, 26.38 | negligible (−0.1 %; spread 0.99) |
| 0 | 1 | 23.05, 22.88 | 22.35, 22.37 | loss (−2.6 %) | 22.38, 22.47 | loss (−2.4 %) |
| 8,192 | 1 | 25.87, 30.74 | 30.18, 28.00 | negligible (+2.8 %; spread 4.87) | 30.21, 30.24 | negligible (+6.8 %; spread 4.87) |

"Both" is a load's two greedy passes' tokens over their time, the lane's own measure. At 8,192 it carries pass 1's
first-request scatter (`f1` 25.87 tok/s on pass 1 against `f2`'s 30.74), which sets the control's spread there and
over both depths. On pass 2 the two `off` loads lie 0.05 and 0.00 tok/s apart, and both settings are a loss.

### `u2` against `u1` (`classes_u2_vs_u1.tsv`, `classes_u2_vs_u1_pass2.tsv`)

| Depth (tokens) | `u1`, pass 2 step, ms | `u2`, pass 2 step, ms | Class | `u1`, pass 2 tok/s | `u2`, pass 2 tok/s | Class |
|---:|---|---|---|---|---|---|
| 0 | 100.88, 100.85 | 100.67, 100.83 | negligible (−0.1 %) | 23.50, 23.50 | 23.55, 23.51 | negligible (+0.1 %) |
| 8,192 | 115.17, 114.91 | 115.12, 115.33 | negligible (+0.2 %) | 31.75, 31.83 | 31.77, 31.71 | negligible (−0.2 %) |

What `2` adds to `1` (the shared expert's one-token fused launch and the fused products with a bias or scale) costs
nothing measurable here. The loss is the part both share: the routed experts' Q2_0 gate and up run apart.

### Prompt reading, tok/s

| Depth (tokens) | Read | `off` | `u1` | Class | `u2` | Class |
|---:|---|---|---|---|---|---|
| 0 | 64 on 0 cached | 53.5, 53.1 | 53.2, 52.3 | negligible (−1.1 %) | 52.1, 53.2 | negligible (−1.3 %) |
| 8,192 | 8,128 on 0 cached | 217.7, 222.1 | 221.2, 224.6 | negligible (+1.4 %) | 224.7, 225.2 | gain (+2.3 %) |

The class of `u2` at 8,192 comes from the rule, but the switch does not reach this work:
- The switch acts only on fused `mul_mat_vec_q` launches, which run for batches of at most 8 tokens.
- The 8,128-token prompt is read in two calls, of 4,096 and 4,032 tokens, in micro-batches of 512 and 448 tokens, and
  the draft catches up in micro-batches of 512. Those run the same kernels in every arm.
- In load order the six prompt speeds were 217.7 (`f1`), 221.2 (`a1`), 224.7 (`b1`), 222.1 (`f2`), 224.6 (`a2`) and
  225.2 (`b2`). The first timed load was the slowest, and `u2`'s loads ran third and last. Ornith's run on the same
  rig the day before showed the same kind of scatter the other way: a "loss" of −1.1 % and −0.5 % for the two arms.

### The sampler pass

One request a depth and load, with the preset's sampler and a fixed seed, so these are not classed:

| Depth (tokens) | `off` | `u1` | `u2` |
|---:|---|---|---|
| 0 | 24.21, 24.04 tok/s | 23.49, 23.45 | 23.44, 23.60 |
| 8,192 | 21.93, 23.86 | 23.49, 23.36 | 23.44, 23.57 |

### Checks

- **Replies:** each request's text md5 and token-id md5 are in `requests.csv`.
  - Pass 1 and pass 2 at temperature 0 gave the same reply on every load and arm. Depth 0: token-id md5
    `2aadc9cfa58da58e1fcec0bc2a78f4de`. Depth 8,192: `2ceddbaabf74614a5a07f13d20b85a49`.
  - The sampler pass gave the same reply on every load and arm. Depth 0: `5c43f559a9db1975a29cd142d17f6f99`. Depth
    8,192: `4229d6696c061a035ebd8244446fb724`.
  - So no KL divergence is needed: the replies are the same bytes.
- **The same work:** the same prompt ids on every load (64 at depth 0, 8,192 at depth 8,192), and 256 tokens out on
  every timed request. The draft accepted the same tokens in every arm: 147 of 321 drafted on pass 1 and 148 of 318
  on pass 2 at depth 0, 185 of 208 and 186 of 205 at 8,192.
- **Memory** (`memory.tsv`): every card held the same MiB in all seven loads (5,007, 5,059, 4,911, 5,075, 4,175,
  4,927, 4,345, 4,939 and 5,545 MiB on cards 0 to 8). The host's lowest MemAvailable during a load was 317 and 277
  MiB for `off`, 274 and 332 for `u1`, and 274 and 284 for `u2`. The RAM watchdog did not stop a load.
- **Graph reallocations:** none in any load's 8 requests.

## What it means, and next

- **The switch is a loss on Flash-Next and stays unset in its preset.** Neither setting earns Flash-Next an agent
  check. On Ornith `2` is a gain on the deep lane and on the agent check, and Ornith's v5 preset sets it.
- **Upstream's #32239 is still an open pull request.** If upstream merges it, its guard turns off every fused
  `mul_mat_vec_q` on Pascal by default, and a merge of that tag would cost Flash-Next this loss. The fork would then
  need a way to keep the fused launch for Q2_0 experts. That is planned row [Q1613](../../../quants/experiments.md),
  parked until upstream merges it.
- **Flint Pascal Q8 GGUF** is Q1594's other half: a dense-FFN file, where `1` changes nothing and `2` takes apart its
  one-token gate, up and GLU. Upstream's own P40 numbers on a dense Llama 3.1 8B were 50.74 → 50.08 tok/s.

## Files

| File | What |
|---|---|
| `table.md` | the lane's tables as `deep_table.py` prints them, with `off` as the control and classes on the load |
| `classes.tsv`, `classes_pass1.tsv`, `classes_pass2.tsv` | each arm against `off`, classed on the load, on pass 1 and on pass 2: metric, depth, arm, each load's value, spread, delta, percent, class |
| `classes_u2_vs_u1.tsv`, `classes_u2_vs_u1_pass2.tsv` | `u2` and `off` against `u1`, on the load and on pass 2 |
| `requests.csv` | one line per request (`deep_client.py`): prompt and generated tokens, times, draft counters, the reply's text and token-id md5 |
| `loads.tsv` | one line per load: arm, start time, seconds to load, its variables and the server line |
| `memory.tsv` | per load: MiB on each card, MemAvailable when the model was up, the lowest MemAvailable during the load (made by C865b's [run/memory_table.py](../2026-11-18_c865b_ornith_glu_rig1/run/memory_table.py)) |
| `run/r205.sh` | the run script: C865's design with the preset, the model and the draft as variables |

## Commands

From the repository's root. The variables:
- `RIG`: the rig's ssh alias.
- `WORK`: the run folder on the rig.
- `BIN`: the build's `llama-server` there.
- `MODEL`: the first shard of the Swift GSQ-RCO Q2_0 file on the rig, with the other two beside it.
- `MODEL_DRAFT`: the MTP draft on the rig.
- `TEXT_LOCAL`: the recorded agent text, the release check's `proxy.jsonl.bodies.gz`, as in R121, R162 and C865.

```bash
export RIG=<rig> WORK=<folder on the rig> \
  BIN=/opt/llama-builds/llama.cpp-20261117-602691153-pascal-p106-v5-cand-b12483/bin/llama-server \
  MODEL=<models>/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf \
  MODEL_DRAFT=<models>/mtp-Qwen3.8-Flash-Next-draft.gguf TEXT_LOCAL=<proxy.jsonl.bodies.gz>
R=docs/data/2026-11-19_r205_flash_glu/run/r205.sh
STEP=dry bash $R      # the three server lines, nothing started
STEP=start bash $R    # the lane, detached on the rig
STEP=wait bash $R     # returns when the lane has ended (here 48 min)
OUT=<folder> STEP=collect bash $R
# the tables, on any machine with Python 3 (here the machine that drove the run):
D=docs/data/2026-11-19_r205_flash_glu
python3 -I bench/deep_lane/deep_table.py <folder> --control off --out $D/table.md --tsv $D/classes.tsv
python3 -I bench/deep_lane/deep_table.py <folder> --control off --class-on pass1 --tsv $D/classes_pass1.tsv
python3 -I bench/deep_lane/deep_table.py <folder> --control off --class-on pass2 --tsv $D/classes_pass2.tsv
python3 -I bench/deep_lane/deep_table.py <folder> --control u1 --tsv $D/classes_u2_vs_u1.tsv
python3 -I bench/deep_lane/deep_table.py <folder> --control u1 --class-on pass2 --tsv $D/classes_u2_vs_u1_pass2.tsv
python3 -I docs/data/2026-11-18_c865b_ornith_glu_rig1/run/memory_table.py <folder> $D/memory.tsv
cp <folder>/requests.csv <folder>/loads.tsv $D/
```

**Provenance:**
- The model file is UkisAI's Swift-1.5 GSQ-RCO Q2_0 of Qwen3.8-Flash-Next with its n-gram table cut into a third
  shard, and the MTP draft, as [release/flash-next-v4](../../../release/flash-next-v4/README.md) describes them.
- The recorded agent text is this project's own release check.
- The tables carry the licence of this repository.
