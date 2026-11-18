# C865b on a P106: Ornith with `GGML_CUDA_MMID_GLU_UNFUSE` unset, `1` and `2`, the timed loads (Q1559, Q350; 2026-11-18)

**Both settings make Ornith's decode faster, with the same replies.** Over both depths, Ornith-1.5-122B Pascal N2P GGUF
decodes at 17.70 and 17.64 tok/s with `1` (the routed gate and up products unfused), against 16.76 and 16.42 with
v4.0's fusions: +6.5 %, a gain. With `2` (upstream's #32239: no fused matrix-vector product on Pascal) it decodes at
17.68 and 17.81: +7.0 %, a gain. Every greedy reply is the same bytes and the same token ids in all three arms, at
depth 0 and at 8,192, and so is the reply of the pass with the preset's sampler. Each card holds the same MiB in every
arm.

On the steady second pass a token takes 50.44 and 50.44 ms with `1` and 50.01 and 50.04 ms with `2`, against 53.64
and 53.70 ms unset at depth 0. That is 3.23 and 3.65 ms less on the means of two loads, a little more than R176's
est. 2.9 ms. At 8,192 tokens it takes 60.84 and 60.93 ms with `1` and 60.16 and 60.23 ms with `2`, against 64.01 and
64.14 ms. `2` is at least as fast as `1` everywhere, and faster at 8,192 (+1.3 %, a gain). The settings file is
unchanged: release settings are chosen on the agent check. There `2` is a gain too, a decode step 6.3 % shorter at the
same context with prompts unchanged, two runs against two ([R204](../2026-11-18_r204_ornith_glu_agent/README.md)).

This is the timed part of C865's P106 row, whose first run on 2026-11-17 stopped before any timed load:
[2026-11-17_c865_ornith_glu_unfuse](../2026-11-17_c865_ornith_glu_unfuse/README.md). The switch, its code and what it
reaches in Ornith's file are in [2026-11-17_c865_glu_unfuse](../2026-11-17_c865_glu_unfuse/README.md); it is listed in
[docs/switches.md](../../switches.md). Registry rows [Q1559](../../../quants/experiments.md) (the switch) and
[Q350](../../../quants/experiments.md) (its first arm). Ornith's step on v4.0, which says where the change should show,
is [2026-11-17_r176_ornith_step_map](../2026-11-17_r176_ornith_step_map/README.md): the fused routed gate, up and GLU
take 12.02 and 12.00 ms of a 53.72 and 53.73 ms token at depth 0.

## What was run

- **Rig and settings:** rig 1, nine P106-090 cards at stock clocks and their 75 W default power limit (no offsets, no
  clock lock), freshly booted.
- **Model and preset:** Ornith-1.5-122B Pascal N2P GGUF (`valid1`,
  `Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf`, 48,196,722,592 bytes) with its v4 preset
  ([release/ornith-v4/settings.env](../../../release/ornith-v4/settings.env)).
- **Build:** one for every arm, fork `pascal-p106-v5` at `1f1110c67`. That is the candidate
  `pascal-p106-v5-cand-b12410`: the three v5 switches on upstream b12410
  ([2026-11-17_v5_cand_b12410](../2026-11-17_v5_cand_b12410/README.md)), with C865's switch merged as `b56fa04e7`.
  - The rig tarball is `llama.cpp-20261117-1f1110c67-pascal-p106-v5-cand-b12410.tar.gz`, md5
    `5bdcfc21c6b4b0f59a9f7a1d8fed2c08`; its `llama-server` md5 is `5b134f43f3a3e885bfba02538ffebcbe`.
  - With every switch unset, this build gives v4.0's replies, MiB per card and launch groups on every release file
    ([2026-11-17_v5_cand_p106_check](../2026-11-17_v5_cand_p106_check/README.md)).
  - C865's first run used rig 2 and the v5 merge `a112712e3`, one step before upstream b12410. The switch's code is
    the same in both builds: the b12410 merge changed none of its lines.
- **No kernel cases:** `test-backend-ops` was not run on the rig. The same-reply check below covers the whole model,
  and the CPU reference of such cases runs on another machine (the first run's README says why).

| Arm | Setting | Loads |
|---|---|---|
| `off` (control) | the switch unset: v4.0's fusions. On this file the routed gate, up and GLU are one launch in every layer, and the shared expert's another | `f1`, `f2` |
| `u1` | `GGML_CUDA_MMID_GLU_UNFUSE=1`: the routed gate and up products as two launches, then the GLU; the shared expert stays fused | `a1`, `a2` |
| `u2` | `GGML_CUDA_MMID_GLU_UNFUSE=2`: upstream's open #32239, no fused matrix-vector product on Pascal, so the shared expert's gate, up and GLU run apart too | `b1`, `b2` |

The deep lane ([bench/deep_lane](../../../bench/deep_lane/README.md)) ran 09:30:14–11:05:33. It started with a
warm-up load of `off` (untimed), then the timed loads `f1 a1 b1 f2 a2 b2`, each a fresh server start. A load read the
file in 680 to 685 s. The quiet check before every load passed: no llama process, 0 MiB on every card. At depth 0 and
at 8,192 tokens each load ran 256 tokens at temperature 0 twice (pass 1 and pass 2), then once with the preset's own
sampler and a fixed seed. Every load's server line printed the switches it set
(`common_params_print_info: switches: 7 set: …`), with `GGML_CUDA_MMID_GLU_UNFUSE=1` or `=2` in `u1`'s and `u2`'s.

## Result

### Decode, tok/s (`classes.tsv`, `classes_pass1.tsv`, `classes_pass2.tsv`)

Each value is one load: both greedy passes' tokens over their time. The class is against `off`. A gain or loss means
the difference of the means is larger than the larger of the two arms' spreads (the gap between an arm's two loads);
otherwise it is negligible.

| Depth (tokens) | `off` | `u1` | Class | `u2` | Class |
|---:|---|---|---|---|---|
| 0 | 18.20, 17.44 | 19.31, 19.20 | gain (+8.0 %) | 18.95, 19.35 | gain (+7.4 %) |
| 8,192 | 15.52, 15.51 | 16.33, 16.31 | gain (+5.2 %) | 16.56, 16.50 | gain (+6.5 %) |
| all | 16.76, 16.42 | 17.70, 17.64 | gain (+6.5 %) | 17.68, 17.81 | gain (+7.0 %) |

The two passes apart. Pass 1 is the first request at a depth; pass 2 is the same request again.

| Depth (tokens) | Pass | `off` | `u1` | Class | `u2` | Class |
|---:|---|---|---|---|---|---|
| 0 | 1 | 17.79, 16.41 | 18.82, 18.62 | gain (+9.5 %) | 18.01, 18.75 | negligible (+7.5 %; spread 1.38) |
| 0 | 2 | 18.64, 18.62 | 19.82, 19.82 | gain (+6.4 %) | 20.00, 19.98 | gain (+7.3 %) |
| 8,192 | 1 | 15.43, 15.42 | 16.23, 16.22 | gain (+5.2 %) | 16.50, 16.40 | gain (+6.7 %) |
| 8,192 | 2 | 15.62, 15.59 | 16.44, 16.41 | gain (+5.2 %) | 16.62, 16.60 | gain (+6.4 %) |

- **Pass 2 is the steady figure.** Pass 1 at depth 0 is the first request after a load, and `f2`'s ran at 16.41
  against `f1`'s 17.79. That one value sets the control's spread at depth 0 (0.76 on the load, 1.38 on pass 1), and
  it is why `u2`'s +7.5 % on pass 1 at depth 0 is classed negligible.
- **Time a token on pass 2:**

  | Depth (tokens) | `off` | `u1` | `u2` |
  |---:|---|---|---|
  | 0 | 53.64, 53.70 ms | 50.44, 50.44 ms | 50.01, 50.04 ms |
  | 8,192 | 64.01, 64.14 ms | 60.84, 60.93 ms | 60.16, 60.23 ms |

  At depth 0, `off`'s 53.64 and 53.70 ms match R176's 53.72 and 53.73 ms on v4.0 on rig 2.

### `u2` against `u1` (`classes_u2_vs_u1.tsv`, `classes_u2_vs_u1_pass2.tsv`)

| Depth (tokens) | `u1` | `u2` | Class on the load | `u1`, pass 2 | `u2`, pass 2 | Class on pass 2 |
|---:|---|---|---|---|---|---|
| 0 | 19.31, 19.20 | 18.95, 19.35 | negligible (−0.6 %) | 19.82, 19.82 | 20.00, 19.98 | gain (+0.8 %) |
| 8,192 | 16.33, 16.31 | 16.56, 16.50 | gain (+1.3 %) | 16.44, 16.41 | 16.62, 16.60 | gain (+1.1 %) |
| all | 17.70, 17.64 | 17.68, 17.81 | negligible (+0.4 %) | 17.97, 17.96 | 18.16, 18.14 | gain (+1.0 %) |

`2` also unfuses the shared expert's gate, up and GLU (Q5_0). That is the difference from `1`.

### Prompt reading, tok/s

| Depth (tokens) | Read | `off` | `u1` | Class | `u2` | Class |
|---:|---|---|---|---|---|---|
| 0 | 64 on 0 cached | 39.3, 39.2 | 39.3, 39.3 | negligible (+0.1 %) | 39.2, 39.3 | negligible (+0.1 %) |
| 8,192 | 8,128 on 0 cached | 217.0, 216.6 | 215.3, 213.6 | loss (−1.1 %) | 215.4, 216.2 | loss (−0.5 %) |

The two classes at 8,192 come from the rule, but the switch does not reach this work:
- The switch acts only on fused `mul_mat_vec_q` launches, which run for batches of at most 8 tokens
  (`MMVQ_MAX_BATCH_SIZE`).
- The 8,128-token prompt is read in two calls, of 4,096 and 4,032 tokens, in micro-batches of 512 and 448 tokens.
  Those run the same matrix kernels in every arm.
- In load order the six prompt speeds were 217.0 (`f1`), 215.3 (`a1`), 215.4 (`b1`), 216.6 (`f2`), 213.6 (`a2`) and
  216.2 (`b2`). Their scatter is wider than the 0.4 between the two `off` loads, so two loads an arm do not span this
  measurement's own variation.

### The sampler pass

One request a depth and load, with the preset's sampler and a fixed seed, so these are not classed:

| Depth (tokens) | `off` | `u1` | `u2` |
|---:|---|---|---|
| 0 | 17.75, 16.52 | 19.19, 17.82 | 17.50, 18.98 |
| 8,192 | 15.56, 15.53 | 16.38, 16.37 | 16.52, 16.48 |

### Checks

- **Replies:** each request's text md5 and token-id md5 are in `requests.csv`.
  - Pass 1 and pass 2 at temperature 0: the same reply on every load and arm. Depth 0: ids md5
    `b8279751702b64bd93868f577002931f`. Depth 8,192: `eea8bef31dda2a8da383789a89b35205`.
  - The sampler pass: the same reply on every load and arm. Depth 0: `50ed7000e6f5f78c12b1021eca028a20`. Depth 8,192:
    `3afd982a8bdb4970b486cc78e4907ad0`.
  - So no KL divergence is needed: the replies are the same bytes.
- **The same work:** the same prompt ids on every load (64 at depth 0, 8,192 at depth 8,192), and 256 tokens out on
  every timed request.
- **Memory** (`memory.tsv`): every card held the same MiB in all seven loads (5,603, 5,635, 5,549, 5,635, 5,243,
  5,351, 5,513, 5,601 and 5,009 MiB on cards 0 to 8). The host's lowest MemAvailable during a load was 1,262 and
  1,265 MiB for `off`, 1,258 and 1,250 for `u1`, and 1,257 and 1,241 for `u2`. The RAM watchdog did not act.
- **Graph reallocations:** none in any load's 8 requests.

## What it means, and next

- **Both arms are gains on decode, with the same replies.** `2` is upstream's own proposal, and it is the faster of
  the two at depth. If upstream merges #32239, a merge of its tag carries it into this engine with no fork change.
  Until then the switch is off by default and the v4 preset does not set it.
- **R204 (Q1593), done 2026-11-18:** Ornith's agent check ([bench/agent_check](../../../bench/agent_check/README.md)) with
  `GGML_CUDA_MMID_GLU_UNFUSE=2` against the v4 preset, two runs each: a gain, a decode step at the same context 6.3 %
  shorter on the means, a task at equal work 6.6 % shorter, prompts unchanged, 24 of 24 tasks
  ([data](../2026-11-18_r204_ornith_glu_agent/README.md)).
- **R205 (Q1594):** `2` turns off every fused matrix-vector product on Pascal, not only Ornith's. It therefore also
  reaches Flint's and Swift Qwen3.8-Flash-Next's one-token decode, and Flash-Next's draft verify passes of 2 to 8
  tokens. Their deep lanes with `1` and `2` against unset tell whether the setting is right for every release file.
  Upstream's own numbers were mixed: a P40 went from 50.74 to 50.08 tok/s on a dense Llama 3.1 8B.
- **Q350's staged form** (one fused launch that reads the two matrices one after the other) stays planned. To be
  worth writing it would have to beat the unfused 50.01 ms a token at depth 0.

## Files

| File | What |
|---|---|
| `table.md` | the lane's tables as `deep_table.py` prints them, with `off` as the control and classes on the load |
| `classes.tsv`, `classes_pass1.tsv`, `classes_pass2.tsv` | each arm against `off`, classed on the load, on pass 1 and on pass 2: metric, depth, arm, each load's value, spread, delta, percent, class |
| `classes_u2_vs_u1.tsv`, `classes_u2_vs_u1_pass2.tsv` | `u2` against `u1`, on the load and on pass 2 |
| `requests.csv` | one line per request (`deep_client.py`): prompt and generated tokens, times, the reply's text and token-id md5 |
| `loads.tsv` | one line per load: arm, start time, seconds to load, its variables and the server line |
| `memory.tsv` | per load: MiB on each card, MemAvailable when the model was up, the lowest MemAvailable during the load |
| `run/memory_table.py` | makes `memory.tsv` from the lane's files |

The run script is C865's, unchanged:
[2026-11-17_c865_ornith_glu_unfuse/run/c865.sh](../2026-11-17_c865_ornith_glu_unfuse/run/c865.sh). Its header names
the build of the first run; here `BIN` named the `1f1110c67` build.

## Commands

From the repository's root. The variables:
- `RIG`: the rig's ssh alias.
- `WORK`: the run folder on the rig.
- `BIN`: the build's `llama-server` there.
- `MODEL`: the Ornith file on the rig.
- `TEXT_LOCAL`: the recorded agent text, the release check's `proxy.jsonl.bodies.gz`, as in R162, R176 and C865.

```bash
export RIG=<rig> WORK=<folder on the rig> \
  BIN=/opt/llama-builds/llama.cpp-20261117-1f1110c67-pascal-p106-v5-cand-b12410/bin/llama-server \
  MODEL=<models>/Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf TEXT_LOCAL=<proxy.jsonl.bodies.gz>
R=docs/data/2026-11-17_c865_ornith_glu_unfuse/run/c865.sh
STEP=dry bash $R      # the three server lines, nothing started
STEP=start bash $R    # the lane, detached on the rig
STEP=wait bash $R     # returns when the lane has ended (here 1 h 35 min)
OUT=<folder> STEP=collect bash $R
# the tables, on any machine with Python 3 (here the machine that drove the run):
D=docs/data/2026-11-18_c865b_ornith_glu_rig1
python3 -I bench/deep_lane/deep_table.py <folder> --control off --out $D/table.md --tsv $D/classes.tsv
python3 -I bench/deep_lane/deep_table.py <folder> --control off --class-on pass1 --tsv $D/classes_pass1.tsv
python3 -I bench/deep_lane/deep_table.py <folder> --control off --class-on pass2 --tsv $D/classes_pass2.tsv
python3 -I bench/deep_lane/deep_table.py <folder> --control u1 --tsv $D/classes_u2_vs_u1.tsv
python3 -I bench/deep_lane/deep_table.py <folder> --control u1 --class-on pass2 --tsv $D/classes_u2_vs_u1_pass2.tsv
python3 -I $D/run/memory_table.py <folder> $D/memory.tsv
cp <folder>/requests.csv <folder>/loads.tsv $D/
```

**Provenance:**
- The model file is Ornith-1.5-122B Pascal N2P GGUF (`valid1`), built from the maker's published weights (see
  [release/ornith-v4](../../../release/ornith-v4/README.md)).
- The recorded agent text is this project's own release check.
- The tables carry the licence of this repository.
