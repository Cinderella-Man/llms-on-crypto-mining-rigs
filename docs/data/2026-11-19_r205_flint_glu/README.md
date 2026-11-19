# R205 on a P106: Flint with `GGML_CUDA_MMID_GLU_UNFUSE` unset, `1` and `2` (Q1594, its Flint half; 2026-11-19)

**On Flint `2` is a small gain and `1` is not, with the same replies.** Flint Pascal Q8 GGUF (`p065`) ran on its v4
preset with up to four pipelined guesses. On the steady second pass a token takes 19.340 and 19.309 ms with `2` at
8,192 tokens of context, against 19.452 and 19.475 ms with the switch unset: −0.7 %, a gain. At depth 0 it takes
18.016 and 18.107 ms against 18.086 and 18.195, which is negligible. Decode on that pass over both depths is 53.54 and
53.45 tok/s with `2`, against 53.28 and 53.09 unset: +0.6 %, a gain. `1` is negligible at depth 0 (−0.3 %) and a loss
at 8,192 (19.547 and 19.521 ms, +0.4 %). Every greedy and sampled reply is the same bytes and the same token ids in
all three arms. The pipeline guessed the same tokens right and wrong in every arm, and each card held the same MiB.

`1` takes apart only the routed experts' fused gate, up and GLU launch, which costs Flint a little, as on Flash-Next.
`2` also takes apart the shared expert's and the dense layers' gate, up and GLU launches, and every product fused
with the add after it. On Flint's Q8_0 and Q5_0 weights those launches run faster apart, and that more than pays for
the routed part: `2` against `1` is a gain on pass 2 at both depths (+0.8 % and +1.1 %). The gain is small, so the
agent check decides it. That check is row [Q1626](../../../quants/experiments.md): Flint's v5 preset against the same
preset with `2`. Ornith's v5 preset already sets `2` ([R204](../2026-11-18_r204_ornith_glu_agent/README.md)).
Flash-Next's leaves the switch unset ([its half](../2026-11-19_r205_flash_glu/README.md): a step +2.3 % with `2`).

The switch, its code and what it reaches are in [2026-11-17_c865_glu_unfuse](../2026-11-17_c865_glu_unfuse/README.md).
It is listed in [docs/switches.md](../../switches.md). Registry rows [Q1594](../../../quants/experiments.md) (this
test, both halves), [Q1559](../../../quants/experiments.md) (the switch) and Q1626 (Flint's agent check with it).
Flint's step on v4.0, taken apart kernel by kernel, is
[2026-11-17_r176_flint_step_map](../2026-11-17_r176_flint_step_map/README.md).

## What was run

- **Rig and settings:** rig 1, nine P106-090 cards at stock clocks and their 75 W default power limit (no offsets, no
  clock lock), booted that morning, idle before the run.
- **Model and preset:** Flint Pascal Q8 GGUF (`flint-near-q8-p065.gguf`, 40,431,236,544 bytes) as
  [release/flint-v4/settings.env](../../../release/flint-v4/settings.env) runs it. That preset has a 262,144-token
  context, one token a step with up to four pipelined guesses (`--pipe-depth 4`) and no draft model. The RAM watchdog's
  floor was `MIN_AVAIL_MB=60`, as in R176.
- **Build:** one for every arm, fork `pascal-p106-v5` at `602691153`: the candidate `pascal-p106-v5-cand-b12483`
  (upstream b12483 merged). Its `llama-server` md5 is `86607661d5a19d1a90d19ee4db10c90b`, and the binary names
  `GGML_CUDA_MMID_GLU_UNFUSE`. With the switch unset this build runs v4.0's fusions.
- **What the switch reaches in this file:** the code at `602691153` (`ggml-cuda.cu`) and the launch groups of a
  one-token pass as R176's group profile named them. The ms are R176's on v4.0, at depth 0 on the repeated request.
  All of these are one-token `mul_mat_vec_q` launches:

| Fused launch in a one-token pass | Weights | ms a pass on v4.0 (R176, `g1`) | `1` | `2` |
|---|---|---|---|---|
| routed experts' gate, up and SWIGLU (`MUL_MAT_ID+MUL_MAT_ID+SWIGLU ffn_gate_exps ffn_up_exps`) | Q8_0; Q5_0 | 4.17; 0.51 | apart | apart |
| attention output with the add after it (`MUL_MAT+ADD attn_output`) | Q8_0; Q5_0 | 2.44; 0.94 | fused | apart |
| shared expert's gate, up and SWIGLU (`MUL_MAT+MUL_MAT+SWIGLU ffn_gate_shexp ffn_up_shexp`) | Q8_0; Q5_0 | 1.05; 0.42 | fused | apart |
| shared expert's down with the add after it (`MUL_MAT+ADD ffn_down_shexp`) | Q8_0; Q5_0 | 0.83; 0.33 | fused | apart |
| dense layers' gate, up and SWIGLU (`MUL_MAT+MUL_MAT+SWIGLU ffn_gate ffn_up`) | Q8_0 | 0.37 | fused | apart |
| dense layers' down with the add after it (`MUL_MAT+ADD ffn_down`) | Q8_0 | 0.18 | fused | apart |

- **More on the switch in this file:**
  - The one launch for a routed and a shared expert together is never made on Flint, with the switch or without it.
    It needs both weights of one shape and type, and Flint's shared expert is 2,048 × 1,024 against a routed
    expert's 2,048 × 512.
  - The routed experts' down product and the router are single products, not fused, and run the same in every arm.
  - Flint's preset never runs a pass of 2 to 8 tokens: its pipelined guesses are one-token passes too.
  - Prompt micro-batches (of 64 and 128 tokens here) run other kernels, which the switch does not change.
- **No kernel cases:** `test-backend-ops` was not run on the rig. Its CPU reference takes hours on the rig's two-core
  host. The same-reply check below covers the whole model.

| Arm | Setting | Loads |
|---|---|---|
| `off` (control) | the switch unset: v4.0's fusions | `f1`, `f2` |
| `u1` | `GGML_CUDA_MMID_GLU_UNFUSE=1`: the routed gate and up products as two launches, then the GLU | `a1`, `a2` |
| `u2` | `GGML_CUDA_MMID_GLU_UNFUSE=2`: upstream's open #32239, no fused `mul_mat_vec_q` on Pascal | `b1`, `b2` |

The deep lane ([bench/deep_lane](../../../bench/deep_lane/README.md)) ran 16:52:08–18:07:31.
- **Loads:** a warm-up load of `off` (untimed), then the timed loads `f1 a1 b1 f2 a2 b2`, each a fresh server start.
  The loads read the file in 590, 589, 589, 588, 590, 591 and 592 s (warm-up first).
- **Quiet checks:** the check before every load passed: no llama process, 0 MiB on every card.
- **Requests:** at depth 0 and at 8,192 tokens each load ran 256 tokens at temperature 0 twice (pass 1 and pass 2),
  then once with the preset's own sampler and a fixed seed. Every reply was kept whole (`KEEP_REPLY=1`).
- **Switches:** every load's server line printed the switches it set (`switches: 5 set` for `off`, `6 set` with
  `GGML_CUDA_MMID_GLU_UNFUSE=1` or `=2` for `u1` and `u2`).

## Result

### Decode, ms a token and tok/s (`classes_pass2.tsv`, `classes_pass1.tsv`, `classes.tsv`)

A token's time is a request's generation time over its 256 tokens. Flint decodes one token a step with pipelined
guesses; a token the pipeline guessed right comes nearly free (R176). The replies and the guesses are the same in
every arm, so every arm does the same work. Each value is one load. The class is against `off`: a gain or loss means
the difference of the means is larger than the larger of the two arms' spreads (the gap between an arm's two loads);
otherwise it is negligible.

| Depth (tokens) | Pass | `off` | `u1` | Class | `u2` | Class |
|---:|---|---|---|---|---|---|
| 0 | 2, ms a token | 18.086, 18.195 | 18.205, 18.198 | negligible (+0.3 %) | 18.016, 18.107 | negligible (−0.4 %) |
| 8,192 | 2, ms a token | 19.452, 19.475 | 19.547, 19.521 | **loss** (+0.4 %) | 19.340, 19.309 | **gain** (−0.7 %) |
| 0 | 2, tok/s | 55.29, 54.96 | 54.93, 54.95 | negligible (−0.3 %; spread 0.33) | 55.51, 55.23 | negligible (+0.4 %; spread 0.33) |
| 8,192 | 2, tok/s | 51.41, 51.35 | 51.16, 51.23 | loss (−0.4 %; spread 0.07) | 51.71, 51.79 | gain (+0.7 %; spread 0.08) |
| all | 2, tok/s | 53.28, 53.09 | 52.98, 53.02 | negligible (−0.3 %; spread 0.19) | 53.54, 53.45 | **gain** (+0.6 %; spread 0.19) |
| 0 | 1, ms a token | 21.369, 21.611 | 21.286, 26.688 | — | 21.170, 21.198 | — |
| 8,192 | 1, ms a token | 21.260, 21.008 | 21.400, 21.839 | — | 21.555, 21.279 | — |
| 0 | 1, tok/s | 46.80, 46.27 | 46.98, 37.47 | negligible (−9.3 %; spread 9.51) | 47.24, 47.17 | gain (+1.4 %; spread 0.52) |
| 8,192 | 1, tok/s | 47.04, 47.60 | 46.73, 45.79 | loss (−2.2 %; spread 0.94) | 46.39, 47.00 | loss (−1.3 %; spread 0.60) |
| 0 | both, tok/s | 50.69, 50.24 | 50.64, 44.56 | negligible (−5.7 %; spread 6.09) | 51.04, 50.88 | gain (+1.0 %; spread 0.45) |
| 8,192 | both, tok/s | 49.13, 49.40 | 48.84, 48.36 | loss (−1.3 %; spread 0.49) | 48.91, 49.28 | negligible (−0.4 %; spread 0.37) |
| all | both, tok/s | 49.90, 49.82 | 49.73, 46.38 | negligible (−3.6 %; spread 3.35) | 49.95, 50.07 | gain (+0.3 %; spread 0.12) |

**Pass 2 is the steady figure.** Pass 1 is the first request at a depth after a load. On it the two `off` loads lie
0.242 ms a token apart at depth 0 and 0.252 at 8,192; on pass 2 they lie 0.109 and 0.023 apart. `u2`'s pass-1 loss at
8,192 is 0.283 ms a token on means whose spreads are 0.252 (`off`) and 0.276 (`u2`). On pass 2 at that depth `u2` is
0.139 ms a token faster, on spreads of 0.023 and 0.031.

**One load was disturbed by its host.** In the first 40 s after `a2` came up, three of its requests took longer than
the same request on any other load:
- pass 1 at depth 0, 17:55:38–17:55:45: 26.688 ms a token, where `f1`, `f2`, `a1`, `b1` and `b2` took 21.369,
  21.611, 21.286, 21.170 and 21.198;
- the sampler pass at depth 0, 17:55:51–17:55:56: 18.810 ms a token, where the others took 16.521, 16.601, 16.657,
  16.621 and 16.210;
- the 8,128-token prompt, 17:55:57–17:56:15: 445.4 tok/s, where the others read 593.3, 592.1, 592.7, 593.4 and 594.3.

`a2`'s pass 2 at depth 0 ran between them, 17:55:45–17:55:50, and took 18.198 ms a token, where `a1`'s took 18.205.
The card log shows the same clocks (P0, the memory at 4,006 MHz), temperatures and throttle flags as in `f2`'s same
requests. The host's lowest MemAvailable in that load was 1,145 MiB, with nothing written to swap. It was a stall on
the host's side, which the switch does not touch. It sets `u1`'s spread on pass 1 and on the load at depth 0.

### `u2` against `u1` (`classes_u2_vs_u1.tsv`, `classes_u2_vs_u1_pass2.tsv`)

| Depth (tokens) | `u1`, pass 2 tok/s | `u2`, pass 2 tok/s | Class |
|---:|---|---|---|
| 0 | 54.93, 54.95 | 55.51, 55.23 | gain (+0.8 %; spread 0.28) |
| 8,192 | 51.16, 51.23 | 51.71, 51.79 | gain (+1.1 %; spread 0.08) |
| all | 52.98, 53.02 | 53.54, 53.45 | gain (+0.9 %; spread 0.08) |

`u1` takes apart only the routed experts' launch, and on Flint that is a loss at 8,192 and negligible at depth 0.
Everything `u2` adds to it is a gain: the shared expert's and the dense layers' gate, up and GLU apart, and the
attention output, the shared expert's down and the dense layers' down apart from the add after each.

### Prompt reading, tok/s

| Depth (tokens) | Read | `off` | `u1` | Class | `u2` | Class |
|---:|---|---|---|---|---|---|
| 0 | 64 on 0 cached | 87.2, 88.7 | 91.5, 86.8 | negligible (+1.4 %) | 89.4, 88.8 | negligible (+1.3 %) |
| 8,192 | 8,128 on 0 cached | 593.3, 592.1 | 592.7, 445.4 | negligible (−12.4 %; `a2`'s stall) | 593.4, 594.3 | negligible (+0.2 %) |

### The sampler pass

One request a depth and load, with the preset's sampler and a fixed seed, so these are not classed:

| Depth (tokens) | `off` | `u1` | `u2` |
|---:|---|---|---|
| 0 | 60.53, 60.24 tok/s | 60.04, 53.16 | 60.16, 61.69 |
| 8,192 | 43.29, 43.21 | 43.16, 43.13 | 43.59, 43.66 |

### Checks

- **Replies:** each request's text md5 and token-id md5 are in `requests.csv`.
  - Pass 1 and pass 2 at temperature 0 gave the same reply on every load and arm. Depth 0: token-id md5
    `576dc4660e25e1b481820cf8464118bc`. Depth 8,192: `e20d7b7ece3980f47cb4e5d6c2e832b7`. These are the replies
    v4.0 wrote on the same file, preset and prompts (R168, R176).
  - The sampler pass gave the same reply on every load and arm. Depth 0: `eb1751516ccb534bb93c2190c5d0ca9b`. Depth
    8,192: `a3384f1e668d02b8c30e4cbae2413b12`.
  - So no KL divergence is needed: the replies are the same bytes.
- **The same work:** the same prompt ids on every load (64 at depth 0, 8,192 at depth 8,192), and 256 tokens out on
  every timed request. The pipeline made the same guesses in every load: 352 guesses with 80 right and 68 wrong on
  each greedy pass at depth 0, and 567 with 93 right and 120 wrong at 8,192.
- **Memory** (`memory.tsv`): every card held the same MiB in all seven loads (5,383, 5,923, 5,081, 5,591, 5,923,
  5,081, 5,545, 5,293 and 4,793 MiB on cards 0 to 8). The host's lowest MemAvailable during a load was 1,164 and
  1,167 MiB for `off`, 1,141 and 1,145 for `u1`, and 1,145 and 1,154 for `u2`. The RAM watchdog did not stop a load.
- **Graph reallocations:** none in any load's 8 requests.
- **The v5 build with the switch unset runs as v4.0:** pass 2 at depth 0 took 18.086 and 18.195 ms a token, where
  v4.0's two loads took 18.09 and 18.09 (R168), and 19.452 and 19.475 at 8,192 against 19.48 and 19.50.

## What it means, and next

- **`2` is a small gain on Flint's deep lane with the same replies; `1` is not.** Release settings are chosen on the
  agent check, so `2` gets one: [Q1626](../../../quants/experiments.md), Flint's v5 preset
  ([release/flint-v5/settings.env](../../../release/flint-v5/settings.env)) against the same preset with
  `GGML_CUDA_MMID_GLU_UNFUSE=2`, two runs each on this build. Until then Flint's v5 preset leaves the switch unset.
- **The switch is set per preset.** On Ornith (N2P and N3P routed experts) `2` is a gain on the agent check and in its
  v5 preset. On Flash-Next (Q2_0 routed experts) both settings are a loss, and it stays unset. On Flint the routed
  part alone costs a little and the rest of `2` gains more.
- **If upstream merges its open #32239,** every fused `mul_mat_vec_q` on Pascal goes off by default. That suits Ornith
  and Flint and costs Flash-Next ([Q1613](../../../quants/experiments.md), parked until then).

## Files

| File | What |
|---|---|
| `table.md` | the lane's tables as `deep_table.py` prints them, with `off` as the control and classes on the load |
| `classes.tsv`, `classes_pass1.tsv`, `classes_pass2.tsv` | each arm against `off`, classed on the load, on pass 1 and on pass 2: metric, depth, arm, each load's value, spread, delta, percent, class |
| `classes_u2_vs_u1.tsv`, `classes_u2_vs_u1_pass2.tsv` | `u2` and `off` against `u1`, on the load and on pass 2 |
| `requests.csv` | one line per request (`deep_client.py`): prompt and generated tokens, times, the reply's text and token-id md5 |
| `loads.tsv` | one line per load: arm, start time, seconds to load, its variables and the server line |
| `memory.tsv` | per load: MiB on each card, MemAvailable when the model was up, the lowest MemAvailable during the load (made by C865b's [run/memory_table.py](../2026-11-18_c865b_ornith_glu_rig1/run/memory_table.py)) |
| `run/r205f.sh` | the run script: the Flash-Next half's with Flint's preset and file, no draft, and R176's watchdog floor |

## Commands

From the repository's root. The variables:
- `RIG`: the rig's ssh alias.
- `WORK`: the run folder on the rig.
- `BIN`: the build's `llama-server` there.
- `MODEL`: Flint Pascal Q8 GGUF on the rig.
- `TEXT_LOCAL`: the recorded agent text, the release check's `proxy.jsonl.bodies.gz`, as in R176 and the Flash-Next
  half.

```bash
export RIG=<rig> WORK=<folder on the rig> \
  BIN=/opt/llama-builds/llama.cpp-20261117-602691153-pascal-p106-v5-cand-b12483/bin/llama-server \
  MODEL=<models>/flint-near-q8-p065.gguf TEXT_LOCAL=<proxy.jsonl.bodies.gz>
R=docs/data/2026-11-19_r205_flint_glu/run/r205f.sh
STEP=dry bash $R      # the three server lines, nothing started
STEP=start bash $R    # the lane, detached on the rig
STEP=wait bash $R     # returns when the lane has ended (here 1 h 15 min)
OUT=<folder> STEP=collect bash $R
# the tables, on any machine with Python 3 (here the machine that drove the run):
D=docs/data/2026-11-19_r205_flint_glu
python3 -I bench/deep_lane/deep_table.py <folder> --control off --out $D/table.md --tsv $D/classes.tsv
python3 -I bench/deep_lane/deep_table.py <folder> --control off --class-on pass1 --tsv $D/classes_pass1.tsv
python3 -I bench/deep_lane/deep_table.py <folder> --control off --class-on pass2 --tsv $D/classes_pass2.tsv
python3 -I bench/deep_lane/deep_table.py <folder> --control u1 --tsv $D/classes_u2_vs_u1.tsv
python3 -I bench/deep_lane/deep_table.py <folder> --control u1 --class-on pass2 --tsv $D/classes_u2_vs_u1_pass2.tsv
python3 -I docs/data/2026-11-18_c865b_ornith_glu_rig1/run/memory_table.py <folder> $D/memory.tsv
cp <folder>/requests.csv <folder>/loads.tsv $D/
```

The ms a token above are `gen_ms / gen_n` from `requests.csv`.

**Provenance:**
- The model file is Flint Pascal Q8 GGUF, made in this project from Unsloth's Q8_0 of Flint
  ([release/flint-v4](../../../release/flint-v4/README.md)).
- The recorded agent text is this project's own release check.
- The tables carry the licence of this repository.
