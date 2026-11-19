# Ornith-1.5-122B's passes of 2, 3 and 4 tokens with the read-once switches (R197, Q1580; 2026-11-19)

What this measured: whether switches already in the v5 engine make Ornith-1.5-122B Pascal N2P GGUF (`valid1`) cheaper
when one pass checks 2, 3 or 4 tokens, the pass any draft model needs. R176 found that on v4.0 a second token in a
pass nearly doubles the cards' work, half of it Ornith's dense Q5_0 and Q8_0 products
([2026-11-17_r176_ornith_step_map](../2026-11-17_r176_ornith_step_map/README.md)). Here arm `base` is Ornith's v5
preset ([release/ornith-v5/settings.env](../../../release/ornith-v5/settings.env)), and arm `ro` is the same preset
with three switches added: `GGML_CUDA_MMVQ_PASCAL_MT=1`, `GGML_CUDA_MMVQ_PASCAL_KCOL_SKIP=q5_0,q8_0` and
`GGML_CUDA_MMVQ_PASCAL_GLU_MT=1` ([docs/switches.md](../../switches.md)). Each arm ran two loads per pass width, at 0
and 8,192 tokens of context. The run used R176's method and nine P106-090 cards, **overclocked**: the second rig's
soaked clock profile ([2026-11-19_r209_oc_v5](../2026-11-19_r209_oc_v5/README.md)) was set before the run and left
alone, so every load ran on one boot and one set of clocks. Registry row Q1580 in
[quants/experiments.md](../../../quants/experiments.md).

Each value is one load. Two values in a cell are the arm's two loads, in run order.

## Result in plain words

- **The switches make a pass of 2 or 3 tokens 3.47 % and 3.68 % cheaper, and a pass of 4 tokens 3.68 % dearer, at depth 0.**
  The verify pass costs 82.98 and 83.08 GPU ms at 2 tokens, against 85.99 and 86.02 without the switches (−3.47 %).
  At 3 tokens it is 107.36 and 107.42 against 111.50 and 111.49 (−3.68 %). At 4 tokens it is 137.82 and 137.67
  against 132.89 and 132.82 (+3.68 %). Each arm's two loads sit within 0.15 ms of each other, so these are a gain at
  2 and 3 tokens and a loss at 4, at both depths.
- **The change is all in Ornith's dense Q5_0 and Q8_0 products, and it depends on their shape.** The kernel that reads
  each weight once wins on the 8,192-wide products (`ssm_out` 9.27 → 6.71 ms at 2 tokens). At 2 and 4 tokens it loses
  on the 3,072-wide ones (`attn_qkv` 14.16 → 15.57 ms at 4 tokens). At 3 tokens it wins on all of them. Picking the
  faster kernel per product would save est. 3.63, 4.20 and 0.78 ms a pass at 2, 3 and 4 tokens (new code row Q1625).
- **A draft still cannot pay on Ornith.** On this boot one token takes 45.60 ms at the start of a context and 55.17
  at 8,192 tokens. That is 15.13 % and 14.35 % less than R176's 53.72 and 64.41, from the v5 preset and the overclock
  together. The multi-token passes shrank less, so a draft now needs more accepted tokens a step, not fewer. With the
  switches, a draft of two tokens (a pass of 3) needs est. 2.73 tokens a step at depth 0 and 2.63 at 8,192, against
  2.82 and 2.72 without them. Ornith's own MTP layer drafting two tokens wrote 2.56 and 2.61 tokens a step on its
  replies without the switches, and 2.67 and 2.06 with them.
- **The replies differ between the arms from their second token on.** Every load of an arm wrote exactly the same
  256 tokens, and each reply reads as a coherent answer. Without the switches the replies are R176's on v4.0, token
  for token.
- **`GGML_CUDA_MMVQ_PASCAL_GLU_MT=1` does nothing on Ornith.** Its kernel takes only Q2_0, Q3_K, Q4_K, Q5_K and
  IQ4_XS pairs, and Ornith's shared expert is Q5_0 (below, "What the switches reach").

## Result

### The verify pass

GPU ms a pass, the nine cards, the repeated request (`p2`). The pass is the group profile's launch groups summed over
the typical verify passes ([compare.md](compare.md) table 1; the first request, [compare_p1.md](compare_p1.md),
agrees within 0.26 ms):

| Width | Depth | base | ro | ro − base, paired | Change of the means | Spread of each arm's two loads (base, ro) | Class |
|---:|---:|---:|---:|---:|---:|---:|---|
| 2 | 0 | 85.99, 86.02 | 82.98, 83.08 | −3.01, −2.95 | −3.47 % | 0.03, 0.10 | **gain** |
| 2 | 8,192 | 97.67, 97.68 | 94.30, 94.41 | −3.37, −3.27 | −3.40 % | 0.01, 0.11 | **gain** |
| 3 | 0 | 111.50, 111.49 | 107.36, 107.42 | −4.14, −4.06 | −3.68 % | 0.01, 0.06 | **gain** |
| 3 | 8,192 | 129.32, 129.38 | 124.68, 124.72 | −4.63, −4.66 | −3.59 % | 0.06, 0.03 | **gain** |
| 4 | 0 | 132.89, 132.82 | 137.82, 137.67 | +4.94, +4.85 | +3.68 % | 0.06, 0.15 | **loss** |
| 4 | 8,192 | 151.95, 152.07 | 154.53, 154.55 | +2.58, +2.48 | +1.66 % | 0.13, 0.02 | **loss** |

The class rule is fixed: a gain when both paired differences are below zero and each is larger than both arms'
spreads, a loss when both are above zero and larger than both spreads, negligible otherwise. Under the same profiler
the server's own step moved with the pass: at 3 tokens and depth 0 it was 150.78 and 150.96 ms for `base` and 146.60
and 146.66 for `ro`; at 4 tokens, 177.43 and 177.48 against 182.90 and 182.78 ([map_base.md](map_base.md),
[map_ro.md](map_ro.md) table 1).

### Where it moved: the dense Q5_0 and Q8_0 products

The 13 launch groups with a plain `MUL_MAT` of a Q5_0 or Q8_0 tensor: the recurrent and attention layers' products,
the shared expert and the output head. Their sum, ms a pass ([compare.md](compare.md) table 3):

| Width | Depth | base | ro | ro − base, paired | Change |
|---:|---:|---:|---:|---:|---:|
| 2 | 0 | 43.29, 43.21 | 40.04, 40.01 | −3.25, −3.19 | −7.44 % |
| 2 | 8,192 | 43.26, 43.21 | 40.07, 40.06 | −3.19, −3.15 | −7.33 % |
| 3 | 0 | 53.66, 53.68 | 49.49, 49.46 | −4.17, −4.23 | −7.82 % |
| 3 | 8,192 | 53.75, 53.80 | 49.58, 49.59 | −4.17, −4.22 | −7.80 % |
| 4 | 0 | 62.77, 62.78 | 66.15, 66.05 | +3.38, +3.27 | +5.30 % |
| 4 | 8,192 | 62.85, 62.90 | 66.25, 66.26 | +3.41, +3.35 | +5.37 % |

These products have the same shapes whatever the reply, so their sums compare like for like. Product by product at
depth 0, the mean of each arm's two loads, ms a pass ([compare.md](compare.md) table 3, `compare.tsv.gz` rows
`label`):

| Product (the label's shape: values in × values out) | 2 tokens: base → ro | 3 tokens: base → ro | 4 tokens: base → ro |
|---|---|---|---|
| `ssm_out` Q5_0, 8192 × 3072 | 9.27 → 6.71 | 11.44 → 9.94 | 12.49 → 11.97 |
| `attn_output` Q5_0, 8192 × 3072 | 3.12 → 2.34 | 3.84 → 3.37 | 4.20 → 4.10 |
| `attn_qkv` Q5_0, 3072 × 12288 | 9.41 → 9.55 | 11.79 → 11.15 | 14.16 → 15.57 |
| `attn_gate` Q5_0, 3072 × 8192 | 6.31 → 6.40 | 7.94 → 7.51 | 9.48 → 10.46 |
| `output` (the head) Q5_0, 3072 × 248320 | 4.96 → 5.05 | 6.31 → 5.97 | 7.61 → 8.38 |
| `attn_q` Q5_0, 3072 × 16384 | 4.08 → 4.14 | 5.15 → 4.87 | 6.18 → 6.82 |
| `attn_k`, `attn_v` Q8_0, 3072 × 512 | 0.31 → 0.21, 0.35 → 0.27 | 0.36 → 0.24, 0.40 → 0.31 | 0.40 → 0.31, 0.45 → 0.38 |
| the shared expert's gate, up, down (Q5_0) | 1.63 → 1.58, 1.32 → 1.28, 1.56 → 1.61 | 1.97 → 1.85, 1.61 → 1.50, 1.99 → 1.95 | 2.37 → 2.47, 2.00 → 2.11, 2.47 → 2.48 |
| **all 13: base, ro, the faster of the two per product (est.)** | **43.25, 40.03, 39.61** | **53.67, 49.47, 49.47** | **62.77, 66.10, 62.00** |

- **The products 8,192 values wide** (`ssm_out`, `attn_output`) gain at every width.
- **The products 3,072 values wide lose a little at 2 tokens** (+0.06 to +0.13 ms each) **and a lot at 4**
  (`attn_qkv` +1.41, `attn_gate` +0.98, the head +0.76, `attn_q` +0.64). At 3 tokens they gain.
- **The rest of the pass moved by the reply, not the switches.** The arms wrote different replies, so their tokens
  picked different routed experts. The routed families moved by −0.16 to +0.95 ms a pass, with signs that change
  with width and depth, and `FLASH_ATTN_EXT` moved by up to 0.18 ms ([compare.md](compare.md) table 2).

### The one-token step on this boot

The host-profile load (`w1h1`, nothing synchronized), the repeated request, ms ([map_base.md](map_base.md) tables 1
and 6):

| Part | Depth 0 | Depth 8,192 |
|---|---:|---:|
| **the step** | **45.60** | **55.17** |
| the server's ms a token (gen_ms / 256) | 45.63 | 55.08 |
| the nine cards' GPU time | 44.05 | 53.46 |
| the cards' part less their GPU time: hops, launch latency, gaps (est.) | 1.22 | 1.32 |
| the host before the pass's inputs are set; after the token | 0.13; 0.19 | 0.19; 0.20 |
| R176 on v4.0 at stock clocks, the step (two loads) | 53.72, 53.73 | 64.41, 64.42 |

The 15.13 % (depth 0) and 14.35 % (8,192) less than R176 is the v5 engine with Ornith's v5 preset and this rig's
overclock together; this run does not split them (R209's agent runs do: [2026-11-19_r209_oc_v5](../2026-11-19_r209_oc_v5/README.md)).
The switches are not on in this load. From the source they cannot change a one-token pass of Ornith's file (below).

### The break-even a draft needs

Tokens a step a draft must reach for a pass of k tokens to beat k one-token steps, est. ([compare.md](compare.md)
table 5). The group profile synchronizes after every evaluation, so the steps of its loads are not the served
steps. The step of k tokens is built from parts instead:
- this run's verify pass, put on the host profile's footing by R176's ratio of the two profilers at the same width
  (1.0068, 1.0062 and 1.0093 at 2, 3 and 4 tokens, depth 0);
- R176's parts outside the verify pass's GPU time: the verify's rest, and the draft's span with its GPU part scaled
  by this boot's one-token GPU time over R176's.

The break-even is that step over this boot's one-token step. The method gives back R176's own measured break-evens
from R176's data (2.02, 2.68 and 3.27 against 2.02, 2.68 and 3.28 at depth 0).

| Width | Depth | base: the step (est.), break-even | ro: the step (est.), break-even | A verify alone, base and ro (any draft's floor) | R176 on v4.0 at stock: step, break-even |
|---:|---:|---|---|---|---|
| 2 | 0 | 97.39, **2.14** | 94.39, **2.07** | 1.98, 1.92 | 108.38, 2.02 |
| 2 | 8,192 | 111.06, **2.01** | 107.72, **1.95** | 1.86, 1.79 | 123.60, 1.92 |
| 3 | 0 | 128.42, **2.82** | 124.30, **2.73** | 2.56, 2.47 | 143.96, 2.68 |
| 3 | 8,192 | 150.02, **2.72** | 145.35, **2.63** | 2.45, 2.36 | 167.91, 2.61 |
| 4 | 0 | 155.68, **3.41** | 160.62, **3.52** | 3.06, 3.17 | 176.00, 3.28 |
| 4 | 8,192 | 179.62, **3.26** | 182.16, **3.30** | 2.88, 2.93 | 202.27, 3.14 |

Ornith's own MTP layer on this greedy reply wrote 1.90, 2.56 and 2.81 tokens a step at 2, 3 and 4 tokens without the
switches at depth 0, and 1.90, 2.67 and 3.24 with them (other replies; [requests.csv](requests.csv)). So with the
switches, two drafted tokens give est. 46.60 ms a token at depth 0 against 45.60 without a draft, and four tokens a
pass give est. 49.56. At 8,192 tokens the `ro` reply was drafted badly (2.06 tokens a step at 3) and the `base` reply
gave est. 57.44 against 55.17. One reply per arm and depth is not an acceptance rate. What carries over is the
break-even column, which is set by the passes alone.

### The replies

md5 of each reply's 256 token ids, first 8 characters ([compare.md](compare.md) table 6):

| Width | Depth | base | ro | First token where they differ | Same as the one-token reply | Same as R176's at this width (v4.0) |
|---:|---:|---|---|---:|---|---|
| 1 | 0 | `b8279751` (w1h1) | | | | `b8279751`: yes |
| 1 | 8,192 | `eea8bef3` (w1h1) | | | | `eea8bef3`: yes |
| 2 | 0 | `b8279751`, `b8279751` | `f1051d98`, `f1051d98` | 1 | base | base |
| 2 | 8,192 | `dacfc86f`, `dacfc86f` | `bb0720f0`, `bb0720f0` | 1 | neither | base |
| 3 | 0 | `b8279751`, `b8279751` | `f1051d98`, `f1051d98` | 1 | base | base |
| 3 | 8,192 | `dacfc86f`, `dacfc86f` | `bb0720f0`, `bb0720f0` | 1 | neither | base |
| 4 | 0 | `69f664b1`, `69f664b1` | `4b74571a`, `4b74571a` | 8 | neither | base |
| 4 | 8,192 | `b5bc3400`, `b5bc3400` | `0aa44b4a`, `0aa44b4a` | 1 | neither | base |

Token 0 comes from the prompt pass, the same in every load. Every reply after it was read: each is a coherent start
of an answer to the cut agent prompt. For example, at depth 0, `base` writes "The user is asking me to create a new
file `test.py` …" and `ro` writes "The test file is quite large. Let me look at the `test_encode` function …". The
deep lane cuts its prompts mid-text, so the first tokens are near ties, and a different sum order flips them. The read-once
kernel keeps each token's sums those of the one-token kernel. The k-column kernel it replaces does not. Neither arm's
verify pass matches one-token decoding exactly, since the routed experts and attention also run as passes of k
tokens. Without the switches the v5 engine with its preset wrote R176's v4.0 replies at every width and depth.

## What the switches reach (the engine at `602691153`)

Read from the source before the run, then seen in the run:

- **`GGML_CUDA_MMVQ_PASCAL_KCOL_SKIP=q5_0,q8_0`** takes Q5_0 and Q8_0 plain products of 1 to 4 tokens away from the
  k-column kernels (`mmvq-pascal-kcol.cu`). At one token the k-column kernels take Q8_0 only in rows whose 64-weight
  chunks are not a multiple of four, and Q5_0 not at all. Every Q5_0 and Q8_0 product in Ornith's main passes takes
  1,024, 3,072 or 8,192 values in (16, 48 and 128 chunks), so one-token passes are unchanged.
- **`GGML_CUDA_MMVQ_PASCAL_MT=1`** then runs those products of 2 to 4 tokens in `mul_mat_vec_q_pascal_mt`, which
  decodes each weight once for every token, with each token's sums those of the one-token kernel (Q211). Without it
  they would go to `mul_mat_vec_q_pascal_mc`. Ornith's routed N2P and N3P experts never reach it: the multi-token
  expert kernel takes them first, as before.
- **`GGML_CUDA_MMVQ_PASCAL_GLU_MT=1`** fuses a gate and up product of 2 to 8 tokens and their SWIGLU into one
  launch, but only for Q2_0, Q3_K, Q4_K, Q5_K and IQ4_XS weights (`mmvq-pascal-glu.cu`, `glu_kind_of`). Ornith's
  shared expert is Q5_0 and its routed experts are `MUL_MAT_ID`, which this kernel never matches. Its gate, up and
  SWIGLU stayed three launches in both arms.
- **With Ornith's v5 preset, `GGML_CUDA_MMID_GLU_UNFUSE=2`:** that switch stops llama.cpp's fused matrix-vector
  launches on Pascal (`ggml_cuda_should_fuse_mul_mat_vec_q`). At 2 to 4 tokens it splits only the routed experts'
  gate, up and GLU, since a plain product is fused only at one token. The three read-once switches decide
  separately, so the two do not interact: `GLU_MT`'s match does not go through that check, and the dense products
  at 2 to 4 tokens were never fused.

## What ran

- **Rig and build:** a nine-card P106-090 rig (Manli M-P106L9-N6G, the stock 4 GB host), nothing else running; the
  v5 engine `pascal-p106-v5` at `602691153` (upstream b12483 merged; the build reports `6026911533`), md5 of
  `llama-server` `86607661d5a19d1a90d19ee4db10c90b`.
- **Clocks:** the rig's soaked profile, every card's memory and core offset as in
  [2026-11-19_r209_oc_v5](../2026-11-19_r209_oc_v5/README.md), set and checked about 30 minutes before the run, with
  the clock kit's watchdog running (70 °C, 80 W for 5 s). Every card read back at its value before and after the run.
  The watchdog never tripped, and the boot logged no GPU error. Over the run the busy SM clock medians were 2,100 to
  2,164 MHz, the hottest card reached 41 °C and the highest single power sample was 91.92 W
  ([cards.md](cards.md)). No card ran busy below 1,800 MHz for 2 s or more.
- **Model:** Ornith-1.5-122B Pascal N2P GGUF (`valid1`), 48,196,722,592 bytes, as R176.
- **The driver:** [bench/deep_lane](../../../bench/deep_lane/README.md) through [run/r197.sh](run/r197.sh), one detached
  run, 14:40:23 to 18:01:25. Per depth (0, 8,192) a load sends a prime, decode pass 1 (the first request at the
  depth: 256 tokens at temperature 0, end of text ignored) and decode pass 2 (the same request again), each after
  4 idle seconds.
- **The arms, each its own load, in this order:** `w1h1` (the preset, one token a pass, host profile and step trace;
  it also warms the rig for the loads after it), then `w2b1 w2r1 w3b1 w3r1 w4b1 w4r1`, then `w2r2 w2b2 w3r2 w3b2 w4r2
  w4b2`. Within each width the second round runs the other way round, so each arm's two loads are about 1 h 30 min
  apart.
  - `w<k>b<n>` is the preset on Ornith's MTP line drafting k − 1 tokens a step (`--spec-type draft-mtp
    --spec-draft-n-max k-1`, `-ts 6,6,6,6,6,5,5,5,5 -ub 256 -c 16384`, as R176), with the group profile
    (`GGML_CUDA_PROFILE` with `GGML_CUDA_PROFILE_MODE=group`).
  - `w<k>r<n>` is the same plus the three switches.
  - Every load's line as it ran is in [loads.tsv](loads.tsv).
- **Loading:** each load spent 597 to 666 s loading the 48 GB file from a USB stick. At 4 tokens and 8,192 tokens of
  context, card 4 lacked 20.70 MiB for a checkpoint store in all four 4-token loads of both arms, as in R176, and the
  server kept those checkpoints in host memory. That slows a request's restore, not the steps.

## How it was read

On the operator's machine after the run, with standard-library Python 3 ([run/tables.sh](run/tables.sh), 3 min 5 s
for the 12 group traces; R176's chain):

1. R121's [split_trace.py](../2026-11-17_r121_flash_step_map/run/split_trace.py) cuts each profile into one file per
   generation request.
2. The one-token load: S37's [step_split.py](../2026-11-10_served_step_split/run/step_split.py) and R176's
   [host_split.py](../2026-11-17_r176_ornith_step_map/run/host_split.py).
3. Each group-profile request: R49's [verify_parts.py](../2026-11-10_served_verify_parts/run/verify_parts.py) (R41's
   parts, 7.6 µs subtracted per launch group, the MTP layer's `eh_proj` graphs counted as the draft's).
4. R176's [step_map.py](../2026-11-17_r176_ornith_step_map/run/step_map.py), once over each arm's loads (each arm's
   loads named `w<k>g<n>` for it): [map_base.md](map_base.md), [map_ro.md](map_ro.md) and their `_p1` versions.
5. [run/compare.py](run/compare.py): the arms against each other, the class, the dense products, the break-even and
   the replies ([compare.md](compare.md), [compare_p1.md](compare_p1.md)).

## Commands

```bash
# the run, from the repository root on a machine that runs only ssh and tar; the rig's paths in the environment
RIG=<rig alias> WORK=<run folder on the rig> BIN=<v5 build on the rig>/bin/llama-server \
  MODEL=<models>/Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf \
  TEXT_LOCAL=<the release check's proxy.jsonl.bodies.gz> DRY=1 STEP=start \
  bash docs/data/2026-11-19_r197_ornith_passes/run/r197.sh
# then the same without DRY=1, then STEP=wait, then STEP=collect OUT=<folder>

# the tables, on any machine with Python 3: S this repository, OUT the run's results, RES the tables
S=<this repository> OUT=<results> RES=<tables> bash docs/data/2026-11-19_r197_ornith_passes/run/tables.sh
```

The twelve group-profile traces (614 MB together) are not published. Every table here comes from them, or from the
host profile and step trace, through the scripts above. Their sizes and md5s:

| Trace | Bytes | md5 |
|---|---:|---|
| `gp-w2b1.csv` | 60,170,982 | `527d2634c7cdc771c167e283e5ebacab` |
| `gp-w2b2.csv` | 60,175,521 | `772e4c96ee67f2d5392abfe39f46fd9e` |
| `gp-w2r1.csv` | 63,337,437 | `e98e3ac240d1de643ea7a66bab467be3` |
| `gp-w2r2.csv` | 63,341,713 | `55aae93eace170996fa1fac18668d5bb` |
| `gp-w3b1.csv` | 46,846,527 | `0122f5895b470e3409f5fd48961460e4` |
| `gp-w3b2.csv` | 46,845,392 | `c63a926c90e5ad8b168ab4a3a43e38a7` |
| `gp-w3r1.csv` | 51,382,352 | `384ffb3a3d7c2673cd041792dffba4ae` |
| `gp-w3r2.csv` | 51,382,494 | `7a168a206243698ade677e71e3baf625` |
| `gp-w4b1.csv` | 43,104,163 | `6bbe8f792932a0ff82cd9d716b83801a` |
| `gp-w4b2.csv` | 43,104,834 | `53aed5fe14c1ea4184a734877cbf6af6` |
| `gp-w4r1.csv` | 41,893,263 | `479299016bc4dc570a4bdee7e01423dd` |
| `gp-w4r2.csv` | 41,893,064 | `52f7d0b81ac77c2ca79b7d4d04dd7b31` |

## Files

| File | What it holds |
|---|---|
| [compare.md](compare.md), [compare_p1.md](compare_p1.md) | `compare.py`: the pass, families, the dense Q5_0 and Q8_0 products, the one-token step, the break-even and the replies, base against ro, for the repeated and the first request |
| `compare.tsv.gz`, `compare_p1.tsv.gz` | every value of those tables in long form: table, width, depth, request, arm, load, name, value (rows `label`: every launch group at depth 0) |
| [map_base.md](map_base.md), [map_ro.md](map_ro.md), [map_base_p1.md](map_base_p1.md), [map_ro_p1.md](map_ro_p1.md) | R176's `step_map.py` per arm: the step, the pass by part and bucket, what grows with depth and width by label, the families, the one-token step's host part |
| `map_base.tsv.gz`, `map_ro.tsv.gz`, `map_base_p1.tsv.gz`, `map_ro_p1.tsv.gz` | every value of the maps in long form, `verify_parts.py`'s labels, parts and buckets of every load and request among them |
| [requests.csv](requests.csv) | every request of every load: prompt and generation times, drafted and accepted tokens, the reply's md5 |
| [loads.tsv](loads.tsv) | every load's environment and server line as it ran (paths as `<OUT>`, `<MODELS>`, `<BIN>`) |
| [table.md](table.md), [deep_classes.tsv](deep_classes.tsv) | the deep lane's own tables (its classes against `w1h1` are not this folder's measure: the arms are widths and profilers) |
| [cards.md](cards.md) | the card log over the run (`scripts/card_table.py`): each card's busy clock, peak power and temperature, throttle reasons |
| `split/w1h1*`, `host/w1h1*` | `step_split.py` and `host_split.py` for the one-token load |
| [run/r197.sh](run/r197.sh), [run/tables.sh](run/tables.sh), [run/compare.py](run/compare.py) | the run and the reading |

## Limits

- **One reply per arm, width and depth, greedy.** The tokens a step are this reply's, and the arms wrote different
  replies. The per-pass costs and the break-even do not depend on the reply. The routed experts' time does, slightly
  (above), and the dense products' does not.
- **The break-even is an estimate:** this run's passes with R176's parts outside the verify pass's GPU time, measured
  on v4.0 at stock clocks. Those parts are host work, hops and the non-GPU part of the draft's span
  (5.26 to 9.22 ms a step in R176 at these widths). The one-token step it divides by is measured on this boot.
- **Overclocked rig:** every load ran on the same boot and clocks, so the comparison holds. The absolute times are
  this rig's with its profile, not stock clocks.
- **The group profile synchronizes after every evaluation,** so its server steps (119.35 to 207.73 ms) are not the
  served steps. Its pass times are this folder's kernel figures; R176 found them 0.6 % to 0.9 % under the host
  profile's GPU time at the same widths.
- **No kernel cases and no correctness check on this rig:** the read-once kernel's correctness on a P106 is Q211's,
  and the replies are compared above.

## Provenance and licence

Code: Miner Llama (MIT, like llama.cpp), the v5 engine. Model: Ornith-1.5-122B by ornith-ai, quantized for this
project with Unsloth's importance matrix (MIT). The prompts are the deep lane's recorded agent text
([bench/deep_lane](../../../bench/deep_lane/README.md)). The tables hold times, counts, kernel labels, md5s and the
first words of two replies, which the model wrote. Measured on a Manli M-P106L9-N6G (nine P106-090).
