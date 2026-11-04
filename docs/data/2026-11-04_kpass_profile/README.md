# C266 / Q822: why a 5-token pass costs 2.6 one-token passes on the P106s (2026-11-04)

[Q591](../2026-11-04_kcurve_p106/README.md) measured T(5) / T(1) = 2.63 for whole-model passes of Swift
Qwen3.8-Flash-Next GSQ-RCO Q2_0 on nine P106-090s, against the ≤ 1.3 the k-column and multi-token kernels aimed at.
This folder profiles the same passes of 1 and 5 tokens with the fork's light profiler (Q681,
[light profiler](../2026-11-04_light_profiler/README.md)) to find which parts of the model grow. Fork branch
`pascal-p106-v3-b12076-ladder3` `7bc1552f4`, the published Q2_0 file (BF16 hyper-connections), every switch at its
default ("on"), and with the k-column and multi-token kernels off (`GGML_CUDA_MMVQ_PASCAL_KCOL=0
GGML_CUDA_MMVQ_PASCAL_MTOK=0`, "off"). The profiler's own cost per timed span, 7.6 µs on a P106
([C140](../2026-11-04_megakernel_rig/README.md)), is subtracted from every span.

## Where the time goes, ms per pass (kernels on)

| Part | 1 token | 5 tokens | 5 / 1 | growth | off: 1 token | off: 5 tokens |
|---|---:|---:|---:|---:|---:|---:|
| **routed experts** (Q2_0) | 8.69 | 55.49 | 6.39 | **+46.8** | 8.68 | 72.37 |
| **attention and recurrent projections** (Q3_K, IQ4_XS, Q4_K, ...) | 27.13 | 61.77 | 2.28 | **+34.6** | 22.27 | 68.55 |
| **hops between cards, host work and gaps** | 11.6 | 40.5 | 3.50 | **+28.9** | 16.8 | 48.4 |
| hyper-connections (BF16 products 9.11 → 19.56, mixing ops 0.92 → 2.68) | 10.03 | 22.25 | 2.22 | +12.2 | 10.07 | 22.24 |
| router (F32) | 2.15 | 5.85 | 2.72 | +3.7 | 2.14 | 5.84 |
| shared expert | 3.24 | 6.36 | 1.96 | +3.1 | 2.99 | 8.21 |
| Gated DeltaNet state and conv | 3.46 | 5.46 | 1.58 | +2.0 | 3.46 | 5.45 |
| other operations without weights | 9.22 | 10.58 | 1.15 | +1.4 | 9.24 | 10.48 |
| attention (FLASH_ATTN_EXT) | 0.63 | 1.69 | 2.68 | +1.1 | 0.62 | 1.69 |
| other weights | 0.59 | 1.25 | 2.11 | +0.7 | 0.60 | 1.55 |
| output layer (Q5_K; these passes ask for the last token's logits only) | 3.58 | 3.58 | 1.00 | 0.0 | 3.58 | 3.58 |
| **the cards' GPU time** (sum of the parts) | 68.7 | 174.3 | 2.54 | +105.6 | 63.6 | 200.0 |
| **the pass, unprofiled** (Q591) | 82.4 | 217.1 | 2.63 | +134.7 | 82.5 | 250.7 |

The parts are the group-mode trace's launch groups by their labels (S21's categories); the hops row is the unprofiled
pass minus the cards' whole evaluations from the graph-mode trace (`hops, host work and gaps` in [parts.tsv](parts.tsv)).

- **The routed experts grow 6.4×.** Each token picks its own 10 of 512 experts, so a 5-token pass reads ~48 distinct
  experts per layer (Q591's bench, with llama-bench's random tokens; real text overlaps more): their bytes grow ~4.8×
  whatever the kernel. Beyond that, the routed down projection never takes the multi-token kernel (Q825: its 180-byte
  rows fail both staging checks) and runs per token. The multi-token gate/up still saves 16.9 ms at 5 tokens (off:
  72.4).
- **The attention and recurrent projections grow 2.3×** although every weight is read once: the k-column kernels' default
  launch shapes are slow on these shapes (Q823: a 32 KB budget makes the whole pass 7.5–21.8 ms faster) and lose to the
  earlier kernels at some types and token counts (Q824); at one token the k-column Q3_K path costs 4.9 ms more GPU time
  than the one-token kernel (22.27 off against 27.13 on).
- **The hops between cards grow 3.5×.** Five tokens' residual crosses each of the eight card boundaries in F32 (C140:
  0.455 ms for one token, ~2.1 ms for five), and the host work per card grows with the pass. Q8_0 hops (C200,
  `LLAMA_HOP_CAST`) cut the bytes 4×.
- **The hyper-connections grow 2.2×**, their BF16 products 2.15× where the kernel bench reaches 1.63× at five tokens
  (Q591's bf16.csv): ~5 ms of room in the 5-token shape.

**These passes use llama-bench's prompt tokens, which are random token ids**, and so is Q591's T(k): their routing is
close to uniform (the ~48 distinct experts per layer at five tokens is Q591's bench under uniform routing, not a count
from these passes). Consecutive tokens of real agent text read fewer: 16.3 / 21.6 / 26.4 / 30.9 distinct experts per layer
at k = 2–5 (Q860's dry runs on real text, against ~19.9 / 29.4 / 38.9 / 48.3 uniform). At the measured ~1.22 ms per pass
per extra distinct expert ((55.49 − 8.69) / (48.3 − 10)), real text's routed experts cost est. ~34 ms at five tokens
instead of 55.5 (about 30.9 / 48 of the share), T(5) est. ~196 ms at the defaults instead of 217.1 and ~184 instead of
205.3 with Q591's 32 KB launch budget: T(5) / T(1) est. 2.4–2.5 rather than 2.6–2.7. Q827 measures it on real text.

## The profiler against the unprofiled pass

| Trace | 1 token | 5 tokens |
|---|---|---|
| `graph` (an event pair around each card's evaluation): the cards' GPU time | 70.80 ms | 176.56 ms |
| `layer` (an event where the layer changes): the sum of the layer spans | 70.56 ms | 176.25 ms |
| `group` (an event between every two launch groups, 7.6 µs each subtracted): the sum of the parts | 68.72 ms | 174.30 ms |
| the pass under the `graph` profiler, wall (host clock) | 83.5 ms | 222.9 ms |
| the pass unprofiled (Q591) | 82.4 ms | 217.1 ms |

`layer` and `graph` agree within 0.3 ms, `group` with the correction within 1–3 %: C140's calibration holds in the
model. All three sum the cards' timelines only, so each sits 11–12 ms (1 token) and 40–43 ms (5 tokens) below the
unprofiled wall; that difference is the hops between cards, the host's work and its gaps. The `graph` profiler costs
1–3 % of the wall; the `group` profiler 28 % at one token (105.8 ms) and 10 % at five (239.9 ms).

`sample=8` (an event pair around every 8th launch group, rotating) under llama-bench with 10 repetitions wrote only each
phase's eager and capture evaluations: a replay's events are read at the same phase's next evaluation, 8 later, and
nothing flushes them at exit (Q826). It ran again with 40 repetitions (below).

## Files

| File | What it is |
|---|---|
| [parts.tsv](parts.tsv) | per trace (`on_` / `off_` with the mode), k and part: ms per pass |
| [growth.tsv](growth.tsv) | per trace and part: ms at 1 and 5 tokens, their ratio and the growth |
| [summaries/](summaries/) | `prof_summary.py`'s report per trace and k (per card, per layer or per launch group, the first ops, the weight types, the check against the unprofiled pass) |
| [run/kpass_prof.sh](run/kpass_prof.sh) | the rig run: one llama-bench load per arm and mode |
| [run/kpass_parts.py](run/kpass_parts.py) | the split of each trace by test, `prof_summary.py` per k, and the parts |

## Run

On the rig, idle, from a folder holding S21's `verify_cost.sh`, `mem_guard.sh` and `smoke_request.json` in `run/`
([verify cost run/](../2026-11-03_verify_cost/run/), [MTP run/](../2026-11-03_flash_next_mtp/run/)), ~5 minutes an arm and
mode:

```bash
BIN=<-ladder3 build>/bin MODEL=<Swift Q2_0 first shard> OUT=<dir> setsid nohup bash run/kpass_prof.sh > kpass.console 2>&1 < /dev/null &
MODES=sample=8 REPS=40 OUT=<dir2> ...   # the sample traces need at least 3 evaluations per phase
```

Each run is `llama-bench -m MODEL -ngl 99 -fa on -b 2048 -ub 512 -ts 6/6/6/6/5/6/5/6/3 -lm dio --no-host 1 -p 1,5 -n 0 -r 10`
under `GGML_CUDA_PROFILE=<csv>` and `GGML_CUDA_PROFILE_MODE=<mode>`. Then, on any machine with Python 3:

```bash
python3 -I run/kpass_parts.py --prof-summary scripts/prof_summary.py --event-us 7.6 --ks 1,5 --wall 1=82.4,5=217.1 \
  --out res on_graph=<on graph csv> on_layer=<on layer csv>
python3 -I run/kpass_parts.py --prof-summary scripts/prof_summary.py --event-us 7.6 --ks 1,5 --per-test 11 \
  --out resg on_group=<on group csv> off_group=<off group csv>
```

The light modes' traces split by test through their graph ids (a pass of another size is another graph); the group
traces, which carry none, by count (`--per-test 11`: llama-bench evaluates each card once for the warm-up and once per
repetition; the warm-up is left out). The traces themselves (0.01–2.7 MB each) are not published.
