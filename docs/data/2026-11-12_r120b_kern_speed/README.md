# The v4 kernel merge on a P106, speed (R120b, Q1391, 2026-11-12)

What this is: the speed of the six verify-pass kernels in `pascal-p106-v4` `d097d6d88` that had never been timed on a
P106, measured on a P106-090 rig after R120a found them exact
([2026-11-12_r120a_kern_exact](../2026-11-12_r120a_kern_exact/README.md)). Three come from
[the v4 kernel merge](../2026-11-11_v4_kernel_merge/README.md): C727's Q6_K products of 2-4 tokens
(`GGML_CUDA_MMVQ_PASCAL_Q6K_MT`), C728's K tiles (`_DMT_KT`) and C729's products of 5-8 tokens (`_RO8`). The other three
come from the day merge: C726's Q3_K products ([`_Q3K_MT`](../2026-11-11_c726_q3k_multitoken/README.md)), C757's one
launch for the shared expert's gate, up and SWIGLU ([`_GLU_MT`](../2026-11-11_c757_shexp_fused/README.md)) and C754's F32
products of 4-8 tokens on `mul_mat_vec_f` ([`GGML_CUDA_MMVF_F32_MT`](../2026-11-11_c754_f32_mmvf/README.md)). Registry
[Q1391](../../../quants/experiments.md), and Q1285, Q1286, Q1287, Q1288, Q1330 and Q1333 for the single switches.

One detached run of [run/rig_kern.sh](../2026-11-11_v4_kernel_merge/run/rig_kern.sh) with `STEPS='perf served'`, on a
nine-card P106-090 rig with nothing else running, with the checked `-kern` rig tarball
(`llama.cpp-20261111-d097d6d88-pascal-p106-v4-b12246-kern.tar.gz`, md5 `37a4c3731f76455dce14013210b2fac7`), 15:15:38 to
16:20:36:

- **perf** (15:15:38-15:32:24, card 0): `test-backend-ops perf -o MUL_MAT` at Qwen3.8-Flash-Next's IQ4_XS, Q4_K, Q5_K
  and Q6_K product shapes, 1-8 tokens, `GGML_TEST_RAW_QUANT=1`, four arms alternated over two passes: `off` (the
  preset's k-column switches `_KCOL_SMEM=32768 _KCOL_TABLE=0`), `mt` (+ the read-once kernels `_DMT=1 _Q5K_MT=1`),
  `kern` (`mt` + `_Q6K_MT=1 _DMT_KT=1 _RO8=1`) and `kernm2` (`kern` + `_DMT_KT_M=2`);
- **served** (15:32:25-16:20:36): R49's profile of a served verify pass
  ([2026-11-10_served_verify_parts](../2026-11-10_served_verify_parts/README.md), `run/verify_prof.sh`): the Miner Llama
  v3 preset exactly (UkisAI's Swift-1.5 GSQ-RCO Q2_0 of Qwen3.8-Flash-Next on nine cards, the MTP draft at three
  tokens), four loads `off`, `on`, `on`, `off`, each a `warm` load (no profile) and a `prof` load (the fork's profiler in
  group mode), each answering the greedy request twice. Both arms add `_DMT=1 _Q5K_MT=1`; `on` adds all six switches:
  `GGML_CUDA_MMVQ_PASCAL_Q6K_MT=1 GGML_CUDA_MMVQ_PASCAL_DMT_KT=1 GGML_CUDA_MMVQ_PASCAL_RO8=1
  GGML_CUDA_MMVQ_PASCAL_Q3K_MT=1 GGML_CUDA_MMVQ_PASCAL_GLU_MT=1 GGML_CUDA_MMVF_F32_MT=1`. Each server's own log lists
  the switches it read (14 set in `on`, 8 in `off`, and the profiler's two more in each `prof` load).

## Result

**The stack of six is a gain in the served verify pass: 85.970, 86.001, 85.958 and 85.991 ms a pass with it against
87.760, 87.874, 87.757 and 87.840 without (−1.828 ms, the means; the arms' spread 0.117).** Inside it, four switches gain
and two lose:

| Switch | Its products in a verify pass | `off`: 1 r1, 1 r2, 4 r1, 4 r2 (ms a pass) | `on`: 2 r1, 2 r2, 3 r1, 3 r2 | on − off | Class |
|---|---|---|---|---:|---|
| `_Q3K_MT` (C726) | the Q3_K attention and recurrent projections | 13.524, 13.529, 13.524, 13.527 | 11.025, 11.031, 11.021, 11.033 | −2.498 | gain |
| `_GLU_MT` (C757) | the shared expert's gate, up and SWIGLU | 2.481, 2.496, 2.490, 2.491 | 1.989, 1.994, 2.006, 1.995 | −0.494 | gain |
| `_Q6K_MT` (C727) | Q6_K `ssm_out` and `attn_output` | 1.583, 1.583, 1.582, 1.584 | 1.347, 1.346, 1.348, 1.352 | −0.235 | gain |
| `_DMT_KT` (C728) | IQ4_XS `ssm_out` | 1.912, 1.911, 1.915, 1.915 | 2.271, 2.276, 2.271, 2.275 | +0.360 | **loss** |
| | Q4_K `ssm_out` | 1.552, 1.549, 1.547, 1.550 | 1.508, 1.511, 1.508, 1.512 | −0.040 | gain |
| | Q4_K `attn_output` | 0.567, 0.569, 0.569, 0.569 | 0.555, 0.555, 0.555, 0.555 | −0.013 | gain |
| `GGML_CUDA_MMVF_F32_MT` (C754) | the router `ffn_gate_inp` (2,560 → 512) | 2.412, 2.405, 2.404, 2.410 | 4.072, 4.072, 4.073, 4.085 | +1.668 | **loss** |
| | the shared expert's gate `ffn_gate_inp_shexp` (2,560 → 1) | 0.610, 0.613, 0.607, 0.618 | 0.297, 0.291, 0.278, 0.276 | −0.327 | gain |
| `_RO8` (C729) | none: it takes products of 5-8 tokens, and the preset's verify passes hold at most 4 | | | | |
| | everything else | 63.118, 63.218, 63.119, 63.175 | 62.908, 62.925, 62.897, 62.908 | −0.248 | gain |
| | **the pass (sum of the parts)** | 87.760, 87.874, 87.757, 87.840 | 85.970, 86.001, 85.958, 85.991 | **−1.828** | **gain** |

([served/stack.tsv](served/stack.tsv), [run/served_stack.py](run/served_stack.py); all nine cards, `verify_parts.py`'s
cut with the profiler's 7.6 µs subtracted per launch group, requests 1 and 2 of each profiled load. A class: every `on`
value beyond every `off` value and the means apart by more than the larger of the two arms' ranges.)

- **C754's switch is a loss as a whole (+1.341 ms a pass):** `mul_mat_vec_f` takes the router's four-token product
  in 4.072, 4.072, 4.073 and 4.085 ms a pass where cuBLAS took 2.412, 2.405, 2.404 and 2.410, and the shared expert's gate in 0.297, 0.291, 0.278 and 0.276 where cuBLAS took 0.610, 0.613, 0.607 and 0.618. A rule that
  leaves the router on cuBLAS keeps the gate's gain (C807).
- **C728's tiles are a loss as a whole (+0.307 ms a pass)**, as the perf step shows for IQ4_XS at 4 tokens (C808).
- **The other products of the pass** do not move beyond their loads' spread, except "everything else" by −0.248 ms
  (hyper-connections, other ops without weights and attention, each under 0.1 ms; the arms run different replies,
  80 and 81 verify passes a request against 77 and 79).
- **The three switches that gain take est. 3.227 ms off the pass together** (the sum of their products: 2.498 + 0.494 +
  0.235; est.: not measured as one load).
- The median raw GPU time of a typical verify pass: 105.58, 105.77, 105.61 and 105.74 ms `off`, 103.26, 103.23, 103.21
  and 103.35 `on` ([served/loads/](served/loads/), `cut.tsv`).

**By name** ([served/by_name.tsv](served/by_name.tsv)), load 1 against load 2, request 1, ms a pass: the Q3_K `attn_qkv`
5.441 → 4.605, `attn_gate` 4.951 → 3.943, `attn_q` 2.144 → 1.819, `ssm_out` 0.835 → 0.544, `attn_k` 0.115 → 0.084
(`attn_v` 0.131 → 0.130); Q6_K `attn_output` 1.057 → 0.900 and `ssm_out` 0.526 → 0.447.

**The step, from the warm loads** (the server's own timings, [served/replies.tsv](served/replies.tsv); ms a step = the
reply's generation time over its steps, steps = generated tokens less accepted drafts):

| Load | Request 1: ms a step | tok/s | Request 2: ms a step | tok/s |
|---|---:|---:|---:|---:|
| 1 `off` | 131.90 | 23.29 | 109.27 | 27.78 |
| 2 `on` | 114.86 | 26.12 | 107.22 | 27.98 |
| 3 `on` | 114.42 | 26.22 | 107.19 | 27.99 |
| 4 `off` | 130.29 | 23.58 | 108.84 | 27.89 |

- **Request 2** (a context checkpoint, 4 prompt tokens read): 107.22 and 107.19 ms a step with the stack against
  109.27 and 108.84 without, about the pass's GPU time.
- **Request 1** (79 prompt tokens, the load's first request): 114.86 and 114.42 against 131.90 and 130.29. Without the stack the first request costs 22.63 and 21.45 ms a step more than the second, with it
  7.64 and 7.23. Most of that difference is one host pause of 1,067 ms before the last pass of `off`'s first reply
  (pass 85 in both `off` loads, `passes.tsv`), which no `on` reply has; the typical steps of requests 1 and 2 take the
  same time from pass to pass (142.47 and 142.13 ms under the profiler in load 1, 140.09 and 139.55 in load 2). Request 2
  is the comparison; what the pause is, is Q1416.
- **The reply's text changes with the stack, as it must when the verify pass's sums change.** All eight `off` replies
  carry one text (checksum `1c45f5e1b111`), all eight `on` replies another (`2d32b6b8e41e`, the text of R79's `on`). The
  drafts accepted: 173 of 243 and 172 of 246 `off` (83 and 84 steps), 171 of 250 in both `on` requests (85 steps).
- These are single-request diagnostics. The kept switches' agentic speed is measured in the stacked agent check (R72).

### perf

[perf/perf.md](perf/perf.md) (from [run/perf_table.py](run/perf_table.py)): every case's µs per product, each pass by
value, and its class against the reference arm. A shape's spread is the largest (max − min) / median over every arm and
pass at the token counts where every arm runs the same kernel (1 token, and the cases no switch takes); a case gains or
loses when it is faster or slower than the reference by more than that spread in both passes. Each case is tagged with
the switch that changes its kernel, from the merged tree's kernel choices
([pick_flashnext_all.tsv](../2026-11-11_v4_kernel_merge/pick_flashnext_all.tsv), and for Q5_K C729's
[table of warps](../2026-11-11_c729_readonce_5to8/README.md)).

`kern` against `mt` (µs per product, pass 1 / pass 2):

| Switch | Product | Tokens | `mt` | `kern` | Change | Class |
|---|---|---:|---:|---:|---:|---|
| `_Q6K_MT` | Q6_K 2,560 × 6,144 | 2 | 172.09 / 171.33 | 104.50 / 104.12 | −39.3 % / −39.2 % | gain |
| | | 3 | 220.99 / 220.29 | 138.97 / 138.79 | −37.1 % / −37.0 % | gain |
| | | 4 | 263.82 / 262.72 | 226.89 / 226.23 | −14.0 % / −13.9 % | gain |
| `_DMT_KT` | IQ4_XS 2,560 × 6,144 | 4 | 112.60 / 113.33 | 132.96 / 133.04 | +18.1 % / +17.4 % | loss |
| | Q4_K 2,560 × 6,144 | 3 | 98.07 / 99.10 | 110.40 / 110.94 | +12.6 % / +11.9 % | loss |
| | | 4 | 142.29 / 141.75 | 136.07 / 135.90 | −4.4 % / −4.1 % | gain |
| `_RO8` | IQ4_XS 2,560 × 6,144 (3 warps a block) | 5 | 258.29 / 257.95 | 169.97 / 169.85 | −34.2 % / −34.2 % | gain |
| | IQ4_XS 2,560 × 6,144 (1 warp a block) | 6 | 277.57 / 279.01 | 595.98 / 595.90 | +114.7 % / +113.6 % | loss |
| | Q4_K 2,560 × 6,144 (2 warps a block) | 5 | 268.07 / 268.42 | 172.45 / 172.86 | −35.7 % / −35.6 % | gain |
| | Q5_K 2,560 × 6,144 (1 warp a block) | 5 | 309.21 / 308.22 | 520.89 / 519.46 | +68.5 % / +68.5 % | loss |

- **`_RO8` gains in 30 of its 32 cases** (every 2,560-wide product at 5-8 tokens, including the output layer's
  Q5_K 248,320 × 2,560: 8,996.59 / 9,009.71 → 4,399.10 / 4,396.49 µs at 5 tokens); its two losses are the launches of
  one warp a block (C809). It takes nothing in the preset's verify passes; it is for longer drafts (C797), and R60 found
  it negligible on prompts.
- **`_DMT_KT_M=2`** (two row blocks a tile round) against `_DMT_KT` alone: 132.96 / 133.04 → 128.48 / 128.06 (IQ4_XS,
  4 tokens), 110.40 / 110.94 → 106.91 / 106.79 (Q4_K, 3), 136.07 / 135.90 → 129.45 / 129.39 (Q4_K, 4): a gain in all
  three, but against `mt` IQ4_XS at 4 tokens and Q4_K at 3 still lose (+14.1 % / +13.0 % and +9.0 % / +7.8 %) and
  Q4_K at 4 gains −9.0 % / −8.7 %. `_DMT_KT_M=2` also moves two cases the kernel choices give no tile: Q6_K at 4 tokens
  226.89 / 226.23 → 234.83 / 234.85 and Q4_K 2,560 × 6,144 at 5 tokens 172.45 / 172.86 → 177.19 / 176.69 (C808 checks
  why).
- **The controls hold:** of the 39 cases `kern` and `mt` run on the same kernel, 35 are negligible; the other four (one
  IQ4_XS and three Q5_K cases) move by at most 2.4 % in a pass, three of them in opposite directions in the two passes
  and the fourth (Q5_K 512 × 2,560 at 2 tokens, +0.8 % / +0.5 %) just past that shape's 0.3 % spread. `mt` against `off` repeats R79
  and R75: `_DMT` gains in 21 of 21 cases and `_Q5K_MT` in 9 of 9.

### What goes into the v4 candidate line

Exact on a P106 (R120a, R112) and, per switch in the served pass: `_Q3K_MT`, `_GLU_MT` and `_Q6K_MT` gain, so they go
into the candidate line (`v4c2`). `GGML_CUDA_MMVF_F32_MT` and `_DMT_KT` lose there and stay out, behind their
switches, until a width rule fixes each (C807, C808). `_RO8` has nothing to take in a verify pass of at most 4 tokens
and is negligible on prompts (R60), so it waits for the longer drafts (C797).

## What ran

- **The host:** MemAvailable at least 3,178,952 kB before every `perf` run and 3,202,524 kB before every load; load
  average at most 1.24.
- **The loads** took 12 min 18 s, 11 min 53 s, 11 min 53 s and 12 min 7 s.
- The controlling laptop restarted at about 15:56, during load 2's profiled requests (15:55:52-15:56:31). The run was
  detached on the rig and ran on; load 2's profiled passes match load 3's (85.970 and 86.001 against 85.958 and
  85.991 ms).
- **The tables ran on a build machine's CPU** at the lowest priority, never on the rig.

## Files

| File | What |
|---|---|
| [perf/us_run.tsv](perf/us_run.tsv) | every case's µs per product: arm, pass, type, rows, K, tokens (no header) |
| [perf/wide.tsv](perf/wide.tsv) | the same per shape and token count, one column per arm and pass |
| [perf/verdict.tsv](perf/verdict.tsv) | every comparison (`kern`, `kernm2` against `mt`; `kernm2` against `kern`; `mt` against `off`) at 2-8 tokens: the switch, each pass, the change, the spread, the class |
| [perf/spread.tsv](perf/spread.tsv) | each shape's same-kernel token counts and spread |
| [perf/perf.md](perf/perf.md) | `perf_table.py`'s report |
| [served/stack.tsv](served/stack.tsv), [served/stack.md](served/stack.md) | the verify pass by switch per load and request, the change and the class |
| [served/products.tsv](served/products.tsv) | per load and request: ms a pass of the one-op weight products by type, the attention and recurrent projections, the pass |
| [served/by_name.tsv](served/by_name.tsv) | per load and request: each IQ4_XS, Q4_K, Q6_K, Q3_K and F32 one-op product by name, launch groups and ms a pass |
| [served/parts.tsv](served/parts.tsv) | per load and request: the pass by R41's parts, all cards |
| [served/replies.tsv](served/replies.tsv) | every reply: text checksum, drafts, accepted, steps, ms a step, tok/s |
| [served/served.md](served/served.md) | R79's `served_table.py` report with the six switches' types |
| [served/loads/](served/loads/) | per load: `verify_parts.py`'s report, its cut per request, every main pass (`passes.tsv`), per card and per launch-group label |
| [run/perf_table.py](run/perf_table.py) | the `perf` tables and classes |
| [run/served_stack.py](run/served_stack.py) | the verify pass by switch |

## Commands

`RIG` is the rig's ssh alias, `W` a folder on a build machine, `R` the run folder on the rig, `KERN` the `-kern` tarball
unpacked on the rig, `S` a folder on the rig with this repository's files (the preset, `rig_server.sh`, the greedy
request), `MODEL` the Swift Q2_0 first shard and `DRAFT` the MTP draft.

```bash
# copy rig_kern.sh and verify_prof.sh to the rig; then on the rig, idle:
STEPS="perf served" BIN=$KERN/bin OUT=$R/out MODEL=$MODEL DRAFT=$DRAFT PRESET=$S/release/flash-next-v3/settings.env \
  RIG_SERVER=$S/bench/agent_check/rig_server.sh REQ=$S/docs/data/2026-11-04_draft_one_card/run/greedy_request.json \
  VERIFY_PROF=$R/verify_prof.sh ORDER="off on on off" \
  SERVED_ENV="GGML_CUDA_MMVQ_PASCAL_Q6K_MT=1 GGML_CUDA_MMVQ_PASCAL_DMT_KT=1 GGML_CUDA_MMVQ_PASCAL_RO8=1 GGML_CUDA_MMVQ_PASCAL_Q3K_MT=1 GGML_CUDA_MMVQ_PASCAL_GLU_MT=1 GGML_CUDA_MMVF_F32_MT=1" \
  setsid nohup bash $R/rig_kern.sh > $R/console.out 2>&1 < /dev/null &
# the tables, on the build machine (the rig's out/ copied to $W/out, each load's prof/prof-verify.csv to $W/traces/<k>-<arm>/prof/),
# from this repository's root
D=docs/data/2026-11-12_r120b_kern_speed/run
nice -n 19 python3 -I $D/perf_table.py $W/out/perf/us_run.tsv --out $W/tables/perf > $W/tables/perf/perf.md
for d in 1-off 2-on 3-on 4-off; do
  mkdir -p $W/tables/served/$d
  nice -n 19 python3 -I docs/data/2026-11-10_served_verify_parts/run/verify_parts.py \
    --kpass-parts docs/data/2026-11-04_kpass_profile/run/kpass_parts.py --prof-summary scripts/prof_summary.py \
    --event-us 7.6 --out $W/tables/served/$d $W/traces/$d/prof/prof-verify.csv > $W/tables/served/$d/report.md
done
nice -n 19 python3 -I docs/data/2026-11-11_r79_c722_p106/run/served_table.py $W/out/served --parts $W/tables/served \
  --out $W/tables/served --types iq4_xs,q4_K,q6_K,q3_K,f32 > $W/tables/served/served.md
nice -n 19 python3 -I $D/served_stack.py $W/tables/served --out $W/tables/served > $W/tables/served/stack.md
```

## Provenance and licence

The model is UkisAI's Swift-1.5 GSQ-RCO Q2_0 of Qwen3.8-Flash-Next (its maker's licence) with its MTP draft layer. The
request is S37's greedy request ([greedy_request.json](../2026-11-04_draft_one_card/run/greedy_request.json)). The
engine is Miner Llama (MIT, as llama.cpp). The tables are this project's measurements.
