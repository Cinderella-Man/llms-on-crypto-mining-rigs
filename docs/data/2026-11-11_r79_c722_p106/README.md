# R79 (Q1277): C722's read-once IQ4_XS and Q4_K kernel on a P106 (2026-11-11)

What this measured: C722's kernel for the 2–4-token IQ4_XS and Q4_K products of Qwen3.8-Flash-Next's verify pass
([2026-11-11_c722_short_batch_products](../2026-11-11_c722_short_batch_products/README.md), switch
`GGML_CUDA_MMVQ_PASCAL_DMT`, [docs/switches.md](../../switches.md)), natively on one P106-090. Three questions, in this
order: are each token's sums the one-token kernel's, bit for bit; how long does each Flash-Next shape take at 1 to 5
tokens beside the k-column kernel the preset uses today; and what does the switch do to a served verify pass and to the
step. The registry row is Q1277 in [quants/experiments.md](../../../quants/experiments.md).

The build is Miner Llama's v4 night merge, `pascal-p106-v4-b12246-night` at `c408dabcb`
([2026-11-11_v4_night_merge](../2026-11-11_v4_night_merge/README.md)), which holds C722's branch. Only this switch was
set; every other switch the merge adds was off, so `off` computes what `pascal-p106-v4` computes. The kernel's native
replay is not repeated here: the merge's replay on a P106 ran the set's twelve `dmt-*` runs, and each passed 315 of
315 cases ([2026-11-11_v4night_replay](../2026-11-11_v4night_replay/README.md)).

## Result

**Exact, and a gain at every shape.** At its fastest setting (R = 1, one row per lane group) the kernel gives every
2–4-token product the one-token kernel's sums bit for bit, and each of the 21 Flash-Next cases at 2–4 tokens is faster
than on the k-column kernel by more than its spread in both passes (the four-token IQ4_XS `attn_qkv`: 146.31 and
147.45 µs against 295.15 and 296.29). In a served verify pass the IQ4_XS and Q4_K products take 8.291, 8.305, 8.314
and 8.313 ms of GPU time with the switch against 16.212, 16.221, 16.202 and 16.214 without; the pass takes 93.068,
93.104, 93.105 and 93.123 ms against 101.030, 101.033, 101.006 and 101.033; and the server's step on the second
request 114.01 and 113.62 ms against 122.10 and 122.03.

The default rule picks R = 2 for most shapes, and R = 2 is slower than R = 1 at every shape where it was tried. So on
this build the switch has to be set with `GGML_CUDA_MMVQ_PASCAL_DMT_R=1`. Since `pascal-p106-v4-b12246-c758`, R = 1
is the default, without the prefetch at IQ4_XS's 2 tokens at K = 2,560
([2026-11-11_c758_dmt_default](../2026-11-11_c758_dmt_default/README.md), Q1335).

### Bit identity

[bitid.tsv](bitid.tsv): `multitoken_bitid dmt` (from [bench/mmvq_pascal](../../../bench/mmvq_pascal/), linked against the
merged tree) compares every 2–4-token product with its tokens' one-token products, bit for bit, on card 0, with the
preset's k-column switches (`GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=32768 GGML_CUDA_MMVQ_PASCAL_KCOL_TABLE=0`).

| Setting | Switches beside the preset's | Comparisons | With differences |
|---|---|---:|---:|
| `on` | `DMT=1` | 60 | 0 |
| `nopf` | `DMT=2` (no register prefetch) | 60 | 0 |
| `r1` | `DMT=1 DMT_R=1` | 60 | 0 |
| `r2` | `DMT=1 DMT_R=2` | 60 | 0 |
| control | none (the k-column kernel) | 60 | 50 |

- **The 60 comparisons** are both types at 2, 3 and 4 tokens over ten shapes: Flash-Next's four (10,240, 6,144 and
  12,288 rows at K = 2,560; 2,560 rows at K = 6,144), and six that test the edges (4,097 × 2,560, 200 × 512,
  1,000 × 1,024, 129 × 2,048, 77 × 4,096, 64 × 8,192).
- **The control differs in 50**, every Flash-Next shape among them: up to 2.59e-3 relative (IQ4_XS, 6,144 × 2,560,
  3 tokens). It is identical in the other 10 (77 × 4,096 at 3 and 4 tokens, 64 × 8,192 at 2 to 4, both types), so
  those 10 do not tell the two kernels apart.

### Kernel time at 1 to 5 tokens

[perf/](perf/): `test-backend-ops perf` on card 0 at the seven Flash-Next shapes of the two types (Q4_K has no
12,288-row product), 1 to 5 tokens, seven arms alternated over two passes. `off` is the preset (the k-column kernel at
2 to 8 tokens). At 1 and 5 tokens every arm runs the same kernels (the one-token kernel, the k-column kernel), so their
spread across the 14 runs is the noise of a case (the Spread column; [spread.tsv](perf/spread.tsv)). A case is a
gain when it beats `off` by more than its shape's spread in both passes.

**`r1` (`DMT=1 DMT_R=1`), the arm the served step ran**, µs per product, pass 1 / pass 2
([verdict.tsv](perf/verdict.tsv)):

| Type | Rows × K | Tokens | `off` | `r1` | Change | Spread | Class | 1 token | `r1` p1 / 1 token |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| IQ4_XS | 2,560 × 6,144 | 2 | 160.73 / 161.45 | 80.35 / 80.70 | −50.0 % / −50.0 % | 1.4 % | gain | 75.83 | 1.06 |
| IQ4_XS | 2,560 × 6,144 | 3 | 198.48 / 199.45 | 88.96 / 88.93 | −55.2 % / −55.4 % | 1.4 % | gain | 75.83 | 1.17 |
| IQ4_XS | 2,560 × 6,144 | 4 | 225.73 / 227.20 | 113.15 / 113.08 | −49.9 % / −50.2 % | 1.4 % | gain | 75.83 | 1.49 |
| IQ4_XS | 6,144 × 2,560 | 2 | 133.32 / 133.58 | 94.00 / 95.42 | −29.5 % / −28.6 % | 2.1 % | gain | 74.88 | 1.26 |
| IQ4_XS | 6,144 × 2,560 | 3 | 175.35 / 175.59 | 85.79 / 85.44 | −51.1 % / −51.3 % | 2.1 % | gain | 74.88 | 1.15 |
| IQ4_XS | 6,144 × 2,560 | 4 | 185.12 / 185.31 | 92.70 / 93.51 | −49.9 % / −49.5 % | 2.1 % | gain | 74.88 | 1.24 |
| IQ4_XS | 10,240 × 2,560 | 2 | 214.01 / 214.85 | 153.56 / 151.83 | −28.2 % / −29.3 % | 2.8 % | gain | 119.25 | 1.29 |
| IQ4_XS | 10,240 × 2,560 | 3 | 271.99 / 273.40 | 131.94 / 133.50 | −51.5 % / −51.2 % | 2.8 % | gain | 119.25 | 1.11 |
| IQ4_XS | 10,240 × 2,560 | 4 | 295.15 / 296.29 | 146.31 / 147.45 | −50.4 % / −50.2 % | 2.8 % | gain | 119.25 | 1.23 |
| IQ4_XS | 12,288 × 2,560 | 2 | 256.12 / 257.10 | 180.54 / 180.89 | −29.5 % / −29.6 % | 2.5 % | gain | 141.49 | 1.28 |
| IQ4_XS | 12,288 × 2,560 | 3 | 323.36 / 324.38 | 155.31 / 156.76 | −52.0 % / −51.7 % | 2.5 % | gain | 141.49 | 1.10 |
| IQ4_XS | 12,288 × 2,560 | 4 | 343.88 / 345.58 | 174.39 / 173.97 | −49.3 % / −49.7 % | 2.5 % | gain | 141.49 | 1.23 |
| Q4_K | 2,560 × 6,144 | 2 | 158.49 / 159.32 | 73.53 / 74.16 | −53.6 % / −53.5 % | 1.2 % | gain | 66.00 | 1.11 |
| Q4_K | 2,560 × 6,144 | 3 | 200.26 / 201.24 | 98.79 / 99.31 | −50.7 % / −50.7 % | 1.2 % | gain | 66.00 | 1.50 |
| Q4_K | 2,560 × 6,144 | 4 | 231.97 / 233.66 | 141.15 / 142.40 | −39.2 % / −39.1 % | 1.2 % | gain | 66.00 | 2.14 |
| Q4_K | 6,144 × 2,560 | 2 | 169.67 / 166.77 | 64.04 / 64.13 | −62.3 % / −61.5 % | 2.0 % | gain | 67.43 | 0.95 |
| Q4_K | 6,144 × 2,560 | 3 | 171.93 / 172.28 | 64.51 / 64.29 | −62.5 % / −62.7 % | 2.0 % | gain | 67.43 | 0.96 |
| Q4_K | 6,144 × 2,560 | 4 | 181.50 / 183.33 | 76.70 / 77.40 | −57.7 % / −57.8 % | 2.0 % | gain | 67.43 | 1.14 |
| Q4_K | 10,240 × 2,560 | 2 | 260.29 / 260.82 | 97.83 / 97.92 | −62.4 % / −62.5 % | 2.6 % | gain | 105.65 | 0.93 |
| Q4_K | 10,240 × 2,560 | 3 | 271.74 / 271.83 | 103.63 / 103.00 | −61.9 % / −62.1 % | 2.6 % | gain | 105.65 | 0.98 |
| Q4_K | 10,240 × 2,560 | 4 | 286.45 / 287.34 | 120.82 / 122.01 | −57.8 % / −57.5 % | 2.6 % | gain | 105.65 | 1.14 |

`1 token` is the median over every arm and pass of the 1-token case (the one-token kernel in all of them).

- **All 21 cases are gains at R = 1.** At K = 2,560 a four-token product costs 1.24, 1.23 and 1.23 times one token's
  for IQ4_XS (6,144, 10,240 and 12,288 rows) and 1.14 and 1.14 for Q4_K (6,144 and 10,240 rows), pass 1 (C722 estimated 246 and 268 load-store clocks a warp iteration against the k-column kernel's 422 and 428: est. 0.58
  and 0.63 of its time; measured at 4 tokens, pass 1: 0.50, 0.50, 0.51 for IQ4_XS and 0.42, 0.42 for Q4_K).
- **K = 6,144 lags at 4 tokens:** 1.49 times one token's for IQ4_XS and 2.14 for Q4_K (0.61 of `off`, near the
  estimate). Four tokens' activations staged whole leave a block 3 warps there; that is C728's case (Q1287).
- **Two Q4_K shapes at 2 and 3 tokens beat the one-token kernel** (6,144 rows: 0.95 and 0.96 of its time; 10,240
  rows: 0.93 and 0.98): the one-token kernel
  does not reach its own bound at those shapes.

**The other arms** ([perf.md](perf/perf.md) has every arm's table, [wide.tsv](perf/wide.tsv) every value):

| Arm | Switches | Gain | Negligible | Loss | Its 4-token cases summed, µs, pass 1 / pass 2 |
|---|---|---:|---:|---:|---:|
| `off` | none | | | | 1,749.80 / 1,758.71 |
| `on` | `DMT=1` (R by the default rule) | 20 | 1 | 0 | 1,317.47 / 1,325.14 |
| `nopf` | `DMT=2` | 17 | 2 | 2 | 1,456.94 / 1,481.15 |
| `r1` | `DMT=1 DMT_R=1` | 21 | 0 | 0 | 865.22 / 869.82 |
| `r2` | `DMT=1 DMT_R=2` | 19 | 0 | 2 | 1,534.53 / 1,530.98 |
| `r1nopf` | `DMT=2 DMT_R=1` | 21 | 0 | 0 | 982.21 / 985.72 |
| `nw2` | `DMT=1 DMT_NW=2` | 18 | 0 | 3 | 1,537.38 / 1,542.38 |

- **The default rule's R = 2 costs the gain.** The rule takes R = 2 wherever a block holds as many rows as at R = 1:
  both types at K = 2,560 and IQ4_XS at K = 6,144. There `on` is slower than `r1` in every case. Its one negligible
  case is IQ4_XS 2,560 × 6,144 at 4 tokens: 228.20 / 230.01 µs against `off`'s 225.73 / 227.20 (+1.1 % / +1.2 %,
  spread 1.4 %), where `r1` takes 113.15 / 113.08. At Q4_K 2,560 × 6,144 the rule takes R = 1, so `on` and `r1` run
  the same kernel there (139.98 / 141.36 and 141.15 / 142.40 at 4 tokens).
- **Forcing R = 2 loses at K = 6,144:** Q4_K 2,560 × 6,144 at 4 tokens takes 341.93 / 341.98 µs (+47.4 % / +46.4 %),
  one warp a block; `nw2` (2 warps a block) the same, 342.30 / 342.46.
- **Without the prefetch at R = 1 (`r1nopf`), 2 tokens of IQ4_XS at K = 2,560 run faster still:** 75.74 / 75.90
  against `r1`'s 94.00 / 95.42 (6,144 rows), 117.74 / 118.73 against 153.56 / 151.83 (10,240), 135.00 / 136.04
  against 180.54 / 180.89 (12,288); at 3 and 4 tokens `r1` is faster. The fastest arm per shape is in
  [fastest.tsv](perf/fastest.tsv): `r1` in 14 of the 21 cases, `r1nopf` in 5 (all at 2 tokens), `on` in 2 (where it
  is R = 1 too).

### The served verify pass

[served/](served/): R49's profile of a served verify pass
([2026-11-10_served_verify_parts](../2026-11-10_served_verify_parts/README.md), `run/verify_prof.sh`): the v3 preset
exactly (UkisAI's Swift-1.5 GSQ-RCO Q2_0 of Qwen3.8-Flash-Next on nine P106-090 cards, the MTP draft at three
tokens), four loads in the order `off`, `on`, `on`, `off`, each a `warm` load (no profile) and a `prof` load (the
fork's profiler in group mode), each answering the greedy request twice. `on` added `GGML_CUDA_MMVQ_PASCAL_DMT=1
GGML_CUDA_MMVQ_PASCAL_DMT_R=1` to the preset's switches: the `perf` step's fastest arm at 4 tokens.

**GPU time per verify pass, all nine cards, ms** ([products.tsv](served/products.tsv), [parts.tsv](served/parts.tsv);
`verify_parts.py`'s cut, the profiler's 7.6 µs subtracted per launch group; requests 1 and 2 of each profiled load):

| Load | IQ4_XS + Q4_K products | The pass (sum of the parts) | Median raw GPU time of a typical pass |
|---|---:|---:|---:|
| 1 `off` | 16.212, 16.221 | 101.030, 101.033 | 118.89, 118.85 |
| 2 `on` | 8.291, 8.305 | 93.068, 93.104 | 110.92, 110.99 |
| 3 `on` | 8.314, 8.313 | 93.105, 93.123 | 110.91, 110.92 |
| 4 `off` | 16.202, 16.214 | 101.006, 101.033 | 118.91, 118.92 |

- **The switch takes 7.91 ms of the products and 7.93 ms of the pass** (the means of the four `on` and the four `off`
  values). The four `off` passes lie within 0.027 ms of each other and the four `on` within 0.055: a gain. C722
  estimated up to 8.6 ms of R41's plain four-token pass.
- **By product** ([by_name.tsv](served/by_name.tsv), load 1 against load 2, request 1): `ssm_out` IQ4_XS 3.875 → 1.905,
  `attn_qkv` IQ4_XS 3.820 → 1.832, `ssm_out` Q4_K 2.535 → 1.543, `attn_q` IQ4_XS 1.378 → 0.649, `attn_qkv` Q4_K
  1.154 → 0.441, `attn_output` Q4_K 0.930 → 0.567, `attn_gate` IQ4_XS 0.771 → 0.376, `attn_gate` Q4_K 0.563 → 0.226;
  and the small products the switch also takes: the shared expert's gate and up where they are IQ4_XS or Q4_K
  (`ffn_gate_shexp` IQ4_XS 0.230 → 0.155, `ffn_up_shexp` IQ4_XS 0.191 → 0.129, `ffn_gate_shexp` Q4_K 0.203 → 0.126,
  `ffn_up_shexp` Q4_K 0.198 → 0.121) and `attn_k` and `attn_v` (IQ4_XS 0.125 → 0.072 and 0.108 → 0.071, Q4_K
  0.066 → 0.040 and 0.065 → 0.038).
- **No other part moves beyond its spread between the loads of one arm** ([served.md](served/served.md), "The pass's
  parts"): the shared expert's part falls by its IQ4_XS and Q4_K products (4.902, 4.908, 4.887, 4.892 ms `off`;
  4.633, 4.628, 4.664, 4.656 `on`); the routed experts, the hyper-connections, the Gated DeltaNet state, attention and
  the output layer do not change.

**The step, from the warm loads** (the server's own timings, [replies.tsv](served/replies.tsv); ms a step = the
reply's generation time over its steps, steps = generated tokens less accepted drafts):

| Load | Request 1: ms a step | tok/s | Request 2: ms a step | tok/s |
|---|---:|---:|---:|---:|
| 1 `off` | 131.90 | 22.48 | 122.10 | 24.86 |
| 2 `on` | 122.43 | 24.22 | 114.01 | 26.01 |
| 3 `on` | 121.58 | 24.39 | 113.62 | 26.10 |
| 4 `off` | 131.27 | 22.59 | 122.03 | 24.88 |

- **The step is shorter by about the pass's GPU time:** request 2 (a context checkpoint of 75 prompt tokens, 4 read)
  takes 114.01 and 113.62 ms a step with the switch against 122.10 and 122.03 without; request 1 (79 prompt tokens,
  the load's one-off launches) 122.43 and 121.58 against 131.90 and 131.27.
- **The reply's text changes with the switch, as it must when the verify pass's sums change.** All eight `off`
  replies carry one text (checksum `e460d038a1f6`, the same as R49's on Miner Llama v3.0); all eight `on` replies
  carry another (`2d32b6b8e41e`). They part at the 53rd word of the reasoning, where `on` adds one short sentence;
  both stay on the task. `off`'s 2–4-token sums are the k-column kernel's, which differ from one-token decoding's
  (the control above); `on` gives these products the one-token sums, so a near-tie in the logits can resolve the other
  way. The drafts accepted: 170 of 255 and 172 of 247 (`off`, requests 1 and 2), 170 of 253 in both `on` requests.
- These are single-request diagnostics. The switch's agentic speed is measured with the other gains of the night
  merge (R90).

## What ran

One detached script on one rig ([rig_c722.sh](../2026-11-11_c722_short_batch_products/run/rig_c722.sh), steps
`bitid perf served`), nothing else on the rig; 10:07:18 to 11:05:14 on 2026-11-11:

| Step | Time | What |
|---|---|---|
| `bitid` | 10:07:18–10:07:40 | `multitoken_bitid dmt` at the four settings and the control, card 0 |
| `perf` | 10:07:40–10:17:14 | `test-backend-ops perf -o MUL_MAT` at the two types' Flash-Next shapes, 1–5 tokens, `GGML_TEST_RAW_QUANT=1`, 7 arms × 2 passes, card 0 |
| `served` | 10:17:14–11:05:14 | four loads of `verify_prof.sh` (`warm`, `prof`) |

- **The harness:** `bench/mmvq_pascal/verify/build_multitoken_bitid.sh` with the merged tree's source and its rig build
  (`build/` of the tarball's build), on a build machine's CPU in a container without a GPU. It loads the rig's own
  CUDA 12.9 runtime, as the build's binaries do.
- **The host:** MemAvailable at least 3,150,348 kB before each step and each `perf` run; load average under 1.2.
- **The loads** took 12 min 0 s, 11 min 57 s, 11 min 56 s and 12 min 7 s.
- **The tables ran on a build machine's CPU**, never on the rig: `run/perf_table.py`, R49's `verify_parts.py` on each
  profiled load, `run/served_table.py`.

## Files

| File | What |
|---|---|
| [bitid.tsv](bitid.tsv) | every comparison of the four settings and the control: type, tokens, rows, K, values, values that differ, largest relative difference |
| [perf/us_run.tsv](perf/us_run.tsv) | every case's µs per product: arm, pass, type, rows, K, tokens (no header) |
| [perf/wide.tsv](perf/wide.tsv) | the same per shape and token count, one column per arm and pass |
| [perf/verdict.tsv](perf/verdict.tsv) | every arm against `off` at 2–4 tokens: each pass, the change, the spread, the class, against one token |
| [perf/spread.tsv](perf/spread.tsv) | each shape's spread at 1 and 5 tokens |
| [perf/fastest.tsv](perf/fastest.tsv) | the fastest arm per shape and token count, beside `r1` |
| [perf/arms4.tsv](perf/arms4.tsv) | each arm's 4-token cases summed per pass (the served step's choice of `on`) |
| [perf/perf.md](perf/perf.md) | `perf_table.py`'s report: every arm's table, the classes, the fastest arm, the sums |
| [served/products.tsv](served/products.tsv) | per load and request: GPU ms a verify pass of the one-op weight products by type, IQ4_XS + Q4_K, the attention and recurrent projections, the pass |
| [served/by_name.tsv](served/by_name.tsv) | per load and request: each IQ4_XS and Q4_K product by name, launch groups and ms a pass |
| [served/parts.tsv](served/parts.tsv) | per load and request: the pass by R41's parts, all cards |
| [served/replies.tsv](served/replies.tsv) | every reply: text checksum, drafts, accepted, steps, ms a step, tok/s |
| [served/served.md](served/served.md) | `served_table.py`'s report |
| [served/loads/](served/loads/) | per load: `verify_parts.py`'s report, its cut per request, per card and per launch-group label |
| [run/perf_table.py](run/perf_table.py) | the `perf` tables and classes |
| [run/served_table.py](run/served_table.py) | the replies and the products per verify pass |

## Commands

`RIG` is the rig's ssh alias, `BUILD` the build machine's, `W` a folder there, `R` the run folder on the rig, `NIGHT`
the night merge's unpacked tarball on the rig, `S` a folder on the rig with this repository's files (the preset,
`rig_server.sh`, the greedy request).

```bash
# on the build machine, CPU only: the harness against the merged tree ($W/src, built by scripts/build_llamacpp_docker.sh)
DOCKER_ARGS="--cpu-shares=256" nice -n 19 bash bench/mmvq_pascal/verify/build_multitoken_bitid.sh $W/src $W/src
# copy bench/mmvq_pascal/build/multitoken_bitid, rig_c722.sh and verify_prof.sh to the rig; then on the rig, idle:
STEPS="bitid perf served" BIN=$NIGHT/bin OUT=$R/out BITID=$R/multitoken_bitid \
  MODEL=<Swift Q2_0 first shard> DRAFT=<the MTP draft> PRESET=$S/release/flash-next-v3/settings.env \
  RIG_SERVER=$S/bench/agent_check/rig_server.sh REQ=$S/docs/data/2026-11-04_draft_one_card/run/greedy_request.json \
  VERIFY_PROF=$R/verify_prof.sh setsid nohup bash $R/rig_c722.sh > $R/console.out 2>&1 < /dev/null &
# the tables, on the build machine (the rig's out/ copied to $W/out), from this repository's root
D=docs/data/2026-11-11_r79_c722_p106/run
nice -n 19 python3 -I $D/perf_table.py $W/out/perf/us_run.tsv --out $W/tables/perf > $W/tables/perf/perf.md
for d in 1-off 2-on 3-on 4-off; do
  nice -n 19 python3 -I docs/data/2026-11-10_served_verify_parts/run/verify_parts.py \
    --kpass-parts docs/data/2026-11-04_kpass_profile/run/kpass_parts.py --prof-summary scripts/prof_summary.py \
    --event-us 7.6 --out $W/tables/served/$d $W/out/served/$d/prof/prof-verify.csv > $W/tables/served/$d/report.md
done
nice -n 19 python3 -I $D/served_table.py $W/out/served --parts $W/tables/served --out $W/tables/served > $W/tables/served/served.md
```

## Provenance and licence

The model is UkisAI's Swift-1.5 GSQ-RCO Q2_0 of Qwen3.8-Flash-Next (its maker's licence) with its MTP draft layer. The
request is S37's greedy request
([greedy_request.json](../2026-11-04_draft_one_card/run/greedy_request.json)). The engine is Miner Llama (MIT, as
llama.cpp). The tables are this project's measurements.
