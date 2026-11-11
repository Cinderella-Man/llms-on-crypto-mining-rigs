# R75 (Q1222): C690's four-row output-head kernel on a P106 (2026-11-11)

What this measured: C690's kernel for Q5_K's products of 2 to 4 rows
([2026-11-11_c690_head_four_rows](../2026-11-11_c690_head_four_rows/README.md), switch
`GGML_CUDA_MMVQ_PASCAL_Q5K_MT`, [docs/switches.md](../../switches.md)), natively on one P106-090. Its target is the
output head of Qwen3.8-Flash-Next (`output`, Q5_K, 2,560 × 248,320), whose four-row product in a served verify pass
took 8.05 and 8.06 ms against 2.86 ms for one row ([R49](../2026-11-10_served_verify_parts/README.md),
[R41](../2026-11-10_release_profile/README.md)). Three questions, in this order: are each row's sums the one-token
kernel's, bit for bit; how long does the head's product take at 1 to 4 rows beside the k-column kernel the preset uses
today; and what does the switch do to a served verify pass and to the step. The registry row is Q1222 in
[quants/experiments.md](../../../quants/experiments.md).

The build is Miner Llama's v4 night merge, `pascal-p106-v4-b12246-night` at `c408dabcb`
([2026-11-11_v4_night_merge](../2026-11-11_v4_night_merge/README.md)), which holds C690's branch. Only this switch was
set; every other switch the merge adds was off, so `off` computes what `pascal-p106-v4` computes. The kernel's native
replay is not repeated here: the merge's replay on a P106 ran the set's ten `q5kmt` runs, and each passed 112 of 112
cases ([2026-11-11_v4night_replay](../2026-11-11_v4night_replay/README.md)).

## Result

**Exact, and a gain at every row count; the head's four rows now cost 1.10 times one row.** At its fastest setting
(R = 1, one row per lane group, without the register prefetch: `GGML_CUDA_MMVQ_PASCAL_Q5K_MT=2
GGML_CUDA_MMVQ_PASCAL_Q5K_MT_R=1`) the kernel gives every 2–4-row product the one-token kernel's sums bit for bit,
and the head's product at four rows takes 3,340.78 and 3,346.73 µs against the k-column kernel's 8,103.16 and
8,100.11. In a served verify pass the head takes 3.149, 3.155, 3.149 and 3.151 ms of GPU time with the switch against
8.105, 8.125, 8.115 and 8.120 without; the pass takes 95.525, 95.558, 95.525 and 95.562 ms against 101.031, 101.045,
101.086 and 101.049; and the server's step on the second request 116.37 and 116.25 ms against 122.11 and 121.81.

The kernel's default (R = 2, two rows per lane group sharing the activation loads, with the prefetch) is slower than
R = 1 at every row count: 5,486.46 and 5,326.59 µs at four rows. So the switch has to be set with `_R=1` (and `=2`
for no prefetch) until the default changes (Q1347).

### Bit identity

[bitid.tsv](bitid.tsv): `multitoken_bitid q5k` (from [bench/mmvq_pascal](../../../bench/mmvq_pascal/), linked against
the merged tree) compares every 2–4-row Q5_K product with its rows' one-row products, bit for bit, on card 0.

| Setting | Switches | Comparisons | With differences |
|---|---|---:|---:|
| `on` | `Q5K_MT=1` | 21 | 0 |
| `nopf` | `Q5K_MT=2` (no register prefetch) | 21 | 0 |
| `r1` | `Q5K_MT=1 Q5K_MT_R=1` | 21 | 0 |
| control | the preset's k-column switches only (`GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=32768 GGML_CUDA_MMVQ_PASCAL_KCOL_TABLE=0`) | 21 | 14 |

- **The 21 comparisons** are 2, 3 and 4 rows over seven shapes: the head (248,320 × 2,560) and six that test the
  edges (4,097 × 2,560, 200 × 768, 1,000 × 1,024, 129 × 2,048, 77 × 4,096, 64 × 8,192).
- **The control differs in 14**, the head at every row count among them (8.10e-05, 5.34e-04 and 1.42e-04 relative at
  2, 3 and 4 rows; 879,561 of its 993,280 values differ at 4). It is identical in the other 7 (77 × 4,096 and
  64 × 8,192 at 2 to 4 rows, 129 × 2,048 at 4), so those 7 do not tell the two kernels apart.
- The `r1nopf` setting the served step ran (`Q5K_MT=2 Q5K_MT_R=1`) was not run by `multitoken_bitid`. It is `nopf`'s
  choice of prefetch at `r1`'s rows per lane group, both exact here, and the merge's replay ran it as
  `q5kmt-r1-nopf` (112 of 112 cases pass against the CPU).

### Kernel time at 1 to 4 rows

[perf/](perf/): `test-backend-ops perf` on card 0 at the head's shape (`type_a=q5_K`, m = 248,320, k = 2,560,
`GGML_TEST_RAW_QUANT=1`), 1 to 4 rows, six arms alternated over two passes. Every arm carries the preset's k-column
switches; `off` is the preset (the k-column kernel at 2 to 4 rows). At 1 row every arm runs the one-token kernel, so
the 12 runs of that case give the spread: (largest − smallest) / median = 0.9 %, median 3,023.54 µs (each run in
[wide.tsv](perf/wide.tsv)). A case is a gain when it beats `off` by more than the spread in both passes.

**`r1nopf` (`Q5K_MT=2 Q5K_MT_R=1`), the arm the served step ran**, µs per product, pass 1 / pass 2
([verdict.tsv](perf/verdict.tsv)):

| Rows | `off` | `r1nopf` | Change | Class | `r1nopf` p1 / 1 row |
|---:|---:|---:|---:|---|---:|
| 2 | 7,508.08 / 7,522.07 | 2,575.94 / 2,574.12 | −65.7 % / −65.8 % | gain | 0.85 |
| 3 | 7,681.59 / 7,691.59 | 2,782.25 / 2,792.19 | −63.8 % / −63.7 % | gain | 0.92 |
| 4 | 8,103.16 / 8,100.11 | 3,340.78 / 3,346.73 | −58.8 % / −58.7 % | gain | 1.10 |

`1 row` is the median of the 1-row case over every arm and pass, 3,023.54 µs.

- **Two and three rows cost less than one row on the one-token kernel** (0.85 and 0.92 of it). At one row the
  one-token kernel reads the head's 437 MB at about 145 GB/s; the new kernel's persistent grid of staged blocks reads
  it faster. That is the subject of Q1348.
- **Four rows cost 1.10 times one row.** C690 estimated 2.7 ms at its default and 2.9 ms for reading the weights; R = 1
  without the prefetch measures 3,340.78 and 3,346.73 µs here and 3.149, 3.155, 3.149 and 3.151 ms in the served
  pass below.

**Every arm** ([perf.md](perf/perf.md) has each arm's table, [wide.tsv](perf/wide.tsv) every value):

| Arm | Switches beside the preset's | 2 rows | 3 rows | 4 rows | Classes at 2 / 3 / 4 rows |
|---|---|---:|---:|---:|---|
| `off` | none | 7,508.08 / 7,522.07 | 7,681.59 / 7,691.59 | 8,103.16 / 8,100.11 | |
| `on` | `Q5K_MT=1` (R = 2, prefetch) | 4,619.20 / 4,552.52 | 5,001.40 / 4,936.07 | 5,486.46 / 5,326.59 | gain / gain / gain |
| `nopf` | `Q5K_MT=2` (R = 2) | 4,495.45 / 4,501.19 | 4,969.65 / 4,981.53 | 5,423.64 / 5,429.28 | gain / gain / gain |
| `r1` | `Q5K_MT=1 Q5K_MT_R=1` | 2,581.42 / 2,585.59 | 2,825.30 / 2,838.42 | 3,401.63 / 3,412.48 | gain / gain / gain |
| `r1nopf` | `Q5K_MT=2 Q5K_MT_R=1` | 2,575.94 / 2,574.12 | 2,782.25 / 2,792.19 | 3,340.78 / 3,346.73 | gain / gain / gain |
| `nw2` | `Q5K_MT=1 Q5K_MT_NW=2` (R = 2, 2 warps a block) | 4,576.82 / 4,535.74 | 7,347.28 / 7,346.84 | 8,207.62 / 8,208.56 | gain / gain / loss |

- **R = 2 costs most of the gain.** Sharing the activation loads between two rows was C690's estimate for the faster
  setting; on a P106 R = 1 is faster at every row count (four rows: 3,401.63 and 3,412.48 µs against `on`'s 5,486.46
  and 5,326.59), as R79 found for C722's kernel.
- **The prefetch does not pay at R = 1:** `r1nopf` against `r1` is −0.2 % and −0.4 % at 2 rows (within the spread),
  −1.5 % and −1.6 % at 3, −1.8 % and −1.9 % at 4.
- **`nw2` loses at four rows** (+1.3 % in both passes; at 3 rows −4.4 % and −4.5 %).

### The served verify pass

[served/](served/): R49's profile of a served verify pass
([2026-11-10_served_verify_parts](../2026-11-10_served_verify_parts/README.md), `run/verify_prof.sh`): the v3 preset
exactly (UkisAI's Swift-1.5 GSQ-RCO Q2_0 of Qwen3.8-Flash-Next on nine P106-090 cards, the MTP draft at three
tokens), four loads in the order `off`, `on`, `on`, `off`, each a `warm` load (no profile) and a `prof` load (the
fork's profiler in group mode), each answering the greedy request twice. `on` added
`GGML_CUDA_MMVQ_PASCAL_Q5K_MT=2 GGML_CUDA_MMVQ_PASCAL_Q5K_MT_R=1` to the preset's switches: the `perf` step's fastest
arm at 4 rows.

**GPU time per verify pass, all nine cards, ms** ([products.tsv](served/products.tsv), [parts.tsv](served/parts.tsv);
`verify_parts.py`'s cut, the profiler's 7.6 µs subtracted per launch group; requests 1 and 2 of each profiled load):

| Load | The head (`output`) | The other Q5_K products | The sampling chain after the head | The pass (sum of the parts) | Median raw GPU time of a typical pass |
|---|---:|---:|---:|---:|---:|
| 1 `off` | 8.105, 8.125 | 1.987, 1.992 | 0.649, 0.650 | 101.031, 101.045 | 118.90, 119.04 |
| 2 `on` | 3.149, 3.155 | 1.433, 1.432 | 0.649, 0.649 | 95.525, 95.558 | 113.32, 113.48 |
| 3 `on` | 3.149, 3.151 | 1.434, 1.435 | 0.650, 0.648 | 95.525, 95.562 | 113.37, 113.43 |
| 4 `off` | 8.115, 8.120 | 1.986, 1.989 | 0.651, 0.650 | 101.086, 101.049 | 118.99, 118.98 |

- **The switch takes 4.97 ms of the head and 5.51 ms of the pass** (the means of the four `on` and the four `off`
  values). The four `off` heads lie within 0.020 ms of each other and the four `on` within 0.006: a gain. Of R49's
  8.05 and 8.06 ms (this build's own `off` loads: 8.105, 8.125, 8.115, 8.120), 3.149, 3.155, 3.149 and 3.151 are
  left; one row costs 2.86 ms (R41), so the served four rows cost 1.10 times one row.
- **The switch also takes the other Q5_K products at four rows** ([by_name.tsv](served/by_name.tsv), load 1 against
  load 2, request 1): `attn_output` 0.964 → 0.709, `ssm_out` 0.713 → 0.528, `attn_v` 0.119 → 0.077, `ffn_up_shexp`
  0.081 → 0.052, `attn_k` 0.069 → 0.041, `ffn_gate_shexp` 0.041 → 0.027.
- **No other part moves** ([served.md](served/served.md), "The pass's parts"): the attention and recurrent
  projections fall by their Q5_K products (34.194, 34.129, 34.156, 34.187 ms `off`; 33.656, 33.687, 33.681, 33.584
  `on`); the routed experts, the shared expert, the hyper-connections, the Gated DeltaNet state, attention and the
  sampling chain do not change. The cards held the same memory in all four loads (card 8: 4,647 MiB once loaded).

**The step, from the warm loads** (the server's own timings, [replies.tsv](served/replies.tsv); ms a step = the
reply's generation time over its steps, steps = generated tokens less accepted drafts):

| Load | Request 1: ms a step | tok/s | Request 2: ms a step | tok/s |
|---|---:|---:|---:|---:|
| 1 `off` | 131.53 | 22.54 | 122.11 | 24.86 |
| 2 `on` | 123.57 | 24.28 | 116.37 | 25.78 |
| 3 `on` | 121.32 | 24.73 | 116.25 | 25.81 |
| 4 `off` | 127.60 | 23.24 | 121.81 | 24.92 |

- **The step is shorter by about the pass's GPU time:** request 2 (a context checkpoint of 75 prompt tokens, 4 read)
  takes 116.37 and 116.25 ms a step with the switch against 122.11 and 121.81 without; request 1 (79 prompt tokens,
  the load's one-off launches) 123.57 and 121.32 against 131.53 and 127.60.
- **The reply's text changes with the switch, as it must when the verify pass's sums change.** All eight `off`
  replies carry one text (checksum `e460d038a1f6`, R49's on Miner Llama v3.0); all eight `on` replies carry another
  (`2d32b6b8e41e`), the same text that C722's switch gave in [R79](../2026-11-11_r79_c722_p106/README.md). `off`'s
  2–4-row head sums are the k-column kernel's, which differ from one-row decoding's (the control above); `on` gives
  the head and the other Q5_K products the one-row sums, so a near-tie in the logits can resolve the other way. The
  drafts accepted: 170 of 255 and 172 of 247 (`off`, requests 1 and 2), 171 of 250 in both `on` requests (85 steps
  against 86 and 84).
- These are single-request diagnostics. The switch's agentic speed is measured with the other gains of the night
  merge (R90).

## What ran

One detached script on one rig ([rig_c690.sh](../2026-11-11_c690_head_four_rows/run/rig_c690.sh), steps
`bitid perf served`), nothing else on the rig; 11:51:35 to 12:43:26 on 2026-11-11:

| Step | Time | What |
|---|---|---|
| `bitid` | 11:51:35–11:51:54 | `multitoken_bitid q5k` at the three settings and the control, card 0 |
| `perf` | 11:51:54–11:55:26 | `test-backend-ops perf -o MUL_MAT` at the head's shape, 1–4 rows, 6 arms × 2 passes, card 0 |
| `served` | 11:55:26–12:43:26 | four loads of `verify_prof.sh` (`warm`, `prof`) |

- **The harness:** `bench/mmvq_pascal/verify/build_multitoken_bitid.sh` with the merged tree's source and its rig build
  (`build/` of the tarball's build), on a build machine's CPU in a container without a GPU (the binary R79 used; its
  `q5k` mode is this repository's). It loads the rig's own CUDA 12.9 runtime, as the build's binaries do.
- **The host:** MemAvailable at least 3,069,384 kB before each step and each `perf` run; load average under 1.2. The
  lowest MemAvailable during a load's requests was 334 MiB.
- **The loads** took 11 min 59 s, 12 min 5 s, 11 min 56 s and 12 min 0 s.
- **The tables ran on a build machine's CPU**, never on the rig: R79's `perf_table.py` on the `perf` step's rows,
  R49's `verify_parts.py` on each profiled load, `run/served_table.py`.

## Files

| File | What |
|---|---|
| [bitid.tsv](bitid.tsv) | every comparison of the three settings and the control: rows, m, k, values, values that differ, largest relative difference |
| [perf/us_run.tsv](perf/us_run.tsv) | every case's µs per product: arm, pass, rows (no header), as `rig_c690.sh` writes it |
| [perf/wide.tsv](perf/wide.tsv) | the same per row count, one column per arm and pass |
| [perf/verdict.tsv](perf/verdict.tsv) | every arm against `off` at 2–4 rows: each pass, the change, the spread, the class, against one row |
| [perf/spread.tsv](perf/spread.tsv) | the spread at 1 row (there is no 5-row case here) |
| [perf/fastest.tsv](perf/fastest.tsv) | the fastest arm per row count, beside `r1` |
| [perf/arms4.tsv](perf/arms4.tsv) | each arm's 4-row case per pass |
| [perf/perf.md](perf/perf.md) | `perf_table.py`'s report: every arm's table, the classes, the fastest arm |
| [served/products.tsv](served/products.tsv) | per load and request: GPU ms a verify pass of the one-op weight products by type, Q5_K, the head, the other Q5_K products, the head and the sampling chain by position, the pass |
| [served/by_name.tsv](served/by_name.tsv) | per load and request: each Q5_K product by name, launch groups and ms a pass |
| [served/parts.tsv](served/parts.tsv) | per load and request: the pass by R41's parts, all cards |
| [served/replies.tsv](served/replies.tsv) | every reply: text checksum, drafts, accepted, steps, ms a step, tok/s |
| [served/served.md](served/served.md) | `served_table.py`'s report |
| [served/loads/](served/loads/) | per load: `verify_parts.py`'s report, its cut per request, its buckets, per card and per launch-group label |
| [run/served_table.py](run/served_table.py) | the replies and the products per verify pass (R79's, with the switch's types and the head as options) |

## Commands

`RIG` is the rig's ssh alias, `W` a folder on the build machine, `R` the run folder on the rig, `NIGHT` the night
merge's unpacked tarball on the rig, `S` a folder on the rig with this repository's files (the preset,
`rig_server.sh`, the greedy request).

```bash
# on the build machine, CPU only: the harness against the merged tree ($W/src, built by scripts/build_llamacpp_docker.sh)
DOCKER_ARGS="--cpu-shares=256" nice -n 19 bash bench/mmvq_pascal/verify/build_multitoken_bitid.sh $W/src $W/src
# copy bench/mmvq_pascal/build/multitoken_bitid, rig_c690.sh and verify_prof.sh to the rig; then on the rig, idle:
STEPS="bitid perf served" BIN=$NIGHT/bin OUT=$R/out BITID=$R/multitoken_bitid \
  MODEL=<Swift Q2_0 first shard> DRAFT=<the MTP draft> PRESET=$S/release/flash-next-v3/settings.env \
  RIG_SERVER=$S/bench/agent_check/rig_server.sh REQ=$S/docs/data/2026-11-04_draft_one_card/run/greedy_request.json \
  VERIFY_PROF=$R/verify_prof.sh setsid nohup bash $R/rig_c690.sh > $R/console.out 2>&1 < /dev/null &
# the tables, on the build machine (the rig's out/ copied to $W/out), from this repository's root
mkdir -p $W/tables/perf $W/tables/served
awk -F'\t' -v OFS='\t' '{print $1, $2, "q5_K", 248320, 2560, $3, $4}' $W/out/perf/us_run.tsv > $W/tables/perf/us_run7.tsv
nice -n 19 python3 -I docs/data/2026-11-11_r79_c722_p106/run/perf_table.py $W/tables/perf/us_run7.tsv \
  --out $W/tables/perf > $W/tables/perf/perf.md
for d in 1-off 2-on 3-on 4-off; do
  mkdir -p $W/tables/served/$d
  nice -n 19 python3 -I docs/data/2026-11-10_served_verify_parts/run/verify_parts.py \
    --kpass-parts docs/data/2026-11-04_kpass_profile/run/kpass_parts.py --prof-summary scripts/prof_summary.py \
    --event-us 7.6 --out $W/tables/served/$d $W/out/served/$d/prof/prof-verify.csv > $W/tables/served/$d/report.md
done
nice -n 19 python3 -I docs/data/2026-11-11_r75_c690_p106/run/served_table.py $W/out/served --parts $W/tables/served \
  --out $W/tables/served > $W/tables/served/served.md
```

[bitid.tsv](bitid.tsv) is `multitoken_bitid`'s lines (`MUL_MAT q5_K n=… m=… k=… : … (d of v values differ, max rel x)`)
as columns, one block per setting.

## Provenance and licence

The model is UkisAI's Swift-1.5 GSQ-RCO Q2_0 of Qwen3.8-Flash-Next (its maker's licence) with its MTP draft layer. The
request is S37's greedy request
([greedy_request.json](../2026-11-04_draft_one_card/run/greedy_request.json)). The engine is Miner Llama (MIT, as
llama.cpp). The tables are this project's measurements.
