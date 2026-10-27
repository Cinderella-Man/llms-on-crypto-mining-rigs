# Overnight rig queue: verify cost, a 4-token step's profile and the job scripts (2026-10-23)

On the evening of 2026-10-23 the rig worked through a queue of jobs with
[scripts/rig_queue.sh](../../../../scripts/rig_queue.sh) until a deadline stopped it at 21:46. The finished jobs fed
registry rows Q19 (verify cost and the drafter no-go), Q54 (Flint at depth) and Q43 (Qwen3.6's MTP head)
([quants/experiments.md](../../../../quants/experiments.md)); the numbers are in those rows. This folder holds the
verify-cost tables, the GPU profile of a 4-token step, and every job script with whether it ran.

| File | What |
|---|---|
| `verify_cost/q19_s4l0_on/`, `verify_cost/q19_s4l0_off/` | [bench/eagle3_flint/verify_cost.sh](../../../../bench/eagle3_flint/verify_cost.sh) part `bench`: llama-bench steps of 1–8 tokens and tg64, Miner S4-L0 on 6 cards, fork build `eeaa59660`, Pascal kernels on / off (`GGML_CUDA_MMVQ_PASCAL=0`); `results.tsv` and `summary.txt` |
| `verify_cost/q19_s4l0_synth/` | part `synth`: llama-server with `-bs` and synthetic full acceptance at widths 1, 3 and 5, 8,192 tokens of context |
| `profile_pp4_by_label.csv`, `profile_pp4_classes.csv`, `profile_pp4_totals.csv` | GPU time of one 4-token prompt step of S4-L0 on 6 cards (`GGML_CUDA_PROFILE`, the last 24 of 25 steps): by launch group, by tensor class, and in total |
| `run/common.inc`, `run/<nn>-<name>.sh` | the queue's jobs, each with its status in the header |

`results.tsv` columns (no header row): method, width, tokens verified per step, ms per step, tok/s, share of drafts
accepted, steps. For `bench_pp` rows only the first four are filled; the `bench_tg` row gives ms and tok/s per token.

| Tokens per step | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| llama-bench, kernels on (ms) | 57.0 | 94.1 | 86.0 | 117.3 | 163.6 | 170.0 | 207.6 | 212.1 |
| llama-bench, kernels off (ms) | 72.3 | 95.2 | 87.3 | 119.0 | 164.7 | 171.8 | 209.3 | 211.1 |

tg64 (CUDA graphs) took 30.8 ms per token with the kernels on and 42.3 ms with them off; llama-bench's prompt steps run
without CUDA graphs. With llama-server and synthetic acceptance, 1.00, 2.90 and 4.05 verified tokens per step took
25.9, 57.7 and 92.1 ms (2.22× and 3.55× one token).

GPU time of one 4-token step (ms, summed over the 6 cards; 130.4 ms wall time while profiled):

| Tensors | Type | Launches | ms |
|---|---|---|---|
| routed experts gate/up | Q5_0 | 47 | 25.65 |
| routed experts down | Q8_0 | 47 | 16.73 |
| attention q | Q4_1 | 49 | 5.84 |
| attention output | Q8_0 / Q4_1 | 33 / 16 | 4.98 / 1.61 |
| shared expert gate, up | Q8_0 | 94 | 4.68 |
| attention k, v | Q8_0 | 98 | 3.13 |
| shared expert down | Q8_0 | 47 | 2.76 |
| router | F16 | 47 | 2.48 |
| output head | Q8_0 | 1 | 2.34 |
| dense FFN, layers 0–1 | Q8_0 | 6 | 1.33 |
| everything else | | 587 | 9.95 |
| **total** | | | **81.49** |

At 4 tokens the step runs on llama.cpp's stock multi-token kernels: the routed experts alone take 52 % of the GPU time
(the one-token step of the same file takes 32.7 ms in total, [Q38](../../2026-10-23_q38/)).

The jobs (run order is name order; three were renamed in the queue, and five unsaved jobs ran between them: an A/B of
the FP32 activation sums and shared-expert fusion on UD-Q4_K_XL, and a Q43 rerun with a warm-up request):

| Job | Row | What | Status |
|---|---|---|---|
| `10-q19-bench-on`, `11-q19-bench-off` | Q19 | verify cost, llama-bench part, kernels on / off | ran |
| `12-q19-prof-pp4` | Q19 | the 4-token profile | ran |
| `13-q19-synth` | Q19 | verify cost, server part with synthetic acceptance | ran |
| `14-q65-host-w4` | Q65 | host timeline of a width-4 verify step | never ran |
| `20-q54-6c-f16`, `21-q54-6c-q8` | Q54 | tg64 at depth 0 / 8k / 32k, 6 cards, f16 and q8_0 KV | ran |
| `22-q54-9c-f16` | Q54 | tg64 at 32k / 64k / 131k, 9 cards | ran |
| `23-q54-9c-q8`, `24-q54-prof32k` | Q54 | 131k with q8_0 KV; profiles at 32k | never ran |
| `30-q19-draft` | Q19 | step time of a random-weight EAGLE-3 head | never ran |
| `40-q68-tool-host` | Q68 | tool-request sampler cost on the rig's CPU | never ran |
| `41-q55-tools-6c` | Q55 | tool-request A/B on 6 cards | never ran |
| `50-q43_plain`, `52-q43_mtp2` | Q43 | Qwen3.6-35B-A3B Q4_0 on 5 cards, plain and MTP with 2 drafted tokens | ran |
| `51-q43_mtp1`, `53-q43_mtp3`, `54-q43_mtp2_synth`, `55-q43_mtp2_kernels_off` | Q43 | the other MTP settings | never ran |
| `60`–`65-q75-chat2-*` | Q75 | a second round of the six Flint chat runs | never ran |

Made on 2026-10-23 on the rig. The queue directory `/tmp/rq` held `jobs/` (the job scripts and `common.inc`) and
`scripts/` (copies of the repository scripts the jobs call, listed in `common.inc`):

```bash
RIG=${RIG:-rig}
ssh "$RIG" 'mkdir -p /tmp/rq/jobs /tmp/rq/scripts'
scp bench/eagle3_flint/verify_cost.sh bench/eagle3_flint/draft_cost.sh bench/tool_sampling/tool_request_ab.sh \
    scripts/server_bench.sh "$RIG":/tmp/rq/scripts/
scp docs/data/2026-10-23_overnight/rig/run/* scripts/rig_queue.sh "$RIG":/tmp/rq/jobs/
ssh "$RIG" 'mv /tmp/rq/jobs/rig_queue.sh /tmp/ && nohup bash /tmp/rig_queue.sh > /dev/null 2>&1 &'
```

`verify_cost.sh` writes `/tmp/eagle3_flint/<TAG>/`. The profile tables were made from the job's `prof-pp4.csv` with
[prof_to_csv.py](../../2026-10-23_q38/run/prof_to_csv.py) and [profile_classes.py](../../2026-10-23_q38/run/profile_classes.py)
(`--tokens 24`). The raw trace (1.45 MB), the llama-bench and server logs and the queue's own log are not published.
