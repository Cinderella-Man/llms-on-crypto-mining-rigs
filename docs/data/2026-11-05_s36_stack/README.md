# The draft length chosen per step on the whole stack (S36, 2026-11-05)

What this measured: the coding-agent check ([bench/agent_check](../../../bench/agent_check)) of Qwen3.8-Flash-Next on
the first 4 GB rig of nine P106-090, on S35's line ([2026-11-04_s35_stack](../2026-11-04_s35_stack/README.md)) with
the draft length chosen per step (`LLAMA_SPEC_ADAPT=1`, Q687), and S35's arm run again as the spread of repeated
sessions. Plain-decode diagnostics (llama-bench tg128) compare the `-ladder4` build with `-kshape`. Registry rows
Q687, Q660 and Q962 in [quants/experiments.md](../../../quants/experiments.md).

Both agent arms ran on S35's build (`-kshape` `39c3ab5f1`) with S35's line exactly: pmix-q2-fast, the MTP draft
line with the draft head's shortlist (`LLAMA_DRAFT_SHORTLIST=32768,8192,2`), `GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=32768
GGML_CUDA_MMVQ_PASCAL_KCOL_TABLE=0` and the Q8_0 hops (`LLAMA_HOP_CAST=q8_0 GGML_CUDA_HOP_STAGED=1`). The bugfix task
set, one attempt each, a 40-minute budget, the RAM watchdog's floor at 60 MiB. The expert-skip arms planned on
`-ladder4` were not run; the skip's agent-check reading moves to the next cumulative build.

## Result

| Arm | Passed | Minutes, all tasks | Gen tok/s | Prompt tok/s | First token per turn, s (median, p90) | Acceptance | Tokens per step | `s35stack`'s work at the arm's rates (12 tasks / 11 without the runaway) |
|---|---|---|---|---|---|---|---|---|
| `s35stack` (S35, 2026-11-04) | 12 / 12 | 34.8 | 18.76 | 67.8 | 7.42, 17.34 | 0.708 | 3.12 | 1.00× / 1.00× |
| `s36ref` (the same arm, today) | 12 / 12 | 29.5 | 18.97 | 69.0 | 6.54, 15.73 | 0.699 | 3.10 | 1.01× / 1.01× |
| `s36adapt` (+ `LLAMA_SPEC_ADAPT=1`) | 12 / 12 | 35.1 | 18.07 | 69.9 | 6.93, 15.81 | 0.789 | 2.80 | 0.99× / 1.02× |

- **The draft length per step: negligible.** At equal work (the reference's prompt and generated tokens at each arm's
  rates, [equal_work.md](equal_work.md)) it is 0.99× over all twelve tasks and 1.02× over the eleven without its one
  runaway (pig-latin, 12 minutes at the time cap, 11 turns, still passing), against 1.00–1.01× between two sessions of
  the same arm. Its generation rate is 18.07 tok/s over the session (the runaway's long contexts) and 19.76 on the
  eleven tasks, against 19.05 and 19.08 for the two reference sessions.
- **What the rule chose** ([adapt.tsv](adapt.tsv), every one of 9,281 steps recomputed from its log with 0 problems):
  verify widths 1 / 3 / 4 in 1,899 / 1,013 / 6,369 steps, 1.80 accepted tokens a step, 2.80 tokens a step against
  3.10–3.12 for the fixed three-token draft; the accepted share of what it verified is 0.789. The cost table was
  C271's real-text T(k) of the build before `-kshape` (`t_k_q2_0_s32k_real.txt`, T(4) 151.6 ms), which prices wide
  verifies above `-kshape`'s own (`t_k_kshape_s32kt0_real.txt`, T(4) 140.4 ms), so the rule drops to one token more
  often than this build needs.
- **Acceptance and tokens per step** ([acceptance.tsv](acceptance.tsv)) are summed over the server's per-request
  lines. The server counts only the steps that drafted, so for `s36adapt` its 3.26 tokens a step leaves out the 1,899
  width-1 steps; 2.80 is over every step.

## Plain decode: `-ladder4` against `-kshape`

llama-bench `-p 0 -n 128 -r 3`, pmix-q2-fast, `-ts 6/5/5/6/5/5/6/5/6 -b 2048 -ub 512 -lm dio`, the k-column setting
above, one model load per row ([bench.tsv](bench.tsv)):

| Build | Hop line (Q8_0 staged hops), tok/s | No hop cast, tok/s |
|---|---|---|
| `-kshape` `39c3ab5f1` | 17.06 (S35), 17.07, 17.05 | 14.98 (S35), 15.49 |
| `-ladder4` `e3b30303d` (every switch off) | 15.98, 16.80 | 15.32 |
| `-ladder4` + `LLAMA_MOE_TAIL=p=0.9,renorm=1,hist=0` | 16.98 | — |
| `-moeskip` `f3072cbe8` | 16.81 | — |
| `-n2p` `1a0832375` | 16.18 | — |
| `-mailbox` `5aebbddd9` | 17.03 | — |
| `-coldtier` `f5e600bd2` (no CUDA code) | 16.19 | — |

- **Two loads of one build differ by up to 0.8 tok/s** (`-ladder4`: 15.98 and 16.80, ~3 ms a token), while the three
  repetitions inside a load agree within 0.05. The first `-ladder4` load read 6 % below `-kshape`; its second load and
  the single branches put the loads in levels near 17.05, 16.8 and 16.2 tok/s that do not follow the code (`-coldtier`
  changes no CUDA code and reads like `-n2p`). A difference between builds smaller than ~1 ms a token needs alternated
  loads, as [2026-11-04_moe_skip](../2026-11-04_moe_skip/README.md)'s switch-off comparison did.
- With `GGML_SCHED_DEBUG=1` both builds give the same decode graph (6,217 nodes, 10 splits) and the same split inputs:
  one 10 KB `hop_narrow` tensor per card boundary. The staged hops count the same copies and chunks in both.

## Files

| File | What it is |
|---|---|
| [report.md](report.md) | the agent check's report of the three arms: per task, per arm, time to first token per turn, the go/no-go checks |
| [equal_work.md](equal_work.md) | the arms at equal work, with and without the task one arm ran away on (`shared_tasks.py`) |
| [acceptance.tsv](acceptance.tsv) | per arm: requests, accepted and drafted tokens, acceptance, steps, tokens per step |
| [adapt.tsv](adapt.tsv) | `s36adapt`'s step log summarized: steps by verify width, drafted, verified and accepted per step |
| [bench.tsv](bench.tsv) | the tg128 diagnostics, one row per model load |
| [run/s36_arms.sh](run/s36_arms.sh) | the run: bench, prep, the arms, the report |
| [run/acceptance.sh](run/acceptance.sh) | acceptance and tokens per step from llama-server's logs |

## Commands

From the repository root, with S12's machine file (`docs/data/2026-11-03_flash_next_mtp/run/s12.env.example`, with
`DRAFT` the MTP draft) and S35's ports and client folders after it, so the report pairs the arms with `s35stack`:

```bash
ENVFILE=<machine file> RIG_PORT=8192 RELAY_PORT=8199 CLIENT_PORT=18493 PROXY_PORT=18494 \
  CLIENT_DIR=<S35's client folder> CLIENT_OUT=<S35's client results folder> \
  MODEL=<pmix-q2-fast first shard on the rig> BUILD=llama.cpp-20261104-39c3ab5f1-pascal-p106-v3-b12076-kshape \
  COST=<t_k_q2_0_s32k_real.txt on the rig> OUT=<local folder> STEPS='prep arms report' ARMS='s36adapt s36ref' \
  bash docs/data/2026-11-05_s36_stack/run/s36_arms.sh
# a bench row: STEPS=bench BUILD=<build> ARMS_BENCH='hop' (or 'hop skip', 'nohop') BT=<file prefix>
python3 docs/data/2026-11-03_retype_agent/run/shared_tasks.py <CLIENT_OUT>/res/attempts.csv s35stack s36ref s36adapt
bash docs/data/2026-11-05_s36_stack/run/acceptance.sh <CLIENT_OUT>/rig/server-*.log
python3 docs/data/2026-11-04_adaptive_draft/run/adapt_log.py --check <CLIENT_OUT>/rig/adapt-s36adapt.jsonl
```

The cost table is `docs/data/2026-11-04_kcurve_p106/cost/t_k_q2_0_s32k_real.txt`.

## Provenance

Run on 2026-11-05: `s36adapt` 13:32–14:13, `s36ref` 14:48–15:23 (local time), the benches 14:14–16:05. The tasks are
the agent check's bugfix set (the task IDs in [report.md](report.md); `bench/agent_check` rebuilds them). The report
holds no prompts or transcripts.
