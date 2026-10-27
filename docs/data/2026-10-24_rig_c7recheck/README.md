# Q155/Q156 short-prompt streams, Q153 KV pad, and the timer-corrected pipelined speed-ups (2026-10-24)

The last rig block of 2026-10-24 and the analysis that came out of it. The block ran the pipelined-serving build
`2650855b2` (fork branch `pascal-p106-v2-pipeserve`, without the persistent MoE kernel) with Flint K2h-r16 on nine cards
with a light last card (`-ts 6,6,6,6,6,6,6,5,2`): several conversations with short prompts at N = 1, 4 and 8, plain and
with the token fed on the card (Q155/Q156), and depth 8 on the edit prompt with the KV cache growing in 256-cell steps
(Q153, `LLAMA_KV_PAD=256`). Reading its traces showed that llama-pipespec's tok/s had counted the wait for the prompt's
tail, so every earlier pipelined summary was re-read from its traces: `corr.txt`. Results: registry rows Q155, Q153,
Q156 and Q87 in [quants/experiments.md](../../../quants/experiments.md),
[pipelined_speculation.md](../../research/notes/pipelined_speculation.md) and
[docs/engine_features.md](../../engine_features.md).

| File | What |
|---|---|
| `corr.txt` | per run set, job and prompt: the trace-based rate (tokens per second from the first to the last sampled token) and speed-up over plain decoding, with llama-pipespec's own tok/s and speed-up in brackets; for [2026-10-24_rig_q87](../2026-10-24_rig_q87/README.md) `k2h-9c` and `k2h-6c` (Q87) and [2026-10-24_q88_persistent](../2026-10-24_q88_persistent/README.md) `v2next-k2h-9c` and `v2next-ud-9c` (Q160) |
| `q155-short-9c/summary.tsv` | [rig_hostloop.sh](../../../bench/pipespec/rig_hostloop.sh) with packed uploads, the embedding on card 1 and fed tokens (`u2f`), chat and code prompts: plain, depth 0 and N = 1/4/8 conversations, each also fed (`-fed`): tokens, total tok/s, host µs per token in submits, waits and the rest, identity with plain runs |
| `kvpad256-9c/summary.tsv` | [rig_run.sh](../../../bench/pipespec/README.md) on the edit prompt, 512 tokens: plain, depth 0, `cont` at depth 8, `copy` and `spec` at depth 3, oracle at depth 8 all right and all wrong, and at depth 3 30 % wrong, with `LLAMA_KV_PAD=256`; tok/s from llama-pipespec's clock |
| `traces.tar.gz` | the 8 per-token traces of these runs in `<run>/<job>.trace` |

The tools are [bench/pipespec/trace_rates.py](../../../bench/pipespec/trace_rates.py) (first-sample time and trace rate
per job, next to the summary's rate) and [trace_speedups.py](../../../bench/pipespec/trace_speedups.py) (the per-depth
tables of `corr.txt`).

## What it shows

**The timer artifact.** llama-pipespec started its clock before the first sample, and the first sample waits for the
prompt's tail (`llama_decode` returns before the cards finish). The wait was 0.1–3.3 s and differed per job, so `tok_s`
in every earlier summary was low and long-prompt speed-ups were wrong in both directions. Fork commit `99c3a7c00` starts
the clock after a `llama_synchronize`, and rig_run.sh now takes `tok_s` and `speedup` from the traces. Best `cont` depth
from the traces (llama-pipespec's figure in brackets):

| Run | chat | code | edit | rewrite |
|---|---|---|---|---|
| K2h-r16, nine cards (Q87) | 1.38× (1.38) | 1.43× (1.42) | 1.41× (1.55) | 1.72× (1.30) |
| K2h-r16, six cards (Q87) | 1.38× (1.37) | 1.35× (1.34) | 1.44× (1.41) | 1.84× (1.39) |
| K2h-r16, nine cards, v2-next (Q160) | 1.31× (1.30) | 1.38× (1.35) | 1.29× (1.27) | 1.81× (1.32) |
| UD-Q4_K_XL, nine cards, v2-next (Q160) | 1.33× (1.32) | 1.26× (1.23) | 1.43× (1.40) | 2.58× (1.76) |

So the Q87 headline of 1.3–1.55× becomes ~1.35–1.85× on the Q87 runs (edits ~1.4, rewrites 1.7–1.85; 2.6 with UD
on v2-next). The oracle ceilings rise too: every guess right gives 220 tok/s at depth 8 on nine cards (6.9×), 222 at
depth 4 on v2-next (6.2×).

**Q153, KV pad 256, edit, depth 8** (trace rates): plain 33.9 tok/s, depth 0 37.0 (1.09×), `cont` depth 8 37.8 (1.11×),
the oracle at depth 8 with every guess right 209.5 (6.17×), every guess wrong 29.1 (0.86×). Plain is not slow: its
summary's 27.6 held a 3.4 s prompt tail (the first long prompt of the process after a 22-token warm-up). At each
256-cell KV step about 7 consecutive tokens take +12 ms (tokens 222–228 and 478–485), where before Q153 18 tokens took
+9.7 ms each: about 0.3 ms per token on average. Every job identical to plain decoding. The 1,024-cell run did not fit
in the block.

**Q155/Q156, short prompts, nine cards:** one conversation 34.3 tok/s plain, 33.9 (n1), 34.4 (n1 fed); four
conversations 146.1 tok/s in total, 152.7 fed (+4.6 %); eight 262.6, 255.2 fed (−2.8 %), ~33 tok/s each. Summed per
conversation the eight-conversation runs reach 282.2 and 278.1 tok/s (−1.5 % fed), and the slowest first token takes
1.38 and 1.54 s. Every output identical to its plain run. A fed token makes card 1's in-order stream wait for its
source's sample, and the other conversations' work queued behind it waits too; with the pipeline full (N = 8) that
costs more than the host round trip it saves, so feeding pays only while fewer items than cards are in flight.

## How it was made

On 2026-10-24 21:11–21:27, nothing else on the rig, the build `llama.cpp-20261024-2650855b2-pascal-p106-v2-pipeserve`
unpacked in `/opt/llama-builds` ([scripts/build_llamacpp_docker.sh](../../../scripts/build_llamacpp_docker.sh)),
bench/pipespec copied to `/tmp/pipespec` on the rig:

```bash
B=/opt/llama-builds/llama.cpp-20261024-2650855b2-pascal-p106-v2-pipeserve/bin
M=/mnt/models/flint/flint-q113-K2h-r16.gguf
ssh "$RIG" "env BIN=$B MODEL=$M GPUS=0,1,2,3,4,5,6,7,8 TS=6,6,6,6,6,6,6,5,2 TAG=q155-short-9c CONFIGS=u2f NS='1 4 8' \
    PROMPTS=/tmp/pipespec/prompts/chat.txt,/tmp/pipespec/prompts/code.txt ENVS='LLAMA_NO_MOE_DOWN_RESCALE=1' \
    bash /tmp/pipespec/rig_hostloop.sh"
ssh "$RIG" "env BIN=$B MODEL=$M GPUS=0,1,2,3,4,5,6,7,8 TS=6,6,6,6,6,6,6,5,2 TAG=kvpad256-9c PROMPTS=edit DEPTHS='8' \
    ENVS='LLAMA_NO_MOE_DOWN_RESCALE=1 LLAMA_KV_PAD=256' bash /tmp/pipespec/rig_run.sh"
```

The traces were packed with `tar --sort=name` and `gzip -9n`. `corr.txt` is the output of

```bash
python3 -I bench/pipespec/trace_speedups.py docs/data/2026-10-24_rig_q87/k2h-9c docs/data/2026-10-24_rig_q87/k2h-6c \
    docs/data/2026-10-24_q88_persistent/v2next-k2h-9c docs/data/2026-10-24_q88_persistent/v2next-ud-9c
```

run after unpacking each folder's `traces.tar.gz` in place. The per-conversation totals and first-token times come from
the run's per-job result lines (not included, nor the nvidia-smi log). Flint K2h-r16: recipe in
[quants/experiments.md](../../../quants/experiments.md) (Q113), Flint is Apache-2.0.
