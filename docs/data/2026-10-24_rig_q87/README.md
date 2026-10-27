# Q87 and Q120/Q121: pipelined speculation and the host loop on the rig (2026-10-24)

The first rig run of pipelined speculation: [bench/pipespec/rig_run.sh](../../../bench/pipespec/README.md) on six and
nine cards (plain decoding against the executor at depth 0, the `cont` guesser at depths 1–8, `copy` and `spec` at
depth 3, oracle ceilings), the identity checks before it, and [rig_hostloop.sh](../../../bench/pipespec/rig_hostloop.sh)
for the host side of a token (Q120/Q121: synchronous against packed uploads, host-fed against device-fed tokens, one
to eight conversations). Flint K2h-r16 on the build `pascal-p106-v2-pipespec2` `f249104ab` (pipelined executor and
Q120/Q121), MoE down rescale off (`LLAMA_NO_MOE_DOWN_RESCALE=1`), greedy on the last card, 512 tokens per job. Results: registry
rows Q87 and Q120 in [quants/experiments.md](../../../quants/experiments.md) and
[pipelined_speculation.md](../../research/notes/pipelined_speculation.md); the speed-ups corrected for the timer
artifact are in [2026-10-24_rig_c7recheck/corr.txt](../2026-10-24_rig_c7recheck/corr.txt).

| File | What |
|---|---|
| `k2h-6c/summary.tsv`, `k2h-9c/summary.tsv` | per job: prompt (chat, code, edit, rewrite), guesser, depth, tokens, tok/s and speed-up over plain from llama-pipespec's own clock, guesses right, wrong and dropped, mean guesses in flight, `same_as_plain`; six cards depths 1–5, nine cards depths 1–6 and 8 |
| `k2h-hostloop/summary.tsv` | per upload mode (`u0` synchronous uploads, `u2` packed, `u2f` packed with the embedding lookup on card 1 and fed tokens) and job (plain, depth 0, N = 1/4/8 conversations, `-fed` with the token fed on the card): tokens, tok/s, host µs per token inside the submit calls, inside the waits and the rest, identity |
| `k2h-6c/host.csv`, `k2h-9c/host.csv`, `k2h-hostloop/host-u0.csv`, `host-u2.csv`, `host-u2f.csv` | the process's CPU use once a second (Unix time, % of one core) |
| `gpu_summary.csv` | per run and card, from the nvidia-smi log (every ~1 s: SM clock, power, temperature, utilization): samples, busy samples (utilization above 0), mean utilization, SM clock mean / min / max, power mean / max, hottest reading. The jobs of a run follow each other in one process with no idle gap, so the log has no job boundaries and the summary is per run |
| `traces.tar.gz` | the 94 per-token traces of these runs (time of each sampled token in µs, guesses in flight, outcome) in `<run>/<job>.trace`; unpacked here they sit next to the summaries, the input of [bench/pipespec/trace_speedups.py](../../../bench/pipespec/trace_speedups.py) |
| `run/q87_block.sh` | the driver: identity (host sampling, then sampling on the last card), rig_run.sh on six and nine cards, rig_hostloop.sh |

## What it shows

| Question | Result |
|---|---|
| Identity, 128 tokens, depths 1/3/5, oracle 0 % and 30 % wrong, copy, continuation, ngram-simple and ngram-map-k guessers, six cards | every run identical to plain decoding in tokens, per-token logits, the KV state and probe logits, with host sampling and with sampling on the last card |
| Identity in the 512-token jobs | every job identical except `cont` on the edit prompt at depth ≥ 4 (six cards d4–d5, nine cards d4–d8): one late divergence, fixed in fork commit `fbb1b24f3` (llama.cpp's `find_slot` restarted the cell search at cell 0 of a full sliding-window cache; [pipelined_speculation.md](../../research/notes/pipelined_speculation.md)) |
| Best `cont` depth, llama-pipespec's clock (plain → pipelined) | six cards: chat 32.6 → 44.8 (d3), code 34.5 → 46.2 (d3), edit 28.4 → 39.9 (d3), rewrite 34.1 → 47.3 (d4); nine cards: chat 31.0 → 42.7 (d5), code 31.1 → 44.2 (d5), edit 26.4 → 40.9 (d2), rewrite 35.9 → 46.7 (d6). From the traces the speed-ups are 1.38 / 1.35 / 1.44 / 1.84× on six cards and 1.38 / 1.43 / 1.41 / 1.72× on nine ([corr.txt](../2026-10-24_rig_c7recheck/corr.txt)) |
| Oracle ceilings (chat) | 70 % of guesses right at depth 3: 83.9 / 81.9 tok/s (six / nine cards); every guess right: 168 at depth 5 on six cards, 186 at depth 8 on nine (traces: 193 and 220) |
| Host loop, nine cards (Q120/Q121) | eight conversations 54.9 tok/s in total with synchronous uploads, 66.3 with packed uploads, 66.6 packed with the embedding on card 1; fed tokens at N = 4 and 8 only 22.1 and 22.3 (the fed token waited for the pipeline to drain: Q156); every output identical to its plain run |
| Clocks during the runs (`gpu_summary.csv`) | busy cards at 1,562–1,807 MHz mean SM clock (lowest in the host-loop run), 33–48 W mean power, at most 54 °C; the idle cards of the six-card run stay at 139 MHz |

The `tok_s` of these summaries comes from llama-pipespec's own clock, which also counted the wait for the prompt's tail
before the first token (0.1–3.3 s per job, fork commit `99c3a7c00` fixed it); the traces do not have that wait. To
reproduce the corrected tables, run in this folder:

```bash
tar xzf traces.tar.gz && python3 -I ../../../bench/pipespec/trace_speedups.py k2h-9c k2h-6c
```

## How it was made

On 2026-10-24 14:41–16:03, the rig otherwise idle, with the build's tarball unpacked in `/opt/llama-builds`
([scripts/build_llamacpp_docker.sh](../../../scripts/build_llamacpp_docker.sh)) and K2h-r16 at
`/mnt/models/flint/flint-q113-K2h-r16.gguf`:

```bash
RIG=rig OUT=. docs/data/2026-10-24_rig_q87/run/q87_block.sh
```

It copies bench/pipespec to the rig, runs `check_identity.sh` (`N=128 DEPTHS='1 3 5' MISSES='0 0.3'`, six cards),
`rig_run.sh` (`TAG=k2h-6c GPUS=0,…,5`; `TAG=k2h-9c GPUS=0,…,8 DEPTHS='1 2 3 4 5 6 8'`) and `rig_hostloop.sh`
(`TAG=k2h-hostloop NS='1 4 8'`), and copies each output folder back. `gpu_summary.csv` was computed from the runs'
`gpu.csv` (the nvidia-smi log rig_run.sh writes); the traces were packed with `tar --sort=name` and `gzip -9n`. The
nvidia-smi logs, the run logs with Flint's outputs and the per-run hash lists are not included. Model: Flint K2h-r16
(recipe in [quants/experiments.md](../../../quants/experiments.md), Q113), Apache-2.0; the prompts are bench/pipespec's.
