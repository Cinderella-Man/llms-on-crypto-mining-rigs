# M4's first MTP heads: trained on agent sessions, scored offline and on the agent check (2026-10-27)

The first heads for Flint trained only on publishable data. The data is agentic run 1 of
[bench/agentic_data](../../../bench/agentic_data/README.md): 1,177 gated sessions written by Flint, 15.4 M generated
tokens. Two heads trained for 240 minutes each (~0.9 epoch), a 3–12-card head (`gen`) and a nine-card head (`nine`).
They were scored on held-out sessions and Q51, then tested on the rig with the agent check. The write-up is
[docs/drafter.md](../../drafter.md), "M4 heads on the agent check" (the M4 method and its changelog). The code is
[bench/staircase_drafter/agentic_head.sh](../../../bench/staircase_drafter/agentic_head.sh) and `train.py`'s agent-session
options (`--spans`, `--ctx`, `--chunk-targets`).

## Results

**The data.** One export of the three writers' runs: 1,177 sessions, 1,053 verified (the task's tests pass), 18,885
turns, 15,426,749 generated tokens. Every session of every 20th module (7 of 74, by name hash) is held out:

- **training:** 1,108 sessions, 14.3 M generated tokens;
- **scored:** 40 sessions, 681k generated tokens;
- **unused:** the held-out modules' other 29 sessions.

The dump replayed 26.95 M tokens of the training sessions (tool results included) at ~1,860 tokens/s over three
dump instances, ~55 KB per token, on the 26-layer grid. Flint's output head on the dumped final states gives the
dumped top-1 at 99.0–99.1 % of positions. The vocabulary has 29,288 ids: every id Flint generated in training plus its
control tokens.

**Offline**, coupled acceptance at nine cards, all positions / fresh text (α 1):

| Head | Validation (sessions of training modules) | Held-out modules (40 sessions) | Q51 |
|---|---|---|---|
| nine | 0.365 | 0.277 / 0.235 | 0.156 / 0.141 |
| gen | 0.343 | 0.256 / 0.218 | 0.145 / 0.135 |

- **Sharpening:** α 3 (the export's default) raises validation by ~0.09: nine 0.455, gen 0.437 (`offline/final_alpha.tsv`).
- **Learning:** validation was still rising when the 240 minutes ended (`offline/val_curve.tsv`).
- **Q51:** low because the heads saw only agent sessions. Their vocabulary misses 14 % of Q51's chat tokens and 5–7 % of
  the rest.

**On the rig** (the v2.0 engine, Flint Q8_0, nine cards, 262k; the agent check with a 25-minute budget per arm):

| Arm | Tasks passed | Minutes per task (median) | First token per turn (median) | Generation tok/s |
|---|---|---|---|---|
| release line | 12 / 12 | 1.2 | 2.30 s | 45.0 |
| + nine-card head | 10 / 10 | 2.1 | 2.19 s | 29.7 |
| + 3–12-card head | 8 / 9 (one session ran to its time cap) | 1.5 | 2.78 s | 30.7 |

One guess costs 5.3 ms on one P106 (median of 300, the release split's 6-row window, 8,192 positions; nine 5.308 ms,
gen 5.293 ms). The heads slow the agent: at this cost and ~20–30 % right guesses on agent text, guessing loses time.
With `--pipe-guess staircase` and no store, the engine also drops the request's continuations, which the release line
uses. The cheaper-guess work starts from that ([2026-10-27_cheap_guess](../2026-10-27_cheap_guess/README.md)).

## Files

| File | What |
|---|---|
| `offline/val_curve.tsv` | validation every ~15 minutes: head, round (`240min`, or `resumed` for the hour after a restart), minutes, step, window (g6, g9, g9light), coupled, top, KL |
| `offline/final_alpha.tsv` | the 240-minute heads' final validation per window: coupled acceptance at α 1, 1.5, 2, 3, 5, top, KL, positions |
| `offline/scores_heldout.md`, `scores_heldout.json` | `score.py` on the 40 held-out sessions (`--spans`), every window and card count, 95 % intervals by session |
| `offline/scores_q51.md`, `scores_q51.json` | the same on Q51 |
| `offline/train_log_gen.jsonl`, `train_log_nine.jsonl` | the 240-minute training logs (every 20 steps; validation every ~15 min; the final evaluation) |
| `offline/train_log_*_resumed.jsonl` | the same heads resumed for an hour (learning rate restarted toward ~3.6 epochs; stopped at step 1,704) |
| `offline/config_gen.json`, `config_nine.json` | `train.py`'s config: profile, sizes, items, every option (`$WORK` = the run folder) |
| `offline/heldout_session_ids.txt` | the 40 scored held-out sessions (row ids of the export's `sessions_seq.jsonl`) |
| `agent/{base,nine,gen}/table.md`, `compare.md`, `gonogo.md` | the agent check's reports per arm |
| `agent/{base,nine,gen}/attempts.csv`, `turns.csv` | per task and per turn: result, minutes, turns, tokens, prefill/decode/fixed time, time to first token |
| `run/head.env.example` | `agentic_head.sh`'s settings for this round |
| `run/head_resume.sh` | snapshot a round and resume both heads (`start`), then gate, score and export (`finish`) |
| `run/rig_test.sh` | the rig test: the guess cost, then the three agent-check arms |

The heads (`head.pt`) and their GGUF side files (0.23 GB each) are not here: files over 1 MB stay off this repository.

## Made with

```bash
. head.env                                   # run/head.env.example, filled in
bash "$BENCH/staircase_drafter/agentic_head.sh"   # export split weights dump vocab train score gguf
# the first training start ran out of memory (pieces of 16,384 targets); relaunched with CHUNK=4096:
STEPS="train score gguf" bash "$BENCH/staircase_drafter/agentic_head.sh"
# Q51's dump input numbers its 38 rows 0-37 (its ids repeat numbers across categories), dumped like the training data:
DUMP_IN=$Q51_SEQS DUMP_OUT=$WORK/seqs/q51.bin DUMP_LAYERS=4,6,8,9,10,12,14,16,17,18,20,22,24,25,27,29,30,33,34,36,37,39,41,42,44,47 \
  DUMP_ALLPOS=4,6,9 DUMP_THREADS=12 DUMP_GPU_TOPK=1 \
  $LLAMA/build-cuda/bin/llama-staircase-dump -m $MODEL -ngl 99 -fa on -c 8192 -b 512 -ub 512
# the rig test (the agent check's machine values from an env file like bench/agent_check/release.env.example):
set -a; . release.env; set +a
OUT=<results> HEADS_SRC=$WORK/gguf RIG_HEADS=<folder on the rig> QUANT=<ssh alias> bash run/rig_test.sh
```

Times are from a shared machine and are not speed results; speed comes only from the rig.

## Provenance and licence

The sessions are Flint's output (Apache-2.0) on the Python standard library (PSF-2.0) and Elixir's library
(Apache-2.0). They are gated at every stage by `bench/agentic_data` (sources, licences, held-out rules): no Q51 prompt
and no agent-check task appears in them. Q51 is the project's held-out test set ([2026-10-23_q51](../2026-10-23_q51/README.md)).
The agent check uses Exercism tasks (MIT, licences in `bench/agent_check/tasks/polyglot/`). The tables here are under
this repository's licence.
