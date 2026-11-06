# Flint's side-by-side on a second rig: the re-read fix against `v3base` (2026-11-06)

The agent check (`bench/agent_check/release_check.sh`: a coding agent fixing 12 small bugs with real tool runs, one
attempt each, 12-minute cap) with Unsloth's Flint Q8_0 on nine P106-090 cards, on the v3 release build `-ladder5`
`1551f5fe1`. Every arm runs Flint's release line, `v3base`: `-c 262144 -b 512 -ub 128 -ts 6,6,5,5,6,5,5,6,6
--pipe-depth 4`, plus the harness's `-np 1 -bs --ctx-checkpoints 4 --cache-ram 0` and Flint's sampling (T 1.0,
top-p 0.95, top-k 20). One switch changes per arm, against `v3base` run on the same rig the same day. It continues
[2026-11-05_release_regression](../2026-11-05_release_regression/README.md), where `v3base` ran on the first rig.

Result pages: [release/flint-v3/README.md](../../../release/flint-v3/README.md), [docs/switches.md](../../switches.md),
[release/notes-v3.0.md](../../../release/notes-v3.0.md).

## Results

| Arm | Switches | Passed | Gen tok/s | Prompt tok/s | TTFT median, p90 s | Minutes, all tasks | Equal work, min a task |
|---|---|---|---|---|---|---:|---|
| `v3base` | none: every switch at its default | 12/12 | 45.51 | 179.7 | 2.64, 7.71 | 31.2 | 2.60 |
| `reread` | `LLAMA_SERVER_TEXT_REUSE=1 LLAMA_SERVER_TAIL_CKPT_MIN=4096` | 12/12 | 45.78 | 178.8 | 2.25, 5.97 | 24.3 | 2.59 (−0.4 %) |

The equal-work column times one fixed amount of work for both arms: `v3base`'s prompt tokens at the arm's prompt
speed, plus `v3base`'s generated tokens at the arm's generation speed, plus `v3base`'s time outside the server, divided
over its 12 tasks ([agent_tables.py](../2026-11-05_keepalive/run/agent_tables.py)). Minutes for all tasks depend on
what each session generated: `v3base`'s sessions decoded 120 s a task (mean), `reread`'s 102.

**The re-read fix (`reread`): negligible.** At equal work it is 2.59 against 2.60 minutes a task (−0.4 %), and
generation is 45.78 against 45.51 tok/s. The fix did what it is for: on 11 requests the prompt's tokens differed from
the cache while its text was the same, and the server kept 87,535 cached tokens in all instead of reading them again
(`v3base`: none). The arm read 2,982 prompt tokens a task against `v3base`'s 5,847, and its first token came at 2.25
against 2.64 s (median), 5.97 against 7.71 s (p90). The equal-work column holds the prompt tokens fixed, so it does
not count tokens the fix saves from being read. One run each: the spread between two runs of `v3base` on this rig was
not measured, so the first-token difference has no class yet. The fix stays behind its switch, out of Flint's preset.

On the first rig, `v3base` on the build before (`1f7215948`) gave 45.9 tok/s, 12/12, 2.67 s first token (median).

**Not run:** the staged hop (`GGML_CUDA_HOP_STAGED=1`), the negligible group (`-t 1 -tb 1`), the answer store
(`--pipe-store`), the keep-alive alone (`GGML_CUDA_KEEPALIVE=decode`) and a second `v3base` for the spread. The
staged arm was stopped during its first task, so it has no result.

## Files

| File | What |
|---|---|
| `agent_t2d.tsv` | both arms side by side: passed, turns, prompt and generation tok/s, first token per turn (median, p90, first turn), minutes, equal work against `v3base` |
| `agent/flint-<arm>/` | each arm's report: `compare.md` (per task, first token per turn), `table.md`, `gonogo.md`, `attempts.csv`, `turns.csv`; `validation.txt` (the task set checked in the sandbox image) with `v3base` |
| `run/merge_arms.py` | puts arms that ran in separate folders under one arm name (`rel`) into one folder with their own names, for `agent_tables.py` |

## How it was made

Run from the repository root, nothing else on the rig, the client (Docker, the agent and the tasks' tests) on another
machine. Each arm is one call of [agent_flint.sh](../2026-11-05_release_regression/run/agent_flint.sh) with its own
client folder. `PREFIX` names the client's containers, so arms of another rig that share the client keep theirs.

```bash
# the machine values as in bench/agent_check/release.env.example: RIG, CLIENT, RIG_PORT, RELAY_PORT, CLIENT_PORT,
# PROXY_PORT, CLIENT_DIR, RIG_LOGDIR; the model and build on the rig
export PREFIX=acr2 MODEL=<flint-Q8_0.gguf> BIN=<llama.cpp-20261105-1551f5fe1-pascal-p106-v3-b12076-ladder5>/bin/llama-server
R=docs/data/2026-11-05_release_regression/run
CLIENT_ROOT=<client folder> OUT=agent_flint ARMS_LIST=v3base BUDGET_MIN_rel=40 bash $R/agent_flint.sh
CLIENT_ROOT=<client folder> OUT=agent_flint ARMS_LIST=reread BUDGET_MIN_rel=40 STEPS_FIRST="run report collect" \
  bash $R/agent_flint.sh
# the side-by-side, from the two client folders
python3 docs/data/2026-11-06_flint_side_rig2/run/merge_arms.py m v3base=<client folder>/out-v3base \
  reread=<client folder>/out-reread
python3 docs/data/2026-11-05_keepalive/run/agent_tables.py m v3base v3base reread   # m/agent.tsv
# requests where the re-read fix kept cached tokens, per arm
grep -c "same text:" m/rig/server-reread.log
```

**Provenance and licence:** the tables summarize runs of Unsloth's Flint Q8_0 on Miner Llama's
`pascal-p106-v3-b12076-ladder5` build at `1551f5fe1`. The texts the agent check runs on are its Exercism-based bug-fix
tasks, built by [build_bugfix.py](../../../bench/agent_check/build_bugfix.py) under their MIT licence.
