# Agentic data: the pilots, the licence scans, the Elixir check and the scripts of run 1 (2026-10-26/27)

The pipeline in [bench/agentic_data](../../../bench/agentic_data/README.md) makes Flint's coding-agent sessions on
licence-checked code, for the answer store, Flint's MTP head and a publishable dataset. This folder holds what is
behind its first runs on the 7-GPU quant machine: the Python pilot's per-session table, the per-file licence scans of
every code base the sessions may use, the Elixir CPU check's yield per library module, and the scripts that drove the
Elixir pilot, long run 1 and the MTP heads trained on it. The write-up is
[agentic_data_pipeline.md](../../research/notes/agentic_data_pipeline.md) (section 7: the pilots and run 1;
section 8: the plan); the registry row is Q165 in [quants/experiments.md](../../../quants/experiments.md). The sessions
themselves are not here: the pilots' sessions were made with the retired first texts (`texts_v1.json`) and are not
data to publish, and run 1's sessions go out only as the dataset itself.

## Results

| Run | Sessions | Verified | Turns | Generated tokens | Notes |
|---|---|---|---|---|---|
| Python pilot (2 V100s, 72 min, 8 standard-library modules) | 22 | 18 | 262 | 234,669 | stage 1 wrote passing tests for 4 of 8 modules; 14 of 14 stage-2 sessions passed; per session in `pilot_sessions.md` |
| Elixir CPU check (no model) | – | – | – | – | 41 of 59 library modules make workspaces (`elixir_workspaces.tsv`); 12 stage-2 tasks from fixture tests of 3 modules |
| Elixir pilot (2 V100s, 81 min, 4 modules) | 7 | 4 | 138 | 134,776 | stage 1: 1 of 4 (`Version`); stage 2 on its tests: 3 of 3; 169 tool calls, 23 failed, 8 to the other profile's names |
| Run 1 (all seven GPUs, three writers on one task list, ~9 h) | 1,177 | 1,053 | 18,885 | 15.4 M | 73 Python and 19 Elixir modules; 18 sessions lost to a shared sandbox tag; settings and resume steps in `run/` |

## Files

| File | What |
|---|---|
| `pilot_sessions.md` | the Python pilot, one row per session: type, turns, generated tokens, longest turn, final sequence length, how it ended, verified, tool calls, failed calls, tool and wall seconds; totals and per-type medians below |
| `elixir_workspaces.tsv` | the Elixir check's workspace builder over all 59 eligible library modules of Elixir 1.18.4: bytes, module references and doctest lines renamed into the `Stdmods` namespace, doctests passing after the rename; or why the module was skipped (5 do not compile after the rename, 2 define no top-level module, 10 fail doctests after it, `Range`'s doctests never finish) |
| `scans/<tree>.json` | `license_scan.py` summary per code base: files, bytes, verdict counts, licence families found, folders with their own licence, the most frequent copyright holders, and the size and SHA-256 of the allow-list a workspace may draw from |
| `scans/<tree>.tsv.gz` | the same scan per file: path, bytes, SHA-256, kind, licence families, SPDX tags, copyright holders, verdict (gzip) |
| `run/cputest.sh` | the Elixir CPU check: sandbox image, workspaces for every module, the workspaces gate, stage-1 tasks, stage-2 tasks from fixture test files |
| `run/pilot-env.sh` | the Elixir pilot's settings for `run.sh` |
| `run/head-run1.env` | the settings `bench/staircase_drafter/agentic_head.sh` ran with on run 1 (two heads, three dump sets, 240 minutes, 4,096-target pieces) |
| `run/head_resume.sh` | the heads' second round: snapshot the 240-minute heads, resume both to 1,000 minutes in all, then gate, score and write the GGUF side files |
| `run/stage2_resume.sh` | run 1's stage 2 resumed after its tasks gate stopped it (219 of 687 rows from 20 gate-excluded parents on one writer): keep the rows the gate passed and run `STEPS="s2run gate export checks"` |
| `run/watch_run1.sh`, `run/wait_event.sh`, `run/wait_head.sh` | the watcher that appended run 1's events to one log, and the waiters that block until its next event or the head run's next step |

| Scan | Licence | Files | `ok` |
|---|---|---|---|
| `cpython-3.12.15-lib` (the pinned `python:3.12-slim` image's `Lib/`) | PSF | 731 | 686 |
| `llama.cpp-d77256999` (`git archive` of that commit) | MIT | 3,880 | 3,708 |
| `img-sympy-1.14.0` | BSD | 1,561 | 1,526 |
| `img-numpy-2.2.6` | BSD | 868 | 783 |
| `img-networkx-3.7` | BSD | 598 | 581 |
| `img-mpmath-1.3.0` | BSD | 87 | 86 |
| `img-fsspec-2026.9.0` | BSD | 58 | 58 |
| `img-dill-0.4.1` | BSD | 46 | 46 |
| `img-jsonschema-4.26.0` | MIT | 38 | 38 |
| `img-einops-0.8.2` | MIT | 27 | 27 |
| `img-referencing-0.37.0` | MIT | 15 | 15 |
| `img-regex-2026.9.29` | Apache-2.0 | 5 | 4 |
| `elixir-1.18.4` (the `elixir:1.18.4-otp-27` image's library sources) | Apache-2.0 | 246 | 246 |
| `otp-27` (the same image's Erlang/OTP 27.3.4.18 `.erl`/`.hrl` sources) | Apache-2.0 | 1,465 | 1,445 |

`img-*` are the site-packages of a local vLLM image. The tree paths in the summaries are the container paths the scans
ran in, with the two copied trees shown as `<trees>/…`. Email addresses in copyright lines are replaced by `[email]`;
the full lines are in each project's sources.

## How it was made

Made on 2026-10-26 and 2026-10-27 on the quant machine with this repository's `bench/` folder at the commits named
below, Flint Q8_0 (Unsloth's GGUF of [RecursiveSI/flint](https://huggingface.co/RecursiveSI/flint)) as the writer, and
the pinned images in [bench/agentic_data/run.sh](../../../bench/agentic_data/run.sh).

From the repository root, with `D=docs/data/2026-10-26_agentic_data` and
`Q51=$PWD/release/flint-v2/reproduce/data/q51/prompts.jsonl`:

```bash
# Licence scans (one per tree; --license is the tree's own licence family, see the table above)
python3 -I bench/agentic_data/license_scan.py <tree> --license MIT --out $D/scans/llama.cpp-d77256999

# Python pilot (2026-10-26): run.sh's defaults are the pilot's (GPUs 3,4, 16 slots of 32,768 tokens, the eight
# modules); stage 2 ran a second round with TOOLS=pi into s2b/
WORK=<out> MODEL=<flint-Q8_0.gguf> HF_DIR=<RecursiveSI/flint small files> Q51=$Q51 LLAMA_SERVER=<llama-server> \
  MOUNTS="-v <dir>:<dir>:ro ..." bash bench/agentic_data/run.sh
python3 -I bench/agentic_data/report.py <out>/s1/sessions <out>/s2/sessions <out>/s2b/sessions --md  # pilot_sessions.md

# Elixir CPU check (2026-10-26), then the Elixir pilot
BENCH=$PWD/bench WORK=<scratch> Q51=$Q51 bash $D/run/cputest.sh
BENCH=$PWD/bench WORK=<out> MODEL=... HF_DIR=... Q51=$Q51 LLAMA_SERVER=... MOUNTS=... \
  bash -c ". $D/run/pilot-env.sh && bash bench/agentic_data/run.sh"

# Run 1's MTP heads (2026-10-27, bench at c5230ed), then their second round (same environment)
export REPO=$PWD RUN1=<run 1 folder> WORK=<out> FEATS2=<second disk>/feats MODEL=... HF_DIR=... LLAMA=<llama.cpp tree>
bash -c ". $D/run/head-run1.env && bash \"\$HEAD_SH\""
HEAD_ENV=$D/run/head-run1.env bash $D/run/head_resume.sh start 1000    # later: ... head_resume.sh finish
```

`elixir_workspaces.tsv` is the per-module log of `workspace.py elixir --modules all` inside `cputest.sh`, turned into
a table. Run 1's three writers ran `bench/agentic_data/run.sh` at `d3a92ef` with the reviewed `texts.json`: a
llama-server writer on two V100s and two vLLM writers (NVFP4, pipeline-parallel) on two RTX 3090s and on two RTX 3080s
plus an RTX 3090, sharing one `CLAIMS` folder; `MODULES=all`, three stage-1 attempts per module, up to ten repairs and
three implementations per passing test file, 48 turns and 45 minutes per session. The three bugs it hit are fixed in
`cb9698c` (a run tag per run), `e1176b5` (`tasks.py --gate`, step `s2run`) and `c5230ed` (the dataset gate's path
check).

The scans describe third-party code under its own licences (PSF, MIT, BSD, Apache-2.0); the summaries and tables here
are this repository's (MIT). The pilot table holds counts only, no session text.
