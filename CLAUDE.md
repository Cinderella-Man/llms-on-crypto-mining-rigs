# llms-on-crypto-mining-rigs

Running LLMs on old crypto-mining hardware. The point of this repo is that anyone who owns the same rig finds clear,
self-explanatory information on what they can do with it and how: thousands of these machines exist. Write for a
technically literate reader with no prior context, and put all technical work here (scripts, benchmarks, research,
build recipes) with machine-identifying details removed. The current rig is a Manli M-P106L9-N6G with nine
NVIDIA P106-090 6 GB cards; its hardware spec is in `docs/mining_rig.md`. Read it before
reasoning about performance: every GPU sits on a PCIe 1.0 x1 link, there is no GPU peer-to-peer, each
card has 5 SMs, and the host has 2 Celeron cores and 4 GB of RAM.

## Working on the rig

- The rig is reached through the SSH alias `rig`, non-interactively: `ssh rig '<command>'`. Each
  contributor defines that alias for their own machine, so the real hostname never enters this repo.
- Build CUDA code with CUDA 12.x for `sm_61`. CUDA 12.x is the last toolkit line and the NVIDIA 580
  driver branch the last driver line that support Pascal.
- Remove anything you leave on the rig, such as test binaries in `/tmp`.

## Measuring speed

- The speed this project optimizes and reports is realistic agentic speed: a coding agent working through a task
  (multi-turn, tool calls and their results, a growing context, prompt caching), measured with `bench/agent_check` as
  time per task, turns, time to first token per turn and tokens per second over the session.
- Single-request numbers (chat, code or edit tok/s, llama-bench tg128 and pp512) are diagnostics for finding causes.
  Release settings are chosen on agentic speed, and the README reports agentic speed first.

## Quant work

- Plan and status of the quants subproject live at the top of `quants/experiments.md` ("Plan", with a "Now" line). Read
  it before starting anything; when you start or stop an experiment, update the "Now" line and the experiment's row.

## The data behind every result

Nobody should have to rerun an experiment because its data or method was not published. An experiment is done when,
in the same change as its result:
- `docs/data/<date>_<experiment>/` holds a `README.md` (what it measured, a link to the result page, a table of the
  files, the exact commands, provenance and licence), the curated tables (CSV, TSV, Markdown) and the scripts that ran
  it in `run/`; reusable tools go to `scripts/` or `bench/<tool>/` instead;
- the result page (benchmark page, registry row or research note) links the folder, and
  [docs/data/README.md](docs/data/README.md) has its row.

Scripts take hosts and paths from the environment (`RIG`, `CLIENT`, `WORK`, `MODELS`, `MODEL`, `BIN`, `OUT`), so the
script that ran is the script committed. A data folder holds tables that summarize raw output, benchmark task IDs with
the command that rebuilds the task file (not the task text), data whose licence allows publishing, and files under
1 MB (gzip up to ~3 MB). Pages link only to this repository and public sources.

## This repo is public

- Never commit hostnames, tailnet or domain names, usernames, IP or MAC addresses, GPU UUIDs or serial
  numbers, whether in docs, scripts, logs or commit messages.
- Keep raw command output out of the repo; summarize what it shows instead.
- When documenting hardware, prefer measurements taken on the rig over spec sheets and GPU databases.
- Write docs as finished specs: state the correct facts plainly. Don't add review commentary,
  verification markers, or notes about what an earlier version or another source claimed.
