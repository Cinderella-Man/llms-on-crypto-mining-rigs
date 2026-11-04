# Q559 and Q607: a RAM watchdog that sees a host out of memory, and a smoke request the size of the agent's first turn (2026-11-04)

Two fixes to the tools that guard the 4 GB rigs while llama-server loads and runs. Nothing here ran on a rig: both were
written and checked on a 7-GPU quant machine (no GPU used), and tomorrow's rig runs are their first. The registry rows are
Q559 and Q607 in [quants/experiments.md](../../../quants/experiments.md); the cause behind both is
[the host-RAM note](../../research/notes/host_ram_4gb.md) §3.1: 410–970 MiB of the 3.8 GiB are CMA-marked pageblocks
(Kexec HandOver scratch) that `MemAvailable` counts and that pinned buffers and the kernel's own allocations cannot use.

## Q559: the watchdog

[scripts/ram_watchdog.sh](../../../scripts/ram_watchdog.sh) samples once a second with bash builtins only (no fork per
sample) and kills the guarded llama-server (SIGKILL) when one limit holds. The same text is embedded, byte for byte, in
[bench/agent_check/rig_server.sh](../../../bench/agent_check/rig_server.sh) (the agent check's server launcher, read from stdin
over ssh, so it cannot call a sibling file) and in [scripts/server_bench.sh](../../../scripts/server_bench.sh);
[scripts/ram_watchdog_test.sh](../../../scripts/ram_watchdog_test.sh) checks that the copies are unchanged.

| What | Default | Where it comes from |
|---|---|---|
| **MemAvailable minus CmaFree** below `MIN_USABLE_MB` for `USABLE_FLOOR_S` samples | 20 MiB, 2 | Q226's host read −18 MiB (949 − 967) for 8 minutes while `MemAvailable` said 949; the lowest reading of a run that finished is 35 MiB |
| **memory pressure**, `/proc/pressure/memory` full avg10 above `PSI_FULL_PCT` for `PSI_FULL_S` samples | 40 %, 10 | the arm the kernel killed (Q558 step 3) reached 59.1 % with 14 s above 40 %, under the old 60 % for 15 s; runs that finished reached 28.3 % |
| the same for **some** avg10, `PSI_SOME_PCT` and `PSI_SOME_S` | 40 %, 10 | that arm 65.2 % and 16 s above 40 %; finished runs 32.5 % |
| **swap-out**: more than `KILL_SWAPOUT_MIB` written to swap within `WINDOW_S` samples (`pswpout` in `/proc/vmstat`) | 600 MiB, 10 | Q558's guard: the load that stopped it wrote 654, Q226's 704; finished runs 557 at most |
| `MemAvailable` alone below `MIN_AVAIL_MB` (the agent check's old floor) | 0 in the script; `rig_server.sh` and `server_bench.sh` pass 400, the agent check's callers 150 | unchanged behaviour of the callers |

Every row's numbers, with the runs they come from, are in [thresholds.tsv](thresholds.tsv) and [history.tsv](history.tsv)
(sixteen guard logs of Q226/Q558, Q561, Q218 and Q551, summarized by [run/summarize.sh](run/summarize.sh)). A limit set to
`off` is disabled. The script also:
- gives the guarded process `oom_score_adj` 1000 (`OOM_SCORE_ADJ`) and reads it back, and gives itself −1000, so the kernel's
  OOM killer takes the server before the desktop or the watchdog;
- runs at a higher priority than the server: `rig_server.sh` and `server_bench.sh` start it with `chrt -f 10` (`nice -n -20`
  without `chrt`) against the server's `SCHED_OTHER`; its first events record the policy of both;
- keeps its script and its log on tmpfs (`/dev/shm`), so that neither waits on a disk; `rig_server.sh` copies the log to
  `<LOGDIR>/ram_watchdog-<arm>.tsv` when the server stops, `server_bench.sh` to `<output folder>/ram_watchdog.tsv`;
- writes every sample to that TSV, one row a second, tab-separated, the first nine columns those of S15's `mem_guard.sh`
  (`epoch avail_mib cma_free_mib usable_mib swap_used_mib swapout_window_mib psi_some_avg10 psi_full_avg10 pid`; the header
  line and the event lines start with `#`, so read it with `comment='#'`). Events: the limits, the process found with its
  `oom_score_adj` and scheduling, `KILL` with the reason, `stopped`;
- after a kill, appends the reason and the numbers (`MemAvailable`, `CmaFree`, `Shmem`, `SwapFree`, the pressure lines) to the
  server's log behind a line starting `watchdog:`, as the agent check's logs have always had;
- is started by `rig_server.sh start` and `server_bench.sh`, which refuse to leave a server unguarded: when the watchdog's
  first sample row does not appear within seconds, the server is stopped and the start fails.

Limits reach the watchdog as environment variables of the launcher. On the ssh command line of `rig_server.sh` they go in
`WD_ENV` (`release_check.sh`, `smoke_dflash.sh` and `smoke_mtp.sh` pass it on): `WD_ENV="PSI_FULL_PCT=30 MIN_USABLE_MB=40"`.

**What the defaults would have done on the recorded runs** ([history.tsv](history.tsv)): none of the sixteen logs has a second
below 20 MiB of usable memory, so the floor stays quiet on every run that finished and is not what stops Q558 step 3 (its last
sample before the kernel's OOM kill, 23.3 minutes after the server started, read 131 MiB). The pressure limits fire on that
arm alone, **6.8 minutes after the server started, 16.5 minutes before the kernel killed it**; the swap-out limit stops Q558
step 1 (654 MiB) as before; no run that finished reaches any limit. The floor covers the state Q226 was in, where
`MemAvailable` is above 900 MiB and nothing is usable.

## Q607: the first-turn request in the draft smoke tests

The agent check's first request (the first bugfix task, Q561) was 2,776 tokens, and `-b 2048` crossed the watchdog's floor on
it; the smoke tests' coding request is 79 tokens. [bench/agent_check/first_turn_request.py](../../../bench/agent_check/first_turn_request.py)
builds a request of that size as a chat completion body: a short system message, the four tools of the agent check's pi
(`read`, `bash`, `edit`, `write`, each with a JSON schema: a request with tools takes the server's tool-call path, backend
sampling off), and the task as the user message: the first line of `tasks/bugfix/order.txt`, its `prompt.md` whole, then its
workspace files (the stub, then the test file) cut at a line boundary to the size. The sampling is the agent check's
(temperature 1.0, top-p 0.95, top-k 20, min-p 0) and the reply is capped at 256 tokens (the agent's first reply is ~80).
[first_turn_request.sh](../../../bench/agent_check/first_turn_request.sh) runs it on a build machine over ssh (`BUILD_SSH`, or
`ssh $CLIENT`), after rebuilding the task folder there with the README's command for the bugfix set,
`python3 build_bugfix.py --out <folder>/tasks/bugfix`, in a temporary folder that is removed afterwards. **No task text is
committed**: the request exists in the smoke test's output folder only.

- **Sizing.** The model's own template and vocabulary, [run/count_tokens.py](run/count_tokens.py), count the request: the
  content weighs 3.6 characters per token and Swift Flash-Next's template adds 250 tokens (a reasoning-effort paragraph and its
  tool format), so a body built for 2,780 renders to 2,760 tokens ([token_count.tsv](token_count.tsv)). The server's own count
  comes back in the reply (`usage.prompt_tokens`) and the smoke scripts print it beside the target. `LONG_TOKENS`, `LONG_CPT`
  and `LONG_TEMPLATE_TOKENS` tune it, for another model's template or if the count is off.
- **Not pi's own text.** The real first turn also carries pi's system prompt and tool descriptions, which are not in this
  repository; the request has the same shape and size with a system message and tool schemas written for it.
- **The smoke scripts** [smoke_dflash.sh](../2026-11-03_flash_next_dflash2/run/smoke_dflash.sh) and
  [smoke_mtp.sh](../2026-11-03_flash_next_mtp/run/smoke_mtp.sh) build it before they touch the rig (a failure stops them
  there), send it **first** after the load, as the agent check's first request is, save the reply as `reply_long.json` and the
  host as `host_after_request_long.txt`, then send their short requests as before. `LONG_REQUEST=off` skips it; `DRY_RUN=1`
  builds it and prints what would be sent. They now exit non-zero when a reply is empty (the watchdog killed the server) and
  report the watchdog's samples and kills. `smoke_dflash.sh` no longer starts S15's `mem_guard.sh` (the launcher's
  watchdog does its job); `GUARD` is not read.

## Checks (a quant machine, no GPU, nothing of the rigs)

[run/validate.sh](run/validate.sh) runs all of this from a machine that only runs ssh. On 2026-11-04:
- **Self-test, 64 of 64** ([selftest.txt](selftest.txt)): 35 checks on synthetic `/proc` files and dummy processes (each limit
  fires with the right reason, a brief excursion and a 39 % reading do not, `off` disables a limit, the TSV has nine columns
  and `usable = avail − cma`, `oom_score_adj` 1000 and −1000, bad input is refused, the scan mode kills only a process of its
  name and keeps running); a **60-second run on the machine's own `/proc`** with every limit off (10 checks: 60 rows, epochs a second
  apart, `MemAvailable`, `CmaFree`, swap and the pressure columns agree with `/proc`; [sample_real_60s.tsv](sample_real_60s.tsv)
  is that log, from a 127 GB machine, which shows the format and not a rig's numbers), and a few seconds under `chrt -f 10`
  that record `SCHED_FIFO rt 10` for the watchdog and `SCHED_OTHER` for the guarded process; the embedded copies equal the
  script; `rig_server.sh` start, a kill by its embedded watchdog and end with a stand-in server (the TSV lands in the log folder
  with the `KILL` event, `/dev/shm` is clean, the start fails when the watchdog cannot start); `server_bench.sh` with the stand-in.
- **Syntax** of every script touched ([syntax.txt](syntax.txt)).
- **The first-turn request**: built through `first_turn_request.sh` ([request_summary.txt](request_summary.txt)) and counted
  ([token_count.tsv](token_count.tsv)); both smoke scripts' dry runs build the same request and stop before the rig
  ([smoke_dry.txt](smoke_dry.txt)).

**Not verified.** The watchdog has not run on a rig: its `chrt -f 10` on the Celeron, `/dev/shm` there and bash 5.2 (the
quant machine has 5.3) are first used tomorrow, and `rig_server.sh` refuses to start unguarded if any of them fails. The
request's size is the template's count, not the server's: `usage.prompt_tokens` in `reply_long.json` is the figure to read (a
few tokens off at most; more means `LONG_CPT` or `LONG_TEMPLATE_TOKENS` is off for that model). The defaults come from
sixteen logs of draft and no-draft runs on the two kernel builds the rigs run (7.0.0-38 and 7.0.0-42), both with the CMA-marked pageblocks; after `kho=off` (S32) `CmaFree` is 0,
the floor equals `MemAvailable`, and the first runs on that kernel are the data to revisit the floor and the pressure limits.

## Commands for the next rig runs

All from the repository root, on the machine that only runs ssh; `CHECK_SSH` and `BUILD_SSH` are ssh commands of a Linux
machine with python3, bash, awk and curl (the agent check's client or the quant machine), never a rig. The values of the S18
environment file (`docs/data/2026-11-03_flash_next_dflash2/run/s18.env.example`) fill in `RIG`, `BENCH`, `RIG_PORT`, `REL_BIN`,
`MODEL`, `DFLASH`, `CTX` and `ARGS_dflash2*`.

```bash
# 1. before the rig (3 minutes, no rig): the self-test, syntax, the request and both dry runs. Expect "64 passed, 0 failed".
CHECK_SSH="ssh <build machine>" bash docs/data/2026-11-04_ram_watchdog/run/validate.sh out/ram_watchdog_check

# 2. a draft smoke test with the first-turn request, the rig quiet (S18's line at -b 2048, two drafted tokens)
set -a; . s18.env; set +a; BUILD_SSH="ssh <build machine>"
D=docs/data/2026-11-03_flash_next_dflash2
R="$D/run/smoke_request.json $D/run/smoke_greedy.json"
DRY_RUN=1 ARGS="$ARGS_dflash2" SMOKE_LOGDIR=$RIG_LOGDIR/smoke REQUESTS="$R" bash $D/run/smoke_dflash.sh out/smoke-dry
ARGS="$ARGS_dflash2" SMOKE_LOGDIR=$RIG_LOGDIR/smoke REQUESTS="$R" bash $D/run/smoke_dflash.sh out/smoke-b2048
#    the MTP draft (smoke_mtp.sh reads ARGS_mtp): ARGS_mtp="$ARGS_mtp2" SMOKE_LOGDIR=$RIG_LOGDIR/smoke-mtp \
#      REQUEST=docs/data/2026-11-03_flash_next_mtp/run/smoke_request.json bash docs/data/2026-11-03_flash_next_mtp/run/smoke_mtp.sh out/smoke-mtp

# 3. a timed arm needs no new command: release_check_s12.sh starts rig_server.sh, which starts the watchdog
#    (WD_ENV="PSI_FULL_PCT=30" adds a limit change; MIN_AVAIL_MB=150 stays the agent check's value)
STEPS=run ARMS=dflash2b bash docs/data/2026-11-03_flash_next_mtp/run/release_check_s12.sh out/agent-dflash2b

# 4. what the watchdog saw, where the files are (awk only): one row per log; the kill, if any, is a '# … KILL' line in it
tar -cf - docs/data/2026-11-04_ram_watchdog/run/summarize.sh -C out/smoke-b2048 ram_watchdog-smoke.tsv \
  | ssh <build machine> 'd=$(mktemp -d); tar -xf - -C "$d"; bash "$d"/docs/data/2026-11-04_ram_watchdog/run/summarize.sh \
    smoke="$d"/ram_watchdog-smoke.tsv; rm -rf "$d"'
grep -h '^watchdog:' out/smoke-b2048/server-smoke.log     # the reason and the numbers, if it stopped the server
grep -h 'request long done' out/smoke-b2048/smoke.log     # the server's prompt tokens against the target 2780
```

A stop by the watchdog ends the smoke test with `request long done (0 bytes)` and a non-zero exit status; read the `KILL`
event for which limit fired, then the rows before it. Do not loosen a limit to get a run through: a run that needs it
is the finding. `rig_vgraphs.sh` (C151r) and the agent-check wrappers of Q551 and Q218 still start S15's `mem_guard.sh` next to the launcher's
watchdog; two guards do no harm (the older one kills at 600 MiB or 60 % only) and add a second log.

## Files

| File | What |
|---|---|
| `thresholds.tsv` | each limit: default, rule, what runs that finished and runs that failed reached, source |
| `history.tsv` | sixteen guard logs of earlier runs (Q226/Q558, Q561, Q218, Q551): samples, lowest usable memory, the longest seconds below 20/48/100 MiB, the most swapped out in 10 s, the highest and longest pressure, kills, and how the run ended |
| `sample_real_60s.tsv` | one minute of the watchdog's TSV on a machine's own `/proc` (a 127 GB quant machine, limits off): the format |
| `selftest.txt`, `syntax.txt` | the self-test's 64 checks and the syntax checks of every script touched |
| `request_summary.txt`, `token_count.tsv`, `smoke_dry.txt` | the first-turn request's size (estimated; counted with the model's template), and the two smoke scripts' dry runs |
| `run/validate.sh` | all the checks above from a machine that only runs ssh |
| `run/summarize.sh` | the per-log summary row (`history.tsv`'s columns); runs on the watchdog's TSV and on `mem_guard.sh`'s logs |
| `run/count_tokens.py` | renders a request with a GGUF's chat template and counts it with llama-tokenize (CPU, vocabulary only) |
| `../../../scripts/ram_watchdog.sh`, `../../../scripts/ram_watchdog_test.sh` | the watchdog and its self-test |
| `../../../bench/agent_check/first_turn_request.{py,sh}` | the first-turn request and the script that builds it on a build machine |

## Provenance

The limits come from the guard logs of the runs listed in `history.tsv` (the rigs' own logs, summarized here; the raw logs stay
with the run folders), and from Q226's kernel OOM report. The tokenizer check used the vocabulary and chat template of
[ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
(its header only; no weights). The tasks are Exercism's, under the MIT licence
([bench/agent_check/tasks/bugfix](../../../bench/agent_check/tasks/bugfix/LICENSE.exercism-python)); none of their text is
stored here.
