# Flash-Next's v3 release preset on the release build, and stock llama.cpp beside it (T1, 2026-11-05 and 2026-11-06)

What this is: release test T1 of Miner Llama v3. Flash-Next (Swift-1.5 GSQ-RCO Q2_0) runs on nine P106-090 cards on
the frozen release build `pascal-p106-v3-b12076-ladder5`, which has every v3 feature behind its switch. The test has
three parts: a greedy identity check against the previous cumulative build, a llama-bench diagnostic, and the agent
check ([bench/agent_check](../../../bench/agent_check/README.md), 12 bugfix tasks an arm) of three switch groups. The
preset also runs on Swift's own Q2_0 file (`t1swift`), so the engine's gain and the file's gain show apart. Stock
llama.cpp at the release build's upstream base runs a tuning pass and the agent check, on Swift's own Q2_0 file. The
preset this chose is [release/flash-next-v3/settings.env](../../../release/flash-next-v3/settings.env).

Every speed below is one measured run, with its file, build and settings named.

## The builds

| Build | Commit | Used for |
|---|---|---|
| Miner Llama `-ladder5` | `1f7215948` | identity, the llama-bench run, arm `t1base` |
| Miner Llama `-ladder5` with the MMVQ row-guard fix | `1551f5fe1` | identity, arms `t1skip`, `t1neg` and `t1swift` |
| Miner Llama `-ladder4` | `e3b30303d` | the identity reference |
| Miner Llama `-keepalive` | `f6c1c6a4a` | the reference runs of [2026-11-05_keepalive](../2026-11-05_keepalive/README.md) |
| stock llama.cpp | `20a51f24a` | arm `stockq2`, the tuning pass. This is `-ladder5`'s upstream merge base (tag `b12076` plus 16 commits), built with `scripts/build_llamacpp_docker.sh 20a51f24a`, no patch |

## Identity: the same greedy output on every build

Each build ran one llama-server with every switch at its default, pmix-q2-fast on nine cards (`-ts 6,5,5,6,5,5,6,5,6`,
`-c 8192`, `-lm dio`). Two prompts ran at temperature 0, 128 tokens each, and the md5 of each reply is compared
([ident.tsv](ident.tsv)).

| Build | code prompt | prose prompt |
|---|---|---|
| `-ladder4` `e3b30303d` | `08d7258fe07b` | `dba941b4967b` |
| `-ladder5` `1f7215948` | `08d7258fe07b` | `dba941b4967b` |
| `-ladder5` `1551f5fe1` | `08d7258fe07b` | `dba941b4967b` |

## The agent check

S35's line ([2026-11-04_s35_stack](../2026-11-04_s35_stack/README.md)) is the base of every Miner Llama arm:
- pmix-q2-fast, with the k-column settings `GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=32768 GGML_CUDA_MMVQ_PASCAL_KCOL_TABLE=0`;
- the Q8_0 hop cast through the staged hop, `LLAMA_HOP_CAST=q8_0 GGML_CUDA_HOP_STAGED=1`;
- the MTP draft with its head's shortlist, `LLAMA_DRAFT_SHORTLIST=32768,8192,2`;
- the server line `-b 2048 -ub 512 -ts 6,6,6,6,5,6,5,6,3 -fit off -lm dio -md <draft> --spec-type draft-mtp
  --spec-draft-n-max 3 -otd token_embd=CUDA4 --spec-draft-n-ubatch 128 --no-host`, with a 65,536-token context.

The arms:

| Arm | Build | On top of S35's line |
|---|---|---|
| `t1base` | `1f7215948` | `GGML_CUDA_KEEPALIVE=decode` (the preset candidate) |
| `t1skip` | `1551f5fe1` | `t1base` + the expert skip, `LLAMA_MOE_TAIL=p=0.9,renorm=1,hist=0` |
| `t1neg` | `1551f5fe1` | `t1base` + every "negligible gain" that applies to this line, together. That is the expert skip, the draft length per step (`LLAMA_SPEC_ADAPT=1` with the real-text T(k) of the `-kshape` kernels at these k-column settings, [t_k_kshape_s32kt0_real.txt](../2026-11-04_kcurve_p106/cost/t_k_kshape_s32kt0_real.txt)), one server thread (`-t 1 -tb 1`), and the keep-alive at each request (`GGML_CUDA_KEEPALIVE=request` instead of `decode`) |
| `t1swift` | `1551f5fe1` | `t1base`'s line and setting exactly, on Swift's own Q2_0 file |
| `stockq2` | stock `20a51f24a` | the same server line on Swift's own Q2_0 file, with the MTP draft and no switch |

Each arm has a 40-minute budget and a 12-minute cap a task. All arms ran on one rig, with the client on another
machine: `t1base`, `t1skip`, `t1neg` and `stockq2` back to back on 2026-11-05, `t1swift` on 2026-11-06. The equal-work column times one fixed amount of work for every arm: `t1base`'s prompt tokens at the
arm's prompt speed, plus `t1base`'s generated tokens at the arm's generation speed, plus `t1base`'s time outside the
server, divided over `t1base`'s tasks ([agent_tables.py](../2026-11-05_keepalive/run/agent_tables.py)).

All 12 tasks ([agent_t1.tsv](agent_t1.tsv)):

| Arm | Passed | Gen tok/s | Prompt tok/s | TTFT median, p90 s | Acceptance | Equal work, min a task |
|---|---|---|---|---|---|---|
| `t1base` | 12/12 | 20.75 | 69.0 | 7.60, 15.79 | 0.714 | 2.45 |
| `t1skip` | 11/12 | 21.29 | 71.9 | 6.37, 16.47 | 0.720 | 2.38 (−2.9 %) |
| `t1neg` | 12/12 | 19.05 | 70.2 | 7.14, 15.73 | 0.817 | 2.58 (+5.3 %) |
| `t1swift` (Swift's own Q2_0 file) | 12/12 | 20.40 | 69.4 | 6.89, 15.89 | 0.718 | 2.47 (+0.8 %) |
| `s35ka` (`-keepalive` `f6c1c6a4a`, the same line and setting as `t1base`) | 12/12 | 20.75 | 77.4 | 6.41, 16.86 | 0.708 | 2.36 (−3.7 %) |

`stockq2` reached 9 tasks before its budget ran out. On those 9 tasks ([agent_t1_9.tsv](agent_t1_9.tsv)):

| Arm | Passed | Gen tok/s | Prompt tok/s | TTFT median, p90 s | Acceptance | Equal work, min a task |
|---|---|---|---|---|---|---|
| `t1base` | 9/9 | 20.93 | 68.1 | 6.96, 15.72 | 0.714 | 1.87 |
| `t1skip` | 8/9 | 21.29 | 70.7 | 6.28, 15.71 | 0.720 | 1.82 (−2.7 %) |
| `t1neg` | 9/9 | 18.82 | 69.8 | 6.89, 15.73 | 0.817 | 1.97 (+5.3 %) |
| `t1swift` | 9/9 | 20.25 | 68.2 | 6.63, 15.89 | 0.718 | 1.90 (+1.6 %) |
| `stockq2` | 9/9 | 8.20 | 53.7 | 10.22, 24.85 | 0.717 | 3.74 (+100.0 %) |

The report's own summary of the arms (minutes, turns, cache hits, runaways) is [report_table.md](report_table.md).
Some tasks ran into the 12-minute cap ("runaway"): `t1skip` one, which it failed (`bugfix-python-scale-generator`), and
`t1neg` one, which it passed. In `t1neg` the draft length varies per step, so 1 + 3 × acceptance does not give its
tokens a step.

**Classes, against `t1base`:**
- **The expert skip (`t1skip`): negligible.** It is −2.9 % a task at equal work. `s35ka`, the same line and setting on
  the `-keepalive` build, is −3.7 % from `t1base`, so the skip's difference is within the difference between two runs
  of one line. It failed one task as a runaway. The skip stays behind its switch, out of the preset.
- **The negligible-gain group together (`t1neg`): a loss.** Generation is 19.05 against 20.75 tok/s and +5.3 % a task
  at equal work. Which member costs is the next question; C534's method answers it with leave-one-out arms, cheapest
  first by T(k).
- **The release preset is `t1base`'s line:** [settings.env](../../../release/flash-next-v3/settings.env).
- **pmix-q2-fast against Swift's own Q2_0 file, on the same build and preset (`t1swift`): negligible.** Swift's file
  takes +0.8 % a task at equal work (2.47 against 2.45 minutes), within the 3.7 % between two runs of one line. Its
  generation is 20.40 against 20.75 tok/s, its prompt reading 69.4 against 69.0. Miner Llama v3 gets its speed from
  the engine on either file.

## Stock llama.cpp against Miner Llama v3

| Engine and file | Arm | Gen tok/s | TTFT median s | Equal work, min a task (the 9 tasks) | Passed |
|---|---|---|---|---|---|
| stock llama.cpp `20a51f24a` + Swift's Q2_0 | `stockq2` | 8.20 | 10.22 | 3.74 | 9/9 (3 not reached) |
| Miner Llama v3 + Swift's Q2_0 | `t1swift` | 20.40 | 6.89 | 1.90 | 12/12 |
| Miner Llama v3 + pmix-q2-fast | `t1base` | 20.75 | 7.60 | 1.87 | 12/12 |

On the 9 tasks stock reached, the engine alone halves a task's time: Miner Llama v3 on Swift's own file does a task's
work in 1.90 minutes against stock's 3.74 on the same file, and generates at 20.40 against 8.20 tok/s. The file adds
little on top: pmix-q2-fast takes 1.87 minutes on the same build and preset.

**Stock's line:** stock `20a51f24a` ran the same server line as Miner Llama, without the fork's environment switches.
That line is the one tuned for Miner Llama, and stock has every option in it: `-lm dio`, the n-gram table read on
demand (`-lzm auto`), `-otd`, `--spec-type draft-mtp`, `--spec-draft-n-max`, `--spec-draft-n-ubatch`, `--no-host` and
`-fit`. Swift's Q2_0 file has no MTP layer of its own (no `nextn` tensors in its shards), so the separate MTP draft
file is stock's only way to draft, and stock runs it: acceptance 0.717, the same as Miner Llama's.

### Stock's tuning pass

One llama-bench load of stock `20a51f24a` on Swift's own Q2_0 file, flash attention on, no switch,
`-ngl 99 -fa 1 -b 2048 -ts 6/5/5/6/5/5/6/5/6 -lm dio -ub 128,256,512 -p 4,2048 -n 128 -r 3` (2026-11-06,
[tune.tsv](tune.tsv)). tok/s, the mean of three repetitions, then each repetition:

| `-ub` | pp4 | pp2048 | tg128 |
|---|---|---|---|
| 128 | 12.69 (12.93, 12.56, 12.58) | 132.43 (131.84, 133.05, 132.40) | 6.37 (6.44, 6.34, 6.34) |
| 256 | 12.55 (12.56, 12.52, 12.57) | **153.44** (152.76, 154.43, 153.12) | 6.37 (6.44, 6.34, 6.34) |
| 512 | 12.54 (12.51, 12.55, 12.55) | 138.93 (139.51, 140.03, 137.26) | 7.16 (7.40, 7.40, 6.67) |

- **`-ub 256` reads a 2,048-token prompt 10.4 % faster than `stockq2`'s `-ub 512`**: 153.44 against 138.93 tok/s,
  with every repetition apart. Stock's tuned line is `stockq2`'s with `-ub 256`.
- **The 4-token pass** (one token and three drafted, the draft's verification) is the same at every `-ub`.
- **tg128** decodes one token a pass, so `-ub` does not enter it. At `-ub 512`, the load's last test, two of its three
  repetitions ran at 7.40 tok/s. The agent arms measure generation over whole sessions.

## llama-bench on S35's no-draft line: a diagnostic

pmix-q2-fast with `GGML_CUDA_KEEPALIVE=decode` and S35's k-column and hop settings, `-ngl 99 -fa 1 -b 2048 -ub 512 -ts
6/5/5/6/5/5/6/5/6 -lm dio -p 1,2,3,4,5 -n 128 -r 5`. The table gives the median ms a pass of k tokens, and for tg128
the mean ms a token:

| Build, run | k = 1 | 2 | 3 | 4 | 5 | tg128 |
|---|---|---|---|---|---|---|
| `-ladder5` `1f7215948` | 64.6 | 82.4 | 94.2 | 105.2 | 123.5 | 52.2 |
| `-keepalive` `f6c1c6a4a`, run 1 | 63.9 | 80.7 | 94.7 | 100.1 | 117.0 | 51.1 |
| `-keepalive` `f6c1c6a4a`, run 2 | 61.8 | 77.7 | 89.4 | 100.0 | 117.1 | 51.1 |

The `-keepalive` runs are from [2026-11-05_keepalive](../2026-11-05_keepalive/README.md). `-ladder5` had one run, so
alternated runs of both builds decide whether its 4- and 5-token passes are slower. On the agent check, `t1base` on
`-ladder5` and `s35ka` on `-keepalive` both generate at 20.75 tok/s.

## Files

| File | What |
|---|---|
| [agent_t1.tsv](agent_t1.tsv) | the agent check per arm, all 12 tasks, against `t1base` |
| [agent_t1_9.tsv](agent_t1_9.tsv) | the same over the 9 tasks `stockq2` reached |
| [report_table.md](report_table.md) | the agent check's own report table (minutes, turns, cache hits, runaways) |
| [ident.tsv](ident.tsv) | the identity replies' md5 and speed per build (the second `1f7215948` load is the reference of the `1551f5fe1` check) |
| [tune.tsv](tune.tsv) | stock's tuning pass: tok/s per `-ub` and test, mean, standard deviation and each repetition |
| [run/rig_t1.sh](run/rig_t1.sh) | the run: identity, llama-bench, the agent arms, stock's arms, `t1swift`, stock's tuning pass |

## Commands

From the repository root, on a machine that only runs ssh. `ENVFILE` is S12's machine file
([s12.env.example](../2026-11-03_flash_next_mtp/run/s12.env.example)) filled in, with `DRAFT`, the client and its
folders. `BUILD` is the build's folder under `/opt/llama-builds` on the rig.

```bash
# identity against -ladder4, the llama-bench run, the client's prep, the three arms and the report
ENVFILE=<machine file> MODEL=<pmix-q2-fast shard 1> BUILD=<-ladder5> IDENT_REF=<-ladder4> COST=<t_k_kshape_s32kt0_real.txt on the rig> \
  OUT=<out> STEPS="ident tk prep arms report" bash docs/data/2026-11-05_release_flashnext/run/rig_t1.sh
# stock on Swift's own Q2_0 file
ENVFILE=<machine file> MODEL=<Swift Q2_0 shard 1> BUILD=<stock 20a51f24a> OUT=<out> STEPS=stock STOCK_ARMS=stockq2 \
  bash docs/data/2026-11-05_release_flashnext/run/rig_t1.sh
# the release preset on Swift's own Q2_0 file; stock's tuning pass
ENVFILE=<machine file> MODEL=<Swift Q2_0 shard 1> BUILD=<-ladder5> OUT=<out> STEPS=swift bash docs/data/2026-11-05_release_flashnext/run/rig_t1.sh
ENVFILE=<machine file> MODEL=<Swift Q2_0 shard 1> BUILD=<stock 20a51f24a> OUT=<out> STEPS=tune bash docs/data/2026-11-05_release_flashnext/run/rig_t1.sh
# tune.tsv from the tuning pass's llama-bench output (<out>/tune.jsonl)
python3 -I -c 'import json,sys
print("build\tn_ubatch\ttest\tavg_tok_s\tstddev_tok_s\tsamples_tok_s")
for l in open(sys.argv[1]):
    r = json.loads(l); t = "pp%d" % r["n_prompt"] if r["n_prompt"] else "tg%d" % r["n_gen"]
    print(r["build_commit"], r["n_ubatch"], t, "%.2f" % r["avg_ts"], "%.2f" % r["stddev_ts"], ",".join("%.2f" % x for x in r["samples_ts"]), sep="\t")' <out>/tune.jsonl > tune.tsv
# the tables, from the agent check's collected folder (any machine with Python 3)
OUT_TAG=_t1 python3 -I docs/data/2026-11-05_keepalive/run/agent_tables.py <client results folder> t1base t1skip t1neg stockq2 s35ka t1swift
SKIP_TASKS=bugfix-javascript-complex-numbers,bugfix-python-pig-latin,bugfix-javascript-variable-length-quantity OUT_TAG=_t1_9 \
  python3 -I docs/data/2026-11-05_keepalive/run/agent_tables.py <client results folder> t1base t1skip t1neg stockq2 s35ka t1swift
```

Provenance: measured on one Manli 9 × P106-090 rig (4 GB host), with the agent check's client on another machine. The
tasks are bench/agent_check's bugfix set. The tables are this repository's measurements, under its licence.
