# The GPU pipe check with sampling, a race, the floor and 256 tokens (C1019, Q1752; 2026-11-24)

What this is: an extension of [bench/pipespec/gpu_pipe_check.sh](../../../bench/pipespec/gpu_pipe_check.sh), the check
that pipelined decoding gives plain decoding's tokens on the cards. The GPU check of every branch and merge runs it
([C1009](../2026-11-24_c1009_pipe_step_cards/README.md)). It decoded 64 greedy tokens a run, so on the cards it missed
what [C1010](../2026-11-24_c1010_pipe_step_paths/README.md) added to the processor's twin of the check (the `pipe` step
of [cpu_tests.sh](../2026-11-05_c565_ladder4fix/run/cpu_tests.sh)): seeded sampling, a race of two samples whose winner
goes on alone, the floor, and 256 tokens. Registry row Q1752 in [quants/experiments.md](../../../quants/experiments.md).

## Result in plain words

- **The GPU pipe check now runs every arm of the processor's check,** on every model, arm and layout:
  - seeded sampling (`sampled`);
  - a race of two samples whose winner goes on alone (`race`);
  - the floor (guesses that are filler tokens) in every identity check;
  - 256 tokens a run instead of 64.

  Every race ended where it was set to, at the first token from position 64 that sample 1 had not emitted before:
  298 of 300 races at token 64 (sample 1 then went on alone for 191 tokens), two at token 65 (Flint's Q4_K copy, whose
  token at 64 had come before). Sample 1 won every race.
- **On the v5 tip every check passed.** The tip is `pascal-p106-v5` `af00bb13b`. Over nine rounds of the new script:
  1,375 verdicts OK and 55 INFO (the kernels lines), none other; the per-type arms with the new checks among them.
- **The older build with C932's Flash-Next pipe fault is now flagged in the tokens themselves.** That build is
  `8a9629942`. Its Flash-Next-type random model (`qwen4exp`) differs in all five checks of every arm, on one card and
  on three: 25 of 25 DIFFERS.
  - With greedy decoding the F32 and Q8_0 copies kept plain decoding's tokens: only the logits and the memory differed.
  - With sampling, 10 of 14 pipelined runs on the F32 model and 14 of 14 on the Q8_0 copy gave other tokens; in the
    race 10 of 12 and 12 of 12. The processor's check saw the same 10 and 10 on F32 (C1010).
  - The mixed-type copy under the Flash-Next preset's switches gave other tokens in every pipelined run, greedy too.

  The tip as head passed all 125 checks.
- **A build takes under three minutes with the new defaults: 165 and 171 s in two runs,** the second with nothing
  else on the machine. The old script took 113 and 112 s.
  - The new defaults run a build's containers side by side, each tool on two processor threads. One container at a
    time, the new checks took 382 s a build.
  - Two threads is what makes side by side pay. With each tool's own thread count (every core, spinning while it
    waits), five containers at once took 706 s: a host-sampled check took 43.5 s against 3.3 s alone, while the
    device-sampled one took 5.0 s.
  - Another session's job held all three cards during the landing-style, fault and per-type rounds, at times keeping
    two of them fully busy, so those rounds took longer: 362 and 401 s a build in the landing-style round, 372 and
    372 s in the fault round.

## The change

- **Two new checks, as the processor step has them** (each one `check_identity.sh` call per model, arm and layout):
  - `sampled`: the host check with seeded sampling (`--temp 1 --top-k 20 --top-p 0.95 --seed 42`, `SAMPLING` and
    `SAMPLE_SEED`), depths 1 and 3, 0 and 50 % wrong guesses, the four guessers, depth 0 and the floor: 14 pipelined
    runs against plain;
  - `race`: two samples (seeds 41 and 42, `RACE`) raced at depths 1 and 3 with 50 % wrong guesses and the floor: 12
    pipelined runs. The plain run decodes the two samples round robin; the pipelined runs keep one token of each in
    flight. Its stop text is the token the sampled check's plain run (the race's sample 1, seed 42) first emits at or
    after token 64 (`RACE_AT`, a quarter of the run), written as two or three letters that the random models' dummy
    tokenizer reads as that one token (`cpu_tests.sh`'s `pipe_race_tok` and `pipe_stop_text`). So the race ends part
    way, sample 1 wins and goes on alone in sequence 1 with the guessers. The winner, the race's tokens and every
    sample's tokens and logits are compared too.
- **The floor in every identity check** (`FLOOR=3`): one more pipelined run at depth 3 in which every guess is a filler
  token (`--pipe-floor 3`), in the `host`, `sampled`, `race` and `fed` checks.
- **256 tokens a run** (`N`, 64 before), which also gives the rollback check four times the rollbacks.
- **`JOBS`**: how many of a build's containers run at once (0, the default, every one). Before, the one-card containers
  ran one per card and then the split ones one at a time. The containers now start split ones first, then one-card
  ones with the cards in turn. These small models leave the cards mostly idle, so containers that share a card slow
  each other less than running them in turn costs.
- **`THREADS`** (default 2): every tool runs on that many processor threads (`-t`). With all layers on the cards the
  processor only waits and copies, and a tool's own default (one thread per core, spinning while it waits) made five
  side-by-side containers crowd the processor.
- **Timings kept:** `times.tsv` (every check's seconds) and `jobs.tsv` (every container's and each build's seconds).
- **The settings are checked first:** `N`, `SAMPLE_SEED`, `RACE`, `FLOOR` and `THREADS` must be whole numbers of at
  least 1, `RACE_AT` and `JOBS` whole numbers; any other value stops the script before it writes anything (exit status 2).
- `check_identity.sh` reads `SAMPLING`, `RACE` and `FLOOR` from its environment, so inside a container the script keeps
  its own settings of those names to itself and each check passes its own.

## The rounds

Three calls of [run/run.sh](run/run.sh) on a build machine's three RTX 3090s (`GPUS=0,1,2`), with `pascal-p106-v5`
`af00bb13b`'s rig build (the v5 tip, C932's fix landed) as `TIP` and `8a9629942`'s (C984's tarball, with C932's fault)
as `FAULT`. Round names in the tables carry the call: `t12/` the first call (the tools' own thread count), `t2/` the
second (`THREADS=2`), `q2/` the third (`THREADS=2`, nothing else on the machine).

| Round | Builds | Settings | Seconds | Verdicts | Shared with |
|---|---|---|---|---|---|
| `t12/old` | tip | the script before the change, its defaults (64 greedy tokens, no sampled or race check, no floor) | 113 | 5 INFO; 75 OK | another session's container at its start |
| `t12/serial` | tip | new defaults, one container at a time (`JOBS=1`) | 384 | 5 INFO; 125 OK | nothing at its start |
| `t12/jobs3` | tip | new defaults, three containers at a time (`JOBS=3`) | 432 | 5 INFO; 125 OK | nothing at its start |
| `t12/jobsall` | tip | new defaults, every container at once (`JOBS=0`) | 707 | 5 INFO; 125 OK | nothing at its start |
| `t2/jobsall` | tip | new defaults, every container at once (`JOBS=0`), `THREADS=2` | 167 | 5 INFO; 125 OK | another session's job on all three cards |
| `t2/jobs3` | tip | new defaults, three containers at a time (`JOBS=3`), `THREADS=2` | 297 | 5 INFO; 125 OK | another session's job on all three cards |
| `t2/defaults` | base and head (both the tip) | new defaults, as a landing runs them, `THREADS=2` | 764 | 10 INFO; 250 OK | another session's job on all three cards |
| `t2/fault` | base (fault) and head (tip) | new defaults, the faulty build as base, `THREADS=2` | 745 | 25 DIFFERS; 10 INFO; 225 OK | another session's job on all three cards |
| `t2/types` | tip | each release type's own arm (`q2_0-fn-c4` ... `iq4_xs-fn-c4`), `THREADS=2` | 476 | 10 INFO; 250 OK | another session's job on all three cards |
| `q2/jobsall` | tip | new defaults, every container at once (`JOBS=0`), `THREADS=2` | 173 | 5 INFO; 125 OK | nothing at its start |
| `q2/old` | tip | the script before the change, its defaults (64 greedy tokens, no sampled or race check, no floor; it has no `THREADS`) | 112 | 5 INFO; 75 OK | nothing at its start |

Each round's seconds cover every build in it (two for `defaults` and `fault`). Per container and build:
[jobs.tsv](jobs.tsv); per check: [times.tsv](times.tsv) and [checktime.tsv](checktime.tsv).

**Each check's mean seconds per architecture** (from [checktime.tsv](checktime.tsv); 256 tokens; 25 checks a round, 50 in `defaults`, `fault` and
`types`):

| Round | host | sampled | race | fed | recurrent |
|---|---|---|---|---|---|
| `t12/serial` | 3.3 | 3.4 | 3.5 | 3.3 | 1.2 |
| `t12/jobs3` | 15.1 | 13.7 | 12.6 | 5.0 | 1.6 |
| `t12/jobsall` | 43.5 | 45.1 | 37.4 | 5.0 | 2.3 |
| `t2/jobsall` | 5.6 | 5.5 | 5.6 | 5.1 | 1.6 |
| `t2/jobs3` | 6.1 | 7.5 | 6.6 | 7.3 | 2.0 |
| `t2/defaults` | 9.1 | 8.9 | 9.6 | 7.6 | 2.3 |
| `t2/fault` | 8.6 | 9.4 | 9.2 | 8.8 | 2.3 |
| `t2/types` | 18.0 | 18.5 | 17.6 | 17.7 | 3.5 |
| `q2/jobsall` | 5.8 | 5.8 | 5.7 | 5.3 | 1.6 |

## Files

| File | What it holds |
|---|---|
| [rounds.tsv](rounds.tsv) | per round: script (old or new), builds (tags), settings, exit status, seconds, verdict counts, how many other jobs' containers ran at its start |
| [verdicts.tsv](verdicts.tsv) | per round, tree, arm, layout and check: how many architectures gave each verdict |
| [checks.tsv](checks.tsv) | every check: round, tree, arm, layout, architecture, check, verdict, pipelined runs differing (of how many, and with other tokens), plain tokens and logits hashes |
| [race.tsv](race.tsv) | every race check: the stop token, its position in sample 1, the stop text, the plain run's winner and race tokens (both samples'), the verdict |
| [times.tsv](times.tsv) | every check's seconds |
| [jobs.tsv](jobs.tsv) | every container's seconds (the models step, each arm and layout) and each build's total |
| [checktime.tsv](checktime.tsv) | per round and check: the seconds summed over every container and architecture, and the mean |
| [run/run.sh](run/run.sh) | the rounds |
| [run/tables.py](run/tables.py) | the tables from the calls' work folders (`python3 -I run/tables.py t12=<work 1> t2=<work 2> q2=<work 3> <data folder>`) |

**The scripts that ran:**
- The first call ran `gpu_pipe_check.sh` at md5 `19dbbe0a59960c24a510e6b082cf9f8e`, which had no `THREADS` (the
  tools ran on their own thread count). Its `old` round ran the script before this change (md5
  `134f3bd6e9295706f9cf125f306d771e`, C1018's).
- The second call's `jobsall` and `jobs3` ran md5 `d34a85a6c7a8a79db297e7e2fdee7540` (`THREADS`, default 2), its
  `defaults`, `fault` and `types` md5 `be90a7a7407bce780d97056442707352` (the same code with its header text
  updated), and so did the third call.
- The committed script is `be90a7a7`. The committed `run.sh` (md5 `f82037b8e8f153f0fd5c15f95d18b079`) ran the second
  and third calls; the first call's copy lacked only the `THREADS` setting it writes into `rounds.tsv`.

## Commands

On the build machine, with a copy of `bench/pipespec/` holding the changed script (`PIPESPEC`), the script before the
change beside `check_identity.sh` and `prompts/rewrite.txt` (`OLDPIPE`), the two rig tarballs' `bin/` (`TIP`, `FAULT`),
a static `llama-quantize` that writes Q2_0 (`QUANT`, as in C1009) and the CUDA preflight:

```bash
WORK=<work 1> PIPESPEC=<copy of bench/pipespec> OLDPIPE=<old copy>/gpu_pipe_check.sh TIP=<tip tarball>/bin \
  FAULT=<fault tarball>/bin QUANT=<llama-quantize> GPUS=0,1,2 MOUNTS='<folders holding the tarballs and quantizer>' \
  DOCKER_ARGS='--label <label> --cpu-shares=256' PREFLIGHT=<cuda_preflight.sh> ROUNDS='old serial jobs3 jobsall' \
  setsid -f bash run/run.sh > <work 1>/driver.log 2>&1 < /dev/null
# each later call in a work folder of its own, the first one's jitcache/ copied in
THREADS=2 WORK=<work 2> ROUNDS='jobsall jobs3' <the same variables> setsid -f bash run/run.sh > <work 2>/driver.log 2>&1 < /dev/null
THREADS=2 WORK=<work 2> ROUNDS='defaults fault types' <the same variables> setsid -f bash run/run.sh >> <work 2>/driver.log 2>&1 < /dev/null
THREADS=2 WORK=<work 3> ROUNDS='jobsall old' <the same variables> setsid -f bash run/run.sh > <work 3>/driver.log 2>&1 < /dev/null
python3 -I run/tables.py t12=<work 1> t2=<work 2> q2=<work 3> <this folder>
```

In a GPU check the defaults need nothing new:

```bash
WORK=<dir> BUILDS='base=<base>/bin head=<branch>/bin' GPUS=0,1,2 QUANT=<llama-quantize> \
  bash bench/pipespec/gpu_pipe_check.sh
# every release type on its own with the new checks
ARMS='q2_0-fn-c4 q4_k-fn-c4 q5_k-fn-c4 q6_k-fn-c4 iq4_xs-fn-c4' <the same>
# the old cost: 64 greedy tokens, no floor, no sampled or race check
CHECKS='host fed recurrent kernels' N=64 FLOOR= <the same>
```

## Provenance and licence

The random models are written by `test-llama-archs` from the build under test and quantized by C1009's
`llama-quantize`; nothing is downloaded. The tables are this project's own (the repository's licence).
