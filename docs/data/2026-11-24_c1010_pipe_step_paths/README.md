# The CPU pipe step widened: sampling, a race, the floor, 256 tokens (C1010, Q1743; 2026-11-24)

What this is: a wider `pipe` step in the CPU test script [cpu_tests.sh](../2026-11-05_c565_ladder4fix/run/cpu_tests.sh)
(C565), which builds every fork branch and merge on a build machine's processor and runs upstream's and the fork's
tests on it. The step checks that Miner Llama's pipelined decoding (tokens guessed ahead and run through the cards as a
pipeline) gives exactly plain decoding's output. C935 added it
([2026-11-24_c935_pipelined_identity](../2026-11-24_c935_pipelined_identity/README.md)) with greedy decoding only, 64
tokens, in about 10 seconds a tree. This change runs more of each model's paths through it: a seeded sampled arm, a race
of two samples, the floor (guesses that are always filler) and four times the tokens. Registry row Q1743 in
[quants/experiments.md](../../../quants/experiments.md).

## Result in plain words

- **The wider step still catches C932's fault, and now in the tokens themselves.** On Miner Llama's `pascal-p106-v5` at
  `8a9629942`, which still carries the fault (its fix is not merged yet), the Flash-Next-type model differs in all four
  arms: greedy 14 of 14 pipelined runs, sampled 14 of 14, the race 12 of 12, and the rollback check. Greedy, the random
  model keeps plain decoding's tokens and only the logits, KV state and probe differ. Sampled, 10 of 14 pipelined runs
  pick other tokens, and in the race 10 of 12 do, with every race's sample tokens differing. The other four models pass
  every arm.
- **With C932's fix applied, everything passes:** all 200 pipelined runs (40 for each of the 5 models) and all 5
  rollback checks give plain decoding's hashes.
- **Nothing else changed:** against C935's run of the same two trees, every other verdict is the same, ctest passes 73
  of 75 test by test in all four columns, and the hand-run tests' logs are line for line the same as far as they go.
- **It costs 36.3 seconds a tree on the faulty tip and 35.8 on the fixed one**, against about 10 seconds before, and a
  whole tree takes about 3 minutes from a warm compiler cache.

## The step

For each architecture that `llama_pipe_init` accepts, plus Flint's, the step makes random models with the tree's own
`test-llama-archs` (`-s 1`, a 4,096-token vocabulary), so each commit is checked on models its own code writes. It
checks every model the generator writes: the MoE model (or the only one) as `<arch>`, a dense model beside a MoE one as
`<arch>-dense`. Then, on the processor (`-ngl 0 -fa on -c 2048`, 256 tokens of
[prompts/rewrite.txt](../../../bench/pipespec/prompts/rewrite.txt) as plain text), four arms:

| Arm | What it runs | Pipelined runs against plain |
|---|---|---|
| `identity` | [check_identity.sh](../../../bench/pipespec/check_identity.sh), greedy: depth 0; the floor (depth 3, no guesser, every guess a filler token, `--pipe-floor 3`); at depths 1 and 3 an oracle with 0 % and 50 % of its guesses wrong on purpose, and the four guessers (copy lookup, continuation counts, n-gram simple, n-gram map) | 14 |
| `sampled` | the same with seeded sampling, `--temp 1 --top-k 20 --top-p 0.95 --seed 42` | 14 |
| `race` | a race of two samples (seeds 41 and 42) until one emits the stop token, then the winner goes on alone with the guessers; depth 0, the floor, and at depths 1 and 3 the oracle at 50 % and the four guessers | 12 |
| `recurrent` | `llama-pipe-recurrent-check`, greedy: the rollbacks the slot rotation rarely makes (a token sent to the slot of the token before it, so the state moves to the stash; wrong guesses dropped; the token in flight dropped and sent again) | its four hashes |

Each `check_identity.sh` arm passes when every pipelined run's tokens, per-token logits, KV state and probe logits are
plain decoding's; the race arm also compares the winner and every sample's tokens and logits. The verdicts go into the
run's `summary.tsv` as `pipe <model> <arm>`: OK, DIFFERS (a pipelined run gives other hashes) or FAIL (a tool stopped
before comparing), and `pipe/times.tsv` holds each check's seconds.

**The race's stop token.** The random models' tokenizer prints every token as its id in hexadecimal, so the usual stop
text `</think>` never appears and a race would run both samples to the token limit (511 race tokens), leaving the
winner nothing to do alone. So the stop is a token the sampled arm's plain run (seed 42, which is the race's sample 1)
first emits at or after token 64 (`PIPE_RACE_AT`, a quarter of the tokens). The step turns that token id into a stop
text of two or three letters and digits by the tokenizer's own rule (it cuts text into 5-byte chunks and hashes each,
`h = 31 h + byte`, modulo the vocabulary). In this run every race ended at sample 1's token 64 (130 race tokens),
sample 1 won in every job, and the winner went on alone in sequence 1 for 191 tokens; the winner's 256 tokens are the
sampled arm's plain tokens bit for bit, in every model.

| Model | md5 | Why it is here |
|---|---|---|
| `qwen4exp` | `3c1f8f2cc6bd` (`qwen4exp-moe.gguf`) | Qwen3.8-Flash-Next's (Swift); delta-net with the n-gram conv history; C932's fault |
| `qwen35` | `ab091bd4f8e9` (`qwen35-dense.gguf`) | Qwen3.5's; delta-net, dense |
| `qwen35moe` | `30b2bfd6a845` (`qwen35moe-moe.gguf`) | Qwen3.5/3.6 MoE; delta-net |
| `qwen3next` | `758bb4d68e8e` (`qwen3next-moe.gguf`) | Qwen3-Next's; delta-net |
| `flint` | `8a91d5448c65` (`flint-moe.gguf`) | Flint's; no recurrent state, so the attention-only path |

None of the five writes a dense model beside its MoE one at this commit: `test-llama-archs` lists qwen4exp and flint as
MoE-only (`moe_mandatory`), qwen35 is dense only, and qwen35moe and qwen3next MoE only. The models are C935's (same
md5). The settings are `PIPE_N` (256), `PIPE_ARMS` (`identity sampled race recurrent`), `PIPE_SAMPLING`,
`PIPE_SAMPLE_SEED` (42), `PIPE_FLOOR` (3; empty leaves the floor out), `PIPE_RACE` (2), `PIPE_RACE_AT` (`PIPE_N`/4),
`PIPE_SEED` (1) and `PIPE_TIMEOUT` (900 s per tool); the step still runs whenever the run's steps name `pipe` or `hand`.

## The run

On a build machine's processor (Ryzen 9 5900X), in containers without network, no GPU; the builds under the machine's
build lock with `-j 8` at a low CPU share, from a compiler cache copied from C935's folder (642 of 660 compiles hit).
One CPU test run, 09:16:19–09:22:18, of two trees:

- **tip**: `pascal-p106-v5` at `8a9629942`, as it is (the driver checked that C932's fix is not in it);
- **fix**: the same commit with C932's fix `80d83fa1a` (fork branch `pascal-p106-v5-pipe-logits`) applied on top, its
  logits checked against the tip's (`@tip`).

The driver lists its steps as the routine drivers do (`tree build ctest archs ident hand`), without `pipe`: the hand
step brought it. **Before** is C935's run of the same two trees by the script with the narrow step, with the same seeds.

### The pipe step

| Model | tip: identity | tip: sampled | tip: race | tip: recurrent | fix: every arm |
|---|---|---|---|---|---|
| `qwen4exp` | **DIFFERS, 14 of 14** (tokens the same; logits, state, probe differ) | **DIFFERS, 14 of 14** (10 with other tokens) | **DIFFERS, 12 of 12** (10 with other tokens; every race's samples differ) | **DIFFERS** (logits from the 2nd token) | OK: 0 of 14, 0 of 14, 0 of 12; recurrent OK |
| `qwen35`, `qwen35moe`, `qwen3next`, `flint` | OK, 0 of 14 | OK, 0 of 14 | OK, 0 of 12 | OK | OK |

The rollback check at 256 tokens sends 85 tokens to the slot before, drops 64 wrong guesses and sends 64 tokens again
through the pipe and 21 through plain decoding, in every tree and model: four times C935's 21, 16, 16 and 5 at 64
tokens. Sampled text from random weights rarely repeats, so the guessers guess little in the sampled and race arms and
the oracle and the floor make most of the rollbacks there ([identity_jobs.tsv](identity_jobs.tsv)); in the fixed tree
the identity arms made 17,560 guesses (11,040 hits, 2,596 misses), the sampled arms 10,711 (3,924 hits, 2,748 misses)
and the race arms 6,197 (984 hits, 2,126 misses), over the five models; a miss also drops the guesses behind it, which
the table counts as wasted tokens.

| Seconds a tree ([pipe_times.tsv](pipe_times.tsv)) | gen | identity | sampled | race | recurrent | total |
|---|---|---|---|---|---|---|
| tip | 2.5 | 9.9 | 10.4 | 10.7 | 2.3 | 36.3 |
| fix | 2.5 | 9.8 | 10.0 | 10.7 | 2.2 | 35.8 |

The total includes each check's container start; `qwen4exp`, whose random model is twice the others' size, takes about
a third of each arm's time.

### Every other check

| Check | Before tip | tip | Before fix | fix |
|---|---|---|---|---|
| build | OK, 0 error lines | OK | OK | OK |
| ctest, 75 tests | 73 pass | 73 | 73 | 73: every test the same in all four columns; `test-kv-cache` and `test-kv-cache-scales` fail as before |
| `test-llama-archs -b CPU -s 1`: plain, writing, comparing; 148 files | OK | OK | OK | OK |
| logits against the base (ident) | — | — | 148 of 148 the same (against the tip) | 148 of 148 the same (against the tip) |
| the nine hand-run tests | 9 pass | 9 | 9 | 9 |
| hand logs, line by line | — | 22 of 24 the same as Before, 2 cut short, 0 different | — | 22 of 24 the same as Before, 2 cut short, 0 different |

The 2 cut short in each comparison are eval-callback's log and its sums, the same as far as they go: the program exits
without flushing llama.cpp's logger (Q1740), so their length changes from run to run.

## Files

| File | What |
|---|---|
| [pipe.tsv](pipe.tsv) | the pipe step's 40 verdicts (2 trees × 5 models × 4 arms) with their details |
| [identity_jobs.tsv](identity_jobs.tsv) | every `check_identity.sh` job of both trees (430): guesses, right and wrong guesses, wasted tokens, and the six hashes against plain (tokens, logits, state, probe, winner, race) |
| [race.tsv](race.tsv) | every race job's winner, race tokens and tokens generated |
| [pipe_times.tsv](pipe_times.tsv) | each check's seconds, per tree, model and arm, and the step's total |
| [models.tsv](models.tsv) | the step's five random models and their md5 in both trees |
| [checks.tsv](checks.tsv) | every other verdict of the script in Before's tip, tip, Before's fix and fix |
| [ctest.tsv](ctest.tsv) | every ctest test's result in the same four columns |
| [hand.tsv](hand.tsv) | the hand logs compared line by line (`compare_hand.py`): Before's tip against tip, Before's fix against fix, tip against fix |
| [run/run.sh](run/run.sh) | the driver: the clone and the fork's two branches checked, the compiler cache, `cpu_tests.sh` on the two trees, then the comparisons |
| [run/tables.py](run/tables.py) | the tables of this folder from the driver's work folder |
| [../2026-11-05_c565_ladder4fix/run/cpu_tests.sh](../2026-11-05_c565_ladder4fix/run/cpu_tests.sh) | the CPU test script with the wider `pipe` step |

## Commands

```bash
# on the build machine; SEED a llama.cpp clone; REPO a folder with cpu_tests.sh, compare_hand.py, run.sh and pipespec/
# (bench/pipespec's check_identity.sh and prompts/rewrite.txt); BEFORE an earlier run's work folder with tip/, fix/ and
# summary.tsv of the same commits
W=<work dir> SEED=<llama.cpp clone> REPO=<scripts> CPU_CCACHE=<an earlier WORK>/ccache BEFORE=<work folder> \
  LOCK=<build lock file> JOBS=8 THREADS=6 bash run.sh
# the tables, from the work folder copied back without builds, models and logits
python3 -I tables.py <W> <this folder>
# the step alone on any tree (it also runs with STEPS naming hand)
WORK=<dir> SRC=<llama.cpp clone> TREES='<tag>=<commit>' STEPS='tree build pipe' bash cpu_tests.sh
```

`run.sh` runs `cpu_tests.sh` as `TREES='tip=8a9629942a4635ea0b2711b72e59556b96a63f86
fix=8a9629942a4635ea0b2711b72e59556b96a63f86+80d83fa1a0e0ca18d20970b1e023ae8e6ea91e45@tip' STEPS='tree build ctest archs
ident hand' CTEST_JOBS=6 THREADS=6`, with labels and `--memory=16g` in `DOCKER_ARGS`; the image is `cpu_tests.sh`'s own
(Ubuntu 24.04 with the compiler, CMake, ccache, Python 3 and Jinja2).

Provenance and licence: every table here was made by these scripts from their own run; llama.cpp and its tests are
MIT-licensed.
