# Pipelined decoding in every CPU test run (C935, Q1673; 2026-11-24)

What this is: a new step in the CPU test script [cpu_tests.sh](../2026-11-05_c565_ladder4fix/run/cpu_tests.sh) (C565),
which builds every fork branch and merge on a build machine's processor and runs upstream's and the fork's tests on it.
The new `pipe` step checks that Miner Llama's pipelined decoding (several tokens guessed ahead, run through the cards
as a pipeline) gives exactly plain decoding's output. Before it, nothing in the routine checks ran pipelined decoding:
ctest never does, and the tools that do (`llama-pipespec`, `llama-pipe-recurrent-check`) were only compiled. So a fault
that a merge brought in on 2026-11-05 went unseen for 16 days, until C932 found it by chance
([2026-11-21_c932_pipe_logits](../2026-11-21_c932_pipe_logits/README.md)): every pipe slot of a Flash-Next-type model
read a stale copy of its conv history. Registry row Q1673 in [quants/experiments.md](../../../quants/experiments.md).

## Result in plain words

- **Every CPU test run now decodes pipelined, and it catches C932's fault.** On Miner Llama's `pascal-p106-v5` at
  `8a9629942`, which still carries the fault (its fix is not merged yet), the step reports the Flash-Next-type model as
  different: all 13 pipelined runs give other logits, KV state and probe hashes than plain decoding, and the rollback
  check too. The other four models pass, as C932 found by hand.
- **With C932's fix applied, every model passes:** all 65 pipelined runs (13 for each of the 5 models) and all 5
  rollback checks give plain decoding's hashes.
- **Nothing else changed:** ctest passes 73 of 75 in all three columns, test by test the same; every other verdict is
  the earlier run's; the hand-run tests' logs are line for line the earlier run's; and the fix leaves the 148
  `test-llama-archs` logits files bit for bit as they were (plain decoding keeps state group 0, so its graph is
  unchanged).
- **It costs about 10 seconds a tree** (five random models, each made, checked with 14 jobs and with the rollback
  check), against about 2 minutes 30 seconds for the whole tree from a warm compiler cache.

## The step

For each architecture that `llama_pipe_init` accepts, plus Flint's, the step makes a random model with the tree's own
`test-llama-archs` (`-s 1`, a 4,096-token vocabulary, the MoE model where the architecture has one), so each commit is
checked on a model its own code writes. Then, on the processor (`-ngl 0 -fa on -c 2048`, greedy, 64 tokens of
[prompts/rewrite.txt](../../../bench/pipespec/prompts/rewrite.txt) as plain text):

| Check | What it runs | Passes when |
|---|---|---|
| `identity` | [bench/pipespec/check_identity.sh](../../../bench/pipespec/check_identity.sh): plain decoding, the pipelined API at depth 0, and at depths 1 and 3 an oracle with 0 % and 50 % of its guesses wrong on purpose plus the four guessers (copy lookup, continuation counts, n-gram simple, n-gram map) | every pipelined run's tokens, per-token logits hashes, KV state and probe logits are plain decoding's |
| `recurrent` | `llama-pipe-recurrent-check`: the rollbacks the slot rotation rarely makes (a token sent to the slot of the token before it, so the state moves to the stash; wrong guesses dropped; the token in flight dropped and sent again through another slot or plain decoding) | its exit status is 0: the same four hashes |

| Architecture | Model | Why it is here |
|---|---|---|
| `qwen4exp` | `qwen4exp-moe.gguf` | Qwen3.8-Flash-Next's (Swift); delta-net with the n-gram conv history; C932's fault |
| `qwen35` | `qwen35-dense.gguf` | Qwen3.5's; delta-net |
| `qwen35moe` | `qwen35moe-moe.gguf` | Qwen3.5/3.6 MoE; delta-net |
| `qwen3next` | `qwen3next-moe.gguf` | Qwen3-Next's; delta-net |
| `flint` | `flint-moe.gguf` | Flint's; no recurrent state, so the attention-only path |

The step's verdicts go into the run's `summary.tsv` as `pipe <arch> identity` and `pipe <arch> recurrent`: OK (every
hash plain's), DIFFERS (a pipelined run gives other hashes) or FAIL (a tool stopped before comparing). **It runs
whenever the run's steps name `pipe` or `hand`**, so every driver that already lists its steps (they all list `hand`)
runs it without a change; `PIPE_ARCHS=''` leaves it out. It needs `check_identity.sh` and `prompts/rewrite.txt`: in a
`pipespec/` folder beside `cpu_tests.sh` (the layout of C932's `check_at.sh`), from the repository it sits in, or
wherever `PIPESPEC` points. Both are copied into the tree's `pipe/` folder, so the record holds the script that ran.
`PIPE_N` (64), `PIPE_SEED` (1) and `PIPE_TIMEOUT` (900 s per tool) set the rest.

## The run

On a build machine's processor (Ryzen 9 5900X), in containers without network, no GPU; the builds under the machine's
build lock with `-j 8` at a low CPU share, from a compiler cache copied from C933's folder (642 of 660 compiles hit).
One CPU test run, 06:28:25–06:33:29, of two trees:

- **tip**: `pascal-p106-v5` at `8a9629942`, as it is;
- **fix**: the same commit with C932's fix `80d83fa1a` (fork branch `pascal-p106-v5-pipe-logits`) applied on top, its
  logits checked against the tip's (`@tip`).

The driver lists its steps as the routine drivers do (`tree build ctest archs ident hand`), without `pipe`: the hand
step brought it. **Before** is C933's second run of the same commit by the script before this step
([2026-11-24_c933_hand_tests_seed](../2026-11-24_c933_hand_tests_seed/README.md)), with the same seeds.

### The pipe step

| Model | tip: identity | tip: recurrent | fix: identity | fix: recurrent |
|---|---|---|---|---|
| `qwen4exp` (md5 `3c1f8f2cc6bd`) | **DIFFERS: 13 of 13** pipelined runs | **DIFFERS**: tokens the same, logits from the 2nd token, state and probe differ | OK, 0 of 13 | OK |
| `qwen35` (`ab091bd4f8e9`) | OK, 0 of 13 | OK | OK, 0 of 13 | OK |
| `qwen35moe` (`30b2bfd6a845`) | OK, 0 of 13 | OK | OK, 0 of 13 | OK |
| `qwen3next` (`758bb4d68e8e`) | OK, 0 of 13 | OK | OK, 0 of 13 | OK |
| `flint` (`8a91d5448c65`) | OK, 0 of 13 | OK | OK, 0 of 13 | OK |

On the tip, every pipelined `qwen4exp` run gives plain decoding's tokens but other logits, KV state and probe hashes,
from depth 0 on (a token is never guessed wrong there: the stale history alone changes the logits). The random weights
keep the greedy tokens the same; on the real Flash-Next file the same fault garbled the replies within four words
(C932). The two qwen4exp and flint models are the ones C932 used at `602691153` (same md5), and each tree made the same
five models ([models.tsv](models.tsv)). The rollback check ran 64 tokens with 21 tokens sent to the slot before, 16
wrong guesses, 16 tokens dropped and sent through the pipe again and 5 through plain decoding, in every tree and model.

### Every other check

| Check | Before | tip | fix |
|---|---|---|---|
| build | OK, 0 error lines | OK | OK |
| ctest, 75 tests | 73 pass | 73 | 73: every test the same in all three columns; `test-kv-cache` and `test-kv-cache-scales` fail as before |
| `test-llama-archs -b CPU -s 1`: plain, writing, comparing; 148 files | OK | OK | OK |
| logits against the base (ident) | 148 of 148 the same (against `3dfad876f`) | — | 148 of 148 the same (against the tip) |
| the nine hand-run tests | 9 pass | 9 | 9 |
| hand logs, line by line | — | 24 of 24 the same as before | 22 of 24 the same as the tip, 2 cut short, 0 different |

The 2 cut short are eval-callback's log (870 of 895 lines) and its sums (95 of 99), the same as far as they go: the
program exits without flushing llama.cpp's logger (Q1740, found by C933).

## Files

| File | What |
|---|---|
| [pipe.tsv](pipe.tsv) | the pipe step's 20 verdicts (2 trees × 5 models × 2 checks) with their details |
| [identity_jobs.tsv](identity_jobs.tsv) | every `check_identity.sh` job of both trees (140): guesses, hits, misses, wasted tokens, and the four hashes against plain |
| [models.tsv](models.tsv) | the step's five random models and their md5 in both trees |
| [checks.tsv](checks.tsv) | every other verdict of the script in Before, tip and fix |
| [ctest.tsv](ctest.tsv) | every ctest test's result in Before, tip and fix |
| [hand.tsv](hand.tsv) | the hand logs compared line by line (`compare_hand.py`): Before against tip, tip against fix |
| [run/run.sh](run/run.sh) | the driver: the clone and the fork's two branches checked, the compiler cache, `cpu_tests.sh` on the two trees, then the comparisons |
| [run/tables.py](run/tables.py) | the tables of this folder from the driver's work folder |
| [../2026-11-05_c565_ladder4fix/run/cpu_tests.sh](../2026-11-05_c565_ladder4fix/run/cpu_tests.sh) | the CPU test script with the `pipe` step |

## Commands

```bash
# on the build machine; SEED a llama.cpp clone holding the tip, REPO a folder with cpu_tests.sh, compare_hand.py, run.sh
# and pipespec/ (bench/pipespec's check_identity.sh and prompts/rewrite.txt); BEFORE an earlier run's tree folder of the
# same commit ($WORK/<tag>)
W=<work dir> SEED=<llama.cpp clone> REPO=<scripts> CPU_CCACHE=<an earlier WORK>/ccache BEFORE=<tree dir> \
  LOCK=<build lock file> JOBS=8 bash run.sh
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
