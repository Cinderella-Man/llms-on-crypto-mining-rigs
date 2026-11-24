# A streams arm in the CPU pipe step (C1020, Q1753; 2026-11-24)

What this is: a fifth arm, `streams`, in the `pipe` step of the CPU test script
[cpu_tests.sh](../2026-11-05_c565_ladder4fix/run/cpu_tests.sh) (C565), which builds every fork branch and merge on a
build machine's processor and runs upstream's and the fork's tests on it. The step checks that Miner Llama's pipelined
decoding (tokens guessed ahead and run as a pipeline) gives exactly plain decoding's output. Its four arms so far
([C1010](../2026-11-24_c1010_pipe_step_paths/README.md)) decode one conversation at a time. The pipe's streams mode,
where several conversations share one context and their prompt chunks interleave with each other's tokens, is what a
server with two slots runs, and only its own check [check_streams.sh](../../../bench/pipespec/check_streams.sh) tested
it, run by hand ([C880](../2026-11-21_c880_batch_api/README.md)). The new arm runs that check in every CPU test run.
Registry row Q1753 in [quants/experiments.md](../../../quants/experiments.md).

## Result in plain words

- **On the landing's head (`pascal-p106-v5` at `af00bb13b`) every verdict is OK:** all 25 (5 models × 5 arms), and in
  the streams arm all 14 conversation runs of every model match their plain runs.
- **The streams arm catches C932's fault too.** On `8a9629942`, the commit before the fix landed, the Flash-Next-type
  model (`qwen4exp`) differs in all 14 conversation runs: logits, KV state and probe differ in each, while the 64
  sampled tokens stay the same. The other four models pass. The four older arms flag the same model, as at the landing.
- **The step stays under a minute a tree:** 42.4 s on `8a9629942` and 43.1 s on `af00bb13b`, of which the streams arm
  is 5.4 s and 5.6 s (about one second a model).
- **Nothing else moved:** the five random models have the landing's md5s, and every hash of the four older arms (every
  job of `identity`, `sampled` and `race`, and the rollback check's line) is the landing's run of the same tree.
- **A gap it leaves:** in its two jobs with guesses, every guess was wrong (0 accepted, 46 dropped a job, over the five
  models), because sampled text from random weights rarely repeats. So the arm checks dropped guesses in streams mode
  but never an accepted one; a greedy streams job is the new row C1090 (Q1823).

## The arm

`check_streams.sh` loads the model once with two KV streams (`-np 2`, 1,024 tokens of context each) and runs, in one
process:

| Job | What it does |
|---|---|
| `ref-c0`, `ref-c1` | plain decoding of each conversation alone in its own sequence, its prompt in chunks of 16 tokens: the reference |
| `all` | both conversations at once, no guesses |
| `arrive` | conversation 1 joins after 16 tokens of conversation 0, so its prompt chunks interleave with conversation 0's tokens |
| `slack` | the same arrivals with guesses (copy lookup of 4 or more, else continuation counts, depth 2) in a shared budget of 4 items in flight |
| `nobudget` | the same guesser with a budget of one item: no guess may go out |
| `multi` | both at once with up to 4 prompt chunks in flight |
| `multi-arrive` | arrivals with up to 4 prompt chunks in flight |
| `multi-gen` | arrivals, 4 chunks in flight while no conversation generates and 1 while one does, guesses in a budget of 6 |

Conversation 0's prompt is [prompts/chat.txt](../../../bench/pipespec/prompts/chat.txt) (2 chunks) and conversation 1's
[prompts/rewrite.txt](../../../bench/pipespec/prompts/rewrite.txt) (36 chunks), so the long prompt's chunks go through
while the first conversation generates. Both sample with the step's seeded settings (`--temp 1 --top-k 20 --top-p 0.95`,
conversation c with seed 42 + c), 64 tokens each. Each of the 14 conversation runs (7 jobs × 2) must give its plain
run's tokens, per-token logits, KV state and probe logits, and `nobudget` must make no guess. The verdict
`pipe <model> streams` is OK when all hold, DIFFERS when a run gives other hashes, and FAIL when a job's result is
missing, the tool stopped, or a guess went out with a budget of one item.

Settings: `PIPE_STREAMS` (2 conversations), `PIPE_STREAMS_N` (64 tokens), `PIPE_STREAMS_CHUNK` (16),
`PIPE_STREAMS_ARRIVE` (16), `PIPE_STREAMS_CTX` (1,024 a conversation); it shares `PIPE_SAMPLING`, `PIPE_SAMPLE_SEED`
and `PIPE_TIMEOUT` with the other arms, and `PIPE_ARMS` now defaults to `identity sampled race recurrent streams`. The
step copies `check_streams.sh` and `chat.txt` into the tree's `pipe/` folder with the other scripts.

## The run

On a build machine's processor (Ryzen 9 5900X), in containers without network, no GPU, at a low CPU share; one run,
21:18:30–21:20:20, of two trees, each building only the step's three tools (`llama-pipespec`,
`llama-pipe-recurrent-check`, `test-llama-archs`) with the landing's CMake options (`-DGGML_RPC=ON`) from the landing's
compiler cache (276 of 300 compiles hit), then the pipe step with all five arms:

- **base**: `pascal-p106-v5` at `8a9629942`, the landing's base, which still has C932's fault (Flash-Next's fused conv
  window writing every pipe slot's history to state group 0);
- **land**: the landing's head `af00bb13b`, which has the fix.

**Before** is the v5 landing's CPU test run of the same two trees with the four-arm step
([2026-11-24_land_v5_batch2](../2026-11-24_land_v5_batch2/README.md)).

| Model | base: streams | base: the four older arms | land: every arm |
|---|---|---|---|
| `qwen4exp` | **DIFFERS, 14 of 14** (tokens 0, logits 14, state 14, probe 14) | **DIFFERS** in all four, as Before | OK; streams 0 of 14 |
| `qwen35`, `qwen35moe`, `qwen3next`, `flint` | OK, 0 of 14 | OK, as Before | OK; streams 0 of 14 |

In the faulty tree the 64 sampled tokens of every conversation stay plain's (the `sampled` arm, at 256 tokens, picks
other tokens in 10 of 14 runs: C1010), so the streams arm sees this fault in the logits, state and probe hashes.

| Streams jobs over the five models ([streams.tsv](streams.tsv)) | guesses | accepted | dropped |
|---|---|---|---|
| `slack`, either tree | 92 | 0 | 46 |
| `multi-gen`, either tree | 93 | 0 | 46 |
| `nobudget` | 0 | 0 | 0 |

| Seconds a tree ([pipe_times.tsv](pipe_times.tsv)) | gen | identity | sampled | race | recurrent | streams | total |
|---|---|---|---|---|---|---|---|
| base | 2.5 | 10.2 | 10.5 | 10.9 | 2.3 | 5.4 | 42.4 |
| land | 2.6 | 10.3 | 10.5 | 11.3 | 2.3 | 5.6 | 43.1 |

The streams arm takes 0.9 to 1.6 s a model (`qwen4exp`, whose random model is twice the others' size, the longest),
each including its container's start.

## Files

| File | What |
|---|---|
| [pipe.tsv](pipe.tsv) | the 50 pipe verdicts (2 trees × 5 models × 5 arms) with their details, and Before's verdict for the four older arms |
| [streams.tsv](streams.tsv) | every streams conversation run (2 trees × 5 models × 14): tokens generated, prompt chunks, guesses, accepted and dropped guesses, and the four hashes against its plain run |
| [arms.tsv](arms.tsv) | the four older arms' jobs whose hashes equal Before's run of the same tree, model and arm |
| [pipe_times.tsv](pipe_times.tsv) | each check's seconds, per tree, model and arm, and the step's total |
| [models.tsv](models.tsv) | the step's five random models and their md5 in both trees (the landing's) |
| [run/run.sh](run/run.sh) | the driver: the clone and both commits checked, the compiler cache, `cpu_tests.sh` on the two trees, then the tables of this folder |
| [../2026-11-05_c565_ladder4fix/run/cpu_tests.sh](../2026-11-05_c565_ladder4fix/run/cpu_tests.sh) | the CPU test script with the `streams` arm |

## Commands

```bash
# on the build machine; SEED a llama.cpp clone holding both commits; REPO a folder with cpu_tests.sh, run.sh and
# pipespec/ (bench/pipespec's check_identity.sh and check_streams.sh, prompts/rewrite.txt and prompts/chat.txt);
# BEFORE the landing's cpu_tests.sh work folder (base/, land/, summary.tsv); CPU_CCACHE its compiler cache
W=<work dir> SEED=<llama.cpp clone> REPO=<scripts> BEFORE=<landing work folder> CPU_CCACHE=<ccache dir> \
    LOCK=<build lock file> JOBS=8 THREADS=6 bash run/run.sh
# the step alone on any tree, every arm (as run.sh calls it):
WORK=<dir> SRC=<llama.cpp clone> TREES='base=8a9629942:llama-pipespec,llama-pipe-recurrent-check,test-llama-archs' \
    STEPS='tree build pipe' CMAKE_ARGS=-DGGML_RPC=ON bash ../2026-11-05_c565_ladder4fix/run/cpu_tests.sh
```

Every container carries the run's labels, `--cpu-shares=256`, `--memory=16g` and no network. Provenance: the random
models are written by each tree's own `test-llama-archs` (`-s 1`, a 4,096-token vocabulary); the prompts are this
repository's. Licence: the repository's.
