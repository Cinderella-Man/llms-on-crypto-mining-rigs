# The GPU pipe check's seed from its own name (C1022, Q1755; 2026-11-24)

What this is: a fix to [bench/pipespec/gpu_pipe_check.sh](../../../bench/pipespec/gpu_pipe_check.sh), the check that
pipelined decoding gives plain decoding's tokens on the cards, which the GPU check of every branch and merge runs
([C1009](../2026-11-24_c1009_pipe_step_cards/README.md)). The script makes random models of five architectures with
the build's own `test-llama-archs -s <seed>`, and it read that seed from the environment variable `SEED`. In the landing
of 2026-11-24 ([2026-11-24_land_v5_batch2](../2026-11-24_land_v5_batch2/README.md)) the driver had its own `SEED`, the
path of a clone, and it reached the script: every `test-llama-archs -s <path>` stopped in `stoull` and every model
step failed on both builds, so the step had to run again. Registry row Q1755 in
[quants/experiments.md](../../../quants/experiments.md).

## Result in plain words

- **A caller's own `SEED` no longer reaches the models.** With `SEED=/some/path` set by the caller, as in the
  landing, the script ran as if no seed were set: seed 1, the same ten model files as the run without it, byte for byte,
  and the same ten files the landing's own run of this build made with seed 1. Every verdict was OK.
- **The seed is now `MODEL_SEED`.** `MODEL_SEED=7` made ten other files (none equal to seed 1's), and the plain run of
  every model gave other token and logits hashes (80 of 80 identity checks). Every verdict was OK.
- **A wrong `MODEL_SEED` stops the script at once.** `MODEL_SEED=/some/path` ended with exit status 2 and a message in
  0 s, and no work folder was made.
- **Drivers that pass `SEED` with a number keep working.** `SEED=7` made `MODEL_SEED=7`'s ten files, byte for byte, and
  the log names `SEED` as the seed's source.
- **Every round that ran the checks passed them all:** 60 of 60 verdicts OK per tree (and 4 kernel lines) in each of
  the four, as in the landing.

## The change

- The seed is `MODEL_SEED` (default 1).
- `SEED` is still read, for the drivers that pass it, but only when it holds a whole number and `MODEL_SEED` is unset.
  Any other `SEED` (a path, say) is left alone: the seed stays 1, and the script's first log line says so.
- A `MODEL_SEED` that is not a whole number of at most 19 digits stops the script at once with exit status 2 and a
  message, before it writes or runs anything.
- The first log line names the seed and where it came from, and `WORK/model_seed.txt` keeps it.

The CPU twin of this step, the `pipe` step of
[cpu_tests.sh](../2026-11-05_c565_ladder4fix/run/cpu_tests.sh), does not have the gap: its seeds have names of their
own (`HAND_SEED`, `PIPE_SEED`, `PIPE_SAMPLE_SEED`), which no caller sets by accident. That it should still stop at once
on a value that is not a whole number is registry row Q1779 (C1046), planned.

## The run

On a build machine with three RTX 3090s, which run the rig build's compute 6.1 code through its PTX: every container
labelled, at a low CPU share, without network, the cards checked with a CUDA preflight before each round. Nothing was
timed but the rounds themselves.

- **The build:** Miner Llama's rig tarball of `pascal-p106-v5` at `af00bb13b` (the landing's head), given as both trees
  (`BUILDS='base=<bin> head=<bin>'`), so every round makes each tree's models twice from the same program.
- **The quantizer:** C1009's static CPU `llama-quantize` (the rig tarballs carry none), with the script's default
  `QUANT_ARGS`.
- **The step's defaults:** arms `f32-stock` and `q8-p106`, layouts `one` and `split` over the three cards, checks
  `host`, `fed`, `recurrent` and `kernels`, five architectures, 64 tokens.

| Round | The caller sets | Exit | Seconds | Seed (the script's first log line) | Model files | Verdicts per tree |
|---|---|---|---|---|---|---|
| plain | nothing | 0 | 242 | 1 (the default) | 10, the landing's seed-1 files | 60 OK, 4 INFO |
| seedpath | `SEED=/some/path` | 0 | 181 | 1 (the default; `SEED` is not a whole number, so not taken) | 10, plain's | 60 OK, 4 INFO |
| model7 | `MODEL_SEED=7` | 0 | 181 | 7 (from `MODEL_SEED`) | 10, all other than plain's | 60 OK, 4 INFO |
| badseed | `MODEL_SEED=/some/path` | 2 | 0 | stopped: "not a whole number of at most 19 digits ... nothing was run" | none | none |
| seednum | `SEED=7` | 0 | 203 | 7 (from `SEED`) | 10, model7's | 60 OK, 4 INFO |

The first four rounds ran in one call of [run/run.sh](run/run.sh) (15:57–16:08), `seednum` in a second
(`ROUNDS=seednum`, 16:08–16:12), which added its rows. The first round started with an empty PTX cache, which the later
rounds shared. Each file's md5 per round is in [models.tsv](models.tsv), every verdict in
[verdicts.tsv](verdicts.tsv).

## Files

| File | What it holds |
|---|---|
| [rounds.tsv](rounds.tsv) | per round and tree: the caller's seed variable, exit status, seconds, the script's seed line, the md5 of the model files' md5 list, the number of model files, the verdict counts |
| [models.txt](models.txt) | which rounds made the same model files |
| [models.tsv](models.tsv) | each model file's md5 per round (the `head` tree; `base` is the same build) and in the landing's run |
| [verdicts.tsv](verdicts.tsv) | every verdict of every round: round, tree, arm, layout, architecture, check, verdict, plain run's tokens and logits hashes |
| [run/run.sh](run/run.sh) | the five rounds and the tables |

## Commands

On the build machine, with a copy of `bench/pipespec/` holding the changed script (`PIPESPEC`), the rig tarball's
`bin/` (`BIN`), a static `llama-quantize` (`QUANT`, as in C1009) and the CUDA preflight:

```bash
WORK=<work dir> PIPESPEC=<copy of bench/pipespec> BIN=<rig tarball>/bin QUANT=<llama-quantize> \
  PREFLIGHT=<cuda_preflight.sh> GPUS=0,1,2 MOUNTS=<folder holding the tarball> REF=<landing work>/gpupipe/head/models.md5 \
  DOCKER_ARGS='--label <label> --cpu-shares=256' \
  setsid nohup bash run/run.sh > <work dir>/driver.log 2>&1 < /dev/null &
```

In a GPU check, a driver now passes the seed as `MODEL_SEED=<n>`:

```bash
MODEL_SEED=1 WORK=<dir> BUILDS='base=<base>/bin head=<branch>/bin' GPUS=0,1,2 QUANT=<llama-quantize> \
  bash bench/pipespec/gpu_pipe_check.sh
```

## Provenance and licence

The random models are written by `test-llama-archs` from the build under test; nothing is downloaded. The build is
Miner Llama's rig tarball of `pascal-p106-v5` at `af00bb13b` (`TESTS=1`, md5 `b51ad2740d5822e7e14af656c413ec22`).
The tables are this project's own (the repository's licence).
