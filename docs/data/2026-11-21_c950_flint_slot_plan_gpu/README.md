# Flint's pipe slots planned once, on three newer cards (C950, Q1687, 2026-11-21)

What this is: the GPU half of the check of [C901](../2026-11-21_c901_pipe_slot_plan/README.md) (registry
[Q1629](../../../quants/experiments.md)), point 4 of the GPU checks of the night's branches (registry
[Q1687](../../../quants/experiments.md)). With pipelined speculation (`--pipe-depth 4` in Flint's preset) every token of a
plain-text reply is computed in one of 10 pipe slots, and each slot has a scheduler of its own. Split over several
cards, those schedulers re-plan their memory: once after a change of device, and then each time the context outgrows
the plan of the moment. C901's branch (`pascal-p106-v5-pipe-slot-plan` `19048bb6d`, one commit on `pascal-p106-v5`
`3dfad876f`) counts those re-plans in the server's log, and its switch `LLAMA_PIPE_SLOT_PLAN=1` plans a slot that
re-plans once more over the whole memory, so that a growing context fits. Its processor check could split a model only
over two network devices; here Flint Pascal Q8 GGUF runs on a build machine's three RTX 3090s, through the rig build's
compute 6.1 PTX (the same binaries a P106 rig runs). Correctness and counts only: nothing here is timed.

## Result

**In plain words:** it passes. Every reply whose tokens are fixed (greedy, or a tool call under a JSON schema) is the
same with the switch unset, with it on, and on the base build, in both rounds of each load. Without the switch the pipe
slots re-planned 200 times in a load's two rounds: 110 as the context grew and 90 after a change of device. With it they
re-planned 90 times, all after a change of device, each followed by a plan over the whole memory that held the step's
graph (90 of 90), and never again as the context grew. The new count line in the server's log gives those numbers, and
the trace's own re-plan lines agree with it request by request. One request sampled at temperature 0.7 with a fixed seed
gave one of two replies; the base build gave both, one in each round, so that is not the switch (registry Q1692).

| Pass criterion (Q1687 point 4) | Result |
|---|---|
| The same tokens, switch unset and on | yes: 12 of 14 requests one token list in all 6 runs (3 loads × 2 rounds), every greedy and every schema request; the sampled R14 (and R15, which continues its conversation) one of two lists in every build, the base included |
| The `pipe reallocs` line counts re-plans after a change of card, fewer or none with the switch | yes: unset 110 / 90 (same devices / changed devices), 0 plans over the whole memory; `=1` 0 / 90, 90 plans over the whole memory, every one "the step's graph fits it" |

**The setup.** Flint Pascal Q8 GGUF with its v5 preset ([release/flint-v5/settings.env](../../../release/flint-v5/settings.env),
`--pipe-depth 4` kept), the layer split changed to the three cards (`-ts 1,1,1`; the log says pipeline parallelism is
on), the trace on (`LLAMA_UBATCH_TRACE=1`: each scheduler re-plan as a `sched-replan` line). Three loads, one at a time:

| Arm | Build | `LLAMA_PIPE_SLOT_PLAN` |
|---|---|---|
| sp0 | the branch, `19048bb6d` | unset |
| sp1 | the branch, `19048bb6d` | `1` (its load line says so) |
| base | the merged v5 engine, `3dfad876f` | not in this build (it has no `pipe reallocs` line; the trace shows its re-plans) |

Each load ran the session twice ([run/session.py](run/session.py)): C949's tool-call session
([2026-11-21_c949_flint_persize_pp_gpu](../2026-11-21_c949_flint_persize_pp_gpu/README.md)) with four plain-text turns
added after its R8, so that the pipe slots meet a context that grows by several 256-cell steps. 14 requests a round:
eight tool calls under a JSON schema (R2 to R7, R10, R11), where the pipe stays off, and six plain-text replies (R1, R8,
R12 to R15), which the pipe slots compute. All greedy but R3 (a tool call) and R14 (plain text), at temperature 0.7,
top-k 8, seed 7. R1 and R12 to R15 ask the server to ignore the end token (a logit-bias sampler), R8 not.

**Tokens** ([tokens.tsv](tokens.tsv)): 84 replies. 12 of the 14 requests give one token list across the six runs. R14
gives two lists of 256 tokens that share their first 23 (the reply names a different file from there on); sp0 gave one
in both rounds, sp1 the other in both, and the base one in each round. R15 continues R14's conversation, so its two lists
pair with R14's. The pipe's own counts are the same in all six runs on every greedy plain request (R1: 395 guesses, 267
right, 32 wrong; R8 194, 70, 31; R12 521, 325, 49; R13 427, 367, 15); on R14 they follow its list (281, 241, 10 with one,
284, 243, 11 with the other). Why a sampled reply under pipelined speculation differs between loads on these cards, from
one token on, is registry Q1692.

**The pipe slots' re-plans** ([replans.tsv](replans.tsv), [causes.tsv](causes.tsv), [arms.tsv](arms.tsv)):

| Arm | Re-plans, same devices | after a change of device | Plans over the whole memory | `sched-replan` lines in the trace (all of them the pipe slots') |
|---|---:|---:|---:|---:|
| sp0 | 110 | 90 | 0 | 200 |
| sp1 | 0 | 90 | 90, all held | 90 |
| base | (no count line) | (no count line) | — | 200, request by request sp0's |

Per round and request (each a multiple of 10, one per slot), sp0 and the base: round 1 R1 10 + 10, R8 0 + 10, R12
20 + 10, R13 10 + 0, R14 10 + 10, R15 10 + 10 (same devices + changed devices); round 2 the same except R1, 0 + 0. sp1:
0 + 10 on R1 (round 1), R8, R12, R14 and R15 (both rounds), 0 on R13 and on round 2's R1.

- **The 110 on the same devices** are all the attention mask (`attn_inp_kq_mask`, in host memory) outgrowing the plan of
  the moment as a reply's context crosses a 256-cell step. Each re-plan after a change of device plans for the current
  context only, so a slot grows again from there: R12 comes after one and crosses two steps (20). With the switch none
  of them happen.
- **The 90 after a change of device** come where a request's sampler chain differs from the one the slot ran last: the
  logit-bias sampler comes (R1, R12) or goes (R8), the random input of the sampler at temperature 0.7 comes (R14) or goes
  (R15). The step's graph then has other nodes at a position than the plan (the trace names the first: `logit_bias`,
  `uniform_0`, the last `norm`), so the slot plans again, with or without the switch. Round 2's R1 follows R15 with the
  same chain, so nothing re-plans there. With the switch each of the 90 is followed by one plan over the whole memory for
  that chain ([plans.tsv](plans.tsv): three graphs, 3,593, 3,597 and 3,599 nodes), and every plan line ends "the step's
  graph fits it"; no slot gave the switch up.
- The context's own schedulers re-planned 0 times in every arm (`graph reallocs` 0 / 0), so every `sched-replan` line
  is a pipe slot's (R181's rule, a re-plan with no micro-batch after it, gives the same split).

**Memory after the session**, MiB per card (one read a load; another load of the same arm can differ by up to 20 MiB on
card 0, C949): sp0 16,153, 17,789, 14,071; sp1 16,157, 17,795, 14,073; base 16,153, 17,793, 14,075. The plans over the
whole memory took no more than that spread.

**For the P106 run** (registry Q1666, the traced agent run on nine cards): there the 10 slots re-planned 380 times as
the context grew (R181), the kind the switch removed here every time. What remains with the switch is one re-plan per
slot whenever a request's sampling settings differ from the slot's last request; an agent run sends the same settings
every turn, so that should be about 10 a load.

## Files

| File | What |
|---|---|
| [tokens.tsv](tokens.tsv) | per request: kind, tokens, the md5 of the token list in every arm and round, how many lists |
| [replans.tsv](replans.tsv) | per arm, round and request: the pipe's guesses, the pipe slots' re-plans (same devices, changed devices) and the plans over the whole memory from the server's count, the trace's pipe-slot and decode-path re-plans, the context's own graph reallocs |
| [causes.tsv](causes.tsv) | per arm: the trace's re-plans by cause (the node at the first changed position, or the tensor that outgrew its place) |
| [plans.tsv](plans.tsv) | the plans over the whole memory by graph size and outcome |
| [arms.tsv](arms.tsv) | per arm: build, switch, load, the pipeline-parallelism and switch lines, totals, MiB per card |
| [run/run.sh](run/run.sh) | the driver: one llama-server per arm in a container, the session twice, the memory read |
| [run/session.py](run/session.py) | the session (C949's with R12 to R15 added; C949's R9 left out) |
| [run/c950_tables.py](run/c950_tables.py) | the tables from the driver's output |

## Commands

On the build machine, detached (the builds are the rig tarballs of `19048bb6d`, md5
`8d2239e8a0d5921a013171b651777a34`, and of `3dfad876f`; `llama-server` md5 `7207345912c9ae9e13b3d916b2698403` and
`5b330eccef892b7dd9f4faf1027319c3`):

```bash
WORK=$WORK NEW=$NEW REF=$REF REPO=$REPO FLINT=$MODELS/flint-pascal-Q8.gguf MOUNTS="$MOUNTS" \
  DOCKER_ARGS="--label pascal-quants=1 --cpu-shares=256" PREFLIGHT=$PREFLIGHT \
  setsid nohup bash docs/data/2026-11-21_c950_flint_slot_plan_gpu/run/run.sh > $WORK/driver.log 2>&1 < /dev/null &
```

The driver runs the arms `sp0 sp1 base` on GPUs 0–2 in the image `nvidia/cuda:12.9.2-devel-ubuntu24.04`, each server as

```bash
env $PRESET_ENV [LLAMA_PIPE_SLOT_PLAN=1] LLAMA_UBATCH_TRACE=1 $BIN/llama-server -m $FLINT $PRESET_ARGS_WITH_TS_1_1_1 -lv 4 \
  --host 0.0.0.0 --port 18950 --no-webui
```

with the PTX compiler's cache in `$WORK/jitcache` (`CUDA_CACHE_PATH`), and `python3 -I run/session.py <server> <arm folder> r<n>`
twice per load. The tables:

```bash
D=docs/data/2026-11-21_c950_flint_slot_plan_gpu
python3 -I $D/run/c950_tables.py $D $WORK/out
```

The trace's lines carry host times; the tables do not read them.

## Provenance and licence

Code: Miner Llama (MIT, like llama.cpp), `pascal-p106-v5-pipe-slot-plan` `19048bb6d` and `pascal-p106-v5` `3dfad876f`,
the rig builds (CUDA 12.x, `sm_61`) run through their compute 6.1 PTX on three RTX 3090s. Model: Flint Pascal Q8 GGUF
(the release file). The prompts are this folder's own text; the tables hold counts and hashes, no reply text. Run on
2026-11-21.
