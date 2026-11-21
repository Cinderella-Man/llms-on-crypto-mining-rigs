# Flint's per-size schedulers with pipeline parallelism on three newer cards: the GPU half of its check (C949; 2026-11-21)

Registry rows [Q1687](../../../quants/experiments.md) (the GPU half of the night's branches; this is its third point,
C949) and [Q1632](../../../quants/experiments.md) (the change). The processor half is
[2026-11-21_c903_persize_pipeline](../2026-11-21_c903_persize_pipeline/README.md).

**Passed.** Flint Pascal Q8 GGUF split over three cards runs with pipeline parallelism, which one processor cannot
have, so only here does the switch's default take the new path. With `LLAMA_SCHED_PER_SIZE=1`, every per-size
scheduler is made with 4 input copies under `LLAMA_SCHED_PER_SIZE_PP=1` and with 1 under `=0`, and the small passes
under `=1` use all four copies. Every reply is the same token list in all four arms (`=0`, `=1`, per-size schedulers
off, and the base build): 10 requests of a tool-call session, 280 replies, one token list per request. The new path
that skips a wait also ran: in the padded prompts, a call's last micro-batch of 1 or 4 tokens started while the
call's earlier micro-batch was still on the cards, which under `=1` skips the wait that `=0` takes; the replies are
the same. Nothing here is timed: the speed half is a P106 rig row.

## What ran where

| | |
|---|---|
| New build | fork branch `pascal-p106-v5-sched-per-size-pp` `c1772c7fc`: one commit on `pascal-p106-v5` `6644c2bc6`, "llama: per-size schedulers made with pipeline parallelism (C903, LLAMA_SCHED_PER_SIZE_PP)". Rig tarball md5 `85259a404c6fcbf567faa61897a5ea2b`, `llama-server` md5 `716f8e99a182c821b20015dc1d6d56b6` ([its build](../2026-11-21_c903_persize_pipeline/README.md)) |
| Base build | `pascal-p106-v5` `3dfad876f` ([its build](../2026-11-21_c908_v5_merged_build/README.md)), rig tarball md5 `0b5b983e22d7296ff44fe363d55f61bb`, `llama-server` md5 `5b330eccef892b7dd9f4faf1027319c3`. It is `6644c2bc6` plus one change to how `LLAMA_PIPE_GRAMMAR` is read, which nothing here sets, so it stands in for the branch's own base |
| Cards | three RTX 3090 24 GB on a build machine, running the rig build's compute 6.1 PTX through the driver's JIT, as in [C947](../2026-11-21_c947_v5_merged_gpu/README.md). A CUDA preflight passed on all three before each run |
| Model | Flint Pascal Q8 GGUF (`p065`, 40,431,236,704 B, sha256 `15c501699c52d8c856cf4064035d9b3b40ae5088d3e17870c6e18c393ece4854`), split over the three cards |
| Settings | Flint's v5 preset ([release/flint-v5/settings.env](../../../release/flint-v5/settings.env)), every switch and flag as there, `--pipe-depth 4` included, with the layer split changed to the three cards (`-ts 1,1,1`); plus `LLAMA_UBATCH_TRACE=1` (each per-size scheduler's creation line, each micro-batch with its graph's reuse and why, and each scheduler pass with the input copy it used) and `-lv 4` (libllama's info lines, among them "pipeline parallelism enabled") |

The arms, one llama-server each, in a container:

| Arm | Build | Switches |
|---|---|---|
| `pp0` | new | `LLAMA_SCHED_PER_SIZE=1 LLAMA_SCHED_PER_SIZE_PP=0` (the per-size schedulers of before the change) |
| `pp1` | new | `LLAMA_SCHED_PER_SIZE=1 LLAMA_SCHED_PER_SIZE_PP=1` (the default with `LLAMA_SCHED_PER_SIZE=1`) |
| `off` | new | none: no per-size schedulers (the preset as released) |
| `base` | base | `LLAMA_SCHED_PER_SIZE=1` |

The session ([run/session.py](run/session.py)), run twice in each load; greedy (temperature 0) but R3, the token ids
returned:

| Request | What | Tokens |
|---|---|---|
| R1 | plain text: a Python snippet continued, no prompt cache | 320 (`ignore_eos`) |
| R2 | a coding task, the first tool call under a JSON schema (a grammar from the first token), no prompt cache | up to 96 |
| R3 | R2's prompt at temperature 0.7, top-k 8, seed 7 | up to 96 |
| R4–R7 | later turns of one conversation, each adding the model's reply and a tool result (a directory listing, a ~45-line module, a ~100-line test file, a one-line result), the prompt cache on: 72, 290, 872 and 31 new prompt tokens | up to 96 each |
| R8 | the same conversation, a short result and a question in plain text | 128 |
| R9, R10, R11 | fresh tool-call prompts (no prompt cache) padded with the server's tokenizer to exactly 264, 645 and 648 tokens | up to 96 each |

The server sends a prompt's last 4 tokens as a call of their own, after a synchronization, and calls of at most 512
tokens (`-b 512`). So in R1–R8 every small pass starts with nothing on the cards. The padded prompts make calls of
129 and 132 tokens, which split into micro-batches of 128 + 1 and 128 + 4: the last one runs on a per-size scheduler
while the 128-token micro-batch is still on the cards. That is the case where `=1` skips the wait and `=0` waits for
every card. R9 was meant to do the same inside Flint's 64-token taper (prompts of up to 600 tokens). No prompt of
exactly 264 tokens came out of the padding search, so R9 was never sent. The taper keeps a 9-token floor inside a
call anyway.

Four runs of the driver:

| Run | Server verbosity | Session | Notes |
|---|---|---|---|
| `run1-lv3` | `-lv 3` (the driver's first version) | R1–R8 | "pipeline parallelism enabled" is a verbosity-4 line in this fork, so it is missing here; the base's creation lines (info lines before the change) too |
| `run2` | `-lv 4` | R1–R8 | |
| `run3` | `-lv 4` | R1–R8, R10 (the padding closed gaps of up to 3 tokens, so R9 and R11 were not sent) | |
| `run4` | `-lv 4` | R1–R8, R10, R11 ([run/session.py](run/session.py) as committed) | |

## Results

**Tokens** ([same.tsv](same.tsv); every reply in [tokens.tsv](tokens.tsv)): one token list per request across every
run, arm and round.

| Request | Replies | Distinct token lists | Token ids' md5 (12 hex digits) |
|---|---|---|---|
| R1 | 32 | 1 | `1e919fd5b8f5` |
| R2 | 32 | 1 | `4d7c67a5b956` |
| R3 | 32 | 1 | `45f26934f6df` |
| R4 | 32 | 1 | `59d4a6353486` |
| R5 | 32 | 1 | `e1a8af0e504e` |
| R6 | 32 | 1 | `a176b48ed464` |
| R7 | 32 | 1 | `d5f28fbf489e` |
| R8 | 32 | 1 | `48858a1a9beb` |
| R10 | 16 | 1 | `0ba792f5b34f` |
| R11 | 8 | 1 | `0ba792f5b34f` (the same tool call as R10's) |

**The schedulers and their passes** ([arms.tsv](arms.tsv); creation lines in [persize.txt](persize.txt), small passes
by graph and copy in [small.tsv](small.tsv)), run 4:

| Arm | "pipeline parallelism enabled" | Per-size schedulers made | Input copies each | Passes of 1–8 tokens, by input copy | Rebuilt while the call's earlier micro-batch was on the cards |
|---|---|---|---|---|---|
| `pp0` | yes | 5 | 1 | 346: copy 0 346 | 4 |
| `pp1` | yes | 5 | 4 "(pipeline parallelism)" | 346: copy 0 91, 1 90, 2 83, 3 82 | 4 |
| `off` | yes | 0 | (the shared scheduler, 4) | 346: copy 0 86, 1 50, 2 168, 3 42 | 4 |
| `base` | yes | 5 | (the line before the change gives no count; one copy) | 346: copy 0 346 | 4 |

The five per-size schedulers are for 1-, 2- and 4-token passes with outputs, and 1- and 4-token passes without
(the padded calls' last micro-batches). Runs 1 to 3 give the same picture with 3, 3 and 4 schedulers: `pp1`'s small
passes (258, 258 and 302) spread over all four copies, and `pp0`'s and `base`'s all use copy 0.

**The skipped wait** ([tails.tsv](tails.tsv)): in runs 3 and 4 every arm has the padded calls' last micro-batch rebuilt
`in_flight` (an earlier graph still on the cards, with pipeline parallelism), 2 and 4 times per load. Under `pp1` that
pass ran on a per-size scheduler with 4 copies and skipped the wait (on copy 0 in the first round, copy 1 in the second);
under `pp0` and `base` it waited for every card first. The replies, R10's and R11's, are the same token list in all of
them.

**Memory** (MiB per card after the session, one read per load, [arms.tsv](arms.tsv)): `pp1` against `pp0` in the same
run, +18/+18/+22 (run 1), +10/+8/+14 (run 2), +10/+8/+14 (run 3) and +24/+26/+28 (run 4, five schedulers). The same arm
read from 16,143 to 16,163 MiB on the first card across the four runs, so a single read is within ~20 MiB of the next
load's. The estimate for the 262,144-cell plan was +10.5 MiB a card for three schedulers; the P106 run measures it on
the 6 GB cards.

## What this does not cover

- No timing: whether the copies and events win back the 2.15 ms a grammar token that the P106 run R216a measured, and
  what the skipped wait saves, is the P106 rig row C919 (rig 2 table row 2b, on rig 1).
- Three cards, not nine: each change of card in a pass is a hop between two 3090s here.

## Files

| File | What |
|---|---|
| [same.tsv](same.tsv) | per request: replies, distinct token lists, their md5s |
| [tokens.tsv](tokens.tsv) | every reply: run, arm, round, request, generated tokens, prompt tokens read, cached tokens, token ids' md5 |
| [arms.tsv](arms.tsv) | per run and arm: the pipeline parallelism line, per-size schedulers and their copies, small passes by copy, passes rebuilt in flight, calls ending in a small micro-batch behind others, MiB per card |
| [small.tsv](small.tsv) | the passes of 1–8 tokens by tokens, outputs, graph (reused, or rebuilt and why) and input copy |
| [tails.tsv](tails.tsv) | the calls of several micro-batches whose last has 1–8 tokens, with each micro-batch's input copy |
| [persize.txt](persize.txt) | every per-size scheduler's creation line |
| [run/run.sh](run/run.sh) | the driver (md5 `fda1ab106d3b91d56273123b8e6e5a89`) |
| [run/session.py](run/session.py) | the session (md5 `ba93e32448732b782475203be1e0e5a9`) |
| [run/c949_tables.py](run/c949_tables.py) | the tables from the server logs (md5 `9a16a8dcf3e3fe2621ebea479a6506bd`) |

## Commands

On the build machine, detached, with the machine's own paths in the environment:

```bash
WORK=<output folder> NEW=<new rig build>/bin REF=<base rig build>/bin REPO=<a copy of this repository> \
  FLINT=<Flint Pascal Q8 GGUF> MOUNTS='<folders holding the builds and the model>' \
  DOCKER_ARGS='--label <your label> --cpu-shares=256' PREFLIGHT=<a CUDA check, optional> RUN=run4 \
  setsid nohup bash run/run.sh > $WORK/driver.log 2>&1 < /dev/null &
python3 -I run/c949_tables.py <tables folder> $WORK/run1-lv3 $WORK/run2 $WORK/run3 $WORK/run4
```

Runs 1 to 3 were the same command with `RUN=run2` and `RUN=run3` (run 1 with the first driver, `-lv 3` fixed and no
`RUN`) and the earlier session versions described above.

Provenance: the replies are Flint Pascal Q8 GGUF's (the model's licence applies); the prompts and the test files in
`session.py` are written for this check. The raw server logs stay with the machine's records.
