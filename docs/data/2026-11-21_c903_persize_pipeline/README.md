# Flint's per-size schedulers made with pipeline parallelism (C903, Q1632, 2026-11-21)

What this is: an engine change and its checks on a CPU. With `LLAMA_SCHED_PER_SIZE=1`, every pass of up to 8 tokens
runs on a scheduler of its own size. That keeps Flint Pascal Q8 GGUF's prompt scheduler on its load-time plan: no
re-plans in the middle of prompts, and the prompts that re-planned before read in 37.1 s instead of 70.1 s
([R216a](../2026-11-19_r216_flint_persize/README.md)). But those small schedulers were made without the pipeline
parallelism the context has: one copy of their inputs and no events on the cards. So at each change of card the host
waited for the previous card to finish. A one-token pass under the tool-call grammar took 23.9 ms (median) to hand its
work to the nine cards, against 2.9 ms on the prompt scheduler, and a grammar token came 26.92 ms after the last,
against 24.77 ms without the switch. The last micro-batch of a prompt (4 tokens under Flint's server) first waited for
the whole prompt: 114.9 ms of host time against 25.0 ms. Registry row [Q1632](../../../quants/experiments.md).

The change (fork branch `pascal-p106-v5-sched-per-size-pp`, one commit `c1772c7fc` on `pascal-p106-v5` `6644c2bc6`,
`src/llama-context.cpp` and `.h`): in a context with pipeline parallelism, each per-size scheduler is now made as the
shared scheduler is, with four copies of its inputs and an event per card and copy. The next card then waits for the
previous one on the cards, and the host hands every card its work without waiting. The last micro-batch of a prompt no
longer waits before it starts, since it writes its own copy of the inputs. The new switch
[`LLAMA_SCHED_PER_SIZE_PP`](../../switches.md#graphs-scheduler-and-memory) is read only with `LLAMA_SCHED_PER_SIZE=1`:
`1` (the default) as described, `0` the schedulers of before, `2` a test that makes them with the copies even without
pipeline parallelism. Without `LLAMA_SCHED_PER_SIZE=1` nothing changes.

## Result (one CPU)

**In plain words:** the change builds, the test suite fails only where its base fails, and on one CPU it gives exactly
the same numbers and tokens as before, with the switch off, on as before, and on in the test mode with four copies.
One CPU has no pipeline parallelism, though, so there the default makes the small schedulers exactly as before. Only
the test mode runs the four copies on a CPU, and the events on the cards and the skipped wait run only on the rig.
Whether it wins back the 2.15 ms a grammar token (est. 5.8 s over R216a's run) and the 4-token pass's wait is for the
traced P106 run below.

| Check | Base `6644c2bc6` | Change `c1772c7fc` |
|---|---|---|
| CPU build, every target, tests on ([builds.tsv](builds.tsv)) | rc 0, 19 s (from the cache) | rc 0, 142 s |
| Compiler warnings | 30 lines | 30 lines, the same set |
| ctest | 71 of 82 pass | 71 of 82 pass, the same 11 failures |

The 11 failures are the base's own: three tests that download (and the three that need their downloads, Not Run), three
that need Python's `numpy`, two `test-jinja` cases that read files through a path the compiler cache made relative, and
`test-kv-cache` (QUEUES code row C881).

**Every architecture, steps of 1 to 8 tokens** ([archs_summary.txt](archs_summary.txt)): `test-llama-archs` with
`LLAMA_ARCHS_LOGITS` decodes 128 tokens of each architecture's random model on the CPU as one batch and again in steps
of 1, 2, … 8 tokens, the sizes that run on per-size schedulers, and writes the logits; the first arm writes them and
every later arm compares value for value.

| Arm | Build | Switches | Architectures compared | Files with any logit different | Rows OK / FAIL |
|---|---|---|---:|---:|---|
| off | change | — | (writes 148) | — | 153 / 0 |
| base_on | base | `LLAMA_SCHED_PER_SIZE=1` | 148 | 0 | 153 / 0 |
| new_on | change | `LLAMA_SCHED_PER_SIZE=1` | 148 | 0 | 153 / 0 |
| new_pp0 | change | and `LLAMA_SCHED_PER_SIZE_PP=0` | 148 | 0 | 153 / 0 |
| new_pp2 | change | and `LLAMA_SCHED_PER_SIZE_PP=2` | 148 | 0 | 153 / 0 |
| new_pp2_r | change | and `=2`, `LLAMA_SCHED_PER_SIZE_RESERVE=2` | 148 | 0 | 153 / 0 |

Flint's architecture traced in each arm ([archs_flint_traced.txt](archs_flint_traced.txt)): the change's build made 8
per-size schedulers (one per step size) with 1 input copy in `new_on` and `new_pp0` and with 4 in `new_pp2` and
`new_pp2_r`, and the logits match the reference with 0 of 32,768 different in every arm. Here each per-size scheduler
builds its graph once and reuses it, so only its first copy is used.

**llama-server with a grammar** ([server_tokens.tsv](server_tokens.tsv), [server_persize.txt](server_persize.txt)): a
random model of Flint's architecture (test-llama-archs, seed 1, GPT-2's vocabulary; 111,158,272 bytes) on six CPU
threads, with Flint's preset switches that act on a CPU (`LLAMA_KV_SEQ_RM_BOUND=1`, `LLAMA_SAMPLING_TOP_K_ROWS=1`,
`LLAMA_SCHED_PER_SIZE_RESERVE=2`) and flags (`-fa on -bs --ctx-checkpoints 4 --cache-ram 0 --pipe-depth 4`, `-b 64
-ub 16`). Four requests: R1 320 tokens of plain text at temperature 0 (the reply crosses 256 cells, where graphs are
built again); R2 a tool call under a JSON schema at temperature 0; R3 the same at temperature 0.7, seed 7; R4 R2's
prompt, R2's reply and a tool result, reusing 22 cached tokens.

| Request | Tokens | off | base_on | new_on | new_pp0 | new_pp2 |
|---|---:|---|---|---|---|---|
| R1 plain, temperature 0 | 320 | `0eeda732b9de` | `0eeda732b9de` | `0eeda732b9de` | `0eeda732b9de` | `0eeda732b9de` |
| R2 schema, temperature 0 | 42 | `ff55ffa0f0a6` | `ff55ffa0f0a6` | `ff55ffa0f0a6` | `ff55ffa0f0a6` | `ff55ffa0f0a6` |
| R3 schema, temperature 0.7 | 51 | `ef014c803223` | `ef014c803223` | `ef014c803223` | `ef014c803223` | `ef014c803223` |
| R4 second turn, schema | 43 | `187f39965511` | `187f39965511` | `187f39965511` | `187f39965511` | `187f39965511` |

(The first 12 hex digits of the md5 of each reply's token list.) In `new_pp2`, traced, the server made 5 per-size
schedulers, each with 4 input copies: 2 tokens with outputs (the server's warm-up), 4 tokens without and with outputs
(a prompt's last micro-batch and its output pass), 6 without, and 1 with outputs (generation). Their passes used all
four copies: 53 passes on copy 1, 43 on copy 2 and 1 on copy 3, beside 388 on copy 0, a count that also holds every
pass of the shared scheduler (one copy on a CPU). In 4 decode calls a prompt's last micro-batch, of 4 or 6 tokens, ran
on a per-size scheduler while the call's earlier micro-batches were still counted in flight; there the change skips
the wait only in a context with pipeline parallelism, so on the CPU the code waited as before (on a CPU a wait does
nothing).

### Memory (est.)

Each per-size scheduler now keeps four copies of its inputs on every card and in the host's pinned memory, where it kept
one. The largest input is the attention mask: under Flint's `LLAMA_SCHED_PER_SIZE_RESERVE=2` it is planned over the
whole 262,144-cell context, 512 KiB a copy for a one-token pass (f16), 1 MiB for 2 tokens and 2 MiB for 4. For the
schedulers of 1, 2 and 4 tokens that is est. 14 MiB a card with four copies against 3.5 MiB with one, est. +10.5 MiB a
card and as much pinned host memory (more if the server makes more sizes). R216a left 72 MiB on cards 1 and 4 at its
peak and 1,030 MB of MemAvailable at its lowest; the P106 run measures it.

## What follows

- **The traced P106 run (C919)**, R216a's run on this build: Flint's v5 preset with `LLAMA_SCHED_PER_SIZE=1` and the
  trace, on rig 1 at stock, compared with R216a. What it should show: a one-token pass under the grammar handed to the
  cards in about the prompt scheduler's time (R216a: 23.9 against 2.9 ms), grammar tokens as far apart as without the
  switch (R216a: 26.92 against 24.77 ms at the first request), the 4-token output pass no longer waiting (R216a: 114.9
  ms of host time), still 0 re-plans of the prompt scheduler, and every card's peak within its 6,144 MiB. Optionally the
  control on the same build (`EXTRA=LLAMA_SCHED_PER_SIZE_PP=0`, arm `f5p0t`), which separates this change from the five
  contributions merged into `6644c2bc6` since R216a's `602691153`.
- Then R216b and R216c, the two untraced runs against R215's control runs, on whichever per-size schedulers win.

## Files

| File | What |
|---|---|
| [builds.tsv](builds.tsv) | the two CPU builds: commit, rc, seconds, warnings, ctest |
| [archs_summary.txt](archs_summary.txt) | `test-llama-archs` per arm: rc, seconds, logits files written and compared, files with any value different, rows OK and FAIL |
| [archs_flint_traced.txt](archs_flint_traced.txt) | Flint's architecture per arm, traced: per-size schedulers and their copies, traced passes, passes on copies 1 to 3, rebuilt passes, the logits line |
| [server_tokens.tsv](server_tokens.tsv) | each arm's four requests: tokens, prompt and cached tokens, md5 of the token list, ms |
| [server_persize.txt](server_persize.txt) | the per-size schedulers each server logged; `new_pp2`'s passes by input copy |
| [run/c903_archs.sh](run/c903_archs.sh) | the archs check |
| [run/c903_server.sh](run/c903_server.sh) | the server check |
| [run/c903_rig.sh](run/c903_rig.sh) | the P106 run: R216a's `r216.sh` with `EXTRA` and the arm `f5ppt` |

## Commands

The fork's two trees (`git archive` of `6644c2bc6` and `c1772c7fc`) were built in Docker on a build machine's CPU
(image: Ubuntu 24.04, gcc 13.3.0, cmake 3.28.3), every target, static, tests on:

```bash
cmake -S src -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=ON \
  -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_UI=OFF
cmake --build build -j 8 -- -k
cd build && ctest -j 8 --timeout 3600 --output-on-failure
```

Then, in a container with no network (the server check's image also has `curl` and `jq`):

```bash
NEW=$WORK/cpu-new/build/bin/test-llama-archs BASE=$WORK/cpu-base/build/bin/test-llama-archs OUT=$WORK/archs \
  bash docs/data/2026-11-21_c903_persize_pipeline/run/c903_archs.sh
NEW=$WORK/cpu-new/build/bin BASE=$WORK/cpu-base/build/bin VOCAB=$WORK/cpu-new/src/models/ggml-vocab-gpt-2.gguf \
  OUT=$WORK/server bash docs/data/2026-11-21_c903_persize_pipeline/run/c903_server.sh
```

The rig's build: `TESTS=1 HF_UI_VERSION=b12483 scripts/build_llamacpp_docker.sh fork/pascal-p106-v5-sched-per-size-pp`
gave `llama.cpp-20261120-c1772c7fc-pascal-p106-v5-sched-per-size-pp.tar.gz` (707,061,369 bytes, md5
`85259a404c6fcbf567faa61897a5ea2b`; its `llama-server` md5 `716f8e99a182c821b20015dc1d6d56b6`). The change is host code
only (`src/llama-context.cpp` and `.h`): it changes no CUDA source.

The P106 run (C919), from the repository's root on the machine that drives the rig, as R216a with the C903 build's
`llama-server` as `BIN`:

```bash
RIG=rig CLIENT=local MODEL=$MODEL BIN=$BIN RIG_LOGDIR=$WORK/agent-logs \
  RIG_PORT=8282 RELAY_PORT=8286 PROXY_PORT=18783 CLIENT_DIR=$CLIENT_DIR CLIENT_OUT=$CLIENT_OUT OUT=$OUT \
  PREFIX=acr-c919 TRACE=1 ARM=f5ppt bash docs/data/2026-11-21_c903_persize_pipeline/run/c903_rig.sh
```

Its tables with R216a's scripts, R216a's run beside it (`r216_grammar.py` and `r216_cards.py` take one `label=file`
argument per run):

```bash
D=docs/data/2026-11-21_c903_persize_pipeline R216=docs/data/2026-11-19_r216_flint_persize
python3 -I $R216/run/r216_replans.py $D $CLIENT_OUT/rig/server-f5ppt.log
python3 -I $R216/run/r216_grammar.py C919=$CLIENT_OUT/rig/server-f5ppt.log R216a=$R216A_CLIENT_OUT/rig/server-f5pst.log > $D/grammar.tsv
python3 -I $R216/run/r216_prompts.py C919=$D/requests.tsv R216a=$R216/requests.tsv > $D/prompts_by_size.tsv
```

## Provenance and licence

Code: Miner Llama (MIT, like llama.cpp). The models are random weights made by the fork's `test-llama-archs`, with
GPT-2's vocabulary file from the llama.cpp tree. The tables hold counts, times and hashes, no text. Built and run on a
build machine's CPU on 2026-11-20 and 2026-11-21, no GPU.
