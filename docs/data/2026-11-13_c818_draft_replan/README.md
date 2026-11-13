# C818: the MTP draft's shared scheduler planned over the whole memory again (2026-11-13)

Registry rows [Q1427](../../../quants/experiments.md) (the change) and Q1447 (its rig check, R143, not run).

**Result on a CPU: with `LLAMA_SCHED_PER_SIZE_RESERVE=2` the MTP draft's shared scheduler re-plans once after a change of
cards and then never again, where `=1` re-plans as the context grows. The drafts, probabilities, draft states and
llama-server's replies are the same in every arm, and with the switch off the build computes what its base computes.**
A CPU never changes cards, so the change was made by a test switch at the draft's first prompt pass:

- `test-llama-archs --mtp-replan`: 9 re-plans with `=1` after it, 1 with `=2` (followed by one plan over the whole
  memory), 0 with `=2` alone. That holds in 16 of 16 runs: four configurations, seeds 1–3 on a CPU build and seed 1 on
  the rig binary's CPU code.
- llama-server on FN84's requests: R60's line re-plans the draft 18 times with `=1` (13 `FILL`, 4 k-pool stage, 1 token
  rows; R130 on the rig: 18 after its change of cards) and once with `=2`. R85's line re-plans it 42 times with `=1`
  and once with `=2`.

The fork branch is `pascal-p106-v4-b12246-c818` at `46c8ae936`, merged off by default into `pascal-p106-v4` as
`7663f3221` (on C822's `2144cf760`). Its P106 check is R143: R130's `plan` load again with `=2`.

## The problem (R130)

Flash-Next decodes with an MTP draft: a small head that guesses the next tokens, on the last card (card 8 of nine
P106-090s). Each pass of a llama.cpp context runs through a scheduler, which plans where every tensor of the graph goes
(which card, which offset in a compute buffer). A plan that fits a graph is kept; a graph that does not fit it is
planned again (a re-plan), which first waits for the cards to finish what is queued (a drain).

With C803's switches (`LLAMA_MTP_GRAPH_KEEP=1` and `LLAMA_SCHED_PER_SIZE_RESERVE=1`) the draft's small passes (up to 8
tokens: drafting and verify catch-ups) have schedulers of their own, each planned once over the whole memory, and those
never re-plan. Passes of more than 8 tokens (the prompts' catch-ups) go to the context's shared scheduler. R130 ran
FN84's load on nine P106 cards with both switches: the shared scheduler still re-planned 19 times in 8 requests
([C803's data, "The rig check (R130)"](../2026-11-12_c803_draft_realloc/README.md#the-rig-check-r130)):

- its start plan is made at load from a graph over the whole memory, so a growing context fits it; but in that graph
  the hidden rows the draft reads come from the host (`mtp_h_input`), while a catch-up pass reads them on card 8
  (a view of the target's hidden-row buffer);
- the load's first 512-token catch-up therefore puts node 40 on another card than the plan has it, and the scheduler
  plans again from that graph, sized for the context of that moment;
- from then on every pass whose context has outgrown that plan re-plans: 18 more (the k-pool's pool mask 8, the k-pool
  tables' update 4, the QSA layer's `FILL` 5, an input view 1), with 2,887.6 ms of drains and 3.714 ms of plans.

## The change

`LLAMA_SCHED_PER_SIZE_RESERVE=2` (fork branch `pascal-p106-v4-b12246-c818`, `src/llama-context.cpp`, `.h`; host code
only) is `=1` plus one rule for the shared scheduler: **when a pass's allocation re-plans the shared scheduler, for any
reason, it is planned again from that pass's graph built over the whole memory, before anything is computed.**

1. The graph allocator counts the plans it makes (new getters `ggml_gallocr_get_n_plans` and
   `ggml_backend_sched_get_n_plans`, host code in ggml): the scheduler's re-plan after a change of cards or a graph
   that outgrew the plan, and the allocator's own re-plan when the scheduler has a single buffer (a CPU, where the
   scheduler's re-plan callback is never called).
2. If the count moved during a pass's allocation in the shared scheduler, the scheduler is reset and planned as the
   start plan is: from the largest batch (`n_ubatch` tokens, the start plan's sequences and outputs), built over the
   whole memory (`init_full`, every pool of a QSA memory re-pooled), but with that pass's kind of input: its hidden
   rows read on the device when the pass reads them there (`sched_shared_plan`). The graph is reserved and never
   computed.
3. The pass's own graph is then built again and allocated in that plan. It has to be built again: on several cards a
   scheduler points a graph's nodes at the input copies of its split, and the next split frees them.

So on the rig the first catch-up still re-plans once, at its change of cards, and in the same pass the scheduler gets
the start plan again with the hidden rows on card 8, which holds every later catch-up at any context. The re-plan has
already waited for the cards, so the new plan waits for nothing.

Why this rule and not the other fix that was listed (reserving the start plan with node 40 on the draft's card):

- It acts on the cause, whatever node moves: v4c2's line adds `LLAMA_QSA_KPOOL_STAGE=1`, `LLAMA_QSA_POOL_MASK_DEV=1`
  and `LLAMA_MTP_CATCHUP_OVERLAP=1`, which place other k-pool tensors, and other models or card counts place other
  nodes. A start plan built for node 40 alone would leave those.
- It plans for the kind of pass that the shared scheduler actually takes. The start plan is made at load, before any
  pass, so it can only guess.
- The cost is one plan per re-plan that would otherwise have been followed by more, and the one re-plan at the change
  of cards stays.

Why a mode of C803's switch and not a switch of its own: it needs the same per-size schedulers. Without them the
shared scheduler takes every pass, and the MTP draft's would change cards twice a step (a drafting pass reads its
hidden row from the host, a catch-up from card 8), each change then followed by a plan over the whole memory. So `=2`
is ignored without per-size schedulers, with `=1`'s warning, and the stacked line moves from `=1` to `=2` in one value.

`LLAMA_SCHED_TEST_REPLAN=1` (a test switch): a CPU has one buffer type, so its scheduler's cards never change. Once per
context, the first shared-scheduler pass of more than 8 tokens plans it from its own graph over the memory as it is,
which is what the rig's first catch-up does after its change of cards. Nothing computed changes.

## The checks (a build machine's CPU, 2026-11-13)

All on `46c8ae936`, in containers without GPUs at a low CPU share, one at a time behind a lock (`run/checks.sh`; every
row in `checks.tsv`). The base is `pascal-p106-v4` `65ac9a47d`.

| Check | Result |
|---|---|
| `--mtp-replan` (`replan_rows.txt`) | 16 of 16: four configurations (f16 with and without the shortlist, with the k-pool, q8_0), seeds 1–3 on the CPU build, seed 1 on the rig binary (its CPU code). C803's three arms as before (`plan` 0 re-plans; `keep` 27, 28 or 30 with the shortlist by seed, 25 without); C818's: `plan+cards` 9 re-plans in prompts, `whole+cards` 1 and 1 plan over the whole memory, `whole` 0 and none; no buffer growth in the three; the same drafts, probabilities and draft states in all six arms |
| C619's `--mtp-graph-keep` (`keep_rows.txt`) | 8 of 8 with `LLAMA_SCHED_PER_SIZE_RESERVE` unset, at `1`, at `2`, and at `2` with `LLAMA_SCHED_TEST_REPLAN=1`; the same rows up to the graph builds in all four (the compute column is `1`'s with both `2` arms) |
| llama-server, FN84's requests, test-llama-archs's tiny Flash-Next with its separate draft (`server_replans.tsv`, `server_trace_summary.txt`) | R60's line (k-pool and indexer switches, shortlist, draft micro-batch 512), 208 draft evaluations: `1` and `2` 0 re-plans; with the test's change of cards `1` 18 (13 `FILL`, 4 `kpool_stage` `CONT`, 1 `mtp_tok_embd` `GET_ROWS`), `2` 1 (`FILL`) and 1 plan over the whole memory. R85's line (shortlist, micro-batch 128), 267 evaluations: 0 and 0; 42 and 1. Every arm's replies the same as its line's `1` |
| the 146 architectures' logits against `65ac9a47d`'s | the same in 146 of 146 with no switch, `LLAMA_SCHED_PER_SIZE_RESERVE=2`, `LLAMA_SCHED_TEST_REPLAN=1`, `LLAMA_SCHED_PER_SIZE=1`, the last with `=2`, and with `=2` and the test switch |
| C565's set | ctest 71 of 71 static and shared, the network test, the architectures' files (146 written and compared), the nine hand tests |
| switch list | 871 names; `LLAMA_SCHED_TEST_REPLAN` new in it |
| `llama-server`'s GPU code against the `65ac9a47d` tip tarball's | SASS the same file (8,540 functions). PTX: 10 `flash_attn_tile` instances differ in 70 lines, all the number in a function-local `__shared__` array's name (as C813 and C779 found); with it masked (C779's `ptx_names.sh`) the same file |
| rig tarball | `llama.cpp-20261113-46c8ae936-pascal-p106-v4-b12246-c818.tar.gz`, 744,437,345 bytes, md5 `45cb414a8f2bec992d08de4eb9a4f019`; `llama-server` md5 `8cc94801546e8d640d13b6a1fc4ae758`, `test-backend-ops` md5 `d136e8567a5419a9baad0fa5553778a9`; sm_61, 0 AVX instructions, GLIBC_2.38 at most, `test-llama-archs` in `bin/` |
| merged `7663f3221` (on C822's `2144cf760`) | static CPU build OK; 146 of 146 architectures the same as `65ac9a47d`'s with every switch off and with `LLAMA_SCHED_PER_SIZE=1 LLAMA_SCHED_PER_SIZE_RESERVE=2 LLAMA_SCHED_TEST_REPLAN=1`; `--mtp-replan` 4 of 4 (seed 1) |

## The rig check (R143, Q1447, not run)

R130's `plan` load again on nine P106 cards, with this branch's tarball and `LLAMA_SCHED_PER_SIZE_RESERVE=2` in place of
`1`: FN84's load (R60's `gains` line, one task of R43's replay with the taper off, one fresh 2,049-token prompt with the
taper off and then on), `-lv 5`, `LLAMA_UBATCH_TRACE=1`, with C803's
[`run/r130_loads.sh`](../2026-11-12_c803_draft_realloc/run/r130_loads.sh):

```bash
RIG=$RIG WORK=$R43 BUILDS=/opt/llama-builds BUILD=llama.cpp-20261113-46c8ae936-pascal-p106-v4-b12246-c818 MODEL=$MODEL DRAFT=$DRAFT NAME=c818-whole ADD_ENV='LLAMA_UBATCH_TRACE=1 LLAMA_MTP_GRAPH_KEEP=1 LLAMA_SCHED_PER_SIZE_RESERVE=2' bash ../2026-11-12_c803_draft_realloc/run/r130_loads.sh
```

It passes when:
- the draft's shared scheduler re-plans once, at the load's first catch-up (its change of cards), followed by one
  `sched_shared_plan` line, and 0 times after, in prompts and in generation (R130 `plan`: 19); the per-size schedulers
  0 times, as in R130;
- the 8 replies and context checkpoints are R130 `plan`'s (and R60's for the 6 replay requests);
- the `switches:` line names `LLAMA_SCHED_PER_SIZE_RESERVE=2`;
- the draft's compute buffers per card at exit, card 8's MiB after loading and at the end, and the lowest
  MemAvailable are given by value against R130 `plan`'s (5.00 / 194.22 / 37.09 MiB on CUDA4 / CUDA8 / host for the
  shared scheduler; 4,789 and 4,891 MiB; 447 MiB). The new plan has the hidden rows on card 8, so the shared
  scheduler's buffer there may grow by est. a few MiB.

If it passes, `LLAMA_SCHED_PER_SIZE_RESERVE=2` replaces `=1` in the stacked line. The tables are R130's commands
([C803's README](../2026-11-12_c803_draft_realloc/README.md#commands)) with the load folder `fn84_c818-whole`, against
R130's `fn84_c803-plan`, plus a count of the `sched_shared_plan:` lines in its `server.log`.

## Files

| File | What it holds |
|---|---|
| `run/checks.sh` | the checks on a build machine, CPU only: the rig tarball, the switch list, `llama-server`'s GPU code against the base's, a CPU build, `--mtp-replan`, `--mtp-graph-keep`, llama-server's trace (C803's `run/trace.sh` and `run/draft_trace.py`), the 146 architectures' logits, ctest, the merged tree |
| `checks.tsv` | every check's row: step, check, verdict, detail (build-machine paths shortened to `$V`) |
| `replan_rows.txt` | `test-llama-archs --mtp-replan`'s table per run (`cpu-s1` to `cpu-s3`: the CPU build, seeds 1–3; `rig-s1`: the rig binary): re-plans per arm in prompts / steps / the target, growths, plans |
| `keep_rows.txt` | `--mtp-graph-keep`'s table per arm (`off`, `plan` `=1`, `whole` `=2`, `wholecards` `=2` with the test switch) |
| `server_replans.tsv` | every draft evaluation that re-planned in llama-server's eight arms (`run/draft_trace.py parse`): request, phase, context, tokens, outputs, the allocator's reason |
| `server_trace_summary.txt` | per arm and context: evaluations, re-plans by phase, buffer growths, and the re-plans' reasons |

## Commands

On a build machine, from a checkout of this repository (`$WORK` the work folder, `$LOCK` a flock file shared with other
builds, `$SRC` a llama.cpp clone with the fork as remote `fork`, `$BASE_TREE` the rig tarball tree of `65ac9a47d`,
`$BASE_CPU` the `bin/` of its static CPU build):

```bash
M=$WORK REPO=$PWD LOCK=$LOCK BASE_TREE=$BASE_TREE BASE_CPU=$BASE_CPU SEED_SRC=$SRC JOBS=8 THREADS=8 \
  bash docs/data/2026-11-13_c818_draft_replan/run/checks.sh
M=$WORK REPO=$PWD LOCK=$LOCK BASE_TREE=$BASE_TREE BASE_CPU=$BASE_CPU SEED_SRC=$SRC JOBS=8 THREADS=8 STEPS=merged \
  bash docs/data/2026-11-13_c818_draft_replan/run/checks.sh       # after the merge into pascal-p106-v4
```

The CPU test on its own, with a build of the branch:

```bash
test-llama-archs --mtp-replan -s 1 -v 1
```

## Provenance and licence

Everything here was made for this repository: the change, the test, the scripts and the tables. FN84's requests and
R130's numbers come from those rows' own published data (C803's folder). The random-weight test models are written by
the fork's own `test-llama-archs`. The scripts are under this repository's licence.
