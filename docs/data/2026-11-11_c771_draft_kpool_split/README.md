# The MTP draft's k-pool update in its layer's split (C771, Q1357, 2026-11-11)

What this is: a change to the MTP draft's graph for Qwen3.8-Flash-Next (`qwen4exp`) and its check on a build
machine's CPU. With C100's `LLAMA_QSA_KPOOL_DEV=1` ([design](../2026-11-11_c100_kpool_on_cards/README.md)) the QSA
k-pool tables stay on every card that runs a QSA layer, and each micro-batch writes only the rows that changed into
the card's copy. The rig run of that switch (R74, [Q1273](../../../quants/experiments.md),
[data](../2026-11-11_r74_c100_p106/README.md)) found that every MTP draft pass then runs on card 8 twice. This change
puts the update into card 8's one split. Registry [Q1357](../../../quants/experiments.md); the rig run is
[Q1361](../../../quants/experiments.md) (R113).

## Why the draft ran on card 8 twice

The Flash-Next preset puts the MTP draft's layer on card 8 and its token embeddings on card 4
(`-otd token_embd=CUDA4`). A draft pass gathers its token rows on card 4 and runs the layer on card 8.

ggml adds a node to the graph when a node that reads it is added, and the scheduler cuts the graph wherever the
device changes. The draft's graph (`graph_mtp` in `src/models/qwen4exp.cpp`) builds the token rows' gather first, but
nothing adds it until the layer's first matrix product reads it. Before that product, the layer's QSA selection
(`build_qsa_sel`, or `build_qsa_kv` in a KV-only pass) adds the update of the card's copy (`build_kpool_dev`: the
stage's copy, a cast and two `SET_ROWS`), which reads only host inputs. So with the tables on the cards the graph runs
card 8 (the update), card 4 (the gather), then card 8 (the layer): two `graph_compute` calls on card 8 a pass, with a
card 4 → card 8 crossing between them that waits for the update's split too. Without the tables on the cards there is
no update and the pass runs card 4, then card 8.

## The change

**`LLAMA_MTP_EMBD_FIRST=1` adds the token rows' gather to the draft's graph as soon as it is built,** so it is the
graph's first node and the update follows it into card 8's one split. The same nodes, in another order. It is read
once per process, off by default, and acts only on a draft pass that reads token ids (`ubatch.token`); the target's
graph is not touched. Fork branch `pascal-p106-v4-b12246-c771` `fa7035917`, one commit on `pascal-p106-v4-b12246-c748`
`5b329df16` (15 lines in `src/models/qwen4exp.cpp`).

With C762's `LLAMA_DRAFT_EMBD_DEV=layer` ([design](../2026-11-11_c762_draft_rows/README.md)) the draft's token
embeddings move to card 8 with its layer, so the gather runs there too and the draft is one card-8 split in either
order, by the scheduler's rule (it cuts only where the device changes; not run here, as C762 is on its own branch):
this switch matters only while the embeddings stay on card 4.

**Why a switch, though the CPU check shows the new order could be the only one:** the results are the same bytes in
every arm, and without the tables on the cards the splits are the same too. The switch stays so that the rig run can
time the same build off and on (R113), and so that a negligible result can join the negligible group, which is tested
switch by switch ([C534](../../../quants/experiments.md)). Once R113 shows the new order is not a loss it can become
the only order.

## Result (build machine, CPU only)

**With the tables on the devices every draft evaluation runs 2 splits instead of 3, one of them on the draft layer's
device instead of two, with the same nodes on each device; replies, top-3 probabilities, draft counts and all 406
verify steps are the same off and on.** Measured with llama-server on the CPU with two RPC devices laid out as cards 4
and 8 (the draft's layer on the second, its token embeddings on the first), from the scheduler's own line for every
evaluation (`LLAMA_UBATCH_TRACE=1`).

| Arm pair (off → on) | k-pool switches | Draft evaluations | Splits each | On the layer's device | Nodes per evaluation (device 1, device 2), off → on | Target evaluations |
|---|---|---:|---|---|---|---:|
| `r` → `re` | none | 1,661 | 2 → 2 | 1 → 1 | `dev1:n1 dev2:n167` (drafting) the same | 429, the same splits |
| `rd` → `rde` | `KPOOL_DEV=1` (R74's `dev`) | 1,661 | **3 → 2** | **2 → 1** | `dev2:n7 dev1:n1 dev2:n169` → `dev1:n1 dev2:n176` | 429, the same splits |
| `rdc` → `rdce` | + `_CHECK=1` (C100's check) | 1,661 | **3 → 2** | **2 → 1** | as `rd` | 429, the same splits |
| `rds` → `rdse` | + `LLAMA_QSA_KPOOL_STAGE=1` (C720) | 1,661 | **3 → 2** | **2 → 1** | `dev2:n7 dev1:n1 dev2:n178` → `dev1:n1 dev2:n185` | 429, the same splits |
| `rdk` → `rdke` | + `LLAMA_MTP_KV_ONLY=1` (C620) | 1,661 | **3 → 2** | **2 → 1** | catch-up passes `dev2:n7 dev1:n1 dev2:n59` → `dev1:n1 dev2:n66` | 429, the same splits |
| `rdb` → `rdbe` | `KPOOL_DEV=1`, `-bs` | 1,661 | **3 → 2** | **2 → 1** | as `rd` | 429, the same splits |

- **The evaluations:** 1,218 drafting passes (1 token, 1 output), 404 four-token catch-up passes after a verify step,
  24 passes of 64 prompt tokens and 15 others; every kind gives the same splits as in the table
  ([splits.tsv](splits.tsv)). Each pair's evaluations match one to one in kind, tokens and outputs.
- **The update's 7 nodes move into the layer's split:** 7 + 169 = 176 nodes on device 2, 1 on device 1, in every
  drafting pass. The target's splits are the same in every pair: the switch touches only the draft's graph.
- **C100's check with the switch on:** `rdce` compared every micro-batch's copies with the host's tables and found no
  difference: 1,661 draft and 429 target micro-batches checked, each with 1 full write, 48,511 and 12,842 rows sent,
  the same counts as `rdc` ([kpool.tsv](kpool.tsv)).
- **The replies:** `off` and `on` (llama-server on the CPU alone, the base `5b329df16` as `base`) and every pair give
  the same tokens, top-3 probabilities, draft counts (`draft_n`, `draft_n_accepted`) and the 406 verify steps of four
  requests (two greedy, two at temperature 1.0, 200 and 600 prompt tokens; [server.tsv](server.tsv)). `r`, on two
  devices, also equals `base`.

### The other checks

| Check | Verdict |
|---|---|
| static CPU builds of `fa7035917`: every target; llama-server and ggml-rpc-server with `GGML_RPC=ON` | OK, OK |
| C748's `test-llama-archs --mtp-kv-only` with the switch on: 7 switch sets (none; the tables on the devices; with C100's check; with C720's stage; with both; `=2` with both; the four QSA switches with their checks) × seeds 1, 2, 3 | 21 of 21 runs, 105 of 105 configurations: the draft's KV after the prompt, every draft and the KV at the end identical with the KV-only pass off and on; with the tables on the devices 1 full write and 574 rows a run, cut and whole ([kvonly.tsv](kvonly.tsv)) |
| every switch off: the base `5b329df16` writes every architecture's logits (`test-llama-archs -b CPU -s 1`), this build compares | 146 of 146 the same; with the switch on 146 of 146 the same; `qwen4exp` with the switch, the tables on the devices and C720's stage, and with C100's check too: the same |
| ctest (without the tests that need a download, the network or torch) | 70 of 70 |
| the rig tarball (`scripts/build_llamacpp_docker.sh`, `TESTS=1`, sm_61) | `llama.cpp-20261111-fa7035917-pascal-p106-v4-b12246-c771.tar.gz`, 656,349,089 B, md5 `5401cab858e39f38d32c4f29881ecbed`; 0 AVX instructions in `llama-bench`, highest glibc symbol `GLIBC_2.38`; `test-llama-archs` in its `bin/` |

`test-llama-archs`'s logits come from each model's own graph, not an MTP draft's, so its arms with the switch show
that nothing outside the draft's graph changes; the server check and the KV-only test run the draft's graph.

The server step ran twice. The first try stopped at its model step: `test-llama-archs --mtp-draft`, which writes the
MTP-only draft file, is C762's addition ([C762](../2026-11-11_c762_draft_rows/README.md)) and not on this branch. The
second try used the two files C762's run wrote (md5 `b67f1d9c…` for the model, `8ba549ab…` for the draft). The CPU
builds of the first try served every later step.

## What it should save on the rig

Est. from R74's host profile (the `hostdev` load, pass 2): counting from the update's entry, a drafting pass's lead is
0.08 ms longer than its crossing with the tables on the cards, against 0.02 to 0.04 without, so the extra split costs
est. ~0.04 to 0.06 ms a draft pass. A step has four draft passes (the catch-up
and three drafting passes), so est. ~0.16 to 0.24 ms a step, inside R74's spreads (0.59 and 0.35 ms): the negligible
class unless the draft's turns follow. R74 also measured the draft's turns before its last pass growing 0.50 ms more
from 16,384 to 49,152 tokens with the tables on the cards than without; if that is the extra split, the saving grows
with depth.

## The rig run (Q1361, R113)

R74's `hostdev` load twice, on one rig, on this branch's tarball: `hostdev` (the switch off) and `hostdevf`
(`LLAMA_MTP_EMBD_FIRST=1`), each `LLAMA_QSA_KPOOL_DEV=1` with the host profile and `-lv 4`, at 0, 16,384 and 49,152
tokens. Est. 32 min of loads (R74's `hostdev` took 16 min) plus the relay.

```bash
# from a machine that reaches the rig over ssh (RIG: its ssh alias; BIN: this branch's tarball's bin/ on the rig;
# MODEL, MODEL_DRAFT: paths on the rig; WORK: a folder on the rig; TEXT_LOCAL: the release check's recorded requests,
# proxy.jsonl.bodies.gz)
RIG=$RIG WORK=$WORK BIN=$BIN/llama-server MODEL=$MODEL MODEL_DRAFT=$MODEL_DRAFT TEXT_LOCAL=$TEXT_LOCAL TAG='^t1swift/' \
  WD_ENV=MIN_AVAIL_MB=60 ENV_ADD=GGML_CUDA_FATTN_SPARSE_PASCAL=1 \
  ARMS="hostdev hostdevf" \
  ARM_ENV_hostdev="LLAMA_QSA_KPOOL_DEV=1 LLAMA_QSA_INPUT_STATS=1 GGML_CUDA_PROFILE_HOST=$WORK/out/hostdev.csv" ARM_ARGS_hostdev="-lv 4" \
  ARM_ENV_hostdevf="LLAMA_QSA_KPOOL_DEV=1 LLAMA_MTP_EMBD_FIRST=1 LLAMA_QSA_INPUT_STATS=1 GGML_CUDA_PROFILE_HOST=$WORK/out/hostdevf.csv" ARM_ARGS_hostdevf="-lv 4" \
  LOADS="hostdev:hostdev hostdevf:hostdevf" DEPTHS="0 16384 49152" PAUSE=4 \
  bash bench/deep_lane/remote.sh start
RIG=$RIG WORK=$WORK bash bench/deep_lane/remote.sh wait
RIG=$RIG WORK=$WORK OUT=$OUT CLEAN=0 bash bench/deep_lane/remote.sh collect
```

- **Read:** R74's `run/fold_updates.py` folds about 2,188 evaluations in `hostdev` and none in `hostdevf`, where each
  draft pass has one card-8 evaluation; then R64's `host_depth.py` per load: a drafting pass's lead against its
  crossing, the draft's turns before its last pass, the step on pass 2 at 16,384 and 49,152.
- **Right when** both loads write R74's reply ids at every depth (`08569d4a…`, `41cfc791…`, `3a311cdd…`).
- **The verdict:** `hostdevf`'s step against `hostdev`'s on pass 2, classed against R74's spreads at the same depths
  (0.59 and 0.35 ms); est. ~0.16 to 0.24 ms a step, more at depth if the turns follow.

## Files

| File | What it holds |
|---|---|
| [summary.tsv](summary.tsv) | every check's verdict and its detail |
| [server.tsv](server.tsv) | per server arm and request: prompt and generated tokens, draft counts, the generated ids |
| [server_verdict.txt](server_verdict.txt) | the server check's verdict lines, `splits.py`'s among them |
| [splits.tsv](splits.tsv) | per RPC arm, evaluation kind (`mtp` the draft, `dec` the target), tokens, outputs and split pattern (each split's device and nodes, in order): the evaluations and the splits on the draft layer's device |
| [kpool.tsv](kpool.tsv) | per RPC arm with the tables on the devices, each memory's counts when freed: devices, micro-batches, full writes, rows sent and read, micro-batches checked |
| [kvonly.tsv](kvonly.tsv) | C748's `--mtp-kv-only` test per arm, seed and configuration, the switch on |
| [run/checks.sh](run/checks.sh) | the whole check: builds, the server check, the KV-only test, every switch off against the base, ctest, the rig tarball |
| [run/server_check.sh](run/server_check.sh) | llama-server's arms on the CPU and on two RPC devices, the requests and the verdict |
| [run/splits.py](run/splits.py) | the splits of every evaluation from the trace lines, and the pairs' verdict |

## Commands

```bash
# on a build machine with Docker (CPU only), from this repository's root, with a llama.cpp clone that has the fork as
# remote "fork" (SRC), the base's static native CPU build (BASE_CPU: bin/ with test-llama-archs; BASE_SRV: bin/ with
# llama-server, both of pascal-p106-v4-b12246-c748 5b329df16) and the random qwen4exp model with one MTP block and its
# MTP-only draft (MODELS: written by test-llama-archs -a '^qwen4exp$' -s 1 -o <dir> --mtp-draft on
# pascal-p106-v4-b12246-c762 950844caf, md5 b67f1d9c… and 8ba549ab…)
WORK=$WORK REPO=$PWD SRC=$SRC BRANCH=pascal-p106-v4-b12246-c771 BASE_CPU=$BASE_CPU BASE_SRV=$BASE_SRV MODELS=$MODELS \
  JOBS=8 LOCK=$WORK/lock bash docs/data/2026-11-11_c771_draft_kpool_split/run/checks.sh
```

Every build and test ran in a container at `nice -n 19` and `--cpu-shares=256`, one at a time under a lock, without a
GPU. Provenance: the model files are random weights written by the fork's own test; the tables are this check's
output. Licence: as this repository.
