# C227 / C228 (Q780, Q781): the night's second wave of Flash-Next engine branches in one tree (2026-11-04)

During the night of 2026-11-03/04 fourteen more engine branches for Qwen3.8-Flash-Next on the nine-P106 rig were written
side by side, most of them on the first merged tree `pascal-p106-v3-b12076-ladder2`
([2026-11-04_ladder2_merge](../2026-11-04_ladder2_merge/README.md)), three on its predecessor. Each one was built and
checked on its own without a GPU. This folder holds their merge into one fork branch, `pascal-p106-v3-b12076-ladder3`,
the checks that ran on the merged tree without a GPU, and the script that runs every branch's GPU check on the merged
build, each switch on and off, so the morning's GPU and rig work runs one build with switches instead of fourteen. The
registry rows are Q780 (the merge) and Q781 (its GPU checks) in [quants/experiments.md](../../../quants/experiments.md).

**Status:** merged, built for the rig and checked on the CPU: every merged branch's own CPU checks pass on the merged tree, the logits equal `-ladder2`'s on every architecture with the switches at their defaults and off, and every kernel a branch changed or added comes out with that branch's sm_61 machine code. Not yet run on a GPU; [run/gpu_checks.sh](run/gpu_checks.sh) does that next, one named run per switch.

## What is merged

Fork branch `pascal-p106-v3-b12076-ladder3` at `7bc1552f4`: thirteen merge commits on `pascal-p106-v3-b12076-ladder2`
(`a330672c1`), each a branch tip as it stood at the merge (later commits on a branch merge on top), in an order that
keeps the kernel branches together and the speculative-loop branches last (`-taper`, pushed after the others, last):

| Order | Merge | Branch, tip (cut from) | What it adds | Its folder | Switches (defaults unchanged) |
|---|---|---|---|---|---|
| 1 | `9bcbfee3f` | `-kcol` `af46311af` (C70's `e1fa4c573`, in `-ladder2`) | Q4_0, Q5_0, IQ4_NL in the k-column kernels; BF16 shapes at 3–4 tokens (C84); one-token Q2_0 fused gate/up at one row per lane on short grids (C85) | [kcol_coverage](../2026-11-04_kcol_coverage/README.md) | `GGML_CUDA_MMVQ_PASCAL_KCOL_SKIP=q4_0,q5_0,iq4_nl`; `GGML_CUDA_MMVQ_PASCAL_Q2_0=2` forces two rows; `GGML_CUDA_MMVQ_PASCAL_NSM` |
| 2 | `ff6b713e0` | `-iqkcol` `0fd6ce5c1` (`-iq2xs` `6c88b6991` on `-ladder2`, plus `-kcol` `af46311af`) | Pascal one-token kernels for IQ2_XXS, IQ2_XS, IQ3_S (C2); the six grid types and IQ1_M in the k-column and multi-token expert kernels, MUL_MAT_ID to 8 tokens for them (C163) | [iq2xs_kernel](../2026-11-04_iq2xs_kernel/README.md), [iq_kcol](../2026-11-04_iq_kcol/README.md) | `GGML_CUDA_MMVQ_PASCAL_SKIP=iq2_xxs,iq2_xs,iq3_s`, `_KCOL_SKIP=<types>`, `_MTOK_SKIP=<types>` |
| 3 | `c96d4a575` | `-ccflags` `9e1a8492e` | launch-bound minimum blocks as macros; `GGML_CUDA_PASCAL_FLAGS`, extra nvcc options for the Pascal kernel files (C131) | [ccflags](../2026-11-04_ccflags/README.md) | build time only: `-DKCOL_MIN_BLOCKS=…` etc. (defaults as before) |
| 4 | `298773a4d` | `-fap` `ff8e1ec24` (`-ladder` `98bbd6a2f`) | FAP's transpose-reduce without a stack frame; a 1,024-key Flint perf case (C135) | [fap_spills](../2026-11-04_fap_spills/README.md) | `GGML_CUDA_FATTN_PASCAL=0` (FAP off by default, as before) |
| 5 | `b2101bf56` | `-qsa` `bdb902d95` (`-ladder` `98bbd6a2f`) | Pascal sparse decode attention for the QSA layers; a linear k-pool pad search; `LLAMA_QSA_INPUT_STATS` (C6, C23) | [qsa_sparse_attn](../2026-11-04_qsa_sparse_attn/README.md) | `GGML_CUDA_FATTN_SPARSE_PASCAL=0` (1, 2), `_NQ`, `_SPLITS`, `_NSM`, `_MIN_RATIO`, `_FAULT` |
| 6 | `cb547461b` | `-lightprof` `973bcf09d` | `GGML_CUDA_PROFILE_MODE=layer`, `sample=<N>`, `graph`: profiler modes that keep CUDA graphs and never synchronize (C142) | [light_profiler](../2026-11-04_light_profiler/README.md) | only with `GGML_CUDA_PROFILE=<file>`; `group` (the default) as before |
| 7 | `c34625f1c` | `-hops` `5b69b0574` | the residual cast to F16/BF16/Q8_0 across card boundaries; a staged copy through pinned host slots (C200, C201) | [card_hops](../2026-11-04_card_hops/README.md) | `LLAMA_HOP_CAST`, `_AT`; `GGML_CUDA_HOP_STAGED`, `_CHUNKS`, `_SLOTS`, `_SLOT_KB`, `_STREAM` (all off) |
| 8 | `9334adfb7` | `-vgraphs` `a476ab83f` | upstream #31156 (graph arenas and schedulers per batch size) and #31274 (scheduler scratch per graph, calloc'd graph metadata) (C151) | [vgraphs](../2026-11-04_vgraphs/README.md) | `LLAMA_GRAPH_PER_SIZE=1`, `LLAMA_SCHED_PER_SIZE=0`, `LLAMA_GRAPH_META_CALLOC=1` (the PR's defaults; `LLAMA_GRAPH_PER_SIZE=0 LLAMA_GRAPH_META_CALLOC=0` = `-ladder2`) |
| 9 | `2454d6db5` | `-bytesw` `fd549b754` (`-draftmem` `1e0caf230`, in `-ladder2`) | fewer routed experts on confident tokens; a shortlist head with an exact bound (Q618, Q619) | [bytes_switches](../2026-11-04_bytes_switches/README.md) | `LLAMA_MOE_TAIL`, `LLAMA_HEAD_BOUND` (off) |
| 10 | `3e7e044e8` | `-ckptrc` `3e49981fb` | upstream #30137, reference-counted checkpoints; #31569 + #31610, a fit that counts the draft (C154, C155) | [ckpt_refcount](../2026-11-04_ckpt_refcount/README.md) | `--ctx-checkpoints-share` (off); `LLAMA_FIT_DRAFT_CTX=0`, `LLAMA_FIT_NEXTN_SKIP=0` give the old fit |
| 11 | `11f98affb` | `-specdump` `dafb09a4a` | one JSON line per speculative verify step, DFlash2's lattice included (Q688) | [tree_replay](../2026-11-04_tree_replay/README.md) | `LLAMA_SPEC_DUMP=<file>` (off) |
| 12 | `6da8ee8b0` | `-adraft` `b52064398` | the draft length chosen per step from the drafter's own probabilities (Q687) | [adaptive_draft](../2026-11-04_adaptive_draft/README.md) | `LLAMA_SPEC_ADAPT=1`, `LLAMA_SPEC_ADAPT_LOG=<file>` (off) |
| 13 | `7bc1552f4` | `-taper` `7b1632aee` | a taper of prompt micro-batches: smaller ones where a call fills or drains the layer-split pipeline (C218, Q771) | [prompt_taper](../2026-11-04_prompt_taper/README.md) | `LLAMA_UBATCH_TAPER`, `--ubatch-taper`, a request's `"ubatch_taper"` (off) |

**The merge is the union of the branches' changes** (`checks/union` in `run/cpu_checks.sh`, [union.tsv](union.tsv)).
Of the 75 files changed against `-ladder2`, 46 are touched by one branch; 45 of those are byte for byte that branch's
file, and the 46th (`tests/test-llama-archs.cpp`, `-bytesw`'s, cut from `-draftmem`) differs only by what `-ladder2`
added to it since. In every file, the merged file's changed lines against `-ladder2` are exactly the union of the
branches' changed lines (sorted line sets), with three explained exceptions:
- `mmvq-pascal-kcol-impl.cuh`: four lines `-kcol` `af46311af` added and `-iqkcol` later rewrote (the type enum, the
  record-order enum, two comments); the merged file is `-iqkcol`'s plus `-ccflags`' macro, line for line.
- `qwen4exp.cpp`: `-qsa`'s `#include <cstdlib>`, which `-ladder2` already has (from `-gdnfuse`).
- `llama-context.cpp`: the three calls below, where `-bytesw` meets `-vgraphs`.

### Where branches meet

Outside the speculative loop, conflicts in five files, each two additions at the same place, kept both:
- `src/models/qwen4exp.cpp` (`-qsa` into the tree with `-gdnfuse`): both added `#include <cstdlib>`; kept once.
- `src/llama-context.cpp` (`-bytesw` after `-hops`): `LLAMA_HOP_CAST`'s block and `LLAMA_MOE_TAIL`'s block both follow
  the screened-head block in the constructor; kept both (they read separate fields).
- `-taper` last: in `llama-context.cpp` the per-size graph switches and then `LLAMA_UBATCH_TAPER` in the constructor,
  `-vgraphs`' scheduler reset and then the taper's per-call hints at the top of `decode()`; in `server-context.cpp`
  `-bytesw`'s per-batch head-bound switch and then the taper's tail hint; `tests/CMakeLists.txt`. The taper's plans keep
  micro-batches of 9 tokens or more, so they use the shared graph arena, not `-vgraphs`' per-size ones (1–8 tokens).

The speculative loop (`-specdump` and `-adraft`, then `-vgraphs`'s per-size graphs under it) conflicted in four files,
every hunk two additions at the same place:
- `common/speculative.h`: `common_speculative_draft_params` gains `info` (the dump) and then `result_p` (the rule);
  the server's aggregate initializer lists them in that order.
- `common/speculative.cpp`: each drafter's push of a drafted token calls `spec_info_add_marginal` (dump) and then
  `spec_push_p` (rule). DFlash2's selector walk keeps `-specdump`'s "past the draft: walk only" step first; the softmax
  runs for `p_min`, sampling or `result_p` (`-adraft`'s condition); after the token the walk index (dump) and then the
  selector's probability (rule) are recorded, before the lattice info closes the position loop.
- `tools/server/server-context.cpp`: the dump's and the rule's statics and per-step functions side by side; at draft
  setup both `spec_info` and `spec_draft_p` are prepared. `tests/CMakeLists.txt`: both tests.

One change beyond the union, where `-bytesw` meets `-vgraphs`: `-bytesw` reads two of the last graph's outputs back
(`moe_tail_collect`'s per-layer kept counts and the head bound's input row `t_head_h`) and looked their backend up, and
synchronized, on the shared scheduler. `-vgraphs` moved every such lookup to `sched_res()`, the scheduler that computed
the graph; those three calls now use it too. With `LLAMA_SCHED_PER_SIZE` at its default `0`, `sched_res()` is the shared
scheduler and nothing changes; with `1`, the lookups find the tensors in the per-size scheduler that holds them instead
of failing their assert.

Clean merges whose files meet: `-ccflags` after `-iqkcol` (the new kernels take the macros: the k-column grid-type
kernels share `mul_mat_kcol`'s `KCOL_MIN_BLOCKS`, the one- and 2–4-token grid-type kernels `mmvqp_min_blocks(_mc)`, the
multi-token grid kernel keeps its own `MMVQP_MTOK_IQ_MIN_BLOCKS`); `-fap` after `-ccflags` (`FAP_MIN_BLOCKS` stays in
the launch bounds of the kernel whose reduce `-fap` rewrote); `-hops` and `-lightprof` in `ggml-cuda.cu`; `-ckptrc`
and `-bytesw` in the server and `common/`. `-iqkcol` already holds `-kcol` `af46311af`, so that merge adds only
`-iq2xs` and `-iqkcol`'s own commits.

### Precedence where features act on the same thing

| Where | Order, all switches set |
|---|---|
| Attention (`fattn.cu`) | FAP (`GGML_CUDA_FATTN_PASCAL`, head 128 × 8: Flint) → the sparse gather (`GGML_CUDA_FATTN_SPARSE_PASCAL`, QSA layers) → the Pascal tile kernel → stock; FAP and the gather take disjoint shapes |
| Dense and expert products (`ggml_cuda_mul_mat_vec_q`) | as `-ladder2`: k-column (no ids, 1–8 columns; now also Q4_0, Q5_0, IQ4_NL, the grid types, IQ1_M) → multi-token experts (Q2_0 and now the grid types and IQ1_M, 2–8 tokens) → the one-token Pascal kernel (now IQ2_XXS, IQ2_XS, IQ3_S too); each `_SKIP` list leaves its types to the next |
| The speculative step | the drafter fills the draft, `spec_info` (dump) and `spec_draft_p` (rule) → `LLAMA_SPEC_ADAPT` cuts the draft to the rule's prefix → the verify pass, in the per-size graph of its width (`LLAMA_GRAPH_PER_SIZE`, `LLAMA_SCHED_PER_SIZE`) → `LLAMA_SPEC_DUMP` writes the step as verified (`draft` = the kept prefix, `cand`/`p` = the whole proposal); a step cut to nothing is neither verified nor dumped. The head bound (`LLAMA_HEAD_BOUND`) acts only on one-output decodes, never on a verify pass |
| A hop between cards | `LLAMA_HOP_CAST` narrows the residual in the graph (a cast on each side of the boundary) → the scheduler copies the narrow tensor → the CUDA copy: the driver's peer copy, or the staged ring with `GGML_CUDA_HOP_STAGED` (prompt-sized hops on the destination's copy stream behind the scheduler's guard event). A per-size scheduler (`LLAMA_SCHED_PER_SIZE=1`) has no events and synchronizes the destination before its input copies, so a staged hop finds it idle |
| A prompt call | llama-server's calls and checkpoint cuts (unchanged) → `LLAMA_UBATCH_TAPER`'s plan of micro-batch ends inside the call (micro-batches of 9 tokens or more; the recurrent tail rule on top) → each micro-batch in the shared graph arena (the per-size arenas serve 1–8 tokens) → its hops at the card boundaries as above |
| Profiling | `GGML_CUDA_PROFILE_MODE` times launch groups inside graph evaluations; staged hops and the scheduler's copies run between them and are not timed |

### `-ladder2`'s behaviour on `-ladder3`

Every switch above at off reproduces `-ladder2` on the host side; the kernel branches change what compute 6.1 runs by
default. The closest `-ladder2` on one binary:

```bash
LLAMA_GRAPH_PER_SIZE=0 LLAMA_GRAPH_META_CALLOC=0 LLAMA_FIT_DRAFT_CTX=0 LLAMA_FIT_NEXTN_SKIP=0 \
GGML_CUDA_MMVQ_PASCAL_SKIP=iq2_xxs,iq2_xs,iq3_s \
GGML_CUDA_MMVQ_PASCAL_KCOL_SKIP=q4_0,q5_0,iq4_nl,iq2_xxs,iq2_xs,iq2_s,iq3_xxs,iq3_s,iq1_m \
GGML_CUDA_MMVQ_PASCAL_MTOK_SKIP=iq2_xxs,iq2_xs,iq2_s,iq3_xxs,iq3_s,iq1_m GGML_CUDA_MMVQ_PASCAL_Q2_0=2 llama-server ...
```

Two differences stay: C84's BF16 shapes at 3–4 tokens have no switch (`GGML_CUDA_MMVF_PASCAL_SHAPE=G,R` fixes one shape
for every call; Q784 adds an off switch), and `GGML_CUDA_MMVQ_PASCAL_Q2_0=2` forces two rows on every one-token Q2_0
product, which `-ladder2` chose per shape. A rig row that compares a switch on and off runs both arms on this binary.

## Checks on the merged tree (no GPU)

All in containers without a GPU on the build machine, from [run/cpu_checks.sh](run/cpu_checks.sh); one line per check
in [cpu_checks.tsv](cpu_checks.tsv) (52 OK; the union check says REVIEW for its three explained exceptions).

| Check (branch) | Result on `7bc1552f4` |
|---|---|
| Build | rig tarball `TESTS=1`, sm_61 + compute_61 PTX, 0 AVX, GLIBC_2.38; the extra tests from the same tree; a static CPU build |
| Union of changed lines | 75 files; 3 with lines outside the union, each explained above ([union.tsv](union.tsv), [union_exceptions.txt](union_exceptions.txt)) |
| k-column decoding, C70 + C93 (`-kcol`) | the bench's header replaced by the tree's (they differ by `-ccflags`' macro alone): 10/10 types PASS, worst 0.0044 of tol; the mutation fails exactly q4_0, q5_0, iq4_nl |
| IQ2_XXS/IQ2_XS/IQ3_S decoding, C2 (`-iq2xs`) | 3 × 400 cases PASS, worst 0.0016 of tol. `MUTATE=1` as published stops ("the mutations did not apply": since C163 one of its edits matches two lines); a copy that edits every site fails exactly iq2_xxs and iq3_s (Q782) |
| Grid types in the k-column and multi-token kernels, C163 (`-iqkcol`) | 6 × (600 + 200) cases PASS, worst 0.0065 of tol, the multi-token results bit for bit the one-token functions'; `MUTATE=1` fails exactly iq2_s, iq3_s, iq1_m |
| Sparse decode attention, C6 (`-qsa`) | `bench/fattn_sparse/check.sh` on the tree: every case passes, the faults and the four mutations fail; `cpu_e2e.sh`: the qwen4exp rows identical to `-ladder2`'s (logits OK 0.00, roundtrip OK, chunked against one batch 3.24e-14), `test-llama-context` 19 passed / 0 failed, the QSA input-stats lines logged |
| FAP's reduce, C135 (`-fap`) | the emulated warp: both forms bit for bit on 640,000 results, the mutation fails |
| Per-kernel SASS (every kernel branch) | 7,591 kernels: 7,262 as `-ladder2`, every kernel a branch changed carried with its machine code (`-kcol` 51, `-iq2xs` 63, `-iqkcol` 324, `-qsa` 4, `-fap` 1); the host-only branches change none; [sass_summary.txt](sass_summary.txt) |
| Logits against `-ladder2` | `test-llama-archs -b CPU`, 136 architectures, rig builds and CPU builds, at the defaults and with the merged switches off: all bit for bit ([identity_rig.tsv](identity_rig.tsv), [identity_cpu.tsv](identity_cpu.tsv)); the test's 7 failures in every run, `-ladder2`'s included, are its own (a second model under the same dump name) |
| GDN fusions, C19/C20 | fused = unfused, 0 of 32,768 logits differ, n_rs_seq 0 and 3 |
| Draft memory, Q564 | `test-draft-mem` 82 checks, 0 failed |
| Device stores, C22/C28 | `test-llama-context` on qwen4exp, qwen35, mamba: 54 passed / 6 skipped / 0 failed, `store` and `store_p` OK; C22's six server arms PASS |
| Per-size graphs, C151 (`-vgraphs`) | `test-backend-ops -b CPU` 26,576/26,576; `test-opt`, `test-alloc`; `test-llama-context` 54/0/6 at all four settings; `test-llama-archs` 148/148 at the default and `LLAMA_SCHED_PER_SIZE=1`; `spec_check.sh` PASS (every arm base's tokens and top probabilities, verify widths 1–4); C22's arms PASS |
| Card hops, C200/C201 (`-hops`) | CPY 555/555 (the 18 hop shapes); round trips 18/0; the random qwen4exp 55/0; identity against `-ladder2`: off and staged bit for bit on 136 architectures, the q8_0 cast at layer 1 changes only qwen4exp; `test-llama-context` 19/0; the server arms PASS |
| Bytes switches, Q618/Q619 (`-bytesw`) | 4/4 arms (F32 and Q5_K heads, as is and zero-tailed): 231–232 checks each, 0 failed |
| Shared checkpoints and the fit, C154/C155 (`-ckptrc`) | `test-llama-context` on three models, C22's arms with and without `--ctx-checkpoints-share` PASS, `server_mem.sh`, `fit_check.sh`: fail=0 |
| Spec dump, Q688 (`-specdump`) | its own CPU build of the merged commit; `test-spec-dump` 155/0; `tree_replay.py --selftest` 16/0; smoke: same tokens with and without the dump, 37 steps, 0 problems |
| Draft length per step, Q687 (`-adraft`) | its own CPU builds (merged commit, `-ladder2`); `test-spec-adapt` 13,507/13,507; off/log/on-all = the base's tokens and draft counts 9/9; greedy replies = plain decoding 16/16; logs check 6/6; a bad cost table stops the server |
| Prompt taper, C218 (`-taper`) | unit (parser, 13 plans, 4,000 schedules) OK; `test-batch-alloc`; the random qwen4exp 56 comparisons, 0 failed; identity: off bit for bit on 136 architectures, `short=4096:9` within each architecture's own one-batch-against-steps nmse (largest 8.0e-7, gemma4); `test-llama-context` 19/0 off and on; the server arms = `-ladder2` (tokens, top-3, `prompt_n`, `cache_n`, checkpoints); the replay dry run |
| Light profiler, C142 (`-lightprof`) | `test-cuda-profile-light` 49,267/0; the rig script's dry run passes |
| Compiler flags, C131 (`-ccflags`) | `GGML_CUDA_PASCAL_FLAGS` reaches 30 CUDA files, the tree's 30 Pascal kernel files by its glob (not `fattn-sparse-pascal.cu`: Q783), 0 without it; a file compiles with it |

## The GPU checks: [run/gpu_checks.sh](run/gpu_checks.sh)

One script for every switch, one named line per run in `console.txt` (`part/run`, the switch, the summary, FAIL lines,
the exit status, what passing means), so a failing switch is found by name. Parts, in order: `ladder2` (-ladder2's own
script: BF16, Q2_0, `-iqmmvq`'s grid types, the k-column and multi-token kernels, the GDN fusions, the persistent MoE
self-check, every architecture, the device checkpoint stores, the whole suite through the PTX), `kcol` (C70's and C93's
rig scripts, correctness parts), `iq2xs`, `iqkcol` (the branches' own scripts and `mmvq_kcol --check-only`), `qsa`
(`qsa_gpu.sh`, parts check fault fasp), `fap` (FAP on, its three faults, off), `hops` (CPY at the hop shapes, the round
trips, and on two cards the staged hop and the casts across a real layer split, as C200g), `vgraphs` (`spec_check.sh`
on the card, `test-llama-context` at the four settings), `ckpt` (C22's `server_check.sh` on the card, without and with
`--ctx-checkpoints-share`), `server` (one llama-server per host switch on the card, greedy tokens against the switch
off).

- **Through the compute 6.1 PTX on another NVIDIA GPU** (an RTX 3090 here): `MODE=ptx`, inside
  `nvidia/cuda:12.9.2-devel-ubuntu24.04` with two cards (`--gpus '"device=<a>,<b>"'`; inside they are 0 and 1), curl and
  jq installed, est. 2–3 h (the whole-suite sweeps of the `ladder2` part are most of it; `PARTS`, and `-ladder2`'s
  `RUNS`/`SKIP`, split it across containers that run side by side). The rig build holds only sm_61 code and compute_61
  PTX, so the card JIT-compiles that PTX by itself; `CUDA_FORCE_PTX_JIT=1` is left off (`PTX_JIT=0`), because it would
  also force cuBLAS onto PTX that cuBLAS does not ship, and every case cuBLAS takes would abort its run (Q876). The
  stock arms (`ladder2`'s `iq-off` and `all-off`, `iqkcol`'s `off` and `skip`) leave out the grid types' cases with
  repeated expert ids at 5–8 tokens: llama.cpp's MMQ takes them there and does not support repeated ids (Q875;
  [run/probe_dup_ids.sh](run/probe_dup_ids.sh) isolates the case). Natively both run as before.
- **On P106-090s natively:** `MODE=native` on a rig alone, cards 0 and 0,1; `PARTS` splits the run between two rigs
  (est. `ladder2 fap hops vgraphs ckpt server` ~3.5 h, `kcol iq2xs iqkcol qsa` ~2.5 h on the rig's Celeron).
- **Card choice:** `DEV`/`DEVS2` in PCI bus order; by default the first entry (and first two) of
  `CUDA_VISIBLE_DEVICES` when it is set, so `CUDA_VISIBLE_DEVICES=0,1,2,4,5,6` keeps every run on those six cards.

```bash
# PTX, from this repository's root on the build machine (paths are the reader's)
docker run --rm --gpus '"device=0,1"' -v "$PWD":/r:ro -v "$WORK":"$WORK" -v "$BENCHES":"$BENCHES":ro \
  -e MODE=ptx -e BIN=$WORK/<tree>/bin -e OUT=$WORK/gpu-ptx -e MODELS=$WORK/checks/ctx/models \
  -e KCOL_BENCH=... -e MTOK_BENCH=... -e KCOL_BENCH2=... -e MMVQ_PASCAL=... -e FASP=... nvidia/cuda:12.9.2-devel-ubuntu24.04 \
  bash -c 'apt-get update -qq && apt-get install -y -qq curl jq > /dev/null; bash /r/docs/data/2026-11-04_ladder3_merge/run/gpu_checks.sh'
# natively on a rig, one half per rig
MODE=native BIN=<tree>/bin OUT=<dir> MODELS=<dir> RUN=<docs/data copy> PARTS="ladder2 fap hops vgraphs ckpt server" \
  KCOL_BENCH=... MTOK_BENCH=... KCOL_BENCH2=... MMVQ_PASCAL=... FASP=... bash gpu_checks.sh
```

The bench binaries: C70's `mmvq_kcol` and R7's `mmvq_pascal` (the `ladder2` part, as -ladder2's handoff), C163's
`mmvq_kcol` built with the IQ types (`KCOL_BENCH2`), C93's `mmvq_pascal` (`MMVQ_PASCAL`), C6's `fasp_check` (`FASP`);
each from `bench/` as its folder says. `MODELS` holds the random models `test-llama-archs -o` writes (the `ctx` check
leaves them in `checks/ctx/models`).

## Files

| File | What |
|---|---|
| `run/cpu_checks.sh` | the build and every check above, CPU only, in containers without a GPU |
| `run/gpu_checks.sh` | every switch's GPU check, `MODE=ptx` or `native` |
| `union.tsv` | per changed file: the branches that touch it, lines outside the union, single-branch identity |
| `union_exceptions.txt` | the changed lines outside the union, per file |
| `cpu_checks.tsv` | one verdict per check of `run/cpu_checks.sh` |
| `decode.txt` | the kernels' host decoding checks and their mutations, the header diff |
| `sass_summary.txt`, `sass_kernels.tsv.gz`, `sass_carried.tsv.gz` | the per-kernel SASS comparison (`sass_identity.py` of the `-ladder2` folder) |
| `identity_rig.tsv`, `identity_cpu.tsv` | per architecture and run: logits against `-ladder2`'s (nmse, largest difference, values that differ) |
| `context_cases.tsv`, `context_runs_vgraphs.tsv` | `test-llama-context` per case (C22's run) and per run and setting (C151's run) |
| `server_verdicts.txt` | the llama-server checks' verdicts (C22's arms in four runs, C151's spec arms, C200's, C218's) |
| `gpu_dry_run_server.txt` | `run/gpu_checks.sh` `PARTS=server` in a container without a GPU (the CPU runs the arms): the harness end to end |

## Reproduce

```bash
# a llama.cpp clone with the fork as remote "fork" (https) in $WORK/src, the branches' tips fetched
WORK=<dir> REPO=<this repository> BASE_BUILD=<-ladder2's unpacked rig tarball, extras in bin/> \
  BASE_CPU=<-ladder2's static native CPU build: llama-server, test-llama-archs> \
  SASS_REFS='ladder2=<dir> ladder=<dir> kcolold=<dir> draftmem=<dir> kcol:kcolold=<dir> iq2xs=<dir> iqkcol=<dir>
             qsa:ladder=<dir> fap:ladder=<dir> bytesw:draftmem=<dir> ccflags=<dir> vgraphs=<dir> specdump=<dir>
             adraft=<dir> ckptrc=<dir> hops=<dir> lightprof=<dir>' \
  JOBS=6 DOCKER_ARGS="<labels>" bash docs/data/2026-11-04_ladder3_merge/run/cpu_checks.sh
```

Each `<dir>` is that branch's unpacked rig tarball (`scripts/build_llamacpp_docker.sh`, `TESTS=1`); the branches'
handoffs name them. Provenance: the fork's commits above; the random models are `test-llama-archs`' own (seed 1 or 42);
no model files or data with a licence of their own are used. Code under llama.cpp's MIT licence.
