# C831 (Q1462): Miner Llama's v4 line on upstream llama.cpp b12360 (2026-11-13)

Before the v4 release candidate is tagged, the v4 line of Miner Llama (`pascal-p106-v4`, `80c94fe0d`, which holds
every v4 feature behind its switch, off by default) takes upstream llama.cpp's newest release, b12360 (`2d8a32502`,
114 upstream commits after the b12246 that C662 merged, [2026-11-10_ladder6](../2026-11-10_ladder6/README.md)). One
merge commit on the new branch `pascal-p106-v4-b12360-up`; every fork switch keeps its default. The registry rows are
Q1462 (this merge) and Q1463 (its checks on a P106) in [quants/experiments.md](../../../quants/experiments.md).

**Result:** the branch builds for the rig, every CPU check passes, and `pascal-p106-v4` is fast-forwarded to it
(`da2920d45`, 2026-11-13 12:55). With no switch set the merged tree computes on a CPU what `80c94fe0d` computes once
upstream's #29810 (the CPU's attention accumulating V in F32) is undone: all 146 random models bit for bit; with it,
143 differ at rounding level (Flash-Next's qwen4exp: nmse 4.718e-13). The line's 44 switches on together change nothing
on either build. The fork's own checks pass (10 of 11 `test-llama-archs` modes; the 11th passes with C740's known
plamo2 abort left out, as on `80c94fe0d`), and llama-server's MTP drafting gives every arm its reference and
`80c94fe0d`'s tokens and drafts. ctest passes but for two upstream tests the test machine cannot run. The sm_61 code of
every fork kernel is unchanged; the one changed code a P106 runs is upstream's `mul_mat_q` (prompt micro-batches past 8
tokens), which R153 (Q1463) runs on a P106 with the GPU checks the quant cards could not run today. `v4-rc1` is not
tagged here: it is the next step.

**On a P106 (R153, below): `v4-rc1` is correct.** Its changed `mul_mat_q` passes every case past 8 tokens of the
`kshape`, `flashnext-moe` and `n2p` suites (3,969 case replays), and `80c94fe0d`, replaying the same records, gives
the same errors on all 3,938 cases both hold. C662's one-card set passes natively as it did through the PTX, the fork
kernels' sets `v4kern` and `ro8kt` pass with every planted fault caught, and the card's logits differ from
`80c94fe0d`'s on the same 143 random models as on the CPU, where C831 traced the difference to upstream's #29810.

## The branch

`pascal-p106-v4-b12360-up` at **`da2920d45`**, one merge commit on `pascal-p106-v4` `80c94fe0d` (the second parent
is upstream's tag b12360, `2d8a32502`): 548 files change against `80c94fe0d`. 14 files conflicted, in 28 hunks. The
merge commit also holds the fixes that upstream's API changes needed in files without a conflict (below), so the
branch builds at every commit.

## The conflicts

| File | Hunks | Upstream's side | The fork's side | Resolution |
|---|---:|---|---|---|
| `ggml/include/ggml-rpc.h`, `ggml/src/ggml.c` | 2, 2 | [#29801](https://github.com/ggml-org/llama.cpp/pull/29801) removes three ops (`ADD1`, `DIAG_MASK_INF`, `DIAG_MASK_ZERO`): 99 ops, RPC protocol major 10 | the fork's four ops (`GDN_CONV`, `KQ_MASK`, `MOE_TAIL`, `RANGE_MASK`): 106 ops, major 109 (stock 9 + 100), patch 4 | 103 ops (99 + 4), **major 110** (stock 10 + 100), patch 4. The op numbers from `GGML_OP_UNARY` on moved, so a build of this branch and an older v4 or v3 build refuse each other over RPC |
| `ggml/src/ggml-cuda/common.cuh` | 3 | [#32001](https://github.com/ggml-org/llama.cpp/pull/32001): the CUDA graph and concurrent-stream state move to a new `graph.cuh`; the context's `cuda_graph()` and constructor move out of line; `concurrent_stream_context` becomes a `unique_ptr` | the fork's fields of `ggml_cuda_graph` (C779/C826's patch templates, chains, groups and node kinds; the profile's phase sets; `capture_gen`) and of the context (`graphs_force_warm`, the Q8_1 row cache, the persistent MoE kernel's state, the Pascal attention kernels' counters, the card hops) | upstream's layout. The fork's `ggml_cuda_graph` fields go into `graph.cuh`'s struct, the profile's phase sets as sets of upstream's `ggml_cuda_graph_instance` ([#31284](https://github.com/ggml-org/llama.cpp/pull/31284)); the context keeps every fork member |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | 5 | #32001's out-of-line context functions; [#31959](https://github.com/ggml-org/llama.cpp/pull/31959): `ggml_cuda_should_use_mmvf` takes the tensor; [#32004](https://github.com/ggml-org/llama.cpp/pull/32004): a reused CUDA graph skips the compatibility check; [#28413](https://github.com/ggml-org/llama.cpp/pull/28413): the `NATIVE_FP8` feature | the card hops (`GGML_CUDA_HOP_STAGED`); the Pascal BF16 path (`mmvf-pascal.cu`) and the F32 multi-token path (`GGML_CUDA_MMVF_F32_MT`) around the MMVF dispatch; the profile's phase choice and the persistent MoE kernel's check mode before the graph check; the `MMVQ_PASCAL`, `MMVF_PASCAL` and `COMPUTE_6_1` features | both, in each hunk: the fork's paths call the new `ggml_cuda_should_use_mmvf`; the graph check is skipped for a reused graph and still never runs in the check mode |
| `ggml/src/ggml-cuda/mmvf.cu`, `mmvf.cuh` | 1, 1 | #31959's signature | C754/C807/C822's `ggml_cuda_mul_mat_vec_f_f32_mt` | the fork's function, calling the new signature. #31959's own change (MMVF for BF16 matrices of few rows) needs MMF, which a Pascal card lacks: there small BF16 matrices already went to MMVF |
| `common/speculative.cpp` | 1 | [#30565](https://github.com/ggml-org/llama.cpp/pull/30565) moves the NextN API into `llama.h`, renames `llama_set_embd_state_src` to `llama_set_ctx_state_src`, and drops the `llama-ext.h` include | C703's two halves of the hidden-row buffer (`LLAMA_MTP_CATCHUP_OVERLAP`) before the buffer is made | the fork's halves with upstream's call; the include stays (the fork's MTP functions are there) |
| `src/llama-ext.h` | 1 | #30565: the NextN declarations leave this header | the fork's declarations after them | upstream's in `llama.h`, the fork's in `llama-ext.h` |
| `src/llama-context.cpp`, `src/llama-context.h` | 5, 1 | [#32009](https://github.com/ggml-org/llama.cpp/pull/32009): a re-reserve keeps the scheduler when its parameters allow (`sched_params`); [#32089](https://github.com/ggml-org/llama.cpp/pull/32089): the graph results are made once in the constructor and reset at each reserve; #30565: `set_warmup` removed, `set_state_src` renamed | the per-size graph arenas (`gf_res_key`, `gf_res_invalidate`, `LLAMA_GRAPH_PER_SIZE`, `LLAMA_MTP_GRAPH_KEEP`), the packed input upload at the scheduler's creation, C812's single-split scheduler (`LLAMA_PP_MULTI_SPLIT`) | **#32009 taken**: `sched_params_cur` is set after the fork's input upload, and C812's call passes the parameters. **#32089 not taken**: its two fixed results do not fit the fork's arenas, which are made per batch size on demand; upstream's constructor allocation and its reset loop are left out (no change at decode time). The fork's methods keep upstream's rename |
| `src/llama-graph.cpp`, `src/llama-graph.h` | 3, 1 | [#32043](https://github.com/ggml-org/llama.cpp/pull/32043): the graph metadata reserve from `new uint8_t[]`, its pages untouched; [40f4f4ac9](https://github.com/ggml-org/llama.cpp/commit/40f4f4ac9): every sampling chain's tensors an output; [#28392](https://github.com/ggml-org/llama.cpp/pull/28392): the KV scales of an FP8 cache moved the memory context up in `build_attn` | the fork's port of the same metadata change (#31274, `calloc`, `LLAMA_GRAPH_META_CALLOC`); the loops of the selected chains, `t_embd_all`, the staircase drafter's taps; K expanded last (`LLAMA_ATTN_K_LAST`) | the fork's port kept (it holds the switch); both loops (setting an output twice is harmless); the fork's K-last expand after upstream's moved lines |
| `src/llama-kv-cache.cpp` | 1 | #28392: each KV layer carries the K and V scales, set in an aggregate | the fork's member-by-name init (the rotations, the card-built cell copy) | the fork's init with the two scales set (a member left out would be uninitialized) |
| `tests/CMakeLists.txt` | 1 | `test-kv-cache` | the fork's tests | both |

The `union` check lists every line the merge changes against `80c94fe0d` that is neither upstream's (its changes since
its merge base with `80c94fe0d`) nor the fork's ([union.tsv](union.tsv)): 15 files, 98 lines added and 83 removed,
which are the resolutions above and the fixes below.

**Fixes for upstream API changes in files without a conflict**, in the merge commit:
- [#32018](https://github.com/ggml-org/llama.cpp/pull/32018) drops the `ids` and `type_src0` arguments of
  `quantize_row_q8_1_cuda`: the persistent MoE kernel's activations (`ggml-cuda.cu`) and C793's GLU kernel
  (`mmvq-pascal-glu.cu`) call it without them.
- [#31896](https://github.com/ggml-org/llama.cpp/pull/31896) renames `GGML_PREC_DEFAULT` to `GGML_PREC_UNDEFINED` (the
  same value, 0): the persistent MoE kernel's router (`moe-persistent.cu`).
- [#32078](https://github.com/ggml-org/llama.cpp/pull/32078): `graph_max_nodes()` takes no token count; the per-size
  schedulers (`LLAMA_SCHED_PER_SIZE`) call it so.
- [#26048](https://github.com/ggml-org/llama.cpp/pull/26048) renames the test class `test_mul_mat_vec_fusion` to
  `test_mul_mat_fusion` and its op name `MUL_MAT_VEC_FUSION` to `MUL_MAT_FUSION`: the class rename is taken, but the
  cases keep the op names `MUL_MAT_VEC_FUSION` and `MUL_MAT_VEC_FUSION_W4A8`, which the fork's recorded checks and
  check scripts select with `-o` (an unknown name would select no case, silently).

## What upstream changed on Flash-Next's paths

Upstream's 114 commits, read for what they do on a P106-090 (compute 6.1) running Flash-Next with its MTP draft, and
what changes with no switch set.

| Upstream change | Path | On a P106, with no switch set |
|---|---|---|
| [#32078](https://github.com/ggml-org/llama.cpp/pull/32078) | qwen4exp, the recurrent layers | upstream's GDN is always the fused op (its graph-level chunked and autoregressive versions are gone); the fork's default path was the fused op already. A context's graph budget no longer grows with `-ub`: 32 nodes a tensor for the hybrid models. **Flash-Next's main context keeps its budget** (1,224 tensors: 39,168 nodes, which was already above 40 × 512 = 20,480 at the preset's `-ub 512`). **The MTP draft file** (34 tensors) **gets 1,088 nodes** instead of 40 × 128 = 5,120 at `--spec-draft-n-ubatch 128`: on a CPU a draft file of the same structure needs 162 nodes (the `draftmd` check below) |
| [#32009](https://github.com/ggml-org/llama.cpp/pull/32009) | the scheduler | a re-reserve (a new sampler chain, an embeddings switch, a LoRA) keeps the scheduler and its compute buffers when the new graph needs no more nodes and the same parallelism; before, every reserve made a new one |
| [#32004](https://github.com/ggml-org/llama.cpp/pull/32004), [#32001](https://github.com/ggml-org/llama.cpp/pull/32001), [#31284](https://github.com/ggml-org/llama.cpp/pull/31284) | CUDA graphs (on for compute 6.1 in Miner Llama, upstream's #27721) | a reused graph skips the compatibility walk over its nodes (less host time a step); the state moved files; the cluster-dimension check returns at once below Hopper |
| [#32100](https://github.com/ggml-org/llama.cpp/pull/32100) | the scheduler's early fetch (`GGML_SCHED_PREFETCH`, C693) | only inputs that earlier splits computed are fetched early. The early fetch never acted on Flash-Next's split (R92, Q1325); the switch is unchanged |
| [#31533](https://github.com/ggml-org/llama.cpp/pull/31533) | MMQ and MMVQ at 2–24 tokens (the verify passes) | a new int8 MMA kernel, from Ampere on only (`ampere_mma_available`, which also needs the build's highest architecture): **nothing changes on a P106**, nor through the compute 6.1 PTX on a newer card. MMVQ's L2 prefetch helper is renamed |
| [#32067](https://github.com/ggml-org/llama.cpp/pull/32067), [#26048](https://github.com/ggml-org/llama.cpp/pull/26048) | MMQ (prompt passes) | the stream-k fixup skips empty MoE tiles: **stream-k is off in every Pascal MMQ configuration** (286 of 286 in `mmq-config-pascal-dp4a.cuh`); the NVFP4 scale epilogue: NVFP4 only |
| [#31959](https://github.com/ggml-org/llama.cpp/pull/31959) | MMVF | BF16 matrices of few rows at small batches on MMVF where MMF would tile them: MMF needs tensor cores, so a P106 already ran them on MMVF (and the fork's Pascal BF16 kernel) |
| [#28413](https://github.com/ggml-org/llama.cpp/pull/28413), [#28392](https://github.com/ggml-org/llama.cpp/pull/28392) | types, attention | F8_E4M3 support in CUDA (MMVF, copies, attention vector kernels) and an FP8 KV cache with per-layer scales read from the model: new options; the presets' caches and models are unchanged |
| [#30285](https://github.com/ggml-org/llama.cpp/pull/30285), [#27997](https://github.com/ggml-org/llama.cpp/pull/27997) | attention | MMA attention for head sizes 40 and 72, an sm70 configuration: tensor-core kernels, not run on a P106 |
| [#31896](https://github.com/ggml-org/llama.cpp/pull/31896) | MUL_MAT precision | a model can ask for its activations' precision (a GGUF precision policy); CUDA refuses the products it cannot run so: none of the presets' models carry a policy |
| [#30542](https://github.com/ggml-org/llama.cpp/pull/30542), [#31919](https://github.com/ggml-org/llama.cpp/pull/31919) | the CUDA memory pool | the VMM pool reserves half the address space again only after a failed reservation; a debug pool behind `GGML_CUDA_POOL_DEBUG` (off) |
| [#29810](https://github.com/ggml-org/llama.cpp/pull/29810) | the CPU's attention | the CPU accumulates V in F32: the CPU's logits change for every model that runs attention there (the CPU checks below); no preset runs attention on the CPU |
| [#29801](https://github.com/ggml-org/llama.cpp/pull/29801), [#30565](https://github.com/ggml-org/llama.cpp/pull/30565) | the APIs | deprecated functions, three ops and `--defrag-thold` are gone (no preset passes it); RPC major 110 |
| [#32098](https://github.com/ggml-org/llama.cpp/pull/32098), [#32113](https://github.com/ggml-org/llama.cpp/pull/32113), [#32115](https://github.com/ggml-org/llama.cpp/pull/32115), [#32120](https://github.com/ggml-org/llama.cpp/pull/32120), [#32133](https://github.com/ggml-org/llama.cpp/pull/32133), [#32025](https://github.com/ggml-org/llama.cpp/pull/32025) | llama-server | a loopback connection's `Host` must name loopback or an allowed origin, but only when `--cors-origins` is not `*`: it is `*` by default and becomes `localhost` only with the server's tools or MCP, which no preset uses, so the agent check's `ssh -L` relay is unaffected. The other changes concern the server's tools, media URLs, the router's child instances, sleep (`--sleep-idle-seconds`) and MCP calls |

Nothing upstream changed in `common/speculative.cpp` but #30565's renames, and nothing in the MTP draft's graph,
`llama-server`'s speculation or the fork's Pascal kernels (MMVQ-Pascal, the k-column, multi-token, persistent MoE and
attention kernels).

## The checks

On the build machine's CPU only ([run/checks.sh](run/checks.sh): Docker at a low CPU share, `nice -n 19`, one job at
a time under a lock; no GPU). Three builds of the CPU code: the tip (`da2920d45`), the base (`80c94fe0d`), and the tip
with upstream's [#29810](https://github.com/ggml-org/llama.cpp/pull/29810) undone (`git apply -R`), to tell its change
of the CPU's numbers apart from everything else. Every verdict line is in [summary.tsv](summary.tsv).

| Check | Result |
|---|---|
| the merge's lines (`union`) | REVIEW, as meant: 548 files change against `80c94fe0d`; 15 hold lines that are neither upstream's nor the fork's (98 added, 83 removed), the resolutions and fixes above ([union.tsv](union.tsv)) |
| the builds | the tip's static CPU build of every target (123 binaries, 0 errors), the base's and the revert build's; the rig tarball (sm_61 code and compute_61 PTX, `TESTS=1` with `test-cuda-keepalive` and `test-llama-archs`, 0 AVX, GLIBC_2.38) and the base's rig build |
| `test-llama-archs -b CPU` on the 146 random models, every switch off, against `80c94fe0d`'s logits ([logits-by-model.tsv](logits-by-model.tsv)) | **with #29810 undone, all 146 give `80c94fe0d`'s logits bit for bit.** The tip itself: 143 of 146 differ (grok, mamba and mamba2 are the same bits), median nmse 1.281e-08, largest 7.319e-07 (gemma4); 77 are above the comparison's own bound (nmse 1e-8), which makes the run exit 1, while every model's own checks pass (151 of 151: against the CPU, the round trip, the mixed batch). **qwen4exp, Flash-Next's architecture: nmse 4.718e-13** (largest difference 1.695e-07). So every difference on a CPU is #29810's, which changes the CPU's attention only: no preset runs attention on the CPU |
| the line's switches on together (44: every host switch that computes what the defaults compute, and the CUDA ones, which change nothing on a CPU; the list is `ALL` in [run/checks.sh](run/checks.sh)) against each build's own logits | **SAME on 146 of 146, on the base and on the tip** |
| the fork's own `test-llama-archs` checks ([modes.tsv](modes.tsv)) | **10 of 11 pass** on the tip: `--kv-shift` 113/113, `--ngram-prefetch` 3/3, `--mtp-kv-only` 5/5, `--mtp-catchup-overlap` 5/5, `--mtp-hrow-lazy` 25/25, `--mtp-graph-keep` 8/8, `--mtp-replan` 4/4, `--verify-prepare` 272/272, `--mtp-h-dev` 5/5, `--mtp-incall-overlap` 5/5. `--kv-seq-rm` aborts at plamo2's 4-sequence test (`GGML_ASSERT` in `ggml.c`, a view past its source), on the base and the revert build alike: the case C740 recorded and left out ([2026-11-11_c740_seq_rm_bound](../2026-11-11_c740_seq_rm_bound/README.md)). With those two tests left out (`LLAMA_ARCHS_SKIP`, as C740 did) both builds pass 436 of 436 |
| llama-server's MTP drafting on the random qwen4exp model with an MTP block (C820's [server_check.sh](../2026-11-13_c820_online_skip_row/run/server_check.sh): C815's online rule, C742's and C820's skipped rows, C744's cap; [server-out.txt](server-out.txt)) | **every arm of the tip equals its reference**: 16 arms over 8 groups (d, w, r, b, p, v, n, c) × 4 requests, the same tokens, top probabilities, drafts, verify steps and rule steps; the rows left out as on `80c94fe0d` (e.g. `v`: =2 leaves out 408 of 408, 396 by the look-ahead). The base's arm `dbase` could not load the tip's random model: the tip writes the optional KV scales of #28392 (6 tensors, `blk.N.attn_k.k_scale` and `attn_v.v_scale` of its 3 attention layers), which `80c94fe0d` does not know. On the base's own random model (the two files differ by those 6 tensors only, the separate draft files by 2: `tensors-*.txt`), with the base's `dbase`: the revert build's `doff` equals it in everything (tokens, top probabilities, drafts, 200 verify steps, 407 rule steps; [server-brev.txt](server-brev.txt)); the tip's has the same tokens, draft counts and top-3 candidates on all four requests (G1, G2, S1, S2), and 157 of its 1,248 top log-probabilities differ by at most 4.768e-07: #29810's rounding ([server-btip.txt](server-btip.txt) fails on them) |
| a separate MTP draft file (`-md`, Flash-Next's draft settings: `--spec-draft-n-ubatch 128`, `-b 2048 -ub 512`) | OK on both builds: each loads the base's model with its MTP block as a separate draft and gives the same 96 greedy tokens and draft counts (549 drafted, 0 accepted: the random model's drafts are never accepted); the tip on its own model too. The draft context's graph has 162 nodes (168 with the sampler); its budget is 1,856 on the tip (32 × 58 tensors) against 5,120 on the base (40 × 128) ([draftmd-graphs.txt](draftmd-graphs.txt)). The tip's log also shows #32009 at work at a re-reserve (`sched_new: reusing scheduler`) |
| the switch list (C770) in the rig builds | the tip's build reads 880 names, `80c94fe0d`'s 872: upstream's `GGML_CUDA_POOL_DEBUG`, OpenCL, SYCL and Vulkan names and two server options are new, `LLAMA_ARG_DEFRAG_THOLD` and the op name `GGML_OP_ADD1` are gone; no fork switch is missing |
| the sm_61 code of every GPU function of `llama-server` against `80c94fe0d`'s rig build ([sass-sass-families.txt](sass-sass-families.txt), [sass-sass-by-function.tsv](sass-sass-by-function.tsv)) | REVIEW: of `80c94fe0d`'s 8,572 functions 6,681 have the same code, 55 the same code compiled into more files (the attention's fixup, combine and mask kernels, now also in the FP8 attention files), 1,834 changed and 2 are gone; 227 are new. **Every fork kernel is unchanged** (the Pascal matrix-vector, k-column, multi-token, read-once, persistent MoE, attention and sparse attention kernels). The changed ones are upstream's: `mul_mat_q` (722: #26048's write-back, which now scales by the NVFP4 weight scale only for NVFP4, a constant 1 otherwise), its stream-k fixup (896: #32067; **stream-k is off in all 286 Pascal MMQ configurations**, so a P106 never launches it) and the tensor-core attention `flash_attn_ext_f16` (216; never chosen below Turing). New: FP8 (#28413: MMVF, attention vector kernels, copies, quantization), few-row MMQ (#31533, Ampere and newer only); gone: `diag_mask_inf_f32` (#29801) and the old `pad_reflect_1d` kernel (#32179). **So the one changed code a P106 runs is `mul_mat_q`**, for products past 8 tokens that the fork's kernels decline (prompt micro-batches): R153 replays it on a P106. Through the compute 6.1 PTX also `flash_attn_tile` (10) and one `mul_mat_vec_f` differ, for the PTX checks on a newer card. The two `sass` lines of `summary.tsv` were written before `sass_diff.py` told copies apart (they count the 55 as changed) |
| C565's CPU test set, static and then shared (ctest: every registered test, the fork's unit tests among them; the test with network; the archs runs; the hand tests) | **OK, static and shared alike, but for two upstream tests the test machine cannot run**: ctest 73 of 75 (the fork's unit tests and `test-backend-ops` among the 73), the test with network 1 of 1, the archs runs 151 of 151 three times with 146 logits files the same, the nine hand tests. The two: `test-kv-cache-scales` (new, #28392) imports `torch`, which the test image lacks (as `test-conversion-fp8`, which the set already leaves out); `test-jinja-py`'s 14 new TTS-normalizer cases ([#32053](https://github.com/ggml-org/llama.cpp/pull/32053)) find their script through `__FILE__`, which the build's ccache base folder makes relative: run from the folder where it resolves, `test-jinja -py` passes 413 of 413 |

## On a P106 (R153, Q1463)

R153 ran the GPU checks natively on one P106-090 of a mining rig (card 0, 2026-11-13 13:15–15:44; the sets recorded 13:14–13:45), with the rig
tarballs of the tip (`da2920d45`, tagged `v4-rc1`) and of `80c94fe0d`; nothing is timed. The rig's 2-core CPU computes
no reference: every `test-backend-ops` run replays a record made on the build machine's CPU with the tip's own
`test-backend-ops` ([scripts/record_checks.sh](../../../scripts/record_checks.sh), `nice -n 19` in a container
without GPUs), so the card computes and the stored CPU results judge it.

**Upstream renamed the cases.** b12360 adds `offset_a` to every `MUL_MAT` case's parameters (`...,pad=0,offset_a=0`)
and renames the fused matrix-vector test class (#26048), and a record finds its case by the md5 of class and
parameters. So sets recorded before the merge do not fit the tip: the `v4kern` set of 2026-11-11 replays 33 of
`kern-all`'s 722 cases on it (689 not recorded), and its two fault runs that fail no case among those 33 are WRONG. Every
set R153 replays is recorded again with the tip's `test-backend-ops`. To give both builds the same stored inputs and
references, [run/alias_base.py](run/alias_base.py) writes each set again under `80c94fe0d`'s case names (hard links,
the tensors unchanged: at `offset_a=0` the view is built as before; it checks every source id against the md5 of its
key), and [run/compare_builds.sh](run/compare_builds.sh) joins the two builds' replays case by case by id.

**Part (1), `mul_mat_q` natively.** The record/replay suites `kshape`, `flashnext-moe` and `n2p`, cut to their cases
past 8 tokens (`EXTRA_P='n=(9|[1-9][0-9]+),'`), the products that reach `mul_mat_q`, the one changed code a P106 runs;
for `MUL_MAT_VEC_FUSION` `n` is the row count, so its cases at every token count are in. Each suite's runs that select
recorded cases ([run/runs_past8.sh](run/runs_past8.sh)) replay with both builds on the same card.

| Set | Cases (GB) | Runs | Tip | `80c94fe0d` (same record) | Case by case |
|---|---:|---:|---|---|---|
| `kshape` | 772 (11.9) | 7 of 18 (the six `mtok` runs, `mtok-iq`) | 7 of 7 ok: 1,872 case replays, 0 failed | 7 of 7 ok: 1,866, 0 failed | every case in both (1,866) the same verdict and the same errors; 6 cases only the tip's test list holds (`MUL_MAT_ID` IQ3_S, 37 experts), passed |
| `flashnext-moe` | 8 (1.96) | 2 of 2 | 2 of 2 ok: 16, 0 failed | 2 of 2 ok: 16, 0 failed | 16 of 16 the same errors |
| `n2p` | 1,313 (17.5) | 9 of 9 (`np-mmq` among them) | 9 of 9 ok: 2,081, 0 failed | 9 of 9 ok: 2,056, 0 failed | every case in both (2,056) the same verdict and the same errors, `np-mmq`'s 40 among them; 25 cases only the tip's test list holds (16 in `reg-half`, 9 in `reg-rows`), passed |
| all three | 2,093 (summed over the sets) | 18 | 3,969 case replays, 0 failed | 3,938, 0 failed | **3,938 of 3,938 the same errors** |

One `kshape` case is held twice by the test list (`MUL_MAT_ID` Q2_0, 4 experts, 2 used, 512 rows, 32 tokens: keys
`#0` and `#1`, two records): replayed five times with each build on card 1 ([run/case_repeat.sh](run/case_repeat.sh)),
each build gives both records' errors, 3.41459379e-08 and 3.5426209e-08, every time.

**Part (2), C662's one-card set natively** (the checks C662 ran through the compute 6.1 PTX:
[2026-11-10_ladder6](../2026-11-10_ladder6/README.md)), the tip's build:

| Check | Result |
|---|---|
| `test-llama-archs -b CUDA0` plain, the fork's paths forced on (`=2`), and the card-built mask checked ([check.sh](../2026-11-03_upstream_rebase/run/check.sh) `NATIVE=1`) | **OK**: 153 of 153 tests pass three times, 0 FAIL lines; 1,579 card masks identical to the host's (C662 through the PTX: the same counts) |
| the persistent MoE kernel's self-check and its planted fault, the fused norm (`MOE_LAYER`, `ADD_RMS_NORM_MUL_Q8`, replayed from the set `onecard`) | **OK**: the self-check 17 of 17 layers, 0 differing from the per-op path (17 of 17 cases pass); the planted routing error (`GGML_CUDA_PERSISTENT_MOE_FAULT=2`) reported in 17 of 17; the fused norm 32 of 32. Every case replayed, none unrecorded (C662 through the PTX: the same counts) |
| the -ladder4 merge's [gpu_checks.sh](../2026-11-04_ladder4_merge/run/gpu_checks.sh) `MODE=native`: skipped expert slots past 8 tokens, with the defaults, with `GGML_CUDA_MMVQ_PASCAL=0` and with `_MTOK=0`, the tail op and the fused router chain (replayed from `onecard`) | **OK**: past 8 tokens 110 of 110, with the defaults 633 of 633, with `GGML_CUDA_MMVQ_PASCAL=0` 633 of 633, Q2_0 with `_MTOK=0` 41 of 41, the tail op and the router chain 544 of 544; 0 FAIL lines, every case replayed (C662 through the PTX: 100, 604, 604, 41, 533; upstream's new cases since) |
| the same: `test-llama-archs` on the card with the tail op (`LLAMA_MOE_TAIL=p=0.8,renorm=1`) as the op, with `skip=0` and with `repeat=1`, against the graph-level ops | **OK**: 146 compared three times, 0 with a difference, exit 0, largest nmse 2.672e-08 (C662 through the PTX: 1.481e-06) |
| the same: the tiered random model with the cut at `-ub` 128 and 1 | **OK** both: the op = the graph-level cut bit for bit; KLD tiered against untiered 0.000000, the cut 0.397970 |
| the same: `test-llama-archs` on the card, every switch off, the tip against `80c94fe0d`'s logits | **REVIEW, as on the CPU**: 143 of 146 models differ, largest nmse 7.303e-07 (gemma4). The same 143 models differ in C831's CPU comparison (none differs on one side only), each by about the same amount (gemma4 7.319e-07 on the CPU, beam 5.942e-07 here and 5.943e-07 there; 62 of 146 the same nmse as printed), where C831 traced it to upstream's #29810. C662 through the PTX, against the v3.0 release: 141 of 143 |
| the fork kernels' sets, recorded again with the tip: `v4kern` (C796's three kernel suites and the three switches together) | **OK**: 47 of 47 runs as expected, 16,240 case replays: every case of the 41 passing runs passes; the 6 faults fail cases (`q6kmt` 42, `dmtkt` 5, `ro8` 193, and the same with all three switches on: 42, 5, 193) |
| `ro8kt` (C794's K tiles at 5–8 tokens) | **OK**: 16 of 16 runs as expected, 3,232 case replays: 202 of 202 in every passing run (the fault with the tiles off among them); the fault fails 34 |

**Not run, by the main session's word** (rig time; C831 found every fork kernel's sm_61 code unchanged since
`80c94fe0d`): gpu_checks.sh's `experts` run (every expert case without skipped slots, 5,121 cases, 42.2 GB to
record) and its `dense` part (`MUL_MAT` at 1–8 tokens, 4,308 cases, 15.5 GB), whose kernels are the fork's; and its
`moeids` part, as the release tarball holds no `test-moe-ids`. A change to those kernels' sm_61 code reopens them.

## Not checked here

- **Speed and memory on a P106.** Nothing is timed. The release tests of `v4-rc1` (Flash-Next, Flint and Ornith, on
  one build) measure them; upstream's changes that act on a P106 are listed above (the scheduler kept at a re-reserve,
  the reused CUDA graphs' skipped walk, the separate draft's graph budget).
- **The release's model files** were not opened by the merged loader here (a full read of each file on a shared
  machine). Upstream's loader changes add optional tensors only (the KV scales) and let qwen3.5 read its merged
  tensors or the separate ones; qwen4exp's loader is unchanged. R153's part (3), a run of its own, loads all three
  with their greedy replies against `80c94fe0d`'s.
- **C662's whole `test-backend-ops` in five shards through the compute 6.1 PTX** (the fork's paths off and forced on):
  it needs a quant card.

## Files

| File | What |
|---|---|
| [run/checks.sh](run/checks.sh) | the CPU checks of the branch on a build machine, step by step (its header lists them) |
| [run/sass_diff.py](run/sass_diff.py) | the GPU functions of two builds compared by name: same, changed, new, gone, by kernel family |
| [run/gguf_names.py](run/gguf_names.py) | the tensors of two GGUF files compared from their headers (no dependencies) |
| [run/tables.sh](run/tables.sh) | makes the tables below from the work folder, its path written as `$M` |
| [summary.tsv](summary.tsv) | every verdict line of the checks |
| [union.tsv](union.tsv) | per file the merge changes: the lines of neither upstream nor the fork (`added_extra`, `removed_extra`) |
| [logits-by-model.tsv](logits-by-model.tsv) | per random model: the tip's and the revert build's logits against `80c94fe0d`'s (nmse, largest difference, logits that differ) |
| [modes.tsv](modes.tsv) | the fork's own `test-llama-archs` checks: how each ended, its summary line |
| `server-*.txt` | the verdicts of the MTP server checks (`out`: every group on the tip's random model; `rout`: the revert build there; `btip`, `brev`: `dbase` and `doff` on the base's model) |
| `tensors-*.txt` | the tip's random models' tensors against the base's |
| [draftmd-graphs.txt](draftmd-graphs.txt) | the separate draft's contexts: their graph budgets (`max_nodes`) and nodes, counted over the arms |
| `sass-*-families.txt`, `sass-*-by-function.tsv` | `llama-server`'s GPU functions against `80c94fe0d`'s build, SASS and PTX: the counts by kernel family, and every function that is not the same |
| [switch-names-tip.txt](switch-names-tip.txt) | the switch names the tip's rig build reads (C770's list) |
| [run/runs_past8.sh](run/runs_past8.sh) | R153: a set's run list cut to the cases a record made with `EXTRA_P` holds |
| [run/alias_base.py](run/alias_base.py) | R153: a set recorded with the tip's `test-backend-ops` written again under `80c94fe0d`'s case names |
| [run/compare_builds.sh](run/compare_builds.sh) | R153: two builds' replays of one record compared case by case |
| [run/case_repeat.sh](run/case_repeat.sh) | R153: one recorded case replayed N times with each build |
| [run/r153_tables.sh](run/r153_tables.sh) | R153: the `rig-*` tables from the rig's output and the record logs |
| [rig-sets.tsv](rig-sets.tsv) | R153: each set recorded with the tip's `test-backend-ops`: cases, bytes, cases linked from another set, when |
| [rig-replays.tsv](rig-replays.tsv) | R153: every replayed run, both builds (and the `oldset` rows: the sets of 2026-11-11 on the tip) |
| [rig-compare.tsv](rig-compare.tsv) | R153 part (1): the two builds case by case, per set and run |
| [rig-archs-off.tsv](rig-archs-off.tsv) | R153: per random model, the tip's logits against `80c94fe0d`'s on the card and on the CPU |
| [rig-part2.txt](rig-part2.txt) | R153 part (2): the consoles of `check.sh` and `gpu_checks.sh` |
| [rig-repeat.tsv](rig-repeat.tsv) | R153: the case held twice, five replays with each build |

## Commands

```bash
# the merge, in a llama.cpp clone with upstream as origin and the fork as remote "fork"
git worktree add <dir> -b pascal-p106-v4-b12360-up 80c94fe0d && cd <dir>
git merge --no-ff --no-commit b12360      # 14 files conflict, in 28 hunks; resolved as in the table above
git commit && git push fork pascal-p106-v4-b12360-up

# on the build machine: the checks (SEED_SRC: a clone with the fork as remote "fork"; the ccaches are optional)
M=<work root> REPO=<this repository> LOCK=<flock file> BASE_COMMIT=80c94fe0d UPSTREAM=b12360 SEED_SRC=<clone> \
  SEED_CCACHE=<rig build ccache> SEED_CPU_CCACHE=<CPU build ccache> JOBS=8 THREADS=8 \
  bash docs/data/2026-11-13_c831_upstream/run/checks.sh
# the second chain (after the first had shown that the base cannot load the tip's random model)
M=<same> REPO=<this repository> ... STEPS='seqrm bmodel bserver draftmd' bash docs/data/2026-11-13_c831_upstream/run/checks.sh
# the tables of this folder
M=<work root> OUT=<dir> bash docs/data/2026-11-13_c831_upstream/run/tables.sh
```

R153 on the P106 (`$TIP` and `$BASE`: the two unpacked rig tarballs; `$S`: the sets; `$O`: the output):

```bash
# on the build machine's CPU, with the tip's test-backend-ops (SUITE kshape first: the others link its cases)
OUT=$S SUITE=kshape BIN=$TIP/bin STEPS=record JOBS=8 EXTRA_P='n=(9|[1-9][0-9]+),' bash scripts/record_checks.sh
OUT=$S SUITE=n2p BIN=$TIP/bin STEPS=record JOBS=8 EXTRA_P='n=(9|[1-9][0-9]+),' LINK_FROM=$S/kshape bash scripts/record_checks.sh
#   (flashnext-moe the same way; onecard with LINK_FROM="$S/kshape $S/n2p $S/flashnext-moe" and no EXTRA_P; v4kern;
#   ro8kt with LINK_FROM=$S/v4kern)
SET=$S/n2p EXTRA_P='n=(9|[1-9][0-9]+),' bash docs/data/2026-11-13_c831_upstream/run/runs_past8.sh
python3 -I docs/data/2026-11-13_c831_upstream/run/alias_base.py $S/n2p $S/n2p-base
# on the rig, card 0: part (1) with both builds, then the comparison (on any machine)
SET=$S/n2p BIN=$TIP/bin OUT=$O/n2p-tip RUNS_FILE=$S/n2p/runs-past8.tsv bash scripts/replay_checks.sh
SET=$S/n2p-base BIN=$BASE/bin OUT=$O/n2p-base RUNS_FILE=$S/n2p-base/runs-past8.tsv bash scripts/replay_checks.sh
TIP=$O/n2p-tip BASE=$O/n2p-base ALIAS=$S/n2p-base/alias.tsv bash docs/data/2026-11-13_c831_upstream/run/compare_builds.sh
# part (2)
NATIVE=1 BIN=$TIP ARCHS=$TIP/bin/test-llama-archs WORK=$O/archs GPU=0 ARCHS_ARGS='-b CUDA0' \
  RUNS='archs archs-on archs-mask' bash docs/data/2026-11-03_upstream_rebase/run/check.sh
GGML_TEST_REPLAY=$S/onecard NATIVE=1 BIN=$TIP ARCHS=$TIP/bin/test-llama-archs WORK=$O/moe GPU=0 \
  RUNS='moe-check moe-fault norm-q8' bash docs/data/2026-11-03_upstream_rebase/run/check.sh
MODE=native DEV=0 BIN=$TIP/bin BASE_BIN=$BASE/bin OUT=$O/gpu-a CHECKS=<random models: checks.sh's coldtier step> \
  TEXT=<text file> PARTS='moeskip combo' RUNS='archs-off archs-tail' bash docs/data/2026-11-04_ladder4_merge/run/gpu_checks.sh
GGML_TEST_REPLAY=$S/onecard MODE=native DEV=0 BIN=$TIP/bin OUT=$O/gpu-b CHECKS=... TEXT=... PARTS=moeskip \
  RUNS='skip-mmq skip skip-mmvq skip-mtok0 tail' bash docs/data/2026-11-04_ladder4_merge/run/gpu_checks.sh
SET=$S/v4kern BIN=$TIP/bin OUT=$O/v4kern bash scripts/replay_checks.sh      # and ro8kt
# the repeated case, on card 1
TIP=... BASE=... TIP_SET=$S/kshape BASE_SET=$S/kshape-base OPS=MUL_MAT_ID \
  P='^type_a=q2_0,type_b=f32,n_mats=4,n_used=2,b=0,m=512,n=32,k=256,m_v=0,ids_offset=0,skip=0$' N=5 DEV=1 OUT=<dir> \
  bash docs/data/2026-11-13_c831_upstream/run/case_repeat.sh
```

The random models of the combo part are those of C662's checks (`checks.sh`'s coldtier step,
[2026-11-04_ladder4_merge](../2026-11-04_ladder4_merge/README.md)), and the text file for `llama-perplexity` the one
C662 used.

Provenance: the fork is [Cinderella-Man/miner-llama](https://github.com/Cinderella-Man/miner-llama) (MIT, as
llama.cpp), upstream is [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) at tag b12360; the random models
are written by `test-llama-archs`.
