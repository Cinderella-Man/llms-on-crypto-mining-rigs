# Upstream status of the Pascal llama.cpp changes

Where each llama.cpp change used on the rig stands in [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp), what a
pull request for it would have to show, and draft text for each. Compiled 2026-10-23 from the GitHub threads and from
upstream master `f64a49819` (2026-10-23). The changes are the CUDA-graphs gate ([PR #27721](https://github.com/ggml-org/llama.cpp/pull/27721))
and the five files in [patches/](../../../patches/README.md). The rig is nine P106-090 cards: Pascal, compute capability 6.1,
5 SMs each, ~173 GB/s measured, behind a 2-core host ([hardware](../../mining_rig.md)).

## Summary

| Change | Upstream item | State on 2026-10-23 | Upstreamable as it is? | Route |
|---|---|---|---|---|
| CUDA graphs on cc 6.1 | [#27721](https://github.com/ggml-org/llama.cpp/pull/27721) (open since 2026-08-25) | No maintainer review. Diff no longer applies: one context line changed. Gate is now `cc < VOLTA` after [#25749](https://github.com/ggml-org/llama.cpp/pull/25749) | Yes: the same one-line change | Add the rig's numbers to #27721 and ask for review; replace it only if its author is gone |
| `mmvq-pascal-2warps.diff`, `mmvq-pascal-2warps-widek.diff` | [#25479](https://github.com/ggml-org/llama.cpp/pull/25479) (open since 2026-07-09), [#20831](https://github.com/ggml-org/llama.cpp/pull/20831) (RDNA only) | #25479: no maintainer review since 2026-07-10, conflicts with master. #20831: stalled | No: both edit the generic table, which every GPU uses | A Pascal-only table with the wide-row rule, measured over every quant type; coordinate with #25479's author |
| `moe-down-rescale-env.diff` | [#30861](https://github.com/ggml-org/llama.cpp/pull/30861), [#30928](https://github.com/ggml-org/llama.cpp/pull/30928) (merged), [#22571](https://github.com/ggml-org/llama.cpp/pull/22571), [#21652](https://github.com/ggml-org/llama.cpp/pull/21652) (open) | Upstream treats the rescale as a stopgap and expects a follow-up to remove its cost | No: an environment variable in the graph builder, and NaN for Q4_0/Q4_1/Q5_0/Q5_1 experts | Report the measured cost; fix the cause (the 16-bit Q8_1 sums) with the maintainers' design |
| `q4_1_64-type.diff` | none | A new `ggml_type` has a high bar; the Prism ML maintainers kept their group-128 types fork-only | Practically no | Keep in the fork |
| `flint-prism-hadamard.diff` | [#29600](https://github.com/ggml-org/llama.cpp/pull/29600) (merged, the mechanism) | No Flint file carries the metadata | Only with a published Flint file that needs it | Keep in the fork |
| MMVQ fusion on cc 6.1 | [#30334](https://github.com/ggml-org/llama.cpp/issues/30334) (issue, no replies) | Gate unchanged in master | There is no patch | Measure first |

Master moved 15 commits past the base `d77256999`. None touches `ggml/src/ggml-cuda/`, `ggml-quants`, `ggml.c` or
`llama-graph.cpp`. Of the files the diffs modify only `src/llama-model.cpp` (an mlock hunk) and `tests/test-backend-ops.cpp`
(a SYCL test hunk) changed, away from the diffs' hunks. All five diffs pass `git apply --check` against `d77256999` and
against `f64a49819`. The diffs of #27721, #25479, #20831, #22571 and #21652 fail against `f64a49819`.

## What a pull request to llama.cpp has to look like

These rules decide the route for every change below ([CONTRIBUTING.md](https://github.com/ggml-org/llama.cpp/blob/master/CONTRIBUTING.md),
[AGENTS.md](https://github.com/ggml-org/llama.cpp/blob/master/AGENTS.md)).

- **The contributor writes every post.** AI-written pull-request descriptions, commit messages and replies to reviewers
  are prohibited and get the PR closed; AI use in code must be disclosed in the PR template. The draft texts below
  hold the numbers and the structure, and have to be rewritten in the contributor's own words. They use ASCII only,
  as AGENTS.md asks (`-`, `->`, `x`, no em dashes or arrows).
- **One open PR for a new contributor.** The bot comments when a second one is opened.
- **Comment on an existing PR instead of opening a duplicate.** [#27721](https://github.com/ggml-org/llama.cpp/pull/27721) and [#25479](https://github.com/ggml-org/llama.cpp/pull/25479)
  are the same changes as ours.
- **Features start with an issue.** Bug fixes need a test that fails before and passes after.
- **Checks before posting:** run `llama-perplexity` and `llama-bench` to show nothing regresses, and `test-backend-ops` on
  at least two backends (the CUDA device and the CPU) for any change under `ggml/`. A new or changed operator needs
  test cases.
- **No kernel change may affect all GPUs.** The reviewer of #20831 refused a change to the shared kernel path and had it
  restricted to the two RDNA generations. Launch shapes live in per-architecture tables in `mmvq.cu` (Turing, GCN, RDNA2/3/4,
  GB10), with a per-quant-type whitelist measured on that architecture.
- **A new quantization type** needs, at least: a small model converted to GGUF with the type and uploaded to Hugging Face;
  perplexity against BF16/FP16 and against types of similar size; KL divergence against BF16/FP16 for the new type and
  similar types; CPU `llama-bench` against similar types.
- **Who decides.** `ggml-cuda` is owned by JohannesGaessler, am17an, IMbackK and ORippler; `ggml-common.h` and `tools/quantize`
  by ggerganov; `src/llama-graph.*` and `src/models/` by CISC (`CODEOWNERS`).

## CUDA graphs on Pascal: PR #27721

**State.** Opened 2026-08-25 by ArtyomITA, one line in `ggml_cuda_graph_set_enabled`: the architecture gate goes from
`GGML_CUDA_CC_VOLTA` (700) to `GGML_CUDA_CC_DP4A` (610). Open, no review, last activity 2026-08-27. The bot flagged the PR
template. am17an asked dfriehs to test it. Master rejects the diff only because [#30327](https://github.com/ggml-org/llama.cpp/pull/30327) (2026-10-13) changed a
context line: the "disabling CUDA graphs" message is now also conditional on `ggml_cuda_graph_debug_enabled()`. The same
one-line change (`cc < GGML_CUDA_CC_DP4A`) still applies by hand; `scripts/build_llamacpp_docker.sh` does this with `sed`.

**Evidence already in the PR.**

- GTX 1080 (author), CUDA 12.4: `test-backend-ops` 13348/13348; identical perplexity with and without graphs on six models;
  10 of 10 greedy generations identical. Generation +6 % (dense 9B Q5_K_M) to +48 % (Ling-3.0-tiny Q6_K, 24 MoE layers);
  prompt processing unchanged. The gain follows the number of graph nodes.
- P102-100 (hschougaard), driver 580, CUDA 12.9: Ling-3.0-tiny Q8_0 tg128 70.80 -> 88.42, pp512 unchanged.
- P100 (dfriehs), sm_60: regressions for single-card and `-sm layer` runs, about -3 % for most quants on two cards; so the
  gate must stay off for cc 6.0. The PR's `cc < DP4A` does that. dfriehs also asked for an opt-in switch for P100.
- History: graphs were turned off before Ampere in 2024 ([#6766](https://github.com/ggml-org/llama.cpp/pull/6766), P40, dense, ~700 nodes,
  53.84 -> 51.67 tok/s). [#25749](https://github.com/ggml-org/llama.cpp/pull/25749) re-enabled them for Volta and Turing on 2026-07-16 with approvals from Gaessler and am17an;
  Gaessler's reason: the old decision predates the current graph logic and nobody had re-measured. That is the argument
  for Pascal as well.

**What reviewers could still ask.**

- The gate also enables cc 6.2 (Jetson TX2), which nobody has tested.
- A crash report on Volta after #25749 ([#26119](https://github.com/ggml-org/llama.cpp/issues/26119), closed) blamed graph capture with speculative decoding and a
  vision projector, through a cuBLAS workspace allocated inside capture. Pascal reviewers may ask for speculative
  decoding with graphs. The rig runs Qwen3.6 with multi-token prediction on the graphs build (23.4 -> 27.7 tok/s with
  backend sampling), without a crash.
- Memory: about 128 bytes per graph node per graph (ORippler); a 1.5k-node graph costs ~180 kB.
- [#30804](https://github.com/ggml-org/llama.cpp/pull/30804) (open) splits graphs into chunks; [#26771](https://github.com/ggml-org/llama.cpp/pull/26771) (merged 2026-10-20) recaptures after a pool flush.
  Both are in or next to the code the gate controls. The rig build has #26771, not #30804.

**Rig measurements** (P106-090, `llama-bench -ngl 99 -fa on`, llama.cpp `d77256999`, graphs off -> on):

| Model | Cards | Generation, tokens/s | Prompt processing, pp512 |
|---|---|---|---|
| Flint, makers' Q4_1 (39.8 B, ~2.5 B active) | 5 | 24.56 -> 27.92 (+13.7 %) | 170.0 -> 170.6 |
| Flint, makers' Q4_1 | 9 | -> 28.09 | 161.1 |
| Qwen3.6-35B-A3B | 4 | 18.30 -> 19.61 (+7.2 %) | 246.8 -> 246.1 |
| granite-3.1-3b-a800m (`-p 0 -n 64`) | 1 | 49.8 -> 57.7 (+15.9 %) | not run |
| Flint, Unsloth UD-Q4_K_XL | 6 | 17.6 -> 19.5 (+11 %) | not run |

The host has two Celeron cores, so the launch-overhead share is large; the same granite model gains +53 % on the GTX 1080
desktop of the PR's author and +16 % here, which suggests that on the rig more of the per-node time is spent on the GPU.

**To do before asking for review.** `test-backend-ops` (all ops on `CUDA0`), `llama-perplexity` and a greedy-output
comparison with and without graphs on the rig, a run with `-sm layer` over 5 and 9 cards (done for speed, not for
equality), `GGML_CUDA_DISABLE_GRAPHS=1` still disabling, and a rebase of #27721's patch on master.

**Draft comment for #27721** (the route that fits the one-PR limit; rewrite in own words):

> Tested on P106-090 (GP106, cc 6.1, 5 SMs, nine cards on a 2-core Celeron host), llama.cpp d77256999 + this change, CUDA
> 12.9.2, driver 580. llama-bench -ngl 99 -fa on -p 512 -n 128 -r 3, graphs off -> on: Flint (39.8B-A2.5B) Q4_1 on 5 cards
> 24.56 -> 27.92 t/s (+13.7%), 28.09 on 9 cards; Qwen3.6-35B-A3B Q4_0 on 4 cards 18.30 -> 19.61 (+7.2%); granite-3.1-3b-a800m
> on 1 card 49.8 -> 57.7 (+15.9%). pp512 is within noise in all runs. MTP speculative decoding with graphs works on the
> same build. [add: test-backend-ops and perplexity results.] The patch no longer applies to master because #30327 changed the
> debug-message condition; the one-line gate change is unchanged. [ask: would a maintainer review this, or should I open a
> rebased PR?]

**Draft PR text if a replacement is needed.**

Title: `CUDA: enable CUDA graphs on Pascal (cc 6.1)`

> ## Overview
> CUDA graphs are disabled for compute capability below 7.0. #25749 enabled them for Volta and Turing after re-measuring.
> This changes the gate in `ggml_cuda_graph_set_enabled` from `GGML_CUDA_CC_VOLTA` to `GGML_CUDA_CC_DP4A`, so cc 6.1 and
> 6.2 use graphs and cc 6.0 (P100) stays off, where #27721's tests showed regressions. `GGML_CUDA_DISABLE_GRAPHS` still
> disables them.
>
> ## Additional information
> [table above, with the commands]. [test-backend-ops result]. [perplexity with and without]. Supersedes #27721.
>
> ## Requirements
> I have read and agree with the contributing guidelines. AI usage disclosure: [state it].

## Pascal MMVQ launch shape: PR #25479, PR #20831 and our two diffs

**What our diffs do.** `mmvq-pascal-2warps.diff` makes `calc_nwarps` return 2 instead of 4 for `ncols_dst == 1` in the
generic table. `mmvq-pascal-2warps-widek.diff` returns 2 only for rows of 2048 values or more and keeps 4 warps (the
small-K path, one row per warp) for narrower rows, by passing `small_k = true` when `should_use_small_k` evaluates its
condition. Both change the table that Ampere and newer GPUs use, which is why the build recipe says "sm_61 only".

**#25479** (animeshsri14, opened 2026-07-09, 14 lines in `mmvq.cu`) does the same thing the upstream way: a new
`MMVQ_PARAMETERS_PASCAL_DP4A` table for cc in [610, 700) with 2 warps at one token, on both the device and the host path.
Evidence: GTX 1060 6 GB (10 SMs), Llama-3.2-3B, `llama-bench -p 0 -n 128 -r 10`, six paired A/B iterations: Q4_0 +4.6 %,
Q4_K_M +5.4 %, Q5_K_M +3.2 %, Q8_0 +2.5 %, Q6_K -0.3 %, geomean +3.1 %; prefill flat at pp512, pp512 with `-ub 128`, and pp2048. Reviews: Gaessler asked
on 2026-07-10 why it was a draft; the author marked it ready and pinged on 2026-07-23; no maintainer has replied since, and
the only later event is a mention in #30334 on 2026-10-11. The diff conflicts with master: `calc_nwarps` gained the
arguments `type`, `small_k` and `halve_iters`, `calc_rows_per_block` gained `type`, and master has a GB10 table that
doubles warps by the same kind of K-loop rule.

**#20831** (JoursBleu, 2026-03-21) clamps `nwarps` to the matrix width to undo a MoE regression that [#19478](https://github.com/ggml-org/llama.cpp/pull/19478) caused on
RDNA3/4. It was restricted to those GPUs at Gaessler's request, who then doubted that the complexity was worth an
inconsistent gain (2026-04-05); last comment 2026-09-14. The P40 rows in Gaessler's test tables read 1.00 because the
restricted change does not touch Pascal; they are regression controls, not Pascal results. It is the precedent for the
wide-row rule, in the opposite direction: fewer warps for narrow matrices.

**What is already in master.** [#20905](https://github.com/ggml-org/llama.cpp/pull/20905) (merged 2026-03-29) added the small-K rule and a Pascal-or-older exclusion list (`IQ3_S`,
`Q2_K`, `Q3_K` never use small-K when `cc < VOLTA`). [#27828](https://github.com/ggml-org/llama.cpp/pull/27828) (merged 2026-10-10, dfriehs) made P100 always use MMVQ for expert products up to
batch 8. Its author tried to cut the warp count on cc 6.0 as well and found gains for some types, but regressions on
larger models, and left it for later; the PR also asked for tests on sm_61 and sm_70 of the small-K type filter and the
halved warps. [#30517](https://github.com/ggml-org/llama.cpp/pull/30517) (merged 2026-10-17) gives low-bit types (`Q1_0`, `Q2_K`, `Q3_K`, `IQ1_S`, `IQ1_M`) two rows per block at one token,
but only in the generic table, measured on Ampere and newer.

**Rig measurements** (P106-090, `llama-bench` tg128; the baseline is the CUDA-graphs build, since upstream master has graphs
off on Pascal and the effects add):

| File | Cards | graphs build | + 2 warps (all rows) | + 2 warps (rows >= 2048 only) |
|---|---|---|---|---|
| Flint, Unsloth UD-Q4_K_XL (Q4_K/Q5_K/Q6_K/IQ4_XS/Q8_0 mix) | 6 | 19.53 | 24.80 (+27.0 %) | 24.58 (+25.9 %) |
| Flint, makers' Q4_1 | 5 | 28.07 | 31.82 (+13.4 %) | not run |
| Flint, mix with Q8_0 expert down-projections (S4-L0) | 6 | 23.09 | 26.29 (+13.9 %) | 28.02 (+21.4 %) |

Against upstream master (graphs off) UD-Q4_K_XL goes from 17.6 to 24.8 tok/s (+41 %). The Q8_0 expert down-projection
has a 512-value row; with 2 warps it reads 43 GB/s against 81 GB/s with the stock 4-warp small-K path, which is why the
wide-row rule exists. A kernel microbenchmark of stock MMVQ at Flint's Q4_1 shapes reaches 75 of 179 GB/s ([bench/mmvq_pascal](../../../bench/mmvq_pascal/README.md)),
so the kernel is instruction-bound on this card; the cause of the gain per warp count has not been profiled.

**Gap between the diffs and an upstreamable change.**

1. A table of its own: `MMVQ_PARAMETERS_PASCAL` for NVIDIA cc in [610, 700), selected on the device path (`__CUDA_ARCH__`) and the host
   path (`ggml_cuda_highest_compiled_arch(cc)`, as the Turing entry does), instead of an edit to the generic case. In that
   table: 2 warps for `ncols_dst == 1` when the small-K rule does not apply, 4 warps (small-K) otherwise, generic values
   for 2 to 8 columns. `calc_rows_per_block` must list the new table where it lists the generic one, and decide whether the
   low-bit two-row rule of #30517 applies on Pascal; `Q2_K`/`Q3_K`/`IQ3_S` stay excluded from small-K there.
2. A per-type result, because every comparable PR whitelists types: the RDNA3 entry is stricter than RDNA4, the Turing entry
   lists K-quants separately. Q6_K was neutral on the GTX 1060 and the P40 gains depend on type ([#30334](https://github.com/ggml-org/llama.cpp/issues/30334)).
3. More than one Pascal card. Gains measured so far: GTX 1060 (10 SMs) +2.5 to +5.4 % in #25479, P106-090 (5 SMs) +13 to +27 %.
   Whether the gain falls with SM count is unknown. Cards that exist among upstream contributors: GTX 1060 (#25479's author),
   GTX 1080 and P102-100 (#27721's testers), P40 (Gaessler uses one), P100 (dfriehs; cc 6.0, outside the table).
4. Both a MoE and a dense model, as in #27828: Ling-3.0-tiny quantized with `llama-quantize --tensor-type "exps=<type>"` for
   every type, and Qwen3.5-9B for the dense case, `llama-batched-bench -npl 1,2,...,8`.

**Evidence to collect.**

- `test-backend-ops -b CUDA0 -o MUL_MAT,MUL_MAT_ID` passing on the P106-090 (the change only reorders the reduction); add
  perf cases for the shapes where the rule flips (K = 512, 1024, 2048, 4096, one column, plus `MUL_MAT_ID` with 256 experts, 6 used) and run
  `test-backend-ops perf -b CUDA0 -o MUL_MAT` before and after.
- `llama-bench -ngl 99 -fa 1 -p 512 -n 128 -r 5`, plus `-d 0,4096`, base against patch, alternating runs, on a dense and a MoE
  model, for every type in the whitelist; report pp512 as the no-regression control, as #25479 did.
- `llama-perplexity` on one file: equal to within the reduction-order noise.
- Run a build with `-DCMAKE_CUDA_ARCHITECTURES=61-real;86` or similar and check that a non-Pascal card keeps its numbers (the table
  must be invisible to them).

**Draft PR text** (supersedes or builds on #25479; ask its author first):

Title: `CUDA: MMVQ parameters for Pascal (cc 6.1/6.2), 2 warps for single-token products with wide rows`

> ## Overview
> Pascal cards with DP4A use the generic MMVQ table: 4 warps per block at one token. This adds a table for cc 6.1/6.2 that
> uses 2 warps per block at one token when the row is at least 2048 values wide, and keeps the existing 4-warp small-K
> path for narrower rows, where 2 warps are slower (Q8_0, K = 512: 43 vs 81 GB/s). Other values and all other GPUs are
> unchanged. Builds on #25479.
>
> ## Additional information
> P106-090 (GP106, 5 SMs, 173 GB/s), CUDA 12.9.2, llama-bench tg128, base = master + CUDA graphs:
> Flint UD-Q4_K_XL 19.53 -> 24.58 t/s (+26 %); Flint makers' Q4_1 28.07 -> 31.82 (+13 %, 2 warps for all rows); a Q4_1/Q8_0
> mix 23.09 -> 28.02 (+21 %). [sweep over types on Ling-3.0-tiny, dense model, GTX 1060 / P40 numbers]. pp512 unchanged.
> test-backend-ops MUL_MAT/MUL_MAT_ID on CUDA0 passes [counts]. Perplexity [numbers].
>
> ## Requirements
> I have read and agree with the contributing guidelines. AI usage disclosure: [state it].

## The MoE down-projection rescale: #30861, #30928, #22571, #21652 and our diff

**What master does.** Flint, Mistral 4 and Mellum divide the input of the expert down-projection, per expert and token, by a
power of two and multiply the output back (`build_amax_src1_f16` in `src/llama-graph.cpp`, from [#30861](https://github.com/ggml-org/llama.cpp/pull/30861), merged 2026-10-21).
The factor is `2^ceil(log2(max|x| / 1024))`, clamped to [1, 2^127], so that the 16-bit sum of 32 values in a Q8_1 block cannot
exceed the FP16 range (headroom of 32 over the 32768 limit). A comment above the call marks it as a stopgap until
backends accept a BF16 source precision for that matrix product ([#30656](https://github.com/ggml-org/llama.cpp/pull/30656) thread). Flint support ([#30928](https://github.com/ggml-org/llama.cpp/pull/30928), merged
2026-10-21) uses the helper and states that the scaling costs 10 to 30 % and has to be addressed in follow-up PRs.

**The underlying problem.** `quantize_q8_1` stores `half2 (d, sum)`; a block of large same-sign activations makes `sum`
infinite and every output of that row NaN. It is old and architecture-independent:

- [#21652](https://github.com/ggml-org/llama.cpp/pull/21652) (bartowski1182, 2026-04-09; clamp the sum) is open. ggerganov declined to patch around bad model data, then
  (2026-04-24) asked for a wider range in the internal Q8_1 type, BF16 or F32, as a separate fix. Gaessler suggested
  scaling activations down before the product and the result up after it, which is what #30861 does in the graph. A user
  on 2026-10-10 reported the same NaN in a video-diffusion model on CUDA and proposed storing `sum / S` for a power of two `S`
  (exact; relative error 0.4-0.7 %, tested on Q4_K with an sm_120 GPU).
- [#22571](https://github.com/ggml-org/llama.cpp/pull/22571) (bartowski1182, draft, 2026-05-01; BF16 instead of FP16 in the Q8_1 activation struct, 8 files) is idle since 2026-05-07.
  Measured ~1 % slower on an RTX 3090, and the author asked for tests on architectures older than Ampere. Gaessler: the
  idea is right, it needs a clean implementation across all matrix-multiplication kernels and no regression. 0cc4m: do it
  for all backends at once, or make Q8_1 private to each backend.
- On 2026-10-21 ggerganov (on #30861) named the F16 block sums as the remaining overflow, and jeffbolznv suggested BF16 for the
  Q8_1 scale factor; ggerganov said he would look into it. No PR yet.
- A different overflow site: [#30950](https://github.com/ggml-org/llama.cpp/issues/30950) (2026-10-22) and its fix [#31019](https://github.com/ggml-org/llama.cpp/pull/31019) (2026-10-23, open): the Q4_1/Q5_1/Q8_1 MMVQ dot
  products multiply the block offset by the sum in FP16 when `FAST_FP16_AVAILABLE` is set. That macro is not defined for cc 6.1
  (`__CUDA_ARCH__ != 610`), so the P106-090 already multiplies in FP32; the NaN of rescale-free Flint Q4_1 comes from the stored sum.
- Related Mellum fallout: [#30704](https://github.com/ggml-org/llama.cpp/issues/30704), [#30733](https://github.com/ggml-org/llama.cpp/pull/30733) (open revert).

**Rig measurements** (`moe-down-rescale-env.diff`, `LLAMA_NO_MOE_DOWN_RESCALE=1`, graphs build, tg128): makers' Q4_1 on 5 cards
28.05 -> 30.31 (+8 %), perplexity NaN on every chunk without the rescale and 19.29 with it; Q5_0 + imatrix 21.33 -> 23.76 (+11 %);
Q8_0 on 9 cards 20.49 -> 21.57 (+5 %). The rescale adds about 11 operations per MoE layer. Files whose three most
overflow-prone layers (46-48) use Q8_0 expert down-projections run without it: no NaN over 41,000 tokens, same KL divergence.

**Verdict on the diff.** Not upstreamable: `getenv` in the graph builder, an unsafe default for four quant types, and the
maintainers are already replacing the mechanism. What can go upstream is information and, later, a CUDA fix:

1. A comment or issue on the follow-up that #30928 promises, with the measured cost on cc 6.1 (+5 to +11 %), since the
   cost is largest on small, instruction-bound cards.
2. If the Q8_1 sum is widened in the CUDA kernels (the design the research notes call "Q8_1 sum scaled by 1/32": store the
   sum divided by 32 in `quantize_q8_1` and `quantize_mmq_q8_1`, multiply back in the Q4_0/Q4_1/Q5_0/Q5_1 dot products and in the
   DS4 MMQ loads), the graph-level rescale becomes unnecessary on CUDA. That PR touches every NVIDIA GPU, so it needs the
   evidence #22571 lacked: `test-backend-ops` with large-activation inputs (the style #31019 adds), perplexity and KL
   divergence unchanged, `llama-bench` on Ampere and Pascal, and the NaN repro without the rescale. It should wait for
   ggerganov's BF16 design or be sent against it.
3. Pascal test data for #22571 if it is rebased: the P106-090 is a pre-Ampere card on the DP4A MMQ path.

**Draft comment** (for the #30928 follow-up or #30861):

> Flint (#30928) on P106-090 (cc 6.1, 5 SMs): turning the down-projection rescale off with a local switch raises tg128 from 28.05
> to 30.31 t/s (+8%) for the makers' Q4_1, 21.33 -> 23.76 (+11%) for a Q5_0 file and 20.49 -> 21.57 (+5%) for Q8_0 on 9 cards.
> Without it the Q4_1 file gives NaN perplexity on every chunk, so the rescale is needed there; files whose down-projections
> are Q8_0 in the three layers where the activations are large run without it with unchanged KL divergence. On small cards
> the rescale costs 5-11% of decode. Is a BF16 or F32 Q8_1 sum in the CUDA quantize kernels the intended way to drop it, and
> would a PR for that be welcome before the F16 sums are changed in other backends?

## MMVQ fusion on cc 6.1: issue #30334

Opened 2026-10-11 by ymr-gif, open, no comments. `ggml_cuda_should_fuse_mul_mat_vec_q` returns false for `cc <= GGML_CUDA_CC_PASCAL`
(600), so the P40 (610) keeps the fused gate/up kernel. The reporter measured turning it off on a P40 (`-ngl 99 -fa 1`, driver 550,
CUDA 12.4): Mixtral Q2_K -5.5 %, Q3_K_M +5.3 %, Llama-3.1-8B Q4_0 +4.7 %, Q4_K_M -1.2 % at one token; with fusion off and the small-K
rule forced on for Q2_K, +12 %. The sign depends on the type, so the fix would be a per-type rule, not a changed comparison. The
P106-090 keeps fusion on; the microbenchmark in [bench/mmvq_pascal](../../../bench/mmvq_pascal/README.md) times the fused kernel. No rig measurement of fusion on or off exists (`GGML_CUDA_DISABLE_FUSION=1`
also switches off every other CUDA fusion, so a clean test needs a local switch like the reporter's `MMVQ_TEST_NO_FUSE`). No
change of ours depends on it.

## `q4_1_64-type.diff`

Adds `GGML_TYPE_Q4_1_64` (id 43, equal to `GGML_TYPE_COUNT` in master) across 23 files: CPU quantizer and dot product,
CUDA MMVQ, dequantization and row lookup, `gguf-py`, `llama-quant`, `llama-model-loader`, `quantize` and a `test-backend-ops` case. It has no MMQ
kernel, so prompts fall back to dequantize plus cuBLAS (about 40 % slower prefill). Benefit on the rig: the makers' Flint
Q4_1 is group-64 data stored with group-32 scales; the new type holds it exactly in 9.4 % fewer bytes, and the 7 % decode gain over
Q4_1 comes with a Pascal kernel.

Upstream outlook, from the rules and precedents:

- The extra criteria for a new type (Hugging Face model, perplexity and KL divergence against BF16 and against similar-size
  types, CPU performance) apply, plus an issue first.
- Prism ML's own maintainers answered a request to merge their group-128 types (`PQ2_0`, `PTQ1_0`) by saying new types add
  maintenance and might stay in their fork ([#29077](https://github.com/ggml-org/llama.cpp/pull/29077) closed; [#29672](https://github.com/ggml-org/llama.cpp/pull/29672) still open with a naming discussion and no maintainer review). A type
  for one model's quantization-aware training layout is a harder case still.
- Master still has `GGML_TYPE_COUNT = 43`. When upstream adds a type, id 43 collides with ours, and GGUF files written with the old id
  would be read as another type. If the fork distributes files, choose an id far from the count (Prism's fork uses 142 and 143).

**Verdict.** Keep it in the fork. If an upstream attempt is made, start with an issue; the draft below is the PR that would follow.

Title: `ggml : add Q4_1_64 (Q4_1 with one scale and offset per 64 weights)`

> ## Overview
> [Why: models trained with group-64 QAT, Flint, are shipped as Q4_1 with duplicated scales.] 4.5 bits per weight. CPU
> reference, CPU dot product, CUDA MMVQ, dequantization, get_rows. No batched CUDA kernel yet.
>
> ## Additional information
> Required by CONTRIBUTING for a new type: [GGUF on Hugging Face], [perplexity vs BF16 and vs Q4_0/Q4_1/Q4_K], [KL
> divergence vs BF16 for the same], [CPU llama-bench vs Q4_0/Q4_1]. test-backend-ops MUL_MAT, MUL_MAT_ID, GET_ROWS pass on CPU and CUDA0. P106-090: tg128 [..] vs Q4_1.
>
> ## Requirements
> I have read and agree with the contributing guidelines. AI usage disclosure: [state it].

## `flint-prism-hadamard.diff`

Adds `LLM_ARCH_FLINT` to the architectures `load_hparams_hadamard` accepts. Master keeps that list to Llama, Qwen3, Qwen3-MoE,
Qwen3.5 (dense and MoE) and Qwen3-Next, because only the matmul helpers `build_lora_mm`/`build_lora_mm_id` apply the input transform, and a
runtime check (`llama_verify_hadamard_graph`) refuses graphs that read a folded table without the inverse. The mechanism
came with Prism Bonsai 2 ([#29600](https://github.com/ggml-org/llama.cpp/pull/29600), merged 2026-10-08; [#30910](https://github.com/ggml-org/llama.cpp/pull/30910), LoRA on folded weights, merged 2026-10-22).
Flint's graph (`src/models/flint.cpp`) goes through `build_attn`, `build_ffn`, `build_moe_ffn` and `build_lora_mm` for the head. The
Flint support in master (#30928) landed one day before this note.

An architecture whitelist is only justified by a published file that needs it; no Flint GGUF carries `prism.hadamard` metadata and the
rig's rotation experiments did not improve quality or speed ([quants/research.md](../../../quants/research.md)). Keep it local. If a rotated Flint file is ever published, the PR is one line plus
a perplexity/KL comparison of the rotated and unrotated file, and a `test-llama-archs` run.

Title: `llama : accept prism.hadamard metadata for Flint`

> ## Overview
> Flint's graph applies all projections through build_lora_mm / build_lora_mm_id, as the already accepted architectures do. [published file]
> needs the architecture in the load_hparams_hadamard whitelist.
>
> ## Additional information
> Perplexity and KL divergence of [file] against the unrotated BF16 reference: [..]. test-llama-archs [..].

## Conflicts with master, one by one

| Our diff | Files | Against `f64a49819` |
|---|---|---|
| `moe-down-rescale-env.diff` | `src/llama-graph.cpp` | Applies. The lines are the post-#30861 code. Disappears when upstream removes the stopgap |
| `flint-prism-hadamard.diff` | `src/llama-model.cpp` | Applies; the master hunk that differs is the mlock block, 150 lines away |
| `q4_1_64-type.diff` | 23 files incl. `ggml.h`, `ggml-common.h`, `mmvq.cu`, `vecdotq.cuh`, `test-backend-ops.cpp` | Applies. Type id 43 collides with the next upstream type; shares `mmvq.cu` with the warp diffs (apply the type first) |
| `mmvq-pascal-2warps.diff`, `-widek.diff` | `mmvq.cu` | Apply. Alternatives: use one. Behavior conflicts: both change the generic table that all other NVIDIA GPUs use; #30517's two-row rule for low-bit types combines with 2 warps unmeasured; the widek diff changes the small-K decision for the generic table |
| CUDA-graphs gate | `ggml-cuda.cu` | #27721's diff fails (stale context); the `sed` in the build script works |

## Plan under the one-PR limit

1. Add the rig data to #27721 and ask a CUDA maintainer to review. This costs no PR slot. If the author is not active, open the
   rebased replacement as the first PR.
2. Prepare the Pascal-table change on a branch while collecting the evidence above; ask the author of #25479 whether to extend it or
   to take over. It is the larger win (+13 to +27 %), and the one most likely to be asked for more data.
3. Post the rescale cost where #30928's follow-up is discussed; wait for the Q8_1 sum design.
4. Keep `q4_1_64-type.diff`, `flint-prism-hadamard.diff` and the environment switch in a fork. In a fork, the commit order is graphs,
   Pascal table, then the three local diffs.

Open measurements: `test-backend-ops` and equality checks for the graphs and warp changes on the rig; the warp rule over every
quant type on a MoE and a dense model; one or two more Pascal cards; fusion on or off on the P106-090.

## Sources

- [PR #27721](https://github.com/ggml-org/llama.cpp/pull/27721) CUDA graphs on Pascal; [#25749](https://github.com/ggml-org/llama.cpp/pull/25749) graphs on Volta and Turing; [#6766](https://github.com/ggml-org/llama.cpp/pull/6766) graphs introduced; [#26119](https://github.com/ggml-org/llama.cpp/issues/26119); [#26771](https://github.com/ggml-org/llama.cpp/pull/26771); [#30327](https://github.com/ggml-org/llama.cpp/pull/30327); [#30804](https://github.com/ggml-org/llama.cpp/pull/30804).
- [PR #25479](https://github.com/ggml-org/llama.cpp/pull/25479) Pascal MMVQ warps; [#20831](https://github.com/ggml-org/llama.cpp/pull/20831) narrow-matrix warps; [#20905](https://github.com/ggml-org/llama.cpp/pull/20905); [#27828](https://github.com/ggml-org/llama.cpp/pull/27828); [#30517](https://github.com/ggml-org/llama.cpp/pull/30517); [#16715](https://github.com/ggml-org/llama.cpp/pull/16715) (MMVQ fusion); [issue #30334](https://github.com/ggml-org/llama.cpp/issues/30334).
- [PR #22571](https://github.com/ggml-org/llama.cpp/pull/22571), [#21652](https://github.com/ggml-org/llama.cpp/pull/21652), [#30861](https://github.com/ggml-org/llama.cpp/pull/30861), [#30928](https://github.com/ggml-org/llama.cpp/pull/30928), [#30656](https://github.com/ggml-org/llama.cpp/pull/30656), [#30733](https://github.com/ggml-org/llama.cpp/pull/30733), [issue #30704](https://github.com/ggml-org/llama.cpp/issues/30704), [issue #30950](https://github.com/ggml-org/llama.cpp/issues/30950), [#31019](https://github.com/ggml-org/llama.cpp/pull/31019).
- [#29600](https://github.com/ggml-org/llama.cpp/pull/29600), [#30910](https://github.com/ggml-org/llama.cpp/pull/30910), [#29077](https://github.com/ggml-org/llama.cpp/pull/29077), [#29672](https://github.com/ggml-org/llama.cpp/pull/29672), [#19762](https://github.com/ggml-org/llama.cpp/pull/19762) (the new-type requirements).
- Rig measurements: [Flint benchmark](../../benchmarks/2026-10-22-flint.md), [Qwen baselines](../../benchmarks/2026-10-22-qwen-baselines.md), [small-MoE overhead](../../benchmarks/2026-10-22-small-moe-overhead.md), [quants/README.md](../../../quants/README.md), [bench/mmvq_pascal](../../../bench/mmvq_pascal/README.md).
