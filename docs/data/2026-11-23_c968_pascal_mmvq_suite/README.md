# The fork's Pascal matrix-vector kernels in the card suite, on three newer cards (C968, Q1698, 2026-11-23)

What this is: registry row [Q1698](../../../quants/experiments.md), point 15 of the GPU checks of the night's branches
(registry [Q1687](../../../quants/experiments.md)). The fork's kernels for small matrix products (1 to 8 tokens:
`mul_mat_vec_q_pascal` and its multi-token kernel, the k-column kernels, the multi-token expert kernels, the BF16
kernels, and the opt-in read-once kernels and fused gate/up launch) turn themselves on only on compute 6.1 cards, the
P106. `GGML_CUDA_MMVQ_PASCAL` and `GGML_CUDA_MMVF_PASCAL` at their default 1 need compute 6.1; `=2` turns them on for any
NVIDIA card. So [C947's card suite](../2026-11-21_c947_v5_merged_gpu/README.md), run on a build machine's RTX 3090s
(compute 8.6) without `=2`, checked llama.cpp's own matrix-vector kernels in their place. This runs C947's three shares
that reach these kernels again with them forced on and a P106's launch shapes. Correctness only: nothing here is timed.

## Result

**In plain words:** the fork's own kernels pass. With them forced on, every matrix product in the card suite computes
correctly on these cards, under the default settings every v5 preset but Flash-Next's uses and under Flash-Next's own
kernel switches. One case failed, a router check whose top-k hit a tie, and it fails the same way with the fork's
kernels off: it is not theirs. Forcing them also runs 278 of the fork's own test cases that C947's run could not.

| Share (card) | Cases | With the fork's kernels forced (live) | Replayed on the same inputs: unset, p106, fn, p106-kcol0, fn-f32 | Cases whose result the fork's kernels change |
|---|---:|---|---|---:|
| `MUL_MAT` (0) | 5,125 | **5,125 of 5,125** pass, p106 and fn | 4,946 of 4,946 pass in every setting | 2,264 (p106), 2,258 (fn) |
| `MUL_MAT_ID` (1) | 2,370 | **2,370 of 2,370** pass, p106 | 2,254 of 2,254 pass unset; 2,370 of 2,370 in every forced setting | 1,042 (p106), 1,043 (fn) |
| fused cases (2) | 5,274 | **5,273 of 5,274** pass, p106 | 5,273 of 5,274 pass in every setting, unset too | 1,931 (p106), 1,949 (fn) |

([bops.tsv](bops.tsv), [verdict.tsv](verdict.tsv), [moved.tsv](moved.tsv))

- **The only failure is a tie in the router's top-k, not a matrix product.** `TOPK_MOE_TAIL(n_expert=256,n_tokens=9,
  n_expert_used=8,with_norm=1,gating_func=1,scale_w=2.5,p=0.8,w=0,renorm=1,n_skip=8,repeat=0)` fails on its cut expert
  ids (normalized squared error 3.5e-5 against the processor, limit 1e-7) while its cut weights and its expert product
  match (5e-15). Replayed on the same inputs it fails the same way with the fork's kernels off. Sigmoid gating of
  logits up to 6 saturates: for one token the processor's 8th expert and the first one left out are 1 float32 ulp
  apart in probability (0.996305525 and 0.996305466), so the card's sigmoid can choose the other one, and the tail
  cuts that 8th slot to weight 0 in every token, so the product never reads the differing id
  ([run/topk_tail_tie.py](run/topk_tail_tie.py)). Which draws tie is random: C947's draw of this case passed, and so did
  the same parameters at 22 tokens here. Its record is kept in [topk_moe_tail/](topk_moe_tail/) and replays the
  failure on its own (command below). [C992](../2026-11-24_c992_topk_tail_tie/README.md) (Q1725) fixed the test: it
  no longer compares the ids of slots cut to weight 0 past `n_skip` tokens and draws its logits without near ties, and
  with it this record passes.
- **The fork's kernels take 5,237 of C947's cases.** Replayed on the same inputs, 2,264 `MUL_MAT`, 1,042 `MUL_MAT_ID`
  and 1,931 fused cases give another result with the kernels forced than without, so those are the cases C947's run
  computed on llama.cpp's kernels where a P106 runs the fork's. Unset replayed twice gives every case the same result,
  so each change comes from the kernel, not from chance.
- **278 cases run only with the fork's kernels on:** 162 `MUL_MAT` and 116 `MUL_MAT_ID` products whose activations
  hold channels large enough that a block of 32 sums past FP16's 65,504, as Flint's last layers' expert inputs do
  (`test_mul_mat_large_act`, 2.5e5). The fork's kernels keep those sums in FP32; the cases declare themselves supported
  only where those kernels are on, so C947's run skipped them as not supported. All 278 pass.
- **Where the fork's kernels act**, from [moved.tsv](moved.tsv):
  - `MUL_MAT` at 1 to 8 tokens, never at 9 or more (405 of 1,063 one-token cases, 1,150 of 1,479 at 2–4, 709 of 1,466
    at 5–8, 0 of 938 at 9+). Types: BF16, the Q types (Q2_0, Q3_K to Q6_K, Q4_0, Q4_1, Q5_0, Q8_0), the IQ types
    but IQ1_S, and the fork's N2P/N3P types. No case of F32, F16, F8_E4M3, MXFP4, NVFP4, Q1_0, Q2_K, IQ1_S, Q5_1 or
    Q4_1_64 changes: those stay on llama.cpp's kernels on a P106 as well.
  - `MUL_MAT_ID` for the same quantized types, BF16 not. 53 of the cases they change have batches of 16 to 512 tokens:
    the expert kernels see each expert's share of the batch, which is a few tokens when the experts are many (256
    experts with 6 used and 128 tokens: about 3 each).
  - The fused cases through their products (cases p106 or fn changes): `MUL_MAT_VEC_FUSION` (1,093 of 2,317), `MUL_MAT_GLU` (612 of 636),
    `MUL_MAT_ID_SHARED` (128 of 131), `MUL_MAT_ADD` (71 of 168), `ADD_RMS_NORM_MUL_Q8` (32 of 32), `MOE_LAYER`
    (17 of 17) and `MUL_MAT_ID_FUSION` (4 of 14). The other fused operators have no product the kernels take.
  - The k-column kernels computed 1,121 of the `MUL_MAT` cases, 46 of the `MUL_MAT_ID` ones and 647 of the fused ones
    under p106: switching them off (p106-kcol0) changes those results, and every one still passes.
  - `GGML_CUDA_MMVF_F32_MT` has its own compute 6.1 gate: at Flash-Next's 1 it does nothing on these cards, at `=2`
    (fn-f32) it moves 35 `MUL_MAT` and 35 fused F32 products of 5 to 8 tokens onto `mul_mat_vec_f`, all passing.

## What ran where

| | |
|---|---|
| Build | `pascal-p106-v5` at its tip `14d679604` (v5 with pull requests 8–10 and the decode-mask fix); rig tarball md5 `d8dd415dea36f9d064ce67e65840c218`, `test-backend-ops` `56890de320df1255d8cada55ad0c3bd8`. Its matrix-vector kernel files are those of C947's build `3dfad876f` (only flash-attention files changed between them), so this checks that build's kernels too |
| Cards | three RTX 3090 24 GB on a build machine, running the rig build's compute 6.1 PTX through the driver's JIT, one share per card; a CUDA preflight passed on all three first |
| Shares | C947's split: `MUL_MAT`, `MUL_MAT_ID`, and the 43 fused-case operators test mode runs that the support listing does not name ([run/ops-testmode-only.txt](run/ops-testmode-only.txt)); every case but the repeated-expert-id ones (`-p "^(?!.*dup=1)"`). C947's fourth share (every other operator) has no matrix-vector product and is not rerun |
| Times | `MUL_MAT` 22:52–23:26 on card 0; `MUL_MAT_ID` 23:28–00:20 on card 1; fused 23:28–00:15 on card 2 |

The settings ([run/run.sh](run/run.sh)):

| Setting | Environment |
|---|---|
| unset | none: llama.cpp's own matrix-vector kernels on these cards, as in C947 |
| p106 | `GGML_CUDA_MMVQ_PASCAL=2 GGML_CUDA_MMVF_PASCAL=2 GGML_CUDA_MMVQ_PASCAL_NSM=5 GGML_CUDA_MMVF_PASCAL_NSM=5 GGML_CUDA_MMVQ_PASCAL_KCOL_NSM=5 GGML_CUDA_MMVQ_PASCAL_KCOL_SMSM=98304`: the fork's kernels with every switch at its default and a P106's launch shapes (5 SMs, 96 KB of shared memory per SM), as [C953](../2026-11-23_c953_ornith_kcol_gpu/README.md) set them. The Flint, Ornith and Qwen3.6 v5 presets set no kernel switch |
| fn | p106 plus every `GGML_CUDA_MMV*` switch of [release/flash-next-v5/settings.env](../../../release/flash-next-v5/settings.env): `_KCOL_SMEM=32768 _KCOL_TABLE=0 _DMT=1 _Q5K_MT=1 _Q3K_MT=1 _GLU_MT=1 _Q6K_MT=1 GGML_CUDA_MMVF_F32_MT=1 _KCOL_EXACT=1 _KCOL_SKIP=q4_0,q5_0` (the read-once kernels for Q3_K, Q4_K, IQ4_XS, Q5_K and Q6_K, the fused gate/up launch, the k-column kernel's exact mode) |
| p106-kcol0 | p106 with `GGML_CUDA_MMVQ_PASCAL_KCOL=0`, so a result that changes was computed by a k-column kernel under p106 |
| fn-f32 | fn with `GGML_CUDA_MMVF_F32_MT=2` |

How each share ran:

- **`MUL_MAT`:** live with p106 and with fn (each case on the card against the processor, fresh random inputs), then
  recorded unset (`--record`: inputs and processor reference written; the 17 cases over 64 MB left out, 12 GB), then
  that record replayed unset, unset again, p106, fn, p106-kcol0 and fn-f32 (`--replay`, `--replay-log`: each case's
  verdict and normalized squared error per compared node).
- **`MUL_MAT_ID` and the fused cases:** recorded with the live check forced (p106), so that one pass is both the live
  check and the record, every case kept (28 GB and 37 GB), then replayed in the same six settings. Their first live
  p106 runs, started like `MUL_MAT`'s, went at under half of C947's pace with every processor core busy on the
  references, so they were stopped after 740 and 2,452 cases, all passing, and replaced by this.

A replay recomputes the record's own inputs on the card and compares with the stored processor result, so two
settings' errors on a case can be compared directly. The records were deleted once the tables were made, except the one
failing case's.

## Files

| File | What it holds |
|---|---|
| [bops.tsv](bops.tsv) | every `test-backend-ops` run: share, run, setting, cases, passed, failed, cases the record lacks, exit code, failing cases |
| [verdict.tsv](verdict.tsv) | per share: cases recorded and replayed, passed and failed per setting, cases the fork's kernels change, cases replayed only forced |
| [moved.tsv](moved.tsv) | per share, operator, weight type and token count: cases, how many repeat unset, how many p106, fn, p106-kcol0 and fn-f32 change |
| [cases.tsv.gz](cases.tsv.gz) | every replayed case: verdict and errors per setting |
| [topk_moe_tail/](topk_moe_tail/) | the failing `TOPK_MOE_TAIL` case's record (inputs, processor reference, metadata; random test data) |
| [run/run.sh](run/run.sh) | the driver: the live runs, the record and the replays per share, one card each |
| [run/tables.py](run/tables.py) | the tables from a copy of the driver's work folder (standard library, `python3 -I`) |
| [run/topk_tail_tie.py](run/topk_tail_tie.py) | the failing case's router read from its record: ranks 7 to 10 per token and the gap between ranks 8 and 9 in ulps |
| [run/ops-testmode-only.txt](run/ops-testmode-only.txt) | the fused share's 43 operators (C947's list) |

## Commands

On the build machine, with the rig tarball unpacked, `run/` and `release/flash-next-v5/settings.env` copied into a work
folder, a folder with ~80 GB free for the records, and the three cards free:

```bash
E="NEW=<14d679604 build>/bin REPO=<work folder>/repo REC=<records folder> MOUNTS='<folders holding the build>' \
DOCKER_ARGS='--cpu-shares=256' PREFLIGHT=<CUDA check, run as \$PREFLIGHT -g 0,1,2> FORCED=mmv"
# MUL_MAT on card 0 (the default chain: live p106 and fn, the unset record at 64 MB a case, five replays; the first
# invocation ran all three shares this way, and MUL_MAT_ID's and the fused share's chains were stopped)
env $E WORK=<work folder> SHARES=mul_mat setsid nohup bash run/run.sh > <work folder>/driver.log 2>&1 < /dev/null &
# MUL_MAT_ID on card 1 and the fused cases on card 2: the record with the live check forced, no cap, six replays
env $E WORK=<work folder> SHARES='mul_mat_id fused' GPUS=1,2,0 STEPS='record replay' RECORD_SETTING=p106 RMAX=0 \
    REPLAYS='unset unset2 p106 fn p106-kcol0 fn-f32' setsid nohup bash run/run.sh > <work folder>/driver.log 2>&1 < /dev/null &
# MUL_MAT's fn-f32 replay on card 0 (its outputs then moved into the main folder)
env $E WORK=<work folder>/x1 SHARES=mul_mat STEPS=replay REPLAYS=fn-f32 setsid nohup bash run/run.sh > <work folder>/x1/driver.log 2>&1 < /dev/null &
# the tables, and the failing case's router
python3 -I run/tables.py <copy of the work folder> <this folder>
python3 -I run/topk_tail_tie.py topk_moe_tail/cases/301f934b85719490d52f.in topk_moe_tail/cases/301f934b85719490d52f.ref
# the failing case replayed alone on any CUDA card, from this folder
test-backend-ops test -b CUDA0 -o TOPK_MOE_TAIL --replay topk_moe_tail --replay-allow-missing \
    -p '^n_expert=256,n_tokens=9,n_expert_used=8,with_norm=1,gating_func=1,scale_w=2\.500000,p=0\.800000'
```

`FORCED=mmv` forces the matrix-vector kernels alone, as this run did; the driver's default since 2026-11-25 forces every
compute-6.1-only path of the fork ([scripts/pascal_forced.sh](../../../scripts/pascal_forced.sh), C991).

The record sizes with no cap, from `--record-dry`: `MUL_MAT` 17.5 GB, `MUL_MAT_ID` 26.7 GB, fused 38.9 GB.
