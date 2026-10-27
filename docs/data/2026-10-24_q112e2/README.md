# Q112 E2/E3 and Q114: GPTQ on K2's body, tool checks and the pipeline (2026-10-24)

The check tables and the runnable pipeline behind "GPTQ for K2h's body" in
[quants/gptq_head/README.md](../../../quants/gptq_head/README.md) (registry rows Q112 and Q114 in
[experiments.md](../../../quants/experiments.md)). Nothing here is a KLD result for E2 or E3: the GPU jobs were staged
and not run. What is here verifies the tools those runs need, and the jobs to run them: Hessians of all 196
always-read inputs from K2-r16's own forward pass, GPTQ for every always-read tensor, one shared 28.3 GB base file with
small second files per variant, and the scores.

| File | What |
|---|---|
| `checks/cputest-check.md` | act_hessian's expert mode against a numpy recount from the dumped rows (K2h, 4 chunks, layers 9 and 47): worst expert 1.36 × 10⁻⁷ (gate/up) and 1.82 × 10⁻⁷ (down) of the largest entry, row counts exact; then a `gptq_experts.py` smoke run with the float64 round-to-nearest (68–95 % of rows equal the mix) |
| `checks/smoke-experts-ggml.md` | the same smoke run with `imat_ggml` (100 % of rows equal the mix); ~40 rows per expert, so the ratios only show that every code path runs |
| `checks/t02-check.md` | act_hessian with `cblas_ssyrk` against the numpy recount (worst 2.55 × 10⁻⁷) and against its plain loops on 10 output files (worst 2.35 × 10⁻⁷) |
| `checks/proxy-o08.md`, `proxy-o08-t12.md`, `proxy-o08-ggml.md` | `gptq_body.py` rerun on E1's attention-output group with other settings: the same errors to three digits; with 12 OpenBLAS threads the donor is byte-identical to E1's, with 8 threads 2.5–4.5 % of the Q5_0 blocks differ; the `ggml` run fills the "= mix" column |
| `run/common.sh` | shared definitions: files, the always-read regex, Docker calls for the GPU and for Python with a fixed BLAS thread count |
| `run/c01-base.sh` | the shared base (K2-r16 without its 343 always-read matrices) and an identity variant, verified |
| `run/c01b-idcheck.sh` | the identity variant against K2-r16 on the CPU |
| `run/t02-blas.sh` | the BLAS-against-loops Hessian check (`checks/t02-check.md`) |
| `run/g01-hess.sh` | Hessians of all 196 always-read inputs, 500 × 512 tokens of wikitext-2 train (~11 GB) |
| `run/c03-solve.sh` | GPTQ for the 120 Q5_0 always-read tensors (same type) and every Q8_0 one into Q5_0, proxy tables, the K2-r16-G5 variant |
| `run/g03-score.sh` | KLD of K2-r16 and K2-r16-G5 against BF16 on the same cards |
| `run/make_candidate.sh` | an E3 candidate: K2-r16-G5 plus GPTQ demotions of the Q8_0 tensors a regex picks |

**The identity check passed.** The shared base plus K2-r16's own always-read tensors as the second file (436 + 343 =
779 tensors, each verified byte for byte) gave, on one 512-token chunk on the CPU, perplexity 19.896902 (equal to
K2-r16's own run), KLD 0.000000 against K2-r16's logits (largest 4.8 × 10⁻⁵, the reference file's 16-bit logit
storage) and the same top token on 100 % of tokens. Every variant built on that base therefore differs from K2-r16 only
in its second file.

Made on 2026-10-24 (checks on the CPU; nothing here used a GPU) with the tools in
[quants/gptq_head](../../../quants/gptq_head/README.md) (`act_hessian/` with `check_hessians.py`, `gptq_body.py`,
`gptq_experts.py`, `split_compose.py`, `test_gptq_head.py`) and three builds of the Q106 tree
([2026-10-24_q106](../2026-10-24_q106/README.md)): CUDA for sm_70 and sm_86 with
`-DHESS_BLAS_LIB=<static OpenBLAS>`, CPU with it, and CPU without it. Run from the repository root:

```bash
export MODELS=... WORK=... BIG=... BIN=... LLAMA_SRC=... GPUS=...   # see run/common.sh
R=docs/data/2026-10-24_q112e2/run
BIN_CPU_BLAS=... BIN_CPU=... bash $R/t02-blas.sh      # check: BLAS capture = numpy recount = plain loops
bash $R/c01-base.sh && BIN_CPU=... bash $R/c01b-idcheck.sh
bash $R/g01-hess.sh && bash $R/c03-solve.sh && bash $R/g03-score.sh
bash $R/make_candidate.sh K2-r16-G5-o08 '^blk\.[0-8]\.attn_output\.weight$'
```

K2-r16 is K2 with F16 routers (`scripts/retype_tensors.py <K2> <K2-r16> 'ffn_gate_inp=F16'`; K2 in
[release/flint-v2/reproduce/model.md](../../../release/flint-v2/reproduce/model.md)). Wikitext-2 is not included
(llama.cpp's `scripts/get-wikitext-2.sh`). The tables and scripts are this project's (MIT).
