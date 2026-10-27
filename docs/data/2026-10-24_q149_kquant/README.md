# Q149: the Pascal Q4_K matrix-vector kernel, its checks and an end-to-end Qwen3.6 run (2026-10-24)

The checks behind Q149, the fork's one-token Q4_K kernel for compute 6.1 (fork branch `pascal-p106-v2-kquant`
`58671c135`, with the Q6_K kernel of Q101 under it, and `GGML_CUDA_MMVQ_PASCAL_SKIP` to leave chosen types to
llama.cpp's kernels; in v2.0): the operator suite with the kernel forced, off and as a mutation build, the SASS identity
of every other Pascal kernel, the static cost of its main loop, and Qwen3.6-35B-A3B Q4_K_M end to end through the
compute_61 PTX. The reusable part, the mutation and SASS-identity kit, is in
[bench/mmvq_pascal/verify](../../../bench/mmvq_pascal/verify/); this folder holds the Q4_K_M tensor list and the end-to-end
run. Results: registry rows Q149 and Q97 in [quants/experiments.md](../../../quants/experiments.md),
[pascal_mmvq_integration.md](../../research/notes/pascal_mmvq_integration.md) section 18 and
[bench/mmvq_pascal](../../../bench/mmvq_pascal/README.md) ("Q4_K (Q149)"; the rig script `kquant_rig.sh`).

| File | What |
|---|---|
| `qwen36-q4km-tensors.tsv` | every tensor of ggml-org's Qwen3.6-35B-A3B Q4_K_M: name, type, shape, bytes (which types the kernels must cover, and how many bytes each streams) |
| `gguf_list.py` | the script that lists a GGUF's tensors (header only, no numpy) |
| `e2e_summary.txt` | the end-to-end runs' perplexity and KLD statistics and the greedy-text comparison (`run/e2e_summary.sh`) |
| `run/e2e.sh`, `run/run_all.sh`, `run/e2e_ctl.sh`, `run/e2e_summary.sh`, `run/placement.sh` | the end-to-end run: the last 2 of 40 layers on one GPU through the compute_61 PTX, the other 38 on the CPU; Pascal kernels off against forced, KLD over 256 wikitext tokens at `-ub 1` and `-ub 4`, greedy text; the controls with Q4_K and Q6_K left to llama.cpp; where the output head landed |
| `run/prompts/p1.txt` … `p4.txt` | the greedy-text prompts (written for this check; p1 and p3 used) |

## What it shows

| Check (RTX 3090 or V100 through the compute_61 PTX of the rig build) | Result |
|---|---|
| `run_suite.sh`, kernels forced (`GGML_CUDA_MMVQ_PASCAL=2`), Q4_K and Q6_K | 270 + 151 + 105 + 10 = 536 of 536 (products, expert products, fused, shared expert), also with staged activations; 504 of 504 with the kernels off (32 large-activation cases need the Pascal feature) |
| Mutation build (Q4_K dot product × 1.1, `q4k-mut.diff`) | exactly the 258 Q4_K cases of 1–4 tokens fail (`tbo_tally.py`), everything else passes; with `GGML_CUDA_MMVQ_PASCAL_SKIP=q4_K` 536 of 536 |
| bench/mmvq_pascal `--check-only` | 662 of 662 (nine types) |
| SASS (`sass_cmp.sh`, against the Q6_K build) | 173 of 173 shared Pascal kernels identical, 15 new (Q4_K) |
| Static cost (`benchloops_k.py`, bench sm_61 SASS) | Q4_K staged `st_r1c1` 2.68 lane instructions per weight byte (2.28 with the gate), direct-load `p_r1c1` 2.49 (2.12); Q5_K 3.07 / 2.77 and 2.89 / 2.63. Registers: bench `st_r1c1` 96 (gate 180), `p_r1c1` 48 (168); fork kernels 125–128 (168) |
| Qwen3.6 Q4_K_M end to end, kernels off against forced | `-ub 1`: mean KLD 0.001916 (max 0.035), same top token 97.6 %, PPL 7.149 → 7.112; `-ub 4`: 0.001686 (max 0.017), 96.9 %; greedy text of 2 prompts × 48 tokens byte-identical |
| The same with Q4_K and Q6_K left to llama.cpp | identical statistics to the fully forced run (0.001916, max 0.035245, 97.638 %): the whole difference is the Q8_0 kernel's summation order; forced against forced-with-skip (the Q4_K and Q6_K kernels alone): KLD −0.000001 ± 0.000001, max 0.000041, same top 100 %, PPL × 1.000005 |
| Port onto the release base (`d379da309`, V100) | forced Q4_K and Q6_K 546 of 546, Q6_K staged activations 169 of 169; the Q4_K+Q6_K mutation fails every case the kernels take; 160 of 160 existing Pascal kernels and 13 of 13 persistent MoE kernels SASS-identical |

In the end-to-end run the output head stayed on the CPU (the `-ot output\.weight=CUDA0` override did not take,
`placement.sh`), so the Q6_K output head ran on the CPU; the CPU layers ran at about 1.2 tokens per second.

## How it was made

On 2026-10-24 on a CUDA host with Docker. The suite, mutation and SASS checks with the kit in
[bench/mmvq_pascal/verify](../../../bench/mmvq_pascal/verify/), on the rig tarball of
[scripts/build_llamacpp_docker.sh](../../../scripts/build_llamacpp_docker.sh) (sm_61 code and compute_61 PTX, `TESTS=1`)
and a second build with `q4k-mut.diff` applied:

```bash
export WORK=<dir> GPU=<n>
bench/mmvq_pascal/verify/run_suite.sh kq-forced 2 <build>
STAGE_ACT=1 bench/mmvq_pascal/verify/run_suite.sh kq-forced-sa 2 <build>
bench/mmvq_pascal/verify/run_suite.sh kq-off 0 <build>
bench/mmvq_pascal/verify/run_suite.sh kq-mut 2 <mutation build>
python3 -I bench/mmvq_pascal/verify/tbo_tally.py $WORK/logs/kq-mut-*.log
SKIP=q4_K bench/mmvq_pascal/verify/run_suite.sh kq-mut-skip 2 <mutation build>
bench/mmvq_pascal/verify/sass_cmp.sh <Q6_K build> <build>
python3 -I bench/mmvq_pascal/verify/benchloops_k.py bench/mmvq_pascal/build/sm61.sass
```

The end-to-end run (a GPU with at least 2.5 GB free, about 20 GB of host RAM for the CPU layers):

```bash
export BUILD=<unpacked rig build> MODEL=<Qwen3.6-35B-A3B-Q4_K_M.gguf> TEXT=<wikitext-2 wiki.test.raw> WORK=<dir> GPU=<n>
docs/data/2026-10-24_q149_kquant/run/run_all.sh && docs/data/2026-10-24_q149_kquant/run/e2e_ctl.sh
docs/data/2026-10-24_q149_kquant/run/e2e_summary.sh > e2e_summary.txt
python3 -I docs/data/2026-10-24_q149_kquant/gguf_list.py <Qwen3.6-35B-A3B-Q4_K_M.gguf> > qwen36-q4km-tensors.tsv
```

The model is ggml-org's Qwen3.6-35B-A3B Q4_K_M (Apache-2.0); wikitext-2's test split is CC BY-SA; the prompts were
written for this check.
