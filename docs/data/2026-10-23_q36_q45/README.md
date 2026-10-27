# Q36, Q44 and Q45: Flint's per-layer allocation and recipe variants (2026-10-23)

The scores behind three registry rows ([experiments.md](../../../quants/experiments.md), Q36, Q44 and Q45): which
8-layer groups of S4's three costliest upgrades can go back to Q4_1 (Q36), whether the makers' 4-bit tensors help a
file made from BF16 (Q44), and eleven recipe variants quantized from BF16 that led to S5 (Q45). The registry gives
rounded values and differences; `scores.tsv` has every file's full score. The exact Q45 recipes are in
[quants/recipes/flint-mixes.tsv](../../../quants/recipes/flint-mixes.tsv), the published S1–S5 recipes in
[quants/recipes](../../../quants/recipes/README.md).

| File | What |
|---|---|
| `scores.tsv` | 41 files: name, registry row, GPU set, what the file is, KLD and its standard error, same top token (%) and its standard error, perplexity and its ratio to BF16, 99th-percentile KLD, and GB read per generated token (the [token_cost.py](../../../scripts/token_cost.py) estimate, where the job computed it) |
| `run/kld_gpu.sh` | the scoring step: KLD against BF16 on 80 × 512 tokens of wikitext-2 test on two GPUs, with Docker environment switches |
| `run/q45_quantize_score.sh` | one Q45 variant (Q1) quantized from BF16 and scored; the others differ only in `EXTRA` |
| `run/q36_compose_score.sh` | one Q36 group probe (a1: attention output Q4_1 in layers 0–8) composed from S4 and S1, scored and deleted |
| `run/q36_final.py` | prints the job for a combined file (S4-L0, L1, L2): groups lowered, router to F16, scored |
| `run/q36_analyze.py` | KLD change per group against the same GPU set's S4, the decode time each group saves (est.), and KLD per ms |

**GPU sets.** Every file was scored on two GPUs of a 7-GPU quant machine, and the GPU set moves KLD by up to 0.005
(`gpu_set`). Q36 used three sets, one per tensor class, each with its own S4 score as the reference: `lane-a` 0.1219
(attention output), `lane-d` 0.1195 (expert down-projections), `lane-u` 0.1170 (expert gate/up). Q45 used `lane-m`,
the set that also scored Q73's rounds 1–4. Compare files within one set.

**S4 with an F16 router** (`flint-mixS4-rf16_nr`, `lane-d`): KLD 0.1213 against S4's 0.1195 on the same GPUs, same top
token 88.88 % against 88.74 %. The difference is within the ~0.005 routing-flip noise of whole-file comparisons, as for
Q35 and Q113.

Made on 2026-10-23 with Flint BF16 (`flint-BF16.gguf`, converted from the published safetensors), Unsloth's importance
matrix (`imatrix_unsloth.gguf`), the makers' Q4_1 file
([RecursiveSI/flint-int4-gguf](https://huggingface.co/RecursiveSI/flint-int4-gguf), `flint-int4-q4_1.gguf`) and
wikitext-2 (llama.cpp's `scripts/get-wikitext-2.sh`; not included here). Run from the repository root:

```bash
# S1, S2, S4 (Q27): quantized from BF16 with the published recipes
for s in 1 2 4; do EXTRA="$(cat quants/recipes/flint-speed-s$s.txt)" scripts/quantize_docker.sh "$LLAMA_SRC" "$MODELS" \
  flint-BF16.gguf flint-mixS$s.gguf Q4_1 imatrix_unsloth.gguf; done
# reference logits, once (CUDA build; the same file scripts/quality_kld.sh's reference step writes on the CPU)
llama-perplexity -m flint-BF16.gguf -f "$WORK/wikitext-2-raw/wiki.test.raw" -c 512 --chunks 80 -ngl 99 -fa on \
  --kl-divergence-base "$WORK/flint-BF16.c512x80.kld"
# Q36 lane references: S4 on each GPU set
GPUS=<set> ENVS=LLAMA_NO_MOE_DOWN_RESCALE=1 docs/data/2026-10-23_q36_q45/run/kld_gpu.sh "$MODELS/flint-mixS4.gguf" _nr_q36a
# Q36 group probes: run/q36_compose_score.sh with the class's donor (a: flint-mixS1.gguf; d, u: flint-mixS2.gguf)
# and the group's layers (1: 0-8, 2: 9-16, 3: 17-24, 4: 25-32, 5: 33-40, 6: 41-48; d within 2-45, u within 2-48)
# combined files
python3 docs/data/2026-10-23_q36_q45/run/q36_final.py flint-S4L0 q36d a5 a6 | GPUS=<set> bash
python3 docs/data/2026-10-23_q36_q45/run/q36_final.py flint-S4L1 q36d a5 a6 d4 d5 d6 | GPUS=<set> bash
python3 docs/data/2026-10-23_q36_q45/run/q36_final.py flint-S4L2 q36a a5 a6 d4 d5 d6 u4 u5 u6 | GPUS=<set> bash
# S4 with an F16 router
python3 scripts/retype_tensors.py "$MODELS/flint-mixS4.gguf" "$MODELS/flint-mixS4-rf16.gguf" 'ffn_gate_inp=F16'
# Q44 probes: S4 with tensors from the makers' Q4_1 (I = flint-int4-q4_1.gguf), e.g. hmk1, the routed experts
python3 scripts/compose_gguf.py "$MODELS/flint-mixS4.gguf" "$MODELS/q36/q36-S4-hmk1.gguf" \
  'ffn_(gate|up)_exps\.weight='"$I" 'blk\.([2-9]|[1-3][0-9]|4[0-5])\.ffn_down_exps\.weight='"$I"
#   hmk2: 'ffn_gate_inp\.weight=I' 'exp_probs_b=I' before hmk1's two rules; hmk3: '^output\.weight=I';
#   hmk5: 'attn_(q|k|v|output)\.weight=I'
# Q45: every variant from BF16 with its EXTRA from quants/recipes/flint-mixes.tsv (base Q4_1), as run/q45_quantize_score.sh
python3 docs/data/2026-10-23_q36_q45/run/q36_analyze.py docs/data/2026-10-23_q36_q45/scores.tsv
```

Every file was scored with `ENVS=LLAMA_NO_MOE_DOWN_RESCALE=1` (the MoE down-projection rescale off, which these
rescale-free recipes allow). The scores are this project's measurements and the recipes its own (MIT); Flint is
Apache-2.0.
