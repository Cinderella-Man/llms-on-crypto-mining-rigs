# Quant experiments

Every experiment in the quants subproject, planned or done, so that nothing is run twice. Check this page before
starting anything; add a row before an experiment starts and fill it in when it ends. Results are summarized here and
written up in [README.md](README.md) or [research.md](research.md); raw logs are kept outside the repository.

Status is **planned**, **running**, **done** or **dropped**. Quality numbers are same top token / mean KL divergence /
perplexity change against the reference named in the row, on 80 × 512 tokens of wikitext-2. Speed is real-chat tok/s on
the rig (llama.cpp `d77256999` + CUDA graphs, `-bs`, 262,144-token context) unless the row says otherwise.

## Flint

| ID | Question | Status | Result | How |
|---|---|---|---|---|
| Q1 | How far are the published 4-bit files from Q8_0? | done | makers' Q4_1 78.6 % / 0.316 / +3.7 %; Unsloth UD-Q4_K_XL 87.6 % / 0.124 / +3.3 % (vs Q8_0) | [quality_kld.sh](../scripts/quality_kld.sh) on a CPU |
| Q2 | One fast format everywhere with Unsloth's importance matrix | done | Q5_0 85.0 % / 0.177, 18.9 tok/s; Q4_1 80.9 % / 0.280; Q4_0 71.4 % / 0.566, PPL +45 % (unusable) | [quantize_docker.sh](../scripts/quantize_docker.sh) from Q8_0, head Q8_0 |
| Q3 | Speed of the published files on the rig | done | makers' Q4_1 24.3 tok/s (tg128 27.9–28.1); UD-Q4_K_XL 15.8 (17.4–19.0); Q8_0 tg128 20.5 on 9 cards | [Flint benchmark](../docs/benchmarks/2026-10-22-flint.md) |
| Q4 | Speed gained by switching off the MoE down-projection rescale | done | +5–11 % tg128; Q4_1 gives NaN without it | [patch](../patches/moe-down-rescale-env.diff) |
| Q5 | Rescale-safe expert down-projections (IQ4_NL or Q8_0) on an imatrix Q4_1 | done (quality) | IQ4_NL 80.5 % / 0.281; Q8_0 81.5 % / 0.257; speed with the rescale off not measured | `--tensor-type ffn_down_exps=…` |
| Q6 | Fewer experts per token (6 → 5 / 4) | done | +4 % / +9 % speed for −2 / −7 points same top token: not worth it | `--override-kv flint.expert_used_count=int:N` |
| Q7 | UD-mirror v1: UD-Q4_K_XL's per-tensor bit budget in fast formats | done (quality) | 87.4 % / 0.129 / +3.8 % vs Q8_0; speed on the rig planned (Q9) | [recipe](recipes/flint-udmirror-v1.txt) |
| Q8 | BF16 as the reference: rescore every Flint quant (Q8_0 included) against the original weights | running | | Flint BF16 converted to GGUF on the quant machine, KLD on GPUs |
| Q9 | Speed of UD-mirror v1 on the rig | planned | Predicted by [token_cost.py](../scripts/token_cost.py) (Q17): tg128 22.1 tok/s, about 17.5 in real chat | rig, graphs build, rescale on |
| Q10 | Decode efficiency of each format on one P106-090 card (Q4_0, Q4_1, Q5_0, Q8_0, IQ4_NL, MXFP4, NVFP4) | planned | | rig, `test-backend-ops -o MUL_MAT perf` and a single-format model |
| Q11 | Is the makers' Q4_1 really group-64 (their MLX export) stored as group-32? | done | Yes: in every Q4_1 tensor, 100 % of neighbouring block pairs share the same scale and offset. A group-64 type would hold the same weights exactly in 4.5 instead of 5.0 bits per weight (−9.4 % bytes per token) | reads every block's scale/offset pair with the `gguf` Python package |
| Q17 | A cost model for decode time: bytes per token in each format × per-format decode efficiency + fixed time per token | done | Fitted to four measured Flint files (11 ms fixed): Q4_1 0.38, Q8_0 0.41, Q5_0 0.30, K-/I-quants 0.19 of 173 GB/s; reproduces all four within 2 %. Flint's small expert matrices decode less efficiently than Llama-2's dense ones (Q8_0 0.54 there) | [token_cost.py](../scripts/token_cost.py) |
| Q12 | Our own importance matrix from BF16 (bartowski's Flint calibration text) against Unsloth's, same mix | planned | | `llama-imatrix` on BF16, then the same Q5_0 / UD-mirror recipe with each |
| Q13 | Residual-stream Hadamard rotation (R1, optionally R2 on value heads) folded into the weights: does it make Q4_0 / Q4_1 usable for attention and the experts' gate/up? | planned | | [rotate_gguf.py](../scripts/rotate_gguf.py), imatrix recomputed on the rotated model, KLD against the unrotated BF16 |
| Q14 | Fused `attn_qkv` tensor: identical output, ~200 fewer launches per token | planned | | [fuse_qkv.py](../scripts/fuse_qkv.py); same perplexity on the quant machine, speed on the rig |
| Q15 | Online Hadamard on the expert down-projection inputs (llama.cpp's `prism.hadamard` metadata): better 4-bit down-projections and no FP16-sum overflow, so the rescale can go | planned | | needs Flint added to the loader's allowed architectures (one line) |
| Q16 | A group-64 Q4_1 format holding the makers' weights exactly: −9.4 % bytes per token | planned | | new GGML type with CUDA decode; correctness on the quant machine, speed on the rig |
