# Ornith-1.5-35B: rig-sized files made from its BF16, against the makers' Q8_0 (2026-10-28)

Registry row Q159. Ornith-1.5-35B-A3B has the shapes of Qwen3.6-35B-A3B: 40 layers (30 Gated DeltaNet, 10 full
attention), 256 routed experts of width 512 with 8 used per token plus a shared expert, and an MTP layer. This folder
makes candidate files from the makers' BF16 weights and measures how far each is from the makers' Q8_0 by KL divergence
on the three texts of [2026-10-27_ornith_kld](../2026-10-27_ornith_kld/README.md): wikitext-2 (80 × 512 tokens),
held-out agent sessions (12 × 8,192) and the agent check's Exercism code (80 × 512). As there, the Q8_0 is the
reference; its perplexity on the three texts is 7.811, 1.703 and 1.362. The model's page is
[models/ornith-1.5-35b](../../../models/ornith-1.5-35b/README.md).

## Results

Mean KL divergence against the Q8_0 / share of tokens with the same top token. GB per token: the tensor bytes a
decoded token reads (8 of 256 routed experts per layer, every other layer tensor, the output layer; no MTP layer).

| File | Recipe | GB | GB per token | wikitext | Agent sessions | Exercism code |
|---|---|---|---|---|---|---|
| Q8_0 (the makers', reference) | | 37.79 | 3.20 | 0 | 0 | 0 |
| Q6_K | llama.cpp's Q6_K | 29.20 | 2.49 | 0.0311 / 92.4 % | 0.0368 / 96.9 % | 0.0124 / 98.4 % |
| **P-Q5** | **routed experts Q5_0, all else Q8_0** | **25.41** | 2.82 | **0.0276 / 93.1 %** | **0.0290 / 97.2 %** | **0.0123 / 98.5 %** |
| Q5_K_M | llama.cpp's Q5_K_M | 25.34 | 2.23 | 0.0500 / 90.4 % | 0.0458 / 96.2 % | 0.0196 / 97.9 % |
| P-Q4 | routed experts Q4_0, all else Q8_0 | 21.28 | 2.69 | 0.0502 / 90.2 % | 0.0396 / 96.5 % | 0.0248 / 97.8 % |
| P-Q4-i | P-Q4 with an importance matrix | 21.28 | 2.69 | 0.0483 / 90.8 % | 0.0430 / 96.5 % | 0.0229 / 97.7 % |
| **PK-Q4-i** | **routed experts Q4_K, all else Q8_0, importance matrix** | **21.28** | 2.69 | **0.0405 / 91.5 %** | **0.0358 / 96.9 %** | **0.0191 / 98.0 %** |
| Q4_K_M-i | llama.cpp's Q4_K_M, importance matrix | 21.70 | 1.98 | 0.0633 / 89.5 % | 0.0541 / 95.7 % | 0.0301 / 97.4 % |
| Q4_K_M | llama.cpp's Q4_K_M | 21.70 | 1.98 | 0.0893 / 87.4 % | 0.0720 / 94.6 % | 0.0416 / 97.0 % |

Standard errors are 0.0006–0.0017 (`kld_35b.tsv`).

- **P-Q5 is the closest file below the Q8_0** on all three texts, ahead of Q6_K, which is 3.8 GB larger, on prose
  and agent sessions, and level with it on code.
- **The dense tensors decide the score.** Every tensor outside the routed experts takes 2.7 GB in Q8_0. At equal
  size, the files that keep them in Q8_0 are 1.6–1.8× closer than llama.cpp's mixes, which round them to 4–6 bits:
  P-Q5 against Q5_K_M, P-Q4 against Q4_K_M.
- **The P- files share the reference's dense tensors byte for byte.** The conversion from BF16 reproduces the makers'
  file exactly (`run/tensor_md5.py`: the F32 router, SSM and norm tensors, and every Q8_0 tensor checked, such as
  `blk.0.attn_qkv`, `blk.30.ssm_out`, `output` and `token_embd`). So their divergence is the routed experts'
  rounding alone, while llama.cpp's mixes also carry the dense tensors' rounding. The Q8_0's own distance from the
  BF16 weights is not measured here.
- **Bytes per token run the other way.** A token reads 8 of each layer's 256 experts but every dense tensor, so the
  dense tensors are ~75 % of a decoded token's bytes in P-Q5 (2.82 GB per token against Q6_K's 2.49 and Q5_K_M's
  2.23). How much that costs in speed is a rig question: a ~3 B-active model on nine cards is not limited by memory
  bandwidth alone.
- **The importance matrix helps the K-quants, not Q4_0.** Computed from 102k tokens of agent sessions outside the
  scored ones (`run/calib_text.py`, `run/imatrix.sh`), it brings Q4_K_M 25–29 % closer on all three texts. For Q4_0
  experts (P-Q4-i) it gains 4 % on prose and 8 % on code and loses 9 % on agent sessions. With 102k tokens over 256
  experts per layer, each expert sees ~3k tokens.
- **PK-Q4-i is the 4-bit file: both levers together.** Routed experts in Q4_K with the importance matrix and dense
  tensors in Q8_0 come 1.1–1.3× closer than P-Q4 at the same 21.3 GB (0.0405 / 0.0358 / 0.0191), and closer than
  llama.cpp's Q5_K_M, which is 4 GB larger, on prose and agent sessions (level on code).
- **Every file keeps the MTP layer** (block 40, quantized by the same recipe), so the MTP head can draft with
  any of them.

## What goes to the rig

1. **The makers' Q8_0** (37.8 GB): no quality loss by definition. It fits nine cards; the full 262,144-token
   context adds about 6 GB, as for Qwen3.6, est. 44–46 GB of the rig's 54 GB.
2. **P-Q5** (25.4 GB): the closest smaller file, with routed experts and dense tensors on the two formats with the
   v2 engine's fastest Pascal kernels (Q5_0, Q8_0). It reads 12 % fewer bytes per token than the Q8_0 and fits six
   cards with the full context.

3. **PK-Q4-i** (21.3 GB): the small file, for five cards with the full context, or two copies of the model on nine
   cards (two agents at once). It reads as many bytes per token as P-Q4 (2.69 GB), its experts on the Pascal Q4_K
   kernel.

The agent check on these lines decides, the Q8_0 and P-Q5 first. If their speeds are within the release line's
spread, the Q8_0 is the file. llama.cpp's own mixes (the makers publish Q4_K_M, Q5_K_M and Q6_K) read fewer bytes
per token but are 1.6–2.2× further from the Q8_0 than the file of the same size here.

## Files

| File | What |
|---|---|
| `kld_35b.tsv` | per file: GB, bits per weight, GB per token; per text the mean KLD ± SE, same top token %, perplexity of the file and of the reference (`run/table.py`) |
| `headers/header-Ornith-1.5-35B-*.txt` | [scripts/gguf_summary.py](../../../scripts/gguf_summary.py) on each file: metadata, tensor bytes per role and type, per block |
| `run/convert.sh` | BF16 safetensors to a BF16 GGUF (llama.cpp's `convert_hf_to_gguf.py` in a PyTorch container) |
| `run/quantize.sh` | one candidate from the BF16 GGUF (CPU llama-quantize in Docker); the recipes above |
| `run/candidates.sh` | every candidate in turn: quantize the next on the CPU while the GPUs score the previous one with [kld_texts.sh](../2026-10-27_ornith_kld/run/kld_texts.sh) |
| `run/calib_text.py`, `run/imatrix.sh` | the importance matrix's calibration text (agentic run 1's sessions outside the 40 held out) and its computation from the Q8_0 |
| `run/table.py` | the logs and headers to `kld_35b.tsv` |
| `run/tensor_md5.py` | MD5 of named tensors, to check that two GGUF files hold identical tensors |

## Made with

From the repository root on the quality machine (Docker, CUDA 12.9), with `W` a work folder on a fast disk, `M` the
models folder and `K` the folder holding the Q8_0's three bases and the agent and code texts
([2026-10-27_ornith_kld](../2026-10-27_ornith_kld/README.md) makes them):

```bash
D=docs/data/2026-10-28_ornith35_kld/run
SRC=$M/ornith-ai__Ornith-1.5-35B-A3B LLAMA=<llama.cpp checkout> OUT=$W/Ornith-1.5-35B-BF16.gguf PYLIB=$W/pylib \
  bash $D/convert.sh
# round 1, no importance matrix
export QUANT=<llama.cpp>/build-cpu/bin/llama-quantize SRC=$W/Ornith-1.5-35B-BF16.gguf OUT=$W/cand \
  REF=$M/ornith-ai__Ornith-1.5-35B-A3B-GGUF/Ornith-1.5-35B-Q8_0.gguf KLD=$K BIN=<llama.cpp>/build-cuda/bin/llama-perplexity \
  GPUS=3,4 WIKI=wikitext-2-raw/wiki.test.raw AGENT=$K/agent-heldout.txt CODE=$K/code-exercism.txt
RECIPES="P-Q5 Q6_K Q4_K_M Q5_K_M P-Q4" bash $D/candidates.sh
# the importance matrix, then round 2, one candidate at a time
python3 -I $D/calib_text.py <run 1 export>/hf/data/sessions.jsonl <run 1>/seqs/test-rows.jsonl $W/calib-agent-train.txt
BIN=<llama.cpp>/build-cuda/bin/llama-imatrix MODEL=$REF TEXT=$W/calib-agent-train.txt OUT=$W/imatrix-agent-train.gguf \
  GPUS=3,4 bash $D/imatrix.sh
for r in P-Q4-i Q4_K_M-i PK-Q4-i; do IMATRIX=$W/imatrix-agent-train.gguf RECIPES=$r bash $D/candidates.sh; done
for f in $W/cand/*.gguf; do python3 -I scripts/gguf_summary.py $f > headers/header-$(basename $f .gguf).txt; done
python3 -I $D/table.py $W/cand $K headers Q8_0 Q6_K P-Q5 Q5_K_M P-Q4 P-Q4-i PK-Q4-i Q4_K_M-i Q4_K_M > kld_35b.tsv
python3 -I $D/tensor_md5.py <file>.gguf blk.0.attn_qkv.weight output.weight token_embd.weight blk.30.ssm_out.weight
```

- **The machine:** the quality machine's two V100 32 GB for the scores and the importance matrix, its CPU at a low
  share for the conversion and the quantizations, with the box's llama.cpp (upstream `d77256999`), whose converter
  knows `Qwen3_5MoeForConditionalGeneration` (the text model and its MTP layer; the vision tower is left out).
- **Run time:** the conversion took 12 minutes (71.1 GB written), a candidate 2–8 minutes to quantize and ~16 minutes
  to score on the three texts, the importance matrix 35 minutes. Times on a shared machine are not speed results.
- **The headers** of the files with an importance matrix name it and its calibration text by file name; the GGUF
  metadata held the work folder's path.

## Provenance and licence

- **Model:** Ornith-1.5-35B-A3B by ornith-ai, MIT. BF16 safetensors from
  [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) at revision `10fbf86fed7e`; the
  Q8_0 from [ornith-ai/Ornith-1.5-35B-A3B-GGUF](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF) at revision
  `12393612fd4f` (37,802,149,280 bytes, SHA-256 `de46c4baf4b4dd85ea438bb0f757f21c38841a353506579979bba114311658c3`).
  The candidate files are not published.
- **Texts:** as in [2026-10-27_ornith_kld](../2026-10-27_ornith_kld/README.md). The calibration text comes from the
  same agentic run 1 (Flint's output, Apache-2.0, gated, publishable), from sessions disjoint from the scored ones; it
  is rebuilt by `run/calib_text.py` rather than stored here.
- **This folder:** the tables and scripts are under this repository's licence.
