# How close each Qwen3.8-Flash-Next file is to the original (2026-11-03)

The quality chain for Qwen3.8-Flash-Next: a Q8_0 reference of Swift-1.5, the KL divergence of every local Flash-Next
file against it, an importance matrix on agent text, and a Pascal-format file for two rigs built with it. The registry
rows are Q216 (the reference and its bases), Q217 (the published files), Q232 (the importance matrix), Q552 (the
two-rig file) and Q579 (the n-gram table's cost) in [quants/experiments.md](../../../quants/experiments.md).

## What runs

One detached job (`run/chain.sh`) on a 7-GPU quality machine (5 × RTX 3090, 2 × RTX 3080, 142 GiB of VRAM), step
after step:

| Step | What | Where |
|---|---|---|
| `fn1a_download` | ukisai's own Swift-1.5 Q8_0 ([ukisai/Swift-1.5-Qwen3.8-Flash-Next-GGUF](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GGUF), revision `0142554c963a8653912c0c8283f05f13f63653f3`, folder `Q8_0/`, five shards, 188.3 GB), each shard checked against the SHA-256 Hugging Face publishes for it | network, fast disk |
| `fn1a_check` | the same model? Seven tensors of [ukisai/Swift1.5-Qwen3.8-Flash-Next](https://huggingface.co/ukisai/Swift1.5-Qwen3.8-Flash-Next)'s BF16 safetensors, requantized bit for bit as the file stores them, must equal its bytes (`run/same_model.py`) | CPU |
| `fn1a_archive` | a copy on the bulk disk, checked against the SHA-256 again | disks |
| `tools` | `llama-imatrix`, `llama-quantize`, `llama-gguf-split` of the Miner Llama v3 base | CPU, Docker |
| `fn1b` | a fit check, then the reference's logits on the three texts (the bases) and the floor: the reference at `-ub 256` against them | all seven GPUs |
| `fn2` | each local file against the bases | all seven GPUs |
| `fn4` | an importance matrix from the reference on publishable agent sessions | all seven GPUs |
| `fn18` | the two-rig file: routed experts Q4_0 with that matrix, every other tensor as in the reference, the published n-gram table; then its KL divergence | CPU, then all seven GPUs |
| `fn22` | the reference with the published IQ4_NL n-gram table in place of its Q8_0 one, against the reference | all seven GPUs |

- **The reference** is ukisai's Q8_0 of Swift-1.5, not one converted here: the converter (`convert_hf_to_gguf.py`)
  needs PyTorch. `fn1a_check` tests that it is the model of the BF16 release: for two hyper-connection projections,
  an expert from each of two fused expert tensors, two routers and a shared expert (layers 3, 20 and 40), the BF16
  values written as the file's type (Q8_0 as ggml's `quantize_row_q8_0_ref`, which `llama-quantize` and the converter
  both follow; F32 by widening) must equal the file's bytes. The check itself was tested on files made from these
  weights: the published GSQ-RCO IQ3_XXS (its BF16 hyper-connections and F32 routers equal), the retyped IQ3_XXS of
  [2026-11-03_retype](../2026-11-03_retype/) (its Q8_0 hyper-connections equal), and a negative control (layer 3's
  weights against layer 4's tensor) that differs in every block.
- **The texts and flags** are those of [2026-10-27_ornith_kld](../2026-10-27_ornith_kld/README.md) and
  [2026-11-03_table_bits](../2026-11-03_table_bits/README.md): wikitext-2 test (80 × 512 tokens), held-out agent
  sessions (12 × 8,192) and the agent check's Exercism code (80 × 512); `llama-perplexity` with `-ub 512 -fa on`,
  every layer on the GPUs, built for compute 8.6 from the Miner Llama v3 base `37e717ff6`.
- **The same cards for everything.** A change of summation order alone moves Flash-Next's KL divergence to a floor
  ([2026-11-03_table_bits](../2026-11-03_table_bits/README.md): 0.016 / 0.023–0.025 / 0.004 on wiki / agent / code),
  and RTX 3080s and 3090s sum differently. So the bases and every file run on the same seven GPUs with the same
  layer split: the fit check's first split that runs is kept in `gpu_args` and passed to every later run. The
  reference at `-ub 256` against its own bases gives this setup's floor.
- **The local files** (`fn2`): Swift-1.5 GSQ-RCO Q2_0, IQ3_XXS, IQ2_XS and IQ3_S
  ([ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF)),
  ISTA's GSQ-RCO IQ3_S and Coder ("IQ1_M", 256 experts;
  [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)),
  and the two retyped Swift files of [2026-11-03_retype](../2026-11-03_retype/). ISTA's files are quantized from the
  original Qwen3.8-Flash-Next, not from Swift-1.5, so against this reference they measure the fine-tune's difference
  as well as their quantization.
- **The importance matrix** (`fn4`): 200 chunks of 512 tokens of agentic sessions outside the held-out ones the
  agent text is scored on (made by [2026-10-28_ornith35_kld](../2026-10-28_ornith35_kld/README.md)'s
  `run/calib_text.py`), so the agent-text scores stay out of sample. Besides each weight column's importance it counts
  the tokens each expert of each layer saw (Q232's expert usage).
- **The two-rig file** (`fn18`, Q552) is requantized from the Q8_0 (`llama-quantize --allow-requantize`), not made
  from the BF16: routed experts (`ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`) Q4_0 with the importance matrix,
  every other tensor copied from the reference (`llama-quantize` copies tensors already in the target type and never
  touches `hc_*`). Its n-gram table shard is then replaced by the table every published GSQ-RCO build ships (IQ4_NL,
  28.8 GB) with `run/swap_tensor.py`; swapping that table into its own shard reproduces the shard byte for byte (MD5
  `98ce140e641bc5d68af9f31df0f1c048`).

## Result

KL divergence against the Swift-1.5 Q8_0 reference, mean ± its error, on the three texts
([kld.tsv](kld.tsv) adds the same-top-token rate and the perplexity ratio). The floor is the reference at `-ub 256`
against its own bases. Which file to run on the rig, from these scores, the fit and the speed model:
[2026-11-04_file_choice](../2026-11-04_file_choice/README.md).

| File | Agent | Wiki | Code |
|---|---:|---:|---:|
| floor (`q8_0-ub256`) | 0.0291 ± 0.0011 | 0.0170 ± 0.0003 | 0.0036 ± 0.0002 |
| the reference with the published IQ4_NL n-gram table (FN22) | 0.0347 ± 0.0013 | 0.0269 ± 0.0006 | 0.0070 ± 0.0004 |
| ISTA IQ3_S | 0.1165 ± 0.0025 | 0.1614 ± 0.0026 | 0.0320 ± 0.0015 |
| Swift IQ3_S | 0.1210 ± 0.0028 | 0.1535 ± 0.0027 | 0.0316 ± 0.0015 |
| Swift IQ3_XXS | 0.1755 ± 0.0038 | 0.2477 ± 0.0039 | 0.0544 ± 0.0023 |
| Swift IQ3_XXS-hcq8 (FN17) | 0.1848 ± 0.0040 | 0.2445 ± 0.0039 | 0.0531 ± 0.0023 |
| Swift IQ2_XS | 0.2248 ± 0.0046 | 0.3512 ± 0.0050 | 0.0750 ± 0.0028 |
| ISTA Coder "IQ1_M" (256 experts) | 0.2735 ± 0.0052 | 0.8455 ± 0.0099 | 0.1308 ± 0.0047 |
| Swift Q2_0-hcq8 (FN17) | 0.2827 ± 0.0055 | 0.4284 ± 0.0058 | 0.1120 ± 0.0039 |
| Swift Q2_0 | 0.2851 ± 0.0056 | 0.4283 ± 0.0059 | 0.1124 ± 0.0040 |
| the two-rig file, Q4_0 experts (FN18) | stopped (GPU fault) | 0.0608 ± 0.0011 | not run |

- **The published IQ4_NL n-gram table costs little:** +0.006 agent, +0.010 wiki and +0.003 code over the floor (FN22,
  Q579).
- **Swift IQ3_XXS is the best file that fits nine cards** (0.176 on agent text, 1.6× below Q2_0); both IQ3_S files are
  better still but do not fit. ISTA's files are quantized from the original Qwen3.8-Flash-Next, so their scores hold
  the fine-tune's difference as well.
- **FN17's retypes cost nothing measurable:** Q2_0-hcq8 equals Q2_0 within its error on all three texts, IQ3_XXS-hcq8
  equals IQ3_XXS on wiki and code and is 0.009 higher on agent text (about twice the error).
- **The two-rig file (FN18)** scored 0.061 on wiki, 4× below Swift IQ3_XXS's 0.248; its agent run stopped on a GPU fault on
  2026-11-04 and its code run did not start, so its agent and code scores are still open (Q552).

## Files

| File | What |
|---|---|
| [kld.tsv](kld.tsv) | per file and text, from each `llama-perplexity` log's summary (`Mean KLD`, `Same top p`, `Mean PPL(Q)/PPL(base)`): the value and its error for each, and a note |
| `run/chain.sh` | the driver: steps, prerequisites, a log and a done or failed marker per step, restartable |
| `run/common.sh` | shared helpers: logging, Docker arguments, waiting until other jobs' containers have left the GPUs, then a CUDA pre-flight on them ([`scripts/cuda_preflight.sh`](../../../scripts/cuda_preflight.sh), run when `PREFLIGHT` is set) |
| `run/fn1a_download.sh`, `run/fn1a_check.sh`, `run/fn1a_archive.sh` | the reference: download, identity check, archive copy |
| `run/same_model.py` | is a GGUF the model of a BF16 safetensors folder: named tensors requantized and compared byte for byte |
| `run/layout.py` | a split GGUF's tensors and bytes per type, the n-gram table's shard, the bytes on the GPUs |
| `run/tools.sh` | the three llama.cpp tools of the v3 base, from an existing build folder of the same commit and flags |
| `run/kld.sh` | the bases or a file's KL divergence on the three texts, in Docker |
| `run/fn1b.sh`, `run/fn2.sh`, `run/fn4.sh`, `run/fn18.sh`, `run/fn22.sh` | the steps above |
| `run/swap_tensor.py` | replaces a tensor that sits alone in its shard with the same-named tensor of another GGUF |
| `run/table.py` | the KL divergence logs to one TSV |
| `run/status.sh` | the chain's state and every finished score on one screen |

## Commands

On the quality machine, with `W` a work folder on a fast disk (the reference, the bases and the copies of files kept
on hard disks are read hard there), an environment file `ENV` that sets the variables the scripts name (`TAG`, `WORK`,
`GPUS_ALL`, `MOUNTS`, `Q8_REPO`, `Q8_REV`, `Q8_SUBDIR`, `Q8_HOT`, `Q8_ARCHIVE`, `Q8_LIST`, `REF`, `BF16`, `PPL`,
`TOOLS`, `SRC`, `BUILD_FROM`, `CCACHE_FROM`, `WIKI`, `AGENT`, `CODE`, `KLD`, `HOT`, `FAST`, `FN2_FILES`, `CALIB`,
`PUBLISHED_TABLE`; optionally `PREFLIGHT`, the path of `scripts/cuda_preflight.sh`, and `PREFLIGHT_CACHE`, where
it keeps its compiled probe), and `Q8_LIST` holding one `<sha256> <bytes> <file>` line per shard:

```bash
mkdir -p $W/logs
ENV=$ENV setsid -f nohup bash run/chain.sh >> $W/logs/chain.log 2>&1 < /dev/null
WORK=$W bash run/status.sh                       # progress
python3 -I run/table.py $W/kld > kld.tsv         # the scores
```

The weights are under the Swift Open License 1.0 and qwen-community-1.0 (Swift-1.5) and ISTA's terms for its files;
the tables here hold only scores and statistics, no weights.
