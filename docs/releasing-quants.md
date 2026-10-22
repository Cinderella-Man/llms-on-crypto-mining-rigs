# Releasing a quantization for these rigs

This project may eventually publish GGUF files tuned for Pascal mining cards: formats the 5-SM P106 decodes quickly,
with an importance matrix to keep quality. This page is the checklist and model-card template for doing that well.
Nothing has been published yet.

## Before a file is worth publishing

- [ ] **Speed on the rig**: `llama-bench` on the smallest card count that fits and on all nine cards, and real chat
      requests through `llama-server` at a long context, with the build and flags recorded
      ([scripts/bench.sh](../scripts/bench.sh), [scripts/server_bench.sh](../scripts/server_bench.sh)).
- [ ] **Quality against a reference**: perplexity and KL divergence against Q8_0 or BF16 on the same text
      ([scripts/quality_kld.sh](../scripts/quality_kld.sh)), with the reference named.
- [ ] **A task check**, because KL divergence alone can mislead (some vendor quants are tuned away from the original
      weights on purpose): a fixed set of coding, tool-calling and reasoning prompts with pass/fail scoring.
- [ ] **Numerical safety**: a GPU perplexity run with the exact build users will run, to catch overflows such as the
      Flint down-projection case (NaN output when the expert down-projection is Q4_x/Q5_x without llama.cpp's rescale).
- [ ] **A clear win**: faster than the best published file at similar quality, or better quality at similar speed.
- [ ] **License check**: the base model's license allows redistribution of modified weights (Flint: Apache-2.0); credit
      the base model and any importance matrix used.

## Recipe

Record the exact commands so anyone can reproduce the file:

```bash
# importance matrix: name the source (e.g. Unsloth's imatrix_unsloth.gguf_file for Flint) or the calibration text
EXTRA="--output-tensor-type q8_0 --tensor-type ffn_down_exps=q8_0" \
  scripts/quantize_docker.sh <llama.cpp source> <models dir> <input Q8_0 or BF16 gguf> <output gguf> Q4_1 <imatrix>
```

## Model card template

```markdown
# <Model>-P106-<format>-GGUF

<Model> (<link to base model>) quantized for NVIDIA Pascal GPUs with few streaming multiprocessors, such as the
P106-090 mining cards: <one sentence on the format choice and why it is fast on these cards>.

## Files
| File | Size | Format mix | Speed on 9 × P106-090 (tokens/s) | KL divergence vs <reference> | Same top token |

## How it was made
<llama.cpp commit, llama-quantize command, importance-matrix source>

## How to run it
<llama.cpp version or build; flags: CUDA-graphs build, -ub 128 for long contexts, -bs, --cache-ram 0, ...>

## Measurements
<links to the benchmark pages in llms-on-crypto-mining-rigs>

## License
<base model license>; derived from <base model> by <author of base model>; importance matrix from <source>.
```
