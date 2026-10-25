# Rebuilding K2h-r16 from Flint's original weights

K2h-r16 is this project's Flint file: Q5_0 and Q8_0 tensors, the formats the Pascal kernels decode fastest, chosen
tensor group by tensor group to stay below Unsloth's UD-Q4_K_XL in error (KL divergence 0.105 against 0.120) while
reading fewer bytes per token. This page rebuilds it from Flint's BF16 weights with public tools, and checks the result
against the shipped file tensor by tensor.

`build_k2h_r16.sh` runs every step. The rest of this page says what each step does, where its inputs come from, and
what to check.

## What you need

| Input | Where | Check |
|---|---|---|
| Flint's weights | [RecursiveSI/flint](https://huggingface.co/RecursiveSI/flint), revision `2a247be0c0aec895ccb0ac7ca92229ec6df37288` (15 safetensors, 79.6 GB) | the SHA-256 of each file in [inputs.sha256](inputs.sha256) (the Hub shows the same per file) |
| Unsloth's importance matrix | `imatrix_unsloth.gguf` from [unsloth/flint-GGUF](https://huggingface.co/unsloth/flint-GGUF) (225 MB) | [inputs.sha256](inputs.sha256) |
| llama.cpp | upstream commit `d77256999` (2026-10-22): `convert_hf_to_gguf.py`, `llama-quantize` (a CPU build is enough), gguf-py | — |
| Python | numpy; torch and safetensors for the conversion (llama.cpp's `requirements/requirements-convert_hf_to_gguf.txt`) | — |
| Disk | ~150 GB: the BF16 GGUF (79.6 GB) and two ~31 GB stages | — |
| GPU | none for the byte-exact rebuild; CUDA GPUs with ~35 GB in total (the 31 GB body plus buffers) to compute the output head yourself | — |

```bash
git clone https://github.com/ggml-org/llama.cpp && git -C llama.cpp checkout d77256999
cmake -S llama.cpp -B llama.cpp/build -DGGML_NATIVE=ON && cmake --build llama.cpp/build -j --target llama-quantize
```

## The steps

### 1. BF16 GGUF

```bash
python llama.cpp/convert_hf_to_gguf.py flint/ --outtype bf16 --outfile flint-BF16.gguf
```

79,625,761,760 bytes. The SHA-256 of the project's own conversion is in [inputs.sha256](inputs.sha256); a conversion
at the same commit is expected to give the same bytes, but that was not repeated here. Every quality number of the
project is measured against this file's logits.

### 2. The body: one llama-quantize run

Base type Q5_0 with Unsloth's importance matrix, and every tensor's type from [k2h-r16.types](k2h-r16.types) (one
`--tensor-type '^name$=type'` per tensor; `tensor_manifest.py quantize-args` writes them). The output head is written as
Q8_0 here and replaced in step 3; the routers stay F32 until step 4.

The mix, by tensor group (layers are Flint's 0–48):

| Group | Type | Why |
|---|---|---|
| Routed experts, gate and up | Q5_0; Q8_0 in layers 2–8 | the cheapest bulk format at the quality bar; the early layers are the most sensitive |
| Routed experts, down | Q5_0; Q8_0 in layers 46–48 | the last layers' outputs feed the head |
| Shared expert (gate, up, down) | Q8_0; Q5_0 in layers 33–45 | read for every token |
| Dense FFN of layers 0–1 | Q8_0 | |
| Attention q | Q8_0 in layers 9–24; Q5_0 in 0–8 and 25–48 | |
| Attention k and v | Q8_0; Q5_0 in layers 33–48 | |
| Attention output | Q8_0 in layers 0–32; Q5_0 in 33–48 | |
| Token embedding | Q5_0 | read one row per token |
| Output head | GPTQ-rounded Q5_0 (step 3) | 150,144 × 2,048, read in full every token |
| Routers (`ffn_gate_inp`) | F16 (step 4) | |
| Norms and small vectors | F32 | |

How the groups were chosen: per-group KLD costs measured one group at a time against the bytes each saves
([Q73, Q77 in the registry](../../../quants/experiments.md#registry)); G3 is the frontier mix, K2 drops the three
cheapest Q8_0 groups to Q5_0 (shared expert 33–45, k/v 33–48, q 0–8).

The same body was first built in two runs and composed, and the one-run command reproduces it:
- `G3`: llama-quantize Q5_0 with the overrides `blk\.([0-9]|1[0-9]|2[0-4])\.attn_q\.weight=q8_0`,
  `blk\.[2-8]\.ffn_(gate|up)_exps\.weight=q8_0`, `blk\.(3[3-9]|4[0-8])\.attn_output\.weight=q5_0`,
  `--output-tensor-type q8_0`, `blk\.(46|47|48)\.ffn_down_exps\.weight=q8_0`, `attn_(k|v)\.weight=q8_0`,
  `_shexp=q8_0`, `blk\.[01]\.ffn_(gate|up|down)\.weight=q8_0`, `attn_output\.weight=q8_0`, `ffn_down_exps=q5_0`,
  `ffn_(gate|up)_exps=q5_0` (llama-quantize uses the first pattern that matches);
- `Z`, the donor: Q5_0 everywhere except the head and the overflow-protected tensors;
- `K2` = `compose_gguf.py G3 K2 'blk\.(33|…|45)\.ffn_(gate|up|down)_shexp\.weight=Z' 'blk\.(33|…|48)\.attn_(k|v)\.weight=Z'
  'blk\.(0|…|8)\.attn_q\.weight=Z'` ([compose_gguf.py](../../../scripts/compose_gguf.py)).

llama-quantize quantizes each tensor on its own, so a composed file equals one run with the same per-tensor types.

### 3. The output head: GPTQ-rounded Q5_0

Round-to-nearest Q5_0 costs +0.023 KLD on Flint's head; GPTQ rounding of the same format costs +0.002
([quants/gptq_head](../../../quants/gptq_head/README.md), [Q56](../../../quants/experiments.md#registry)).

- **Byte-exact:** take `output.weight` from the shipped file (`HEAD_FROM=flint-K2h-r16.gguf`).
- **From scratch:** `quants/gptq_head/run.sh` on the step-2 body:
  1. `llama-head-dump` (in `quants/gptq_head/head_dump/`, built inside a llama.cpp tree with CUDA) writes the final
     hidden state after the output norm for every token of 500 chunks of 512 tokens of wikitext-2 train (255,500 rows)
     and 40 later chunks (held out);
  2. H = Σ x xᵀ in float64 (2,048 × 2,048);
  3. GPTQ on the BF16 head: Q5_0 blocks of 32 as GPTQ groups, column order, 1 % damping, block scales by ggml's
     weighted search (`--scale search`);
  4. the head alone on the held-out rows against the BF16 head (KLD 0.0030, 97.35 % same top token on wikitext for
     the shipped head's recipe).

  The shipped head was calibrated on the head inputs of F3, an earlier mix of the same family, on wikitext-2 train
  chunks 0–499 plus Flint's own outputs: the 38 transcripts of
  [bench/expert_overlap](../../../bench/expert_overlap/README.md) (81,723 rows, regenerated by its `prompts.py` and
  `generate.py`). Wikitext alone gives the same full-model quality within noise (KLD 0.1133 against 0.1136 on F3), so
  a head computed from scratch is equivalent but not byte-identical: GPU dumps and sampled transcripts vary in the
  last bits between GPUs.

### 4. The routers in F16

```bash
python scripts/retype_tensors.py flint-k2h.gguf flint-k2h-r16-raw.gguf 'ffn_gate_inp=F16'
```

The 47 routers are F32 in llama-quantize's output (99 MB per token); F16 halves that at no measurable cost (KLD 0.1048
against 0.1092 with F32, within the ~0.005 routing-flip noise; [Q113](../../../quants/experiments.md#registry)).

### 5. The release header

```bash
python scripts/gguf_release_header.py flint-k2h-r16-raw.gguf flint-K2h-r16.gguf --preset flint --name "Flint K2h-r16" --quantized-by "<who>"
```

Sets `general.license = apache-2.0` and the base model (Flint by RecursiveSI), writes the makers' sampling
(`general.sampling.temp` 1.0, `top_p` 0.95, `top_k` 20, `min_p` 0), and keeps `quantize.imatrix.*`, which credits
Unsloth's importance matrix. The tensors are copied byte for byte and verified.

## Checking the result

**Tensor by tensor:** `tensor_manifest.py hash flint-K2h-r16.gguf` against [k2h-r16.tensors.tsv](k2h-r16.tensors.tsv)
(`build_k2h_r16.sh`'s `verify` step). With `HEAD_FROM` every tensor must be equal; with a computed head, every tensor
but `output.weight`. The header is not compared: names and tool versions in it differ between builds.

**Quality,** against the BF16 file's logits on 80 × 512 tokens of wikitext-2 test:

```bash
llama-perplexity -m flint-BF16.gguf -f wiki.test.raw -c 512 --chunks 80 -ngl 99 -fa on --kl-divergence-base flint-BF16.c512x80.kld
llama-perplexity -m flint-K2h-r16.gguf -f wiki.test.raw -c 512 --chunks 80 -ngl 99 -fa on --kl-divergence-base flint-BF16.c512x80.kld --kl-divergence
```

| | Mean KLD | Same top token | PPL vs BF16 |
|---|---|---|---|
| Shipped K2h-r16 | 0.1048 | 88.84 % | +2.65 % |
| A rebuild with a computed head | 0.105 ± 0.002 | 88.8 ± 0.2 % | |
| Unsloth UD-Q4_K_XL, for comparison | 0.120 | 87.7 % | +3.9 % |

The BF16 reference run needs ~80 GB of GPU memory across cards; a rebuild with `HEAD_FROM` needs no GPU at all.

**Speed per token:** 2.091 GB read per generated token (`scripts/token_cost.py`), against 2.213 for UD-Q4_K_XL.

## Time and size

| Step | Time | Output |
|---|---|---|
| Download | ~25 min at 50 MB/s | 79.6 GB safetensors |
| Convert | TBD | 79.6 GB |
| Quantize | TBD | ~31 GB |
| Head (computed) | one model pass over ~260k tokens on GPUs for the dumps, then minutes of CPU for H and GPTQ | 0.21 GB |
| Router, header, verify | minutes each | 29.4 GB |
