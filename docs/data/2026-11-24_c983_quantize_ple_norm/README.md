# llama-quantize keeps Flash-Next's ple_norm tensors unquantized (C983, Q1716; 2026-11-24)

What this is: one rule in Miner Llama's quantizer and a test for it, on the fork branch
`pascal-p106-v5-quantize-ple-norm`, checked on a build machine's processor. Flash-Next (llama.cpp's `qwen4exp`
architecture) has three per-layer-embedding norm tensors, `blk.N.ple_norm_key`, `ple_norm_query` and `ple_norm_conv`,
which the graph multiplies elementwise. `llama-quantize`'s default rules quantized them whenever they were stored as a
matrix, and the file then stopped at its first token. Registry row Q1716 in
[quants/experiments.md](../../../quants/experiments.md); the fault was found by C957
([2026-11-23_c957_batch_api_gpu](../2026-11-23_c957_batch_api_gpu/README.md)).

## Result in plain words

- **A Flash-Next file quantized with the default rules now runs whatever shape its norm tensors have.** On a random
  Flash-Next model the old rules gave a file that aborted at its first token; with the new rule the same command gives
  a file that runs. It is byte for byte the file the old rules make when told to keep the norms F32
  (`--tensor-type ple_norm=f32`, C957's workaround).
- **Files made from the real model were never hit.** The original checkpoint stores these tensors flat (one row of
  10,240 values per layer), the converter writes them that way as F32, and `llama-quantize` never quantizes a 1-row
  tensor. Only files that store them as a 4 × 2,560 matrix were hit, such as the random models of the fork's own tests.
- **Nothing else changes.** On 150 random models of every architecture, each quantized by both builds, the only
  tensor types that differ are qwen4exp's three norms (Q8_0 before, F32 after). ctest gives the same result for every
  test, except the new one, which fails before and passes after.
- **Upstream has the same gap.** Its newest tag `b12679` (2026-11-23) has the same rule, and no pull request or issue
  mentions these tensors.

## The fault and its reach

`tensor_allows_quantization` in `src/llama-quant.cpp` keeps every tensor whose name contains `_norm.weight`
unquantized. Flash-Next's per-layer-embedding norms are named `ple_norm_key.weight`, `ple_norm_query.weight` and
`ple_norm_conv.weight`, so that rule misses them, although `llama-arch.cpp` lists them as `GGML_OP_MUL` operands like
the other norms. Quantized to Q8_0, they reach an elementwise multiply with a Q8_0 operand, which no backend computes:
the CPU backend aborts with `binary_op: unsupported types: dst: f32, src0: f32, src1: q8_0`.

Whether the rules reach them depends on how the file stores them. The loader accepts them flat or as `{n_embd, hc}`
(`TENSOR_ALLOW_RESHAPE`), and `llama-quantize` quantizes only tensors with two or more dimensions:

| File | `ple_norm_key` of layer 1 |
|---|---|
| the original checkpoint, Swift-1.5 Qwen3.8-Flash-Next (safetensors) | BF16, shape [10240] |
| Unsloth Qwen3.8-Flash-Next UD-Q5_K_XL | F32, 10240 |
| Swift-1.5-Qwen3.8-Flash-Next Q8_0 | F32, 10240 |
| Swift Qwen3.8-Flash-Next GSQ-RCO Q2_0, IQ3_S and IQ1_M | F32, 10240 (all three norms) |
| a random qwen4exp model from `test-llama-archs` | F32, 256 × 4 (`n_embd` × `hc`) |

So a reader who converts the real model and quantizes it with the default rules gets F32 norms and a file that runs,
on the old rules as on the new. A file whose norms are stored 2-D, as `test-llama-archs` writes them, got them
quantized and stopped. The rule makes the result independent of the stored shape.

## The change

Two commits on `pascal-p106-v5` at `8a9629942`:

- `9320d0e05` **the test**: `test-llama-archs --quantize` saves each random model as F32, quantizes it with
  `llama_model_quantize`'s default rules (Q8_0, no tensor named), loads it on the CPU and decodes one token. The file
  must load, decode, give finite logits and hold at least one quantized tensor. A tensor the rules wrongly quantize
  aborts the decode, which fails the test. The new ctest `test-quantize-default-rules` runs it on qwen4exp, qwen35moe
  and flint.
- `91eead67a` **the rule**: `quantize &= name.find("ple_norm_") == std::string::npos;` next to the `_norm.weight` rule.

## The checks

[run/run.sh](run/run.sh) on a build machine's processor, every program in a container without network. Two trees:
**base** is the test commit (`9320d0e05`: the old rules plus the new test), **head** the rule commit (`91eead67a`).

| Check | Result |
|---|---|
| build, every target, static ([checks.tsv](checks.tsv)) | both OK, no error lines |
| ctest, test by test ([ctest.tsv](ctest.tsv)) | 76 tests, the same result in both but one: `test-quantize-default-rules` aborts on base (qwen4exp) and passes on head. Base 73 passed, head 74. `test-kv-cache` (aborts) and `test-kv-cache-scales` (fails) fail in both, as on the same base in [C929's run](../2026-11-24_c929_rpc_graph_ids/README.md) |
| `test-llama-archs -b CPU -s 1`, logits written and compared | 153 of 153 cases pass in both; head's 148 logits files the same as base's |
| every random model quantized to Q8_0 by both builds' `llama-quantize` ([types.tsv](types.tsv)) | 150 files (147 models of every architecture and 3 with a 4,096-token vocabulary), 11,029 tensors: the same type in both builds except 3 tensors of `qwen4exp-moe` (both vocabularies): `blk.0.ple_norm_conv`, `ple_norm_key`, `ple_norm_query`, Q8_0 on base and F32 on head (36 and 33 of 103 tensors quantized). `qwen35moe-moe` (21 of 77 quantized) and `flint-moe` (19 of 76) have the same type for every tensor |
| one token of `llama-completion` on each Q8_0 file, with its own build ([cli.tsv](cli.tsv)) | base: `qwen4exp-moe` aborts (exit 134, `binary_op: unsupported types: dst: f32, src0: f32, src1: q8_0`), `qwen35moe-moe` and `flint-moe` run. Head: all three run |
| base's `llama-quantize --tensor-type ple_norm=f32` against head's default rules ([override.tsv](override.tsv)) | the same file, md5 for md5, for all three models |
| `test-llama-archs --quantize -s 1`, each architecture in its own process ([sweep.tsv](sweep.tsv)) | 174 architectures, 143 models in 136 of them (the rest have no model the test can save and quantize). Base: 142 OK, `qwen4exp` MoE aborts. Head: 143 OK |

The models are `test-llama-archs -s 1`'s: every architecture, as ctest's `test-generate-models` writes them but with seed 1, and
qwen4exp, qwen35moe and flint with a 4,096-token vocabulary (`LLAMA_TEST_ARCHS_N_VOCAB=4096`) from each tree's build,
the same files from both.

## Files

| File | What |
|---|---|
| [checks.tsv](checks.tsv) | `cpu_tests.sh`'s verdicts per tree: build, ctest, test-llama-archs, logits against base |
| [ctest.tsv](ctest.tsv) | every ctest test's result in base and head |
| [types.tsv](types.tsv) | per quantized model: exit codes, tensor count, quantized tensors in each build, the tensors whose type differs |
| [cli.tsv](cli.tsv) | one token of `llama-completion` per Q8_0 file and build: exit code, first error line |
| [override.tsv](override.tsv) | md5 of base's file with `--tensor-type ple_norm=f32` and of head's default file |
| [sweep.tsv](sweep.tsv) | `test-llama-archs --quantize` per architecture: exit code and result rows in each build |
| [run/run.sh](run/run.sh) | the build machine's driver: sources, builds, ctest, models, types, the token runs, the sweep |
| [run/c983_tables.py](run/c983_tables.py) | the tables above from a copy of the driver's work folder |

The driver also uses [cpu_tests.sh](../2026-11-05_c565_ladder4fix/run/cpu_tests.sh) (builds, ctest, test-llama-archs)
and [model_types.py](../2026-11-23_c957_batch_api_gpu/run/model_types.py) (a GGUF file's tensor types from its header).

## Commands

```bash
# on the build machine: W holds everything, REPO (inside W) run.sh, cpu_tests.sh and model_types.py
W=$WORK SEED=<a llama.cpp clone> REPO=$WORK/repo CPU_CCACHE=<a compiler cache> LOCK=<the build lock> \
  setsid nohup bash $WORK/repo/run.sh > $WORK/run.out 2>&1 < /dev/null &
# the tables, from a copy of $WORK without sources, builds and models
python3 -I docs/data/2026-11-24_c983_quantize_ple_norm/run/c983_tables.py docs/data/2026-11-24_c983_quantize_ple_norm $COPY
```

The builds (each tree's `git archive`, Docker image Ubuntu 24.04 with gcc 13.3.0 and cmake 3.28.3):

```bash
cmake -S src -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=ON \
  -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF
cmake --build build -j 8 -- -k                                                    # cpu_tests.sh: every target
cmake --build build -j 8 --target llama-quantize llama-completion test-llama-archs  # the quantize checks
```

The checks by hand, with one build's `bin`:

```bash
LLAMA_TEST_ARCHS_N_VOCAB=4096 bin/test-llama-archs -a '^qwen4exp$' -s 1 -o models
bin/llama-quantize models/qwen4exp-moe.gguf qwen4exp-moe-q8_0.gguf Q8_0 4
bin/llama-completion -m qwen4exp-moe-q8_0.gguf -p 'The quick brown fox' -n 1 -c 256 -t 4 -no-cnv --temp 0
bin/test-llama-archs --quantize -s 1 -a '^(qwen4exp|qwen35moe|flint)$'
```

It ran on 2026-11-24 07:55–07:59, from a warm compiler cache (97 % hits in the builds of every target, 95 % in the quantize builds).

## Provenance and licence

The engine is Miner Llama (llama.cpp's MIT licence). The models are random-weight test models from the fork's
`test-llama-archs` (seed 1) and their Q8_0 copies. The real files' tensor shapes were read from their headers only (the
GGUF tensor table, the safetensors header). No model file, download or private data. Tables and scripts here: the
repository's licence.
