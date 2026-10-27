# Flint quantizations: KL divergence per chunk and in full (2026-10-22)

The quality columns of the Flint page's quantization table and of its "Fewer experts per token" table
([2026-10-22-flint.md](../../benchmarks/2026-10-22-flint.md)) come from seven `llama-perplexity --kl-divergence`
runs against Unsloth's Q8_0 on 80 chunks of 512 tokens of wikitext-2. This folder holds every statistic those runs
printed, overall and after each chunk, so the spread across chunks and the tails of the distribution can be checked.

| File | What |
|---|---|
| `kld_summary.csv` | one row per file: chunks, perplexity of the file and of Q8_0 (± standard error), their ratio and log ratio, mean KL divergence ± standard error, its maximum, 99.9th, 99th, 95th, 90th, median and 10th percentile, RMS of the change in the top token's probability (%), same top token as Q8_0 (%) |
| `kld_by_chunk.tsv` | 560 rows (7 files × 80 chunks): what `llama-perplexity` prints after each chunk, i.e. running means over chunks 1..N of perplexity, log perplexity ratio, KL divergence, RMS Δp and same top token, each with its standard error |
| `run/kld_chunks.py` | builds both files from the `*.kld.log` files |

The seven files (names as in the tables):

| `file` | What it is |
|---|---|
| `flint-int4-q4_1` | the makers' Q4_1 ([RecursiveSI/flint-int4-gguf](https://huggingface.co/RecursiveSI/flint-int4-gguf)) |
| `flint-int4-q4_1_e5`, `flint-int4-q4_1_e4` | the same file with 5 or 4 experts per token instead of 6 (`--override-kv flint.expert_used_count=int:N`) |
| `flint-UD-Q4_K_XL` | Unsloth's [UD-Q4_K_XL](https://huggingface.co/unsloth/flint-GGUF) |
| `flint-imat-Q4_0`, `flint-imat-Q4_1`, `flint-imat-Q5_0` | requantized from the Q8_0 file with Unsloth's importance matrix, output head at Q8_0, everything else in one format |

Made on 2026-10-22 on a PC (CPU only; quality does not depend on the GPU) with
[scripts/quality_kld.sh](../../../scripts/quality_kld.sh) (default 80 chunks of 512 tokens), the imatrix files with
[scripts/quantize_docker.sh](../../../scripts/quantize_docker.sh):

```bash
scripts/quality_kld.sh <llama.cpp source> <models dir> <work dir> unsloth/flint-GGUF/flint-Q8_0.gguf \
    RecursiveSI/flint-int4-gguf/flint-int4-q4_1.gguf unsloth/flint-GGUF/flint-UD-Q4_K_XL.gguf \
    custom/flint/flint-imat-Q4_0.gguf custom/flint/flint-imat-Q4_1.gguf custom/flint/flint-imat-Q5_0.gguf
for n in 5 4; do
    CAND_ARGS="--override-kv flint.expert_used_count=int:$n" TAG=_e$n \
        scripts/quality_kld.sh <llama.cpp source> <models dir> <work dir> unsloth/flint-GGUF/flint-Q8_0.gguf \
        RecursiveSI/flint-int4-gguf/flint-int4-q4_1.gguf
done
python3 run/kld_chunks.py . <work dir>/*.kld.log
```

Model outputs are not included; the numbers are statistics of Flint's (Apache-2.0) next-token distributions on
wikitext-2 (CC BY-SA 4.0).
