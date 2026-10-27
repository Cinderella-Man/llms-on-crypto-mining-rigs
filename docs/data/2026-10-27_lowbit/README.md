# Q164: a 2–3-bit format for Pascal, the CPU quality proxy (2026-10-27)

The data behind [pascal_lowbit_format.md](../../research/notes/pascal_lowbit_format.md) section 4 and registry row Q164
in [quants/experiments.md](../../../quants/experiments.md): the importance-weighted reconstruction error of 2-, 3- and
4-bit formats on real routed-expert matrices of Ornith-1.5-122B, and the check that Miner Llama's N2P/N3P quantizer in
C gives the same errors as the proxy's numpy version. CPU only; no model output and no speed is measured here.

| File | What |
|---|---|
| `proxy-main.tsv` | the main run: 32 formats (llama.cpp's own and the candidates) × 84 matrices (layers 2, 10, 18, 26, 34, 42, 47; four experts each; gate, up, down) |
| `proxy-shift.tsv` | the shifted-table formats (ik_llama.cpp's IQ2_K/IQ3_K levels with and without the shift bit) on the same 84 matrices |
| `proxy-first-layer2.tsv` | the first run, 43 formats on layer 2 only (12 matrices), stopped for time; its six Hadamard-rotated variants are the rotation evidence |
| `proxy-shift-2layers.tsv` | the first look at the shift bit, layers 10 and 34 (24 matrices) |
| `c-quantizer.tsv` | fork branch `pascal-p106-v3-lowbit`'s N2P/N3P quantizer (through `libggml-base`) against the proxy's `ik2x_s4x8`/`ik3x_s5x32` on layers 10 and 34: identical errors |
| `summary.txt` | `lowbit_summary.py` over `proxy-main.tsv` and `proxy-shift.tsv` together (per format, the expert mixes), then the per-layer error ratios |
| `tables.json` | the N2P/N3P-style tables fitted on layer 6 (`nl2`, `nl3`), used by every run |
| `run/proxy.sh` | the runs: CPU-only `libggml-base`, `scripts/lowbit_proxy.py`, `scripts/lowbit_summary.py` |
| `run/check_n2p_n3p.py` | the CPU consistency check of N2P/N3P: vec_dot against the dequantized dot product (≤ 1e-5 relative), Gaussian round trip, an importance vector |

TSV columns: `layer`, `expert`, `kind` (gate, up, down), `count` (tokens the importance matrix routed to the expert),
`format`, `bpw`, `err_imp` (importance-weighted relative error, the result), `err_plain` (unweighted), `seconds`.

## How it was made

- **Weights:** `ornith-ai/Ornith-1.5-122B` BF16 safetensors (MIT), read one expert slice at a time (about 0.5 GB per full
  run). Experts per layer: the two the importance matrix routed to most and two random ones (seed 7).
- **Importance matrix:** `imatrix_unsloth.gguf_file` from `unsloth/Ornith-1.5-122B-GGUF` (694 chunks of 512 tokens);
  imp_j = in_sum2 / counts for the expert.
- **Reference formats:** llama.cpp's quantizers and dequantizers (`ggml_quantize_chunk`) from a CPU-only `libggml-base`
  of llama.cpp `d77256999`; N2P/N3P (`c-quantizer.tsv`) from fork branch `pascal-p106-v3-lowbit` at `67181fcd9`.
- On a separate CPU machine (Docker, numpy 1.26, 8 worker processes, low CPU priority).

Commands (`run/proxy.sh`, with `QM`, `WORK`, `SRC`, `MODEL`, `IMATRIX` set):

```bash
run/proxy.sh lib
RUN=run1 ARGS="--formats <run 1 list below>" run/proxy.sh proxy   # stopped after layer 2's 12 matrices: proxy-first-layer2.tsv
RUN=run2 run/proxy.sh proxy                             # defaults of lowbit_proxy.py: proxy-main.tsv
RUN=run3 ARGS="--layers 10,34 --formats IQ2_S,IQ2_XS,IQ3_XXS,IQ3_S,Q3_K,nl2_s4x16,nl2_s4x8,nl3_s6x32,nl3_s4x32,nl3_s6x16,nl2x_s4x16,nl2x_s4x8,ik2_s4x16,ik2x_s4x16,nl3x_s6x32,nl3x_s4x32,ik3x_s5x16,ik3x_s6x32,ik3_s6x32" run/proxy.sh proxy
RUN=run4 ARGS="--formats ik2x_s4x16,ik2x_s5x16,ik2x_s6x16,ik2_s4x8,ik2x_s4x8,ik3_s6x32,ik3x_s4x32,ik3x_s5x32,ik3x_s6x32,ik3x_s5x16" run/proxy.sh proxy
SRC=<pascal-p106-v3-lowbit tree> run/proxy.sh lib       # then:
RUN=run5 ARGS="--layers 10,34 --formats N2P,ik2x_s4x8,N3P,ik3x_s5x32,IQ2_S,IQ3_XXS" run/proxy.sh proxy
```

Run 1's 43 formats: `Q4_0,IQ4_XS,Q3_K,IQ3_S,IQ3_XXS,Q2_K,IQ2_S,IQ2_XS,IQ2_XXS,Q2_0,u4a_32,u3s_32,u3a_32,u3s_64,
u3s_s6x32,u3a_s6x32,u3s_s4x32,u3s_s4x16,u3a_s4x16,u3a_s6x16,nl3_32,nl3_64,nl3_s6x32,nl3_s4x32,nl3_s4x16,nl3_s6x16,u2s_32,
u2m_32,u2s_s6x32,u2s_s4x16,u2a_s4x16,u2s_s4x8,u2a_s4x8,nl2_32,nl2_s4x16,nl2_s6x16,nl2_s4x8,u3s_s6x32+r256,u3a_s4x16+r256,
nl3_s6x32+r256,u2s_s4x16+r256,u2a_s4x8+r256,nl2_s4x16+r256` (one argument, without the line breaks). Run 1 fitted
`tables.json`; every later run reused it.
`summary.txt` is `lowbit_summary.py --ref IQ3_S --mix …` over the main and shift tables concatenated, followed by the
per-layer ratios of the paired formats.
