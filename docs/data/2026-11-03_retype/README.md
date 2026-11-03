# Q218: Flash-Next's hyper-connections in Q8_0 and its router in F16 (2026-11-03)

On the rig, Qwen3.8-Flash-Next spends 43 of its 136 ms of GPU time per token in BF16 matrix-vector products
([decode_profile_v3.txt](../2026-11-03_flash_next_baseline/decode_profile_v3.txt)), almost all in the hyper-connection
projections `hc_attn_up`, `hc_ffn_up` (320 × 10,240) and `hc_attn_down`, `hc_ffn_down` (10,240 × 320): Pascal has no
fast BF16 path, and these run about 8× slower than their bytes need. The router (`ffn_gate_inp`, 2,560 × 512) is F32
and takes another 2.5 ms. This step rewrites the two Swift-1.5 GSQ-RCO files that fit one rig with those tensors in
Q8_0 and F16, from the files' own BF16 and F32 tensors (no BF16 download), every other tensor byte for byte, and
measures what that costs in quality. The speed is measured on the rig. The registry row is Q218 in
[quants/experiments.md](../../../quants/experiments.md).

## Result: exact, 0.73 GB less on the cards, and about twice the KLD of a change of summation order

- **The rewrite is exact.** In both files the 290 Q8_0 tensors are byte for byte what llama-quantize writes from the
  same BF16 tensors, and the 96 F16 tensors what `ggml_quantize_chunk` writes; the key/value pairs and every other
  tensor are unchanged (`verify`), and shard 3 keeps the table's md5 `98ce140e641bc5d68af9f31df0f1c048`. A file takes
  ~9 minutes on an NVMe disk with eight processes at the lowest priority.
- **The cards hold 725,729,280 bytes less per file** ([sizes.tsv](sizes.tsv)): 599,654,400 from the hyper-connections
  (1,279,262,720 bytes of BF16 become 679,608,320 of Q8_0) and 126,074,880 from the router. Shards 1–2, all the cards
  hold, go from 37,749,814,496 to 37,024,085,216 bytes (Q2_0) and from 47,165,934,816 to 46,440,205,536 (IQ3_XXS).
- **What it costs** ([kld.tsv](kld.tsv); every row against the unmodified file's logits from the same four cards):

  | File | Text | Retyped: mean KLD | same top token | PPL change | `-ub 256`: mean KLD | same top token | PPL change |
  |---|---|---:|---:|---:|---:|---:|---:|
  | Q2_0 | wiki | 0.0155 ± 0.0003 | 95.17 % | +0.24 % | 0.0082 ± 0.0001 | 96.46 % | +0.17 % |
  | Q2_0 | agent | 0.0166 ± 0.0007 | 97.49 % | −0.40 % | 0.0079 ± 0.0003 | 98.23 % | −0.32 % |
  | Q2_0 | code | 0.0049 ± 0.0002 | 99.00 % | −0.19 % | 0.0024 ± 0.0001 | 99.34 % | +0.11 % |
  | IQ3_XXS | wiki | 0.0293 ± 0.0005 | 94.05 % | −0.44 % | 0.0158 ± 0.0004 | 95.42 % | +0.17 % |
  | IQ3_XXS | agent | 0.0418 ± 0.0015 | 96.49 % | −0.58 % | 0.0234 ± 0.0010 | 97.37 % | −0.56 % |
  | IQ3_XXS | code | 0.0079 ± 0.0004 | 99.04 % | +0.09 % | 0.0043 ± 0.0003 | 99.29 % | +0.12 % |

  `self`, the unmodified file against its own logits, gives 0 and 100 % on wiki for both files: on a fixed set of cards
  the scores are deterministic. PPL change is the file's perplexity against the unmodified file's on the same text.
- **The retype is not free, but it is small.** Summing the same weights in another order (`-ub 256`) already moves
  this model's output distributions: KLD 0.008 (Q2_0) and 0.016 (IQ3_XXS) on wiki, with 3.5–4.6 % of the tokens
  changing their top choice. The retype moves them 1.8–2.1× as far on every file and text, and leaves perplexity
  within the same ±0.6 % that `-ub 256` moves it, in either direction. Whether that costs agentic quality is the agent
  check's to say, together with the speed: R28 (Q2_0) and S13 (IQ3_XXS) on the rigs.
- **The router's F16 costs nothing; the cost is the hyper-connections' Q8_0.** The same rewrite with the router left
  in F32 (`hcq8r32`) scores the same within the standard errors:

  | File | Text | Hyper-connections only: mean KLD | same top token | Both (as above): mean KLD | same top token |
  |---|---|---:|---:|---:|---:|
  | Q2_0 | wiki | 0.0157 ± 0.0003 | 95.05 % | 0.0155 | 95.17 % |
  | Q2_0 | agent | 0.0167 ± 0.0006 | 97.45 % | 0.0166 | 97.49 % |
  | Q2_0 | code | 0.0052 ± 0.0002 | 99.00 % | 0.0049 | 99.00 % |
  | IQ3_XXS | wiki | 0.0287 ± 0.0005 | 93.99 % | 0.0293 | 94.05 % |
  | IQ3_XXS | agent | 0.0386 ± 0.0013 | 96.61 % | 0.0418 | 96.49 % |
  | IQ3_XXS | code | 0.0081 ± 0.0004 | 98.91 % | 0.0079 | 99.04 % |

  F16 would hold the hyper-connections almost exactly ([f16_exact.tsv](f16_exact.tsv)): 99.985 % of their 640 M
  weights are exact in F16, the other 97,006 lie below 2⁻¹⁷ and would move by less than 3 × 10⁻⁸, and the largest is
  8.56. F16 keeps 2 bytes a weight, so it pays only if llama.cpp's F16 path runs these shapes on a P106 near their
  bytes, which BF16's does not.
- **Base logits must come from the same cards as the scores.** The same unmodified IQ3_XXS scored on these four cards
  (two RTX 3080s and two RTX 3090s) against base logits from three RTX 3090s
  ([2026-11-03_table_bits](../2026-11-03_table_bits/README.md)) gives KLD 0.0159, 0.0225 and 0.0040 on wiki, agent and
  code (95.6, 97.5 and 99.3 % same top token), as large as `-ub 256` and as that folder's re-encoded-table control
  (the floor of Q574): llama.cpp's matrix kernels divide their work by the card's number of multiprocessors, which
  differs between the two models, so the same products are summed in another order. The retyped file against those
  base logits gives 0.0299, 0.0435 and 0.0079, close to its scores on its own cards:

  | Text | Unmodified, other cards' base | Retyped, other cards' base | Retyped, same cards' base |
  |---|---:|---:|---:|
  | wiki | 0.0159 | 0.0299 | 0.0293 |
  | agent | 0.0225 | 0.0435 | 0.0418 |
  | code | 0.0040 | 0.0079 | 0.0079 |

## How it was made

**The files.** Swift-1.5 GSQ-RCO Q2_0 and IQ3_XXS
([ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
revision `99bb8f7f95c7aa7b24a36a7786a4f657b30f5d3d`), in the three-shard layout of
[Q435](../2026-11-03_ssd_room/README.md) that gives the n-gram table a shard of its own.

**The rewrite** ([scripts/gguf_retype.py](../../../scripts/gguf_retype.py), standard-library Python, eight worker
processes at the lowest CPU priority) with two rules, `hc_=Q8_0@BF16` and `ffn_gate_inp=F16@F32`:

| Tensors | Per file | From | To |
|---|---:|---|---|
| `blk.*.hc_{attn,ffn}_{up,down}` (320 × 10,240 and 10,240 × 320) | 192 | BF16 | Q8_0 |
| `blk.*.hc_{attn,ffn}_inject` (10,240 × 4) | 96 | BF16 | Q8_0 |
| `output_hc_up`, `output_hc_down` (the head's hyper-connection) | 2 | BF16 | Q8_0 |
| `blk.*.ffn_gate_inp` (2,560 × 512, the router) | 48 | F32 | F16 |
| `blk.*.ffn_gate_inp_shexp` (2,560, the shared expert's gate) | 48 | F32 | F16 |

The hyper-connection norms (`hc_*_norm`, F32 vectors) stay as they are. Q8_0 follows ggml's
`quantize_row_q8_0_ref` bit for bit (per block of 32 weights the scale is the largest magnitude over 127, the weights
are rounded half away from zero, the scale is stored in F16), and F16 rounds to nearest even as
`ggml_fp32_to_fp16_row`. Each tensor stays in its shard; the key/value pairs, tensor order and every other tensor's
bytes are unchanged, so llama.cpp reads the files without any engine change: stock llama.cpp has Q8_0 and F16 kernels
for these shapes.

**The checks**, on each file:
- `gguf_retype.py verify`: the same key/value pairs, tensor names, shapes and order as the unmodified shards, and every
  tensor that was not converted identical byte for byte;
- the md5 of every new shard; shard 3, the table, must keep the table's md5 `98ce140e641bc5d68af9f31df0f1c048`;
- [scripts/gguf_retype_test.py](../../../scripts/gguf_retype_test.py) with the real files: the 290 BF16 tensors are
  written in their source type into one GGUF with shard 1's key/value pairs and converted to Q8_0 by llama-quantize
  itself (the `qwen4exp` architecture does not exclude `hc_*` from quantization), and the 96 F32 tensors go through
  `ggml_quantize_chunk` to F16 (llama-quantize never converts `ffn_gate_inp`); each must equal its tensor in the new
  shards. Both come from a CPU build of the Miner Llama v3 base commit `37e717ff65f641a2b365eebeef33fc1e1292e707`
  (`run/build.sh`). The test's synthetic part (edge cases: rounding ties, all-zero and all −0.0 blocks, scales below
  F16's normal range and below its smallest step, F16 ties) fails for a tool that rounds half to even, cuts the scale
  to F16 instead of rounding it, or keeps the sign of an all −0.0 block.

**The scores** (`run/kld.sh`, the script of [2026-11-03_table_bits](../2026-11-03_table_bits/README.md) with a new
header): llama-perplexity's KL divergence of each retyped file against its unmodified file on the three texts of
[2026-10-27_ornith_kld](../2026-10-27_ornith_kld/README.md) (wikitext-2 test, 80 × 512 tokens; held-out agent sessions,
12 × 8,192; the agent check's Exercism code, 80 × 512), with that folder's flags (`-ub 512 -fa on`, all layers on the
cards), built for compute 8.6 from the same v3 base commit. Every score of a file comes from one set of cards, two RTX
3080s and two RTX 3090s, the same as its base logits. Two references measure the measurement: `self`, the unmodified
file against its own logits, and `ub256`, the unmodified file with `-ub 256`, the same weights summed in another order.
A third file per model, `hcq8r32`, has only the hyper-connections converted (the rule `hc_=Q8_0@BF16`), to tell the
two conversions apart. The IQ3_XXS files were also scored against the base logits of
[2026-11-03_table_bits](../2026-11-03_table_bits/README.md), made on three RTX 3090s.

**F16's exactness** (`run/f16_exact.py`): every BF16 weight of the `hc_*` tensors classified by its exponent as exact
in F16, rounded (below 2⁻¹⁷, where F16's subnormals no longer hold BF16's 8 significant bits) or overflowing (from
2¹⁶).

## Files

| File | What |
|---|---|
| [kld.tsv](kld.tsv) | one row per file, variant and text (`hcq8`: both conversions, `hcq8r32`: the hyper-connections only, `ub256`, `self`, and the IQ3_XXS scores against the other cards' base logits): mean KLD and its standard error, the 99th and 99.9th percentile KLD, same top token and its standard error, RMS of the top token's probability change, perplexity of the variant and of the unmodified file |
| [sizes.tsv](sizes.tsv) | per shard of the `hcq8` files: the size before and after, and the tensors converted from each type with their bytes before and after |
| [f16_exact.tsv](f16_exact.tsv) | per hyper-connection tensor (layers summed): weights exact in F16, rounded, overflowing, and the largest magnitude |
| [checksums.md5](checksums.md5) | md5 of every shard of the four retyped files (`hcq8` and `hcq8r32` of each model) |
| [run/build.sh](run/build.sh) | llama-quantize and libggml-base.so (CPU) from the v3 base commit, for the comparison with llama.cpp |
| [run/retype.sh](run/retype.sh) | one file: the rewrite, `verify`, the md5 and the comparison with llama.cpp |
| [run/kld.sh](run/kld.sh) | the KL divergence runs (base logits, then each variant) |
| [run/table.py](run/table.py), [run/sizes.py](run/sizes.py), [run/f16_exact.py](run/f16_exact.py) | the logs and headers to the three tables |

## Commands

On the quality machine (Docker, CUDA 12.9), with `S` this repository's `scripts/`, `D` this folder's `run/`, `M` the
folder of the two three-shard files, `O` a folder for the retyped files, `W` a work folder, `K` a folder for base
logits (44 GB a file), `T` the folder with the agent and code texts of
[2026-10-27_ornith_kld](../2026-10-27_ornith_kld/README.md), `L` a llama.cpp checkout at `37e717ff6`, `B` the
llama-perplexity of [2026-11-03_table_bits](../2026-11-03_table_bits/README.md)'s `run/build.sh` and `TB` that
folder's base logits of the unmodified IQ3_XXS:

```bash
WORK=$W/quantize SRC=$L JOBS=8 DOCKER_ARGS="--label <label> --cpu-shares 256" bash $D/build.sh
python3 -I $S/gguf_retype_test.py $W/test                     # the synthetic self-test
Q=$W/quantize/build/bin
for f in Q2_0 IQ3_XXS; do                                      # hcq8: both rules; hcq8r32: the router left in F32
  for v in "hcq8 hc_=Q8_0@BF16 ffn_gate_inp=F16@F32" "hcq8r32 hc_=Q8_0@BF16"; do
    set -- $v; n=$1; shift
    RULES="$*" SCRIPTS=$S MODEL=$M/Swift-Qwen3.8-Flash-Next-GSQ-RCO-$f-00001-of-00003.gguf OUT=$O/$f-$n \
      PREFIX=Swift-Qwen3.8-Flash-Next-GSQ-RCO-$f-$n QUANTIZE=$Q/llama-quantize GGML=$Q/libggml-base.so \
      TEST=$W/test-$f-$n JOBS=8 bash $D/retype.sh
  done
done
# a variant folder per score: symlinks to the shards it reads, and `args` with extra arguments
for f in Q2_0 IQ3_XXS; do
  p=Swift-Qwen3.8-Flash-Next-GSQ-RCO-$f
  for n in hcq8 hcq8r32; do
    mkdir -p $W/v/$f-$n
    ln -s $O/$f-$n/$p-$n-0000[12]-of-00003.gguf $W/v/$f-$n/
    ln -s $M/$p-00003-of-00003.gguf $W/v/$f-$n/$p-$n-00003-of-00003.gguf   # the unchanged table
  done
  mkdir -p $W/v/$f-self $W/v/$f-ub256
  ln -s $M/$p-0000?-of-00003.gguf $W/v/$f-self/; ln -s $M/$p-0000?-of-00003.gguf $W/v/$f-ub256/
  echo "-ub 256" > $W/v/$f-ub256/args
done
export BIN=$B GPUS=0,1,5,6 WIKI=wikitext-2-raw/wiki.test.raw AGENT=$T/agent-heldout.txt CODE=$T/code-exercism.txt \
  MOUNTS="$M $O $W $K $T $TB" DOCKER_ARGS="--label <label> --cpu-shares 256 --name kld"
for f in Q2_0 IQ3_XXS; do
  m=$M/Swift-Qwen3.8-Flash-Next-GSQ-RCO-$f-00001-of-00003.gguf
  OUT=$K/$f bash $D/kld.sh $m $W/v/$f-hcq8 $W/v/$f-ub256 $W/v/$f-hcq8r32
  OUT=$K/$f TEXTS=wiki bash $D/kld.sh $m $W/v/$f-self
done
mkdir -p $K/other; ln -s $TB/base.{wiki,agent,code}.kld $K/other/     # IQ3_XXS against the 3090s' base logits
OUT=$K/other bash $D/kld.sh $M/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00003.gguf \
  $W/v/IQ3_XXS-hcq8 $W/v/IQ3_XXS-self
python3 -I $D/table.py $(for f in Q2_0 IQ3_XXS; do for n in hcq8 hcq8r32 ub256 self; do
  echo $f-$n=$K/$f/$f-$n; done; done) "IQ3_XXS-hcq8 (base from 3 x RTX 3090)=$K/other/IQ3_XXS-hcq8" \
  "IQ3_XXS-self (base from 3 x RTX 3090)=$K/other/IQ3_XXS-self" > kld.tsv
{ for f in Q2_0 IQ3_XXS; do
    SCRIPTS=$S python3 -I $D/sizes.py $f $M/Swift-Qwen3.8-Flash-Next-GSQ-RCO-$f-00001-of-00003.gguf \
      $O/$f-hcq8/Swift-Qwen3.8-Flash-Next-GSQ-RCO-$f-hcq8-00001-of-00003.gguf
  done; } | awk 'NR == 1 || !/^file/' > sizes.tsv
for f in Q2_0 IQ3_XXS; do
  SCRIPTS=$S python3 -I $D/f16_exact.py hc_ $M/Swift-Qwen3.8-Flash-Next-GSQ-RCO-$f-0000[12]-of-00003.gguf |
    awk -v f=$f 'NR == 1 { print "file\t" $0; next } { print f "\t" $0 }'
done | awk 'NR == 1 || !/^file/' > f16_exact.tsv
cat $O/*/*.md5 > checksums.md5
```

The runs went one at a time on the four cards (on the quality machine the variant folders were named in lower case,
`hcq8r32` as `hc`; `kld.tsv` uses the names above), and `kld.sh` skips a run whose log is complete.

The model's weights are under the Swift Open License 1.0 and qwen-community-1.0; this folder holds sizes, checksums
and scores, no weights.
