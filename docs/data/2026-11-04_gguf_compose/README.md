# Q597 and Q620: a split-aware GGUF composer, and Flash-Next's hyper-connections below Q8_0 (2026-11-04)

Two standard-library tools for building Qwen3.8-Flash-Next files on a machine with no NumPy or PyTorch, and the
files they built for the next quality runs:

- [scripts/gguf_compose.py](../../../scripts/gguf_compose.py) (Q597) writes a GGUF whose chosen tensors come byte for
  byte from another file of the same model and every other tensor byte for byte from the base. Both inputs may be
  split, and the output keeps the base's shards or is split again. It builds Q218's Pascal spines: a Swift file whose
  dense matrices come from a Q8_0 of the same model, its routed experts and n-gram table left as they are.
- [scripts/gguf_retype.py](../../../scripts/gguf_retype.py) (Q620) now also writes Q6_K, Q5_0 and Q4_0, from F32, F16,
  BF16 or Q8_0, beside its BF16 → Q8_0 and F32 → F16: bit for bit what ggml's reference quantizer writes, which is what
  llama-quantize writes without an importance matrix. On the rig, Flash-Next's hyper-connection projections still read
  0.68 GB a token after Q218's retype to Q8_0; Q5_0 cuts that by 35 %, and Q6_K, which needs rows of 256, by 11 % with
  the 320-wide up projections left in Q8_0 or by 29 % with them in Q5_0, if the KL divergence holds.

The registry rows are Q597 and Q620 in [quants/experiments.md](../../../quants/experiments.md); the spine is FN3 of
Q218. The scores are still to run: [run/score.sh](run/score.sh) scores every file here against the Swift-1.5 Q8_0
reference of [2026-11-03_flash_next_quality](../2026-11-03_flash_next_quality/README.md), with that folder's texts,
cards and split, once its chain has ended.

## Result

- **Both tools are exact on the real files.** In each Swift Q2_0 file every converted tensor is byte for byte what
  `ggml_quantize_chunk` writes and, for the quantized types, what llama-quantize writes ([checks.txt](checks.txt)):

  | File | Converted (from BF16, the router from F32) | Identical to ggml and llama-quantize |
  |---|---|---|
  | `q2_0-hcq6k` | 193 hyper-connection tensors to Q6_K (down and inject projections), 97 to Q8_0 (the 320-wide ups), 96 router tensors to F16 | all 386 |
  | `q2_0-hcq5` | 290 hyper-connection tensors to Q5_0, 96 router tensors to F16 | all 386 |
  | `q2_0-hcq6k5` | 193 hyper-connection tensors to Q6_K, 97 to Q5_0 (the ups), 96 router tensors to F16 | all 386 |

  The F16 router tensors are checked against ggml only: llama-quantize never converts `ffn_gate_inp`.

  The IQ3_XXS spine `iq3_xxs-spine8` holds 301 tensors identical to the Q8_0's and 923 to its base's (`verify`);
  in `iq3_xxs-spine5` the 300 dense tensors requantized from that Q8_0 to Q5_0 are identical to
  `llama-quantize --allow-requantize`'s and to `ggml_quantize_chunk`'s from ggml's own dequantization. Every
  other tensor of every file is identical to its input's, and llama.cpp's loader reads each file and finds every
  tensor inside the file bounds.
- **What the cards hold** ([sizes.tsv](sizes.tsv)): the hyper-connections take 679.6 MB in `hcq8`
  ([2026-11-03_retype](../2026-11-03_retype/README.md)), 601.7 MB in `hcq6k` (−11.5 %), 482.5 MB in `hcq6k5`
  (−29.0 %) and 439.7 MB in `hcq5` (−35.3 %), and every one of those bytes is read once a token. The spine moves
  the other way: the IQ3_XXS file's dense matrices take 2.30 GB in their K- and IQ-quants, 3.77 GB in Q8_0 (+1.46 GB
  on the cards) and 2.68 GB in Q5_0 with the output head kept in Q8_0 (+0.37 GB).
- **The scores are next:** each file against the Q8_0 reference's logits on the three texts, beside
  `swift-q2_0-hcq8` and `swift-iq3_xxs-hcq8` of [2026-11-03_flash_next_quality](../2026-11-03_flash_next_quality/README.md),
  with [run/score.sh](run/score.sh).
- **Building beside a KL divergence run:** at the lowest CPU and disk priority, six workers spread over all cores
  still slowed the host's part of the KL divergence runs beside them (llama-perplexity computes the divergence on the
  host): the agent text, with the most scored tokens, took 1.53 times as long as the wiki text for the file scored
  then, against 1.18 and 1.31 for the files scored before the build. Pinned to two of the twelve cores (`CPUS`), the
  build left the runs as fast as before: the Q2_0
  `hcq8` file scored in 3:43, 4:33 and 2:39 (wiki, agent, code) beside the build, against 3:45, 4:56 and 2:36 for the
  unmodified Q2_0 with nothing beside it.

## The tools

**gguf_compose.py.** `compose --base SHARD... --donor SHARD... --take PATTERN... --out-dir DIR` takes the tensors that
match a pattern (a name, `glob:<pattern>` or `re:<regex>`) from the donor, every other from the base. It refuses two
files whose architectures, tensor names or shapes differ, and a pattern that matches nothing. `--layout` keeps the
base's shards (the default), follows the donor's, writes one file, or re-splits by size; `--isolate` gives the tensors
that match it a last shard of their own, as [scripts/gguf_reshard.py](../../../scripts/gguf_reshard.py) does. Every
tensor keeps its source slot, so a base shard with nothing taken comes out byte for byte as it was, and
`--link-unchanged` writes a symbolic link to it instead (Flash-Next's 28.8 GB table shard is not copied).
`--headers-only` writes the headers and sizes the files, for a fit check without the data. `verify` reads every tensor
of the output against the input it came from.

**gguf_retype.py.** A rule `<regex>=<type>[@<from type>]` converts the matching tensors; the first matching rule wins.
The new conversions follow ggml's C step by step in float32:

| To | Per block | From |
|---|---|---|
| Q5_0, Q4_0 (`quantize_row_q5_0_ref`, `_q4_0_ref`) | 32 values: `max` the first value of the largest magnitude, `d = max / -16` (Q4_0 −8), `q = min(31, trunc(x / d + 16.5))` (Q4_0 `min(15, trunc(x / d + 8.5))`), `d` in F16 | F32, F16, BF16, Q8_0 |
| Q6_K (`quantize_row_q6_K_ref`) | 256 values in 16 sub-blocks: each sub-block's scale from `make_qx_quants`' search over 19 trial scales (squared error weighted by x²), the 16 scales stored as int8 against the largest, `d` in F16, `q = nearest(x / (d · scale))` in [−32, 31]; a sub-block whose `d · scale` is 0 keeps the search's own levels | F32, F16, BF16, Q8_0 |

A Q8_0 source is read as `dequantize_row_q8_0` reads it (`q · d`, exact in float32), so Q8_0 → Q5_0 is what
`llama-quantize --allow-requantize` writes. Rows must hold whole blocks: llama-quantize would switch such a tensor to
another type (Q6_K to Q8_0), so the 320-wide up projections get a rule of their own first. Each float32 step is done
in double and rounded to float32 on its way into an `array('f')`, which gives float32's own result; the C code's loop
over a block runs across all blocks at once, one column of the block at a time, in the C code's order. Tensors are
converted in pieces of 2²⁰ values across worker processes: on the quality machine one core converts a million BF16
values to Q8_0 in 0.20 s, to Q5_0 in 0.27 s and to Q6_K in 7.3 s.

## How the tools are checked

- [scripts/gguf_compose_test.py](../../../scripts/gguf_compose_test.py): a base in three shards (the last holding one
  tensor alone, as the table shard does) and a donor of the same model in two shards of another layout and other
  types. It checks every layout (base with the untouched shard linked, then copied with its old SHA-256; donor; one
  file; a re-split by size with one tensor isolated), the key/value pairs and split keys of every shard, every
  tensor's type and bytes, `--headers-only` against the full files and `verify`. Negative controls that must fail: a
  donor with one shape changed, a donor of another architecture, a pattern that matches nothing, an output that is an
  input, and `verify` of an output with one byte changed in a taken tensor and in a base tensor.
- [scripts/gguf_retype_test.py](../../../scripts/gguf_retype_test.py): its Q8_0 and F16 file as before, and a second
  synthetic file of BF16, F32, F16 and Q8_0 tensors converted to Q6_K, Q5_0 and Q4_0, against a line-by-line
  transcription of ggml's C (`quantize_row_q5_0_ref`, `_q4_0_ref`, `_q6_K_ref` with `make_qx_quants` and
  `nearest_int` as written, with its float trick, and `dequantize_row_q8_0`). Its blocks hold the edge cases: two
  equal magnitudes of either sign first, the caps at 31 and 15, values one float32 step below a truncation boundary,
  `nearest_int` ties, Q6_K sub-blocks that are zero, whose int8 scale rounds to 0, or whose block's F16 `d`
  underflows (both keep `make_qx_quants`' levels), an outlier that moves the scale search, a tensor of two conversion
  pieces, a 320-wide tensor kept in Q8_0 by a first rule, and a shard with nothing to convert, linked. Negative
  controls: a Q6_K rule on rows of 320 and an impossible conversion must fail, and the transcriptions with one step
  changed (Q6_K without the scale search, Q5_0's truncation in double instead of float32) must differ from the tool.
  Four deliberate bugs in the tool each fail the test (Q5_0 rounding instead of truncating, Q6_K without the search,
  the last instead of the first largest value, Q6_K's zero-scale sub-blocks given level 32), and a different piece
  size passes.
- On the real files: every converted tensor of each retyped file goes through `ggml_quantize_chunk` from
  libggml-base (the function llama-quantize calls), from float32 values made by ggml's own `to_float` for a Q8_0 or
  F16 source, and every Q8_0, Q6_K, Q5_0 and Q4_0 tensor through llama-quantize itself (`--pure`, with
  `--allow-requantize` for a Q8_0 source); each must equal the tool's bytes, and ggml's bytes of one tensor must
  differ from the tool's of another of the same shape. Both come from a CPU build of the Miner Llama v3 base commit
  `37e717ff65f641a2b365eebeef33fc1e1292e707` ([2026-11-03_retype/run/build.sh](../2026-11-03_retype/run/build.sh)).
  Each file also passes `verify` against its inputs, and llama.cpp's loader reads its headers and checks every
  tensor against the file bounds (`llama-quantize --dry-run`).

## The files

Each file has three shards, as its inputs do; shard 3 of every file is a link to the unchanged n-gram table shard,
so only shards 1–2 are new (md5 in [checksums.md5](checksums.md5)). "Cards" is
every tensor but the table and the token embeddings ([sizes.tsv](sizes.tsv)).

| File | Made from | What changes | Cards (GB) | Against its input |
|---|---|---|---:|---:|
| `swift_q2_0_hcq6k` | Swift Q2_0 | hyper-connections Q6_K, their 320-wide ups Q8_0; router F16 | 36.662 | −77.9 MB against `hcq8` |
| `swift_q2_0_hcq5` | Swift Q2_0 | hyper-connections Q5_0; router F16 | 36.500 | −239.9 MB against `hcq8` |
| `swift_q2_0_hcq6k5` | Swift Q2_0 | hyper-connections Q6_K, their ups Q5_0; router F16 | 36.543 | −197.1 MB against `hcq8` |
| `swift_iq3_xxs_spine8` | Swift IQ3_XXS `hcq8` + Swift-1.5 Q8_0 | the 301 dense matrices (attention, recurrent output, shared expert, output head) from the Q8_0 | 47.618 | +1,462.4 MB |
| `swift_iq3_xxs_spine5` | `swift_iq3_xxs_spine8` | those matrices requantized to Q5_0, the output head kept in Q8_0 | 46.527 | +371.2 MB |

The dense matrices are the 300 tensors that match `^blk\.[0-9]+\.(attn_(q|k|v|qkv|gate|output)|ssm_out|ffn_(gate|up|down)_shexp)\.weight$`
and `output.weight`; in the IQ3_XXS file they are Q4_K, Q5_K, Q6_K, IQ3_S, IQ4_XS, IQ4_NL and (8 shared-expert down
projections) Q2_0. The file's other BF16 tensors (`ple_key`, `ple_value`, `ssm_alpha`, `ssm_beta`, the indexer) are
not part of the spine and stay as they are. [run/build.sh](run/build.sh) also builds `iq3_xxs-spine4` (Q4_0) and the
spine of the Q2_0 file itself (`q2_0-spine8`, `q2_0-spine5`).


## Files here

| File | What |
|---|---|
| [checksums.md5](checksums.md5) | md5 of every shard written (shard 3 of each file is a link to the unchanged table shard, md5 `98ce140e641bc5d68af9f31df0f1c048`) |
| [sizes.tsv](sizes.tsv) | per file: the bytes of shards 1–2 (what the cards hold) and the bytes each conversion or composition moved |
| [checks.txt](checks.txt) | per file: the conversion or composition totals, the `verify` lines, the loader's tensor counts and the comparison with llama.cpp |
| [run/build.sh](run/build.sh) | every file of this folder, one after another as one detached job at the lowest CPU and disk priority, with its checks |
| [run/sizes.py](run/sizes.py) | the headers to `sizes.tsv` |
| [run/score.sh](run/score.sh) | the KL divergence of each file against the Swift-1.5 Q8_0 reference's logits, with [2026-11-03_flash_next_quality](../2026-11-03_flash_next_quality/README.md)'s scripts |

## Commands

On the quality machine, with `S` this repository's `scripts/`, `D` this folder's `run/`, `O` the models folder, `W` a
work folder, `L` a llama.cpp checkout at `37e717ff6`, and the inputs: `Q2` the first shard of Swift-1.5 GSQ-RCO Q2_0 in
the three-shard layout of [2026-11-03_ssd_room](../2026-11-03_ssd_room/README.md), `Q2H` and `IQ3H` the first shards
of the `hcq8` files of [2026-11-03_retype](../2026-11-03_retype/README.md), `Q8` the first shard of ukisai's
Swift-1.5 Q8_0, `TABLE` a file of the table shard (`-00003-of-00003`, md5 `98ce140e…`):

```bash
python3 -I $S/gguf_compose_test.py $W/test-compose               # the self-tests
python3 -I $S/gguf_retype_test.py $W/test-retype
WORK=$W/quantize SRC=$L JOBS=8 bash ../2026-11-03_retype/run/build.sh   # llama-quantize and libggml-base.so (CPU)
B=$W/quantize/build/bin
SCRIPTS=$S OUT=$O WORK=$W Q2_0=$Q2 Q2_0_HCQ8=$Q2H IQ3_HCQ8=$IQ3H Q8_0=$Q8 TABLE=$TABLE QUANTIZE=$B/llama-quantize \
  GGML=$B/libggml-base.so JOBS=6 CPUS=10,11,22,23 setsid nohup bash $D/build.sh >> $W/logs/build.log 2>&1 < /dev/null &
# FILES="iq3_xxs-spine4 q2_0-spine8 q2_0-spine5" builds the others
P=Swift-Qwen3.8-Flash-Next-GSQ-RCO; f() { echo "$O/swift_$1/$P-$2-00001-of-00003.gguf"; }
python3 -I $D/sizes.py swift-q2_0=$Q2 swift-q2_0-hcq8=$Q2H:swift-q2_0 \
  swift-q2_0-hcq6k=$(f q2_0_hcq6k Q2_0-hcq6k):swift-q2_0-hcq8 swift-q2_0-hcq5=$(f q2_0_hcq5 Q2_0-hcq5):swift-q2_0-hcq8 \
  swift-q2_0-hcq6k5=$(f q2_0_hcq6k5 Q2_0-hcq6k5):swift-q2_0-hcq8 swift-iq3_xxs-hcq8=$IQ3H \
  swift-iq3_xxs-spine8=$(f iq3_xxs_spine8 IQ3_XXS-hcq8-spine8):swift-iq3_xxs-hcq8 \
  swift-iq3_xxs-spine5=$(f iq3_xxs_spine5 IQ3_XXS-hcq8-spine5):swift-iq3_xxs-hcq8 > sizes.tsv
cat $O/swift_{q2_0_hcq6k,q2_0_hcq5,q2_0_hcq6k5,iq3_xxs_spine8,iq3_xxs_spine5}/*.md5 > checksums.md5
```

`checks.txt` is each file's log (`$W/logs/<file>.log`) and loader log (`<prefix>.load.log` beside its shards),
filtered to the totals, `verify` lines, tensor counts and comparisons, with folder names cut.

These files were built by `run/build.sh` before it had `CPUS`: its workers were pinned to the same two cores with
`taskset` 20 minutes in, and the loader's tensor counts in `checks.txt` come from the loader logs, which the job's
log had filtered without the K types.

The scores, once [2026-11-03_flash_next_quality](../2026-11-03_flash_next_quality/README.md)'s chain has ended, with
`E` its environment file, `K` its kld folder (base logits and split), `H` a folder on the fast disk and `TF` a copy of
the table shard there:

```bash
P=Swift-Qwen3.8-Flash-Next-GSQ-RCO
ENV=$E QUALITY=../2026-11-03_flash_next_quality/run BASES=$K OUT=$W/kld HOT=$H TABLE_FAST=$TF FILES="\
swift-q2_0-hcq6k=$O/swift_q2_0_hcq6k/$P-Q2_0-hcq6k-00001-of-00003.gguf
swift-q2_0-hcq5=$O/swift_q2_0_hcq5/$P-Q2_0-hcq5-00001-of-00003.gguf
swift-q2_0-hcq6k5=$O/swift_q2_0_hcq6k5/$P-Q2_0-hcq6k5-00001-of-00003.gguf
swift-iq3_xxs-spine8=$O/swift_iq3_xxs_spine8/$P-IQ3_XXS-hcq8-spine8-00001-of-00003.gguf
swift-iq3_xxs-spine5=$O/swift_iq3_xxs_spine5/$P-IQ3_XXS-hcq8-spine5-00001-of-00003.gguf" \
  setsid nohup bash $D/score.sh >> $W/logs/score.log 2>&1 < /dev/null &
python3 -I ../2026-11-03_flash_next_quality/run/table.py $W/kld q8_0-ub256 swift-q2_0 swift-q2_0-hcq8 \
  swift-q2_0-hcq6k swift-q2_0-hcq5 swift-q2_0-hcq6k5 swift-iq3_xxs swift-iq3_xxs-hcq8 swift-iq3_xxs-spine8 \
  swift-iq3_xxs-spine5 > kld.tsv
```

## Provenance and licence

The inputs: ukisai's Swift-1.5 GSQ-RCO Q2_0 and IQ3_XXS
([ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
revision `99bb8f7f95c7aa7b24a36a7786a4f657b30f5d3d`), their `hcq8` retypes of
[2026-11-03_retype](../2026-11-03_retype/README.md), and ukisai's Swift-1.5 Q8_0
([ukisai/Swift-1.5-Qwen3.8-Flash-Next-GGUF](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GGUF) revision
`0142554c963a8653912c0c8283f05f13f63653f3`, `Q8_0/`, each shard's SHA-256 checked against Hugging Face's). The weights
are under the Swift Open License 1.0 and qwen-community-1.0; this folder holds sizes, checksums and check results, no
weights.
