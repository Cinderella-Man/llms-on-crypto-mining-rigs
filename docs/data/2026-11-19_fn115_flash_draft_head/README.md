# FN115 on a P106: Flash-Next with a published distilled MTP draft head (Q1545; 2026-11-19)

**The published head cannot run on Miner Llama's v5 engine as it is published, so nothing was measured with it.** The
engine loads it, then stops on an internal check during the draft's first test pass, before the server answers any
request. The head's file marks its one layer as a plain-attention layer. The preset's own head, and every
full-attention layer of Flash-Next, use the sparse attention (QSA) instead. The engine's draft graph always builds the
sparse attention's inputs, and with no sparse layer to read them, one of those inputs has no memory. No setting of the
preset avoids it for the draft alone, because every sparse-attention switch also changes the main model. The test
therefore stopped at that point, and the head is parked until the engine handles a plain-attention draft layer
(Q1623). After that, the lane in [run/fn115.sh](run/fn115.sh) runs unchanged.

The head is [hyukudan/Qwen3.8-Flash-Next-MTP-distillG-Q8_0](https://huggingface.co/hyukudan/Qwen3.8-Flash-Next-MTP-distillG-Q8_0)
(`mtp-Qwen3.8-Flash-Next-shared-distillG-Q8_0.gguf`, 2,786,568,256 bytes, sha256 `5516d3ce…4add4c3c`, Qwen Community
License 1.0). Its maker distilled it against Unsloth's UD-Q4_K_XL target, not against the 2-bit file this preset runs.
Registry row [Q1545](../../../quants/experiments.md) (this test), with the new rows Q1623 (the engine change) and Q1624
(the head's experts at the preset head's types).

## The two heads, header against header

[head_compare.md](head_compare.md) lists every metadata key and tensor of both files
([run/head_compare.py](run/head_compare.py) reads only the headers).

| | the preset's head (`mtp-Qwen3.8-Flash-Next-draft.gguf`) | the published head |
|---|---|---|
| tensors, bytes | 34, 2,686,492,672 | 32, 2,775,621,632 |
| vocabulary | 248,320 tokens | the same: the token, type and merge arrays have equal md5s |
| layer shapes, hyperparameters | Flash-Next's MTP layer (block 48 of 49) | the same, except the next row |
| the layer's compress ratio (`attention.compress_ratios[48]`) | 4: a sparse-attention (QSA) layer, like layers 3, 7 … 47 | **0: plain attention over every cell** |
| token table, output head | its own: `token_embd` Q8_0 (675,430,400 B), `output` Q6_K (521,472,000 B) | **none** (`nextn_shared_target_tensors`): it reads the main model's |
| the layer's experts (gate, up, down) | Q4_K, Q4_K, IQ4_NL (471,859,200 B each) | Q8_0 (891,289,600 B each) |
| attention and shared expert | Q4_K and Q6_K; shared down Q8_0 | Q8_0 |
| indexer projections | Q8_0 | BF16 |

The maker's card says that only the trained tensors were re-encoded to Q8_0, and that the header and the other tensors
are those of the base head. The BF16 indexer projections are therefore the base head's own. A layer at compress ratio 0
never uses them.

What the published head needs from the engine:
- **Its token table and output head come from the main model.** The fork's `LLAMA_DRAFT_SHARE=embd,head` (Q564, Q635)
  does this: the draft then reads Swift Q2_0's Q3_K token table, on the host, and its Q5_K output head. Without the
  switch the file cannot load, since it has no `token_embd`.
- **The shortlist works with the shared head** (`LLAMA_DRAFT_SHORTLIST=32768,8192,2`). The probe's server set up the
  same 32,801 ids as with the preset's head.
- **The plain-attention layer is where it stops.** `qwen4exp`'s MTP graph builds the sparse attention's k-pool input
  whenever the draft has an indexer cache and the model's pool size is above 0. The pool size is 4 here, from the other
  layers. The layer itself at ratio 0 never reads that input, so its `k_idxs` tensor gets no buffer, and setting it
  stops on `GGML_ASSERT(buffer)` (`ggml-backend.cpp:347`). The full stack is in the probe's server log: from
  `llama_kv_cache::set_input_k_idxs` through `llm_graph_input_kpool::set_input`, `llama_context::decode`,
  `common_context_can_seq_rm` and the server's `load_model`. The sparse-attention switches of the preset
  (`LLAMA_QSA_*`) act on the main model as well, so turning one off would test another main-model line, not the head.

## What was run

- **Rig and build:** rig 1, nine P106-090 cards at stock clocks and their 75 W limit (no offsets, no clock lock), booted
  that morning and idle before the run. The tree is `pascal-p106-v5` `602691153` (the v5 candidate with upstream b12483
  merged, the tree of [R205](../2026-11-19_r205_flash_glu/README.md) the same morning).
- **The copy:** 2,786,568,256 bytes reached the rig in 88.66 s, with its sha256 equal to the maker's, and its licence
  file beside it.
- **The probe:** one load of the published head on the deep lane (`STEP=probe` in [run/fn115.sh](run/fn115.sh)). It
  used [release/flash-next-v4/settings.env](../../../release/flash-next-v4/settings.env) with its model (UkisAI's
  Swift-1.5 GSQ-RCO Q2_0, three shards), the head in place of the preset's, and two variables added:
  `LLAMA_DRAFT_SHARE=embd,head` and the step trace. Its whole line is [probe_line.txt](probe_line.txt). The load began
  at 16:35:44. The draft loaded and its shortlist was set up, and at 16:41:03 the server stopped on the check above.
  No request was sent.
- **Memory at the stop** ([probe_memory.tsv](probe_memory.tsv)): card 8, the card of the draft's layer, held
  5,970 MiB. With the preset's head, after its load in R205's `f1`, it held 5,545 MiB: +425 MiB. Each of cards 0–7 held
  75 MiB more than in `f1` at that moment, so est. 350 MiB of card 8's difference is the head's own. During R205's
  requests the preset's head took card 8 up to 5,858 MiB, 313 MiB above its after-load value. On that pattern the
  published head would need est. 6,208 MiB on card 8, 64 MiB over the card's 6,144 (est.). The lowest host
  MemAvailable during the probe was 456 MiB.
- **No kernel cases** ran on the rig, and nothing else ran on it during the probe. Rig 1 was clean afterwards (no llama
  process, 0 MiB on every card).

## The lane that runs once the head loads

[run/fn115.sh](run/fn115.sh) is R205's deep-lane design with the arm being the draft file: `own` (the preset's head)
and `dist` (the published head, with `LLAMA_DRAFT_SHARE=embd,head`); loads `warm:own f1:own d1:dist f2:own d2:dist`;
depths 0 and 8,192; 256 tokens at temperature 0 twice a depth plus the preset's sampler pass, every reply kept. Every
timed load also writes a step trace (`LLAMA_STEP_TRACE`), so each load is its own arm in the lane. [run/relabel.py](run/relabel.py)
names the arms `own` and `dist` again for `bench/deep_lane/deep_table.py`. [run/trace_table.py](run/trace_table.py)
reads the traces: the acceptance by draft position, the steps by how many tokens they accepted, and the draft's span a
step. [run/tables.sh](run/tables.sh) runs all three. The main model verifies every drafted token, so the greedy replies
must be the same bytes in both arms, and only the acceptance and the time may differ.

**The preset's head, by draft position.** The trace reader was checked on R121's two traced loads of the preset's head
on v4.0 ([2026-11-17_r121_flash_step_map](../2026-11-17_r121_flash_step_map/README.md), loads `w4h1` and `w4h2`, the
host profile on). Every request's steps added up to the server's own drafted and accepted counts. Those traces give
this head's acceptance by position, which no earlier table had ([r121_positions.md](r121_positions.md)):

| Depth, pass | tokens a step | P_1 | P_2 | P_3 | a_2 | a_3 | the draft's span, median ms |
|---|---:|---:|---:|---:|---:|---:|---|
| 0, greedy repeat | 2.370 | 0.642 | 0.434 | 0.321 | 0.676 | 0.739 | 10.32, 10.34 |
| 4,096, greedy repeat | 3.241 | 0.857 | 0.766 | 0.675 | 0.894 | 0.881 | 11.71, 11.70 |
| 8,192, greedy repeat | 3.606 | 0.942 | 0.870 | 0.841 | 0.923 | 0.967 | 12.31, 12.32 |
| 8,192, sampled | 3.459 | 0.903 | 0.861 | 0.764 | 0.954 | 0.887 | 12.32, 12.48 |

P_j is the share of steps whose j-th drafted token was accepted. a_j is the share accepted at position j of the steps
that accepted every token before it. Both are over the steps that drafted all three tokens, and both loads wrote the
same replies, so their counts are the same. The span runs from `s.drf` to `s.drf.end`: the drafter's three passes, with
the verify pass prepared inside the last one. On the greedy repeat at 0, 4,096 and 8,192 tokens, the steps that accepted
the first token accepted the second in 0.676, 0.894 and 0.923 of cases, and the steps that accepted two accepted the
third in 0.739, 0.881 and 0.967.

## Files

| File | What |
|---|---|
| [head_compare.md](head_compare.md) | every metadata key and tensor of the two heads, side by side, and the differing array entry |
| [probe_line.txt](probe_line.txt) | the probe's server line: the variables and flags, with the files by name |
| [probe_memory.tsv](probe_memory.tsv) | each card's MiB at the probe's stop, against R205's `f1` after its load and R205's peak |
| [r121_positions.md](r121_positions.md), [r121_positions.tsv](r121_positions.tsv) | the preset's head by draft position and its span a step, per request, from R121's traces |
| [run/fn115.sh](run/fn115.sh) | the run script: `STEP=dry`, `probe`, `pwait`, `pcollect`, `start`, `wait`, `collect` |
| [run/head_compare.py](run/head_compare.py) | two GGUF headers side by side, standard library only |
| [run/trace_table.py](run/trace_table.py) | acceptance by position and the draft's time a step from `LLAMA_STEP_TRACE` files |
| [run/relabel.py](run/relabel.py) | the lane's arms renamed for its tables |
| [run/tables.sh](run/tables.sh) | the tables of a collected run |

## Commands

```bash
# from the repository root on a machine that runs only ssh and tar; the rig's paths in the environment
RIG=<rig alias> WORK=<run folder on the rig> BIN=<pascal-p106-v5 602691153>/bin/llama-server \
  MODEL=<models>/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf \
  MODEL_DRAFT=<models>/mtp-Qwen3.8-Flash-Next-draft.gguf \
  MODEL_DIST=<models>/mtp-Qwen3.8-Flash-Next-shared-distillG-Q8_0.gguf \
  TEXT_LOCAL=<the release check's proxy.jsonl.bodies.gz> STEP=probe \
  bash docs/data/2026-11-19_fn115_flash_draft_head/run/fn115.sh
# then STEP=pwait, STEP=pcollect OUT=<folder>; the lane itself: STEP=dry, STEP=start, STEP=wait, STEP=collect OUT=<folder>
# the header table, on any machine with Python 3 (here the rig, both files there)
python3 -I docs/data/2026-11-19_fn115_flash_draft_head/run/head_compare.py <preset head> <published head> > head_compare.md
# the preset's head by position from R121's traces: a folder with loads.tsv, requests-<load>.csv and st-<load>.jsonl
python3 -I docs/data/2026-11-19_fn115_flash_draft_head/run/trace_table.py <folder> --arm-map w4h1=own,w4h2=own \
  --out r121_positions.md --tsv r121_positions.tsv
```

## Provenance and licence

The published head is hyukudan's, under the Qwen Community License 1.0 (`qwen-community-1.0`), and is not
redistributed here; only its header's contents are summarised. The preset's head and the model are those of
[release/flash-next-v4](../../../release/flash-next-v4/settings.env): UkisAI's Swift-1.5 GSQ-RCO Q2_0 of
Qwen3.8-Flash-Next with its MTP draft. The tables are this project's own measurements.
