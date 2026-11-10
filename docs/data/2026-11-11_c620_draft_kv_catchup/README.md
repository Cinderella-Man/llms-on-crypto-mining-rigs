# C620 (Q1105): the MTP draft's catch-up writes only its caches (2026-11-10, the night's code rows)

With the MTP draft on, llama-server runs the draft layer over every token the target reads: each prompt chunk, and
the tokens each verify step accepts. These passes are the draft's catch-up (`process()` in the `draft-mtp` driver,
`common/speculative.cpp`). They mark no output row, because the next draft pass reads the target's hidden row and the
draft layer's caches, never a catch-up row's result. Yet the draft's graph computes the whole layer for every row: the
queries, the attention over the layer's whole cache, the output projection, the MoE, the hyper-connection combines and
the head's mixer. On the deep lane this costs the release line 28.4 % of prompt speed on an empty cache and 41.3 % on
32,188 cached tokens: the draft adds 2.23 and 5.89 ms a prompt token
([R39b](../2026-11-10_deep_lane/README.md#the-second-pair-the-mtp-draft-on-against-off-q1153)). A larger micro-batch
for the draft takes back 9.3 % of that, and the part that grows with depth not at all
([R53](../2026-11-10_deep_lane/README.md#the-fourth-run-the-drafts-micro-batch-in-prompt-reading-q1215)).

This folder holds the change, behind a new switch on Miner Llama's v4 line, and its CPU check. The registry row is
Q1105 in [quants/experiments.md](../../../quants/experiments.md); the switch is listed in
[docs/switches.md](../../switches.md). The idea is Warabii1248's `kvonly` (the Qwen-V100 fork, read in the
[Qwen-V100 note](../../research/notes/qwen_v100_llamacpp.md) §6.5) and llama.cpp
[#29523](https://github.com/ggml-org/llama.cpp/pull/29523) (closed unmerged). Upstream's open draft
[#31750](https://github.com/ggml-org/llama.cpp/pull/31750) does the same for Qwen3.5, Qwen3.5 MoE and Qwen3-Next, not
for Qwen3.8-Flash-Next's architecture (`qwen4exp`), whose draft layer is a sparse-attention (QSA) layer with its own
indexer cache.

**Result:** the switch `LLAMA_MTP_KV_ONLY=1` cuts the draft's catch-up passes to the writes of its caches. On the
CPU backend the draft's sequence state after a prompt, every draft pass's logits and hidden row, and the state at the
end (after a verify step's catch-up and one more draft pass) are the same bytes with the switch off and on. That holds
in 30 of 30 runs: five configurations (F16 and Q8_0 caches, flash attention on and off, the draft's top-k backend
chain) at three seeds, each with the rig build's CPU code and with a native CPU build. On the test's model the cut
catch-up computes 531 graph nodes where the full one computes 1,458. With the switch off the branch gives the base's
logits bit for bit on all 146 architectures `test-llama-archs` builds. The rig tarball builds (0 AVX instructions).
The gain on the cards, and the drafts' identity there, are R75 (Q1278).

## The change

Branch `pascal-p106-v4-b12246-c620` at **`ca3055285`**, two commits on `pascal-p106-v4` (`84f7bd3d2`, the `-ladder6`
tree): `9d2bcb94d` the change and its check, `ca3055285` the gate's test for `dist` (below):

| File | What |
|---|---|
| `src/llama-cparams.h`, `src/llama-context.cpp` | `LLAMA_MTP_KV_ONLY`, read when a context is made (`cparams.mtp_kv_only`); an MTP context logs one line when it is on |
| `src/llama-graph.cpp`, `src/llama-graph.h` | `kv_only_pass()`, the test of a pass whose result nobody reads, and `build_attn_kv_store()`, K and V rotated and stored as `build_attn` stores them (#31750's function, the same name and arguments) |
| `src/models/qwen4exp.cpp`, `src/models/models.h` | the draft graph (`graph_mtp`): on such a pass, after the attention mixer, `build_layer_attn_kv()` writes the layer's caches and the graph ends |
| `tests/test-llama-archs.cpp` | `--mtp-kv-only`: the CPU check below |

**A pass is cut when** the switch is on and the pass has no output row, extracts no unmasked hidden rows and no
embeddings. The `draft-mtp` driver's catch-up is such a pass: it asks for the hidden rows masked and marks no row.
One more case keeps the whole pass: a backend sampler chain that holds `dist`. Every graph that carries backend
samplers builds a chain per possible output, also when it has no row, and `dist` draws a random number for each chain
in every graph. A cut pass builds no chains, so with `dist` the later draws could come out of step. The test reads the
samplers' names, and a backend sampler's name carries a `+` once it runs on the backend (`+dist`): the first commit
compared it with `dist` and cut such passes too, which the check's fifth configuration caught (2,043 nodes off, 531 on,
where the counts must be equal), and the second strips the mark. The draft's own chain is top-k only (`-bs`, the
preset), so the catch-up is cut there.

**What a cut pass computes,** for every row as the full pass does, in the same ops and the same order:

1. the token row and the target's hidden row, normed and projected (`eh_proj`) into the four hyper-connection streams;
2. the attention mixer of those streams (`build_hc_mix`), the attention block's input;
3. on a QSA layer, the indexer's writes, in `build_qsa_sel`'s order: each row's raw indexer key into its cache row,
   then the blocks of four cells this micro-batch completes or regroups pooled, normed, rotated and scattered into
   their pooled slots;
4. K (projected, normed, rotated) and V (projected), rotated by the cache's Hadamard matrices when it has them, and
   stored.

**What it skips:** the queries and the gate, the indexer's queries, scores, top-k and mask, the attention itself, the
output projection, both hyper-connection combines, the MoE mixer, the routed and shared experts, the head's mixer and
the head. The skipped part holds everything that grows with the cache: the indexer scores every row against every
pooled key and the attention reads the cells it selects.

**With the switch off nothing changes:** `kv_only_pass()` is false, and the draft graph builds the same tensors in the
same order as before. The full path's functions (`build_layer_attn`, `build_qsa_sel`) are not touched; the cut pass
has its own copy of their cache writes.

**Inputs a cut pass no longer reads** stay unallocated, as they already do on graphs that need them not: the attention
mask (the hybrid memory's and the attention input's `set_input` skip a mask without a buffer), the output ids (not
built), the hidden-row carry's output index (not built). The k-pool inputs the QSA layers share are filled for every
micro-batch as before, also the selection inputs only the skipped part reads (`pool_mask`, `pool_cells`, `pool_idxs`,
`tail_idxs`); not filling them is Q1279.

**Graph reuse:** a cut pass has no output row, so its graph is reused only by another pass without outputs (the
reuse test compares the output count), and a draft pass builds its own graph as before.

## The CPU check

`run/checks.sh` on a build machine, CPU only. It builds the branch's rig tarball (`TESTS=1`, sm_61 code, no AVX) and
`test-llama-archs` from the same build tree, then the branch without CUDA (static, native), and runs:

- **kvonly:** `test-llama-archs --mtp-kv-only` with both builds' binaries at seeds 1, 2 and 3. The rig build runs in
  a container without a GPU against CUDA's stub driver, so its CUDA backend has no device and the CPU backend runs.
- **off:** `test-llama-archs -b CPU -s 1` on every architecture the test builds, the logits written by the base's
  CPU build (`84f7bd3d2`) and compared by the branch's. With the switch off the branch must compute what the base
  computes.

`--mtp-kv-only` builds a random `qwen4exp` model of the test's fixture (two trunk layers, hyper-connections of four
streams, QSA with blocks of four, MoE) with one MTP block after the trunk, also a QSA layer, and runs an MTP context
through what the `draft-mtp` driver does, once with the switch off and once on, from the same model, tokens and
hidden rows (random rows stand in for the target's):

1. the catch-up over a 100-token prompt in three batches without outputs (37, 38 and 25 tokens), in micro-batches of
   18, so that blocks of four complete across micro-batches;
2. the draft's sequence state (`llama_state_seq_get_data`: K, V, the indexer's raw and pooled keys);
3. a draft step: a pass at position 100 with the target's row, then one at 101 with the first pass's own row;
4. the draft region removed (positions 100 and on), the catch-up over the step's three accepted tokens, which
   regroups the blocks from position 100;
5. the next step's first pass at 103, and the sequence state again.

| Config | Cache | Flash attention | Backend sampler on the draft |
|---|---|---|---|
| `f16 fa` | F16 | on | none |
| `f16` | F16 | off | none |
| `q8_0 fa` | Q8_0 | on | none |
| `f16 fa top-k` | F16 | on | top-k 10, the `draft-mtp` driver's chain |
| `f16 fa dist` | F16 | on | top-k 10 and dist |

A configuration passes when the state after the prompt, every draft pass's logits (with a backend sampler: its sampled
logits, candidates and token) and hidden row, and the state at the end are the same bytes with the switch off and on,
and when the catch-up computed fewer graph nodes with the switch on (with `dist`: the same number).

### Results

| Step | Verdict | Detail |
|---|---|---|
| rig tarball | OK | `llama.cpp-20261110-ca3055285-pascal-p106-v4-b12246-c620.tar.gz`, 647,388,245 bytes, md5 `f6347afe9083e519d1d8b689a0c8b3a6`; sm_61, `TESTS=1`, 0 AVX instructions in `llama-bench`, GLIBC_2.38 at most |
| `test-llama-archs`, rig tree | OK | built from the tarball's CUDA build tree into its `bin/` |
| CPU build | OK | static, native, without CUDA, `ca3055285` |
| kvonly, rig build | OK | seeds 1, 2, 3: 5 of 5 configurations each |
| kvonly, CPU build | OK | seeds 1, 2, 3: 5 of 5 configurations each |
| off | OK | the base (`84f7bd3d2`) wrote 146 architectures' logits, the branch compared 146: 0 differ; 151 of 151 test cases pass in both |

[summary.tsv](summary.tsv) holds the verdicts, [kvonly.tsv](kvonly.tsv) one row per build, seed and configuration.
The rows are the same at every seed and in both builds:

| Config | KV after the prompt | Drafts | KV at the end | Catch-up nodes, off / on |
|---|---|---|---|---|
| `f16 fa` | identical | identical | identical | 1,458 / 531 |
| `f16` | identical | identical | identical | 1,503 / 549 |
| `q8_0 fa` | identical | identical | identical | 1,566 / 585 |
| `f16 fa top-k` | identical | identical | identical | 1,647 / 531 |
| `f16 fa dist` | identical | identical | identical | 2,043 / 2,043 |

The node counts are those of the four catch-up calls together (the prompt's three and the verify step's one), counted
by the graph callback. With a backend sampler the full pass also builds its four chains for the rows it does not have
(189 nodes more with top-k, 585 with top-k and dist); a cut pass builds none. With `dist` the pass stays whole, as the
gate says.

**The first run, on `9d2bcb94d`,** stopped after its first seed: four configurations passed as above and `f16 fa dist`
failed on its node counts (2,043 off, 531 on). The gate had cut the passes with a `dist` chain (the name was `+dist`).
Its caches and drafts were identical all the same, because the draws of chains without a row did not reach the drafts
in this check. The gate stays, because a sampler whose state such a draw moves would make the drafts depend on the
switch. The second commit fixed the name test, and every step ran again on `ca3055285`.

## How to run it

On a build machine with Docker, from a checkout of this repository, with the fork as remote `fork` of the work
folder's llama.cpp clone (`scripts/build_llamacpp_docker.sh`, its header):

```bash
WORK=<work dir> BUILD=scripts/build_llamacpp_docker.sh BRANCH=pascal-p106-v4-b12246-c620 \
BASE_CPU=<bin/ of the base's static native CPU build> JOBS=8 LOCK=<flock file> \
setsid -f nohup bash docs/data/2026-11-11_c620_draft_kv_catchup/run/checks.sh > <work dir>/logs/driver.out 2>&1 < /dev/null
```

Verdicts land in `$WORK/summary.tsv`, the check's table in `$WORK/checks/kvonly/kvonly.tsv`
([kvonly.tsv](kvonly.tsv) here), and `$WORK/C620_DONE` marks the end. One configuration by hand:

```bash
<build>/bin/test-llama-archs --mtp-kv-only -s 1
```

## On the rig

Not run yet. R75 (Q1278) relays this tarball to a rig and runs two things. First, untimed loads of the deep lane at 0
and 32,768 tokens with `LLAMA_SPEC_DUMP` in the control and in the cut arm: every verify step's drafted tokens and
probabilities must be the same wherever the control repeats itself. Second, R53's deep-lane primes with the control,
the cut arm (`ARM_ENV_kvonly=LLAMA_MTP_KV_ONLY=1`) and no draft, two loads each, read with
[`draft_ubatch.py`](../2026-11-10_deep_lane/run/draft_ubatch.py) for the share of the draft's prompt cost the cut
takes away at each depth. FN68 (Q1280) checks the identity through the compute 6.1 PTX on a build machine's card
first, if a card is free. The commands, with the `wait` and `collect` steps of
[bench/deep_lane](../../../bench/deep_lane/README.md) after each:

```bash
# (a) identity, untimed; the two dumps are written in the rig's work folder: copy them off before it is removed
RIG=rig BIN=<c620 build>/bin/llama-server \
MODEL=<models>/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf \
MODEL_DRAFT=<models>/mtp-Qwen3.8-Flash-Next-draft.gguf \
TEXT_LOCAL=<recorded>/proxy.jsonl.bodies.gz TAG='^t1swift/' WD_ENV=MIN_AVAIL_MB=60 \
ENV_ADD=GGML_CUDA_FATTN_SPARSE_PASCAL=1 N_PREDICT=16 DEPTHS="0 32768" ARMS="draft kvonly" \
ARM_ENV_draft=LLAMA_SPEC_DUMP=/tmp/deep-lane/dump-draft.jsonl \
ARM_ENV_kvonly="LLAMA_MTP_KV_ONLY=1 LLAMA_SPEC_DUMP=/tmp/deep-lane/dump-kvonly.jsonl" \
LOADS="draft1:draft kvonly1:kvonly draft2:draft" \
bash bench/deep_lane/remote.sh start

# (b) the prompt table: R53's primes
RIG=rig BIN=<c620 build>/bin/llama-server \
MODEL=<models>/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf \
MODEL_DRAFT=<models>/mtp-Qwen3.8-Flash-Next-draft.gguf \
TEXT_LOCAL=<recorded>/proxy.jsonl.bodies.gz TAG='^t1swift/' WD_ENV=MIN_AVAIL_MB=60 \
ENV_ADD=GGML_CUDA_FATTN_SPARSE_PASCAL=1 N_PREDICT=16 \
ARMS="draft kvonly nodraft" ARM_ENV_kvonly=LLAMA_MTP_KV_ONLY=1 ARM_DRAFT_nodraft=off \
LOADS="warm:draft draft1:draft kvonly1:kvonly nodraft1:nodraft draft2:draft kvonly2:kvonly nodraft2:nodraft" \
bash bench/deep_lane/remote.sh start
python3 -I docs/data/2026-11-10_deep_lane/run/draft_ubatch.py <results folder> --out draft_ubatch.md --tsv draft_ubatch
python3 -I docs/data/2026-11-10_deep_lane/run/host_memory.py <results folder> > host_memory.tsv
```

**What to expect (est.):** in R53 the preset's draft added 217.7 s to a load's four prompt chunks. Its cost per prompt
token rose by 3.60 ms from an empty cache to 32,188 cached tokens, and R53 reads that growing part as the draft layer's
attention (not measured). All of the layer's work that grows with the cache is in the skipped part. If the growing
part goes, the chunk read on 32,188 cached tokens takes est. 10.69 and 10.67 ms a token instead of the 14.29 and
14.27 measured in R53's two loads (70.0 and 70.1 tok/s): est. 93.5 and 93.7 tok/s, against 119.5 and 119.6 without the
draft. The share of the constant part (2.23 ms a token on an empty cache in R39b) that goes with the queries, the MoE
and the head is not known. The cut keeps `eh_proj` over
the four streams, the mixer, K, V and the indexer's keys.

## Files

| File | What |
|---|---|
| [run/checks.sh](run/checks.sh) | the build and the CPU check (steps `build`, `cpubuild`, `kvonly`, `off`), as run |
| [summary.tsv](summary.tsv) | the verdict of every step on `ca3055285` |
| [kvonly.tsv](kvonly.tsv) | `--mtp-kv-only`: one row per build, seed and configuration (the identity of the KV after the prompt, of the drafts and of the KV at the end; the catch-up's node counts off and on) |

The check itself is `test-llama-archs --mtp-kv-only` in the fork's `tests/test-llama-archs.cpp` (branch
`pascal-p106-v4-b12246-c620`).

## Provenance and licence

Every number here comes from random models that `test-llama-archs` builds from its seeds; no model file, prompt or
recorded text was read. The fork is MIT-licensed, as llama.cpp; these tables and the script are published with this
repository.
