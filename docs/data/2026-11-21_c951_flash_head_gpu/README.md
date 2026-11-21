# Flash-Next's draft-head fix, on three newer cards (C951, Q1687, 2026-11-21)

What this is: the GPU half of the check of [C898](../2026-11-21_c898_mtp_plain_attn_build/README.md) (registry
[Q1623](../../../quants/experiments.md)), point 5 of the GPU checks of the night's branches (registry
[Q1687](../../../quants/experiments.md)). A published distilled MTP head for Qwen3.8-Flash-Next marks its one layer as
plain attention (compress ratio 0), where the preset's kind of head marks a sparse-attention (QSA) layer (ratio 4). On
the v5 engine `602691153` the draft still built the sparse attention's k-pool inputs for such a layer; nothing read
them, so they got no memory, and the server stopped on `GGML_ASSERT(buffer)` before its first reply
([FN115](../2026-11-19_fn115_flash_draft_head/README.md), on a P106 rig). C898's branch (`pascal-p106-v5-mtp-plain-attn`
`c9c3134d4`, one commit on `602691153`) gives the draft its indexer cache and k-pool inputs only when its own block is a
QSA block. Its processor check passed; here the real files run on a build machine's three RTX 3090s, through the rig
build's compute 6.1 PTX (the same binaries a P106 rig runs). Correctness only: nothing here is timed.

## Result

**In plain words:** it passes. With a head of the preset's kind, the branch and the base give the same replies and the
same drafted and accepted counts on every request, greedy and sampled, on two target files. The distilled head now
loads, reads the target's token table and output head, gets no indexer cache of its own, drafts on every request, and
its greedy replies are the other head's, token for token. The base build with the distilled head stops on the same
`GGML_ASSERT(buffer)` FN115 saw on the P106 rig.

One criterion set before the run was wrong. It asked for the distilled head's greedy replies to equal plain decoding's
(no draft). They do on the coding request. On the prose and deep requests they leave plain decoding's at token 56 and
32 (Swift GSQ-RCO Q2_0) and at token 79 and 20 (Swift Qwen3.8-Flash-Next Pascal Q2 GGUF). The preset's kind of head
leaves at exactly the same tokens, on the base build too, so neither the head nor the fix causes it. It is registry
Q971's effect: a 4-token verify pass and 1-token decodes take different code paths, so a near tie can flip. Q971
already says to compare drafting arms with each other on real files, not with plain decoding. By that rule the
distilled head passes: its greedy replies are the preset-kind head's in all 12 greedy replies.

| Pass criterion (Q1687 point 5) | Result |
|---|---|
| (a) the preset's kind of head: the same replies and the same drafted/accepted counts as the base `602691153` | **yes**: all 20 replies a file (5 requests × 2 rounds) the same token ids, and the same drafted and accepted counts, on both files; the same log lines (the target's and the draft's indexer caches, the draft's k-pool tables and pool mask) and the same MiB per card after the load and after the requests |
| (b) the distilled head loads and drafts | **yes**: it answered all 10 requests on each file, drafting on every one (greedy: 233 to 286 drafted, 159 to 177 accepted); one indexer cache line (the target's), no k-pool table or pool mask for the draft |
| (b) its greedy replies are plain decoding's | **no on 2 of 3 greedy requests, for both heads and both builds alike** (above); its greedy replies are the preset-kind head's in all 12 (3 requests × 2 rounds × 2 files) |
| the base with the distilled head (as FN115) | stops on `ggml-backend.cpp:347: GGML_ASSERT(buffer) failed`, exit 134, on both files |

**The setup.** Qwen3.8-Flash-Next with its v5 preset ([release/flash-next-v5/settings.env](../../../release/flash-next-v5/settings.env),
`--spec-type draft-mtp`, three drafted tokens a step), the layer split changed to the three cards (`-ts 1,1,1`), `-lv 4`
(libllama's info lines, among them each "creating indexer KV cache"). Two target files, each in three shards (the third
the n-gram table): Swift GSQ-RCO Q2_0, the file the preset names (UkisAI's Swift-1.5 GSQ-RCO Q2_0), and Swift
Qwen3.8-Flash-Next Pascal Q2 GGUF (`pmix-q2-fast`, the published file). Two heads:

- **own**, a head of the preset's kind: Swift-1.5's own MTP layer
  ([2026-11-10_swift_own_mtp_head](../2026-11-10_swift_own_mtp_head/README.md)), written with the preset's Gyro draft's
  tensor names, order, shapes and types; its block is a QSA block, so it takes the code path the fix leaves as it was.
  The preset's own Gyro draft is kept on the rigs only.
- **dist**, the published distilled head
  ([hyukudan/Qwen3.8-Flash-Next-MTP-distillG-Q8_0](https://huggingface.co/hyukudan/Qwen3.8-Flash-Next-MTP-distillG-Q8_0),
  `mtp-Qwen3.8-Flash-Next-shared-distillG-Q8_0.gguf`, 2,786,568,256 bytes, sha256 `5516d3ce…4add4c3c` as published):
  no token table or output head of its own, so it runs with `LLAMA_DRAFT_SHARE=embd,head`; its experts in Q8_0; its
  block at ratio 0.

Five loads a file, one at a time:

| Arm | Build | Draft |
|---|---|---|
| own-new | the branch, `c9c3134d4` | own |
| own-ref | the base, `602691153` | own |
| dist-new | the branch | dist, `LLAMA_DRAFT_SHARE=embd,head` |
| plain-new | the branch | none: `-md` and every `--spec-*` flag removed |
| dist-ref | the base | dist, `LLAMA_DRAFT_SHARE=embd,head` (FN115's case) |

Each load that answered ran the requests twice ([run/session.py](run/session.py)), every one a fresh prompt (no prompt
cache), 256 tokens, the token ids returned: C854b's coding request and prose paragraph
([2026-11-18_c854b_p106_check](../2026-11-18_c854b_p106_check/README.md)), greedy; a deep prompt of 8,189 tokens (the
first 21,924 characters of [docs/switches.md](../../switches.md) as of this commit, cut at a line end, and a question
about them), greedy, so the draft attends over a cache 8,189 cells deep, the depth of FN115's deep lane; the deep prompt
and the coding request again with the preset's sampler (temperature 1.0, top-p 0.95, top-k 20, min-p 0) at seed 7.

**Every reply of the preset's kind of head is the same on both builds** ([agree.tsv](agree.tsv), [replies.tsv](replies.tsv)).
Drafted / accepted tokens, the same on the branch and the base:

| File | Round | code | prose | deep | deep, sampled | code, sampled |
|---|---|---|---|---|---|---|
| Swift GSQ-RCO Q2_0 | 1 | 236 / 176 | 276 / 162 | 238 / 174 | 236 / 176 | 246 / 172 |
| Swift GSQ-RCO Q2_0 | 2 | 236 / 176 | 270 / 164 | 232 / 176 | 234 / 176 | 241 / 174 |
| Pascal Q2 GGUF | 1 | 237 / 175 | 280 / 161 | 259 / 168 | 18 / 11 | 213 / 184 |
| Pascal Q2 GGUF | 2 | 234 / 176 | 277 / 162 | 250 / 171 | 18 / 11 | 213 / 184 |

The coding and prose requests on Swift GSQ-RCO Q2_0 give 236 / 176 and 276 / 162 in round 1, as C947 found on the
merged engine with the same head ([2026-11-21_c947_v5_merged_gpu](../2026-11-21_c947_v5_merged_gpu/README.md)). A
greedy reply is the same in both rounds of a load, but some counts move in round 2, alike on both builds: the draft
carries state from one request to the next. A sampled reply under the draft can move between rounds too, since
token-by-token rejection draws on what was drafted. Plain decoding's sampled replies repeat.

**The distilled head** (dist-new). Drafted / accepted tokens:

| File | Round | code | prose | deep | deep, sampled | code, sampled |
|---|---|---|---|---|---|---|
| Swift GSQ-RCO Q2_0 | 1 | 236 / 176 | 276 / 162 | 250 / 170 | 189 / 131 (195 tokens) | 239 / 175 |
| Swift GSQ-RCO Q2_0 | 2 | 233 / 177 | 276 / 162 | 247 / 171 | 308 / 151 | 229 / 178 |
| Pascal Q2 GGUF | 1 | 251 / 171 | 286 / 159 | 271 / 164 | 99 / 74 (108 tokens) | 210 / 184 |
| Pascal Q2 GGUF | 2 | 242 / 174 | 280 / 161 | 259 / 168 | 99 / 74 (108 tokens) | 213 / 184 |

Its greedy replies are the own head's token ids in every case; its sampled replies differ from the own head's (the
drafts differ, so the rejection draws differ). These counts come from a 3090 running the P106's code paths through the
PTX and from three requests; the head's acceptance against the preset's head on the P106 rig is FN115's lane, not this.

**Against plain decoding** (plain-new): the greedy coding request is the same with either head; prose and deep leave
plain decoding's at the same token with both heads (Swift GSQ-RCO Q2_0: 56 and 32; Pascal Q2 GGUF: 79 and 20),
in both rounds, and the base's own head with them.

**The log lines and memory** ([loads.tsv](loads.tsv)). Own head, both builds: two "creating indexer KV cache" lines
(the target's and the draft's), the draft's k-pool tables on its card and its pool mask. Distilled head on the branch:
one indexer cache line (the target's), neither of the draft's two. Distilled head on the base: the draft's indexer
cache and pool mask made, no k-pool tables (their set-up already skipped ratio-0 layers), then the assertion. MiB per
card after the load, cards 0, 1, 2 (the draft on card 2):

| File | own (both builds) | dist (branch) | plain (branch) |
|---|---|---|---|
| Swift GSQ-RCO Q2_0 | 13917, 13191, 16009 | 13917, 13191, 16327 | 13803, 13085, 12737 |
| Pascal Q2 GGUF | 13915, 13207, 15999 | 13915, 13207, 16317 | 13801, 13101, 12727 |

The distilled head takes 318 MiB more than the own head on the draft's card after the load (its experts in Q8_0, with
no token table or output head of its own and no indexer cache for the draft); after the requests the gap is 316 and
318 MiB. These are 3090 figures with a 24 GiB card; whether the head fits a P106's 6 GiB card is FN115's probe.

## Files

| File | What |
|---|---|
| [agree.tsv](agree.tsv) | per file, round and request: own head branch against base (token ids, drafted/accepted), dist against own, each head against plain decoding (or the first token that differs), each load's two rounds alike |
| [replies.tsv](replies.tsv) | every reply: prompt tokens, tokens, md5 of the token ids and of the text, drafted, accepted, stop reason |
| [loads.tsv](loads.tsv) | per load: build, head, answered, exit, the indexer cache lines, the draft's k-pool table and pool mask lines, the assertion, MiB per card |
| [run/run.sh](run/run.sh) | the driver: one llama-server per arm in a container, the requests twice, the memory read |
| [run/session.py](run/session.py) | the five requests |
| [run/c951_tables.py](run/c951_tables.py) | the tables from the driver's output |

## Commands

On the build machine, detached (the builds are the rig tarballs of `c9c3134d4`, md5
`9f0ab5f1cb5eb21cb57e1c2cd6a15014`, and of `602691153`; `llama-server` md5 `554c5313306e89ab97bbf09eaada002e` and
`86607661d5a19d1a90d19ee4db10c90b`; `$REPO` a copy of this repository at this commit):

```bash
WORK=$WORK NEW=$NEW REF=$REF REPO=$REPO OWN=$MODELS/mtp-Swift-1.5-Qwen3.8-Flash-Next-draft.gguf \
  DIST=$MODELS/mtp-Qwen3.8-Flash-Next-shared-distillG-Q8_0.gguf \
  MODEL_maker=$MODELS/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf \
  MODEL_pmix=$MODELS/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-pmix-fast-00001-of-00003.gguf MOUNTS="$MOUNTS" \
  DOCKER_ARGS="--label pascal-quants=1 --cpu-shares=256" PREFLIGHT=$PREFLIGHT \
  setsid nohup bash docs/data/2026-11-21_c951_flash_head_gpu/run/run.sh > $WORK/driver.log 2>&1 < /dev/null &
```

The driver runs the arms `own-new own-ref dist-new plain-new dist-ref` for each file on GPUs 0–2 in the image
`nvidia/cuda:12.9.2-devel-ubuntu24.04`, each server as

```bash
env $PRESET_ENV [LLAMA_DRAFT_SHARE=embd,head] $BIN/llama-server -m $MODEL [-md $HEAD] $PRESET_ARGS_WITH_TS_1_1_1 -lv 4 \
  --host 0.0.0.0 --port 18951 --no-webui
```

(the plain arm without `-md` and the `--spec-*` flags), with the PTX compiler's cache in `$WORK/jitcache`
(`CUDA_CACHE_PATH`), and `python3 -I run/session.py <server> <arm folder> r<n> $REPO/docs/switches.md 8192` twice per
load. The tables:

```bash
D=docs/data/2026-11-21_c951_flash_head_gpu
python3 -I $D/run/c951_tables.py $WORK/out $D
```

The server's timing lines carry host times; the tables do not read them.

## Provenance and licence

Code: Miner Llama (MIT, like llama.cpp), `pascal-p106-v5-mtp-plain-attn` `c9c3134d4` and `pascal-p106-v5` `602691153`,
the rig builds (CUDA 12.x, `sm_61`) run through their compute 6.1 PTX on three RTX 3090s. Models: UkisAI's Swift-1.5
GSQ-RCO Q2_0 of Qwen3.8-Flash-Next, Swift Qwen3.8-Flash-Next Pascal Q2 GGUF and Swift-1.5's own MTP layer, as in
[release/flash-next-v5](../../../release/flash-next-v5/settings.env) and the folders linked above. The distilled head
is hyukudan's, under the Qwen Community License 1.0 (`qwen-community-1.0`), and is not redistributed here. The prompts
are C854b's and this repository's own text; the tables hold counts and hashes, no reply text. Run on 2026-11-21.
