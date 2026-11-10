# Why the draft line's greedy reply differs from the plain line's (Q1175; 2026-11-10)

What this is: the Miner Llama v3 release line ([release/flash-next-v3/settings.env](../../../release/flash-next-v3/settings.env):
Qwen3.8-Flash-Next, Swift Q2_0, an MTP draft of three tokens) is meant to write the model's own answers: the draft only
proposes tokens, and the main model keeps a proposed token when it is the token it would have written. At temperature
0 the line with the draft and the line without it (the plain line) should then write the same text. In the run that
counted the draft's acceptance by depth
([2026-11-10_draft_acceptance_by_depth](../2026-11-10_draft_acceptance_by_depth/README.md), registry row Q1172) they
did not, at any of five context depths, while each line repeated its own reply on a second request. This run finds
out why. Registry row Q1175 in [quants/experiments.md](../../../quants/experiments.md).

Three causes were possible:

1. **The card split.** That run left the split to llama-server. The residual crosses each card boundary as Q8_0
   (`LLAMA_HOP_CAST=q8_0`), which is lossy, so a boundary at another layer in one arm would change the logits.
2. **Rounding at a near-tie.** The draft line checks a draft in one pass over up to four tokens, the plain line
   decodes one token a pass. Results that differ in the last bits can turn a choice between two tokens of almost equal
   probability.
3. **A fault in the speculative path,** which would show the same way, but with tokens the main model clearly does
   not prefer.

No time is read or reported here: the run was on newer cards through the build's compute 6.1 PTX.

## Result

**Not the card split, and not a near-tie: where the two lines part, the two candidates are 0.07 to 1.78 apart in
log-probability. Not the draft's accept step or its rollback either: a draft of one token and a draft of three, whose
steps fall differently, write the same 256 tokens with the same log-probabilities to the last digit. The lines part
because a pass over two or more tokens is another arithmetic than a one-token pass, and this model turns such a
difference into tenths of a log-probability within a few tokens. The plain line does the same to itself: by reading
its own tokens back in one pass, and at 8,192 tokens and deeper from one load to the next.**

A lead is the difference of two tokens' log-probabilities at one position, which is the difference of their logits.
The line between rounding and more is 0.05.

### Where the two replies part

Both arms ran with the same explicit split, and every layer sat on the same card in both ([split.tsv](split.tsv)).
The replies still part at every depth ([divergence.tsv](divergence.tsv)):

| Depth (tokens) | First differing token | Plain line's token, draft line's token | Plain line while generating: lead of its token | Draft line while generating: lead of its token | Plain line, the common tokens read back in one request: first choice, lead of the plain reply's token |
|---:|---:|---|---:|---:|---|
| 0 | 7 | ` verify`, ` run` | 0.2100 | 0.2032 | the draft line's token, −0.6213 |
| 8,192 | 2 | ` response`, ` listing` | 1.7798 | 0.7537 | the plain line's, 1.6685 |
| 16,384 | 17 | ` test`, ` whole` | 0.1516 | 0.2965 | the plain line's, 0.4698 |
| 32,768 | 26 | ` I`, ` The` | 0.0692 | 0.1086 | the plain line's, 0.1044 |
| 49,152 | 43 | ` has`, ` uses` | 0.1443 | 0.1581 | the plain line's, 0.2151 |

- **Each line wrote the first choice of its own distribution.** Over all five depths every token of both replies is
  the most probable token the line listed for that position (end of text aside, which the requests ignore).
- **The same two candidates, opposite leads.** At depth 0 the plain line gives ` verify` 0.2100 over ` run`; the
  draft line, after the same seven tokens, gives ` run` 0.2032 over ` verify`.
- **The plain line turns the same choice by itself.** Reading the seven common tokens back as a prompt (the prompt's
  last four tokens and the seven: 11 tokens in passes of several) it prefers the draft line's token by 0.6213. That
  is the same line, the same tokens and no draft: only the passes over several tokens differ from generating.
- **The gap opens with the first verify pass.** At depth 0 the log-probability of the first reply token is the same
  in both lines to the last digit (it comes from the prompt's last pass). For the next six common tokens the two lines
  differ by 0.0138, 0.0109, 0.0073, 0.0017, 0.0009 and 0.3214 ([path_shift.tsv](path_shift.tsv)).

### Both replies teacher-forced on the plain line

For every token of a reply the plain line was given the prompt and the reply up to that token and asked for one
token, one token a pass ([teacher_forced.tsv](teacher_forced.tsv), every miss in
[off_first_choice.tsv](off_first_choice.tsv)). "Not its first choice" leaves out end of text.

| Depth (tokens) | The plain line's own reply: tokens, not its first choice (above 0.05), largest lead lost | The same log-probability as while generating | The draft line's reply from the first differing token: tokens, not the plain line's first choice (above 0.05), largest lead lost |
|---:|---|---:|---|
| 0 | 256, 0 | 256 of 256 | 249, 5 (3), 0.3781 at token 37 |
| 8,192 | 256, 4 (3), 0.5409 at token 206 | 22 of 256 | 254, 5 (5), 0.3936 at token 3 |
| 16,384 | 256, 3 (3), 0.5914 at token 17 | 44 of 256 | 239, 3 (2), 0.0825 at token 24 |
| 32,768 | 256, 5 (5), 0.4607 at token 68 | 23 of 256 | 230, 10 (10), 0.3640 at token 44 |
| 49,152 | 256, 5 (4), 0.5577 at token 141 | 19 of 255 | 213, 3 (3), 0.9267 at token 118 |
| all | 1,280, 17 (15), 0.5914 | | 1,185, 26 (23), 0.9267 |

- **At depth 0 the plain line repeats itself exactly and the draft line's reply misses 5 of 249 first choices** (2.0 %),
  by 0.0382 to 0.3781. This is the clean measure of the verify pass against one-token decoding: nothing else differs
  there.
- **At 8,192 tokens and deeper the plain line misses its own first choice as often as the draft line's reply does:**
  17 of 1,024 tokens of its own reply against 21 of 936 of the draft line's. To score a reply the server reads the
  prompt again, and the second reading does not give the first one's state: at 8,192 the plain line's first own
  token loses by 0.4449 to another.
- **The distance between the lines is the distance of the plain line from itself.** The KL divergence over the listed
  tokens, the draft line generating against the plain line teacher-forced on the same tokens, is 0.0055, 0.0044,
  0.0011, 0.0101 and 0.0056 at the five depths. The plain line generating against itself teacher-forced: 0.0000,
  0.0049, 0.0057, 0.0057 and 0.0049.

### The plain line against itself, load against load

The plain line was loaded twice in this run with the same requests in the same order ([repeat.tsv](repeat.tsv)).

| Depth (tokens) | The same 256 tokens | First differing token | Log-probability of the same token before it: exactly equal, largest difference | The first token's difference |
|---:|---|---:|---|---:|
| 0 | yes | — | 256 of 256, 0.0000 | 0.0000 |
| 8,192 | no | 24 | 0 of 24, 0.1130 | 0.0236 |
| 16,384 | no | 17 | 0 of 17, 0.1139 | 0.0088 |
| 32,768 | no | 26 | 0 of 26, 0.1440 | 0.0806 |
| 49,152 | no | 141 | 11 of 140, 0.3599 | 0.0003 |

- At depth 0 both arms also wrote the acceptance run's replies again, hash for hash. At the four depths from 8,192
  neither arm did.
- One load can also repeat a reading exactly. In the first plain load of this run (stopped during teacher forcing by
  a fault of the client, see Method) the 8,128-token prompt read a second time from nothing gave all 256
  log-probabilities of the reply to the last digit; in the second load it gave 22 of 256.
- In that first load the request with probabilities, which starts from a restored checkpoint, parted from the
  request before it at 49,152 tokens, at token 144, where its token led by 0.5174. In the other 24 of 25 cells of
  this run (five loads, five depths) the two requests wrote the same tokens.
- **So at 8,192 tokens and deeper a greedy text is one draw.** Row Q575 of the registry follows the run-to-run
  difference; row Q1177 what turns a rounding difference into tenths on this model (the experts' and the attention's
  selections are the candidates). The older Flint measured the same way in row Q21.

### The draft at one token, a second load of the draft line, and the verify steps

Because the leads are beyond 0.05, the draft line ran twice more: with a draft of one token
(`--spec-draft-n-max 1`) and with the preset's three again, both with the server's record of every verify step. A
cell is the first eight characters of the hash of a reply's 256 token ids ([hashes.tsv](hashes.tsv)):

| Load | 0 | 8,192 | 16,384 | 32,768 | 49,152 |
|---|---|---|---|---|---|
| the acceptance run, plain | `84d85f5e` | `9eb51303` | `41cfc791` | `a423bdef` | `9c6a0daa` |
| plain, first load | `84d85f5e` | `fb838580` | `31d479c4` | `479cf06a` | `6a5ce791` |
| plain, second load (the one scored) | `84d85f5e` | `339e9d80` | `2ea4b33d` | `9c38c0d1` | `6c8165de` |
| the acceptance run, draft | `dcd8ac53` | `2ceddbaa` | `aa1e6568` | `f0401cae` | `6c850da2` |
| draft, first load (the one scored) | `dcd8ac53` | `df205624` | `bebe917b` | `833c3ec1` | `8fd44563` |
| draft, second load | `dcd8ac53` | `df205624` | `bebe917b` | `88326a47` | `09680ed9` |
| draft at one token | `dcd8ac53` | `2f10ac7a` | `41cfc791` | `09915dee` | `ca910dfe` |
| different replies | 2 | 6 | 5 | 7 | 7 |

- **At depth 0 there are two texts: the plain line's in all three of its loads and the draft line's in all four of
  its loads,** the one-token draft among them.
- **The draft at one token and the draft at three agree in all 256 log-probabilities to the last digit at depth 0**
  ([repeat.tsv](repeat.tsv)), although their steps fall differently: 135 steps with 15 drafts rejected against 84
  steps, 26 of them partly accepted and 12 with nothing accepted
  ([steps_draft1.tsv](steps_draft1.tsv), [steps_draft3.tsv](steps_draft3.tsv)). A fault in the accept step or in the
  rollback after a rejected token would have to leave the same trace in both to stay hidden. So a verify pass of
  two tokens and one of four are the same arithmetic, and the one-token pass is another.
- **From 8,192 tokens on the text depends on the load, not on the draft:** the seven loads wrote 6, 5, 7 and 7
  different texts. The draft line's two loads agree in every log-probability at 8,192 and 16,384 and part at 32,768
  (token 33) and 49,152 (token 43). At 16,384 the draft at one token wrote the 256 tokens the plain line wrote in the
  acceptance run.
- **Where the first differing token sits in the draft line's steps** (three tokens; the second load wrote the first
  load's tokens at depths 0, 8,192 and 16,384). Depth 0, token 7: an accepted draft token of a step that kept all
  three, after a step that kept two of three. 8,192, token 2: the own token of a step that kept none of its three,
  after a step that kept none. 16,384, token 17: the own token of a step that kept one of three, after a step that
  kept all three. So it follows a draft that was not kept whole at two of the three depths.
- **The misses gather after drafts that were not kept whole, as uncertain tokens do.** Of the draft line's 742 scored
  tokens at those three depths 286 follow such a step (39 %); of the 13 that are not the plain line's first choice,
  10 do ([off_tokens_draft3.tsv](off_tokens_draft3.tsv)). A draft is rejected where the next token is uncertain,
  and a small lead is turned there by any change. The one-token line, whose rollbacks fall elsewhere, writes the
  same tokens.

**What it decides.**
- **The split is cleared** as the cause: the server's own split and this run's explicit one are the same layers on
  the same cards, and the replies part with it.
- **The accept step and the rollback are cleared at depth 0,** where a text can be compared exactly: two draft
  lengths, the same log-probabilities. At depth the gap between the lines has the size of the gap between two loads
  of one line. The search moves from the speculative path to the pass shapes (new row Q1179) and to rows Q1177 and
  Q575.
- **What the draft line keeps:** every token it writes is the main model's first choice in the pass that verified it.
  It does not keep the plain line's text: at depth 0, where nothing else varies, the two part after 7 tokens, and 2.0 %
  of the draft line's later tokens are not the plain line's first choice. A sentence on a model page should say the
  first and not promise the second.
- **A greedy text is not a correctness check between two arms of this line** from 8,192 tokens on, as row Q1171
  found for the attention kernels: there the plain line parts from its own second load within 17 to 141 tokens.
  Depth 0 can be compared hash for hash.
- New rows: Q1179 (which operation computes one token another way than two or more, and a verify pass that repeats
  one-token decoding bit for bit) and Q1180 (whether a pass over several tokens is worse than one-token passes or
  only different: KL divergence by pass width against one reference).

**Limits.**
- **One tail, one load of the draft line scored, 256 tokens per depth.** At 8,192 tokens and deeper another load of
  either line writes another text, so each cell there is one sample; depth 0 repeats in every load of both arms.
- **Newer cards, llama.cpp's own kernels.** On a card newer than compute 6.1 the build's Pascal kernels stay off at
  the preset's defaults (row Q1178), so the verify pass and the one-token pass compared here are llama.cpp's
  kernels. The P106's kernels for passes of two to five tokens are other code than its one-token kernels; how far
  they move the same choices is not measured here.
- **Ten tokens are listed per position.** Every lead in the tables is between listed tokens. For three tokens whose
  text ends inside a character the server lists no probabilities; they count by the token the server wrote.
- **Teacher forcing reads the reply one token a pass** after reading the prompt again. The draft line's reply starts
  from the prompt's tail and the common tokens read in one pass, so its first scored token stands on a state the
  plain line's own pass does not have (at depth 0 that is why the first differing token counts as the plain line's
  first choice in the draft line's row).

## Method

- **The line:** the release preset without its nine-card split (`-ts`), the draft's device override (`-otd`) and the
  keep-alive variable, with one explicit split for both arms, on three cards (two RTX 3090 and an RTX 3080 12 GB).
  The arm `nodraft` also drops `-md` and the `--spec-*` flags. The build is `pascal-p106-v3-b12076-ladder5`
  (`1551f5fe1`); it holds only sm_61 code and compute 6.1 PTX, which the newer cards' driver compiles.
- **The split:** `-ts 23861,23861,11681`, the three cards' free memory in MiB in llama-server's device order (the two
  RTX 3090 first), which are the proportions the server takes by itself. Each arm's line was started once at `-lv 5`,
  which prints the card of every layer, and stopped when those lines were out: layers 0 to 19 on the first card, 20
  to 39 on the second, 40 to 48 on the third, in both arms. After loading, the cards held 16,322, 16,284 and 9,465
  MiB with the draft and 16,188, 16,152 and 7,223 MiB without it, the same as in the acceptance run.
- **The prompts:** tail 1 of the acceptance run, token for token (its `prompts.json`): recorded coding-agent
  requests rendered by the server's chat template and cut to 0, 8,192, 16,384, 32,768 and 49,152 tokens, each ending
  in the same 64-token tail (depth 0 is the tail alone).
- **The replies** ([run/exact_client.py](run/exact_client.py), phase 1): per arm one server load, and per depth the
  request order of [bench/deep_lane](../../../bench/deep_lane/README.md): the prompt without its tail, then the whole
  prompt with 256 tokens at temperature 0 (`top_k` 1, end of text ignored), then the same request with `n_probs` 10.
  The second request returns, for every token written, the ten most probable tokens with their log-probabilities as
  the line computed them while generating: one token a pass on the plain line, the verify pass on the draft line.
- **The lead at a divergence point** is read three ways: from the plain line while it generated, from the draft line
  while it generated, and from the plain line given the prompt and the common tokens as a prompt, one token asked
  (`probe`; the server restores its checkpoint four tokens before the prompt's end and reads those four and the
  common tokens).
- **Teacher forcing** (phase 2, on the plain line): for every token i of a reply, the prompt and the reply's first i
  tokens are sent with the prompt cache on, and one token is asked with `n_probs` 10. The cache already holds all but
  the last token, so the server reads one token: the path of one-token decoding. Before each reply the prompt is
  read again as in phase 1. The plain line's own reply is scored from token 0, the draft line's from the first
  differing token (before it the prompts are the same).
- **End of text** is ignored by every request, as in the acceptance run, so a reply goes on where the model would
  stop. A token listed above the one a greedy line wrote was therefore not allowed there, and
  [run/exact_table.py](run/exact_table.py) leaves such tokens out of a first choice (16 positions in the 2,465
  scored).
- **Three loads of the plain line.** The first attempt gave the split in the cards' bus order and ran out of memory
  on the 12 GB card while loading. In the second, the client stopped at depth 8,192 on a reply token whose text ends
  inside a character (the server lists no probabilities for such a token); its replies at all five depths are the
  first load of the load-against-load table, and its tables are in [first_plain_load/](first_plain_load/). The third
  load, with the client corrected, is the one scored above.
- **The draft at one token and the step dump:** two more loads of the draft line, `draft1` with
  `--spec-draft-n-max 1` and `draft3` with the preset's three, both with `LLAMA_SPEC_DUMP=<file>`, the server's own
  record of every verify step: the position, the drafted tokens and the tokens the step yielded. The dump changes no
  reply: with it the draft line wrote the same hashes at depths 0, 8,192 and 16,384 as without it.
  [run/exact_steps.py](run/exact_steps.py) places each reply token in its step.
- **Not located:** which operation makes a one-token pass differ from a pass of two or more tokens. The run shows
  that they differ and that passes of two and four tokens do not; row Q1179 takes it from there.

## Files

| File | What |
|---|---|
| [summary.md](summary.md) | all tables below as `exact_table.py` printed them, with its verdict line per depth |
| [divergence.tsv](divergence.tsv) | per depth: the first differing token, the two candidates (id and text), their log-probabilities and lead as each line generated and as the plain line read the common tokens back, whether each reply equals the acceptance run's, and whether each line's second request wrote the first one's tokens |
| [teacher_forced.tsv](teacher_forced.tsv) | per depth and reply: tokens scored on the plain line, positions where end of text is the most probable token, tokens not its first choice, those above 0.05, the largest lead lost and where; a last row per reply over all depths |
| [off_first_choice.tsv](off_first_choice.tsv) | every scored token that is not the plain line's first choice, with the lead it loses by |
| [path_shift.tsv](path_shift.tsv) | per depth and pair of paths: the difference in the log-probability of the same token after the same tokens (exactly equal, median, 90th percentile, largest) and the KL divergence over the listed tokens |
| [repeat.tsv](repeat.tsv) | two loads of one line: the first differing token, the differences before it, each load's lead of its own token there |
| [split.tsv](split.tsv) | the card of every layer in both arms, and the cards' memory after loading |
| [first_plain_load/](first_plain_load/) | `divergence.tsv`, `teacher_forced.tsv`, `off_first_choice.tsv` and `path_shift.tsv` of the draft line's load against the first plain load (teacher forcing: depth 0 whole, 8,192 the plain reply and 100 tokens of the draft line's) |
| [hashes.tsv](hashes.tsv) | every load's reply per depth as the first eight characters of its hash: the acceptance run's two arms and this run's five loads |
| [steps_draft3.tsv](steps_draft3.tsv), [steps_draft1.tsv](steps_draft1.tsv) | per depth: the load's verify steps (drafted, accepted, steps by outcome) and the step that wrote the first token differing from the scored plain reply |
| [off_tokens_draft3.tsv](off_tokens_draft3.tsv), [off_tokens_draft3_rates.tsv](off_tokens_draft3_rates.tsv) | the draft line's tokens that are not the plain line's first choice at depths 0, 8,192 and 16,384, each with its step; the same two facts counted over all scored tokens |
| [draft_one_token/](draft_one_token/), [draft_second_load/](draft_second_load/) | `divergence.tsv` and `path_shift.tsv` of those two loads against the scored plain load |
| [run/run_exactness.sh](run/run_exactness.sh) | the loads, one after another: the split lines, the draft line's replies, the plain line's replies and scoring; hosts and paths from the environment |
| [run/exact_client.py](run/exact_client.py) | one load's requests: the replies, the probe, the teacher forcing |
| [run/exact_table.py](run/exact_table.py) | the tables from the client's files |
| [run/exact_repeat.py](run/exact_repeat.py) | two loads of one line, token by token |
| [run/exact_steps.py](run/exact_steps.py) | the draft line's verify steps around each token that is not the plain line's |

Not published: the prompts' and replies' token ids and text, each request's listed tokens and the server logs
(recorded text and paths).

## Commands

On the machine with the cards, from a checkout of this repository (`REPO`), with the acceptance run's
`tail1/prompts.json` and `tail1/requests.csv` and the model files:

```bash
WORK=<work folder> REPO=<checkout> BIN=<build>/bin/llama-server \
MODEL=<models>/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf \
MODEL_DRAFT=<models>/mtp-Qwen3.8-Flash-Next-draft.gguf \
PROMPTS=<acceptance run>/tail1/prompts.json REF_CSV=<acceptance run>/tail1/requests.csv \
GPUS=<three cards, as docker names them> TS=23861,23861,11681 \
MOUNTS="<the folders of WORK, the build, the models>" DOCKER_ARGS="--cpu-shares=256" NAME=fn44 PORT=18144 \
setsid -f nohup nice -n 19 bash docs/data/2026-11-10_draft_line_exactness/run/run_exactness.sh > console.txt 2>&1 < /dev/null
```

It runs the steps `split:nodraft split:draft gen:draft score:nodraft` and ends with `WORK/chain.done`. The plain
line's second load is the same call with `STEPS="score:nodraft"` after the first load's files are moved aside.
The two loads with the step dump are the same call with other steps:

```bash
STEPS="gen:draft1 gen:draft3" ENV_draft1="LLAMA_SPEC_DUMP=<work folder>/spec_dump_draft1.jsonl" \
ENV_draft3="LLAMA_SPEC_DUMP=<work folder>/spec_dump_draft3.jsonl" <the variables above> \
setsid -f nohup nice -n 19 bash docs/data/2026-11-10_draft_line_exactness/run/run_exactness.sh > console2.txt 2>&1 < /dev/null
```

Then the tables (`P` is the folder of the first plain load's files, with a copy of `replies_draft.json`):

```bash
cd <work folder> && R=<checkout>/docs/data/2026-11-10_draft_line_exactness/run
nice -n 19 python3 -I $R/exact_table.py . --out tables
nice -n 19 python3 -I $R/exact_table.py $P --out $P/tables
for a in draft1 draft3; do nice -n 19 python3 -I $R/exact_table.py . --draft $a --out tables_$a; done
nice -n 19 python3 -I $R/exact_repeat.py $P/replies_nodraft.json replies_nodraft.json \
  --label "nodraft: two loads" --out tables/repeat.tsv
nice -n 19 python3 -I $R/exact_repeat.py replies_draft.json replies_draft3.json \
  --label "draft: two loads (the second with the step dump)" --append --out tables/repeat.tsv
nice -n 19 python3 -I $R/exact_repeat.py replies_draft.json replies_draft1.json \
  --label "draft at three tokens against draft at one token" --append --out tables/repeat.tsv
nice -n 19 python3 -I $R/exact_steps.py --dump spec_dump_draft3.jsonl --replies replies_draft3.json \
  --plain replies_nodraft.json --same-as replies_draft.json --requests requests_nodraft.jsonl --tf-reply draft --out tables
nice -n 19 python3 -I $R/exact_steps.py --dump spec_dump_draft1.jsonl --replies replies_draft1.json \
  --plain replies_nodraft.json --out tables
nice -n 19 python3 -I $R/exact_hashes.py --out tables/hashes.tsv --ref-csv <acceptance run>/tail1/requests.csv \
  --ref-label "the acceptance run" "nodraft, first load=$P/replies_nodraft.json" \
  "nodraft, second load (scored)=replies_nodraft.json" "draft, first load (scored)=replies_draft.json" \
  "draft, second load (step dump)=replies_draft3.json" "draft at one token (step dump)=replies_draft1.json"
```

Each server line is the preset's without `-ts`, `-otd` and `GGML_CUDA_KEEPALIVE`, with `-ts $TS` and a
`CUDA_CACHE_PATH` for the compiled kernels, inside
`docker run --rm --init --network host --gpus "device=<cards>" … nvidia/cuda:12.9.2-devel-ubuntu24.04`;
`run_exactness.sh` holds the whole call. `CUDA_FORCE_PTX_JIT` is not set.

## Provenance and licence

The prompts are the acceptance run's, cut from this project's release check of 2026-11-05
([docs/data/2026-11-05_release_flashnext](../2026-11-05_release_flashnext/README.md)): the agent's sessions on 12
bug-fix tasks (Exercism exercises, MIT). The tables hold token positions, log-probabilities, counts and hashes, and
the text of the two candidate tokens at each divergence point; no other prompt or reply text. The scripts are under
this repository's licence.
