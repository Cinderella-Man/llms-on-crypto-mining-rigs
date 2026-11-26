# C1128: how far a verify pass's logits drift from one-token passes, over whole replies

What this is: registry [Q1861](../../../quants/experiments.md) (C1128), on the build machine's RTX 3090s running the
rig build through its compute 6.1 PTX with the fork's Pascal card paths forced, 2026-11-26. Correctness only, nothing
timed. On a P106, [C964](../2026-11-25_c964_flash_mtp_greedy_p106/README.md) found that Flash-Next's greedy replies
with the MTP draft leave plain decoding's text on all three of its requests, each time at a near tie between two
tokens that the draft's verify pass (four tokens a call) ranks the other way round from a one-token pass. This row
measures how far a four-token pass moves the next-token distributions from one-token passes at every position of
those replies, against a second one-token load as the floor, to tell whether those three flips are the rounding every
position shows or a few positions that drift much further.

**In plain words:** the flips are ordinary rounding that met a near tie. Reading the replies four tokens a call moves
the model's next-token odds at every one of the 768 positions, and at the typical position by more than at C964's
coding flip; a second one-token load repeats the reference to the last bit. So there is no handful of positions that
go far astray: a four-token pass is a slightly different arithmetic everywhere, and about one position in a hundred
sits close enough to a tie to change its first choice. Over a 256-token greedy reply that makes a flip nearly certain,
as C964 saw on all three requests.

## The result

The reading rule below was committed before the run. Its rule 3 applies: **rounding everywhere.**

**The 768 verify-pass positions** (the `reply` reading, rows 1–256 of the three requests; [summary.tsv](summary.tsv),
[rule.tsv](rule.tsv)), each arm against the reference load `base`:

| Arm | Same bits | Same top token | KL divergence: mean | median | 99th pct. | largest | Logit gap move: median | 90th pct. | 99th pct. | largest |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `four` (four tokens a call, the verify pass) | 0 | 760 | 0.000870 | 0.0000754 | 0.0111 | 0.0288 | 0.0839 | 0.2651 | 0.6391 | 1.7359 |
| `floor` (a second one-token load) | 768 | 768 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `one_rs3` (one token a call, `n_rs_seq` 3) | 768 | 768 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `four_rs0` (four tokens a call, `n_rs_seq` 0) | 0 | 760 | 0.000870 | 0.0000754 | 0.0111 | 0.0288 | 0.0839 | 0.2651 | 0.6391 | 1.7359 |

- **The rule:** the floor's 99th percentile is 0, below a tenth of the `four` arm's median (rule 1 holds); no position
  of the `four` arm is the same bits (rule 2 does not apply); its median logit gap move, 0.0839, is above a third of
  the smallest flip's move (0.0156), so rule 3 applies. C964's three flips moved the logit gap by amounts that rank at
  the 30.5th (code, 0.0467), 88.2nd (prose, 0.2459) and 54.6th (deep, 0.0943) percentile of the `four` arm's moves.
- **The setting `n_rs_seq` changes nothing:** `one_rs3` is the reference to the bit and `four_rs0` is `four` to the
  bit, so the whole drift comes from reading four tokens in one call.
- **The prompt's last position** (row 0, read by the same prompt call in every arm) is the same bits in all arms.
- **Where the candidates are close** ([contested.tsv](contested.tsv)): at the 106 positions whose most likely token is
  below 0.6, the probability gap between the two candidates moves by 0.0268 at the median (90th percentile 0.0682,
  99th 0.1156). C964's probability moves rank at the 43.4th (code, 0.0219), 93.4th (prose, 0.0849) and 39.6th (deep,
  0.0190) percentile there: the prose flip is a large move, the other two typical ones. Of the 30 positions whose
  two candidates are closer than 0.0849, 8 change their top token; of the 15 closer than 0.03, 6.
- **The 8 positions that change top token** (code 192; prose 131, 136, 142; deep 28, 30, 163, 201) all sit at
  near ties: the reference's two candidates 0.2274 against 0.2232 to 0.5082 against 0.4849, moved by 0.0132 to 0.1172
  in probability ([reply_rows.tsv](reply_rows.tsv), `same_top` 0). 8 of 768 is 1.04 % of positions; at that rate a
  256-token greedy reply keeps the reference's text throughout with a chance of about 7 % (est.).
- **The place in the four-token call does not matter** ([offsets.tsv](offsets.tsv)): the mean KL divergence is
  0.000851, 0.000896, 0.000844 and 0.000888 for the call's first to fourth token, and none of the 192 positions at
  any place is the same bits. The first token of a verify pass (the accepted one) drifts as much as the drafted ones.

**C964's flip positions read on these cards** ([flips.tsv](flips.tsv)): none flips here, because the 3090s' reference
ranks those candidates with other gaps than the P106 did.

| Request, row | Reference's two candidates | `four`: the same two | Probability gap move | Logit gap move (percentile) | C964's move on the P106 |
|---|---|---|---:|---:|---:|
| code, 124 | 0.4953, 0.4390 | 0.4722, 0.4689 | −0.0530 | −0.1137 (61.8) | −0.0219 |
| prose, 67 | 0.3603, 0.3253 | 0.3356, 0.3296 | −0.0290 | −0.0843 (50.3) | −0.0849 |
| deep, 21 | 0.2330, 0.1373 | 0.2314, 0.1411 | −0.0054 | −0.0344 (23.5) | −0.0190 |

At code row 124 the 3090s' one-token pass puts the other candidate first (token 78296, 0.4953) than the P106's plain
decoding did (token 4058, the reply's), and the four-token pass moves the gap by −0.0530 towards 4058, 2.4 times
C964's move and the other way round: the size of the move is the same rounding, its direction is not the P106's.

**The reference against the P106's text** ([reference.tsv](reference.tsv)): the 3090s' one-token pass puts the P106's
reply token first at 255 of 256 positions on code (not at row 124), 251 on prose (first not at row 131, where C964
found the 3090s' plain decoding leave the P106's) and 253 on deep (first not at row 0, the token C964 found the 3090s
choose differently). With the Pascal paths forced the 3090s come close to the P106's arithmetic, not to its bits.

**The whole text read four tokens a call** (the `whole` reading, `llama-perplexity`'s way; [summary.tsv](summary.tsv),
[depth.tsv](depth.tsv)): over 9,003 positions the `four` arm keeps the same top token at 8,762 (97.32 %), no position
is the same bits, the mean KL divergence is 0.00506 (median 0.00116, 99th percentile 0.0457, largest 0.5597), and the
median logit gap move 0.1233. On the deep request the drift does not grow with depth: the mean KL divergence is
0.00994 over its first 256 positions, 0.00393 at positions 2,048–4,095, 0.00637 at 6,144–8,188 and 0.00314 over the
reply. That last figure is 2.4 times the reply's 0.00131 when the prompt is read the server's way: a prompt read four
tokens a call carries its own drift into the cached state. The floor and `one_rs3` are the same bits at all 9,003
positions here too.

**One more arm, outside the rule** (`four_exact`): the v5 preset leaves `GGML_CUDA_MMVF_PASCAL_EXACT` off, the switch
that gives the BF16 products of three and four tokens the one-token sums (Q1325). With it added to `four`, the
verify-pass positions show 0 the same bits, 754 the same top token, a mean KL divergence of 0.000819 and a median logit
gap move of 0.0843; over the whole text 23 of 9,003 positions are the same bits and the mean is 0.00437. It changes
where the rounding falls, not its size: other operations of a four-token pass carry most of the drift.

**What this decides:** C964's flips are not the tail of a few positions with a bad kernel path, so Q971's search will
not find an outlier to fix: it would have to find what makes every four-token pass differ from one-token passes,
which is [Q1179](../../../quants/experiments.md)'s question (a verify pass that repeats one-token decoding bit for bit).
With the v5 preset's exact product switches on, and the BF16 one on top, the drift is still everywhere, so the
remaining operations (attention, the delta-net layers, the experts' path) are where Q1179 looks next. Until then a
check on real files compares drafting arms with each other (Q971's rule).

## Files

| File | What |
|---|---|
| [summary.tsv](summary.tsv) | per arm, reading (reply rows 1–256, reply row 0, whole) and request: positions, same bits, same top token, KL divergence, logit and probability gap moves |
| [rule.tsv](rule.tsv) | the reading rule's quantities and verdict |
| [contested.tsv](contested.tsv) | the `four` arm at the positions whose top token is below 0.6 or 0.5, and at near ties; C964's moves ranked there |
| [flips.tsv](flips.tsv) | C964's three flip positions read here, per arm |
| [offsets.tsv](offsets.tsv) | the four-token arms by place in their call |
| [depth.tsv](depth.tsv) | the four-token arms' whole reading of the deep request by depth |
| [reference.tsv](reference.tsv) | the reference load's top token against the P106's plain reply |
| [reply_rows.tsv](reply_rows.tsv) | every reply row of every arm: the reference's two candidates and the per-position measures |
| [run/verify_drift.cpp](run/verify_drift.cpp) | the tool: one load reads the requests at one call size and writes or compares every position's logits |
| [run/build.sh](run/build.sh) | builds it (and `run/sysinfo.cpp`) against a rig build's own build tree |
| [run/sysinfo.cpp](run/sysinfo.cpp) | the backend's feature line under an environment (C981's), the check that the forced paths are on |
| [run/make_inputs.py](run/make_inputs.py) | C964's prompts and plain replies as the tool's inputs |
| [run/run.sh](run/run.sh) | the driver: one load per arm, the forced environment, the preset's arguments |
| [run/summarize.py](run/summarize.py) | the tables above from the driver's output |

## Method

- **Text:** C964's three requests (its coding request, 27 prompt tokens; its prose paragraph, 19; the deep prompt,
  8,189), each followed by plain decoding's 256-token reply on the P106 (C964's `tokens.tsv`, arm `plain`), as token
  ids. `llama-perplexity` cannot read this text: it tokenizes a text file without parsing special tokens (the replies
  hold `<think>`-like special tokens), and it scores only the second half of each window, which would leave out the
  first ~120 reply tokens of the two short requests. So a small tool, [run/verify_drift.cpp](run/verify_drift.cpp),
  does what the row asks of `llama-perplexity` with the exact token ids: one load per arm reads every token after the
  prompt *S* tokens a call, keeps every position's logits, and writes them (the reference load) or compares them
  with the reference's, position by position.
- **Two readings per request**, each on a cleared memory:
  - `reply`: the prompt read as llama-server reads a prompt call (calls of 4,096 tokens, the last one cut at
    *N* − 4 − 512 and *N* − 4 with in-call checkpoints, the server's hints), then the reply *S* tokens a call. Row 0 is
    the prompt's last position; row *i* is reply token *i* − 1's position and predicts reply token *i*. This is how the
    two servers read C964's replies: the prompt alike, then one token a call (plain decoding) or four (the draft's
    verify pass).
  - `whole`: the prompt and the reply from the first token, *S* tokens a call: `llama-perplexity` at `-ub S`, the
    deep prompt's 8,189 positions included.
- **Arms (one load each):**

  | Arm | Tokens a call | `n_rs_seq` | Role |
  |---|---:|---:|---|
  | `base` | 1 | 0 | the reference: plain decoding's server (`--kl-divergence-base` at `-ub 1`) |
  | `four` | 4 | 3 | the drafting server's verify pass (`--kl-divergence` at `-ub 4`) |
  | `floor` | 1 | 0 | `base` again in a fresh load: the load-to-load floor |
  | `one_rs3` | 1 | 3 | the snapshot setting alone |
  | `four_rs0` | 4 | 0 | the call size alone |
| `four_exact` | 4 | 3 | `four` with `GGML_CUDA_MMVF_PASCAL_EXACT=1` added (run after the others, outside the rule) |

  `n_rs_seq` is the number of recurrent-state snapshots a context keeps for rolling back drafted tokens: llama-server
  sets 3 with the preset's MTP draft of three tokens and 0 without a draft.
- **Per position** (against `base`): the KL divergence of the arm's distribution from the reference's (in double, over
  the whole 248,320-token vocabulary), whether the top token agrees, whether the logit rows are the same bits, and how
  far the gap between the reference's two most likely tokens *a* and *b* moved: in probability,
  (p(a) − p(b)) − (p_ref(a) − p_ref(b)), C964's measure, and in logits, (x(a) − x(b)) − (x_ref(a) − x_ref(b)), which
  does not depend on how likely the two tokens are.
- **Line:** [release/flash-next-v5/settings.env](../../../release/flash-next-v5/settings.env)'s `PRESET_ENV` with
  [scripts/pascal_forced.sh](../../../scripts/pascal_forced.sh)'s `pascal_force_preset` (every compute-6.1-only path
  at `2`, a P106's launch shapes, `LLAMA_NO_MOE_DOWN_RESCALE=tg`), and the preset's arguments that act on a load or
  on the arithmetic: `-ngl 99 -fa on -c 65536 -b 4096 -ub 512 -fit off -lm dio --no-host`, the taper
  `short=1400:64`, the layer split `-ts 1,1` on two RTX 3090s. Engine: Miner Llama `pascal-p106-v5` at `2d839da1d`
  (the rig build `llama.cpp-20261125-2d839da1d-pascal-p106-v5-land-batch3`; the tool is linked against that build's
  own tree, whose `llama-perplexity` has the tarball's md5). Target: Swift GSQ-RCO Q2_0, the preset's file.

## The reading rule (fixed before the run)

C964's three flips moved the probability gap between the two candidates by 0.0219 (code), 0.0849 (prose) and 0.0190
(deep), and the logit gap by 0.0467, 0.2459 and 0.0943 (from the probabilities in C964's table). The rule reads the
`four` arm's |logit gap move| over the 768 verify-pass positions of the `reply` reading (rows 1–256 of the three
requests, every output of a four-token call):

1. **The floor first:** the `floor` arm's 99th percentile must be below a tenth of the `four` arm's median, or the
   drift is not resolved from load-to-load noise and nothing below is concluded.
2. **Exact:** if the `four` arm's rows are the same bits as the reference's at 99 % of the positions or more, the
   four-token pass is exact on these card paths, and C964's flips come from something this run does not reproduce.
3. **Rounding everywhere:** if the `four` arm's median is at least a third of the smallest flip's move (0.0467 / 3 =
   0.0156), a typical position moves by the size of the flips, and the flips are that rounding meeting a near tie.
4. **A tail:** if the median is below a tenth of it (0.0047) and all three flips' moves lie above the `four` arm's
   99th percentile, the flips are a few positions that drift much further than the rest (Q971's starting point).
5. **Otherwise** the result is between the two, reported with each flip's percentile rank.

The KL divergence, the same-top-token rate, the positions where C964 flipped (code row 124, prose 67, deep 21, read on
these cards) and the other arms are reported, not judged.

## How it ran

- **Machine:** two RTX 3090s of the build machine (CUDA 12.9 image, the rig build's compute 6.1 PTX compiled by the
  driver), nothing timed. `run/sysinfo.cpp` under the run's environment lists `MMVQ_PASCAL = 1` and
  `MMVF_PASCAL = 1` (the forced Pascal paths on), a CUDA preflight on both cards passed before each driver run.
- **The tool's build:** linked against the rig build's tree (its `llama-perplexity` md5 `8ebeab4e6ab839cfff0454625fb6db0b`,
  the tarball's); `verify-drift` md5 `2ce0fd215bd229f3f81fe8a234633660`.
- **Inputs:** the prompts tokenize to 27, 19 and 8,189 tokens, C964's counts; the deep prompt's md5 is C964's
  (`dea4956d2c52140e5ebfedf0dd46c770`); the replies are C964's `plain` token ids (256 each).
- **Loads:** `base` 3 min 1 s, `four` 2 min 4 s, `floor` 3 min 14 s, `one_rs3` 3 min 14 s, `four_rs0` 2 min 3 s (one
  driver run, every load rc 0), then `four_exact` 2 min 56 s (a second driver run with `ARMS=four_exact`, against the
  same reference file). Each load wrote or compared 9,774 rows: 771 of the `reply` reading (257 a request) and 9,003 of
  the `whole` reading.

```bash
# on the laptop: the inputs (the deep prompt from docs/switches.md at commit 6a9cf9d9)
git show 6a9cf9d9:docs/switches.md > switches-6a9cf9d9.md
python3 -I docs/data/2026-11-26_c1128_verify_drift/run/make_inputs.py docs/data/2026-11-25_c964_flash_mtp_greedy_p106 \
    switches-6a9cf9d9.md inputs
# copy inputs/, docs/data/2026-11-26_c1128_verify_drift/run/, release/flash-next-v5/settings.env and
# scripts/pascal_forced.sh (as repo/...) into $WORK-run on the build machine, then there:
TREE=<the rig build's work folder> OUT=$WORK-run/tool \
    DOCKER_ARGS='--label pascal-quants=1 --cpu-shares=256' LOCK=<build lock> bash run/build.sh
I=$WORK-run/inputs
WORK=$WORK BASE=<bulk disk>/base.rows TOOL=$WORK-run/tool/verify-drift SYSINFO=$WORK-run/tool/sysinfo \
  MODEL=$MODELS/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf REPO=$WORK-run/repo \
  REQS="code=$I/code.prompt.txt=$I/code.ids;prose=$I/prose.prompt.txt=$I/prose.ids;deep=$I/deep.prompt.txt=$I/deep.ids" \
  GPUS=0,1 MOUNTS="<folders holding the model, inputs and tool>" PREFLIGHT=<CUDA preflight script> \
  DOCKER_ARGS='--label pascal-quants=1 --cpu-shares=256' bash run/run.sh
# the extra arm against the same reference: the same line with ARMS=four_exact
# on the laptop, with $WORK copied back (each arm's drift.rows.tsv):
python3 -I docs/data/2026-11-26_c1128_verify_drift/run/summarize.py <copy of WORK> docs/data/2026-11-26_c1128_verify_drift
```

The reference file holds every row's 248,320 logits (9.7 GB); the per-row tables of every arm are summarized here, and
[reply_rows.tsv](reply_rows.tsv) keeps every row of the `reply` reading.

## Provenance and licence

Code: Miner Llama (MIT, like llama.cpp); the tool, scripts and tables here under the same licence. Model file:
UkisAI's Swift-1.5 GSQ-RCO Q2_0 of Qwen3.8-Flash-Next, the [v5 preset](../../../release/flash-next-v5/settings.env)'s
file. The text is C964's: prompts written for this project, a slice of this repository's `docs/switches.md`, and the
model's own replies as token ids. The tables hold token ids, probabilities and divergences; no reply text.
