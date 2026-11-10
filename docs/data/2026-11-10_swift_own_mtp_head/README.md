# Swift's own MTP head against the draft in use: the file and tokens a step (Q1168; 2026-11-10)

What this is: the Miner Llama v3 release preset
([release/flash-next-v3/settings.env](../../../release/flash-next-v3/settings.env)) runs UkisAI's Swift-1.5 fine-tune
of Qwen3.8-Flash-Next (Swift GSQ-RCO Q2_0) and drafts three tokens a step with the MTP draft of agentionai's Gyro
repository, a head made from the base model. Swift-1.5's own BF16 checkpoint also holds an MTP layer. This
experiment

1. cuts that layer out of the checkpoint into a draft file with the tensor names, types and metadata of the draft in
   use, and compares the two files tensor by tensor;
2. counts how many tokens one step of the main model yields with each file as the draft, on the release session's 66
   recorded requests, with the preset's sampler (`pv0`) and with probabilistic verification
   (`--spec-draft-sampling probabilistic --spec-draft-p-min 0`, `pv1`).

Registry row Q1168 in [quants/experiments.md](../../../quants/experiments.md). The method of the second part, its
scripts and the four loads of the draft in use that each pair is made with are those of
[2026-11-10_prob_verify_acceptance](../2026-11-10_prob_verify_acceptance/README.md) (Q1173), run on the same cards
and build earlier the same night.

Why: a draft is right more often the closer it follows the model it drafts for. If Swift's checkpoint held a head
trained along with Swift, it would be a free gain in accepted tokens. Acceptance depends on the two models, the
sampler and the text, not on the card, so it was counted on a machine with newer cards. No time is read or
reported here.

## Result

**Swift-1.5 has no head of its own: the MTP layer in its checkpoint is the base model's, and as a draft it is no
gain. With the preset's sampler it yields 3.086 and 3.073 tokens a step in its two loads, against 3.043, 3.072 and
3.090 in three loads of the draft in use (+0.011, the three 0.048 apart): negligible. With probabilistic
verification it yields 3.238 and 3.279 against 3.283 and 3.304 (−0.035; its own loads 0.041 apart): no gain, and
lower in 40 of the 66 requests. The preset keeps the Gyro draft.**

- **The file.** 25 of the head's 34 tensors are the base model's bit for bit: 24 have the same bytes as in the
  draft in use (the token embeddings, the router, every norm, the three sets of hyper-connection weights, the
  indexer, the input projection, the shared expert's down projection), and Swift's output head rounds to the same
  Q8_0 bytes as a base-model file's.
  The other nine (the block's attention matrices and its experts) differ from the draft in use only as two
  roundings of the same values differ. What differs between the two draft files is how those nine and the output
  head were rounded to 4 and 6 bits.
- **The new file loads and drafts** on the release build with `--spec-type draft-mtp`, in the same memory as the
  draft in use.

### What the checkpoint holds

Swift-1.5's BF16 checkpoint (131 safetensors shards, 1,658 tensors) holds 31 tensors under `mtp.`
([checkpoint_tensors.tsv](checkpoint_tensors.tsv)): one full-attention block with 512 routed experts and a shared
one, its two hyper-connections, the mixer that feeds the output head, the two input projections and their norms.
Its `config.json` says `mtp_num_hidden_layers: 1`. With the token embeddings and the output head these are the 33
tensors an MTP-only file is made of.

The draft in use (`mtp-Qwen3.8-Flash-Next-draft.gguf`, 2,697,439,648 bytes, 34 tensors, 55 key/value pairs, file
type Q4_K_M) holds the same layer of the base model as block 48: the token embeddings in Q8_0, the output head in
Q6_K, the routed experts in Q4_K (gate, up) and IQ4_NL (down), the attention in Q4_K and Q6_K, the hyper-connections,
the indexer and the input projection in Q8_0, the norms and the router in F32.

### How the new file was made

No PyTorch was at hand, so `convert_hf_to_gguf.py --mtp` could not run. What it does for this architecture
(llama.cpp's `conversion/qwen4exp.py` and `conversion/qwen.py`) is written out in
[run/mtp_head_from_safetensors.py](run/mtp_head_from_safetensors.py), which needs numpy and gguf-py only
([tensors.tsv](tensors.tsv) lists each tensor's source):

- `mtp.layers.0.*` becomes block 48; `mtp.pre_fc_norm_embedding` and `mtp.pre_fc_norm_hidden` its `nextn.enorm` and
  `nextn.hnorm`; `mtp.hyper_connection_mixer` its `nextn.hc_head_*`;
- `mtp.fc_embedding` and `mtp.fc_hidden` are joined along the input into `nextn.eh_proj`;
- the experts' `gate_up_proj` is cut in half into `ffn_gate_exps` and `ffn_up_exps`, the indexer's `index_qk_proj`
  into `indexer.q_proj` (the first 512 rows) and `indexer.k_proj`;
- every `*norm.weight` is stored plus one;
- the norms, the router and the shared expert's gate are written in F32, every other tensor in BF16, the
  checkpoint's own bytes; names, order, shapes and every key/value pair come from the draft in use
  (`general.name` aside), whose tokenizer arrays equal the target's.

`llama-quantize` then gives each tensor the type it has in the draft in use ([types.txt](types.txt), on a Q4_K_M
base), and [scripts/gguf_retype.py](../../../scripts/gguf_retype.py) converts the indexer's two projections to
Q8_0, which `llama-quantize` leaves in the source type. The result has the draft in use's 34 tensors, types and
size, 2,697,439,648 bytes. The three steps took about two minutes on the CPU
([run/make_head.sh](run/make_head.sh)).

### The two files, tensor by tensor

[compare_heads.tsv](compare_heads.tsv), from [run/compare_heads.py](run/compare_heads.py): every tensor decoded to
float32 and compared as one vector. "From the checkpoint" is |file − checkpoint| / |file|, the distance of a file's
decoded values from Swift's BF16 values; it is what the rounding costs when the file was rounded from those values.

| Tensors | Type | Values | Same bytes in both files | From the checkpoint: the draft in use | From the checkpoint: the new file |
|---|---|---:|---|---:|---:|
| token embeddings | Q8_0 | 635,699,200 | yes | 0.00547 | 0.00547 |
| input projection `nextn.eh_proj` | Q8_0 | 13,107,200 | yes | 0.01062 | 0.01062 |
| hyper-connections: 6 down and up, 2 inject | Q8_0 | 19,742,720 | yes, all 8 | 0.00643 to 0.00863 | the same |
| indexer `q_proj`, `k_proj`; shared expert's down | Q8_0 | 3,276,800 | yes, all 3 | 0.00651 to 0.00825 | the same |
| router, shared expert's gate, 9 norms | F32 | 1,357,568 | yes, all 11 | 0 | 0 |
| output head | Q6_K | 635,699,200 | no | 0.01923 | 0.01834 |
| attention `attn_v` | Q6_K | 1,310,720 | no | 0.02610 | 0.02450 |
| attention `attn_q` | Q4_K | 31,457,280 | no | 0.08385 | 0.08358 |
| attention `attn_k` | Q4_K | 1,310,720 | no | 0.09517 | 0.09479 |
| attention `attn_output` | Q4_K | 15,728,640 | no | 0.07825 | 0.07802 |
| routed experts, gate | Q4_K | 838,860,800 | no | 0.07845 | 0.07352 |
| routed experts, up | Q4_K | 838,860,800 | no | 0.07810 | 0.07315 |
| routed experts, down | IQ4_NL | 838,860,800 | no | 0.08241 | 0.07774 |
| shared expert, gate | Q4_K | 1,638,400 | no | 0.08963 | 0.08917 |
| shared expert, up | Q4_K | 1,638,400 | no | 0.08553 | 0.08520 |

- **24 tensors have the same bytes in both files**, 673,183,488 values. Q8_0 is rounded without a search (the
  largest magnitude of 32 values over 127 as the step, each value to the nearest step) and F32 is not rounded at
  all, so equal bytes mean equal values in the two checkpoints: these parts of Swift's checkpoint are the base
  model's.
- **The output head is the base model's too.** [output_head_check.tsv](output_head_check.tsv): Swift's output head
  and token embeddings, rounded to Q8_0, equal the Q8_0 copies in a base-model file (agentionai's Gyro-L) in all
  19,865,600 blocks each. The two draft files still hold the output head in different bytes: both are Q6_K
  of the same values, rounded by two procedures. That gives the measure for the rest: for values known to be the
  same, the draft in use sits 0.01923 from them and the new file 0.01834, and 50.8 % of their decoded values are
  equal.
- **The other nine tensors show the same picture and nothing more.** The draft in use sits as far from Swift's
  values as the new file's own rounding does, to within 0.3 to 6.8 % (4.9 % for the output head). A layer trained
  further would add its difference on top. The comparison cannot exclude a small one: a difference of up to about
  1 to 3 % of a tensor's norm would hide in the gap between the two roundings, and these nine hold two thirds of
  the head's values. Comparing the checkpoint shards' hashes with the base repository's would settle it (registry
  Q1182).
- The draft in use was rounded by another procedure than plain `llama-quantize`: its Q4_K blocks differ from the
  new file's in nearly every block, and it sits a little further from the values in plain distance, which is what
  an importance-weighted rounding looks like. Its key/value pairs name no importance matrix, so how it was rounded
  is not known.

### Every load

Tokens a step is generated tokens over steps; a step is one pass of the main model that yielded tokens (generated
minus accepted draft tokens). Pooled means the counters are summed over the 66 requests before dividing.

| Arm | Draft | Sampler | Load | Generated | Accepted of drafted | Share accepted | Steps | Tokens a step (pooled) | Tokens a step (mean of the 66 requests) |
|---|---|---|---|---:|---|---:|---:|---:|---:|
| `pv0` (Q1173's run) | in use | preset | 1 | 17,410 | 11,688 of 16,958 | 0.6892 | 5,722 | 3.043 | 3.127 |
| `pv0` (Q1173's run) | in use | preset | 2 | 17,359 | 11,708 of 16,743 | 0.6993 | 5,651 | 3.072 | 3.143 |
| `gyro-pv0` (this run's control) | in use | preset | 1 | 17,346 | 11,733 of 16,629 | 0.7056 | 5,613 | 3.090 | 3.147 |
| `own-pv0` | Swift's | preset | 1 | 17,174 | 11,608 of 16,488 | 0.7040 | 5,566 | 3.086 | 3.140 |
| `own-pv0` | Swift's | preset | 2 | 17,439 | 11,764 of 16,812 | 0.6997 | 5,675 | 3.073 | 3.131 |
| `pv1` (Q1173's run) | in use | probabilistic | 1 | 18,788 | 13,066 of 17,050 | 0.7663 | 5,722 | 3.283 | 3.344 |
| `pv1` (Q1173's run) | in use | probabilistic | 2 | 18,784 | 13,099 of 16,944 | 0.7731 | 5,685 | 3.304 | 3.369 |
| `own-pv1` | Swift's | probabilistic | 1 | 15,085 | 10,426 of 13,869 | 0.7517 | 4,659 | 3.238 | 3.293 |
| `own-pv1` | Swift's | probabilistic | 2 | 16,254 | 11,297 of 14,744 | 0.7662 | 4,957 | 3.279 | 3.283 |
| the recorded session (the rig, live) | in use | preset | — | 27,067 | 18,447 of 25,704 | 0.7177 | 8,620 | 3.140 | 3.191 |

A replayed reply ends at the model's own stop or at the recorded reply's length, whichever comes first, so a load
generates fewer tokens than the recorded session did. The `own-pv1` loads generated fewer tokens than the `pv1`
loads mostly because of two long replies: one request with a limit of 8,747 tokens stopped after 1,039 tokens in
both `own-pv1` loads and after 2,836 in both `pv1` loads (1,148 or 1,603 in the five loads with the preset's
sampler), and one with a limit of 3,233 stopped after 1,815 in both `own-pv1` loads and ran to the limit in every
other load. [loads.tsv](loads.tsv); every request in every load in [requests.tsv](requests.tsv).

### The control

This run's one load of the draft in use with the preset, `gyro-pv0`, gave 3.090 tokens a step: 0.018 above the
higher of Q1173's two loads of the same line (3.043 and 3.072) and +0.033 above their mean (95 % interval over
requests −0.007 to +0.073). It wrote the same reply as those two loads in 43 and 34 of the 66 requests, as they do
with each other (37). So it is a third load of the preset, and the three are 0.048 apart, not 0.029: two loads
understate how far loads of one line differ. The classes below are read against the three loads.

### Swift's head against the draft in use

An arm's value is the mean of its loads. The spread is the difference between an arm's highest and lowest load.
Class: gain when the arm is above the other by more than the spread, loss when below by more than it, negligible
otherwise. The intervals are 95 % bootstrap intervals of the difference: over requests (each request keeps its rows
in all loads, so the comparison stays paired) and over whole sessions (the 12 tasks). [compare.tsv](compare.tsv).

| Swift's head | Against | Metric | Its loads | The other's loads | Difference | % | Its spread | The other's spread | Class | 95 % (requests) | 95 % (sessions) |
|---|---|---|---|---|---:|---:|---:|---:|---|---|---|
| `own-pv0` | the preset's three loads | tokens a step | 3.086, 3.073 | 3.043, 3.072, 3.090 | +0.011 | +0.36 | 0.013 | 0.048 | **negligible** | −0.016 to +0.040 | −0.020 to +0.046 |
| `own-pv0` | the preset's three loads | share accepted | 0.7040, 0.6997 | 0.6892, 0.6993, 0.7056 | +0.0039 | +0.55 | 0.0043 | 0.0163 | **negligible** | −0.0054 to +0.0137 | −0.0068 to +0.0160 |
| `own-pv0` | Q1173's two `pv0` loads | tokens a step | 3.086, 3.073 | 3.043, 3.072 | +0.022 | +0.72 | 0.013 | 0.029 | **negligible** | −0.008 to +0.053 | −0.011 to +0.066 |
| `own-pv0` | this run's control | tokens a step | 3.086, 3.073 | 3.090 | −0.011 | −0.36 | 0.013 | one load | — | −0.049 to +0.030 | −0.045 to +0.018 |
| `own-pv1` | Q1173's two `pv1` loads | tokens a step | 3.238, 3.279 | 3.283, 3.304 | −0.035 | −1.07 | 0.041 | 0.021 | **negligible** by its own spread and by the preset's 0.048; a loss by `pv1`'s 0.021 | −0.101 to +0.048 | −0.090 to +0.051 |
| `own-pv1` | Q1173's two `pv1` loads | share accepted | 0.7517, 0.7662 | 0.7663, 0.7731 | −0.0107 | −1.39 | 0.0145 | 0.0067 | the same | −0.0321 to +0.0167 | −0.0292 to +0.0175 |

Per request (each request's tokens a step averaged over its arm's loads, then the mean difference over the 66
requests):

- `own-pv0` against the preset's three loads: −0.003 (95 % −0.039 to +0.030), higher in 30 requests, lower in 25,
  equal in 11;
- `own-pv1` against `pv1`: −0.068 (95 % −0.125 to −0.014), higher in 23, lower in 40, equal in 3. Both `own-pv1`
  loads are below both `pv1` loads.

So with the preset's sampler the two files draft alike, and with probabilistic verification Swift's file is not
better and leans lower. That fits the files: with the preset's sampler a draft is accepted when its first choice
is the token the main model samples, which a coarser rounding of the same weights rarely changes; probabilistic
verification accepts by the ratio of the two models' probabilities, which follows the draft's whole distribution.

Probabilistic verification itself gains on Swift's file as it does on the draft in use: `own-pv1` against
`own-pv0` +0.179 tokens a step (+5.82 %; 95 % +0.120 to +0.229), against +0.237 (+7.74 %) in Q1173.

### By position in the draft

Verify steps of three drafted tokens by how many the main model kept, and the share of those drafts whose first,
second and third token was accepted (a token counts only when every token before it was)
([by_position.tsv](by_position.tsv), [steps.tsv](steps.tsv)):

| Load | Steps | None | One | Two | All three | First accepted | Second | Third |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `pv0` 1 | 5,643 | 1,029 | 766 | 650 | 3,198 | 0.8177 | 0.6819 | 0.5667 |
| `pv0` 2 | 5,572 | 963 | 749 | 648 | 3,212 | 0.8272 | 0.6927 | 0.5765 |
| `gyro-pv0` 1 | 5,534 | 943 | 742 | 581 | 3,268 | 0.8296 | 0.6955 | 0.5905 |
| `own-pv0` 1 | 5,487 | 938 | 740 | 583 | 3,226 | 0.8291 | 0.6942 | 0.5879 |
| `own-pv0` 2 | 5,595 | 975 | 757 | 604 | 3,259 | 0.8257 | 0.6904 | 0.5825 |
| `pv1` 1 | 5,677 | 675 | 684 | 590 | 3,728 | 0.8811 | 0.7606 | 0.6567 |
| `pv1` 2 | 5,642 | 656 | 657 | 562 | 3,767 | 0.8837 | 0.7673 | 0.6677 |
| `own-pv1` 1 | 4,615 | 602 | 553 | 527 | 2,933 | 0.8696 | 0.7497 | 0.6355 |
| `own-pv1` 2 | 4,906 | 599 | 545 | 555 | 3,207 | 0.8779 | 0.7668 | 0.6537 |

No position sets the two files apart with the preset's sampler. With probabilistic verification Swift's file is
0.003 to 0.014 lower at the first position and 0.003 to 0.032 lower at the third, load against load.

### The server logs

Every load ran with `LLAMA_TRACE=1`, which prints one line per verify step and changes no computation
([log_check.tsv](log_check.tsv), with Q1173's four loads beside this run's five).

| Load | Requests with a grammar | Their sampled tokens, drawn on the device | Rejected by the grammar after a device draw | Verify steps | Steps that restored a checkpoint | Checkpoint restores between requests | Error lines |
|---|---:|---|---:|---:|---:|---:|---:|
| `gyro-pv0` 1 | 66 | 17,355 of 17,355 | 1 | 5,552 | 0 | 65 | 0 |
| `own-pv0` 1 | 66 | 17,186 of 17,186 | 1 | 5,506 | 0 | 65 | 0 |
| `own-pv0` 2 | 66 | 17,451 of 17,451 | 1 | 5,614 | 0 | 65 | 0 |
| `own-pv1` 1 | 66 | 15,134 of 15,134 | 16 | 4,632 | 0 | 64 | 0 |
| `own-pv1` 2 | 66 | 16,295 of 16,295 | 12 | 4,924 | 0 | 63 | 0 |

- The server loaded the new file as the draft in every `own` load, built the same shortlist for its output head as
  for the draft in use (32,801 ids), and the three cards held the same memory after loading as with the draft in
  use (13,938, 13,162 and 15,042 MiB).
- The accepted and drafted tokens add up to the same numbers three ways in every load: over the step lines, over
  the server's per-request lines and over the `timings` the replay keeps.
- Two loads of Swift's file wrote the same reply in 32 (`own-pv0`) and 31 (`own-pv1`) of 66 requests; an `own-pv0`
  load and the control in 31 and 31 ([same_reply.tsv](same_reply.tsv)). Two loads of one line differ as much (37
  for `pv0`, 35 for `pv1`), so the replies do not tell the two files apart.

**What it decides.**
- **The preset and its next candidate keep the Gyro draft.** Swift's checkpoint offers no head fitted to Swift, and
  the same head rounded by plain `llama-quantize` is no better with the preset's sampler and leans lower with
  probabilistic verification.
- **A head fitted to Swift exists only by training one** (registry Q572, parked: acceptance is at the published
  level).
- **A third load of the preset moves its spread from 0.029 to 0.048 tokens a step.** Replay comparisons on this
  line read differences under about 0.05 as within the loads' spread.

**Limits.**
- **Counts, not time**, on newer cards; the new file was not run on a rig.
- **One session's requests:** 66 requests of 12 bug-fix tasks. Two loads an arm, one control load.
- **Nine of the 34 tensors are compared through their rounding only** (see above).
- **The text differs between loads**, so "paired by request" pairs the prompt, the seed and the reply limit, not
  the reply. Two long replies carry most of the difference in generated tokens between `own-pv1` and `pv1`.
- One target file (Swift GSQ-RCO Q2_0), one draft length, the sm_61 build's compute 6.1 PTX on newer cards.

## Method

- **The line, the requests, the counters:** as in
  [2026-11-10_prob_verify_acceptance](../2026-11-10_prob_verify_acceptance/README.md): the release preset without
  its nine-card split (`-ts`), the draft's device override (`-otd`) and the keep-alive variable, on three RTX 3090
  cards; build `pascal-p106-v3-b12076-ladder5` (`1551f5fe1`), `CUDA_FORCE_PTX_JIT` not set; the 66 recorded
  requests of the release check's session `t1swift` re-sent by
  [bench/agent_check/replay.py](../../../bench/agent_check/replay.py) in their recorded order with the same seed
  per request and the recorded reply length as the limit; `draft_n`, `draft_n_accepted` and `predicted_n` of the
  server's `timings`.
- **The arms:** `<draft>-<sampler>`. Draft `gyro` is the draft in use, `own` the new file; sampler `pv0` is the
  preset's, `pv1` adds `--spec-draft-sampling probabilistic --spec-draft-p-min 0`. Five loads in the order
  `gyro-pv0 own-pv0 own-pv1 own-pv0 own-pv1`, one llama-server process per load
  ([run/run_own_head.sh](run/run_own_head.sh)).
- **The pairs:** `own-pv0` against Q1173's `pv0` loads and this run's control, `own-pv1` against Q1173's `pv1`
  loads, each request paired with itself. [run/tables.sh](run/tables.sh) links the loads of each comparison into
  one folder and runs Q1173's `pv_table.py` on it (10,000 bootstrap resamples, seed 1).
- **The file comparison:** gguf-py's decoders; the tensors over 2^27 values in slices.

## Files

| File | Holds |
|---|---|
| [checkpoint_tensors.tsv](checkpoint_tensors.tsv) | the 33 tensors read from the checkpoint: name, type, shape, shard, offset, bytes |
| [tensors.tsv](tensors.tsv) | the new file's 34 tensors: type before `llama-quantize`, shape, how each is made and from which checkpoint tensor |
| [types.txt](types.txt) | `llama-quantize`'s type per tensor, read from the draft in use |
| [compare_heads.tsv](compare_heads.tsv) | per tensor: the draft in use against the new file, each against the checkpoint's values (cosine, relative distance, largest difference), equal decoded values, equal bytes, equal blocks |
| [output_head_check.tsv](output_head_check.tsv) | Swift's output head and token embeddings rounded to Q8_0 against a base-model file's Q8_0 copies: equal blocks and values |
| [loads.tsv](loads.tsv) | per load (this run's five and Q1173's four): generated, drafted, accepted, share accepted, steps, tokens a step; the recorded session's row |
| [compare.tsv](compare.tsv) | each comparison (`all`: against Q1173's two `pv0` loads; `pre3`: against the preset's three loads; `night`: against this run's control; `pv1`; `own`: `own-pv1` against `own-pv0`): every load by value, difference, spreads, class, intervals, per-request counts; an arm's second load against its first |
| [requests.tsv](requests.tsv) | per request and load: prompt tokens, reply limit, generated, drafted, accepted, steps, tokens a step, tokens drawn on the device, grammar counters, finish reason, the first 12 characters of the reply's MD5 |
| [steps.tsv](steps.tsv), [by_position.tsv](by_position.tsv) | per load: verify steps by drafted and accepted tokens; the share accepted at each position of a three-token draft |
| [same_reply.tsv](same_reply.tsv) | per pair of loads: requests with the same reply and with the same counters |
| [log_check.tsv](log_check.tsv) | per load, from the server log: sampling on the device, grammar rejections, steps, restores, error lines, and the three sums of accepted and drafted tokens |
| [run/mtp_head_from_safetensors.py](run/mtp_head_from_safetensors.py) | the MTP head of a Qwen3.8-Flash-Next checkpoint as a BF16 GGUF with a reference draft's names and metadata (numpy, gguf-py) |
| [run/make_head.sh](run/make_head.sh) | extract, quantize, retype, compare: the new file from the checkpoint |
| [run/compare_heads.py](run/compare_heads.py), [run/output_head_check.py](run/output_head_check.py) | the two file comparisons |
| [run/st_headers.py](run/st_headers.py) | lists a sharded safetensors checkpoint's tensors from its headers (standard library) |
| [run/run_own_head.sh](run/run_own_head.sh) | the five loads: one llama-server container per load, the replay, a check that each load is whole |
| [run/tables.sh](run/tables.sh) | the comparisons and tables from the loads' rows |

The tasks are the bug-fix set of [bench/agent_check](../../../bench/agent_check/README.md) (`tasks/bugfix`); the
tables name them by task id and hold no prompt or reply text.

## Commands

On the machine with the cards. `REPO` is a checkout of this repository, `WORK` a work folder, `MODEL_DIR` the
folder of Swift-1.5's BF16 safetensors, `LIKE` the draft in use, `TARGET` the first shard of Swift Q2_0 (the other
two beside it), `GGUF_PY` the `gguf-py` folder of the build's llama.cpp tree, `QUANTIZE` a `llama-quantize` of the
same line, `PY_IMAGE` a container image with python3 and numpy, `MOUNTS` the folders the containers read.

The checkpoint's tensors, then the new file (CPU only):

```bash
python3 -I "$REPO/docs/data/2026-11-10_swift_own_mtp_head/run/st_headers.py" "$MODEL_DIR" \
  '^mtp\.|embed_tokens|^lm_head|language_model\.norm' > checkpoint_tensors.tsv
MODEL_DIR=$MODEL_DIR LIKE=$LIKE TARGET=$TARGET REPO=$REPO GGUF_PY=$GGUF_PY QUANTIZE=$QUANTIZE WORK=$WORK/head \
  NAME=mtp-Swift-1.5-Qwen3.8-Flash-Next-draft TITLE="Swift-1.5 Qwen3.8-Flash-Next MTP head" PY_IMAGE=$PY_IMAGE \
  MOUNTS="$MOUNTS" DOCKER_ARGS="--label pascal-quants=1 --cpu-shares=256" \
  setsid -f nohup nice -n 19 bash "$REPO/docs/data/2026-11-10_swift_own_mtp_head/run/make_head.sh" \
  > "$WORK/head/make_head.console.txt" 2>&1 < /dev/null
```

It is done when `$WORK/head/head.done` exists; the file is usable from `head.ready` on. The output head's check
reads a base-model file that holds `output.weight` and `token_embd.weight` in Q8_0 (`BASE`, here Gyro-L, read with
the `gguf-py` of a tree that knows its expert types, `GGUF_PY_BASE`), in the same image:

```bash
PYTHONPATH=$GGUF_PY_BASE nice -n 19 python3 "$REPO/docs/data/2026-11-10_swift_own_mtp_head/run/output_head_check.py" \
  "$WORK/head/mtp-Swift-1.5-Qwen3.8-Flash-Next-draft-BF16.gguf" "$BASE" --out "$WORK/head/output_head_check.tsv"
```

The five loads (`BIN` the build's `llama-server`, `LOG` the recorded `proxy.jsonl` with its `proxy.jsonl.bodies.gz`
beside it, `GPUS` the three cards as Docker names them):

```bash
cd "$WORK" && WORK=$WORK REPO=$REPO BIN=$BIN MODEL=$TARGET MODEL_DRAFT=$LIKE \
  DRAFT_own=$WORK/head/mtp-Swift-1.5-Qwen3.8-Flash-Next-draft.gguf LOG=$LOG GPUS=$GPUS MOUNTS="$MOUNTS" \
  DOCKER_ARGS="--label pascal-quants=1 --cpu-shares=256" NAME=own-head PORT=18140 \
  setsid -f nohup nice -n 19 bash "$REPO/docs/data/2026-11-10_swift_own_mtp_head/run/run_own_head.sh" \
  > console.txt 2>&1 < /dev/null
```

It is done when `$WORK/done` exists (`$WORK/failed` otherwise); the five loads ended 33 minutes after the start.
Then, with `PRIOR` a folder that holds Q1173's `replay-pv0-{1,2}.jsonl`, `replay-pv1-{1,2}.jsonl` and their server
logs:

```bash
WORK=$WORK PRIOR=$PRIOR REPO=$REPO bash "$REPO/docs/data/2026-11-10_swift_own_mtp_head/run/tables.sh"
```

## Provenance

- The checkpoint is UkisAI's Swift-1.5 fine-tune of Qwen3.8-Flash-Next in BF16 safetensors (Swift Open License 1.0
  and Qwen Community License 1.0); the draft in use is `mtp-Qwen3.8-Flash-Next-draft.gguf` of
  [agentionai/Qwen3.8-Flash-Next-Gyro-GGUF](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF)
  (sha256 `00818378…`), as in the release preset. The new draft file is not published here: it is rebuilt from the
  checkpoint with the commands above (md5 `e0aa166bbd5328f39026f2f42b458ff6`, sha256
  `985340d5f9ad75d925d899143c37c724f7f8274f4527636b6bea7ace063b0d5a` with `llama-quantize` of the fork's
  `0b8427657`).
- The requests are the release check's own recording of this project's agent check; the tables hold counters and
  hashes only. Q1173's four loads are that experiment's own rows.
- Raw output (server logs, the replay's rows with their times, the cards' memory) is not published.
