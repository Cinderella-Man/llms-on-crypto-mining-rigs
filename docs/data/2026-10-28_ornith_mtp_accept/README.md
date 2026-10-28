# Ornith-1.5's built-in MTP layer: how often its first draft would be accepted (2026-10-28)

Registry row Q341. Ornith-1.5-35B and -122B ship one multi-token-prediction layer (the `nextn` block: the last block of
the GGUF, a full-attention MoE layer trained with the model) that guesses the token after next from the model's own
hidden state. llama.cpp runs it as the draft of `--spec-type draft-mtp`, which [the 122B's rig run](../2026-10-27_ornith/README.md)
used. This measures, offline and teacher-forced on held-out text, how close its distribution is to the model's own:
the fraction of drafts that speculative sampling would accept, at the first draft position.

## Method

`run/mtp_accept.cpp`, built against llama.cpp's own CUDA build (`run/build_tool.sh`), so it runs exactly the MTP graph
and kernels that `llama-server --spec-type draft-mtp` runs. Per chunk of N tokens of the text:

1. The model decodes the chunk and keeps, at every position k, its logits (which predict token k+1) and its
   post-norm hidden state h_k, the MTP layer's input.
2. An MTP context reads token k at position k with h_(k-1), as the server's draft driver pairs them (zeros at k = 0).
   Its logits at k predict token k+1 too.
3. At every position from 1 to N-2, with p the model's distribution and q the MTP layer's:
   - **acceptance** Σ min(p, q): the probability that speculative sampling accepts a draft drawn from q, at
     temperature 1, at 0.6, and under the makers' full sampling chain applied to both (top-k 20, top-p 0.95, then
     temperature 0.6, as llama.cpp's sampler orders them);
   - **top-1 agreement:** the two most likely tokens are the same (the acceptance of greedy drafting with greedy
     verification);
   - whether each top-1 is the text's own next token;
   - everything binned by the model's top probability at temperature 1.

Teacher-forced means every position conditions on the text's true tokens, not on tokens the model sampled itself: the
acceptance of one draft after a correct prefix. Chained drafts (the MTP layer run again on its own output, as
`--spec-draft-n-max 3` does) are measured by the second tool ([below](#chained-drafts-registry-q366)).

## Results

Expected acceptance of the first MTP draft, Σ min(p, q), and top-1 agreement, over every position of the text:

| Model, file (MTP layer's type) | Text | Positions | T 1 | T 0.6 | Makers' chain | Top-1 agree | Model's top-1 = text | MTP's top-1 = text |
|---|---|---|---|---|---|---|---|---|
| 35B, the makers' Q8_0 (Q8_0) | agent sessions | 98,280 | 0.866 | 0.876 | **0.876** | 0.875 | 0.848 | 0.817 |
| 35B, the makers' Q8_0 (Q8_0) | Exercism code | 40,800 | 0.899 | 0.908 | **0.908** | 0.909 | 0.888 | 0.860 |
| 122B, Unsloth's UD-Q2_K_XL (Q8_0) | agent sessions | 98,280 | 0.825 | 0.845 | **0.845** | 0.849 | 0.867 | 0.797 |
| 122B, Unsloth's UD-Q2_K_XL (Q8_0) | Exercism code | 40,800 | 0.844 | 0.865 | **0.865** | 0.871 | 0.899 | 0.832 |
| 122B, AtomicChat's AD-IQ3_XXS-IQ2_S (Q4_K) | agent sessions | 98,280 | 0.762 | 0.798 | **0.799** | 0.807 | 0.861 | 0.761 |
| 122B, AtomicChat's AD-IQ3_XXS-IQ2_S (Q4_K) | Exercism code | 40,800 | 0.787 | 0.830 | **0.831** | 0.843 | 0.897 | 0.808 |

By the model's own confidence (top probability at T 1), agent sessions, acceptance under the makers' chain:

| Model's top probability | 35B: share of positions | 35B: acceptance | 122B UD: share | 122B UD: acceptance |
|---|---|---|---|---|
| ≥ 0.99 | 55.7 % | 0.987 | 56.8 % | 0.939 |
| 0.90–0.99 | 15.0 % | 0.916 | 15.3 % | 0.888 |
| 0.70–0.90 | 9.8 % | 0.793 | 9.8 % | 0.763 |
| 0.50–0.70 | 8.2 % | 0.668 | 8.0 % | 0.632 |
| 0.30–0.50 | 6.7 % | 0.560 | 6.4 % | 0.534 |
| < 0.30 | 4.7 % | 0.425 | 3.6 % | 0.433 |

- **The 35B's MTP layer is about as close to its model as built-in MTP layers are reported to be:** 0.88 of first
  drafts accepted on agent sessions and 0.91 on code under the makers' sampling, 87–91 % top-1 agreement.
- **The 122B's is 3–4 points behind in its rig file:** 0.845 and 0.865. The gap sits where the model is surest: at
  top probability ≥ 0.99 (57 % of positions) the 35B's MTP layer is accepted 98.7 % of the time and the 122B's 93.9 %,
  and there the 122B's MTP top-1 misses the text's token 5.5 % of the time against its model's 0.2 %. In
  UD-Q2_K_XL the MTP layer is Q8_0 but it reads hidden states of a 2–3-bit model, not those it was trained on; the
  122B's Q8_0 (132.6 GB, seven GPUs) would separate the two.
- **Acceptance follows the model's confidence:** above 0.9 top probability (71–72 % of positions) 0.89–0.99 of drafts
  are accepted, below 0.5 about half.
- **The makers' top-k and top-p change nothing beyond temperature 0.6.** T 0.6 raises acceptance by 1–2 points over
  T 1 overall (3–4 on the AD file): up where the model is confident, down below 0.5.
- **The file's quantization of the MTP layer matters:** AtomicChat's AD file keeps it in Q4_K and is accepted 4.6
  and 3.4 points less often than UD-Q2_K_XL, which keeps it in Q8_0.
- **Against the rig:** with the AD file and `--spec-draft-n-max 3`, stock llama.cpp accepted 41–56 % of its drafts
  ([serve_probe.csv](../2026-10-27_ornith/serve_probe.csv), `*_drafted` and `*_accepted`), across up to three chained
  drafts per step, on text the model wrote itself. Offline its first draft is accepted 0.80–0.83 of the time, so the
  second and third drafts carry most of the loss.

## Chained drafts (registry Q366)

`run/mtp_chain.cpp` chains drafts as llama-server's draft-mtp driver does for a single-head model. Draft 1 is the
draft above. Draft j+1 reads draft j's token at the next position with the MTP layer's own post-norm hidden state from
draft j, while the layer's KV cache holds the true entries up to the last accepted position and the drafts after it.
Teacher-forced, the draft tokens are the text's own, so draft j is scored given that the drafts before it were
accepted. S positions at a time get their own sequence holding a copy of the true entries up to their position, a
metadata copy in the unified KV cache, so drafts 2 and 3 of 64 positions run as one batch each. Its draft-1 numbers
equal those of `run/mtp_accept.cpp` (0.8603 and 0.8604 on four code chunks; positions 1 to N-4 against 1 to N-2).

Acceptance of each draft given the earlier ones, and the expected number of drafts accepted per step with one, two or
three drafts (E_n = a₁ + a₁a₂ + …, per position; a step also yields the model's own token), makers' sampling:

| Model, file | Text | Draft 1 | Draft 2 | Draft 3 | E₁ | E₂ | E₃ |
|---|---|---|---|---|---|---|---|
| 35B, the makers' Q8_0 | agent sessions | 0.876 | 0.786 | 0.711 | 0.88 | 1.59 | 2.14 |
| 35B, the makers' Q8_0 | Exercism code | 0.908 | 0.823 | 0.757 | 0.91 | 1.68 | 2.30 |
| 122B, UD-Q2_K_XL | agent sessions | 0.845 | 0.738 | 0.666 | 0.85 | 1.49 | 1.94 |
| 122B, UD-Q2_K_XL | Exercism code | 0.865 | 0.770 | 0.702 | 0.87 | 1.55 | 2.07 |

- **Each further draft loses about 0.07–0.11 of acceptance:** the MTP layer reads its own hidden state instead of the
  model's, one step further from what it was trained on.
- **Three drafts give ~1.9–2.3 accepted per step on these texts**, so ~2.9–3.3 tokens per step. The rig's 2.2–2.7
  tokens per step with three drafts (stock llama.cpp, the AD file, the model's own sampled text) are in line with the
  AD file's lower first-draft acceptance.

## Cheaper drafts (registry Q396)

`run/mtp_variant.cpp` keeps the model as the target and takes the draft from a second GGUF, as llama-server's
draft-mtp does with a draft model file. The MTP graph reads only that file's MTP block, token embedding, head norm and
output head, so a variant is a file with those in other types (`run/draft_file.sh`; the rest of the file, never read,
is TQ1_0 and stays in host memory). Further variants are a draft-only metadata override (fewer routed experts), the
routed experts' down projection zeroed (`run/tensor_zero.py`: the shared expert alone), a shortlist of the draft head
(`run/shortlist.py`, FR-Spec style: draft probability only on the K most frequent token ids, the target unchanged),
and a screened head (the draft head ranks every token, the model's own head rows give exact logits for its top 256).
With the draft file holding the model's own types it gives Q341's numbers to the last digit (0.8604 on four code
chunks). Each variant runs on the first 40 code chunks (20,400 positions) and the first two agent chunks (16,380), under
the makers' sampling; the ref row is the same slices, so every change is paired.

Draft bytes and P106 time are est. from the shapes and the measured Pascal rates of the
[cheap MTP draft note](../../research/notes/cheap_mtp_draft.md) (Q8_0 169 GB/s, Q5_0 150, Q4_0 145, Q6_K 139; Q4_K
~120 and Q2_K ~70 GB/s, est.), plus ~1.35 ms per draft for norms, router, sampling and the host (~1.15 ms with a
shortlist's small sampling).

**Ornith-1.5-35B** (the makers' Q8_0 as the target; acceptance change in points against ref):

| Variant | Code | Agent | MTP block, MB | Head, MB | est. ms per draft |
|---|---|---|---|---|---|
| ref: block, embedding, head Q8_0 | **0.919** | **0.823** | 70 | 540 | 5.0 |
| head Q5_0 | −0.0 | −0.0 | 70 | 350 | 4.1 |
| head Q4_0 | −0.1 | −0.1 | 70 | 286 | 3.7 |
| screened head (Q4_0 ranks, Q8_0 rows rescore the top 256) | +0.0 | +0.0 | 70 | 287 | 3.9 |
| shortlist head, 64k ids (Q8_0 rows) | −0.6 | −0.1 | 70 | 143 | 2.4 |
| shortlist head, 32k ids | −4.0 | −0.1 | 70 | 71 | 2.0 |
| shortlist head, 16k ids | −11.7 | −0.1 | 70 | 36 | 1.8 |
| shortlist head, 8k ids | −15.6 | −0.7 | 70 | 18 | 1.7 |
| MTP block Q4_K | −0.2 | −0.3 | 38 | 540 | 4.9 |
| routed experts Q2_K, rest of the block Q8_0 | −0.3 | −0.5 | 52 | 540 | 4.9 |
| MTP block Q2_K | −4.0 | −6.8 | 23 | 540 | 4.9 |
| top-4 of 8 routed experts | −0.5 | −1.0 | 57 | 540 | 4.9 |
| top-2 | −2.4 | −3.9 | 50 | 540 | 4.9 |
| the shared expert alone | −8.8 | −9.1 | 43 | 540 | 4.8 |

- **The head is the draft:** 540 of a draft's 610 MB. A Q4_0 or Q5_0 head costs nothing measurable and saves
  ~1.3 ms; shrinking the block or its experts saves under 0.2 ms, whatever it costs in acceptance.
- **Shortlists cost what they leave out.** The ids are ranked by their count in 8.3 M tokens of agentic run 1's
  sessions (outside the 40 held out) and Ornith's own replies in the agent check. The agent text, the same kind of
  text, loses ≤ 0.7 points even at 8k; the code text loses 0.6 points at 64k and 4–16 at 32k and below, as its
  coverage falls (64k ids cover 99.0 % of its tokens, 32k 96.0 %, 16k 88.7 %, 8k 82.8 %). Ornith's own replies, the
  text a draft meets on the rig, sit between the two (lists from run 1's counts alone cover 89.4 % of them at 8k and
  94.1 % at 16k).
- **Fewer experts or a 2-bit block cost acceptance for little time:** top-4 −0.5 to −1.0 points, a Q2_K block −4 to −7;
  Q2_K on the routed experts alone costs 0.3–0.5 points and halves their memory, which matters for card memory
  (R8's state), not for time.

**Ornith-1.5-122B** (Unsloth's UD-Q2_K_XL as the target; its draft path as shipped: MTP block Q8_0, head Q6_K,
embedding Q4_K; the head variants requantized from that Q6_K, the block variants from its Q8_0):

| Variant | Code | Agent | MTP block, MB | Head, MB | est. ms per draft |
|---|---|---|---|---|---|
| ref: as in UD-Q2_K_XL | **0.874** | **0.824** | 197 | 626 | 7.6 |
| head Q5_0 | −0.0 | −0.0 | 197 | 524 | 6.6 |
| head Q4_0 | −0.1 | −0.1 | 197 | 429 | 6.1 |
| shortlist head, 64k ids (Q6_K rows) | −0.5 | −0.1 | 197 | 165 | 4.1 |
| shortlist head, 32k ids | −3.5 | −0.1 | 197 | 82 | 3.5 |
| shortlist head, 16k ids | −10.4 | −0.1 | 197 | 41 | 3.2 |
| MTP block Q4_K | −2.9 | −5.8 | 106 | 626 | 7.3 |
| routed experts Q2_K, rest of the block Q8_0 | −0.7 | −1.8 | 142 | 626 | 7.5 |
| top-4 of 8 routed experts | −2.3 | −3.4 | 157 | 626 | 7.4 |
| top-2 | −12.9 | −11.3 | 137 | 626 | 7.3 |

- **The heads behave as on the 35B:** a Q5_0 or Q4_0 head costs ≤ 0.1 points, a 64k shortlist 0.5 on code and 0.1 on
  agent sessions, 32k and below lose on code what they leave out.
- **The 122B's MTP block is far more sensitive than the 35B's:** Q4_K costs 3–6 points (the 35B's 0.2–0.3), top-4
  2–3 (0.5–1.0), Q2_K experts 0.7–1.8 (0.3–0.5). Its block's weights are best left as the makers trained them, which
  matches AtomicChat's Q4_K block accepting 3–5 points less than UD's Q8_0 one (above).
- The block's time here includes its attention reading 2 KiB of KV per position (~0.6 ms at 6k positions) and ~0.35 ms
  of norms, router and gates, per the note; the fixed sampling and host share is ~1 ms (~0.8 ms with a shortlist).

## valid1's own MTP layer (registry Q397)

valid1, the N2P/N3P file of the 122B ([2026-10-28_ornith_valid1](../2026-10-28_ornith_valid1/README.md)), keeps its
MTP block in its own formats: routed experts N2P (gate, up) and N3P (down), the rest Q5_0, 105 MB per pass against
UD-Q2_K_XL's 197 MB; head Q5_0, token embedding Q8_0. The three tools, built against the low-bit branch
(`pascal-p106-v3-lowbit` at `d7a75deb3`), on the full texts as above (makers' sampling):

| File (target and its own MTP layer) | Text | Draft 1 | Draft 2 | Draft 3 | E₁ | E₂ | E₃ | Top-1 agree |
|---|---|---|---|---|---|---|---|---|
| valid1 | agent sessions | 0.843 | 0.740 | 0.656 | 0.84 | 1.49 | 1.94 | 0.848 |
| UD-Q2_K_XL | agent sessions | 0.845 | 0.738 | 0.666 | 0.85 | 1.49 | 1.94 | 0.849 |
| valid1 | Exercism code | 0.864 | 0.760 | 0.676 | 0.86 | 1.54 | 2.04 | 0.873 |
| UD-Q2_K_XL | Exercism code | 0.865 | 0.770 | 0.702 | 0.87 | 1.55 | 2.07 | 0.871 |

- **valid1's MTP layer is as good as UD-Q2_K_XL's at half the bytes:** the same drafts accepted per step on agent
  sessions (1.49 with two drafts, 1.94 with three), and within 0.01–0.03 on code. The cost note had predicted 0 to −4
  points.
- **On UD-Q2_K_XL's trunk**, valid1's draft path (its block, Q5_0 head and Q8_0 embedding, through
  `run/mtp_variant.cpp`) is accepted more often than UD's own: 0.889 against 0.874 on the code slice and 0.828 against
  0.824 on the agent slice. Its block is no closer to the original weights than UD's Q8_0 block, so the difference
  points at the token embedding the MTP layer reads (Q8_0 in valid1, Q4_K in UD); that test is not finished (below).

## Where this stopped (2026-10-29)

Done: Q341 (first drafts), Q366 (chained drafts), Q396's variant curves for the 35B and the 122B, Q397 (valid1's own
MTP layer). **The point for Q211's kernels:** keep the MTP block as the makers ship it (Q8_0, or valid1's N2P/N3P
block, which accepts as well) with all 8 routed experts, and draft through a shortlist head of the 64k most frequent
ids with Q4_0 rows: est. −0.5 to −0.6 points on code and −0.1 on agent sessions, a draft of ~3.7 ms instead of ~7.6
on the 122B (UD) and ~2.0 instead of ~5.0 on the 35B (the two head changes are measured apart; their sum is est.).
Fewer experts, a K-quant block or a 2-bit block save under 0.3 ms per draft and cost the 122B 1–13 points.

Not finished, in order:
1. **The token embedding's part in valid1's +1.5 points on UD's trunk:** UD's draft path with the makers' Q8_0
   embedding (`run/compose_gguf.py`-style compose of the reference draft file with `^token_embd\.weight$` from the
   Q8_0; the code slice ran on 2026-10-29, the agent slice did not).
2. **The recommended point measured as one variant** on both models: draft file with HEAD=q4_0, run with
   SHORTLIST=shortlist-65536.txt (and, for the 122B, the Q8_0 embedding).
3. Then the R8 engine work (registry R8 rows: accepted-path state replay for the pipelined executor, with this MTP
   layer as the guesser).

Commands for both steps: `run/draft_file.sh` (BLOCK=48, HEAD=q4_0, EMBD=q4_k, ALLOW_REQUANT=1 from UD-Q2_K_XL), then
`FAST=1 DRAFT=<file> DRAFT_CPU='^blk\.([0-9]|[1-3][0-9]|4[0-7])\.' SHORTLIST=shortlist-65536.txt BIN=<mtp-variant>
MODEL=<UD-Q2_K_XL> TEXT=<code text> CHUNK=512 CHUNKS=40 GPUS=3,4 OUT=<tsv> bash run/run_accept.sh`, and the same with
the agent text, CHUNK=8192 CHUNKS=2.

## What it means for the engine

Speculation with the MTP layer pays when a verification step, the model on the sampled token plus the draft, costs
much less than two plain steps. With one draft accepted ~0.85–0.9 of the time, a step yields ~1.85–1.9 tokens. On the
rig it does not pay yet: v2.0 with the MTP head ran 7.9 against 8.8 tok/s on the 122B's agent check
([the benchmark](../../benchmarks/2026-10-27-ornith-122b.md)), and the Pascal IQ2_S/IQ3_XXS expert kernels run 2–4
tokens at 0.5–0.8× llama.cpp's speed, a 2-token expert product at 3–4× the cost of one token
([2026-10-27_iq_kernels](../2026-10-27_iq_kernels/README.md)).

- **First: expert kernels for 2–8 tokens** (registry Q211). With a verification that reads each weight once, and the
  agent-session acceptances above, est. decode against no drafts:

  | Drafts per step | 35B P-Q5 (GB per token) | 122B UD-Q2_K_XL (ms of GPU time per token) |
  |---|---|---|
  | 0 | 2.82 | 93 |
  | 1 | 2.18 (1.29×) | 76 (1.23×) |
  | 2 | **2.06 (1.37×)** | **74 (1.25×)** |
  | 3 | 2.09 (1.35×) | 78 (1.20×) |

  Two drafts are the most: every further token in the verification reaches ~7–8 more of each layer's 256 experts.
  The model: the dense tensors once per step; the routed experts by the distinct experts the step's tokens choose
  (15.8, 23.3 and 30.6 for 2, 3 and 4 tokens against 8); per draft one MTP layer and the output head (~0.6 GB, ~6 ms).
  The 122B's split is the tuned IQ kernels' profile (36 ms experts, 36 ms dense, 21 ms the rest). This needs no new
  speculation code: llama.cpp's `draft-mtp` already drafts with the layer and rolls the recurrent state back.
- **Then: the recurrent state by accepted-path replay** (QUEUES R8, Q162), so that the MTP layer can guess in Miner
  Llama's pipelined executor (Q341), which fills the cards that sit idle. It rests on the same multi-token kernels.
- **The draft's own cost:** the output head (248,320 tokens: 0.54 GB in the 35B's Q8_0, 0.63 GB in UD-Q2_K_XL) is most
  of one MTP draft's bytes; Flint's screened head (Q91, Q92) cuts that cost.

## Files

| File | What |
|---|---|
| `35b-q8_0-{agent,code}.tsv`, `122b-ud-{agent,code}.tsv`, `122b-ad-{agent,code}.tsv` | per run: positions, share, acceptance at T 1, T 0.6 and under the makers' chain, top-1 agreement, each top-1's hit rate on the text, mean top probability; all positions and per bin of the model's top probability |
| `chain-35b-q8_0-{agent,code}.tsv`, `chain-122b-ud-{agent,code}.tsv` | chained drafts per run: positions, share, acceptance of drafts 1–3 given the earlier ones (makers' sampling), their top-1 agreement, expected drafts accepted per step E₁–E₃; all positions and per bin of the model's top probability |
| `*-chunks.tsv` | the acceptance numbers per chunk of each run |
| `variants-35b.tsv`, `variants-122b.tsv` | Q396 per variant: positions, makers'-chain acceptance, top-1 agreement and the change against ref, on the code and agent slices (`run/variant_table.py`) |
| `122b-valid1-{agent,code}.tsv`, `chain-122b-valid1-{agent,code}.tsv` | Q397: valid1 as its own target, first drafts and chained drafts (full texts) |
| `var-122b-valid1mtp-{agent,code}.tsv` | Q397: valid1's draft path on UD-Q2_K_XL's trunk (slices, FAST) |
| `shortlist-65536.txt` | the 64k shortlist, token ids most frequent first (`run/shortlist.py`; the 8k–32k lists are its first rows) |
| `run/mtp_variant.cpp`, `run/draft_file.sh`, `run/variants.sh`, `run/tensor_zero.py`, `run/shortlist.py`, `run/variant_table.py` | Q396's tool, draft files, driver, the shared-expert-only zeroing, the shortlists and the table |
| `run/mtp_accept.cpp` | the first-draft measurement (C++ against llama.cpp's API: `llama_process`, `common_batch::set_embd_state`, `llama_get_embeddings_nextn_ith`) |
| `run/mtp_chain.cpp` | the chained-draft measurement (the same API, plus sequence copies in the MTP layer's unified KV cache) |
| `run/build_tool.sh` | either tool's build against an existing llama.cpp CUDA build, in the CUDA image |
| `run/run_accept.sh` | one run of either tool on CUDA GPUs in Docker |

## Made with

```bash
D=docs/data/2026-10-28_ornith_mtp_accept/run
LLAMA=<llama.cpp checkout> BUILD=build-cuda OUT=<work>/bin/mtp-accept bash $D/build_tool.sh
BIN=<work>/bin/mtp-accept MODEL=<Ornith-1.5-35B-Q8_0.gguf> TEXT=code-exercism.txt CHUNK=512 CHUNKS=80 GPUS=3,4 \
  OUT=35b-q8_0-code.tsv bash $D/run_accept.sh
BIN=... MODEL=<Ornith-1.5-35B-Q8_0.gguf> TEXT=agent-heldout.txt CHUNK=8192 CHUNKS=12 GPUS=3,4 OUT=35b-q8_0-agent.tsv \
  bash $D/run_accept.sh
BIN=... MODEL=<Ornith-1.5-122B-UD-Q2_K_XL.gguf> TEXT=code-exercism.txt CHUNK=512 CHUNKS=80 GPUS=3,4 OUT=122b-ud-code.tsv \
  bash $D/run_accept.sh
BIN=... MODEL=<Ornith-1.5-122B-UD-Q2_K_XL.gguf> TEXT=agent-heldout.txt CHUNK=8192 CHUNKS=12 GPUS=3,4 \
  OUT=122b-ud-agent.tsv bash $D/run_accept.sh
# the same two runs with Ornith-1.5-122B-AD-IQ3_XXS-IQ2_S.gguf (122b-ad-code.tsv, 122b-ad-agent.tsv)
# per-chunk tables from each run's log (OUT with .log)
awk 'BEGIN{print "chunk\tacc_t1\tacc_t06\tacc_chain\ttop1_agree"} /^chunk /{split($2,a,"/"); print a[1]"\t"$4"\t"$6"\t"$8"\t"$10}' \
  35b-q8_0-code.log > 35b-q8_0-code-chunks.tsv
# chained drafts: the same four runs (35B, 122B UD) with the second tool
TOOL=mtp_chain LLAMA=<llama.cpp checkout> BUILD=build-cuda OUT=<work>/bin/mtp-chain bash $D/build_tool.sh
BIN=<work>/bin/mtp-chain MODEL=<Ornith-1.5-35B-Q8_0.gguf> TEXT=code-exercism.txt CHUNK=512 CHUNKS=80 GPUS=3,4 \
  OUT=chain-35b-q8_0-code.tsv bash $D/run_accept.sh
awk 'BEGIN{print "chunk\ta1\ta2\ta3\tE1\tE2\tE3"} /^chunk /{split($2,a,"/"); print a[1]"\t"$4"\t"$6"\t"$8"\t"$10"\t"$12"\t"$14}' \
  chain-35b-q8_0-code.log > chain-35b-q8_0-code-chunks.tsv
```

- **The engine:** llama.cpp `d77256999` as built on the quality machine (CUDA 12.9, compute 7.0 and 8.6), the build that
  made the KLD bases; two V100 32 GB.
- **The texts:** those of [2026-10-27_ornith_kld](../2026-10-27_ornith_kld/README.md): the agent check's Exercism
  code (80 chunks of 512 tokens) and held-out agent sessions (chunks of 8,192 tokens).
- **The files:** the makers' Q8_0 of the 35B (its MTP layer in Q8_0, as every tensor); the 122B's two rig-sized
  files, Unsloth's UD-Q2_K_XL (the rig's file: MTP layer Q8_0, the rest mostly 2–3-bit) and AtomicChat's
  AD-IQ3_XXS-IQ2_S (MTP layer Q4_K), so the 122B numbers measure each MTP layer against the quantized model the rig
  runs. The 122B's Q8_0 (132.6 GB) needs all seven GPUs of the machine and was not run.
- **Run time:** 7–16 minutes per run on the two V100s, most of it the statistics over 248,320-token distributions on
  the CPU at a low share. Times on a shared machine are not speed results.

## Provenance and licence

- **Models:** Ornith-1.5-35B-A3B and Ornith-1.5-122B by ornith-ai (MIT); the files as in
  [2026-10-28_ornith35_kld](../2026-10-28_ornith35_kld/README.md) and [2026-10-27_ornith](../2026-10-27_ornith/README.md).
- **Texts:** as in [2026-10-27_ornith_kld](../2026-10-27_ornith_kld/README.md).
- **This folder:** the tables and scripts are under this repository's licence.
