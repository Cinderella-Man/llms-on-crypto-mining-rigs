# Where a Flint file should spend its bytes for long agent conversations: UD-Q4_K_XL's tensor map against ours, recipes at equal speed, and a cheap 8k sensitivity map (2026-10-24)

Scope: with the screened head (Q92) on both files, Unsloth's UD-Q4_K_XL reads est. 2.060 GB per token and our K2 2.102,
so our files no longer win on bytes. They must win on quality at equal bytes, and quality must hold on long agent text,
not only on 512-token wikitext: at 8,192 tokens on Flint's own BFCL conversations every candidate is 1.03–1.04× UD's KLD
(Q106), and G3 trails UD on miss_param tasks (Q41 E2, exploratory). This note asks:
- where UD's advantage on long agent text comes from, tensor class by tensor class;
- which mixes at UD's bytes per token could beat it, with expected numbers;
- how to get an 8k sensitivity map without a full KLD run per variant;
- what in a KLD dump could predict miss_param failures;
- which experiments come first, and the rule for "our file beats UD at equal speed".

Conventions are those of the other notes:
- "(est.)" marks this note's own arithmetic or inference; the basis is given next to it.
- "P" and "S" in the source list mean primary (papers, code, model cards, measurements in this repository) and
  secondary.
- Nothing ran on the rig or the quant machine. UD-Q4_K_XL's tensor types were read from its GGUF header with
  standard-library Python. Q106's and Q107's per-token dumps and Q41's E2 transcripts (kept outside the repository)
  were re-analysed on a laptop; the new splits in section 1.1 come from that.
- Ratios are KLD(file) / KLD(UD) on the same tokens. "Read" tokens are what Flint prefills and never samples (tool
  schemas and system text, user turns, tool output); "written" tokens are what it samples (reasoning, calls,
  answers). "sr" in a recipe name means screened head plus F16 router.

Starting point, measured ([registry](../../../quants/experiments.md)):
- **Flint:** 49 layers (2 dense, 47 MoE with 256 routed experts, top-6, and a shared expert), hidden 2,048, 32 query
  and 4 KV heads of 128; 14 global layers without positional encoding (0, 1, 5, 9, …, 45, 48) and 35 sliding-window
  layers (513 keys, RoPE); vocabulary 150,144.
- **512-token KLD against BF16** (Q77's GPU set): G3 0.1040, G3h 0.1063, K2 0.1068, K2h 0.1092, K1h 0.1121; UD
  0.120. F16 router (Q113): no measurable cost.
- **Agent text at 8k (Q106/Q107),** ratio to UD, against Q8_0 (24 chunks) / BF16 (12 chunks):

  | File | All tokens | Read | Written | After `</think>` |
  |---|---|---|---|---|
  | G3 | 1.033 / 1.042 | 1.037 / 1.064 | 0.887 / 0.892 | 0.69 / 0.59 |
  | G3g (global attention Q8_0) | 1.032 / 1.035 | 1.037 / 1.074 | 0.865 / 0.883 | 0.60 / 0.59 |
  | K2h | 1.032 / 1.036 | 1.046 / 1.039 | 1.013 / 0.999 | 0.53 / 0.49 |
  | K1h | 1.042 / — | 1.051 / — | 1.038 / — | 0.69 / — |
  | Unsloth Q8_0 | — / 0.845 | — / 0.837 | — / 0.537 | — / 0.15 |

- **Bytes per token** (screened head 174 MB in place of a 327 MB Q8_0 head): UD 2.212 → 2.059; G3 2.326 → 2.124 with
  the F16 router too; K2 → 2.053; K1 → 2.009. On the fork's v2 build UD (26.84 ms) and S4-L0 (26.8 ms at 2.236 GB)
  run at the same speed per byte (Q76), so equal speed means equal bytes, ~2.06 GB.
- **Q112 E1:** GPTQ recovers 0.79 / 0.63 / 0.45 of Q5_0 rounding's cost on attention output 0–8, shared expert 9–16 and
  attention k/v 9–16.

## Verdict

- **UD has no advantage on what Flint writes; its edge is on predicting tool-schema text, and nothing so far ties that
  to how Flint behaves.**
  - On written tokens at 8k G3 is 0.89× UD (both references); K2h is level.
  - Against BF16 the read-token gap sits entirely in tool schemas and system text (G3 1.058, K2h 1.049). Tool output
    is level (0.999, 0.989), user turns favour our files (0.82, 0.87).
  - It does not grow along the chunk (schema tokens at 4–5k: 1.061; 6–7k: 1.057) or with copy distance (copies from
    512–1k back 1.106, from 4k+ back 1.026). On the copied token itself G3 is as good as UD (mean NLL excess +0.355
    against +0.352 nats).
  - It is not a long-range retrieval effect: restoring all global-layer attention to Q8_0 left it unchanged (Q107;
    read −0.001 ± 0.002 for G3g − G3).
- **Read-token KLD is a noisy, heavy-tailed yardstick.**
  - Top 5 % of read tokens carry 52 % of UD's read KLD; the median read token has 0.020 nats.
  - At ~0.6 % of unique read positions one model puts less than e⁻⁸ on a token the other is sure of. The Q8_0
    reference itself is the confidently wrong one at 180 such positions, the quant at 170–220.
  - The 24 chunks share schema prefixes: 19 % of scored tokens repeat another chunk's exact prefix, and the chunks
    form only 15 independent schema contexts. Bootstrapping by those 15 clusters widens G3's read interval from
    1.012–1.061 to 1.000–1.074 (against Q8_0).
  - Body changes of ~25 tensors move the read ratio by ±0.02 through routing flips alone (K2hg − K2h: +0.018 ± 0.011
    for a change that adds precision).
- **The bytes that matter on written tokens are now visible, and they are not where the 512-token sweeps put them.**
  Paired by chunk on Q106's dumps (ratio-to-UD units, ± SE):
  - **The GPTQ Q5_0 head costs +0.073 ± 0.009 on written tokens, nothing on read tokens.** That is +0.0015 nats per
    written token, close to its absolute cost at 512 tokens (+0.0023), on a base 5× smaller. The screened head is exact and 37 MB
    lighter, so it replaces the GPTQ head in every candidate.
  - **K2's groups (shared expert 33–45, k/v 33–48, q 0–8 at Q5_0) cost +0.069 ± 0.013 on written tokens** (K2h − G3h),
    nothing on read tokens, against +0.0028 KLD at 512.
  - **K1's groups (shared expert and attention output 25–32) cost +0.026 ± 0.007 written and +0.013 ± 0.004 on schema
    tokens**: the only measured change that moves the schema gap.
  - **Global-layer attention in Q8_0 helps written tokens a little:** −0.012 ± 0.007 (G3g) and −0.013 ± 0.007 (K2hg).
- **The structural difference left after Q107** is 19 sliding-layer attention tensors that UD keeps at Q8_0 and G3 at
  Q5_0: q in 27, 28, 30–32, 34, 36, 38, 40, 42, 44, 47 and output in 34, 36, 38, 40, 42, 44, 47 (60 MB per token). The
  other differences are UD's K-quant super-blocks (Q5_K) for the experts of layers 3–18 at G3's byte count, and G3's
  extra bytes on routed experts (+113 MB), the shared expert (+32 MB) and the F32 router (+49 MB).
  - Schema text is repeated JSON structure a few hundred tokens apart: within the 513-key window, where sliding-layer
    induction heads copy it.
  - Jin et al. find the massive q/k values that contextual tasks depend on in RoPE layers, which in Flint are the
    sliding ones.
  - So sliding-layer attention q/o in layers 25–48 is the most plausible carrier (est.). Two swap probes test it
    directly.
- **Best recipe at UD's bytes:**
  - **G3-6sr:** G3 with the screened head, the F16 router, and Q6_K instead of Q8_0 for the shared expert 25–45,
    k/v 33–48 and q 0–8.
  - **Numbers:** 2.066 GB per token (UD + screen 2.059), 512-token KLD est. 0.105–0.106 (0.88× UD), est. written
    0.90–0.91× UD, read est. 1.04–1.06× UD.
  - **It needs** the Pascal Q6_K kernel to pass its rig gate (Q101, ≥ 130 GB/s).
  - **Available today:** K2sr (K2 + screen + F16 router), 2.053 GB, KLD 0.107 measured, est. written ~0.94×, read
    1.04× (K2h measured).
  - **Both beat UD at 512 tokens and on written tokens, and both trail it ~4 % on schema tokens.**
- **The cheap sensitivity map is mostly a statistics problem, not a GPU-hours one.**
  - Q106's scoring runs took 2–6 minutes per file on two RTX 3090s, mostly model load. The limits are the
    noise (SE ~0.011 per body-variant comparison at 24 chunks), only 15 independent schema contexts in the material,
    and rewriting 30 GB per variant.
  - **The method:**
    1. Class-level probes, scored on more chunks with distinct schemas. A top-256 sparse reference makes 48+ chunks
       fit the scratch disk, and split-file variants avoid rewriting 30 GB.
    2. A per-tensor H-weighted error map from Hessians captured on schema and written positions at 8k, for the fine
       grain.
    3. A cross-prefill test (one file prefills, another writes) that measures whether reading fidelity changes
       writing at all.
- **miss_param needs its own proxy.**
  - Q106's dumps hold only 5 post-clarification turns (115 call tokens; G3 0.97×).
  - The G3 failures after asking that could be read are action decisions, not garbled values: re-asking instead of
    calling the obvious next tool, a name-lookup tool called with a ticker symbol, a prerequisite call skipped.
  - The proxy is a decision-point replay: log-odds of the reference action at each step boundary after the
    clarification, every file teacher-forced on the same transcripts. It is validated against Q41's discordant pairs.
- **First experiments** (in order; ~4 GPU-hours to the first decision):
  1. K2sr on agent 8k against BF16 (0.3 GPU-h): the number the release needs.
  2. Two swap probes, G3 with UD's attention and G3 with UD's routed experts, plus G3a (G3 + the 29 attention tensors
     UD keeps at Q8_0), ~1.5 GPU-h: says which class carries the schema gap.
  3. The cross-prefill test (~1 GPU-h + a day of code): says whether it matters.

## UD-Q4_K_XL against G3 and K2, by tensor class

Types per layer from the GGUF headers (UD) and the recipes (G3 = Q73's G3; K2 = Q77's round-5 mix). MB read per token;
routed experts count 6 of 256.

| Class (layers) | UD-Q4_K_XL | MB | G3 | MB | K2 | MB |
|---|---|---|---|---|---|---|
| Attention q, global (14) | Q8_0 ×13, Q5_K in 45 | 121.6 | Q8_0 0–21, Q5_0 25–48 (7) | 102.8 | Q5_0 in 0, 1, 5 and 25–48 (10), Q8_0 ×4 | 93.3 |
| Attention q, sliding (35) | Q8_0 ×29, Q5_K in 19, 26, 35, 39, 43, 46 | 293.1 | Q8_0 2–24 (18), Q5_0 26–47 (17) | 258.5 | Q5_0 ×23, Q8_0 ×12 | 239.6 |
| Attention k + v (49) | Q8_0 ×42, Q5_K in the same 7 | 103.6 | Q8_0 | 109.2 | Q8_0 0–32, Q5_0 33–48 | 96.6 |
| Attention output, global (14) | Q8_0 ×13, Q5_K in 45 | 121.6 | Q8_0 0–29, Q5_0 33–48 (5) | 109.1 | as G3 | 109.1 |
| Attention output, sliding (35) | Q8_0 ×29, Q5_K ×6 | 293.1 | Q8_0 2–32, Q5_0 34–47 (11) | 277.3 | as G3 | 277.3 |
| Shared expert 2–24 | Q8_0 ×22, Q5_K in 24 | 151.4 | Q8_0 | 153.7 | Q8_0 | 153.7 |
| Shared expert 25–48 | Q8_0 ×12, Q5_K ×11, IQ4_XS in 42 | 131.1 | Q8_0 | 160.4 | Q5_0 33–45, Q8_0 | 129.8 |
| Dense FFN 0–1 | Q8_0 | 80.2 | Q8_0 | 80.2 | Q8_0 | 80.2 |
| Routed gate/up 2–18 | Q5_K ×16, IQ3_S in 2 | 143.8 | Q8_0 2–8, Q5_0 9–18 | 180.1 | as G3 | 180.1 |
| Routed gate/up 19–48 | IQ4_XS ×25, Q4_K 21–23, Q5_K 27, 32 | 205.7 | Q5_0 | 259.5 | as G3 | 259.5 |
| Routed down 2–18 | Q5_K ×15, Q8_0 in 4, Q6_K in 8 | 76.7 | Q5_0 | 73.5 | as G3 | 73.5 |
| Routed down 19–48 | IQ4_XS ×16, Q5_K ×14 | 114.0 | Q5_0 19–45, Q8_0 46–48 | 136.8 | as G3 | 136.8 |
| Router | BF16 | 49.3 | F32 (F16: 49.3) | 98.6 | F32 | 98.6 |
| Output head | Q8_0 (screened: 174) | 327 | Q8_0 (G3h: GPTQ Q5_0 211; screened 174) | 327 | GPTQ Q5_0 in K2h (screened 174) | 211 |
| **Total** | **screened 2,059** | **2,212** | **screened + F16 router 2,124** | **2,326** | **screened + F16 router 2,053** | **2,140 (K2h)** |

- **Where UD spends more bytes than G3:** attention q (+56.6 MB) and output (+34.6 MB), all in layers 25–48. After
  Q107's global-layer change, 19 sliding tensors (60 MB) remain.
- **Where G3 spends more:** routed gate/up (+90.1 MB) and down (+22.8 MB), the shared expert (+31.7 MB, all in
  25–48), the F32 router (+49.3 MB).
- **Where the formats differ at equal bytes:** UD's routed experts in layers 3–18 are Q5_K (256-weight super-blocks
  with 6-bit sub-scales and minimums), G3's are Q5_0 (32-weight blocks, one FP16 scale, no minimum). On Llama-3-8B
  Q5_K_S has 25 % less KLD than Q5_0 at the same size ([Pascal forks and quants](pascal_forks_and_quants.md) §4.3).
- **K2 against UD:** UD has more attention bytes (q +84.9, output +34.6, k/v +8.6 MB) in 60 tensors. 84 MB of that is
  in sliding layers.
- **What both files share:** dense layers, k/v of layers 0–32, most of attention in 0–24 at Q8_0, and Unsloth's
  importance matrix. Its header records 687 chunks of 512 tokens, so neither file's calibration saw more than 512
  tokens of context.

## Ranked table

Bytes with the screened head on both files. Expected KLD at 512 tokens against BF16 on Q77's GPU set. Agent ratios
are to UD + screen at 8k (the screen is exact, so UD's measured 8k numbers stand for it). GPU-hours are job time on the
7-GPU quant machine; Q106's scoring runs took 2–6 minutes per file on two RTX 3090s.

| # | Recipe or method | GB per token (est.) | 512-token KLD (est.) | Agent read ratio to UD (est.) | Agent written ratio to UD (est.) | GPU-hours | First experiment | Kill criterion |
|---|---|---|---|---|---|---|---|---|
| 1 | **K2sr:** K2 + screened head + F16 router | 2.053 | 0.107 (K2 measured 0.1068) | 1.04 (K2h measured 1.039–1.046) | ~0.94 (K2h 1.013 minus the GPTQ head's +0.073) | 0.3 | L1: K2h-r16 (Q113) with G3's Q8_0 head swapped in (the screen's stand-in, composed in minutes), on agent 8k against Q8_0 (24 chunks) and BF16 (12) | Written upper 95 % bound ≥ 1.00 against BF16: no edge over UD + screen at equal bytes |
| 2 | **Swap probes:** SW-A = G3 + UD's attention (196 tensors), SW-X = G3 + UD's routed experts, G3a = G3 + Q8_0 on the 29 attention tensors UD keeps at Q8_0 | 2.403 / 2.217 / 2.417 (Q8_0 heads, F32 router; probes, not candidates) | — | the probe's result | — | 1–1.5 | L2: compose from G3 and UD (and the Q107 donor for G3a); agent 8k against BF16 (12 chunks) and Q8_0 (24); SW-X with the rescale on, since UD's Q5_K down-projection of layer 46 reads FP16 activation sums in prompt processing | No probe moves G3's schema-token ratio by ≥ 0.02 toward UD against BF16: the gap is not class-attributable; stop spending bytes on it |
| 3 | **Cross-prefill test:** UD prefills the conversation, G3 writes the last turn (and the reverse) | — | — | — | the test's result | 0.5–1, plus ~1 day of code | L3: 24–48 conversations × 4 prefill/write pairs, written-token KLD against BF16 | G3 writing over UD's prefix within ±2 % of G3 over its own: reading fidelity doesn't reach writing, read tokens become a diagnostic |
| 4 | **G3-6sr:** G3 + screen + F16 router + Q6_K for shared expert 25–45, k/v 33–48, q 0–8 | 2.066 | 0.105–0.106 | 1.04–1.06 (G3's, unchanged by these groups) | 0.90–0.91 (G3 0.887 + ¼ of the Q5_0 cost of the same groups) | ~1 | L6: after Q101's rig gate, compose with a Q6_K donor from BF16; 512 and agent 8k | Q6_K < 130 GB/s on the rig, 512-token KLD > 0.110, or written ratio not below K2sr's |
| 5 | **Decision-point replay for miss_param** | — | — | — | log-odds at call/ask/stop decisions | 1–2, plus ~1 day | L5: every Q41 miss_param transcript, both producers, replayed to each step boundary after the clarification | AUC < 0.6 for the per-task margin difference predicting Q41's discordant pairs: no cheap proxy; miss_param goes to task runs |
| 6 | **Same-token context contrast:** the 24 agent chunks rescored at `-c 2048`, paired per token with the 8k run | — | — | separates format from length | — | 0.5 | L4: Q8_0 base at 2k (29.5 GB, once), UD, G3, K2h | — (diagnostic: if the schema gap is the same at 2k, future maps run at short context) |
| 7 | **Sparse top-256 reference + 48 chunks with distinct schemas** | — | — | SE ~0.008 instead of ~0.011 | same | ~0.1 per variant; ½–1 day of code | L7: rescore UD and G3 on Q106's chunks with the sparse base | Per-token KLD correlation with the full base < 0.99, or ratios off by > 0.005 |
| 8 | **H-weighted error map** from Hessians captured on schema and written positions at 8k | — | per group | per group | per group | ~1, plus ~1 day | L8: `act_hessian` with `--parse-special` and a position mask; `gptq_body.py proxy` per tensor and format | Fewer than 7 of the 8 measured signs in section 3.4 reproduced, or Spearman < 0.7 on Q77's 20 groups |
| 9 | **G3a-6sr** (conditional): G3a + screen + F16 router + Q6_K shared expert, dense, gate/up 2–8, down 46–48; shared expert 33–45 and q 0–8 at Q5_0 | 2.061 | 0.112–0.116 | 1.00–1.02 if attention carries the gap | 0.95–1.00 | ~1 | Only if L2 finds attention and L3 finds that reading reaches writing | Read upper bound ≥ 1.03, or 512-token KLD > 0.118 |
| 10 | **K2sr-X** (conditional): K2sr with routed gate/up 9–18 and down 2–18 in Q5_K | 2.053 (+0.12 ms) | 0.105–0.106 | 1.00–1.03 if the experts carry it | ~0.93 | ~0.5 | Only if SW-X moves the schema ratio | As row 1 |
| 11 | Fisher information on written tokens, back-propagated through 8k of context (torch port of Flint) | — | — | — | — | 5–10, plus 3–4 days | Not first: L3 answers the file-level question for a fraction of the cost | — |

## 1. Where UD's long-context advantage comes from

### 1.1 What Q106's and Q107's dumps show, split further

Q106's per-token dumps hold, for every scored token, the KLD, the same-top flag and both models' log-probability of
the actual token. Re-labelled with `analyze.py`'s classes and split further:

**By class and reference.** All against UD:

| Class | Share of tokens | G3 vs Q8_0 | G3 vs BF16 | K2h vs BF16 | Unsloth Q8_0 vs BF16 |
|---|---|---|---|---|---|
| Tool schemas and system text | 40 % | 1.039 | 1.058 | 1.049 | 0.879 |
| Tool output | 11 % | 1.030 | 0.999 | 0.989 | 0.764 |
| User turns | 2 % | 0.896 | 0.816 | 0.867 | 0.423 |
| Reasoning | 39 % | 0.847 | 0.868 | 1.006 | 0.512 |
| Calls | 2 % | 0.953 | 0.963 | 1.172 | 0.846 |
| First token after `</think>` | 0.2 % | 0.689 | 0.589 | 0.488 | 0.150 |

- Against BF16, which removes the Q8_0 reference's bias, the read gap is a tool-schema gap.
  - Non-copied tool output favours our files (G3 0.981, K2h 0.969); copied tool output is 1.06 on 557 tokens.
  - The Q8_0 reference flatters UD: UD's Q8_0 tensors are byte-identical to Unsloth's Q8_0 file (five tensors
    checked), while only 4 of G3's 321 Q8_0 tensors are. Decisions use BF16.
- **No length signature:**
  - Schema tokens against BF16 by position: 1.061 (4,096–5,119), 1.056, 1.057, 0.964 (7,168–8,191).
  - Long-range copies by distance to their source, G3 against BF16: 1.106 (512–1k), 0.972 (1–2k), 1.101 (2–4k), 1.026
    (4k+). Against Q8_0: 0.928, 1.074, 1.056, 1.018. K2h shows no monotone trend either.
- **The copied token itself:** mean NLL excess over the reference (nll − nll_base) is +0.352 nats for UD, +0.355 for
  G3, +0.342 for G3g, +0.380 for K2h and +0.382 for K1h. G3 retrieves as well as UD; the K family loses ~0.03 nats,
  which fits its Q5_0 k/v in five global layers.
- **By BFCL category:** G3's read ratio is 1.023–1.051 in all four. miss_param, where G3 loses tasks, has G3's
  lowest written ratio (0.836).
- **By producer:** our files trail more on UD's own conversations (read 1.04–1.06) than on G3's (1.02–1.04). Each
  file's transcripts may favour it a little; the material mixes both equally.

**Heavy tails and bistable positions.**
- **Tail share:** in UD's read-token KLD the top 1 % of tokens carry 17.5 %, the top 5 % 52.8 % and the top 20 %
  92.8 %. The median read token has 0.020 nats and the 99th percentile 11.8 nats. The same holds for every file.
- **Confident disagreements** among the 33,188 unique read positions (nll > 8 nats where the other side's nll < 1):
  - the quant wrong: UD 192, G3 218, K2h 174, K1h 171;
  - the Q8_0 reference wrong and the quant right: 180–204.
  - None occurs on written tokens.
  - Example: after `"properties": {"brakePedalStatus": {"type": "` the schema's next token, `string`, gets 27 nats of
    NLL from UD and 0.05 from the reference, identically in the two chunks that share this prefix; K2h gets it right.
  - These are copy-source choices among repeated JSON blocks: a near-tie that any rounding can flip.
- **Not a few catastrophes:** removing the 100 rows with the largest |G3 − UD| difference leaves +3.6 % of the
  +4.1 % gap (unique read rows), 300 rows leave +2.7 %, 1,000 leave +1.6 %. The gap is broad, on a heavy-tailed base.
- **Why read KLD is large for every file:** quantization KL tracks the full model's next-token entropy (ρ 0.92,
  Lotfi et al., in the [answer-time note](answer_time.md)). Schema descriptions are high-entropy, and the bistable
  positions add the tail. That is why even Q8_0 keeps 0.84× UD's read KLD while it has half of UD's on written tokens.

**Correlated chunks.**
- 18,764 of the 98,280 scored tokens (19 %) sit at positions whose whole prefix, token included, equals an earlier
  chunk's; same-scenario conversations share their tool schemas.
- Grouped by a shared prefix reaching the scored half, the 24 chunks form 15 clusters.
- Cluster bootstrap, token-pooled ratios against Q8_0:

  | File | Read ratio | Chunk bootstrap | Cluster bootstrap (15) |
  |---|---|---|---|
  | G3 | 1.037 | 1.012–1.061 | 1.000–1.074 |
  | K2h | 1.034 | 1.015–1.052 | 1.008–1.061 |
  | K1h | 1.043 | 1.026–1.059 | 1.019–1.065 |
  | G3g | 1.036 | 1.010–1.061 | 0.997–1.074 |

  On unique rows only, the point estimates stay (G3 1.041, K2h 1.041, K1h 1.046). The gap is real for the K family
  and borderline for G3 against Q8_0.

**Paired effects of the changes already measured** (by chunk, in units of UD's KLD, ± SE over 24 chunks against Q8_0):

| Change | Tensors, MB per token | Read | Schema | Written | 512-token KLD |
|---|---|---|---|---|---|
| GPTQ Q5_0 head instead of Q8_0 (G3h − G3) | head, −116 | +0.0003 ± 0.0005 | −0.0005 ± 0.0004 | **+0.073 ± 0.009** | +0.0023 |
| K2's groups at Q5_0 (K2h − G3h) | shared 33–45, k/v 33–48, q 0–8; −72 | −0.004 ± 0.006 | −0.004 ± 0.007 | **+0.069 ± 0.013** | +0.0029 (K2 − G3: +0.0028) |
| K1's groups at Q5_0 (K1h − K2h) | shared 25–32, output 25–32; −44 | +0.009 ± 0.004 | **+0.013 ± 0.004** | +0.026 ± 0.007 | +0.0029 (K1 − K2: +0.0057) |
| Global attention to Q8_0 (G3g − G3) | 12 tensors, +38 | −0.001 ± 0.002 | −0.002 ± 0.002 | −0.012 ± 0.007 | −0.0009 |
| Global attention to Q8_0 (K2hg − K2h) | 25 tensors, +52 | +0.018 ± 0.011 | +0.027 ± 0.013 | −0.013 ± 0.007 | −0.0003 |

- **Written-token KLD is small (0.021 nats per token for UD),** so a fixed added error shows up large there. The GPTQ
  head adds ~0.0015 nats per written token, close to its absolute cost at 512 tokens (+0.0023). On read tokens it is
  invisible under 0.9 nats.
- **On written tokens at 8k the always-read late-layer tensors are worth ~3× their 512-token price.** K2's groups
  cost 2.4 % of UD's KLD at 512 and 6.9 % of UD's written KLD at 8k.
- **The only measured lever on schema tokens is layers 25–32** (attention output and shared expert): +1.3 %.
- **The head-only change has a paired SE of 0.0005 on read tokens; body changes 0.002–0.013.** Routing flips make body
  comparisons noisy.

### 1.2 The structural difference

- **UD's strategy:**
  - Attention at Q8_0 in 42 of 49 layers. The seven Q5_K layers (19, 26, 35, 39, 43, 45, 46) are sliding except 45.
  - Routed experts at 4.25–5.5 bits: Q5_K in 3–18, IQ4_XS/Q4_K in 19–48, IQ3_S in layer 2.
  - Shared expert at Q8_0 except twelve layers in 24–44.
  - Unsloth does not publish its selection rule ([survey A](quant_speed_a_file_only.md) §3.3).
- **Our strategy:** Q8_0 always-read tensors in layers 0–24 (Q77's sweep), Q5_0 attention q 25–48 and output 33–48,
  and 5–8 bits on every routed expert.
  - All our sweeps (Q36, Q45, Q73, Q77, Q90) ranked layer groups on 512-token wikitext.
  - In that test attention q/o in layers 25–48 cost little.
- **The 512-token frontier and the agent-text frontier disagree on the same groups.** K2's groups are the cheapest at
  512 tokens and dear on written tokens at 8k.
  - Unsloth's calibration text is chat, agentic coding and tool use (S); ours was wikitext for every sweep. The two
    allocations look tuned to their tuning domains.
  - The imatrix is shared, so it is not the difference.

### 1.3 Which tensor classes plausibly carry copying and reading of tool output

- **Copying from more than 513 tokens back** can only pass through the global layers' attention. Q107 restored all of
  it to Q8_0 without changing copies (G3g 1.049 against G3's 1.056 vs BF16). G3 already matches UD on the copied
  token's NLL, so there is nothing to recover there for G3.
  - The K family's Q5_0 k/v in global layers 33, 37, 41, 45, 48 is the one place where copies lose (+0.03 nats). It is
    within the noise of the read ratio.
- **Reading tool output** shows no gap against BF16, so nothing carries a deficit there.
- **Predicting tool-schema text** is the gap.
  - A schema repeats the same JSON skeleton every 100–300 tokens (`"type": …, "description": …, "required": …`). The
    next key is copied from the previous block: an induction pattern within the 513-key window.
  - Induction heads match the previous occurrence of the current token through q·k and copy what followed through v
    and the output projection (Olsson et al.).
  - So the carriers are the attention q, k, v and output of the sliding layers, and of the global ones at short range.
  - **Global attention** is ruled out by Q107. **k and v** are Q8_0 in G3 everywhere. That leaves **sliding q in
    layers 26–47 and sliding output in 34–47**, the 19 tensors UD keeps at Q8_0, plus K1's evidence for layers 25–32.
- **Why q in RoPE layers is plausible (est.):**
  - Jin et al. find the massive values that contextual retrieval depends on concentrated in q and k and caused by
    RoPE. Disrupting them takes passkey retrieval from 100 % to 0–9 %.
  - In Flint the RoPE layers are the sliding ones.
  - Per-head RMSNorm bounds q's length, but a 5-bit error still changes its direction, and the low-frequency RoPE
    dimensions carry retrieval (zeroing 32 of 128 such dimensions in retrieval heads drops recall from 1.00 to 0.18,
    arXiv 2606.21249).
- **The experts' format is the second candidate.**
  - UD's Q5_K super-blocks in layers 3–18 are ~25 % more accurate than Q5_0 at equal bytes on Llama-3-8B.
  - Experts act on each position alone, but which experts fire depends on the text, and schema JSON routes
    differently from wikitext.
- **The decision between the two is cheap** (L2): G3 with UD's attention, and G3 with UD's routed experts.

### 1.4 Literature

New sources for this note (the long-context, retrieval-head, decision-margin and agent-task literature is in
[long-context quality](long_context_quality.md) §1 and not repeated):
- **Red Hat / Neural Magic, RULER on quantized Llama-3.1 8B and 70B.**
  - W4A16 (INT4 weights), W8A8 INT8 and FP8 recover 99.2 % (8B) and 98.6 % (70B) of the unquantized score on average.
    Most formats stay above 99.5 % at every length from 4k to 64k.
  - INT4 falls to about 85 % (8B) and 88 % (70B) at 128k, where the unquantized models also score below 65.
  - With Mekala et al. (no loss at 8k), weight quantization at 4+ bits shows no long-context loss below 64k.
- **"Does RoPE Prevent or Degrade Retrieval Heads?"** (arXiv 2606.21249).
  - Masking 87 detected retrieval heads drops OLMo-2's needle recall from 1.00 to 0.00; matched random heads do
    nothing.
  - Zeroing the lowest-frequency 32 of 128 RoPE dimensions of those heads drops recall to 0.18, against 0.98 for random
    dimensions.
  - It does not cover NoPE or interleaved models, or quantization.
- **Olsson et al., induction heads** (arXiv 2209.11895): prefix matching plus copying is the mechanism for repeating
  earlier sequences in context. It is the mechanism schema prediction needs.
- **Second-order sensitivity for allocation:**
  - **SqueezeLLM** (ICML 2024) weights each weight's rounding error by the diagonal Fisher information from 100
    calibration samples (10 are "typically sufficient"). LLaMA-7B at 3 bits: wikitext-2 PPL 6.13 against 6.27 for
    GPTQ g128 (FP16 5.68).
  - **HAWQ-V2** (NeurIPS 2020) ranks layers by the average Hessian eigenvalue (the trace).
  - **FGMP** (arXiv 2504.14152) keeps blocks in FP8 or NVFP4 by Fisher-weighted perturbation: Llama-2-7B within 1 % of
    an all-FP8 perplexity.
  - All of them measure sensitivity on short calibration text. None scores positions by role (read against written)
    or at long context.
- **Not found:**
  - any measurement of weight-quantization sensitivity by token role (prefilled against generated);
  - any on structured-text prediction (JSON schemas) under quantization;
  - any study of interleaved sliding/global models under quantization ([long-context quality](long_context_quality.md)
    §1.5 found none either).

## 2. A byte budget at equal speed

### 2.1 What the free moves give

| Step | Bytes per token |
|---|---|
| UD-Q4_K_XL, Q8_0 head | 2.212 GB |
| UD + screened head (Q92), the file to beat | 2.059 GB |
| G3 / K2 / K1, Q8_0 head, F32 router | 2.326 / 2.255 / 2.211 GB |
| + screened head (−153 MB) + F16 router (−49 MB, Q113) | 2.124 / 2.053 / 2.009 GB |

- K2sr already sits at UD's bytes.
- G3sr is 65 MB (3.2 %) over; K1sr 50 MB under.
- The screened head beats the GPTQ Q5_0 head on both counts: 37 MB lighter, and it avoids the GPTQ head's +0.073
  written-token cost.

### 2.2 Byte-neutral trades, priced on both yardsticks

| Trade | MB per token | 512-token KLD | Written at 8k | Schema at 8k | Basis |
|---|---|---|---|---|---|
| K2's groups Q8_0 → Q5_0 | −72 | +0.0028 | +0.069 | −0.004 | Q77, Q106 (measured) |
| Same groups → Q6_K | −46 | +0.0004–0.0007 (est.) | +0.009–0.017 (est.) | ~0 (est.) | Q6_K costs 0.13–0.25 of Q5_0 (Q90: attention in Q6_K +0.008 against ~0.06 summed for Q5_0; llama.cpp's Llama-3-8B table 0.0055 against 0.0222) |
| Shared expert 25–32 → Q6_K | −12 | +0.0003–0.0005 (est.) | +0.002–0.003 (est.) | small | Q77 sh25-32 +0.0021 at Q5_0 |
| K1's groups → Q5_0 | −44 | +0.003–0.006 | +0.026 | +0.013 | Q77, Q106 |
| Global attention → Q8_0 (G3g) | +38 | −0.0009 | −0.012 | −0.002 | Q107 |
| 19 sliding attention tensors UD keeps at Q8_0 → Q8_0 | +60 | −0.002 (est., G3g's per tensor) | −0.01 to −0.02 (est.) | unknown: L2 | — |
| Q112's GPTQ groups (o0-8, sh9-16, kv9-16) → Q5_0 | −53 | +0.0087 | unknown | unknown | Q112 |
| Routed gate/up 2–8 Q8_0 → Q5_0 | −33 | +0.0060 | unknown | unknown | Q77 |
| Routed experts 9–18 (gate/up) and 2–18 (down) Q5_0 → Q5_K | 0 (+0.12 ms at 125 against 138 GB/s) | −0.001 to −0.003 (est.) | small (est.) | unknown: L2 | 25 % lower error at equal bits; Q5_0 experts cost ~0.015 in all (Pascal forks note) |
| GPTQ Q5_0 head instead of the screen | +37 | +0.0023 | +0.073 | 0 | Q56, Q106 |

The ordering on written tokens differs from the 512-token one: K2's groups are the cheapest 72 MB at 512 tokens and
cost as much as the GPTQ head at 8k. With a 6-bit rung the same bytes come out at a quarter of the price (est.).

### 2.3 The recipes

**R1, G3-6sr (best expected; needs Q101).**
- **Recipe:** G3 + screened head + F16 router, with Q6_K instead of Q8_0 for the shared expert in 25–45, attention
  k/v in 33–48 and attention q in 0–8 (241 M weights, −58 MB).
- **Bytes:** 2.066 GB per token, UD + screen + 0.3 %. If Q6_K streams at 130 rather than ~138 GB/s, the 198 MB of Q6_K
  cost ~0.09 ms more, ~0.3 % (est.).
- **512-token KLD:** G3's 0.1040 + 0.0008–0.0016 (Q77's Q5_0 costs of these groups sum to 0.0063, × 0.13–0.25) =
  **0.105–0.106 (est.)**, 0.88× UD.
- **Written at 8k:** G3's 0.887 + 0.011–0.020 = **0.90–0.91× UD (est.).** Against BF16 from 0.892.
- **Read at 8k:** **1.04–1.06× UD (est.)**, G3's own: K2's groups did not move it.
- **Build:** a Q6_K donor quantized from BF16 with Unsloth's importance matrix (`--tensor-type` for these 104
  tensors), composed into G3 with `compose_gguf.py`; with GPTQ-rounded Q6_K (Q115) later.
- **Why not G3 itself:** G3sr is 3.2 % over UD's bytes. The cheapest written-token-aware way down is the 6-bit rung,
  not Q5_0 cuts.

**R2, K2sr (available now).**
- **Recipe:** K2 + screened head + F16 router: 2.053 GB.
- **512-token KLD:** 0.107 (measured with a Q8_0 head; the screen is exact).
- **Written at 8k:** K2h's measured 1.013 minus the GPTQ head's +0.073 = **~0.94× UD (est.).** It needs measuring:
  the head's effect was measured on G3, not on K2.
- **Read at 8k:** 1.04× (K2h measured; the head does not move read tokens).
- **Status:** the fallback if Q101 fails, and the number to measure first (L1).

**R3, G3a-6sr (attention-first; only if L2 and L3 point there).**
- **Recipe:** G3 + Q8_0 for the 29 attention tensors that UD keeps at Q8_0 and G3 holds at Q5_0 (+91 MB). Then
  screen + F16 router, Q6_K for the whole shared expert, the dense layers, gate/up 2–8 and down 46–48, and Q5_0 for the
  shared expert 33–45 and q 0–8 to come back to 2.061 GB.
- **512-token KLD:** G3a ~0.101 (est.: G3g 0.1031 minus ~0.002 for the sliding tensors) + 0.010 (Q90: shared expert
  + dense + gate/up 2–8 in Q6_K, measured) + ~0.003 for the Q5_0 groups = **0.112–0.116 (est.)**.
- **Written at 8k:** 0.95–1.00 (est.): Q6_K on the early shared expert and dense layers costs written tokens; the
  attention restoration recovers ~1–2 %.
- **Read at 8k:** 1.00–1.02 if attention carries the gap.
- **Trade:** ~0.008 of 512-token KLD and est. 5–10 % of written-token KLD for ~4 % of schema-token KLD. Worth it only if
  reading reaches writing.

**R4, K2sr-X (expert-format variant; only if SW-X moves the schema ratio).**
- **Recipe:** K2sr with the routed experts' gate/up in 9–18 and down in 2–18 in Q5_K (160 MB per token, same bytes).
- **Cost:** Q5_K streams at 120–128 GB/s on the fork (Q46) against Q5_0's 138: +0.12 ms (est.), ~0.5 %.
- **512-token KLD:** 0.105–0.106 (est.).

**Not proposed:**
- **G3g or K2hg** (Q107: no read effect, 1.3 % on written for 38–52 MB);
- **the F1 class** (its bytes go to routed experts, which move neither yardstick's gap);
- **a q8_0 KV cache** (untested at 8k, Q62).

## 3. Cheap long-context sensitivity

### 3.1 What a variant costs today

- **GPU time is small.** Q106's agent-text scoring took 2 minutes per file loaded from NVMe and ~6 from the HDD on
  two RTX 3090s. The BF16-referenced runs on 12 chunks took 1–2 minutes. That is 0.07–0.2 GPU-hours per variant, not
  one.
- **What costs:**
  - **Disk:** a composed variant is a 25–33 GB file. Q112's `split_compose.py` (a shared first file plus a small
    second file per group) makes variants cost only their changed tensors.
  - **The reference:** 300 KB per scored token, 29.5 GB for 24 chunks of 8,192. With ≥ 50 GB free required on the
    scratch disk, about 30 chunks fit there today; the HDD takes 59 GB for 48 chunks at ~150 MB/s, ~7 minutes of
    reading per variant.
  - **Statistics:** the binding constraint.

### 3.2 How many chunks

From Q106's dumps (paired SE of the ratio difference over 24 chunks against Q8_0):

| Comparison | Read | Written | Chunks for SE 0.01 on read |
|---|---|---|---|
| Head-only change (G3h − G3) | 0.0005 | 0.009 | 1 |
| 12 attention tensors (G3g − G3) | 0.002 | 0.007 | 1 |
| 25 tensors incl. k/v (K2hg − K2h) | 0.011 | 0.007 | ~30 |
| Different mixes (G3 − UD, K2h − UD) | 0.011–0.015 | 0.021–0.027 | 31–53 |

- **Fewer chunks work only for changes that cause no routing flips** (the head, the last layers). Body changes need
  ≥ 24 independent chunks for a 3 % effect at 2 SE and ~50 for 2 %.
- **"Independent" means distinct tool schemas.** The 24 chunks hold 15. Adding Flint's own agent transcripts (Q51's
  38 conversations) and other tool sets raises the count; BFCL alone repeats schemas across scenarios.
- **Read-token-only scoring saves no GPU time** (the forward pass is the same). It sharpens the metric, and the dump
  already allows it.

### 3.3 The method

1. **L4, same-token context contrast (0.5 GPU-h, no code).**
   - Rescore Q106's 24 chunks at `-c 2048` against a 2k Q8_0 base. Each token at positions 4,096–8,191 whose 2k window
     scores it is then paired with its 8k score.
   - If the schema gap is the same at 2k, it is a format effect, not a length effect. The map can then use 2k slices
     of many more conversations (any slice of a schema works), which multiplies the independent schema contexts.
   - This is the long-context note's M1b, still not run, done on the same tokens.
2. **L7, a sparse reference (½–1 day).**
   - A `--kl-divergence-base` format holding the reference's top 256 log-probabilities and the tail mass: ~1.5 KB
     instead of 300 KB per token.
   - KLD over the top 256 plus one tail bucket is a lower bound of the full KLD, and equal to it wherever the top 256
     hold the mass. Q91 found 256 candidates enough to hold Flint's top 20 on every one of 40,527 rows.
   - 48–96 chunks then fit anywhere.
   - **Validation:** rescore UD and G3 on Q106's 24 chunks. Per-token correlation with the full-base KLD ≥ 0.99, and
     the class ratios within 0.005.
3. **L2, class-level probes on agent text** (swaps and restorations, split-file variants, ~0.15 GPU-h each).
   - They resolve effects ≥ 3 % at 24–48 chunks: enough to find which class carries the schema gap, not to rank 8-layer
     groups.
4. **L8, a per-tensor map from H-weighted errors (~1 GPU-hour + ~1 day).**
   - **Capture:** extend `act_hessian` (Q112) with `--parse-special` and a per-position mask. Capture H = Σ x xᵀ for
     every always-read input (attention norm, attention result, FFN norm, shared and dense down inputs) on 24–48 chunks
     at 8k, separately over schema positions and written positions.
     - Float64 H for ~200 tensors is ~10 GB of host RAM.
     - The accumulation is ~1.3·10¹⁴ multiply-adds on the CPU, ~15 minutes on 12 threads (est.); the forward pass takes
       minutes.
   - **Proxy:** `gptq_body.py proxy` gives tr(ΔW H ΔWᵀ) / tr(W H Wᵀ) per tensor and format. ΔW comes from the
     BF16 weights against each candidate format, and against UD's own tensors, dequantized.
     - Q5_0/Q8_0/Q4_x are quantized by script; K- and I-quant ΔW come from llama-quantize outputs or from UD's file.
     - Minutes on one GPU.
   - **Gains:** fit per-class gains on Q77's 20 measured 512-token group costs (H from wikitext at 512), then predict
     the 8k effects with the agent-text H.
   - **Output:** which classes' error grows most from wikitext to schema text (the ratio E_schema / E_wiki per class).
     That is the map the 8k runs can't resolve per group.
5. **L3, the cross-prefill test**, which says whether the map should target read positions at all (section 4.3).

### 3.4 How the methods validate against Q106

- **The sparse base** must reproduce Q106's ratios on the same chunks (above).
- **The H-weighted map** must reproduce the measured pairwise effects of section 1.1, in sign and order of magnitude:

  | Pair | Read | Written |
  |---|---|---|
  | GPTQ head | ~0 | large positive |
  | K2's groups | ~0 | positive |
  | K1's groups | positive on schema | positive |
  | G3g | ~0 | small negative |

  That makes eight signs; at least seven must come out right, together with Spearman ≥ 0.7 on Q77's 20 groups.
  - Q112's proxy predicted the recovered share in the right order but conservatively (0.71 / 0.45 / 0.45 against
    0.79 / 0.63 / 0.45).
- **The proxy has a blind spot:** each tensor's error is measured on the reference's inputs, so it misses routing
  flips and errors that travel through later positions' attention. The cross-prefill test and the full 8k runs cover
  those.
- **Fisher information on written tokens** (row 11) would cover them too, by back-propagating the written tokens'
  log-likelihood through the whole context into every weight. It is the principled version (SqueezeLLM's diagonal
  Fisher, at long context and restricted to written tokens). It needs Flint in torch across five cards without
  peer-to-peer: 3–4 days of work and est. 5–10 GPU-hours for 100k tokens. It comes after L3 says reading matters.

## 4. Task-level link: what could predict miss_param failures

### 4.1 What the KLD split shows, and doesn't

- **The miss_param chunks look like the others:** G3's read ratio 1.041, written 0.836 (its best category).
- **Q106's chunks contain only five post-clarification turns** (a user turn after a turn in which the model asked):
  79 clarification tokens and 115 call tokens.
  - On those calls G3 is 0.97× UD (cluster interval 0.94–1.25), K2h 1.02×; on the whole turn's written tokens G3 is
    0.91×.
  - The first token after `</think>`, which decides between a call and a reply, is 0.69× UD for G3 over all 230
    occurrences.
- **Mean KLD over token classes therefore gives no warning for this failure type,** in line with the long-context
  note's reading that the E2 gap is a decision effect.

### 4.2 What the failures are

- **The four G3 failures after asking whose transcripts were available locally**, against UD's passes on the same
  tasks:
  - Given the second city's name, G3 looked up the first city's zip code and then asked again for the second instead
    of looking it up (UD called the lookup twice and the distance tool).
  - Given a ticker symbol, G3 called the name-to-symbol lookup with the symbol, got "not found", and stopped (UD
    called the quote tool with the symbol).
  - In the ask turn G3 replied as if it would book, without asking for the destination, and then booked without the
    price check UD ran first.
  - In one task both files made the same calls in the clarification turn; the failure lies later.
- **None is a garbled copied value.** All are action decisions (call next, ask again, stop, which tool), and two
  depend on knowing a tool's arguments from its schema.
- These are the low-margin decisions that quantization flips first: margins shrink as m′ ≈ c·m + b, c < 1, and
  whether to call is more fragile than which tool (Wu, Dhiman and Koshiyama, arXiv 2608.06564; in the long-context
  note).

### 4.3 Proxies

**L5, decision-point replay (the proxy to build).**
- **Material:** every miss_param transcript of Q41 (47 exploratory + 139 confirmatory tasks, both producers, ~370),
  rendered with `multi_turn.py render` up to each step boundary from the ask turn on: est. 3–6 boundaries per
  transcript.
- **Scores, each file teacher-forced on the same prefix and the producer's reasoning:**
  - at the token after `</think>`, the log-odds of `<tool_call>` against reply text (act against ask or stop);
  - over the tool-name tokens, the log-odds of the BFCL ground-truth next call against the producer's actual choice;
  - the same under BF16, which sets the reference margin m.
- **Statistics per file:**
  - the margin shrink c (regression of m′ on m, as in arXiv 2608.06564);
  - the flip rate on decisions with BF16 margin < 2 nats;
  - the per-task mean margin difference G3 − UD.
- **Counterfactual reading:** on G3's failed transcripts, does UD (and BF16) also pick G3's action after G3's
  reasoning? If yes, the decision was set by the sampled reasoning, not the decision token, and only task runs can
  judge it.
- **Validation:** logistic regression of Q41's discordant outcomes (UD-only and G3-only passes) on the per-task margin
  difference, fitted on the 47 exploratory tasks and tested on the confirmatory 139. A proxy is useful at AUC ≥ 0.7
  (est.) and dead below 0.6.
- **Cost:**
  - Prefixes grow step by step, so llama-server's prompt cache reuses each conversation's previous boundary: est.
    ~2 M prefill tokens per file (370 transcripts of ~5–6k tokens), 10–15 minutes on two 3090s.
  - The BF16 reference with experts on the CPU takes ~1 GPU-hour.
  - One day of scripting: `/completion` with `n_probs` on prefixes ending at `</think>`, or the sparse-base dump at
    the selected positions.

**Free first step:** classify every post-clarification failure in Q41's raw outputs (exploratory and confirmatory, both
files) as act/ask/stop errors, tool-choice errors, argument errors (`unknown_arg` is already in the per-attempt
table) or later-turn errors. It tells which decision the replay must score.

**L3, the cross-prefill test (does reading fidelity reach writing?).**
- **Steps:**
  1. File A prefills a conversation's prompt: tool schemas, user turns, tool output and the earlier turns.
  2. A saves the sequence state; file B loads it (`llama_state_seq_get_data` / `set_data`; same architecture and KV
     layout).
  3. B scores the last turn's written tokens against BF16.
- **Four pairs:** UD→UD, G3→G3, UD→G3, G3→UD on 24–48 conversations of 6–16k tokens.
- **Reading:** (UD→G3) − (G3→G3) is the effect of UD's better reading on G3's writing; (G3→UD) − (UD→UD) the
  reverse.
  - If both are within ±2 %, the schema gap does not reach what Flint writes, and read-token KLD stops being a release
    metric.
  - If UD's prefix lowers G3's written KLD by ≥ 3 %, reading matters, and R3 or R4 becomes the target.
- **Cost:** ~150 lines on top of llama.cpp's save-load-state example; the KV state at 16k is ~0.45 GB (28 KiB per token);
  est. 0.5–1 GPU-hour.

## 5. Plan

### 5.1 Experiments, ranked

| Order | Experiment | GPU-h (est.) | Decides | Kill criterion |
|---|---|---|---|---|
| 0 | Q41's confirmatory miss_param test (running until 12:00) | 0 more | whether a task-level gap exists | p ≥ 0.05: miss_param is not a release gate beyond E2's margin; L5 drops to the end |
| 1 | **L1:** K2h-r16 with G3's Q8_0 head swapped in, on agent 8k against Q8_0 (24 chunks) and BF16 (12) | 0.3 | K2sr's written and read ratios at UD's bytes | Written upper 95 % bound ≥ 1.00 against BF16: K2sr has no edge |
| 2 | **L2:** SW-A, SW-X and G3a, against BF16 and Q8_0 | 1–1.5 | which class carries the schema gap | No probe moves G3's schema ratio by ≥ 0.02 toward UD against BF16 |
| 3 | **L3:** cross-prefill test | 0.5–1, plus ~1 day | whether reading reaches writing | Both cross terms within ±2 % |
| 4 | **L4:** same-token contrast at 2k | 0.5 | format against length | — |
| 5 | **L5:** decision-point replay (after the free failure classification) | 1–2, plus ~1 day | a cheap miss_param proxy | AUC < 0.6 on the confirmatory pairs |
| 6 | **L6:** G3-6sr, after Q101's rig gate | ~1 | the best recipe at UD's bytes | Q6_K < 130 GB/s; 512-token KLD > 0.110; written ratio not below K2sr's |
| 7 | **L7:** sparse reference + 48 chunks with distinct schemas, then the release comparison (best candidate against UD + screen) | ~1, plus ½–1 day | the final 8k numbers at SE ~0.008 | — |
| 8 | **L8:** H-weighted map | ~1, plus ~1 day | the per-group 8k map | < 7 of 8 signs, or Spearman < 0.7 |
| 9 | R3 or R4, whichever L2 and L3 point to | ~1 | an attention-first or expert-format file | As in the ranked table |

Total to the first decision (rows 1–3): est. 2.5–4 GPU-hours on one GPU pair, all quality work, plus a day of code
for L3.

### 5.2 The rule: "our file beats UD-Q4_K_XL at equal speed"

All of the following, with Q41's non-inferiority rules unchanged:

1. **Equal footing.**
   - Both files with the screened head (or both with Q8_0 heads) on the same build and cards.
   - Bytes per token at most 1 % above UD + screen (≤ 2.08 GB) and rig decode within −1 % of UD's (Q125).
2. **KLD against BF16, never against Unsloth's Q8_0** (UD's Q8_0 tensors are that reference's own):
   - **512-token wikitext:** ratio to UD ≤ 0.95 with the upper 95 % bound < 1.00.
   - **Agent text at 8k, written tokens:** upper 95 % bound of the ratio ≤ 1.00 (superiority on what Flint writes).
     - Cluster bootstrap by schema set, ≥ 24 chunks, ≥ 15 schema clusters.
     - Decision tokens (first token after `</think>`, calls) reported, with an upper bound ≤ 1.10.
   - **Agent text at 8k, read tokens:** upper 95 % bound ≤ 1.05 (non-inferiority). It gates only if L3 shows that
     reading reaches writing; otherwise it is reported.
3. **Tasks** (Q41):
   - E1 lower 90 % bound > −1.5 points; E2 on 799 tasks at K ≥ 2 > −3; E3 at 65,536 tokens > −3.
   - Malformed plus runaway ≤ +0.5 points.
   - **If the confirmatory miss_param test confirms G3's deficit (p < 0.05)**, a miss_param lane is added: the 186
     tasks at K = 5 (paired SE ~1.6 points, est. 12–18 GPU-hours per pair), lower 90 % bound > −5 points.
4. **Otherwise UD wins.** If a candidate meets the non-inferiority parts but not the superiority in 2, it ties UD. A
   tie goes to UD unless time per correct answer on E1/E2 ([answer time](answer_time.md)) favours ours by ≥ 2 %. Then
   v2 recommends UD-Q4_K_XL with the screened head on the fork.

Where the candidates stand today against this rule:
- **K2sr:** passes the 512-token test (0.89×). It should pass the written-token test (est. 0.94×, upper bound est.
  ~0.99) and the read-token test (K2h's BF16 interval 0.999–1.081 brushes 1.05). E1 passed for K2h; E2 and miss_param
  are open.
- **G3-6sr:** est. better on both KLD yardsticks, pending Q101.

## Already covered

- **Global-layer attention in Q8_0 (G3g, X1) and its GPTQ variant (X2):** [long-context quality](long_context_quality.md),
  Q107 (measured: no read effect). This note adds its small written-token effect (−1.2 ± 0.7 %).
- **Transcript KLD at 8k, wikitext at 8k, the format control M1b, synthetic retrieval, RULER, the decision procedure
  with sample sizes:** [long-context quality](long_context_quality.md) §3, §5, Q106. L4 is M1b on the same tokens.
- **GPTQ for always-read groups, its proxy, `act_hessian`, `split_compose.py`:** [GPTQ for the body](gptq_body.md),
  Q112. L8 reuses them with agent-text Hessians.
- **Q6_K as the 6-bit rung, its quality (Q90) and kernel (Q101), GPTQ-Q6_K (Q115):**
  [Pascal forks and quants](pascal_forks_and_quants.md), [GPTQ for the body](gptq_body.md).
- **K-quant rungs in the allocation search (Q5_K experts at Q5_0's bytes):** [Pascal forks and quants](pascal_forks_and_quants.md)
  idea 8, not yet run. R4 is its expert-format case, gated by L2.
- **Screened head (Q91, Q92), F16 router (Q23, Q35, Q113), same head treatment for release claims (Q125):**
  [breakthrough candidates](breakthrough_candidates.md), [answer time](answer_time.md).
- **Per-layer sensitivity at 512 tokens:** Q18, Q36, Q45, Q73, Q77, Q90.
- **The decision-margin literature, retrieval heads, NoPE sensitivity, massive values, long-context quantization
  studies:** [long-context quality](long_context_quality.md) §1.
- **Reasoning length and time per correct answer:** [answer time](answer_time.md), Q123–Q125.
- **A speed-aware allocator (cost table plus knapsack) and fake-quant sensitivity scans:** [survey A](quant_speed_a_file_only.md)
  §3.3. Its scan design is what L2/L8 run on agent text.

## Not verified

- **K2sr's written-token ratio:** est. ~0.94 assumes the GPTQ head costs K2 what it cost G3 (+0.073). L1 measures it.
- **Q6_K's cost on written tokens:** taken as 0.13–0.25 of Q5_0's, from 512-token data (Q90) and Llama-3-8B. Not
  measured on Flint at 8k.
- **Q6_K's rig speed:** Q101's kernel is correct but untimed (gate ≥ 130 GB/s).
- **Which class carries the schema gap:**
  - The sliding attention hypothesis rests on the structural difference, induction-head mechanics, the RoPE/massive-
    value literature and K1's +1.3 % on layers 25–32. No swap has been run.
  - The expert-format hypothesis is untested too.
- **Whether reading fidelity reaches writing:** the cross-prefill test has not been run. The written-token KLD at 8k
  already contains the effect of each file's own reading, so the written-token comparisons above stand either way.
- **The source weights:**
  - UD's Q8_0 tensors equal Unsloth's Q8_0 file (five tensors checked byte for byte), but only 4 of G3's 321 Q8_0
    tensors do.
  - If Unsloth's BF16 differs from the BF16 our files were made from, the BF16 reference favours our files slightly,
    and UD's schema lead against it is a lower bound.
- **The copy-source flips:** the reading of the confidently-wrong positions as near-ties between repeated JSON blocks
  comes from inspecting examples, not from attention maps.
- **The miss_param failure taxonomy:** four transcripts were readable locally; the other failures are on the quant
  machine. The free classification step covers them.
- **Unsloth's calibration:** content (chat, agentic, tool use) from its documentation (S). The imatrix chunk length of
  512 is from the file's header (P). The rule Unsloth used to pick layers is not published.
- **Cost estimates:** capture and solve times for L8, the cross-prefill tool's size, and the replay's prefill counts
  are arithmetic, not measured.

## Sources

Papers and reports (P), read 2026-10-24:
- Red Hat Developer, [How well do quantized models handle long-context tasks?](https://developers.redhat.com/articles/2024/02/03/how-well-do-quantized-models-handle-long-context-tasks)
  (RULER, Llama-3.1 8B/70B, W4A16, W8A8 INT8, FP8, 4k–128k; recovery averages and the 128k INT4 figures).
- [Does RoPE Prevent or Degrade Retrieval Heads? A Mechanistic Analysis Across Model Families](https://arxiv.org/abs/2606.21249)
  (arXiv 2606.21249): head masking, low-frequency RoPE dimension patch.
- C. Olsson et al., [In-context Learning and Induction Heads](https://arxiv.org/abs/2209.11895) (arXiv 2209.11895).
- S. Kim et al., [SqueezeLLM: Dense-and-Sparse Quantization](https://arxiv.org/abs/2306.07629) (arXiv 2306.07629,
  ICML 2024): Fisher-diagonal weighting, 100 calibration samples, Table 1.
- Z. Dong et al., [HAWQ-V2: Hessian Aware trace-Weighted Quantization](https://arxiv.org/abs/1911.03852) (NeurIPS 2020).
- C. Hooper et al., [FGMP: Fine-Grained Mixed-Precision Weight and Activation Quantization](https://arxiv.org/abs/2504.14152)
  (arXiv 2504.14152).
- Via [long-context quality](long_context_quality.md) and [answer time](answer_time.md) (read for those notes): Jin
  et al., Massive Values in Self-Attention Modules (arXiv 2502.01563); Wu et al., Retrieval Head (arXiv 2404.15574);
  Mekala et al. (arXiv 2505.20276); Wu, Dhiman, Koshiyama (arXiv 2608.06564); Lotfi et al. (KL against entropy).

Secondary (S):
- Unsloth, [Dynamic 2.0 GGUFs](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs): calibration content.

This repository (P):
- [quants/experiments.md](../../../quants/experiments.md): Q12, Q18, Q35, Q36, Q41, Q45, Q46, Q56, Q73, Q76, Q77, Q90,
  Q91, Q92, Q101, Q106, Q107, Q112, Q113, Q115, Q125.
- [bench/long_context_kld](../../../bench/long_context_kld/README.md) (`analyze.py`'s token classes and long-range copy
  rule); Q106's and Q107's per-token dumps (kept outside the repository), re-analysed for section 1.1: class ×
  reference splits, copy distance, position bins, tail shares, confident disagreements, prefix duplication and cluster
  bootstrap, paired effects with chunk SEs, post-clarification turns.
- Q41's E2 per-attempt table and transcripts (kept outside the repository): section 4.2.
- UD-Q4_K_XL's GGUF header (tensor types per layer, sliding-window and RoPE patterns) and Unsloth's importance-matrix
  header (687 chunks of 512 tokens), read with the header parser of [token_cost.py](../../../scripts/token_cost.py);
  five Q8_0 tensors of UD-Q4_K_XL compared byte for byte with Unsloth's Q8_0 file.
- [quants/gptq_head](../../../quants/gptq_head/README.md) (`act_hessian`, `gptq_body.py`, `split_compose.py`),
  [compose_gguf.py](../../../scripts/compose_gguf.py).
- llama.cpp fork source: `ggml/src/ggml-cuda/mmq.cuh` (`mmq_get_q8_1_ds_layout`: Q5_0 and Q8_0 use the D4 layout
  without activation sums in prompt processing, Q4_K/Q5_K the DS4 layout), the save-load-state example
  (`llama_state_seq_get_data` / `llama_state_seq_set_data`).
- Notes: [long-context quality](long_context_quality.md), [GPTQ for the body](gptq_body.md),
  [answer time](answer_time.md), [K-quant decode](kquant_decode_pascal.md),
  [Pascal forks and quants](pascal_forks_and_quants.md), [unconventional ideas](unconventional_ideas.md),
  [survey A](quant_speed_a_file_only.md), [attention at long context](pascal_attention_long_context.md),
  [Maya](maya_expert_tiering.md), [breakthrough candidates](breakthrough_candidates.md).
