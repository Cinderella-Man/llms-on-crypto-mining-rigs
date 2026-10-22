# EAGLE-3 for Flint: a one-hour check before training a drafter

Speculative decoding lets a small, cheap *drafter* propose the next k tokens; the big model then checks all k in one
step and keeps the ones it agrees with, plus one token of its own. When checking several tokens costs little more
than generating one, every accepted draft is a free token. [EAGLE-3](https://arxiv.org/abs/2503.01840) is the drafter
llama.cpp runs for any target model (`--spec-type draft-eagle3`): one transformer layer that reads three of the
target's internal layers (for Flint the inputs of layers 2, 24 and 46) and drafts with its own small output head.

[Flint](https://huggingface.co/RecursiveSI/flint) has no drafter: no multi-token-prediction layer and no small model
with its vocabulary. Training an EAGLE-3 head takes the [quant machine](TRAINING_PLAN.md) several days. Whether that
pays off on this rig depends on two costs that can be measured **without** a trained head:

- **v(n)**, the time of one Flint decode step that checks n tokens at once (n = k + 1 for k drafts). On these 5-SM
  cards a step is limited by per-kernel latency more than by memory bandwidth, so checking 2–4 tokens should cost only
  10–40 % more than one ([Flint benchmarks](../../docs/benchmarks/2026-10-22-flint.md): two conversations in parallel
  run at 87–97 % of single-conversation speed each). The risk is the opposite case seen on other hardware, where
  checking a draft activates more experts and an EAGLE-3 head made a Mixture-of-Experts model *slower*
  ([llama.cpp PR #18039, comment of 2026-01-15](https://github.com/ggml-org/llama.cpp/pull/18039#issuecomment-3755925892):
  gpt-oss-120b at 0.46–0.71× on a DGX Spark).
- **D(k)**, the drafter's own time per round of k drafts. It does not depend on the head's weights, so a head with
  random weights and the right shapes times it exactly.

The remaining unknown, how many drafts a trained head gets accepted, is taken from published heads on similar
Mixture-of-Experts models. If the arithmetic says "go", [TRAINING_PLAN.md](TRAINING_PLAN.md) is the next step.

## Files

| File | What it does |
|---|---|
| [verify_cost.sh](verify_cost.sh) | Measures v(n) for n = 1–8 three ways (llama-bench, llama-server with synthetic acceptance, llama-server with real n-gram drafting) |
| [draft_cost.sh](draft_cost.sh) | Times EAGLE-3 rounds with a random head at k = 1–4 drafts, then prints D(k), the speed-up estimates and a verdict |
| [make_random_head.py](make_random_head.py) | Writes a random-weight EAGLE-3 checkpoint with a target's shapes (Hugging Face format) |
| [TRAINING_PLAN.md](TRAINING_PLAN.md) | How to train a real head on the quant machine if the check passes |

Both scripts run on the rig through `ssh rig "... bash -s" < script`, write to `/tmp/eagle3_flint/<TAG>/`, always
pass the 4 GB-host settings (`-ub 128`, `--cache-ram 0`, `--ctx-checkpoints 0`), use backend sampling (`-bs`) like
the [real-chat benchmarks](../../docs/benchmarks/2026-10-22-speedups.md), and kill llama-server if free host memory
drops below 400 MiB. They default to the CUDA-graphs build
`/opt/llama-builds/llama.cpp-20261022-d77256999-graphs61/bin` (set `BIN` for another) and all nine cards.

## The random heads

Four heads, each a Q8_0 GGUF of a one-layer EAGLE-3 drafter with Flint's attention shape (hidden size 2048, 32 query
heads and 4 key-value heads of 128 dimensions) and random weights:

| GGUF | Draft vocabulary | FFN width | Parameters | Size | Question it answers |
|---|---|---|---|---|---|
| `flint-eagle3-rand-v32k-Q8_0.gguf` | 32,000 tokens + `d2t` map | 6,144 | 145 M | 160 MB | **The main candidate**: draft cost at k = 1–4 |
| `flint-eagle3-rand-v150k-Q8_0.gguf` | all 150,144 tokens, own head | 6,144 | 387 M | 417 MB | What the reduced vocabulary saves |
| `flint-eagle3-rand-nohead-Q8_0.gguf` | all 150,144, borrows Flint's own output head | 6,144 | 80 M | 90 MB | Drafting through the target's 4-bit head instead of a trained one |
| `flint-eagle3-rand-v32k-ffn12288-Q8_0.gguf` | 32,000 + `d2t` | 12,288 | 183 M | 200 MB | The cost of SpecForge's wider FFN for MoE targets (its Qwen3-30B-A3B config) |

A 32k draft vocabulary is the usual choice (SpecForge, vLLM speculators): the head drafts only the 32,000 most
frequent tokens, and llama.cpp maps them back to Flint's ids through the `d2t` table, so the drafter reads a 70 MB
head per draft token instead of 327 MB. Flint's token embedding is not part of any head: llama.cpp borrows the
target's, which it keeps in host memory.

To rebuild them (any machine; nothing here needs a GPU):

```bash
# 1. Flint's config and tokenizer only (a few MB; never the weights)
mkdir flint-meta && cd flint-meta
for f in config.json tokenizer.json tokenizer_config.json generation_config.json; do
    curl -sfLO "https://huggingface.co/RecursiveSI/flint/resolve/main/$f"; done
cd ..

# 2. random checkpoints (torch-cpu + safetensors through uv; no system pip)
UV="uv run --no-project --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match"
$UV --with torch==2.11.0 --with safetensors python make_random_head.py --target-dir flint-meta --out rand-v32k --draft-vocab 32000
$UV --with torch==2.11.0 --with safetensors python make_random_head.py --target-dir flint-meta --out rand-v150k --draft-vocab 0
$UV --with torch==2.11.0 --with safetensors python make_random_head.py --target-dir flint-meta --out rand-nohead --no-lm-head
$UV --with torch==2.11.0 --with safetensors python make_random_head.py --target-dir flint-meta --out rand-v32k-ffn12288 \
    --draft-vocab 32000 --intermediate-size 12288

# 3. GGUF with llama.cpp's converter (commit d77256999 or later); the tokenizer comes from the target folder
for v in v32k v150k nohead v32k-ffn12288; do
    $UV --with torch==2.11.0 --with numpy~=2.2.6 --with transformers==4.57.6 --with sentencepiece --with protobuf \
        --with safetensors --with pyyaml --with tqdm --with requests \
        python <llama.cpp>/convert_hf_to_gguf.py rand-$v --target-model-dir flint-meta --outtype q8_0 \
        --outfile flint-eagle3-rand-$v-Q8_0.gguf
done
```

The converter recognises the checkpoint as EAGLE-3 because its config has `draft_vocab_size` and one layer, takes the
tokenizer from `--target-model-dir`, and records the target layers [2, 24, 46] (its default `[2, L/2, L−3]` for
Flint's 49 layers; llama.cpp reads the *inputs* of those layers).

## Running the check on the rig, in order (about an hour)

Each llama-server start reads the whole 26 GB Flint file from the USB stick (~1–2 minutes), and the server's
speculative settings can't be changed per request in this llama.cpp version, so every measured configuration costs one
model load. Use the Flint file you serve (below: the makers' Q4_1).

**0. Copy the heads (2 minutes).** About 870 MB in total, e.g. to `/mnt/models/eagle3/` (`rsync` over SSH takes under
a minute; the 160 MB 32k head alone is enough for steps 1–2).

**1. Verify cost (25–30 minutes, 8 model loads).**

```bash
ssh rig "MODEL=/mnt/models/RecursiveSI/flint-int4-gguf/flint-int4-q4_1.gguf bash -s" < bench/eagle3_flint/verify_cost.sh
```

- `bench`: `llama-bench -p 1,2,…,8 -n 64 -r 5`, the time of a step that processes n tokens.
- `synth`: llama-server once per width n = 1…6. Drafts come from n-gram lookup (free) and `--spec-synth-len n` accepts
  all of them, so each step verifies up to n tokens through the real speculative path with backend sampling.
- `ngram`: real n-gram drafting on a prompt whose answer repeats the input (adding type hints to a Python module).

It prints ms per step against the tokens actually verified per step, and the ratio to plain generation. The plain
server step is **T1**.

**2. Draft cost (12–15 minutes, 4 loads).**

```bash
ssh rig "MODEL=/mnt/models/RecursiveSI/flint-int4-gguf/flint-int4-q4_1.gguf \
    HEADS=/mnt/models/eagle3/flint-eagle3-rand-v32k-Q8_0.gguf bash -s" < bench/eagle3_flint/draft_cost.sh
```

For k = 1…4 it runs `--spec-type draft-eagle3 --spec-draft-n-max k --spec-draft-p-min 0` with the head on the last
card (`-devd CUDA8`, the card that also holds Flint's output head and last layers) and `--spec-synth-len k+1`, so every
round drafts k tokens and verifies k + 1, exactly as with a perfect head. The round time does not depend on how many
drafts are accepted, so this is the cost of a round for any trained head of this shape. At log level 4 llama.cpp also
reports the time spent inside the drafter's `draft()` call, which covers the k draft passes but not the encoder and
cache-filling passes, so round(k) − draft() is an upper bound on v(k + 1) and cross-checks step 1. The script then
reads step 1's results and prints D(k), the speed-up estimates of the next section, and a verdict.

**3. Head variants, if time allows (8–10 minutes, 3 loads).**

```bash
ssh rig "MODEL=... TAG=draft_variants DRAFT_N=3 HEADS='/mnt/models/eagle3/flint-eagle3-rand-v150k-Q8_0.gguf \
    /mnt/models/eagle3/flint-eagle3-rand-nohead-Q8_0.gguf /mnt/models/eagle3/flint-eagle3-rand-v32k-ffn12288-Q8_0.gguf' \
    bash -s" < bench/eagle3_flint/draft_cost.sh
```

The head without its own output matrix uses Flint's, so it has to run on the card that holds that matrix: keep
`DRAFT_DEV` at the last card.

**4. Decide (5 minutes)**, below. Copy `/tmp/eagle3_flint/` off the rig before any reboot.

## Deciding

Definitions (all in milliseconds, from the two scripts):

- **T1**: one plain decode step (server, `-bs`), i.e. ms per token without drafting.
- **v(n)**: one step that verifies n tokens.
- **round(k)**: one speculative round with k drafts, measured by `draft_cost.sh`; **D(k) = round(k) − v(k + 1)** is the
  drafter's share (encoder, catch-up pass and k draft passes).
- **τ(k)**: tokens a round produces on average (accepted drafts plus the target's own token), at most k + 1.

Speed-up over plain generation:

```text
S(k) = τ(k) · T1 / round(k) = τ(k) · T1 / (v(k+1) + D(k))
break-even acceptance:  τ* = round(k) / T1
```

**Planning values for τ(k)**, from published chain-drafting EAGLE-3 heads on Mixture-of-Experts targets (medians over
each model card's task rows):

| k | [Qwen3-30B-A3B head](https://huggingface.co/RedHatAI/Qwen3-30B-A3B-speculator.eagle3) | [gpt-oss-20b head](https://huggingface.co/RedHatAI/gpt-oss-20b-speculator.eagle3) | τ mid (mean of the two) | τ conservative (0.85 × mid) |
|---|---|---|---|---|
| 1 | 1.74 | 1.64 | 1.69 | 1.44 |
| 2 | 2.25 | 2.03 | 2.14 | 1.82 |
| 3 | 2.60 | 2.26 | 2.43 | 2.07 |
| 4 | 2.82 | 2.40 | 2.61 | 2.22 |

Both heads were trained on 300k+ conversations by their publishers and measured in vLLM. A first Flint head trained
on ~60k conversations, and used at Flint's recommended temperature 1.0, should be expected somewhat lower; the
conservative column allows 15 % for that. The Qwen3-30B-A3B column doubles as the optimistic case.

**Fill in** (draft_cost.sh prints the same table):

| k | v(k+1) | D(k) | round(k) | τ* = round/T1 | S mid = τ mid · T1 / round | S conservative |
|---|---|---|---|---|---|---|
| 1 | | | | | | |
| 2 | | | | | | |
| 3 | | | | | | |
| 4 | | | | | | |

**Expected values** (arithmetic from the earlier rig measurements, T1 = 41.2 ms at 24.3 tokens/s served, to be
replaced by the measurement):

| k | v(k+1) / T1 | D(k), 32k head | round(k) | τ* | S mid |
|---|---|---|---|---|---|
| 1 | 1.03–1.13 | 4–7 ms | 46–54 ms | 1.13–1.30 | 1.30–1.50× |
| 2 | 1.10–1.25 | 6–10 ms | 51–62 ms | 1.25–1.49 | 1.43–1.72× |
| 3 | 1.20–1.40 | 8–13 ms | 57–71 ms | 1.39–1.72 | 1.42–1.74× |
| 4 | 1.30–1.55 | 10–16 ms | 64–80 ms | 1.54–1.94 | 1.35–1.69× |

D(k) here is bytes plus call overhead. In llama.cpp's EAGLE-3 loop a round of k drafts runs k + 2 drafter passes: an
encoder pass over the k + 1 verified tokens, a pass that writes their keys and values into the drafter's cache, and k
single-token draft passes. The 71 MB decoder layer is therefore read k + 1 times and the 70 MB head k times
(~225 / 367 / 507 / 648 MB for k = 1…4, 2.5–7.2 ms at the ~90 GB/s these cards reach with Q8_0), plus an estimated
0.5–1.5 ms of launch and host overhead per pass on the 2-core host.

**The rule:**

- **GO** (start [TRAINING_PLAN.md](TRAINING_PLAN.md), pilot first) if the best k reaches **S mid ≥ 1.4×** and
  **S conservative ≥ 1.15×** there. A drafter must clearly beat the cheaper trained lever, partial quantization-aware
  training of the per-token tensors (+10–20 % predicted), and must not lose speed even if the head turns out weak.
- **NO-GO** if **S mid < 1.2×** for every k. Below that, sampling at temperature 1.0 and the requests where backend
  sampling switches off (tool calls, see below) would erase the gain.
- **MARGINAL** in between: don't train yet. Look at which side is expensive. If D(k) is large, step 3 shows whether a
  smaller head would fix it (the 32k vocabulary, a Q4_0 head, a narrower FFN). If v(n) grows steeply (v(4) > 1.5·T1),
  verification reads extra experts on these cards too, and only k = 1–2 can work.

Two further readings:

- **Faster plain decoding shrinks the gain.** D(k) stays the same when T1 drops (a quant without the MoE rescale, a
  QAT file), so S falls. Recompute with the T1 of the file you will actually serve.
- **Tool-call requests.** llama-server turns backend sampling off for requests that carry tools. Each verify step then
  copies k + 1 rows of 150,144 logits (600 KB each) over the last card's PCIe 1.0 x1 link (~3 ms per row) and
  samples them on the CPU, which costs roughly 20–25 ms per k = 3 round. For agent traffic S is therefore lower than
  the table's; measure it once a real head exists.

### Why these methods and not others

- **`llama-batched-bench`** (`-npp/-ntg/-npl`) runs n *different* sequences per step and copies every output row's
  logits to the host. For Flint that is 600 KB per row over the x1 link, ~3 ms each, which a `-bs` verify step never
  pays, so it overstates v(n) by ~3 ms per verified token.
- **`llama-bench` ppN** processes n tokens of one sequence but outputs one row and samples nothing; on the dense
  Qwen3.8-27B it overstated the real verify cost ([speed-ups](../../docs/benchmarks/2026-10-22-speedups.md)). It is
  kept for its smooth n = 1…8 curve.
- **Synthetic acceptance** (`--spec-synth-len L`, llama-server) accepts each draft with a fixed probability instead of
  comparing it with the target's choice. With L = k + 1 every draft is accepted, so the step width is fixed and the
  measured time is the true cost of that width through the real server path. It replaces verification entirely and
  the generated text is meaningless.
- **n-gram drafting** costs nothing to draft, so with synthetic acceptance it isolates v(n). Its drafts are shorter
  than requested when the matching token occurred recently, which is why the scripts report the tokens actually
  verified per step and `draft_cost.sh` interpolates v between the measured widths.
