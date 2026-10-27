# Ornith-1.5-122B: quality of the rig-sized files against Q8_0, on prose, agent sessions and code (2026-10-27)

Only 2–3-bit files of Ornith-1.5-122B fit nine 6 GB cards. This measures how far the candidates drift from the makers'
Q8_0. The BF16 model (244 GB) does not fit the quality machine's 158 GB of VRAM, so the 132.6 GB Q8_0 is the reference.

The files:
- **AD-IQ3_XXS-IQ2_S** (AtomicChat, 47.6 GB);
- **UD-Q2_K_XL** (Unsloth, 44.5 GB);
- **N2P-N3P-Q5_0dense** (45.4 GB, 3.05 bits per weight): file c of the
  [Pascal low-bit format](../../research/notes/pascal_lowbit_format.md), Miner Llama's own N2P (expert gate/up) and N3P
  (expert down) types, the dense tensors at Q5_0, quantized from the Q8_0 with Unsloth's imatrix
  ([its data](../2026-10-27_lowbit/README.md)).

The texts:
- **wiki:** the wikitext-2 test file, 80 chunks of 512 tokens (this repository's "within noise" test);
- **agent:** 12 chunks of 8,192 tokens of 40 Flint-written agent sessions held out from training
  ([2026-10-27_mtp_head](../2026-10-27_mtp_head/README.md), [bench/agentic_data](../../../bench/agentic_data/README.md));
- **code:** 80 chunks of 512 tokens of the agent check's Exercism material: each task's reference solution and hidden
  test file, Python and JavaScript.

The speed side of the same files is the [Ornith-1.5-122B benchmark](../../benchmarks/2026-10-27-ornith-122b.md).

## Results

Against Ornith-1.5-122B Q8_0 (mean KL divergence ± standard error / same top token):

| File | wiki | agent | code |
|---|---|---|---|
| UD-Q2_K_XL | **0.184 ± 0.003 / 83.5 %** | **0.054 ± 0.001 / 94.9 %** | **0.071 ± 0.003 / 96.3 %** |
| AD-IQ3_XXS-IQ2_S | 0.233 ± 0.003 / 81.4 % | 0.066 ± 0.001 / 94.3 % | 0.081 ± 0.003 / 96.2 % |
| N2P-N3P-Q5_0dense | 0.203 ± 0.003 / 82.8 % | 0.057 ± 0.001 / 94.7 % | 0.072 ± 0.003 / 96.4 % |

- **UD-Q2_K_XL is closer to Q8_0 on all three texts.** Of the two downloaded files it is also the smaller one and decodes
  ~10 % faster on the rig, so it is the better candidate.
- **N2P-N3P-Q5_0dense sits between the two:** within noise of UD on code, about two standard errors behind it on the
  agent sessions, and clearly behind on prose, for 0.9 GB more. Its case is speed: its Pascal kernels are estimated to
  read an expert weight ~1.8× faster than the IQ2_S/IQ3_XXS kernels, which only a rig run can confirm.
- **Prose shows the largest drift.** Agent sessions and code are far more predictable: the Q8_0's perplexity is 6.25 on
  wikitext, 1.57 on the agent sessions and 1.28 on code (`bases.tsv`).
- **Perplexity:** UD's falls slightly below the reference's on all three texts (`kld_122b.tsv`, `ppl_q` against
  `ppl_base`). That is noise around a reference that is itself quantized, not a gain. AD's and N2P-N3P's rise above it.
- **Across models:** these numbers measure each file against Ornith's own Q8_0 and do not compare with Flint's KL
  divergences against Flint's BF16.

Pending, added here when they finish:
- AD and UD with a Q5_0 output layer (`run/head_q5_0.py`);
- the low-bit format's file b (N2P/N3P experts, dense at Q8_0 as in AD);
- the Ornith-1.5-35B Q8_0's bases on the same texts.

## Files

| File | What |
|---|---|
| `kld_122b.tsv` | per file and text: perplexity of the file and of the reference, mean KLD ± SE, median, 99th percentile and maximum KLD, same top token ± SE, RMS of the probability change |
| `bases.tsv` | the Q8_0's perplexity on each text (chunks, context) |
| `run/kld_texts.sh` | the bases and the scores on CUDA GPUs (llama-perplexity in Docker), with the seven-card split the 122B Q8_0 needs |
| `run/agent_text.py` | the agent text from the held-out sessions of the agentic data export |
| `run/code_text.py` | the code text from `bench/agent_check/tasks/polyglot` |
| `run/head_q5_0.py` | the Q8_0's output layer requantized to Q5_0 (524 MB instead of 811 MB, relative rms error 0.044), as a one-tensor GGUF for `scripts/compose_gguf.py` |

## Made with

```bash
python3 -I run/agent_text.py <export>/hf/data/sessions.jsonl <work>/seqs/test-rows.jsonl agent-heldout.txt
python3 -I run/code_text.py bench/agent_check/tasks/polyglot code-exercism.txt
BIN=<llama.cpp>/build-cuda/bin/llama-perplexity OUT=<folder> GPUS=0,1,2,3,4,5,6 TS=2.5,3,7,11,11,8,7.5 \
  WIKI=wikitext-2-raw/wiki.test.raw AGENT=agent-heldout.txt CODE=code-exercism.txt \
  bash run/kld_texts.sh Ornith-1.5-122B-Q8_0.gguf Ornith-1.5-122B-AD-IQ3_XXS-IQ2_S.gguf Ornith-1.5-122B-UD-Q2_K_XL.gguf
# a file in Miner Llama's own types names its build: ... file.gguf=<that build>/bin/llama-perplexity
python run/head_q5_0.py Ornith-1.5-122B-Q8_0.gguf output-q5_0.gguf          # needs numpy and gguf-py
python scripts/compose_gguf.py Ornith-1.5-122B-UD-Q2_K_XL.gguf UD-headQ5_0.gguf '^output\.weight$=output-q5_0.gguf'
```

- **The machine:** the quality machine's 3 × RTX 3090, 2 × V100 32 GB and 2 × RTX 3080, with the box's llama.cpp build
  (upstream `d77256999` with local changes, which runs `qwen35moe`).
- **Run time:** each Q8_0 base took ~25 minutes, most of it reading 132.6 GB from a hard disk. Times on a shared machine
  are not speed results.
- **The split:** the default split by free memory and `-ts 3,3,7,11,11,7,6` both put four layers (10.2 GB) on the
  10 GB card; `run/kld_texts.sh` explains why.

## Provenance and licence

- **Models:** Ornith-1.5-122B by ornith-ai; the GGUFs are the makers' Q8_0, AtomicChat's AD file and Unsloth's UD file,
  all from Hugging Face at pinned revisions.
- **Texts:** wikitext-2 (CC BY-SA 3.0, used for measurement only, not republished here); the agent sessions (Flint's
  output, Apache-2.0, on the Python standard library and Elixir's library, gated by `bench/agentic_data`); the Exercism
  tasks (MIT, licences in `bench/agent_check/tasks/polyglot/`).
- **This folder:** the tables and scripts are under this repository's licence.
