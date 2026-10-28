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
`--spec-draft-n-max 3` does) are not measured here.

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

## What it means for the engine

Speculation with the MTP layer pays when a verification step, the model on the sampled token plus the draft, costs
much less than two plain steps. With one draft accepted ~0.85–0.9 of the time, a step yields ~1.85–1.9 tokens. On the
rig it does not pay yet: v2.0 with the MTP head ran 7.9 against 8.8 tok/s on the 122B's agent check
([the benchmark](../../benchmarks/2026-10-27-ornith-122b.md)), and the Pascal IQ2_S/IQ3_XXS expert kernels run 2–4
tokens at 0.5–0.8× llama.cpp's speed, a 2-token expert product at 3–4× the cost of one token
([2026-10-27_iq_kernels](../2026-10-27_iq_kernels/README.md)).

- **First: expert kernels for 2–8 tokens** (registry Q211). With a 2-token verification that reads each weight once,
  est. 1.3× decode on the 35B (P-Q5: 2.18 against 2.82 GB per token, the draft's MTP layer and output head included)
  and ~1.2× on the 122B (UD-Q2_K_XL: ~82 against ~99 ms of GPU time per token). This needs no new speculation code:
  llama.cpp's `draft-mtp` already drafts with the layer and rolls the recurrent state back.
- **Then: the recurrent state by accepted-path replay** (QUEUES R8, Q162), so that the MTP layer can guess in Miner
  Llama's pipelined executor (Q341), which fills the cards that sit idle. It rests on the same multi-token kernels.
- **The draft's own cost:** the output head (248,320 tokens: 0.54 GB in the 35B's Q8_0, 0.63 GB in UD-Q2_K_XL) is most
  of one MTP draft's bytes; Flint's screened head (Q91, Q92) cuts that cost.

## Files

| File | What |
|---|---|
| `35b-q8_0-{agent,code}.tsv`, `122b-ud-{agent,code}.tsv`, `122b-ad-{agent,code}.tsv` | per run: positions, share, acceptance at T 1, T 0.6 and under the makers' chain, top-1 agreement, each top-1's hit rate on the text, mean top probability; all positions and per bin of the model's top probability |
| `*-chunks.tsv` | the four acceptance numbers per chunk of each run |
| `run/mtp_accept.cpp` | the measurement (C++ against llama.cpp's API: `llama_process`, `common_batch::set_embd_state`, `llama_get_embeddings_nextn_ith`) |
| `run/build_tool.sh` | its build against an existing llama.cpp CUDA build, in the CUDA image |
| `run/run_accept.sh` | one run on CUDA GPUs in Docker |

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
