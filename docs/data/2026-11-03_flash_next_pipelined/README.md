# Q550: pipelined speculation for Qwen3.8-Flash-Next, a recurrent-state group per pipe slot (2026-11-03)

On a layer split one card works at a time: Flash-Next's decode passes nine P106 cards in turn, each busy for 12–18 ms
of a token's 136 ms of GPU time ([Q215](../2026-11-03_flash_next_baseline/README.md)), so eight cards sit idle.
Pipelined speculation ([pipelined_speculation.md](../../research/notes/pipelined_speculation.md)) sends guessed tokens
into the first cards while the later ones still work on the current token, and keeps a guess only when the token
sampled from the real pass equals it. A wrong guess is dropped: its KV cells are removed. Flash-Next is a hybrid model,
36 of its 48 layers Gated DeltaNet layers, whose recurrent state cannot drop a token once it has absorbed it, so
pipelined decoding refused it until now. This run ports Q162's fix (built for Qwen3.6 on a v2 branch) to v3 and extends
it to Flash-Next's own state rows, checks that pipelined decoding then gives plain decoding's output exactly, and
sizes the memory and the expected gain on the rig. The registry row is Q550 in
[quants/experiments.md](../../../quants/experiments.md).

## Result

### The engine: fork branch `pascal-p106-v3-hybrid`

Branch `pascal-p106-v3-hybrid` of the fork ([Cinderella-Man/llama.cpp](https://github.com/Cinderella-Man/llama.cpp)),
`f377294d6`, off `pascal-p106-v3` (`37e717ff6`, the v2.0 commit). Its seven commits:

- **Q162's state groups, rebased** (five commits from `pascal-p106-v2-hybrid`; its copy of the card-built mask was
  already in v3): `llama_pipe_init` gives the recurrent memory one group of state rows per pipe slot plus a stash
  group. Each slot writes its tokens' states to its own group, at an offset fixed in the slot's graph (so graph reuse,
  CUDA graphs and the fused state write stay); every evaluation reads its source state from whichever group holds it.
  Per row the memory records which position's state each group holds, so dropping a wrong guess
  (`seq_rm(seq, p, -1)`) moves the sequence back to the group that holds the state at p − 1. The one state a rollback
  may still need, before the oldest token not yet read, is copied to the stash before a slot overwrites it. The
  hybrid memory's attention part can build its mask on the cards.
- **Flash-Next (`qwen4exp`)**: it keeps three state rows per Gated DeltaNet layer: the delta-net state, written by the
  shared delta-net code that already wrote its slot's group, and two convolution histories that the model writes with
  its own code, the delta-net's conv state and the n-gram embedding's conv history (layer 1). Both now go to the
  ubatch's group too. Its memory type (`llama_memory_hybrid_idx`, the hybrid memory plus the sparse-attention
  indexer's cache) is a hybrid memory, so the groups reach its recurrent part; the indexer cache is a KV cache and
  drops a guess's cells like the attention cache. `llama_pipe_init` now accepts `qwen4exp`.
- The stash copy runs on the slot's own backend (with `LLAMA_PIPE_CHUNK_STREAM` a prompt-chunk slot has streams of its
  own), and `llama_pipe_init` logs the pipe slots' compute buffers per device.

Recurrent models whose state writes do not know the groups are still refused, and `llama-server --pipe-depth` needs
`n_rs_seq 0`, so it does not combine with `--spec-type draft-mtp` (which keeps rollback snapshots instead).

### Pipelined output = plain output

`bench/pipespec/check_identity.sh` compares every pipelined run with plain decoding of the same prompt: the generated
tokens, every token's logits (a hash per token), the sequence's state with its recurrent part after generation, and the
logits of probe tokens decoded afterwards. `llama-pipe-recurrent-check` forces the rollbacks the executors rarely make:
tokens submitted to the slot of the token before them (the stash copy), dropped guesses, and the token in flight
dropped as llama-server's `pipe_leave` does, then submitted again or decoded with `llama_decode`. Per run in
[checks.tsv](checks.tsv):

| Model, devices | Pipelined runs | Tokens | Guesses | Wrong | Dropped | Verdict |
|---|---:|---:|---:|---:|---:|---|
| Random `qwen4exp`, CPU, greedy, depths 1–4 | 29 + recurrent check | 3,712 | 3,522 | 604 | 1,510 | all identical |
| Random `qwen4exp`, 2 virtual devices on one GPU (delta-net layer / attention layer), greedy, depths 1–4 | 29 | 3,712 | 3,522 | 604 | 1,510 | all identical |
| Same, T 1 with a seed, depths 1 and 3 | 15 + recurrent check | 1,920 | 1,419 | 392 | 783 | all identical |
| **Negative control:** the same branch with the two conv rows written to group 0 again, CPU | 29 | 3,712 | 5,022 | 1,462 | 3,592 | **all differ** |

- The guessers: an oracle replaying the plain run with 0, 10 or 50 % of its guesses made wrong, the copy lookup
  (match ≥ 2), copy ≥ 4 else the request's continuation counts (llama-server's `cont`), `ngram-simple` and
  `ngram-map-k`, plus depth 0 (the pipelined calls with no guess).
- **The random model** comes from llama.cpp's `test-llama-archs -a qwen4exp` (seed 1): two layers, a Gated DeltaNet
  layer that carries the n-gram embedding and its conv history and an attention layer with the sparse-attention
  indexer, hyper-connections, two routed experts and a shared one, a 128-token test vocabulary. It exercises every state row Flash-Next has.
- **The negative control** shows the checks catch a state the groups do not keep: with only the conv rows left in
  group 0, every pipelined run differs in tokens, logits, state and probe, depth 0 included (the next evaluation reads
  the conv rows from a group the slot never wrote).
- **Not run yet: the real file.** Swift-1.5 GSQ-RCO Q2_0 (48 layers) on two RTX 3090s and on four virtual devices
  over them, and the same check on the rig's P106s before any timing (the first step of Q551's speed ladder).

### Memory per depth on the rig

`llama-server --pipe-depth D` makes 2 (D + 1) one-token pipe slots, so the recurrent memory holds 2 D + 3 state groups
where plain decoding holds one. One group of Swift Q2_0 is 112.6 MiB over the nine cards with Q215's split
(`-ts 6,5,5,6,5,5,6,5,6`): 15.9, 12.5, 9.4, 15.6, 12.5, 9.4, 15.6, 12.5 and 9.4 MiB on cards 0–8 (3.12 MiB per
delta-net layer; card 0 also holds the n-gram conv history), the 112.6 MiB of a context checkpoint. What the extra
groups take (exact, from Q215's load log):

| Depth | Pipe slots | State groups | Extra on the cards | On card 6 (fullest: 735–811 MiB free at 64k, `-ub 512`) | On card 0 (869 MiB free) |
|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 5 | 450 MiB | 62 MiB | 64 MiB |
| 2 | 6 | 7 | 676 MiB | 94 MiB | 96 MiB |
| 3 | 8 | 9 | 901 MiB | 125 MiB | 128 MiB |
| 4 | 10 | 11 | 1,126 MiB | 156 MiB | 159 MiB |
| 6 | 14 | 15 | 1,576 MiB | 218 MiB | 223 MiB |

- The free memory is Q213's projection (735 MiB on card 6) and Q215's reading after the load (6,144 MiB less the 5,333
  and 5,275 MiB in use on cards 6 and 0).
- **The pipe slots' own compute buffers** (one-token graphs over the whole 64k context, per slot) come on top and are
  not measured yet. llama.cpp's `--fit` projection does not include pipelining, so `run/memory.sh` loads the model at
  each depth on the rig's nine cards (header-only sparse copies of the shards give the same buffers and load faster)
  and reads the state groups and the slots' buffers per card from `llama_pipe_init`'s log, nvidia-smi's use per card,
  and whether pipelining came up or failed to allocate. The host gets one pinned logits buffer per slot (0.95 MiB: 7.6 MiB at depth 3)
  plus the slots' host-side buffers, on a host that had 230–460 MiB available during Q215 (Q549).

### Expected gain: est. 1.3× for agent sessions at depth 3

[bench/ngram_replay/pipeline_sim.py](../../../bench/ngram_replay/pipeline_sim.py)'s event model, fed with Flash-Next's
per-card GPU time from Q215's profile (15.6 / 14.8 / 14.0 / 18.1 / 14.3 / 15.0 / 16.2 / 12.0 / 16.1 ms, scaled to a
plain token of 112, 140 or 185 ms), 0.6 ms per hop between cards and 3 ms of host time per sample, replays Q51's 38
Flint transcripts (chat, code, edits and agent sessions; 65,558 generated tokens) with llama-server's guessers.
Speed-up over plain decoding (140 ms; 112 and 185 ms give the same within 0.01), from [sim_speedups.md](sim_speedups.md):

| Guesser | Depth 1 | 2 | 3 | 4 | 6 |
|---|---:|---:|---:|---:|---:|
| `cont` (copy ≥ 4, else the request's counts), agent sessions | 1.22 | 1.29 | **1.31** | 1.33 | 1.34 |
| `cont`, all transcripts | 1.29 | 1.40 | **1.45** | 1.48 | 1.51 |
| copy ≥ 4, else request counts and a store of earlier answers (`--pipe-store`), agent | 1.26 | 1.35 | 1.38 | 1.40 | 1.42 |
| a guesser right 70 % of the time (an MTP head's range), all | 1.67 | 2.06 | 2.30 | 2.43 | 2.54 |

- The guess rates are Flint's, on its own transcripts: Flash-Next's output and tokenizer differ, so this is an
  estimate. On six P106 cards the same model predicted Flint's 1.44× where the rig measured 1.3–1.55×, and llama-server
  served Flint 39.2 → 48.5 tok/s at depth 3 (1.24×, Q160).
- The cards hold about 7.5 tokens at once (136 ms over a slowest stage of 18 ms), so depth is not what limits the
  gain: the guesses are. Edits and file rewrites, mostly copied, gain 1.9–3.9× at depth 3; fresh text gains little.
- A drafter as good as an MTP head would more than double the gain over `cont` (Q554): the pipeline runs its guesses
  on the idle cards instead of in a verify batch.
- The agent check sends only temperature, top-p, top-k, min-p and a seed, so its requests qualify (llama-server
  pipelines a request without penalties or token probabilities).

## Files

| File | What |
|---|---|
| `checks.tsv` | every check: label (model and devices), check, run (job name: guesser, depth, share made wrong), generated tokens, guesses, right, wrong, dropped tokens, same/DIFF for tokens, logits, state and probe, the verdict; `llama-pipe-recurrent-check` as one row per run |
| `sim_speedups.md` | the event model's full table: per setup, guesser and depth the speed-up per prompt class (chat, code, edits, agent, all, file rewrites), the right and wrong guess shares and the card time spent on dropped guesses |
| `run/build_test.sh` | the test build (RTX 3090 code plus compute 6.1 PTX) with the check tools and `test-llama-archs` |
| `run/build_negative.sh`, `run/neg_qwen4exp_conv_rows.diff` | the negative control: the test build with the conv-row change reversed |
| `run/checks.sh` | the checks: the random model (written by `test-llama-archs`) on the CPU and on 2 virtual devices, the real file on the CPU, on the GPUs and on 4 virtual devices; `check_identity.sh` and `llama-pipe-recurrent-check` |
| `run/checks_table.py` | `checks.tsv` from the checks' output folders |
| `run/memory.sh`, `run/memory_table.py` | llama-server loaded at each depth with the rig's split, stopped once it listens; per device the model, KV, state-group and compute buffers, the pipe slots' buffers and nvidia-smi's use, whether pipelining came up, and the host's memory |
| `run/sim.sh` | the event-model replay |

## How it was made

On 2026-11-03, on the 7-GPU quant machine (correctness and memory only; nothing here was timed). The fork branch
`pascal-p106-v3-hybrid` at `f377294d6`, built in `nvidia/cuda:12.9.2-devel-ubuntu24.04` by `run/build_test.sh` from a
`git archive` of the branch next to it. The checks ran in the same image with the build's `bin/` and its CUDA 12
libraries on `LD_LIBRARY_PATH`; the CPU runs had no GPU in the container and the toolkit's stub `libcuda.so` as
`libcuda.so.1`.

```bash
# the random model and the checks (BIN: the build's bin/; PS: bench/pipespec of this repository)
BIN=<bin> PS=bench/pipespec OUT=<out>/cpu STEPS='rand-cpu rand-prc-cpu' \
  bash docs/data/2026-11-03_flash_next_pipelined/run/checks.sh            # no GPU: -ngl 0, 8 threads
BIN=<bin> PS=bench/pipespec OUT=<out>/rand STEPS='rand-gpu rand-seeded rand-prc-gpu' \
  bash docs/data/2026-11-03_flash_next_pipelined/run/checks.sh            # one GPU, GGML_CUDA_DEVICES=2 inside
# the negative control: run/build_negative.sh, then the CPU check with its binary and the same random model
BIN=<bin-neg> RAND=<out>/cpu/models/qwen4exp-moe.gguf PS=bench/pipespec OUT=<out>/neg STEPS='rand-cpu' \
  bash docs/data/2026-11-03_flash_next_pipelined/run/checks.sh
python3 -I docs/data/2026-11-03_flash_next_pipelined/run/checks_table.py checks.tsv \
  random-cpu=<out>/cpu random-2vdev=<out>/rand negative-cpu=<out>/neg
# the event model on Q51's transcripts (release/flint-v2/reproduce/data/q51)
SIM=bench/ngram_replay GEN=<q51 gen.jsonl> OUT=<out>/sim bash docs/data/2026-11-03_flash_next_pipelined/run/sim.sh
```

The real file: Swift-1.5 GSQ-RCO Q2_0 as three shards (the n-gram table alone in the third,
[Q435](../2026-11-03_ssd_room/README.md)), from
[ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
`99bb8f7f95c7aa7b24a36a7786a4f657b30f5d3d` (Swift Open License 1.0 and qwen-community-1.0). The prompts are
`bench/pipespec/prompts/` of this repository. The rig tarball of the branch, built by
[scripts/build_llamacpp_docker.sh](../../../scripts/build_llamacpp_docker.sh) `fork/pascal-p106-v3-hybrid` with
`TESTS=1`, is `llama.cpp-20261103-f377294d6-pascal-p106-v3-hybrid.tar.gz` (293,541,289 bytes, md5
`8e53e99a8293d18bc1c8df37ad2c0815`).
