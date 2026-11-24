# Q1723: does a Q8_0 cache keep Ornith-1.5-122B's quality at 65,536 and 98,304 tokens? (2026-11-24)

A Q8_0 attention cache is half the size of Ornith's F16 cache, so it allows a longer context on the same cards. Up to
49,152 tokens its quality was within noise of the F16 cache
([2026-11-23_c976_ornith_q8_cache_kld](../2026-11-23_c976_ornith_q8_cache_kld/README.md), Q1707), with one caveat: past
8,192 tokens its perplexity ran a few tenths of a percent higher on average (+0.28 % on six stretches of 16,384 tokens,
+0.35 % on four of 32,768), which that check could neither show nor rule out. With pull request 9's decode kernel the
cache was a gain on agentic speed on the rig at 49,152 tokens ([2026-11-24_r232b_ornith_q8_agent](../2026-11-24_r232b_ornith_q8_agent/README.md)).
Before the rig tries a longer context ([Q1807](../../../quants/experiments.md)), this folder checks the cache at
65,536 and 98,304 tokens: does the perplexity rise stay where C976 put it, or grow with the context? It runs the same
comparison as C976 (Ornith-1.5-122B Pascal N2P GGUF with `-ctk q8_0 -ctv q8_0` against the same file with its F16
cache, on held-out agent text, through the rig build's compute 6.1 PTX on three RTX 3090s), through the Pascal prompt
kernel and through the decode kernel. It is a correctness and quality check only: no time on these cards means
anything for a P106.

**Result: pending** (the run started 2026-11-24 20:47).

## What ran

| | |
|---|---|
| Build | Miner Llama `pascal-p106-v5` at `af00bb13b`, which holds pull request 9's decode kernel (rig tarball `llama.cpp-20261124-af00bb13b-pascal-p106-v5-land-batch2.tar.gz`, md5 `b51ad2740d5822e7e14af656c413ec22`; its `llama-perplexity` md5 `1c967216e551e89cff64fb769bf1de83`). Its GPU code is sm_61 machine code and compute 6.1 PTX only, so a newer card runs the PTX |
| Model | Ornith-1.5-122B Pascal N2P GGUF (`valid1`: `Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf`, 48,196,722,624 bytes), read from a hard disk |
| Text | the agent text of [2026-10-27_ornith_kld](../2026-10-27_ornith_kld/README.md) (`agent-heldout.txt`, 4,157,046 B, md5 `f25e8e0ba6688053522b8f1b486aedf5`), as in C976 |
| Depths | 98,304 tokens (run first: its reference run is the fit check) and 65,536 tokens, three chunks each (`-c D --chunks 3`): each chunk is its own stretch of the text, scored on its second half (positions D/2 to D−1), so a depth is three independent repeats |
| Cards | three RTX 3090 (compute 8.6) in one container per run that sees only those cards; the CUDA preflight passes before each depth |
| Settings in every run | C976's: `-ngl 99 -fa on -fit off --no-host -ts 1,1,1 -b 4096 -lv 4`; the v5 preset's engine variables `GGML_CUDA_HOP_STAGED=1 GGML_CUDA_GRAPHS_FORCE=1 GGML_CUDA_MMID_GLU_UNFUSE=2`; the fork's Pascal kernels on as a P106 runs them, with a P106-090's launch shapes: `GGML_CUDA_FATTN_TILE_PASCAL=2 GGML_CUDA_MMVQ_PASCAL=2 GGML_CUDA_MMVF_PASCAL=2 GGML_CUDA_MMVQ_PASCAL_NSM=5 GGML_CUDA_MMVQ_PASCAL_KCOL_NSM=5 GGML_CUDA_MMVQ_PASCAL_KCOL_SMSM=98304`; `GGML_SCHED_DEBUG_REALLOC=1` |

The arms (each one load of `llama-perplexity`; every arm is scored against `f16`'s log-probabilities on the same
tokens):

| Arm | Cache | `-ub` | Switches | What it is |
|---|---|---:|---|---|
| `f16` | F16 | 512 | | the reference: the v5 preset's cache and micro-batch |
| `q8` | Q8_0 | 512 | `GGML_CUDA_FATTN_TILE_PASCAL_MIN=1 GGML_CUDA_FATTN_DEC_PASCAL=4` | the cache as the rig's agent runs used it, through the Pascal prompt kernel |
| `q8k` | Q8_0 | 1 | the same | every token its own pass: the cache through pull request 9's decode kernel at every position up to the depth |
| `f16d` | F16 | 1 | | every token its own pass, as the preset decodes: the decode path's F16 control, so `q8k` − `f16d` is the cache's cost through the decode path |

`GGML_CUDA_FATTN_DEC_PASCAL=4` sends one to four query tokens with a Q8_0 cache at head 256 to the decode kernel,
which is reached only from `GGML_CUDA_FATTN_TILE_PASCAL_MIN` tokens, hence `_MIN=1`. With 512-token micro-batches both
switches leave the prompt kernel in charge; at one token a pass every attention call of `q8k` goes to the decode kernel.

## Rule (fixed before the run)

Per depth D and per stretch s (three each), each arm's ln(PPL(arm)/PPL(`f16`)), mean KLD and top-token disagreement
(100 − same top %), from `llama-perplexity`'s per-chunk lines. Across the three stretches: the mean, its standard
error (the spread of the three divided by √3) and t = mean / standard error. The spread across stretches is the error
that holds for neighbouring tokens being correlated (C976 found the per-token error too small for that).

1. **The answer (prompt kernel, `q8` against `f16`), per depth:**
   - **grows with the context** if its mean ln PPL ratio m exceeds C976's at 32,768 tokens (+0.003490 ± 0.002556 over
     four stretches) by t = (m − 0.003490) / √(SE² + 0.002556²) ≥ 3;
   - **not measured growing** if that t is below 3;
   - **within 1 %** if m + 3 SE ≤ 0.01 (C976 judged its repeats by the same 1 %, at two standard errors).
   KLD and top-token disagreement are reported beside it, against C976's per-stretch means (16,384: 0.009697 and
   1.778 points over six stretches; 32,768: 0.012008 and 1.688 over four).
2. **The decode kernel, per depth:** per stretch, Δ = (`q8k` − `f16d`) − (`q8` − `f16`), in ln PPL ratio, KLD and
   top-token disagreement (`f16` against itself is zero). **The decode kernel adds an error at this length** if Δ's
   t ≥ 3 in KLD or top-token disagreement (one-sided, as [2026-11-24_c993_floor_measured](../2026-11-24_c993_floor_measured/README.md)
   judges a fault) or |t| ≥ 3 in ln PPL ratio. The cache's cost through the decode path, `q8k` − `f16d` in ln PPL ratio,
   is **within 1 %** if its mean + 3 SE ≤ 0.01.
3. **Validity** (a depth's verdict holds only if all of these hold): every run exits 0 and logs `computing over 3
   chunks, n_ctx=D`; the logs of `f16` and `f16d` show an F16 cache and those of `q8` and `q8k` a Q8_0 cache; `q8`'s
   log names the prompt kernel at head 256 with Q8_0 K/V, `q8k`'s the decode kernel (`head 256, q8_0 K/V, up to 4
   rows`), and `f16d`'s no decode kernel.
4. **Fit:** the reference at 98,304 tokens finishing is the fit on three 24 GB cards and this host; the lowest host
   memory available during the run is reported. If it does not fit, the largest multiple of 16,384 tokens whose
   reference runs takes its place, and this page says so.

What size of attention error this kind of check sees was measured on Ornith the same day
([2026-11-24_c993_floor_measured](../2026-11-24_c993_floor_measured/README.md)): against an unfaulted run, an error of
1 % in the attention output by KLD and of 2 % by top-token disagreement, on twelve stretches of 8,192 tokens. Three
stretches give a larger standard error; the rule above takes it from the data.

## Files

| File | What it is |
|---|---|
| [run/run.sh](run/run.sh) | the run: both depths, four arms each, the memory log |
| [run/kld_depth.sh](run/kld_depth.sh) | C976's driver, unchanged (md5 `68dceb3dd6f1a7e8e55696cabe0a0a49`) |

## Commands

```bash
WORK=<work dir on a disk with ~130 GB free> BIN=<build>/bin MODEL=<Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf> \
  TEXT=<agent-heldout.txt> GPUS=0,1,2 MOUNTS="<folders the container reads>" \
  DOCKER_ARGS="--cpu-shares=256" PREFLIGHT=<cuda check script> \
  setsid nohup bash run/run.sh > driver.log 2>&1 < /dev/null &
```

Host memory: at 98,304 tokens the reference run holds the chunk's log-probabilities for the whole chunk (98,304 ×
248,324 × 2 bytes, 48.8 GB) and the logits of its scored half (49,152 × 248,320 × 4 bytes, 48.8 GB), about 95 GiB in
all; a scored arm about 72 GiB. Base files: 73.2 GB at 98,304 and 48.8 GB at 65,536.

## Provenance and licence

- **Model:** Ornith-1.5-122B by ornith-ai; `valid1` is Miner Llama's N2P/N3P quantization of it with Unsloth's
  importance matrix (MIT).
- **Text:** the held-out agent sessions' text of [2026-10-27_ornith_kld](../2026-10-27_ornith_kld/README.md) (Flint's
  output, Apache-2.0; `run/agent_text.py` there); used for measurement only, not republished here.
- **This folder:** the tables hold only scores and statistics; tables and scripts are under this repository's licence.
