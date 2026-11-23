# Qwen3.6-35B-A3B on Miner Llama v5

A preset for Qwen3.6-35B-A3B on four GTX 1070 8 GB cards: Unsloth's UD-Q5_K_S with its MTP layer, a 131,072-token
context, and the model's own MTP head guessing the pipeline's next token. It needs the fork's branch
`pascal-p106-v5` with PRs [#3](https://github.com/Cinderella-Man/miner-llama/pull/3)–[#7](https://github.com/Cinderella-Man/miner-llama/pull/7)
(merged 2026-11-20). The model's page is [models/qwen3.6-35b-a3b](../../models/qwen3.6-35b-a3b/README.md).

**This is a contributor's rig, not the reference rig:** four GTX 1070 8 GB cards (compute 6.1) on PCIe x1 risers, a
Celeron G3900, 8 GB of host memory, Windows 10 with NVIDIA driver 582.78, and a local CUDA 12.9 sm_61 build. The
switches are not Windows-specific, but the line has not been run on Linux or on P106 cards. Every number on this page
was taken on that rig at temperature 0. Where a setting has two values ("68.5 and 72.9 tok/s"), they are the slowest
and fastest of the contributor's runs; the runs between them were not recorded.

## What it gets

Measured 2026-11-20, `pascal-p106-v5` with the five PRs, this preset as it is:

| | |
|---|---:|
| Plain text, 1,500 tokens | 79.6 and 82.7 tok/s |
| Tool call on a fresh 39k-token prompt | 69.1 s: prompt 663 tok/s, writing 71.7 tok/s |
| Tool call at 72k (32.8k new tokens on 39k cached) | 105.5 s: prompt 367 tok/s, writing 57.7 tok/s |
| 17k-token prompt | read at 879.6 tok/s |
| 49-token JSON-schema answer, cold | 1.9 s, valid |
| Free memory at load, GPU 0-3 | 665 / 663 / 897 / 1,191 MiB (same line on v4.0) |

The same line on v4.0 with the same changes gives the same outputs (identical MD5s for the plain text and both tool
calls' arguments) and the same speed within 1 %.

**Against v4.0 tuned without these changes**, same rig, 2026-11-17:

| | v4.0, tuned | v4.0 + these changes |
|---|---:|---:|
| Plain text, 1,500 tokens | 53.6, 54.8 tok/s | 78.9, 82.2 tok/s |
| Tool call at ~39k, fresh | 153.0 s | 79.3 s |
| Tool call at ~72k, fresh | 240.8 s | 125.6 s |
| 49-token JSON-schema answer, warm | 1.58, 1.39 s | 1.09, 0.85 s |

That comparison used a q8_0 cache and `-ub 256` on both sides; the preset has since moved to an f16 cache and
`-ub 128` (below), which is why its tool calls are faster than the right-hand column.

## Run it

Build the fork from the branch `pascal-p106-v5` for compute 6.1 (CUDA 12.9: CUDA 13 cannot target Pascal), and
download the file:

```bash
wget -c -P /mnt/models https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-Q5_K_S.gguf
```

```bash
MODELS=/mnt/models; . llms-on-crypto-mining-rigs/release/qwen3.6-v5/settings.env && env $PRESET_ENV llama-server -m "$MODELS/$PRESET_MODEL" $PRESET_ARGS --host 127.0.0.1 --port 8080
```

On Windows (PowerShell), the same line:

```powershell
$env:LLAMA_DRAFT_SHORTLIST='65536'; $env:GGML_CUDA_HOP_MAILBOX='1'; $env:GGML_CUDA_HOP_MAILBOX_TIMEOUT_MS='1000'
$env:GGML_CUDA_HOP_MAILBOX_MAX_KB='0'; $env:GGML_CUDA_HOP_MAILBOX_GATE_KB='64'; $env:LLAMA_PIPE_GRAMMAR='1'
$env:LLAMA_ARG_CTX_CHECKPOINTS_VRAM='2'; $env:GGML_CUDA_GRAPHS_FORCE='1'; $env:LLAMA_SERVER_YIELD='2'
$env:LLAMA_SAMPLING_TOP_K_ROWS='1'; $env:LLAMA_KV_SEQ_RM_BOUND='1'
.\llama-server.exe -m D:\models\Qwen3.6-35B-A3B-UD-Q5_K_S.gguf --alias qwen36 -ngl 99 -c 131072 -np 1 `
  --load-mode none -fa on --jinja --reasoning off -ctk f16 -ctv f16 -ts 0.85/1/1.05/1.1 -ub 128 `
  --spec-type draft-mtp -ctkd q8_0 -ctvd q8_0 --spec-draft-backend-sampling `
  -bs --pipe-depth 2 --pipe-guess mtp --pipe-mtp-walk 1000 --mtp-device 0 `
  --spec-ckpt-vram --ctx-checkpoints-incall --cache-ram 3072 --host 127.0.0.1 --port 8080
```

On this rig the server process runs at High priority (Windows): a tool call at 39k writes at 42.2 and 42.4 tok/s against
38.3 and 39.3 at Normal (measured 2026-11-16 on an earlier build of this line); plain text at short context gained
~10 tok/s. A second core made no difference to tool calls.

## The preset: [settings.env](settings.env)

| Switch | What it does here | Measured |
|---|---|---|
| `--spec-type draft-mtp -ctkd q8_0 -ctvd q8_0` | the model's own MTP layer as the draft model, its cache in q8_0 | |
| `-bs --pipe-depth 2 --pipe-guess mtp` | pipelined speculation with the MTP head guessing each next token, one row launched ahead | 1,290 right / 208 wrong per 1,500 tokens, against 458 / 585 for `cont`; output byte-identical to plain decoding (PR #7) |
| `--pipe-mtp-walk 1000` | a right guess keeps the MTP chain going instead of re-anchoring every token | 68.5 and 72.9 against 62.9 and 65.4 tok/s at depth 2 (PR #7) |
| `--mtp-device 0` | the MTP layer and a copy of the output head (515 MiB) on GPU 0 | on the last card an MTP row waited behind pipe work under WDDM: 36 to 47 ms a row, 15.5 tok/s; on GPU 0, 38 and 40 tok/s. May not matter on Linux (PR #7) |
| `LLAMA_DRAFT_SHORTLIST=65536` | the MTP head scores token ids below 65,536, special tokens and the tokens in the context, instead of all 248k | the guesser 50.4 tok/s instead of 29.4 with the same hit rate (PR #6) |
| `LLAMA_PIPE_GRAMMAR=1` | the pipe keeps running under a grammar; the host checks each drawn token and a rejection counts as a wrong guess | tool call at 39.5k 53.1 and 55.6 tok/s instead of 31.5 and 32.1, JSON schema 89 instead of 47; output differs from the batched path at near ties, both valid, and repeats exactly (PR #7) |
| `GGML_CUDA_HOP_MAILBOX_MAX_KB=0 GGML_CUDA_HOP_MAILBOX_GATE_KB=64` | hops above 64 KiB go through pinned host buffers ordered by CUDA events; smaller ones keep the polling mailbox | fresh prompts 19.3k 763 vs ~390 tok/s, 39.5k 572 and 591 vs 281 and 289, 102k 318 vs 122 and 155; a 125k prompt passes, where the mailbox timed out at ~50k to 60k and the display driver reset (PR #4) |
| `--spec-ckpt-vram` | the per-step speculation checkpoint of the recurrent state (62.8 MiB) stays on the card instead of crossing the x1 link every step | 49-token JSON answer 3.9 s → 1.5 and 1.7 s, tool call 5.3 → 3.8 s, same output (PR #5) |
| PR #3 (no switch) | at temperature 0 a speculation replay after a checkpoint restore is accepted as it is, as at temperature > 0 | JSON-schema requests that livelocked (the same restore and one-token replay hundreds of times) answer in 2–4 s |
| `-ctk f16 -ctv f16` | f16 cache for the main model | on Pascal the stock q8_0 decode attention read ~28 GB/s against ~145 for f16: tool-call writing +34 % at 39k, +48 % at ~104k |
| `-ub 128` | prompt micro-batches of 128 | a ~104k prompt in 263.6 s instead of 278.7 with 256; ~330 MB more free on GPU 0 |
| `-ts 0.85/1/1.05/1.1` | layers weighted off GPU 0, which also holds the MTP layer and its head copy | keeps every card above ~650 MiB free at load |
| `-c 131072` | half the model's context | 262,144 does not fit in f16 on 4 × 8 GB with this split |

## What is left out, and why

| Switches | Why not here |
|---|---|
| A q8_0 cache for the main model, 262,144 tokens | slower decode attention on Pascal (above); new compute-6.1 kernels for it are in progress and would bring it back |
| `--pipe-guess cont` | 458 right / 585 wrong per 1,500 tokens against the MTP head's 1,290 / 208 |
| `GGML_CUDA_HOP_MAILBOX_MAX_KB=64` without `GATE_KB` | the cap that worked before PR #4; long prompts read at half the speed |

## Not measured yet

- The agent check of [bench/agent_check](../../bench/agent_check/README.md) and the deep lane: the numbers above
  come from the contributor's own scripts (plain text, tool calls at 39k and 72k, a JSON schema, a 17k prompt).
- Linux, P106 cards, and fewer or more than four cards.
- Quality of the UD-Q5_K_S file against Q8_0 (the model page's table has Q4 files only).
