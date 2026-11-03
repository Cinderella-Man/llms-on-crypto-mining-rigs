# Context checkpoints on the cards and inside long prompts (C22, C28; 2026-11-04)

Registry rows [Q440, Q448 and Q548](../../../quants/experiments.md). Flash-Next (`qwen4exp`) is a hybrid model: its
recurrent state (Gated DeltaNet) cannot be rolled back, so llama-server saves context checkpoints of it and restores
one when a prompt diverges from what the slot holds. On the rig each checkpoint is 112.6 MiB of host RAM (four are kept,
~450 MiB of a 4 GB host that an agent session leaves 140–460 MiB free), a save takes 0.67 s and a restore 0.63 s over
the cards' PCIe 1.0 x1 links ([Q215](../2026-11-03_flash_next_baseline/README.md)), and they are taken only after the
chat template's prefix and in a prompt's last ~512 tokens, so a follow-up that changes history further back re-reads
the whole prompt (a 31k-token follow-up: 209 s to its first token).

Two server switches on Miner Llama's `pascal-p106-v3-b12076-ckpt` branch change that, both off by default:

- **`--ctx-checkpoints-vram N` (C22, Q440):** a slot's newest N checkpoints stay on the cards, each layer's state copied
  into a buffer on its own card (a copy inside each GPU, no PCIe transfer, no host memory). Older ones move to the
  host in the background (`--ctx-checkpoints-vram-sync`: at once). Underneath is a small llama API, numbered device
  stores of a sequence state (`llama_state_seq_get_data_store` and the four calls beside it in `include/llama.h`),
  built on upstream's single-slot `LLAMA_STATE_SEQ_FLAGS_ON_DEVICE` path. If a store cannot be allocated, the server
  logs it and keeps its checkpoints on the host from then on.
- **`--ctx-checkpoints-spread N` (C28, Q548, Q448):** checkpoints at most N tokens apart inside a prompt: at the first
  message start N/2 or more tokens after the previous checkpoint, or N tokens after it when no message starts; when
  the list is full, the checkpoint whose removal leaves the smallest gap goes (the new checkpoint counts as the newest
  one's neighbour, so of the two checkpoints at a prompt's end the earlier one goes before one far back does).

The responses' `timings` carry `ckpt_save_ms` (time spent saving checkpoints for the request) and `ckpt_dev_n` (how
many of them went to the cards) beside the existing `cache_restore_ms`, `cache_src` and `cache_ckpt_n`.

## Result: both paths give identical numbers; on the CPU so far

Checked on the CPU only, in a container without a GPU, with random-weight models from `test-llama-archs`: qwen4exp
(Flash-Next's architecture), qwen35 (Gated DeltaNet and attention) and mamba (recurrent only). The rig numbers (save
and restore times on the cards, host memory, the history edit and the agent check) are the next step.

- **The stores, against the host path** (`test-llama-context`, new cases `store` and `store_p`, whole and partial
  states, each also against a cache pre-filled with a pattern): a state restored from a store continues with the
  same logits, bit for bit, as the state restored from the host; both follow the live continuation; the store's host
  copy is the host state byte for byte, also when it is made on another thread while the context decodes; two stores
  of one sequence keep their own states; a freed store, and the blob of another fill, restore nothing. All 20 cases
  pass or skip as before on the three models: 54 OK, 6 SKIP (cases for DeepSeek V4, and mamba's rollback cases, which
  need `n_rs_seq`), 0 FAIL ([context_cases.tsv](context_cases.tsv)).
- **llama-server** ([server_check.sh](run/server_check.sh), the random qwen4exp, five requests: a 3,000-token prompt,
  its continuation, the same history with one token changed at 2,200, the original again, and a continuation of the
  edit; [server_requests.tsv](server_requests.tsv), [server_verdict.txt](server_verdict.txt)):

| Arm | Switches | Tokens and top-3 probabilities, P1–P5 | Checkpoints saved (on the cards) | Moved to the host | History edit (P3): resumed from |
|---|---|---|---|---|---|
| `base` | none | reference | 10 (0) | — | nothing: re-read 3,516 tokens |
| `vram` | `--ctx-checkpoints-vram 8` | = base | 10 (10) | 0 | nothing: re-read 3,516 |
| `vram2` | `--ctx-checkpoints-vram 2` | = base | 10 (10) | 2, in the background | nothing: re-read 3,516 |
| `vram2s` | `--ctx-checkpoints-vram 2 --ctx-checkpoints-vram-sync` | = base | 10 (10) | 2, at once | nothing: re-read 3,516 |
| `spread` | `--ctx-checkpoints-spread 512` | = base | 22 (0) | — | the checkpoint at 2,048: re-read 1,468 |
| `spread_v2` | `--ctx-checkpoints-spread 512 --ctx-checkpoints-vram 2` | = base | 22 (22) | 13, in the background | the checkpoint at 2,048: re-read 1,468 |

  Where a checkpoint lives changes no number: the store arms give `base`'s tokens and probabilities exactly. The
  spread arms give them too, although they split the prompt into other batches and resume the edited history from
  2,048 tokens instead of re-reading it. No server logged an error. The CPU times in the table are not speeds: the
  model is a 19 MB random one.

## Files

| File | What |
|---|---|
| [context_cases.tsv](context_cases.tsv) | `test-llama-context` per model and case: verdict and skip reason |
| [server_requests.tsv](server_requests.tsv) | `server_check.sh` per arm and request: prompt tokens processed and reused, where the reuse came from, the restored checkpoint's position, checkpoints saved (on the cards), save and restore ms, the generated tokens |
| [server_verdict.txt](server_verdict.txt) | its comparisons |
| [run/chain.sh](run/chain.sh) | the rig build of the branch, `test-llama-context` and `test-llama-archs` from the same tree, then `checks.sh` in a container without a GPU |
| [run/checks.sh](run/checks.sh) | the random models, `test-llama-context` on each, `server_check.sh` |
| [run/server_check.sh](run/server_check.sh) | the six server arms and their comparison |
| [run/history_edit.sh](run/history_edit.sh) | for the rig: Q215's single-request line (`fn_line.sh` with its 32k follow-up that changes the last ~1,500 tokens) once per checkpoint setting |

## Commands

The build machine needs Docker and a llama.cpp clone with the fork as remote `fork` in `$WORK/src`
([scripts/build_llamacpp_docker.sh](../../../scripts/build_llamacpp_docker.sh)):

```bash
WORK=<work dir> REPO=<this repository> BRANCH=pascal-p106-v3-b12076-ckpt JOBS=6 THREADS=8 \
  DOCKER_ARGS='--label <your label>' bash docs/data/2026-11-04_ckpt_cards/run/chain.sh
cat $WORK/checks/server/verdict.txt    # PASS
```

Build: `pascal-p106-v3-b12076-ckpt` at `b5ed2f4fe` (three commits on `pascal-p106-v3-b12076-ladder` `98bbd6a2f`), image
`nvidia/cuda:12.9.2-devel-ubuntu24.04`, sm_61, no AVX, GLIBC_2.38: `llama.cpp-20261103-b5ed2f4fe-pascal-p106-v3-b12076-ckpt.tar.gz`,
304,224,527 bytes, md5 `b7d9461b92faee80ca886781473a31f2`. Server check alone, on any build:

```bash
BIN=<build>/bin MODEL=<random qwen4exp gguf> OUT=<dir> SARGS='--override-kv qwen4exp.context_length=int:16384' \
  bash docs/data/2026-11-04_ckpt_cards/run/server_check.sh
```

Provenance: the models are `test-llama-archs -a '^<arch>$' -s 1 -o <dir>` of the same build (random weights, a test
tokenizer); no other data. Licence: the scripts and tables are this repository's.
