# Does a verify pass copy its logits to the host? (C338 step 2, 2026-11-04)

What this checked: whether llama-server's speculative verify pass on the rig's MTP draft line copies a full logits row
per verified position from the last card to the host. Four such rows would be 3.97 MB a step (248,320 F32 each), est.
~20 ms over PCIe 1.0 x1, about half of the ~41 ms a step that no card's clock holds in
[2026-11-04_verify_host](../2026-11-04_verify_host/README.md). If it did, the plan ([Q900](../../../quants/experiments.md),
C338 step 2) was to sample the verify batch on the last card behind a new switch.
The method is a reading of Miner Llama's source at `pascal-p106-v3-b12076-kshape` `d27845b67`. Its server, sampling,
context, graph and CUDA op-support code is identical to `-ladder3` `7bc1552f4`, the build rig 2 measured (an empty
`git diff` over those files). Nothing was built or run.

## Result

**No: with `-bs` the verify batch is already sampled on the last card, and no full logits row crosses PCIe.** Every
agent-check and smoke arm passes `-bs` (`release_check_s12.sh` and `smoke_dflash.sh` put it in their common
arguments). So the design C338 step 2 asked for is already what runs: the drafted positions' rows stay on card 8, the
request's own top-k/top-p/temperature chain samples each of them there, and the host gets back the sampled id and the
chain's candidates. C338 step 2 is closed without code.

| Bytes back to the host per step (MTP draft, `--spec-draft-n-max 3`, 4-token verify) | Without `-bs` | With `-bs` (the rig's line) |
|---|---:|---:|
| the verify pass's sampling data, 4 rows | 3,973,120 (4 × 248,320 F32) | ~976 (4 × 244: the id, 20 logits, 20 probs, 20 ids) |
| the target's hidden rows for the MTP draft (`h_nextn`, n_embd 2560) | 40,960 | 40,960 |
| each draft pass's sampling data | 993,280 | 80 (10 logits, 10 ids: the draft's device top-k 10) |
| a full row after a tool-grammar rejection (Q67's resample, synchronous) | — | 993,280, only on a rejection |

At the link's ~190 MB/s the line's ~42 KB a step is ~0.2 ms, not ~20 ms.

**How the path goes** (file:line in the fork at `d27845b67`):

| Step | Where | What happens |
|---|---|---|
| the request's chain goes to the context | `tools/server/server-context.cpp:2938-2948` | with `-bs`, `llama_set_sampler(ctx_tgt, slot, chain)`; it stays off only for `n_probs > 0` without post-sampling probabilities, and for TTS |
| one chain per verified row | `server-context.cpp:57-77`, `common/speculative.cpp:3358` | `n_outputs_max_per_seq` = draft length + 1 (4 here), so each row of the verify batch gets its own chain |
| the graph samples every row | `src/llama-graph.cpp:4679` (`build_sampling`) | `n_outputs_max_per_seq` chains per sequence, built on the output card |
| the logits stay on the card | `src/llama-context.cpp:3210` (`needs_raw_logits`), `3534-3540` | every output has a sampler, so `t_logits` is never read back; its rows are marked as on the device |
| what comes back | `llama-context.cpp:3175` (`copy_tensor_async_rows`), `3664-3671` | `ggml_nbytes` of each row's sampled id, logits, probabilities and candidate ids, after top-k: 20 of each |
| the chain on the card | `src/llama-sampler.cpp` top-k `1586`, top-p `1736`, dist `1286`; the order `common/common.h:292` | at `--top-k 20 --top-p 0.95 --min-p 0 --temp 1.0` the penalties, DRY, top-n-sigma, typical, min-p, XTC and temperature samplers are empty, so the whole chain runs on the device; with several outputs per sequence the dist sampler's draws are committed only as tokens are accepted |
| the CUDA backend runs it | `ggml/src/ggml-cuda/ggml-cuda.cu` (`supports_op`) | TOP_K, ARGSORT (CUB), CUMSUM, STEP, LOG, SET_ROWS, GET_ROWS on I32, CPY F32→I32, SOFT_MAX, SUM, CLAMP, SCALE, ADD, SUB and PAD carry no compute-capability condition, so the P106s run the chain; an unsupported op would log `does not have support for op` and leave the rest of the chain on the host |
| the host's part of a row | `common/sampling.cpp:146-176` (`set_logits`), `690-760` | the 20 candidates; the card's token is taken, and with a tool grammar only that token is checked (Q67) |
| token-by-token rejection | `common/sampling.cpp:904-1008` | the target's p of the drafted token and the residual come from the 20 candidates |
| a full row | `sampling.cpp:746` → `get_logits_ith_fetch` (`llama-context.cpp:1339`) | only when the grammar rejects the card's token (Q67's resample) or the grammar applies first |
| the MTP draft | `common/speculative.cpp:1963-1985`, `common/common.h:363` | `--spec-draft-backend-sampling` is on by default: a device top-k 10 chain per draft row, the host's dist on those 10 |

**Does it change anything?** Yes, in two ways:
- **The ~41 ms of a 179 ms step that no card's clock holds is not logits.** By the same reading it is the target's eight
  hops between cards (est. ~9 ms) and the host's turnarounds. Those include the four draft passes: the draft's host span
  is 22.2 ms over 8 evaluations, and with `-otd token_embd=CUDA4` each pass crosses card 4 → card 8 through the host and
  waits for its sampled id before the next is queued. They also include the server's bookkeeping between the draft and
  the next verify pass (`before_ms` 14.5). These are estimates. The split is now its own measurement (Q960, C405), and
  the draft's turnarounds are a candidate change (Q961, C406).
- **`-bs` is worth ~4 MB a step on this line:** a server line without it pays the four rows' copy and the host's
  sampling of 248,320 values a row. Every release and agent-check line keeps `-bs`.

## Files

This folder is the README only: there is no run, table or script. The source lines above are the evidence.

## Commands

None: a source reading. To repeat it, check out the fork at the commit below and read the lines in the table.

## Provenance

Miner Llama `pascal-p106-v3-b12076-kshape` `d27845b67` (llama.cpp b12076 base, MIT licence), read on 2026-11-04. The
bytes are computed from the tensor shapes the code builds. The `h_nextn` width is Flash-Next's `n_embd` (2560, from a
server log's model print); the 993,280-byte row is 248,320 F32.
