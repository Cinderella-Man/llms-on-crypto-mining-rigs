# Flash-Next on two rigs: Swift IQ3_S across 18 cards (2026-11-04)

Scope: how to run the best-quality Qwen3.8-Flash-Next file, which does not fit one rig, on two identical rigs (nine
P106-090 cards each, 4 GB of host RAM, gigabit on one switch), and how fast it would be. It builds on
[multi_rig.md](multi_rig.md) (the link, the traffic of each way of splitting, llama.cpp's RPC), [distributed
inference](distributed_inference.md) (the RPC protocol, the gigabit cost model) and [card hops](card_hops_x1.md) (what a
40 KB hop between two cards costs on PCIe 1.0 x1), and on the file choice ([Q811](../../data/2026-11-04_file_choice/README.md)).
Its own data, the RPC build's CPU checks, the 18-card fit and the run for the rigs are in
[docs/data/2026-11-04_two_rigs](../../data/2026-11-04_two_rigs/README.md). "(est.)" marks this note's arithmetic.

## Summary

- **The RPC build works on the fork now (Q743).** The fork's two extra ops (`GGML_OP_GDN_CONV`, `GGML_OP_KQ_MASK`)
  had stopped every `GGML_RPC=ON` build at a static assert. Branch `pascal-p106-v3-b12076-rpc` (`ba068ae6e`, on
  `-ladder3`) fixes the op count, gives the fork its own RPC major version (108) so that a fork build and a stock build
  refuse each other at the handshake instead of running the wrong ops, and rejects unknown ops on the server. On the
  CPU, with two local `ggml-rpc-server` processes standing in for a second rig, a random-weight Flash-Next model gives
  **the same tokens and the same probabilities** as a local run: whole model on one server, the layers split over two
  servers, part local and part remote, cached-prefix requests (recurrent state and checkpoints restored on remote
  buffers), sampling, and speculative decoding. Every op of the architecture ran remotely: the Gated DeltaNet
  fusions (`GDN_CONV`, `GATED_DELTA_NET`), the hyper-connections (`DSV4_HC_PRE`/`POST`), sparse attention
  (`LIGHTNING_INDEXER`, `TOP_K_MASK`, `FLASH_ATTN_EXT`), routing and `MUL_MAT_ID`.
- **The file: Swift IQ3_S** (agent KLD 0.121 against Swift-1.5's Q8_0, against IQ3_XXS's 0.176 and Q2_0's 0.285). It
  is the fine-tune the rig runs, with the same reference; ISTA's IQ3_S (0.117) comes from the original model.
- **It fits 18 cards with room to spare:** 1,780 MiB left on the fullest card at 64k (`-ub 512`, whole layers, F16
  cache), 1,761 with the DFlash2 drafter, 1,188 at 128k, and 262k at `-ub 256` with 967. llama.cpp puts the RPC
  devices first, so rig 2 holds layers 0–25 and rig 1 layers 26–47 and the output head: the logits never cross the LAN.
- **The n-gram table stays on rig 1's SSD only.** With RPC, only the host that runs llama-server opens the GGUF; the
  table is read lazily there, and its only consumer is layer 1 (`ple.layers = [1]`), on rig 2's first card. So each
  token sends 10 KB of gathered rows and the 10 KB input embedding to rig 2. Rig 2 needs no copy of the file.
- **est. decode: 5.5–6.3 tok/s plain** (158–181 ms a token, on Q2_0's measured 128–136 ms a token on one rig), against
  6.5–6.9 for Swift IQ3_XXS on one rig: 8–17 % longer per token for 1.45× lower agent KLD. Of the extra over IQ3_XXS,
  est. 4–9 ms is IQ3_S's heavier spine and experts and est. 9.5–17 ms the second rig over stock RPC: eight blocking
  copies between rig 2's cards (each a hop plus a round trip from rig 1), one 40 KB crossing back to rig 1, and the
  inputs sent one by one.
- **Ranking:** (1) **stock RPC now**: built, CPU-checked, the only way to IQ3_S quality tomorrow; (2) **the fork's own
  rig-to-rig pipeline link** (Q325): est. 4–9 ms a token less and carries pipelined speculation, weeks of engine work,
  worth it if the measured RPC overhead matches this estimate; (3) **two independent servers** on files that fit one
  rig: twice the throughput for two agent sessions, no quality gain, no code.

## 1. The RPC build on the fork

Stock `ggml-rpc.h` asserts `GGML_OP_COUNT == 102`. The fork adds `GGML_OP_GDN_CONV` (Gated DeltaNet's conv window as one
op) before `GGML_OP_UNARY` and `GGML_OP_KQ_MASK` (the attention mask built on the card from cell positions) after
`GGML_OP_GLU`, so its count is 104 and every op from `GGML_OP_UNARY` on has another number than in stock llama.cpp.
`rpc_tensor` sends the op as that number. The fix
(`pascal-p106-v3-b12076-rpc` `ba068ae6e`, one commit on `-ladder3` `7bc1552f4`):

| Change | Why |
|---|---|
| `static_assert(GGML_OP_COUNT == 104)`, `RPC_PROTO_PATCH_VERSION` 1 → 2 | the build; the patch number is upstream's convention for a changed op table |
| `RPC_PROTO_MAJOR_VERSION` 8 → 108 | a client accepts only a server with its own major version, so a fork client and a stock `rpc-server` (or the other way round) refuse each other at the handshake; with the same major version, a fork server would run a stock client's `UNARY` as `GDN_CONV`, and a stock server would read every fork op from `UNARY` on as the op after it |
| the server rejects a tensor whose op is beyond its own op table | as it already rejects an unknown type |

**What crosses the wire.** A graph node is serialized as its op number, `op_params` (all of them), flags, sources,
view source and name; `use_count` too, so fusions inside the remote backend see the same graph. The fork's CUDA work
(the Pascal matrix-vector and MoE kernels, the k-column paths, the GDN fusions, the sparse gather) is chosen inside the
backend that computes the node, so on rig 2 it runs inside `ggml-rpc-server` exactly as it does inside llama-server.
Three things do not carry over by themselves:
- **Environment switches** (`GGML_CUDA_*`) are read by the process that runs the CUDA backend: set them for
  `ggml-rpc-server` on rig 2 as well as for llama-server.
- **The RPC backend answers "supported" for every op** (upstream's `supports_op` is a TODO). Locally, an op the CUDA
  backend rejects falls back to the CPU; over RPC it reaches the remote CUDA backend and aborts the server. On the
  CPU check every op ran remotely; on the rigs, tomorrow's smoke test is the check (Q818).
- **The fork's host-side input packing** (one upload per split from a pinned buffer, Q120) does not apply to RPC
  devices: each input is its own `SET_TENSOR` message.

**CPU check** (`run/cpu_rpc_check.sh`; random-weight `qwen4exp`, two blocks, n-gram table, hyper-connections, sparse
attention, 512 experts' layout at small width):

| Arm | Layout | Tokens and top-5 probabilities against the local run (greedy 64, cached prefix 16 + 32, sampled 48) |
|---|---|---|
| rpc1 | every layer on one rpc-server | 4/4 identical, largest probability difference 0 |
| rpc2 | block 0 on one server, block 1 and the output on the other (`-ts 1,2`) | 4/4 identical, 0 |
| mixed | rpc2 with block 1 on the local CPU (remote → local → remote) | 4/4 identical, 0 |
| draft-r | rpc2 with speculative decoding (the file as its own drafter, 4 a step), against the same locally | 4/4 identical, 0 |
| hop | rpc2 with `LLAMA_HOP_CAST=f16` at the boundary between the servers | greedy identical (2.5e-7); sampled diverges at token 27 (F16 rounding, expected) |

Each CPU rpc-server peaked at 18–47 MiB resident (107 MiB with the drafter as well): the 18 MB of weights, which are a CPU
server's device memory, plus one message buffer while loading.

## 2. Which file

| File | Agent KLD (same top token) | On the cards | Shards on disk | The table |
|---|---|---|---|---|
| **Swift IQ3_S** | **0.121 (93.7 %)** | 52,066 MiB | 44.90 + 38.84 GB | in shard 1 with 372 other tensors: needs `scripts/gguf_reshard.py split --isolate` (C160) before the rig reads it from its SSD |
| ISTA IQ3_S | 0.117 (93.4 %) | 51,945 MiB | 54.82 + 28.80 GB | alone in shard 2 |
| Swift IQ3_XXS (fits one rig) | 0.176 (92.3 %) | 44,710 MiB | 3 shards (the rig's set) | alone in shard 3 |

Swift IQ3_S is the file for the one goal: the target is Swift's fine-tune, and every speed and quality number on the
rig so far is Swift's. ISTA's file is quantized from the original model; its slightly lower KLD against Swift's
reference does not make it the fine-tune. The two differ only in the routers (F32 against BF16, Q811).

**One table copy for every file.** The rig keeps the 28.8 GB IQ4_NL table once, on rig 1's SSD, as shard 3 of both
its 3-shard sets (Q2_0 and IQ3_XXS: the same file, md5 `98ce140e…`). Swift IQ3_S and ISTA IQ3_S hold the same table
byte for byte (data md5 `421370d4…` in all four files, [table_md5.tsv](../../data/2026-11-04_two_rigs/table_md5.tsv)),
and Swift IQ3_S has the same 1,224 tensors, so resharded into three shards with the table alone in the third it
reuses that file through a symlink: only shards 1–2 (16.1 + 38.8 GB) go to rig 1's USB disk. Rig 1's SSD has room
for one table, not two.

## 3. The split across 18 cards

**Device order.** llama.cpp lists RPC devices before local ones ("to minimize network transfers"), so `-ts` takes rig 2's
nine cards (RPC0–8) first and rig 1's nine (CUDA0–8) after them. The output head is on the last device, rig 1's card 8.
That is the right way round for this model: the other order would send 248,320 F32 logits (~1 MB, ~8.5 ms) over the
LAN every token, while this one sends ~20 KB of inputs one way and one 40 KB residual back. Backend sampling (`-bs`)
stays on rig 1.

**The fit** (`bench/file_choice/fit.py --cards 18`, the per-card costs llama.cpp projected for Flash-Next in Q213,
6,001–6,007 MiB free per card; [fit18.md](../../data/2026-11-04_two_rigs/fit18.md)). Smallest free MiB on any card, F16
cache, whole layers:

| Swift IQ3_S | `-ub 64` | `-ub 128` | `-ub 256` | `-ub 512` |
|---|---|---|---|---|
| 64k | 2,342 | 2,274 | 2,110 | **1,780** (DFlash2 1,761; MTP 594) |
| 128k | 2,098 | 1,989 | 1,729 | **1,188** (DFlash2 1,169) |
| 262k | 1,608 | 1,420 | **967** (DFlash2 948) | 10 |

The 64k line at `-ub 512`: `-ts 3,3,3,3,3,3,2,3,3,2,3,3,3,3,2,2,2,3` (26 layers on rig 2, 22 and the output on rig 1).
Half the cards' memory is spare: room for a larger file than IQ3_S (Q819) or a long context with a drafter.

**The n-gram table.** `per_layer_token_embd` (IQ4_NL, 160 × 320,001,536, 28.8 GB) is a lazily read host tensor: its
rows are gathered on the host that runs llama-server, from the GGUF that host opened. With stock RPC that is rig 1, so
the table lives on rig 1's SSD only, and rig 2 never reads the file at all (its weights arrive over the network, then from
its own cache, `ggml-rpc-server -c`). The gathered rows (16 heads × 160 = 2,560 floats, 10 KB a token) go to the card
that holds layer 1, the file's only n-gram layer, which is rig 2's first card. With the pipeline link (section 5) the
first rig would hold layer 1 and its own table, and nothing table-related would cross.

## 4. Decode speed

### What crosses per token, measured on the CPU

`GGML_RPC_DEBUG=3` on the client counts every command; the decode share is (65 new tokens − 1) / 64
([commands.tsv](../../data/2026-11-04_two_rigs/commands.tsv)). On the random file split over two servers, a decoded
token sent **26 commands: 21 `SET_TENSOR` (the splits' inputs, ~20 KB, no reply), 2 `GRAPH_RECOMPUTE` (each server's
graph re-run by its id, 24 B) and 3 `GET_TENSOR` with a reply** (the residual read from the first server to be set on
the second, the same for the last block's second input, and the logits), the same at 32 and 160 tokens of context.
A full `GRAPH_COMPUTE` (76–128 KB) went out only when the graph's shape changed, 3 times in 65 tokens. Per input that is
one message of a ~300-byte tensor header plus its data; the fork's resident mask cells make the attention inputs a few
hundred bytes, not n_kv × 8 bytes.

### The same on the rigs (est.)

On two rigs, rig 2's nine cards are one server, so the residual passes between them by `COPY_TENSOR` on that server
(no relay through rig 1), and the crossing to rig 1 is one `GET_TENSOR`. Each of these waits for a reply, and the server
synchronizes its cards before every command that touches buffer data, so a boundary on rig 2 costs one hop plus one
network round trip, with the next card's launch behind it.

| Per decoded token | One rig | Two rigs, stock RPC (est.) | Two rigs, pipeline link (est.) |
|---|---|---|---|
| Card boundaries | 8 local hops (0.5–0.9 ms each, card_hops) | 8 on rig 1 as before + 8 on rig 2 as blocking `COPY_TENSOR` (hop + round trip 0.1–0.3 ms + launch 0.1–0.3 ms + ~10 inputs as single messages 0.3–0.5 ms: 1.05–2.0 ms each) | 8 on rig 1 + 8 local hops on rig 2 (0.5–0.9 ms each) |
| LAN | — | one 40 KB `GET` (0.35 ms of bytes at 117 MB/s + round trip + the two PCIe legs 0.4 ms: 0.9–1.1 ms); ~20 KB of inputs to rig 2, no reply | one 40 KB one-way message (~0.8 ms) and a 4-byte token id back |
| Extra over one rig | — | **9.5–17 ms** | **5–8 ms** |
| With F16 hops (`LLAMA_HOP_CAST=f16`, Q231) | −1.6 ms | −1.8 ms more (eight PCIe hops and the LAN crossing at half the bytes) | the same |

**Per token, plain decoding, short context** (est.; "measured basis" = Swift Q2_0's 128–136 ms a token in the agent check
and R13, plus the file-choice model's GPU-time differences; "GPU basis" = the file-choice model's GPU time alone,
tonight's kernels at their expected speed, a lower bound):

| Line | GPU over Q2_0 | Second rig | Measured basis | tok/s | GPU basis | tok/s |
|---|---|---|---|---|---|---|
| Swift Q2_0, one rig | — | — | 128–136 ms | 7.4–7.8 | 73.0 ms | 13.7 |
| Swift IQ3_XXS, one rig | +16.4 / +19.0 | — | 145–155 ms | 6.5–6.9 | 91.9 ms | 10.9 |
| **Swift IQ3_S, two rigs, stock RPC** | +20.0 / +27.8 | +9.5–17 | **158–181 ms** | **5.5–6.3** | 110–118 ms | 8.5–9.1 |
| Swift IQ3_S, two rigs, pipeline link | +20.0 / +27.8 | +5–8 | 153–172 ms | 5.8–6.5 | 106–109 ms | 9.2–9.5 |

So the second rig costs more than the heavier file does: est. 9.5–17 ms against 4–9 ms. The measured 18-card overhead
comes from tomorrow's `ident` step (the same IQ3_XXS file on 9 and on 18 cards, so only the second rig differs).

### Speculation across rigs

- **The drafters sit on rig 1.** DFlash2 and the MTP draft follow the output head onto rig 1's last card, so drafting
  never crosses the LAN. DFlash2 conditions on the target's hidden states at a few layers; those on rig 2 come back with
  one more `GET` each per verify pass (est. +0.1–0.4 ms each).
- **A verify pass of k tokens pays the per-boundary costs once.** The round trips, launches and inputs are per pass;
  only the bytes grow with k (k × 40 KB per boundary: 1.4 ms more on the LAN at k = 4, F32). At ~2–2.5 tokens per pass
  the second rig's cost per token falls to est. 4–8 ms. The CPU check ran speculative decoding over RPC with the same
  tokens as locally.
- **Pipelined speculation (`--pipe-depth`) does not survive stock RPC.** Same-server copies block the client
  (`cpy_tensor_async` is not implemented), so several tokens cannot be in flight on rig 2's cards; the fork's pipelined
  executor would degrade to plain speed on the remote half (multi_rig.md §4). The pipeline link keeps it.

## 5. Prompt speed (est.)

A 512-token micro-batch carries 21 MB of residual (F32) across each boundary: 0.2–0.5 s per hop inside a rig and
~0.18 s on the LAN. A turn whose new tokens fit one micro-batch passes all 18 cards in turn, so it pays 9 more hops than
on one rig: est. +2–4.9 s per 512 tokens (+1–2.5 s with F16 hops) on ~9–11 s of compute, about **0.65–0.8× one rig's
prompt speed** for IQ3_S on 18 cards against IQ3_XXS on nine. Longer prompts (the first turn's ~2,780 tokens, six
micro-batches) can overlap micro-batches across cards; stock RPC has overlapped them across RPC devices since PR #18626
(+30–70 % with two servers upstream), but the blocking copies on rig 2 hold the client back. Est. 0.65–1.2× in all;
the `ident` step measures the first turn on 9 and 18 cards.

## 6. Host memory on the 4 GB rigs

| | Rig 1 (llama-server, the client) | Rig 2 (`ggml-rpc-server`) |
|---|---|---|
| CUDA contexts | nine, as today | nine: est. 360–630 MiB (Q565, unmeasured) |
| Weights in host memory | the token embeddings (322 MiB) and the table's page cache, as today | none once on the cards |
| While loading | the client copies each remote tensor into one message: up to the largest rig-2 tensor, `ffn_down_exps` in IQ4_NL, 450 MiB, at a time | receives each tensor whole into one buffer (450 MiB peak); with `-c`, a cached tensor is read whole from its file the same way |
| Per token | a few hundred KB of RPC buffers | the same |

Est. peak on rig 2: ~0.9–1.2 GB, far below what llama-server needs on rig 1. Rig 1's load peak rises by up to ~0.45 GB
over today's; the RAM watchdog guards both (tomorrow's script runs `scripts/ram_watchdog.sh` on rig 2 too).

## 7. The options, ranked

| | Stock RPC (this build) | The fork's pipeline link (Q325) | Two independent servers |
|---|---|---|---|
| File | Swift IQ3_S (KLD 0.121) | Swift IQ3_S | Swift IQ3_XXS or Q2_0 per rig (0.176 / 0.285) |
| est. plain decode | 5.5–6.3 tok/s | 5.8–6.5 tok/s | 6.5–6.9 (IQ3_XXS) / 7.4–7.8 (Q2_0) per session |
| Pipelined speculation | no (remote half at plain speed) | yes | yes, per rig |
| Load | rig 1 reads all 84 GB and sends rig 2's ~28 GB over the LAN (~4–5 min the first time), with `-c`, later loads send hashes | each rig reads its own layers from its own disk | each rig its own file |
| Code | none beyond Q743 | weeks: one process per rig, one-way messages, cache edits in order, partial loading | none |
| For | the best quality now | the best quality at the best speed, after Q816 measures RPC | two agents at once; the three coming rigs |

**Recommendation: stock RPC with Swift IQ3_S now.** It is the only route to the best file tomorrow and makes a
token est. 8–17 % longer than IQ3_XXS on one rig. If Q816 measures the second rig's cost near this estimate, two cheap
RPC changes come first (Q817: same-server copies without a reply, inputs packed into one message per split, est. −3 to
−5 ms), and the pipeline link after them if pipelined speculation is wanted across rigs.

## 8. Ideas registered

- **Q816** (C260): the two-rig run, `run/two_rigs.sh`: network baseline, the RPC smoke test, the 9- against 18-card
  identity on Swift IQ3_XXS (same greedy text; the measured cost of the second rig), Swift IQ3_S's smoke test with and
  without F16 hops, then one agent-check arm.
- **Q817** (C261): RPC changes in the fork for the rig-2 boundaries: `cpy_tensor_async` for copies within one server (no
  reply, ordered on the server's queue) and one packed `SET_TENSOR` per split for its inputs. est. −3 to −5 ms a token.
- **Q818** (C262): `supports_op` over RPC: ask the server once per op shape and cache it, so an op the remote CUDA
  backend lacks falls back to rig 1's CPU instead of aborting `ggml-rpc-server`. Needed only if the smoke test aborts.
- **Q819** (C263, parked): a larger Flash-Next file on two rigs: 18 cards leave ~1.8 GB per card at 64k beyond IQ3_S,
  ~30 GB in all, room for IQ4-class experts. Reopens if Q816's agent arm holds its speed and a Swift file of that size
  exists (or FN's quantization chain makes one).
