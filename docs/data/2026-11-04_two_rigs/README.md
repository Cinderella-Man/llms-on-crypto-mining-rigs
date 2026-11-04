# Flash-Next on two rigs: the RPC build, its CPU checks, the 18-card fit and the run for the rigs (2026-11-04)

Registry rows Q743 (llama.cpp's RPC backend on the fork), Q505 (Flash-Next across two rigs) and Q816–Q819. The
write-up, with the speed estimates and the ranking of the options, is the research note
[flash_next_two_rigs.md](../../research/notes/flash_next_two_rigs.md). Nothing here ran on a GPU or a rig: the
build machine's CPU ran the checks, and the rigs' run is written for the next session with both rigs up.

## Result

- **The fork's RPC backend builds and runs.** Fork branch `pascal-p106-v3-b12076-rpc` `ba068ae6e` (one commit on
  `-ladder3` `7bc1552f4`): `ggml-rpc.h` asserts the fork's 104 ops (stock: 102) and moves the patch version to 2, the
  fork's RPC major version is 108 (a fork build and a stock build refuse each other at the handshake, since every op
  from `GGML_OP_UNARY` on is numbered differently), and the server rejects ops beyond its table. Rig tarball
  `llama.cpp-20261104-ba068ae6e-pascal-p106-v3-b12076-rpc.tar.gz` (562,298,827 B, md5
  `fb252e63f9128879a13727a4154ef6e3`; `TESTS=1 RPC=1`, `ggml-rpc-server` beside llama-server, sm_61, no AVX, GLIBC_2.38).
- **On the CPU, remote devices give the local tokens exactly** ([cpu_checks.tsv](cpu_checks.tsv), [tokens.tsv](tokens.tsv)):
  a random-weight `qwen4exp` (Flash-Next's architecture: Gated DeltaNet, hyper-connections, the n-gram table, sparse
  attention, routed experts) served by llama-server with its layers on one `ggml-rpc-server`, split over two (the
  residual crossing between them inside the model), and part local, part remote: every greedy, cached-prefix and
  sampled request has the local run's tokens and top-5 probabilities to the bit (12 of 12 requests over the three
  layouts, largest difference 0); speculative decoding on the remote devices gives its local twin's tokens (4 of 4). The fork's F16 hop
  (`LLAMA_HOP_CAST=f16`) at the boundary between the two servers keeps the greedy reply (probabilities within 2.5e-7).
- **Every op of the architecture ran on the remote devices** ([remote_ops.tsv](remote_ops.tsv)), the fork's
  `GDN_CONV` among them, with `GATED_DELTA_NET`, `DSV4_HC_PRE`/`POST`, `LIGHTNING_INDEXER`, `TOP_K_MASK`,
  `FLASH_ATTN_EXT`, `MUL_MAT_ID`; only the token-embedding and n-gram-table gathers stayed on the client's CPU, where
  their weights are.
- **What a decoded token sends** ([commands.tsv](commands.tsv), [socket_bytes.tsv](socket_bytes.tsv)): 26 RPC
  commands, of which 21 input uploads without a reply, 2 graph re-runs by id and 3 reads that wait for a reply (two
  4 KB residual reads to relay between the servers, the logits), the same at 32 and 160 tokens of context; ~24 KB out
  and ~9 KB back on this small file.
- **Swift IQ3_S fits 18 cards with room** ([fit18.md](fit18.md), [fit18.tsv](fit18.tsv)): 1,780 MiB left on the fullest
  card at 64k and `-ub 512` (DFlash2 1,761), 1,188 at 128k, 967 at 262k and `-ub 256`, whole layers, F16 cache.
- **Swift IQ3_S's n-gram table is the rig's table** ([table_md5.tsv](table_md5.tsv)): the same 28,800,138,240 bytes
  (md5 `421370d4…`) as in the rig's Swift Q2_0 and IQ3_XXS sets, and in ISTA's IQ3_S too; with the same tensor count
  (1,224), Swift IQ3_S resharded with the table alone in a third shard reuses the table file already on rig 1's SSD.

## Files

| File | What it holds |
|---|---|
| [cpu_checks.tsv](cpu_checks.tsv) | one verdict line per check, both runs (run 1 split `-ts 1,1`: both blocks on one server and the output on the other; its op listing failed on the log's prefixes and its 2,048-token byte count overran the random file's 256-token context, both fixed in run 2, `-ts 1,2`) |
| [tokens.tsv](tokens.tsv) | run 2, per arm and request: identical or not against the reference, how many tokens, the largest top-5 probability difference |
| [remote_ops.tsv](remote_ops.tsv) | the ops of one decoded token's graph per device (two rpc-servers, the client's CPU), then each split's inputs and their sizes |
| [commands.tsv](commands.tsv) | the client's RPC commands per decoded token: count, bytes out and back, how many waited for a reply, their time on loopback |
| [socket_bytes.tsv](socket_bytes.tsv) | the client's TCP counters per decoded token per server (`ss -ti`), and the prompt's totals |
| [rpc_server_memory.tsv](rpc_server_memory.tsv) | each rpc-server's peak and final resident memory per arm |
| [fit18.md](fit18.md), [fit18.tsv](fit18.tsv) | the 18-card fit of Swift IQ3_S, ISTA IQ3_S and Swift IQ3_XXS (32k–262k, F16 and q8_0 cache, no draft, DFlash2, MTP; whole layers and routed experts moved); the planner names the cards CUDA0–17, which over RPC are RPC0–8 (rig 2) and CUDA0–8 (rig 1) |
| [table_md5.tsv](table_md5.tsv) | the n-gram table's type, shape, bytes and md5 in each file, and the md5 of its shard's header where it sits alone |
| `run/cpu_rpc_check.sh`, `run/rpc_tools.py` | the CPU check (build, model, tokens, ops, bytes) and its comparison, op and byte counters |
| `run/fit18.sh` | the fit on 18 cards through `bench/file_choice/fit.py` |
| `run/table_identity.py` | the table's md5 in each file |
| `run/reshard_iq3_s.sh` | Swift IQ3_S into three shards, the table alone in the third, checked against the rig's table file |
| `run/two_rigs.sh` | the run on the two rigs (Q816): network baseline, the RPC smoke test, 9 against 18 cards on Swift IQ3_XXS, Swift IQ3_S's smoke test with and without F16 hops, one agent-check arm |

## Commands

The fork's branch and the rig tarball, on a build machine with Docker (the public build script's new `RPC=1` adds
`-DGGML_RPC=ON` and `ggml-rpc-server`):

```bash
WORK=<work dir> TESTS=1 RPC=1 JOBS=6 DOCKER_ARGS="--cpu-shares=256" \
  bash scripts/build_llamacpp_docker.sh fork/pascal-p106-v3-b12076-rpc
```

The CPU check (`WORK/src` a clone of the fork at `pascal-p106-v3-b12076-rpc`; ~6 min after the build):

```bash
WORK=<work dir> REPO=<this repo> JOBS=6 THREADS=4 LOCK=<build lock file> \
  bash docs/data/2026-11-04_two_rigs/run/cpu_rpc_check.sh          # STEPS="tokens ops bytes" reruns without the build
```

The fit and the table check (headers, then one read of each table):

```bash
FILES="swift-iq3_s=<shard1>,<shard2> ista-iq3_s=<shard1>,<shard2> swift-iq3_xxs=<s1>,<s2>,<s3>" OUT=<dir> \
  bash docs/data/2026-11-04_two_rigs/run/fit18.sh
python3 -I docs/data/2026-11-04_two_rigs/run/table_identity.py swift-q2_0=<shards> swift-iq3_xxs=<shards> \
  swift-iq3_s=<shards> ista-iq3_s=<shards> > table_md5.tsv
SRC="<Swift IQ3_S shard 1> <shard 2>" OUT=<dir> TABLE_MD5=<md5 of the rig's table shard file> \
  bash docs/data/2026-11-04_two_rigs/run/reshard_iq3_s.sh
```

The rigs' run: both rigs on the RPC tarball, Swift IQ3_S's shards 1–2 on rig 1 with shard 3 a link to the table on
its SSD, the agent check's client folder and settings as in [the DFlash2 run](../2026-11-03_flash_next_dflash2/README.md)
(`run/s18.env.example`), then

```bash
set -a; . s18.env; set +a
RIG2=<rig 2's ssh alias> RIG2_ADDR=<rig 2's LAN address> RPC_BIN=<tarball>/bin/ggml-rpc-server RPC_CACHE=<rig-2 drive>/rpc-cache \
  REL_BIN=<tarball>/bin/llama-server MODEL=<Swift IQ3_S shard 1> MODEL_ID=<Swift IQ3_XXS shard 1> \
  bash docs/data/2026-11-04_two_rigs/run/two_rigs.sh <out dir>
```

`ggml-rpc-server` is unauthenticated: bind it to the rigs' own LAN address, never a public or VPN one.

## Provenance and licence

The CPU check's model is random weights written by the fork's `test-llama-archs` (`-a qwen4exp -s 1`). The fit and the
table check read the GGUF headers and the table bytes of Swift-Qwen3.8-Flash-Next GSQ-RCO (ukisai, swift-open-license-1.0)
and ISTA's Qwen3.8-Flash-Next GSQ-RCO files; no weights are published here. The per-card costs in the fit are
llama.cpp's own projection for Flash-Next on nine cards
([2026-11-03_flash_next_fit](../2026-11-03_flash_next_fit/README.md)). Tables and scripts: this repository's licence.
