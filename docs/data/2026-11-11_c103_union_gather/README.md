# Q643: verify passes and prompt micro-batches read the union of their tokens' kept cells (2026-11-11)

Qwen3.8-Flash-Next's 12 QSA layers attend to about 2,051 kept cells per query token, and Miner Llama's sparse attention
below Turing ([Q230](../2026-11-04_qsa_sparse_attn/README.md)) reads only those. Each query token of a call read its
own list, so a cell that the 4 tokens of a draft verify all keep came from memory 4 times, and a cell that 128 tokens
of a prompt micro-batch keep came 128 times. That is why the switch's mode 1 took a verify only once the cache held
1.5 × 2,051 cells per query token (12,306 cells for 4 tokens): the dense kernels read the cache once for all tokens.
The deep lane's sixth run ([Q1214](../2026-11-10_deep_lane/README.md#the-sixth-run-prompt-micro-batches-through-the-sparse-kernel-q1214))
sent 128-token prompt micro-batches through the per-token kernel and read prompts 46.7 % faster, and the prompt rule
([Q1246](../2026-11-11_c705_sparse_prompt_rule/README.md)) gave such calls a threshold of their own. This folder holds
the union kernel: a block takes several consecutive query tokens and reads the union of their kept cells once. It holds
its checks without a GPU, the check set for a P106 and the rig commands that check and time it.

**Status: written, built and checked on a CPU; the check set for a P106 is recorded; not yet run on any GPU.** Miner
Llama branch `pascal-p106-v4-b12246-c103` `4a3423134` (six commits on the prompt
rule's `pascal-p106-v4-b12246-c705` `baac08803`), behind `GGML_CUDA_FATTN_SPARSE_PASCAL_UNION` (off by default; then the
switch takes exactly the calls it took on `-c705`, and the per-token kernel keeps its source and machine code).

## The union kernel

`fattn-sparse-pascal-impl.cuh` holds the union kernel (`flash_attn_ext_sparse_pascal_union<D, type>`) beside the
per-token one, whose text is unchanged (so its machine code stays the same); the union kernel repeats its tile loop and
merge:

1. **One list per group of tokens.** The call's query tokens are taken in groups of `tpb` consecutive tokens of a
   sequence, as many as a block of 12 warps holds: a warp takes 4 query heads, so Flash-Next's 12 heads per KV head
   need 3 warps a token and `tpb` is 4 (4 heads per KV head: 8 tokens, the compaction's largest group; 8: 6; 16: 3).
   Upstream's compaction (`flash_attn_mask_to_sparse_indices`, which the Turing kernel already runs in groups of 8)
   lists the cells any token of a group keeps, in cache order, at most `tpb` × `n_kv_max` of them; instances for
   groups of 2 to 7 are added.
2. **Every warp of a block walks that list in the same order.** With Flash-Next's 3 warps a token, warp `y` takes
   the heads of warp `y % 3` for the group's token `y / 3` (`fasp_union_map`, shared with the CPU check). The block's
   12 warps read each K or V row at about the same time, so it comes from memory once and then from the SM's cache, as
   the per-token kernel already shares a row among one token's 3 warps. Each warp scores with its own token's mask
   row, so a cell its token does not keep scores −∞.
3. **A tile without any of a token's cells is skipped.** Such a tile would leave the warp's running maximum, sum and
   accumulators exactly as they are (factor 1, weights 0), so skipping it changes no bit and saves its reads.
4. **The merge is the per-token kernel's:** each warp writes its partial output per split, and the last block of a
   group's column merges every split for each of the group's tokens.

The output of each token equals the per-token kernel's to rounding: the token's cells are the same, but tiles and
splits cut the list elsewhere, so the online softmax rescales at other points. Where every token of a group keeps the
same cells and the split count is the same, the two kernels compute the same bits (checked below).

| Variable | Values |
|---|---|
| `GGML_CUDA_FATTN_SPARSE_PASCAL_UNION` | `0` off (default); `1` a call of two or more query tokens that the switch takes runs the union kernel |
| `GGML_CUDA_FATTN_SPARSE_PASCAL_UNION_MIN_RATIO` | default 1.5: with `_UNION`, mode 1 takes a decode or verify call from this × `n_kv_max` cells per union list (one list for a verify of up to 4 tokens) instead of `_MIN_RATIO` per query token |
| `GGML_CUDA_FATTN_SPARSE_PASCAL_FAULT=4` | testing the tests: the union kernel scores each token with the next token's mask row |

Prompt micro-batches keep the prompt rule (`_PROMPT`, `_PROMPT_NQ`, `_PROMPT_MIN_RATIO`): its threshold is already per
call.

At Flash-Next's 2,051 kept cells and 12 query heads per KV head, mode 1 takes a call from these caches
([rule.txt](rule.txt); `never`: the dense kernels keep it):

| Setting \ query tokens | 1 | 2 | 3 | 4 | 5 | 8 | 9 to 128 | 129, 512 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `=1` (the switch before) | 3,077 | 6,153 | 9,230 | 12,306 | 15,383 | 24,612 | never | never |
| `=1 _MIN_RATIO=1.0 _PROMPT=1` | 2,051 | 4,102 | 6,153 | 8,204 | 10,255 | 16,408 | 2,051 | never |
| `=1 _UNION=1` | 3,077 | 3,077 | 3,077 | 3,077 | 6,153 | 6,153 | never | never |
| `=1 _UNION=1 _PROMPT=1` | 3,077 | 3,077 | 3,077 | 3,077 | 6,153 | 6,153 | 2,051 | never |
| `=1 _UNION=1 _UNION_MIN_RATIO=1.0` | 3,077 | 2,051 | 2,051 | 2,051 | 4,102 | 4,102 | never | never |
| `=1 _UNION=1 _UNION_MIN_RATIO=0.5` | 3,077 | 1,026 | 1,026 | 1,026 | 2,051 | 2,051 | never | never |

A verify of up to 4 tokens is one union list, so with `_UNION` it goes to the kernel from the cache size a decode
token does, where the per-token lists needed 12,306 cells (8,204 at the eighth run's ratio of 1.0). The cache sizes of
the eighth run's verify passes (about 4,352 cells at 4,096 tokens, 8,448 at 8,192, 12,544 at 12,288, est. from the
256-cell padding) all lie above 3,077. Decode calls keep `_MIN_RATIO`'s rule.

## Checks without a GPU

On the build machine's CPU, no GPU, every step under the night's shared lock at a low CPU share
([check.txt](check.txt), [rule.txt](rule.txt)):

| Check | Result |
|---|---|
| Fork build (`TESTS=1`, sm_61 + compute_61 PTX) | `4a3423134`: compiles; tarball 649,847,005 B, md5 `c83704625726bdd9246796acebca0fc0` |
| Registers ([registers.tsv](registers.tsv), ptxas for sm_61 in `fasp_check`'s build of the same header) | the per-token instances as on `pascal-p106-v4` (113, 118, 158, 161); the union instances 114, 120, 159, 162 under their cap of 168 (384 threads a block), no stack, no spills |
| `fasp_check`, 58 cases: the per-token kernel's 26 and the union kernel's 32 | **58 / 58 pass**: union cases within 2.26·10⁻⁶ of the largest output against FP64 and NMSE at most 4.55·10⁻⁵ against ggml's CPU kernel |
| Union against per-token on the same problem (the per-token model at its own splits) | largest difference 7.77·10⁻⁷ of the largest output (rows sharing 90 % of their pools, Q8_0); **bit for bit** in the 4 cases where every row keeps the same cells (4 and 128 tokens, F16 and Q8_0) |
| Union lists in the cases | a 4-token verify on 32,768 cells whose rows share 90 % of their pools: 2,655 cells against 2,051 per token (1.29×); with independent selections per row the first list of a case holds up to 5,727 |
| Faults 1, 2, 3 | 58 failing cases each |
| Fault 4 (each token scored with the next token's mask) | the 32 union cases fail, the per-token ones pass |
| Six source mutations of the header | Q8_0 rows without their scale: 29 failing; splits' sums merged unscaled: 36; every split but the first skipping an entry: 36; scores mapped to the wrong key: 58; **union warps on the wrong token of their list: 32; every sequence reading the first sequence's union lists: 2** (the two two-sequence cases) |
| `fasp_check --rule`, 454,896,000 settings × calls | pass: with `_UNION` off 0 decode or verify calls differ from the switch before, 0 prompt calls from the prompt rule; with `_UNION` on 0 calls of one token a list differ from the switch before and 0 calls of more from the union rule (12,405,528 taken) |
| Six mutations of the rule | each fails `--rule` (the prompt rule's three, and union lists counted per query token, the union rule with `_UNION` off, one-token calls by the union rule) |
| GPU code against `-c705`'s, kernel by kernel ([code.txt](code.txt), `cuobjdump -sass` and `-ptx` of `llama-server` and `test-backend-ops`) | **every one of the base's 8,346 functions (8,765 copies) the same in all four listings**, the per-token sparse kernel included; 10 new: the union kernel's 4 instances and the compaction's groups of 2 to 7 |

## The check set for a P106

`scripts/record_checks.sh`'s suite `qsa-union`: every `FLASH_ATTN_EXT` case with the `n_kv_max` hint, recorded with the
branch's own `test-backend-ops` (md5 `b3d8a3b5adfedd5b2669b707f98a5f23`) in a container without a GPU, so a P106 replays
them against the stored CPU references ([scripts/replay_checks.sh](../../../scripts/replay_checks.sh)).

| | |
|---|---|
| Cases | 135 ([set.tsv](set.tsv)): the 50 new union cases (25 shapes × F16/Q8_0) and the 85 sparse-hinted cases of the prompt rule's set (decode, verify and prompt calls at Flash-Next's shape, the other head shapes, the gather cases) |
| Size | 4,462,731,372 B of cases; the tar 4,463,237,120 B, md5 `4f7fa912d83889c50cb84a353abecea1`, equal to the folder (`tar -d`) |
| Recorded | 31 s, 8 workers, at nice 19 |
| Runs on a P106 ([runs.tsv](runs.tsv)) | `qu-stock` (`=0`), `qu-sparse1` (`=1`, per-token lists), `qu-union` (`=1 _UNION=1`), `qu-uprompt` (and `_PROMPT=1`), `qu-ur05` (`_UNION_MIN_RATIO=0.5`, `_PROMPT_NQ=512`), `qu-m2` (`=2 _PROMPT=1 _PROMPT_NQ=512`, per-token lists), `qu-m2-union` (the same through the union kernel), `qu-m2-union-s1` and `qu-m2-union-s32` (1 and 32 splits), `qu-fault1` (passes: within the test's limit), `qu-fault2`, `qu-fault3` and `qu-fault4` (must fail cases) |
| Through the compute_61 PTX ([runs-ptx.tsv](runs-ptx.tsv)) | `qu-stock` and the eight mode-2 runs |

Which kernel each multi-token case takes in each run, from the rule (`S` the per-token kernel, `U` the union kernel,
`d` the dense ones; [rule.txt](rule.txt), its last table); `qu-stock` takes `d` everywhere, `qu-m2` `S` and
`qu-m2-union` `U` everywhere:

| Query tokens, cells, `n_kv_max`, sequences (head 256, 12 heads per KV head) | `qu-sparse1` | `qu-union` | `qu-uprompt` | `qu-ur05` |
|---|---|---|---|---|
| 2, 3, 4 on 4,096, 2,051, 1 | d | U | U | U |
| 5, 8 on 4,096 | d | d | d | U |
| 128 on 4,096 | d | d | U | U |
| 2 on 12,288 | S | U | U | U |
| 4, 5, 8 on 12,288 | d | U | U | U |
| 128 on 12,288 | d | d | U | U |
| 2, 4, 8 on 49,152 | S | U | U | U |
| 128 on 49,152 | d | d | U | U |
| 6 on 8,192, two sequences | d | U | U | U |
| 2 on 8,192 (the per-token set's verify cases) | S | U | U | U |
| 3, 5, 8 on 8,192 | d | U | U | U |
| 4 on 4,096, two sequences | d | U | U | U |
| 9 to 128 (the prompt set's cases) | d | d | U | U |
| 129, 256, 512 on 4,096 | d | d | d | U |
| head 128, 4 heads per KV head: 8 on 4,096, 333 | S | U | U | U |
| the same, 13 tokens | d | d | U | U |
| head 128, 8 heads per KV head, sinks: 7 on 4,096, 1,027 | d | U | U | U |
| 16 heads per KV head: 5 on 4,096, 31 | S | U | U | U |

## On a P106

Two parts on one rig, nothing else running on it (registry row Q1310).

**1. The native check** (est. 35 min): the set on card 0 with the branch's own `test-backend-ops`. It passes when every
console line ends `ok` (`qu-fault2`, `qu-fault3` and `qu-fault4` must fail cases, every other run passes every case),
and when each case's error in a run equals its error in `qu-stock` where the table above says `d` and differs where it
says `S` or `U` (the per-case errors are in `OUT/<run>.tsv`). Then `fasp_check --gpu --nsm 5` (every case of the CPU
check, the per-token and the union kernel on the card, each launched twice) and `fasp_check --bench --nsm 5`: µs per
call of the union kernel against the per-token one on the same call, for selections that share 90 % of their pools
and for selections that share nothing beyond chance.

**2. The timing on the deep lane** (est. 3 h 20 min): [bench/deep_lane](../../../bench/deep_lane/README.md) on the
branch's build, the release preset's line with the sparse switch's `_NSM=5`, two loads an arm after a warm-up load,
256 tokens a decode request at 0, 4,096, 8,192, 16,384, 32,768 and 49,152 tokens. The arms take the verify side of the
eighth run ([Q1212](../2026-11-10_deep_lane/README.md#the-eighth-run-the-sparse-kernels-threshold-for-verify-passes-q1212))
and the prompt side of the sixth:

| Arm | Switch | `-ub` | What it shows |
|---|---|---:|---|
| `ub512` | `=1` | 512 | the preset (control) |
| `r100` | `=1 _MIN_RATIO=1.0` | 512 | the eighth run's best verify line, per-token lists (a 4-token verify from 8,204 cells) |
| `u150` | `=1 _UNION=1` | 512 | union lists for verify passes (a 4-token verify from 3,077 cells) |
| `pr128` | `=1 _MIN_RATIO=1.0 _PROMPT=1` | 128 | the prompt rule with per-token lists (R73's arm) |
| `pu128` | `=1 _UNION=1 _PROMPT=1` | 128 | the prompt rule with union lists |
| `pu512` | `=1 _UNION=1 _PROMPT=1 _PROMPT_NQ=512` | 512 | 512-token micro-batches through the union kernel |

Read: the repeated step at each depth (`deep_table.py`; `u150` against `r100` and `ub512`, most of all at 4,096 and
8,192, where the per-token lists keep a verify on the dense kernels) and prompt tok/s per depth (`prompt_depth.py`,
control `ub512`; `pu128` against `pr128`, `pu512` against both). The union gather does what it is for when `u150`'s
step is at most `r100`'s at every depth and below it at 4,096, and when `pu128` reads prompts faster than `pr128` from
16,384 tokens on; `pu512` then says whether a whole 512-token micro-batch through the kernel beats 128-token ones. Its
step at depth 0 must stay `ub512`'s: the decode calls are the same.

## Files

| File | What |
|---|---|
| [run/night.sh](run/night.sh) | the build, the CPU check, the GPU code against the base build kernel by kernel, and the recording, on a build machine, CPU only, each step under a lock |
| [run/funcs.py](run/funcs.py) | compares two `cuobjdump` listings kernel by kernel (the per-token kernel under its new template name) |
| [rule.txt](rule.txt) | `fasp_check --rule`: the grid's verdict, the switch points, the kernel each prompt and union case takes per replay run |
| [check.txt](check.txt) | `bench/fattn_sparse/check.sh`'s verdicts and every run's table |
| [code.txt](code.txt) | the GPU code of `llama-server` and `test-backend-ops`, branch against base, kernel by kernel |
| [registers.tsv](registers.tsv) | registers, stack and spills of every instance of the sparse kernel (ptxas, sm_61) |
| [set.tsv](set.tsv) | the check set: one line per recorded case (id, case string, input and reference bytes) |
| [runs.tsv](runs.tsv), [runs-ptx.tsv](runs-ptx.tsv) | the set's runs on a P106 and through the compute_61 PTX on a newer card |
| [bench/fattn_sparse](../../../bench/fattn_sparse/README.md) | the CPU check (the union kernel's host model and cases, fault 4, five new mutations, the union rule in `--rule`, the union bench are new here) |
| [scripts/record_checks.sh](../../../scripts/record_checks.sh), [scripts/replay_checks.sh](../../../scripts/replay_checks.sh) | the recording (suite `qsa-union` is new here) and the replay |

Test cases added to `tests/test-backend-ops.cpp`: Flash-Next's QSA call (head 256, 2 × 12 heads, 2,051 cells) at 2 to
8 and 128 query tokens on 4,096 and 12,288 cells and at 2, 4, 8 and 128 on 49,152, 6 tokens in two sequences on 8,192, head 128 with 4 query
heads per KV head at 8 and 13 tokens, with 8 and sinks at 7, and head 256 with 16 at 5 tokens and 31 cells, F16 and
Q8_0.

## Commands

The night's steps on a build machine with Docker, CPU only ([run/night.sh](run/night.sh); `WORK/src` a clone of
llama.cpp with the fork as remote `fork`, `BASE` the base build's rig tarball, `LOCK` a lock file other jobs share;
`funcs.py` needs Python 3):

```bash
WORK=<work folder> REPO=<this repository> BRANCH=pascal-p106-v4-b12246-c103 LOCK=<lock file> \
  BASE=<work>/llama.cpp-20261110-baac08803-pascal-p106-v4-b12246-c705.tar.gz DOCKER_ARGS="--label <job>=1" \
  setsid -f nohup bash docs/data/2026-11-11_c103_union_gather/run/night.sh > night.out 2>&1 < /dev/null
```

Each step alone: `STEPS=build`, `STEPS=check`, `STEPS=code`, `STEPS=record` (the record step is
`OUT=<work>/sets SUITE=qsa-union BIN=<tree>/bin bash scripts/record_checks.sh`). The CPU check by itself:

```bash
SRC=<llama.cpp tree of the branch> DOCKER_ARGS="--cpu-shares=256" bench/fattn_sparse/check.sh
bench/fattn_sparse/build/fasp_check --rule      # the rule alone, seconds
```

**On the rig.** The build, the set and `fasp_check` (built by `check.sh` from the branch's header, CPU code without
AVX and the CUDA runtime linked in), relayed from the build machine (`CLIENT`) through the operator's machine, with the
md5s compared on arrival:

```bash
T=llama.cpp-20261111-4a3423134-pascal-p106-v4-b12246-c103
ssh $CLIENT "cat $WORK/$T.tar.gz" | ssh $RIG "tee >(md5sum > /tmp/$T.md5) | tar xzf - -C /opt/llama-builds; cat /tmp/$T.md5"
ssh $CLIENT "cat $WORK/sets/qsa-union.tar" | ssh $RIG "mkdir -p $DIR && tee >(md5sum > $DIR/qsa-union.md5) | tar xf - -C $DIR"
ssh $CLIENT "cat $WORK/fasp_check" | ssh $RIG "cat > $DIR/fasp_check && chmod +x $DIR/fasp_check && md5sum $DIR/fasp_check"
ssh $RIG "cat > $DIR/replay_checks.sh" < scripts/replay_checks.sh
```

The replay, detached, then `fasp_check` on card 0 (its CPU references take the Celeron about 10 min):

```bash
ssh $RIG "SET=$DIR/qsa-union BIN=/opt/llama-builds/$T/bin OUT=$DIR/out DEV=0 CPU_CHECK=200 \
  setsid nohup bash $DIR/replay_checks.sh > $DIR/replay.log 2>&1 < /dev/null &"
ssh $RIG "cd $DIR && CUDA_VISIBLE_DEVICES=0 setsid nohup sh -c './fasp_check --gpu --nsm 5 --threads 2 > fasp_gpu.txt 2>&1; \
  ./fasp_check --bench --nsm 5 --threads 2 > fasp_bench.txt 2>&1' > /dev/null 2>&1 < /dev/null &"
```

Through the compute_61 PTX on a newer card (mode 1 leaves such a card on its own kernels, so the list holds the mode-2
runs): `RUNS_FILE=$DIR/qsa-union/runs-ptx.tsv` with the same replay command in a container with one card, and
`CUDA_FORCE_PTX_JIT=1 fasp_check --gpu --nsm 5`.

The deep lane, with the deep lane's `wait` and `collect` commands
([2026-11-10_deep_lane](../2026-11-10_deep_lane/README.md#commands); `DRY=1` first: its six server lines differ only
in `-ub` and the switch's variables):

```bash
S=GGML_CUDA_FATTN_SPARSE_PASCAL
RIG=rig BIN=/opt/llama-builds/llama.cpp-20261111-4a3423134-pascal-p106-v4-b12246-c103/bin/llama-server \
MODEL=<models>/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf \
MODEL_DRAFT=<models>/mtp-Qwen3.8-Flash-Next-draft.gguf \
TEXT_LOCAL=<recorded>/proxy.jsonl.bodies.gz TAG='^t1swift/' WD_ENV=MIN_AVAIL_MB=60 \
DEPTHS="0 4096 8192 16384 32768 49152" N_PREDICT=256 ENV_ADD=${S}_NSM=5 \
ARMS="ub512 r100 u150 pr128 pu128 pu512" \
ARM_ENV_ub512="$S=1" \
ARM_ENV_r100="$S=1 ${S}_MIN_RATIO=1.0" \
ARM_ENV_u150="$S=1 ${S}_UNION=1" \
ARM_ENV_pr128="$S=1 ${S}_MIN_RATIO=1.0 ${S}_PROMPT=1" ARM_UB_pr128=128 \
ARM_ENV_pu128="$S=1 ${S}_UNION=1 ${S}_PROMPT=1" ARM_UB_pu128=128 \
ARM_ENV_pu512="$S=1 ${S}_UNION=1 ${S}_PROMPT=1 ${S}_PROMPT_NQ=512" \
LOADS="warm:ub512 ub512_1:ub512 r100_1:r100 u150_1:u150 pr128_1:pr128 pu128_1:pu128 pu512_1:pu512 ub512_2:ub512 r100_2:r100 u150_2:u150 pr128_2:pr128 pu128_2:pu128 pu512_2:pu512" \
bash bench/deep_lane/remote.sh start
```

Its tables:

```bash
python3 -I bench/deep_lane/deep_table.py <results folder> --out table.md --tsv classes.tsv
python3 -I docs/data/2026-11-10_deep_lane/run/prompt_depth.py <results folder> --control ub512 --out prompt_depth.md --tsv prompt_depth
python3 -I docs/data/2026-11-10_deep_lane/run/host_memory.py <results folder> > host_memory.tsv
```

## Provenance and licence

Code: Miner Llama (MIT, like llama.cpp), branch `pascal-p106-v4-b12246-c103`. The check's inputs are random tensors
drawn by test-backend-ops and `fasp_check`; no model or text is in the set. The rig row's model: Swift-1.5 GSQ-RCO
Qwen3.8-Flash-Next Q2_0 ([ukisai](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF), Swift Open
License 1.0 and qwen-community-1.0), as retyped by [Q218](../2026-11-03_retype/README.md), and the deep lane's recorded
agent text ([bench/deep_lane](../../../bench/deep_lane/README.md)).
