# C865 on a P106: Ornith with `GGML_CUDA_MMID_GLU_UNFUSE` unset, `1` and `2` (Q1559, Q350; 2026-11-17, a partial run)

**No speed answer yet: the run was stopped before its timed part, and its loads run again as their own row.** Every
kernel case that finished passed: 65 with the switch unset and 4 with `=1`, none failed. The check spent 1 h 46 min
on the first of three settings, because the rig's two-core Celeron computes the CPU reference of every case at
Ornith's expert sizes. The deep lane was then stopped during its untimed warm-up load, so no load was timed.

The switch, its code and what it reaches on Ornith's file are in
[2026-11-17_c865_glu_unfuse](../2026-11-17_c865_glu_unfuse/README.md); the switch is listed in
[docs/switches.md](../../switches.md). Registry rows [Q1559](../../../quants/experiments.md) (the switch) and
[Q350](../../../quants/experiments.md) (its first arm). Ornith's step on v4.0, which says where a change should show,
is [2026-11-17_r176_ornith_step_map](../2026-11-17_r176_ornith_step_map/README.md): the fused routed gate, up and GLU
take 12.02 and 12.00 ms of a 53.72 and 53.73 ms token at depth 0.

## What was run

Nine P106-090 cards at stock clocks and their 75 W default power limit, Ornith-1.5-122B Pascal N2P GGUF (`valid1`)
with its v4 preset ([release/ornith-v4/settings.env](../../../release/ornith-v4/settings.env)), on one build for
every arm: fork `pascal-p106-v5` `a112712e3` (the v5 merge of three switches, each off by default;
[2026-11-17_v5_merge_switches](../2026-11-17_v5_merge_switches/README.md)), rig tarball md5
`d34dfe3a64d195d8dd068cfb3485c523`, `llama-server` md5 `42f0b1f2ff2de02cd2e83836b5ff9841`, `test-backend-ops` md5
`cfd5c96a77f587c10e22071889e98107`. With every switch unset, every kernel a P106 runs is v4.0's.

| Arm | Setting |
|---|---|
| `off` | the switch unset: v4.0's fusions |
| `u1` | `GGML_CUDA_MMID_GLU_UNFUSE=1`: the routed gate and up products apart, then the GLU; the shared expert fused |
| `u2` | `GGML_CUDA_MMID_GLU_UNFUSE=2`: upstream's open #32239, no fused matrix-vector product on Pascal |

1. **Kernel cases** (`STEP=tbo`, card 0, untimed), 17:11–18:58: `test-backend-ops -b CUDA0 -o
   MUL_MAT_VEC_FUSION,MUL_MAT_ID_SHARED,MUL_MAT_ID,ADD_RMS_NORM_MUL_Q8 -p 'type(_a)?=(n2p|n3p|q5_0),'` per arm, then
   `-o GLU`.
2. **The deep lane** (`STEP=start`, [bench/deep_lane](../../../bench/deep_lane/README.md)), from 18:58: loads
   `warm:off f1:off a1:u1 b1:u2 f2:off a2:u1 b2:u2`, depths 0 and 8,192, 256 tokens at temperature 0 twice per depth
   and a pass with the preset's sampler. Stopped at 19:00 by the rig's owner, so another run could use the rig.

## Result

### Kernel cases (`kernel_cases.tsv`)

| Arm | Cases run | OK | Failed |
|---|---:|---:|---:|
| `off` | 65: `MUL_MAT_ID` 44, `MUL_MAT_VEC_FUSION` 11, `MUL_MAT_ID_SHARED` 6, `ADD_RMS_NORM_MUL_Q8` 4 | 65 | 0 |
| `u1` | 4: `ADD_RMS_NORM_MUL_Q8` 4 | 4 | 0 |
| `u2` | none | — | — |

- The `off` arm's first part ran 1 h 40 min (17:11:00 to its last case at 18:50:54) and was stopped at 18:57:51
  while still in the N3P `MUL_MAT_ID` cases at Ornith's shapes (1,024 × 3,072, 32 experts, 1 and 2 tokens). The
  cases took about 1.6 min each, almost all of it the CPU reference on the Celeron (no AVX).
- **The `-o GLU` part ran no case** ("0/0 tests passed"): test-backend-ops names a GLU case by its kind
  (`ggml_op_desc`: `SWIGLU`, `GEGLU`, `REGLU` and the others), not `GLU`. Ornith's GLU is `SWIGLU`, so the filter is
  `-o SWIGLU`.
- Cases of N2P and N3P at Ornith's shapes therefore belong off the rig's CPU: the reference is recorded on another
  machine and only the card's half runs on the P106 (the record-and-replay checks of C372), or the kernel cases are
  left to the lane's same-reply check, which covers the whole model.

### The deep lane

No timed load: the warm-up load (`off`, untimed) was reading the 48 GB file when the run was stopped, and
`loads.tsv` has no line. The quiet check before it passed (no llama process, 0 MiB on every card). The run's card log
(`scripts/card_log.py`, every card every 0.5 s) covers only that file load, so it shows nothing about decode.

## Next

The timed loads run again as their own row (C865b): the same build and loads, est. 1 h 30 min (Ornith takes
540–670 s to load on this rig, R176). Then each arm's ms a token at 0 and 8,192 tokens, load by value, the class of
`u1` and `u2` against `off` by the spread of the loads, and the same greedy replies or the KL divergence.

## Files

| File | What |
|---|---|
| `kernel_cases.tsv` | every kernel case that finished: arm, switch, part, the case as test-backend-ops prints it, result |
| `run/c865.sh` | the run script (C865's runner; its header names the merged build): `STEP=tbo`, `dry`, `start`, `wait`, `tables`, `collect` |

## Commands

From the repository's root; `RIG` is the rig's ssh alias, `WORK` the run folder on the rig, `BIN` the build's
`llama-server` there (its `test-backend-ops` beside it), `MODEL` the Ornith file on the rig, `TEXT_LOCAL` the recorded
agent text (the release check's `proxy.jsonl.bodies.gz`, as R162 and R176):

```bash
export RIG=<rig> WORK=<folder on the rig> BIN=<build>/bin/llama-server MODEL=<Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf> \
  TEXT_LOCAL=<proxy.jsonl.bodies.gz>
R=docs/data/2026-11-17_c865_ornith_glu_unfuse/run/c865.sh
STEP=tbo bash $R      # the kernel cases (stopped here after 1 h 46 min; see above)
STEP=dry bash $R      # the three server lines
STEP=start bash $R    # the lane, detached on the rig
WORK=$WORK bash bench/deep_lane/remote.sh stop   # how it was stopped (c865.sh has no stop step)
OUT=<folder> STEP=collect bash $R
```

Provenance: the model file is Ornith-1.5-122B Pascal N2P GGUF (`valid1`), built from the maker's published weights
(see [release/ornith-v4](../../../release/ornith-v4/README.md)); the recorded agent text is this project's own release
check. Licence of the tables: as this repository.
