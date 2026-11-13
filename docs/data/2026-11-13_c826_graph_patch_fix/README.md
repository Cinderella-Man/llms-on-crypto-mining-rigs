# Forced prompt graphs that C779 can patch on a P106 (C826, Q1449, 2026-11-13)

Code and CPU checks of a fix to `GGML_CUDA_GRAPHS_PATCH` (C779), on the fork branch `pascal-p106-v4-b12246-c826`
(`b248a54e2`, one commit on `pascal-p106-v4` `2144cf760`), merged into `pascal-p106-v4` as `36ef48889` (on C818's `7663f3221`). It stays inside
C779's switch, off by default; with it unset, nothing changes. Its P106 check is R145 (Q1450), not run. Registry row:
[Q1449](../../../quants/experiments.md).

## The problem

In R136's part 1 on nine P106-090 cards, the patch load gave the force load's replies and its 40 verify steps, but
patched none of the 768 captures of a 32,768-token prime
([C779's rig check](../2026-11-12_c779_graph_replay/README.md#part-1-on-a-p106-2026-11-13-the-same-replies-and-steps-but-no-capture-patched)).
Every capture fell back to a full one:

- **`structure`, 705.** C779's check called a node changed in structure when any source (`src[j]`) or its view source
  was another tensor object. Under pipeline parallelism the scheduler gives each split input four copies and rotates
  them from one micro-batch to the next: each copy is a separate tensor at its own address, so every node that reads a
  split input changed its structure every micro-batch.
- **`api`, 62 of the 63 captures whose structure held.** CUDA refused a parameter update after 21 CUDA nodes had been
  set in each try (1,302 in all). The log did not name the call.
- **`nodes`, 1.**

## The change

Host code only, in `ggml/src/ggml-cuda/graph-patch.h`, `common.cuh` and `ggml-cuda.cu`, all inside the switch.

1. **A source is compared by its kind, not as an object.** The structure check now compares a node's op, type and
   flags and, for each of its sources and its view source, whether it is there and its kind: type, op, flags, buffer
   and `extra` (`ggml_cuda_graph_patch_kind_compare`). What a source holds (its data address, shape and strides) was
   already among the node's properties, so a rotated copy is now a parameter change: the node's group is dispatched
   again and its kernels get the new address. A source that becomes another tensor of the same kind still changes the
   node's properties (the node's copy holds its source pointers), so its group and the window before it are
   dispatched again and their fusion decisions are checked anew.
2. **A memset or memcpy whose executable already holds its new parameters is left as it is.** CUDA updates a memcpy
   in an instantiated graph only between 1-D operands, and a memset in little more than its address and value. A group
   dispatched again because a node near it changed often makes the same copy as before (a 2-D copy in top-k's argsort
   path, or a strided copy of the same type), and setting it anyway is refused. The captured graph each executable was
   instantiated from (its template) now mirrors the executable's memset and memcpy parameters: a node whose new
   parameters equal its template node's is kept; any other is set in the executable and then in the template
   (`ggml_cuda_graph_patch_action_for`). Kernels are set as before.
3. **Every refused update is named.** The first refusal of each kind is logged with its call, node type, the group's
   ggml op and node, the CUDA error, and the parameters (new and held):
   `GGML_CUDA_GRAPHS_PATCH: CUDA refused <call> <node type> <ggml op> <CUDA error>: node "<name>" (<index> of <n>, group
   of <k> ggml nodes), chunk <c>, CUDA node <j> of <m>: new ..., held ...`. Every 64 decisions, after the counts line,
   a line lists every refused kind with its count: `GGML_CUDA_GRAPHS_PATCH: refused updates: <call> <type> <op> <error>
   <count>; ...`. Each `api` fallback is exactly one refused update, so the two must add up.
4. **The counts line ends with the memsets and memcpys kept** (`; memset and memcpy nodes kept <n>`), and the first 16
   structure changes of a process (other than a new node count) are logged with their node and the field that changed:
   `GGML_CUDA_GRAPHS_PATCH: structure changed at node <i> of <n> (<name>, <op>): <field> of <the node | its source j |
   its view source>`.

[run/patch_counts.py](run/patch_counts.py) reads all of these from a server log: the counts lines (C779's, with an
empty `kept` column, and C826's), `--refused` (each refused kind with its count, node and parameters, and whether the
`api` count equals the refused updates counted) and `--structure`.

**Which call R136's P106 refused (est., from the code).** No copy between cards is captured into a graph: the
scheduler copies each split's inputs between cards before the split's graph runs, outside the capture. The memcpy
nodes a Flash-Next split captures come from same-type copies (`cpy`, 1-D, or 2-D when strided), top-k's argsort path (a
2-D copy of each row's first k entries), `argsort`, `solve_tri` and `concat` (1-D); memsets from the matrix kernels'
padding. The 2-D copies are the ones CUDA refuses to update at all, and their parameters hold from one micro-batch to
the next; est. R136's refusal was one of them, which change 2 now leaves alone. R145 names it either way.

## Results on a CPU

All on a build machine, CPU only, 2026-11-13: the branch tip `b248a54e2` 09:23–10:19, the merged tree `36ef48889` 10:20–10:38
([checks.md](checks.md)); nothing ran on a GPU.

| Check | Result |
|---|---|
| `ggml-cuda.cu` alone, compiled as the rig build compiles it (sm_61) | compiles, no warning |
| The rig tarball (`TESTS=1`; sm_61, no AVX) | built: `llama.cpp-20261113-b248a54e2-pascal-p106-v4-b12246-c826.tar.gz`, 678,730,034 B, md5 `c2cc27fa673fcc30a6c79e95fb9430b8`; 0 AVX instructions, highest symbol GLIBC_2.38; `bin/llama-server` md5 `f8b9d0a70038b44c483b1aa7b5bfb2cd`, `bin/test-backend-ops` `fa45c59032e2a640fa5b973dbc6f8944`; the base build's 34 warning lines |
| The switch list (C770) | 870 names, as before; `GGML_CUDA_GRAPHS_PATCH`, `_WINDOW` and `_MAX` among them |
| `llama-server`'s SASS against the `-c822` tarball's (`cc97bb6b6`, whose tree is `2144cf760`'s) | the same file: 8,540 functions |
| `llama-server`'s PTX against the same | 8,540 functions; the same 10 `flash_attn_tile` instances as in C779 differ in 70 lines, each the number nvcc gives a function-local `__shared__` array's name. With that number masked as well ([C779's ptx_names.sh](../2026-11-12_c779_graph_replay/run/ptx_names.sh)), the whole PTX dumps are the same file (md5 `80551b469e1173101aab20a1fb7ebee5`) |
| The static CPU build (every target) | builds; its 3 warning lines are the base's (`common/sampling.cpp`) |
| `test-cuda-graph-patch` | 24,545 checks, 0 failed, the same output in two runs (below) |
| `test-cuda-graph-force` (C702's, unchanged) | 6,070 checks, 0 failed |
| `test-llama-archs` against the base's logits (a static CPU build of `65ac9a47d`, to which `2144cf760` adds only CUDA host code) | 146 of 146 the same with no switch, with `GGML_CUDA_GRAPHS_FORCE=1`, with both switches, and with both at window 0 and a 100 % limit |
| C565's set, static and shared | ctest 72 of 72, the network test 1 of 1, archs 146 written and compared, the nine hand tests pass |
| The merged `pascal-p106-v4` (`36ef48889`: `-c826` on C818's `7663f3221`; it adds exactly `-c826`'s diff, and C818 changed none of its files) | a static CPU build of every target; `test-cuda-graph-patch` 24,545 of 24,545; `test-llama-archs` 146 of 146 the same as the base with every switch off and with the force and patch switches |

**The model test.** `tests/test-cuda-graph-patch.cpp` now also checks the kind comparison field by field (a rotated
copy is the same kind; a source that appears or goes, or another type, op, flags, buffer or `extra` of a source or the
view source, is a structure change, named by field and source) and the action for each CUDA node. Its model gained:
input readers whose source rotates through four separate tensor objects at their own addresses; a fusion of A and B
only when B reads A's tensor object, as `ggml_can_fuse` checks; nodes linked to another source (a property) and
sources that change kind (a structure change); a node T that makes a kernel and then a 2-D copy whose parameters
change only with T's shape, which the model's executable refuses to update, as CUDA does; and templates that mirror the
executable's memset and copy parameters. In 2,000 random runs of 300 micro-batches (600,000 evaluations):

| Run | Patched | Full captures by reason | Executables a fresh capture's |
|---|---|---|---|
| C826 (window 5, the longest fusion's reach) | 73,391 (42.3 groups dispatched again each; 571,184 memset and copy nodes kept) | not_ready 6,643, structure 8,774, share 216,579 (the runs at a 60 % limit), skip 0, pool 227,780, fuse 52,128, nodes 423, api 9,578 (2-D copies whose shape changed) | after every evaluation |
| C779's check by tensor object, the same runs | 11,876 | structure 531,499 | after every evaluation |
| Control: window 2 | | | not in 34,816 evaluations of 322 runs |
| Control: templates keep the recording's parameters | | | not in 2,465 evaluations of 71 runs: a memset left with stale parameters once its input copy comes round again |

The model's other full captures are its own design (a 60 % limit in half the runs, pool use held into the next group);
what it shows is that a rotated copy no longer forces a full capture, that keeping unchanged copies is exact only
while the template follows the executable, and that every patch gives a fresh capture's executables.

## The rig check (R145)

R136's part 1 again, on the `-c826` tarball (rig 1, est. 20 min): [R136's script](../2026-11-12_c779_graph_replay/run/r136_run.sh)
with `STEP=lane`, the force load (`GGML_CUDA_GRAPHS_FORCE=1`) and then the patch load (with `GGML_CUDA_GRAPHS_PATCH=1`
too), 16 tokens at depths 0 and 32,768, every verify step dumped. It passes when:

- most captures patch: in the patch load's last counts line, `patched` is more than half the decisions;
- the patch load gives the force load's greedy replies, and its dumps are the force load's at every verify step
  (R82's `dump_compare.py`, the force load as both controls);
- every refused update is named: `patch_counts.py --refused` ends with `named` (the `api` count equals the refused
  updates the log lists).

If it passes, R136's parts 2 and 3 run on the same tarball. If `structure` fallbacks remain, `--structure` names the
node and field; if `api` remains, `--refused` names the call, node and parameters.

## Commands

```bash
# the checks, on a build machine (CPU only; M a work folder, REPO a checkout of this repository, LOCK a flock file,
# BASE_TREE an unpacked rig tarball of pascal-p106-v4-b12246-c822 cc97bb6b6 (its tree is pascal-p106-v4 2144cf760's),
# BASE_CPU the bin/ of a static CPU build of 65ac9a47d (2144cf760 adds CUDA host code only), SEED_SRC a llama.cpp clone
# with the fork as remote "fork")
M=$M REPO=$REPO LOCK=$LOCK BASE_TREE=$BASE_TREE BASE_CPU=$BASE_CPU SEED_SRC=$SEED_SRC JOBS=8 \
    bash $REPO/docs/data/2026-11-13_c826_graph_patch_fix/run/checks.sh
# after the merge into pascal-p106-v4: the merged tree's "every switch off" check
M=$M ... STEPS=merged bash $REPO/docs/data/2026-11-13_c826_graph_patch_fix/run/checks.sh

# R145, from a machine that only runs ssh, at this repository's root (RIG, WORK, BUILDS, MODELS and RECORDED as in the
# script's header)
export BUILD=llama.cpp-20261113-b248a54e2-pascal-p106-v4-b12246-c826
STEP=lane bash docs/data/2026-11-12_c779_graph_replay/run/r136_run.sh
STEP=lanewait bash docs/data/2026-11-12_c779_graph_replay/run/r136_run.sh
STEP=lanecollect OUT=<here>/lane bash docs/data/2026-11-12_c779_graph_replay/run/r136_run.sh

# R145's reading, on a build machine's CPU, in a folder that holds the collected lane/
python3 -I $REPO/docs/data/2026-11-11_r82_c702_p106/run/dump_compare.py --control lane/dump-force.jsonl \
  --control2 lane/dump-force.jsonl --arm lane/dump-patch.jsonl --out lane/dump_compare.tsv
P=$REPO/docs/data/2026-11-13_c826_graph_patch_fix/run/patch_counts.py
python3 -I $P lane/server-patch1.log > lane/patch_counts.tsv
python3 -I $P --refused lane/server-patch1.log > lane/refused.tsv
python3 -I $P --structure lane/server-patch1.log > lane/structure.tsv
```

## Files

| File | What it holds |
|---|---|
| [checks.md](checks.md) | every check's line from the build machine (`summary.tsv`) |
| [run/checks.sh](run/checks.sh) | the checks: `ggml-cuda.cu` alone, a static CPU build, `test-cuda-graph-patch`, the rig tarball, the switch list, `llama-server`'s GPU code against the base's (SASS and PTX by function, and the whole PTX with internal names' numbers masked), `test-llama-archs` against the base's logits, C565's ctest set static and shared, the merged tree |
| [run/patch_counts.py](run/patch_counts.py) | a server log's `GGML_CUDA_GRAPHS_PATCH` lines as TSV: the counts, the refused updates, the structure changes |

## Limits

- **Not yet patched on a P106.** The CPU checks show that nothing changes with the switch unset or on a CPU, that the
  GPU code is the base's, and that in a model with rotating source objects and 2-D copies a patch gives a fresh
  capture's executables. Whether the P106's driver accepts the updates now, how many groups a split dispatches again
  and what that saves are R145's and R136's parts 2 and 3.
- **The refused call is an estimate** until R145's log names it.
- **A source that becomes another tensor of the same kind is a parameter change.** That is safe only because the node's
  group and the groups up to the window before it are dispatched again, so every fusion decision that reads it is made
  anew; a decision that reads further back than the window would not be. C779's window limit applies unchanged.
- **The template mirrors only the memset and memcpy parameters.** Its kernel nodes keep the recording's parameters;
  they are used only to name the executable's nodes.

## Provenance and licence

Code and checks of 2026-11-13, on the Miner Llama fork of llama.cpp (MIT). The checks ran on a build machine with no
GPU in use. No model output is in this folder. Licence: the repository's.
