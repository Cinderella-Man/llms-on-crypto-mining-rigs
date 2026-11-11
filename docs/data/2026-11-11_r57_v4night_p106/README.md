# The v4 night build on a P106, Flash-Next's preset, against v3.0 (R57, 2026-11-11)

What this is: the v4 line's first run on a P106 rig. The build is Miner Llama `pascal-p106-v4-b12246-night`
`c408dabcb` ([the night merge](../2026-11-11_v4_night_merge/README.md)): the v3.0 release tree merged onto upstream
llama.cpp b12246 ([`-ladder6`](../2026-11-10_ladder6/README.md)) plus the night's 17 engine branches, every one of them
behind a switch that is off here. The reference is the v3.0 release build `pascal-p106-v3-b12076-ladder5` `1551f5fe1`.
Both run Flash-Next's v3 preset ([release/flash-next-v3/settings.env](../../../release/flash-next-v3/settings.env)):
Swift-1.5 GSQ-RCO Q2_0 on nine P106-090 cards with the MTP draft, a 65,536-token context. The run has three parts, in
this order: the load (memory per card, and where upstream #31927 puts the draft's hidden rows), greedy identity, and the
agent check ([bench/agent_check](../../../bench/agent_check/README.md), 12 bugfix tasks an arm) on both builds. The
registry row is Q1224 in [quants/experiments.md](../../../quants/experiments.md).

Every speed below is one measured run, with its build and settings named.

**Result:** the night build runs Flash-Next's preset on the P106s as v3.0 does. It loads with 82 MiB more on card 8
and the same memory to the MiB on cards 0 to 7: upstream #31927's buffer for the draft's hidden rows,
80.04 MiB on card 8, so the rows stay on the card. Greedy decoding gives the same bytes as v3.0 on both prompts. On the
agent check it passes 12 of 12 tasks and does a task's work in 2.75 minutes at equal work, the same as v3.0's 2.75
today (+0.0 %); without the one task both builds ran to the 12-minute cap, 1.90 against 1.91 minutes (−0.5 %).
v3.0's own run of 2026-11-06 on the same rig, file and preset sits +0.7 % and +2.1 % from today's. **Class:
negligible.** The difference is within the spread between two runs of v3.0, and the build is not measurably slower.

## The builds

| Build | Commit | Upstream base | What |
|---|---|---|---|
| Miner Llama `pascal-p106-v4-b12246-night` | `c408dabcb` | b12246 | the v4 night merge; rig tarball md5 `736363d6ca12e6c81ff5f5b684ef31b3`, every file on the rig checked against it by md5 before the run |
| Miner Llama `pascal-p106-v3-b12076-ladder5` | `1551f5fe1` | b12076 + 16 commits | the v3.0 release (tag `miner-llama-v3.0`) |

## The load: upstream #31927's buffer on card 8

Upstream #31927 (in b12246) lets the MTP draft read the target model's hidden rows from a buffer on the card instead of
copying them to host memory: 2,049 rows of 10,240 F32 values at `-b 2048`, est. 80.0 MiB, on the card that holds the
target's output (card 8, which also holds the draft). When it cannot, the load log warns that the rows go through host
memory.

One llama-server per build with the preset exactly (`run/r57.sh`, step `load`), memory per card read with `nvidia-smi`
20 s after the server first answered ([load_memory.tsv](load_memory.tsv)):

| Card | v3.0 `1551f5fe1`, MiB | night `c408dabcb`, MiB | Difference |
|---|---|---|---|
| 0 | 5,059 | 5,059 | 0 |
| 1 | 5,095 | 5,095 | 0 |
| 2 | 4,963 | 4,963 | 0 |
| 3 | 5,111 | 5,111 | 0 |
| 4 | 4,875 | 4,875 | 0 |
| 5 | 4,979 | 4,979 | 0 |
| 6 | 4,381 | 4,381 | 0 |
| 7 | 4,991 | 4,991 | 0 |
| 8 | 4,565 | 4,647 | **+82** |
| all nine | 44,019 | 44,101 | +82 |

- **The buffer is on card 8, and nothing else moved:** cards 0 to 7 hold the same memory to the MiB on both builds;
  card 8 holds 82 MiB more, the estimated 80.0 MiB buffer and its allocation. Card 8 keeps 1,497 MiB free (of 6,144).
- **The rows stay on the card.** At llama-server's default verbosity (3) the library's warnings print and its INFO lines
  do not; the night build's load log has no warning that the rows go through host memory. One more load of the
  night build at `-lv 4` (step `loadlv`) prints the line, `h_nextn_dev_alloc: CUDA8 h_nextn buffer size = 80.04 MiB`,
  and every other buffer per card ([load_buffers_v4night.txt](load_buffers_v4night.txt)); its memory per card is the
  first load's to the MiB. Upstream's buffer is the default here. The fork's switch `LLAMA_MTP_H_DEV`
  ([C693](../2026-11-11_c693_upstream_switches/README.md)) turns it off for a later comparison.
- Both servers answered after 226 s (night) and 225 s (v3.0). The host's MemAvailable 20 s later: 918,544 kB (night)
  and 942,516 kB (v3.0), one reading each.

## Greedy identity

[run/rig_greedy.sh](../2026-11-05_ladder5/run/rig_greedy.sh) of the v3.0 release tests: `llama-completion` decodes 256
tokens at temperature 0 (one token a pass) from one prompt with each build, every switch at its default, Flash-Next's
no-draft split (`-ngl 99 -fa on -ts 6,5,5,6,5,5,6,5,6 -fit off -lm dio`, `-c 4096`), and compares the generated text
byte for byte. Two prompts: a coding request (the script's default) and a prose paragraph with one Python example
([greedy.tsv](greedy.tsv)).

| Prompt | v3.0 `1551f5fe1` | night `c408dabcb` | Bytes | Verdict |
|---|---|---|---|---|
| code | `4384ebc02147be1ee6f38d1ce4a43f21` | `4384ebc02147be1ee6f38d1ce4a43f21` | 745 | the same |
| prose | `b21ce2b8a3573b92754e8cac6068b811` | `b21ce2b8a3573b92754e8cac6068b811` | 1,200 | the same |

Both prompts give the same bytes on both builds: upstream's changes between b12076 and b12246 on the path these runs
take (the fused MoE router, MMQ, the lightning indexer) change no greedy token of either reply. Without `-bs`,
`llama-completion` samples on the host, so device sampling's new radix top-k is not on this path; the preset turns
device sampling on (`-bs`), and the agent check below runs it. As a diagnostic only, a decode step took 70.51 and 70.17
ms on the night build against 73.37 and 73.72 ms on v3.0 (code, prose; one load each).

## The agent check

The preset on each build, one arm each, back to back on one rig: `v4night` (night `c408dabcb`) 08:55–09:35, then
`v3rel` (v3.0 `1551f5fe1`) 09:35–10:12. Both arms take exactly the server line and switches of v3.0's release arm
`t1swift` ([2026-11-05_release_flashnext](../2026-11-05_release_flashnext/README.md), `run/rig_t1.sh`): the 12 bugfix
tasks, one attempt each, a 12-minute cap a task, a 40-minute budget an arm (neither arm reached it), the client on
another machine at `--cpu-shares 256`. `t1swift` itself (v3.0 `1551f5fe1`, the same rig, file and preset, 2026-11-06) is
put beside them as a second run of v3.0 ([run/merge_arm.sh](run/merge_arm.sh)), which shows how far two runs of one
build lie apart. The equal-work column times one fixed amount of work for every arm: `v3rel`'s prompt tokens at the
arm's prompt speed, plus `v3rel`'s generated tokens at the arm's generation speed, plus `v3rel`'s time outside the
server, divided over `v3rel`'s tasks ([agent_tables.py](../2026-11-05_keepalive/run/agent_tables.py)).

All 12 tasks ([agent.tsv](agent.tsv)):

| Arm | Build | Passed | Gen tok/s | Prompt tok/s | TTFT median, p90 s | Acceptance | Equal work, min a task |
|---|---|---|---|---|---|---|---|
| `v3rel` | v3.0 `1551f5fe1`, 2026-11-11 | 11/12 | 20.97 | 68.7 | 7.50, 16.39 | 0.700 | 2.75 |
| `t1swift` | v3.0 `1551f5fe1`, 2026-11-06 | 12/12 | 20.40 | 69.4 | 6.89, 15.89 | 0.718 | 2.77 (+0.7 %) |
| `v4night` | night `c408dabcb`, 2026-11-11 | 12/12 | 20.65 | 69.8 | 6.94, 15.23 | 0.733 | 2.75 (+0.0 %) |

One task, `bugfix-javascript-variable-length-quantity`, ran into the 12-minute cap on both builds today: `v4night`
passed it after 6 turns, `v3rel` failed it after 2 (its second reply ran on to the cap); `t1swift` passed it within the
cap. The 11 other tasks ([agent_11.tsv](agent_11.tsv)):

| Arm | Build | Passed | Gen tok/s | Prompt tok/s | TTFT median, p90 s | Acceptance | Equal work, min a task |
|---|---|---|---|---|---|---|---|
| `v3rel` | v3.0 `1551f5fe1`, 2026-11-11 | 11/11 | 20.94 | 69.0 | 7.01, 16.20 | 0.700 | 1.91 |
| `t1swift` | v3.0 `1551f5fe1`, 2026-11-06 | 11/11 | 20.32 | 68.6 | 6.74, 15.80 | 0.718 | 1.95 (+2.1 %) |
| `v4night` | night `c408dabcb`, 2026-11-11 | 11/11 | 21.03 | 69.0 | 6.59, 15.23 | 0.733 | 1.90 (−0.5 %) |

- **The night build against v3.0: negligible.** At equal work it is +0.0 % on all 12 tasks and −0.5 % on the 11,
  within the +0.7 % and +2.1 % between two runs of v3.0 on this rig, and not slower. Generation is 20.65 against
  20.97 tok/s on all 12 and 21.03 against 20.94 on the 11; prompt reading 69.8 against 68.7 and 69.0 against 69.0.
- Device sampling (`-bs`, with upstream's radix top-k on the night build) ran on every request of both arms: the
  server logs count no request with backend sampling off.
- The agent check's own report ([report_table.md](report_table.md), [compare.md](compare.md),
  [gonogo.md](gonogo.md)) marks two checks NO-GO, both the same on the two builds and as in v3.0's release run. The
  proxy rows: each arm's capped request was cancelled mid-reply, so it has a row without timings. Memory: the host's
  lowest MemAvailable was 206,680 kB in `v4night` and 197,336 kB in `v3rel`, each in its first minutes, with swap use
  peaking at 914,968 and 1,028,584 kB (v3.0's release run: 98 MB and 1,019 MB). No watchdog stop in either arm.

## Files

| File | What |
|---|---|
| [load_memory.tsv](load_memory.tsv) | memory per card with the preset loaded, 20 s after the server answered, per build |
| [load_buffers_v4night.txt](load_buffers_v4night.txt) | the night build's load lines at `-lv 4`: model, KV, recurrent-state, compute and draft buffers per card, and the `h_nextn` buffer |
| [greedy.tsv](greedy.tsv) | greedy identity: per prompt and build the md5 of the generated text, its bytes, the eval line |
| [agent.tsv](agent.tsv) | the agent check per arm, all 12 tasks, against `v3rel` |
| [agent_11.tsv](agent_11.tsv) | the same without the task both builds ran to the cap |
| [report_table.md](report_table.md), [compare.md](compare.md), [gonogo.md](gonogo.md) | the agent check's own report of `v3rel` and `v4night` |
| [run/r57.sh](run/r57.sh) | the run: the loads, greedy identity, the agent check of both builds, the report |
| [run/merge_arm.sh](run/merge_arm.sh) | one folder with this run's arms and v3.0's `t1swift`, for the tables |

## Commands

From the repository root, on a machine that only runs ssh. The machine values go in the environment: `RIG`, `CLIENT`,
`MODEL` (Swift Q2_0's first shard on the rig), `DRAFT`, `NEW` and `REF` (build folders under `/opt/llama-builds`),
`RIGDIR`, `OUT`, `RIG_PORT`, `RELAY_PORT`, `CLIENT_PORT`, `PROXY_PORT`, `CLIENT_DIR`, `CLIENT_OUT`, `CPU_SHARES=256`.

```bash
# the client's bench files, image and task validation (no rig step)
STEPS=prep bash docs/data/2026-11-11_r57_v4night_p106/run/r57.sh
# the two loads, greedy identity, the two agent arms (v4night, then v3rel), the report collected to OUT/agent
STEPS="load greedy agent report" bash docs/data/2026-11-11_r57_v4night_p106/run/r57.sh
# the night build's load at -lv 4
STEPS=loadlv bash docs/data/2026-11-11_r57_v4night_p106/run/r57.sh
# load_memory.tsv: OUT/load-<arm>/mem.csv of both loads side by side; greedy.tsv: OUT/greedy/greedy-flashnext-{code,prose}.tsv
# on the client: v3.0's t1swift beside this run's arms, then the tables
NOW=<CLIENT_OUT> OLD=<v3.0's release check CLIENT_OUT> OLD_ARM=t1swift DEST=<dir> \
  bash docs/data/2026-11-11_r57_v4night_p106/run/merge_arm.sh
python3 -I docs/data/2026-11-05_keepalive/run/agent_tables.py <dir> v3rel t1swift v4night
OUT_TAG=_11 SKIP_TASKS=bugfix-javascript-variable-length-quantity \
  python3 -I docs/data/2026-11-05_keepalive/run/agent_tables.py <dir> v3rel t1swift v4night
```

Provenance: measured on one Manli 9 × P106-090 rig (4 GB host), with the agent check's client on another machine. The
tasks are bench/agent_check's bugfix set. The tables are this repository's measurements, under its licence.
