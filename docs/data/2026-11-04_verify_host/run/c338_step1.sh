#!/usr/bin/env bash
# C338 step 1 (Q900): what a real verify step costs beyond the cards' GPU time. One load of S34's MTP line (pmix-q2-fast,
# -ladder3, GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=32768: S33's mtp2d line) through docs/data/2026-11-03_flash_next_dflash2/run/
# smoke_dflash.sh (the RAM watchdog, floor 60), one short untimed request (S12's smoke request), with the fork's host
# profile GGML_CUDA_PROFILE_HOST (per card and graph evaluation: host entry and exit of graph_compute and the GPU time
# between two events around its work; no synchronize), not the kernel profiler, which slows the pass. Then
# run/host_steps.py on the file, on any machine with python3.
# Run from the repository root. ENVFILE is S12's machine file (docs/data/2026-11-03_flash_next_mtp/run/s12.env.example,
# with DRAFT); MODEL the file's first shard on the rig; BUILD the -ladder3 folder under /opt/llama-builds; OUT local.
set -uo pipefail
: "${ENVFILE:?}" "${MODEL:?}" "${BUILD:?}" "${OUT:?}"
M=$MODEL
set -a; . "$ENVFILE"; set +a
export MIN_AVAIL_MB=60 MODEL=$M REL_BIN=/opt/llama-builds/$BUILD/bin/llama-server
LD=/tmp/c338/smoke
MT="-b 2048 -ub 512 -ts 6,6,6,6,5,6,5,6,3 -fit off -lm dio -md $DRAFT --spec-type draft-mtp --spec-draft-n-max 3 -otd token_embd=CUDA4 --spec-draft-n-ubatch 128 --no-host"
LONG_REQUEST=off REQUESTS=docs/data/2026-11-03_flash_next_mtp/run/smoke_request.json ARGS="$MT" \
  SMOKE_ENV="GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=32768 GGML_CUDA_PROFILE_HOST=$LD/host_profile.csv" SMOKE_LOGDIR=$LD \
  bash docs/data/2026-11-03_flash_next_dflash2/run/smoke_dflash.sh "$OUT/smoke"
python3 -I docs/data/2026-11-04_verify_host/run/host_steps.py "$OUT/smoke/host_profile.csv" > "$OUT/steps.md"
