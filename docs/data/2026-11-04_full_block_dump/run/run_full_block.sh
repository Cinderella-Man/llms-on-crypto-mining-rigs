#!/usr/bin/env bash
# Q804: DFlash2's acceptance at drafted positions 1-7 on recorded agent sessions, with the line drafting 2 tokens as
# the rig's S18 line does. llama-server (fork branch pascal-p106-v3-b12076-specdump2) drafts and verifies N_MAX = 2
# tokens a step, as the line does, and after each draft decodes the drafter's whole trained block a second time
# (LLAMA_SPEC_DUMP_FULL_BLOCK=1) so the dump holds the drafter's proposals at all 7 positions after the anchor; the
# target's own continuation, which scores positions 3-7, comes from the same request's next steps. Everything else is
# ../../2026-11-04_tree_replay/run/run_quant.sh, which takes the same environment (BIN, MODEL, DRAFT, OUT, LOG, REPO,
# GPUS, MOUNTS, ...; see its header): the smoke request whose dump must carry the second block, S18's recorded
# sessions re-sent in order, the tables by position (bench/tree_replay: block*.csv, summary.md).
# Usage: BIN=... MODEL=... DRAFT=... OUT=... LOG=... REPO=... [GPUS=2,3,4] MOUNTS='-v ...' bash run_full_block.sh
# Acceptance only: no timing from this run is a speed result (one more drafter pass per step, a shared machine).
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export N_MAX=${N_MAX:-2} FULL_BLOCK=${FULL_BLOCK:-1}
exec bash "$here/../../2026-11-04_tree_replay/run/run_quant.sh"
