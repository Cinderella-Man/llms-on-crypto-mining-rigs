#!/usr/bin/env bash
# R229 (Q1642): Flash-Next's agent check with its v5 preset (release/flash-next-v5/settings.env) as it is, whose MTP
# draft, after a shortlist miss, drafts two rounds on its full output head (LLAMA_DRAFT_SHORTLIST=32768,8192,2), against
# the same preset with LLAMA_DRAFT_SHORTLIST=32768,8192,1, one such round (R175, docs/data/2026-11-20_r175_flash_shortlist/:
# a step 1.31 ms shorter where it acts). Four runs on one rig, engine and boot, alternated: s2a, s1a, s2b, s1b.
# It starts the preset as C912 did (docs/data/2026-11-24_c912_flash_agent/): R227's run/flash_taper.sh with ARM=t1400a,
# that is C854's launcher (docs/data/2026-11-18_c854b_p106_check/run/flash_agent.sh), R156's driver
# (docs/data/2026-11-13_r156_agent_v4r/run/r156.sh), the harness's arm v4r, then the report. r156.sh takes the
# shortlist from release/flash-next-v3/settings.env's PRESET_ENV, which no environment variable overrides, so this
# script writes into OUT a copy of r156.sh with one line added after the line that reads that file,
#   PRESET_ENV=${PRESET_ENV/LLAMA_DRAFT_SHORTLIST=32768,8192,2/LLAMA_DRAFT_SHORTLIST=32768,8192,<rounds>}
# and a copy of flash_agent.sh that runs that copy, and runs it. Both arms take this path; for s2a and s2b the added
# line changes nothing, so their started command is C912's.
#   ARM=s2a|s2b  the v5 preset (two full-head rounds after a miss)
#   ARM=s1a|s1b  the same with LLAMA_DRAFT_SHORTLIST=32768,8192,1, so the started command differs from the control's
#                in that one value
# Run from the repository root on the machine that drives the rig. Everything machine-specific comes from the
# environment, as flash_agent.sh lists it: RIG CLIENT MODEL DRAFT NEW REF RIGDIR OUT CLIENT_DIR CLIENT_OUT, and
# RIG_PORT RELAY_PORT PROXY_PORT. Give every run its own OUT and CLIENT_OUT (the driver skips attempts it finds
# finished there) and, when another agent check shares the client, its own PREFIX (the client's container names,
# acrel-<arm>-* by default). DRY=1 prints the environment the run would use and the added line, and stops.
set -uo pipefail
DRIVER=docs/data/2026-11-13_r156_agent_v4r/run/r156.sh
LAUNCHER=docs/data/2026-11-18_c854b_p106_check/run/flash_agent.sh
READS_PRESET='. release/flash-next-v3/settings.env  # PRESET_ENV, PRESET_ARGS'
FROM=LLAMA_DRAFT_SHORTLIST=32768,8192,2
case ${ARM:?} in
  s2a|s2b) TO=LLAMA_DRAFT_SHORTLIST=32768,8192,2 ;;
  s1a|s1b) TO=LLAMA_DRAFT_SHORTLIST=32768,8192,1 ;;
  *) echo "unknown ARM $ARM"; exit 1 ;;
esac
# R227's t1400a arm (run/flash_taper.sh): R156's v4r line with text reuse, the sparse ratio 0.75 and the 1,400 taper
V4R_DEFAULT=$(sed -n 's/^V4R_ENV=\${V4R_ENV:-\(.*\)}$/\1/p' "$DRIVER")
GAINS_DEFAULT=$(sed -n 's/^GAINS_ENV=\${GAINS_ENV:-\(.*\)}$/\1/p' "$DRIVER")
[ -n "$V4R_DEFAULT" ] && [ -n "$GAINS_DEFAULT" ] || { echo "no V4R_ENV or GAINS_ENV default in $DRIVER"; exit 1; }
RATIO=GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO
[[ " $GAINS_DEFAULT " == *" $RATIO=1.0 "* ]] || { echo "the GAINS_ENV default has no $RATIO=1.0"; exit 1; }
export V4R_ENV="$V4R_DEFAULT LLAMA_SERVER_TEXT_REUSE=1"
export GAINS_ENV="${GAINS_DEFAULT/"$RATIO=1.0"/"$RATIO=0.75"}"
export V4C2_TAPER=short=1400:64
# the shortlist: one line added to a copy of r156.sh, right after it reads the v3 preset
# shellcheck disable=SC1091
PRESET_ENV=$(. release/flash-next-v3/settings.env && echo "$PRESET_ENV")
[[ " $PRESET_ENV " == *" $FROM "* ]] || { echo "release/flash-next-v3/settings.env has no $FROM"; exit 1; }
[ "$(grep -cxF "$READS_PRESET" "$DRIVER")" = 1 ] || { echo "$DRIVER does not read the v3 preset in one known line"; exit 1; }
ADD="PRESET_ENV=\${PRESET_ENV/$FROM/$TO}"
echo "$(date +%H:%M:%S) arm $ARM: GAINS_ENV=$GAINS_ENV"
echo "$(date +%H:%M:%S) arm $ARM: V4R_ENV=$V4R_ENV"
echo "$(date +%H:%M:%S) arm $ARM: V4C2_TAPER=$V4C2_TAPER"
echo "$(date +%H:%M:%S) arm $ARM: added to r156.sh after it reads the v3 preset: $ADD"
[ "${DRY:-0}" = 1 ] && exit 0
mkdir -p "${OUT:?}"
awk -v src="$READS_PRESET" -v add="$ADD" '{ print } $0 == src { print add }' "$DRIVER" > "$OUT/r156_shortlist.sh"
[ "$(grep -cxF "$ADD" "$OUT/r156_shortlist.sh")" = 1 ] \
  && [ "$(($(wc -l < "$DRIVER") + 1))" = "$(wc -l < "$OUT/r156_shortlist.sh")" ] \
  || { echo "the copy of $DRIVER was not written with the one added line"; exit 1; }
sed "s|bash $DRIVER\$|bash $OUT/r156_shortlist.sh|" "$LAUNCHER" > "$OUT/flash_agent_shortlist.sh"
grep -q "bash $OUT/r156_shortlist.sh\$" "$OUT/flash_agent_shortlist.sh" \
  && [ "$(diff "$LAUNCHER" "$OUT/flash_agent_shortlist.sh" | grep -c '^[<>]')" = 2 ] \
  || { echo "the copy of $LAUNCHER does not run the copy of $DRIVER"; exit 1; }
exec bash "$OUT/flash_agent_shortlist.sh"
