#!/usr/bin/env bash
# C1128 (Q1861): the build machine's driver. One load of the model per arm, each a verify-drift run (run/verify_drift.cpp,
# built by run/build.sh) over C964's three requests (its prompts and plain decoding's replies, as token ids):
#   base     one token a call, n_rs_seq 0 (plain decoding's server): writes the reference rows
#   four     four tokens a call, n_rs_seq 3 (the drafting server's verify pass): against base
#   floor    base again in a fresh load: against base (the load-to-load floor)
#   one_rs3  one token a call with n_rs_seq 3: against base (the snapshot setting alone)
#   four_rs0 four tokens a call with n_rs_seq 0: against base (the call size alone)
# The environment is release/flash-next-v5/settings.env's PRESET_ENV with the fork's compute-6.1 card paths forced the
# way a P106 takes them (scripts/pascal_forced.sh, pascal_force_preset); of the preset's arguments, those that act on
# a load or on the arithmetic (checked present in PRESET_ARGS), the layer split set to the cards.
# Env:
#   WORK (outputs), BASE (the reference rows file, ~1 MB per row: put it on a bulk disk), TOOL (verify-drift),
#   SYSINFO (the sysinfo probe), MODEL, REPO (a copy of this repository: release/flash-next-v5/settings.env,
#   scripts/pascal_forced.sh), REQS (verify-drift's KLD_REQS, paths readable in the container), GPUS (docker's device
#   list, default 0,1), TS (layer split, default 1 per card), ARMS (default "base four floor one_rs3 four_rs0"),
#   PARTS (default reply,whole), THREADS (8), MOUNTS (mounted read-only at their own paths), DOCKER_ARGS, IMAGE,
#   PREFLIGHT (a CUDA check run as "$PREFLIGHT -g $GPUS"), JIT_SEED (a PTX JIT cache to start from), TIMEOUT (seconds
#   per load, default 2700)
set -uo pipefail
: "${WORK:?}" "${BASE:?}" "${TOOL:?}" "${SYSINFO:?}" "${MODEL:?}" "${REPO:?}" "${REQS:?}"
GPUS=${GPUS:-0,1}
NCARDS=$(tr ',' '\n' <<< "$GPUS" | grep -c .)
TS=${TS:-$(printf '1%.0s,' $(seq 1 "$NCARDS") | sed 's/,$//')}
ARMS=${ARMS:-base four floor one_rs3 four_rs0}
PARTS=${PARTS:-reply,whole}
THREADS=${THREADS:-8}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
TIMEOUT=${TIMEOUT:-2700}
PREFLIGHT=${PREFLIGHT:-}

say() { echo "$(date +%H:%M:%S) $*"; }
finish() { say "end rc=$1"; echo "$1" > "$WORK/DONE"; exit "$1"; }
smi() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | tr -d ' ' | tr '\n' ' '; }
rm -f "$WORK/DONE"
mkdir -p "$WORK/jitcache" "$(dirname "$BASE")"
[ -n "${JIT_SEED:-}" ] && [ -z "$(ls -A "$WORK/jitcache")" ] && cp -a "$JIT_SEED/." "$WORK/jitcache/"

unset PRESET_MODEL PRESET_DRAFT PRESET_ENV PRESET_ARGS
# shellcheck disable=SC1091
. "$REPO/release/flash-next-v5/settings.env"
# shellcheck disable=SC1091
. "$REPO/scripts/pascal_forced.sh"
for f in "-ngl 99" "-fa on" "-c 65536" "-b 4096 -ub 512" "-fit off -lm dio" "--no-host" "--ctx-checkpoints-incall" \
        "--ubatch-taper short=1400:64" "--spec-type draft-mtp --spec-draft-n-max 3"; do
    [[ " $PRESET_ARGS " == *" $f "* ]] || { say "the preset's arguments lack '$f'"; finish 1; }
done
ENVS="$PRESET_ENV $(pascal_force_preset "$PRESET_ENV")"
ARGS="-m $MODEL -ngl 99 -fa on -c 65536 -b 4096 -ub 512 -ts $TS -fit off -lm dio --no-host -t $THREADS"
md5sum "$TOOL" "$SYSINFO" "$MODEL" "$REPO/release/flash-next-v5/settings.env" "$REPO/scripts/pascal_forced.sh" \
    "$(dirname "$0")"/* > "$WORK/md5.txt" 2>&1
for f in ${REQS//;/ }; do md5sum "$(cut -d= -f2 <<< "$f")" "$(cut -d= -f3 <<< "$f")"; done >> "$WORK/md5.txt" 2>&1
say "start: arms '$ARMS', parts $PARTS, on cards $GPUS (-ts $TS)"
if [ -n "$PREFLIGHT" ]; then
    $PREFLIGHT -g "$GPUS" > "$WORK/preflight.txt" 2>&1
    rc=$?; say "preflight rc=$rc"; [ $rc = 0 ] || finish 1
fi

VOL=()
for m in ${MOUNTS:-}; do VOL+=(-v "$m:$m:ro"); done
VOL+=(-v "$WORK:$WORK" -v "$(dirname "$BASE"):$(dirname "$BASE")" -v "$(dirname "$TOOL"):$(dirname "$TOOL"):ro")
run_c() {  # <container name> <timeout> <command>: one container on the cards with the run's environment
    # shellcheck disable=SC2086
    timeout "$2" docker run --rm ${DOCKER_ARGS:-} --name "$1" --gpus "\"device=$GPUS\"" "${VOL[@]}" \
        -e "CUDA_CACHE_PATH=$WORK/jitcache" -e CUDA_CACHE_MAXSIZE=4294967296 -e OMP_NUM_THREADS="$THREADS" \
        "${@:4}" "$IMAGE" bash -c "$3"
    local rc=$?
    [ $rc = 124 ] && docker stop "$1" > /dev/null 2>&1
    return $rc
}

# the forced card paths, listed by the backend under the run's environment
run_c c1128-sysinfo 300 "env $ENVS $SYSINFO" > "$WORK/sysinfo.txt" 2>&1
say "sysinfo rc=$?: $(grep -o 'MMVQ_PASCAL = [0-9]*\|MMVF_PASCAL = [0-9]*' "$WORK/sysinfo.txt" | tr '\n' ' ')"

arm() {  # <name> <mode> <step> <rs>
    local o=$WORK/$1 rc
    rm -rf "$o"; mkdir -p "$o"
    printf 'env: %s\nKLD_MODE=%s KLD_STEP=%s KLD_RS=%s KLD_PARTS=%s KLD_REQS=%s\nargs: %s %s\n' \
        "$ENVS" "$2" "$3" "$4" "$PARTS" "$REQS" "$TOOL" "$ARGS" > "$o/command.txt"
    say "$1: start ($2, $3 a call, n_rs_seq $4); cards $(smi)"
    run_c "c1128-$1" "$TIMEOUT" "env $ENVS $TOOL $ARGS" \
        -e KLD_MODE="$2" -e KLD_STEP="$3" -e KLD_RS="$4" -e KLD_PARTS="$PARTS" -e KLD_REQS="$REQS" \
        -e KLD_BASE="$BASE" -e KLD_OUT="$o/drift" > "$o/run.log" 2>&1
    rc=$?
    say "$1: rc=$rc; $(grep '^verify-drift: [0-9]* rows' "$o/run.log" | tail -n 1); cards $(smi)"
    return $rc
}

for a in $ARMS; do
    case $a in
    base)     arm base base 1 0 || finish 1 ;;
    four)     arm four test 4 3 || finish 1 ;;
    floor)    arm floor test 1 0 || finish 1 ;;
    one_rs3)  arm one_rs3 test 1 3 || finish 1 ;;
    four_rs0) arm four_rs0 test 4 0 || finish 1 ;;
    *) say "unknown arm $a"; finish 1 ;;
    esac
done
finish 0
