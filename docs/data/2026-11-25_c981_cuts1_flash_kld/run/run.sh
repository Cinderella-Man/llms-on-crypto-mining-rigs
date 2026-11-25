#!/usr/bin/env bash
# C981 (Q1714): the build machine's driver. One load of the model per pass, each a kld-cuts run (run/kld_cuts.cpp,
# built by run/build.sh) over the call sizes with four arms: the taper off, the v5 preset's short=1400:64, the same with
# cuts=1, and short=1400:64 again (the repeat). The environment is release/flash-next-v5/settings.env's PRESET_ENV with
# the fork's compute-6.1 card paths forced the way a P106 takes them (scripts/pascal_forced.sh, pascal_force_preset);
# of the preset's arguments, those that act on a load or on the arithmetic (checked present in PRESET_ARGS), the layer
# split set to the cards. The context's n_rs_seq is 3, what llama-server sets for the preset's MTP draft of three.
# Passes:
#   fresh  every size in SIZES, CHUNKS calls each, from the text's first token, each call on an empty memory
#   deep   every size in DEEP_SIZES, DEEP_CHUNKS calls each, from the token after the fresh pass's last, each call read
#          after a DEEP_PREFIX-token prompt call of the text before it (an agent's next turn on a cached context)
# Env:
#   WORK, TOOL, MODEL, TEXT, REPO (a copy of this repository: release/flash-next-v5/settings.env,
#   scripts/pascal_forced.sh), GPUS (docker's device list, default 0,1), TS (layer split, default 1 per card),
#   PASSES (default "fresh deep"), SIZES, CHUNKS (8), DEEP_SIZES, DEEP_CHUNKS (2), DEEP_PREFIX (4096), DECODE (16),
#   THREADS (8: the tool's CPU threads and OpenMP threads), MOUNTS (mounted read-only at their own paths),
#   DOCKER_ARGS, IMAGE, PREFLIGHT (a CUDA check run as "$PREFLIGHT -g $GPUS"), JIT_SEED (a PTX JIT cache to start
#   from), TIMEOUT_FRESH, TIMEOUT_DEEP (seconds)
set -uo pipefail
: "${WORK:?}" "${TOOL:?}" "${MODEL:?}" "${TEXT:?}" "${REPO:?}"
GPUS=${GPUS:-0,1}
NCARDS=$(tr ',' '\n' <<< "$GPUS" | grep -c .)
TS=${TS:-$(printf '1%.0s,' $(seq 1 "$NCARDS") | sed 's/,$//')}
PASSES=${PASSES:-fresh deep}
SIZES=${SIZES:-569,577,583,599,610,775,803,815,896,936,1086,1130,1300,1359,449,1028}
CHUNKS=${CHUNKS:-8}
DEEP_SIZES=${DEEP_SIZES:-569,577,583,599,610,775,803,815,896,936,1086,1130,1300,1359}
DEEP_CHUNKS=${DEEP_CHUNKS:-2}
DEEP_PREFIX=${DEEP_PREFIX:-4096}
DECODE=${DECODE:-16}
THREADS=${THREADS:-8}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
TIMEOUT_FRESH=${TIMEOUT_FRESH:-9000}
TIMEOUT_DEEP=${TIMEOUT_DEEP:-6000}
PREFLIGHT=${PREFLIGHT:-}

say() { echo "$(date +%H:%M:%S) $*"; }
finish() { say "end rc=$1"; echo "$1" > "$WORK/DONE"; exit "$1"; }
smi() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | tr -d ' ' | tr '\n' ' '; }
rm -f "$WORK/DONE"
mkdir -p "$WORK/jitcache"
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
ARGS="-m $MODEL -f $TEXT -ngl 99 -fa on -c 65536 -b 4096 -ub 512 -ts $TS -fit off -lm dio --no-host -t $THREADS"
md5sum "$TOOL" "$MODEL" "$TEXT" "$REPO/release/flash-next-v5/settings.env" "$REPO/scripts/pascal_forced.sh" \
    "$(dirname "$0")"/* > "$WORK/md5.txt" 2>&1
say "start: passes '$PASSES' on cards $GPUS (-ts $TS)"
if [ -n "$PREFLIGHT" ]; then
    $PREFLIGHT -g "$GPUS" > "$WORK/preflight.txt" 2>&1
    rc=$?; say "preflight rc=$rc"; [ $rc = 0 ] || finish 1
fi

VOL=()
for m in ${MOUNTS:-}; do VOL+=(-v "$m:$m:ro"); done
VOL+=(-v "$WORK:$WORK" -v "$(dirname "$TOOL"):$(dirname "$TOOL"):ro")

pass() {  # <name> <sizes> <chunks> <prefix> <start> <timeout>
    local name=$1 o=$WORK/$1 c=c981-$1 rc
    rm -rf "$o"; mkdir -p "$o"
    printf 'env: %s\nKLD_SIZES=%s KLD_CHUNKS=%s KLD_PREFIX=%s KLD_START=%s KLD_DECODE=%s KLD_RS=3\nargs: %s %s\n' \
        "$ENVS" "$2" "$3" "$4" "$5" "$DECODE" "$TOOL" "$ARGS" > "$o/command.txt"
    say "$name: start (sizes $2, $3 each, prefix $4, from token $5); cards $(smi)"
    # shellcheck disable=SC2086
    timeout "$6" docker run --rm ${DOCKER_ARGS:-} --name "$c" --gpus "\"device=$GPUS\"" "${VOL[@]}" \
        -e "CUDA_CACHE_PATH=$WORK/jitcache" -e CUDA_CACHE_MAXSIZE=4294967296 -e OMP_NUM_THREADS="$THREADS" \
        -e KLD_SIZES="$2" -e KLD_CHUNKS="$3" -e KLD_PREFIX="$4" -e KLD_START="$5" -e KLD_DECODE="$DECODE" -e KLD_RS=3 \
        -e KLD_OUT="$o/kld" "$IMAGE" bash -c "env $ENVS $TOOL $ARGS" > "$o/run.log" 2>&1
    rc=$?
    [ $rc = 124 ] && docker stop "$c" > /dev/null 2>&1
    say "$name: rc=$rc, $(grep -c '^kld-cuts: chunk' "$o/run.log") chunks; cards $(smi)"
    [ -f "$o/kld.pos.tsv" ] && gzip -f "$o/kld.pos.tsv"
    return $rc
}

# tokens the fresh pass takes: CHUNKS x the sum over SIZES of (size + DECODE + 1)
fresh_tokens() {
    local s=0 n
    for n in ${SIZES//,/ }; do s=$((s + n + DECODE + 1)); done
    echo $((s * CHUNKS))
}

for p in $PASSES; do
    case $p in
    fresh) pass fresh "$SIZES" "$CHUNKS" 0 0 "$TIMEOUT_FRESH" || finish 1 ;;
    deep)  pass deep "$DEEP_SIZES" "$DEEP_CHUNKS" "$DEEP_PREFIX" "$(fresh_tokens)" "$TIMEOUT_DEEP" || finish 1 ;;
    *) say "unknown pass $p"; finish 1 ;;
    esac
done
finish 0
