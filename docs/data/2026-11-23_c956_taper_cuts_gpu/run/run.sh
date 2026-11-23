#!/usr/bin/env bash
# C956 (Q1687 point 10): the GPU half of C905's check (fork branch pascal-p106-v5-taper-cuts 1720f26d3, two commits on
# pascal-p106-v5 3dfad876f: the taper key cuts=1, which plans each stretch of a prompt call between the server's
# in-call checkpoint cuts on its own, so no piece under the taper's 9-token floor is left beside a cut), on a build
# machine with three newer NVIDIA cards that run the rig build's compute 6.1 PTX. Correctness only, nothing timed.
#
# Each load is one llama-server of Flash-Next's v5 preset (release/flash-next-v5/settings.env: Swift GSQ-RCO Q2_0 with
# an MTP draft of three, -b 4096 -ub 512, in-call checkpoints, --ubatch-taper short=1400:64), the layer split changed
# to the three cards (-ts 1,1,1) and the log at verbosity 4 (-lv 4). The draft is Swift's own MTP head (the preset's
# draft file is on the rigs only). Loads (LOADS):
#   base   the base (3dfad876f, C908's rig tarball): sizes passes srv, s2048; then the replay, passes 0 and 1
#   head   the branch: sizes passes srv, v5c, s2048, s2048c, srv2; then the replay, passes 0 to 4
#   headc  the branch with the server's setting --ubatch-taper short=1400:64,cuts=1: sizes pass srv
# Sizes (client.py): 24 requests a pass, prompts sized where the cuts fall (C905's server check's sizes and the 14 call
# sizes cuts=1 changes), 64 greedy reply tokens with top-3 log-probabilities. Passes: srv = the server's own setting,
# v5c = each request sending short=1400:64,cuts=1, s2048 / s2048c = short=2048:64 without and with cuts=1 (where
# C222 logged the 1,534- and 1,985-token calls' pieces), srv2 = srv again on the same load.
# Replay (taper_replay.py of docs/data/2026-11-04_prompt_taper, as C222 and R225 ran it on a rig): the agent check's
# 12 tasks and 62 turns (docs/data/2026-11-03_retype_agent/agent/turns.csv), 8 greedy tokens a turn, no fresh
# prompts; per request the taper spec of its pass. Pass 0 is the warm-up (its first turn goes out cold), then
# short=1400:64, short=1400:64,cuts=1, short=1400:64,cuts=1, short=1400:64 on the branch; one short=1400:64 pass on
# the base. The replay script also writes times; they are not used (no speed on this machine).
# compare.py then compares the loads and passes and checks each call's pieces against C905's CPU plan check.
#
# The fork's Pascal kernels turn themselves on only on compute 6.1 cards (the matrix-vector kernels at their default
# 1, the sparse attention at the preset's 1 only below Turing), so they are forced on with a P106's launch shapes
# (5 SMs, 96 KB of shared memory per SM), as C953 ran them: the pieces of 4 to 6 tokens this branch removes run the
# matrix-vector kernels on a P106.
ON="GGML_CUDA_MMVQ_PASCAL=2 GGML_CUDA_MMVF_PASCAL=2 GGML_CUDA_MMVQ_PASCAL_NSM=5 GGML_CUDA_MMVF_PASCAL_NSM=5 \
GGML_CUDA_MMVQ_PASCAL_KCOL_NSM=5 GGML_CUDA_MMVQ_PASCAL_KCOL_SMSM=98304 GGML_CUDA_FATTN_SPARSE_PASCAL=2"
# Run it on the build machine itself, detached: setsid nohup bash run.sh > $WORK/driver.log 2>&1 < /dev/null &
# Env:
#   WORK      a folder for the outputs (mounted read-write into every container)
#   NEW, REF  bin/ folders of the two rig builds (NEW = the branch, REF = the base)
#   REPO      a copy of this repository (release/flash-next-v5/settings.env, this run/, the replay's script and turns,
#             C905's plans.tsv, the text files of TEXTS)
#   FN        Flash-Next's preset file (first shard, the others beside it); DRAFT its MTP draft
#   MOUNTS    folders the containers mount read-only at the same paths (the builds, the model)
#   DOCKER_ARGS  extra `docker run` arguments (labels, --cpu-shares) the machine's policy asks for
#   PREFLIGHT    a CUDA check run before the loads as "$PREFLIGHT -g <ids>" (optional)
#   GPUS (0,1,2), TS (1,1,1), PORT (18956), IMG (nvidia/cuda:12.9.2-devel-ubuntu24.04), LOADS ("base head headc"),
#   MAX_TOKENS (64), LOAD_TIMEOUT (2400 s), CLIENT_TIMEOUT (3600 s), NAME (c956: the containers' name prefix)
# Output: WORK/load/<load>/ (server.log, split.txt: the log's line count when the replay starts, sizes.jsonl,
# replay/requests.tsv, command.txt, cards.tsv), WORK/compare/ (compare.py's tables), WORK/md5.txt, WORK/DONE.
set -uo pipefail
: "${WORK:?}" "${NEW:?}" "${REF:?}" "${REPO:?}" "${FN:?}" "${DRAFT:?}"
IMG=${IMG:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
GPUS=${GPUS:-0,1,2}
TS=${TS:-1,1,1}
PORT=${PORT:-18956}
LOADS=${LOADS:-base head headc}
MAX_TOKENS=${MAX_TOKENS:-64}
LOAD_TIMEOUT=${LOAD_TIMEOUT:-2400}
CLIENT_TIMEOUT=${CLIENT_TIMEOUT:-3600}
NAME=${NAME:-c956}
PREFLIGHT=${PREFLIGHT:-}
D=$REPO/docs/data/2026-11-23_c956_taper_cuts_gpu/run
REPLAY=$REPO/docs/data/2026-11-04_prompt_taper/run/taper_replay.py
TURNS=$REPO/docs/data/2026-11-03_retype_agent/agent/turns.csv
PLANS=$REPO/docs/data/2026-11-21_c905_taper_cuts/plans.tsv
TEXTS="$REPO/docs/data/2026-11-21_c905_taper_cuts/README.md $REPO/docs/data/2026-11-20_c222_flash_microbatch/README.md \
$REPO/docs/data/2026-11-20_r225_flash_taper1400/README.md"
URL=http://127.0.0.1:$PORT

say() { echo "$(date +%H:%M:%S) $*"; }
VOL=()
for m in ${MOUNTS:-}; do VOL+=(-v "$m:$m:ro"); done
VOL+=(-v "$WORK:$WORK")
# the PTX JIT's cache, shared by every container (the cards compile the build's compute 6.1 PTX once)
JIT=(-e "CUDA_CACHE_PATH=$WORK/jitcache" -e CUDA_CACHE_MAXSIZE=4294967296)
# shellcheck disable=SC2206
DR=(docker run --rm ${DOCKER_ARGS:-} "${VOL[@]}" "${JIT[@]}")
mkdir -p "$WORK/jitcache"
rm -f "$WORK/DONE"
# shellcheck disable=SC2086
md5sum "$NEW/llama-server" "$REF/llama-server" "$DRAFT" "$D"/* "$REPLAY" "$TURNS" "$PLANS" $TEXTS \
    "$REPO/release/flash-next-v5/settings.env" > "$WORK/md5.txt" 2>&1
say "start: loads '$LOADS', NEW=$NEW REF=$REF, forced: $ON"

smi() {  # every card's memory use, "index:used/total" in MiB
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits -i "$GPUS" |
        awk -F', *' '{printf "%s%s:%s/%s", (NR > 1 ? " " : ""), $1, $2, $3}'
}
mark() { printf '%s\t%s\n' "$1" "$(smi)" >> "$o/cards.tsv"; }
up() { curl -sf -m 5 "$URL/health" > /dev/null 2>&1; }
alive() { docker ps -q --filter "name=^$c\$" | grep -q .; }

preset() {  # PRESET_ENV and PRESET_ARGS of release/flash-next-v5/settings.env: the split changed to the cards, -lv 4
    unset PRESET_MODEL PRESET_DRAFT PRESET_ENV PRESET_ARGS
    # shellcheck disable=SC1090
    . "$REPO/release/flash-next-v5/settings.env"
    ARGS=$(sed -E "s/-ts [0-9,]+/-ts $TS/" <<< "$PRESET_ARGS")
    [[ "$ARGS" == *"-ts $TS"* && "$ARGS" == *"--ubatch-taper short=1400:64"* && "$ARGS" == *"--ctx-checkpoints-incall"* ]] ||
        { say "the preset's -ts, taper or in-call checkpoints not found"; exit 1; }
    ARGS="$ARGS -lv 4"
    CUTS=$(sed -E "s/--ubatch-taper short=1400:64/--ubatch-taper short=1400:64,cuts=1/" <<< "$ARGS")
}

load() {  # <load> <bin folder> <server arguments> <sizes passes> <replay passes or ->
    local l=$1 bin=$2 args=$3 sp=$4 rp=$5 t0 up1=0
    o=$WORK/load/$l
    c=$NAME-$l
    rm -rf "$o"; mkdir -p "$o"
    docker rm -f "$c" > /dev/null 2>&1
    printf 'bin: %s\nenv: %s %s\nargs: -m %s -md %s %s\n' "$bin" "$PRESET_ENV" "$ON" "$FN" "$DRAFT" "$args" > "$o/command.txt"
    mark before
    say "$l: start"
    # shellcheck disable=SC2086
    "${DR[@]}" -d --name "$c" --gpus "\"device=$GPUS\"" -p "127.0.0.1:$PORT:$PORT" "$IMG" \
        bash -c "env $PRESET_ENV $ON $bin/llama-server -m $FN -md $DRAFT $args \
            --host 0.0.0.0 --port $PORT --no-webui > $o/server.log 2>&1; echo rc=\$? > $o/server.rc" > /dev/null
    t0=$(date +%s)
    while [ $(( $(date +%s) - t0 )) -lt "$LOAD_TIMEOUT" ]; do
        up && { up1=1; break; }
        alive || break
        sleep 2
    done
    if [ $up1 = 1 ]; then
        mark up
        # shellcheck disable=SC2086
        timeout "$CLIENT_TIMEOUT" python3 -I "$D/client.py" "$URL" "$o/sizes.jsonl" "$sp" "$MAX_TOKENS" $TEXTS \
            > "$o/sizes.out" 2>&1
        say "$l: sizes rc=$? ($(wc -l < "$o/sizes.jsonl") requests, $(grep -c '"error"' "$o/sizes.jsonl") failed)"
        wc -l < "$o/server.log" > "$o/split.txt"
        if [ "$rp" != - ]; then
            mkdir -p "$o/replay"
            timeout "$CLIENT_TIMEOUT" python3 -I "$REPLAY" --port "$PORT" --turns "$TURNS" --arm hcq8 --passes "$rp" \
                --gen 8 --fresh 0 --label "$l" --out "$o/replay" > "$o/replay.out" 2>&1
            say "$l: replay rc=$? ($(( $(wc -l < "$o/replay/requests.tsv") - 1 )) requests)"
        fi
        mark replied
    fi
    docker stop -t 60 "$c" > /dev/null 2>&1
    docker rm -f "$c" > /dev/null 2>&1
    mark stopped
    say "$l: up=$up1 $(cat "$o/server.rc" 2>/dev/null), taper lines $(grep -ac 'ubatch taper:' "$o/server.log")," \
        "in-call checkpoints $(grep -aci 'checkpoint' "$o/server.log"), restored $(grep -ac 'restored context checkpoint' "$o/server.log")," \
        "asserts $(grep -ac 'GGML_ASSERT\|GGML_ABORT' "$o/server.log"), features: $(grep -aoE 'MMV[QF]_PASCAL = 1' "$o/server.log" | sort -u | paste -sd' ')"
}

[ -n "$PREFLIGHT" ] && { $PREFLIGHT -g "$GPUS" > "$WORK/preflight.txt" 2>&1; say "preflight rc=$?"; }
preset
for l in $LOADS; do
    case $l in
        base)  load "$l" "$REF" "$ARGS" "srv:- s2048:short=2048:64" "short=1400:64 short=1400:64" ;;
        head)  load "$l" "$NEW" "$ARGS" "srv:- v5c:short=1400:64,cuts=1 s2048:short=2048:64 s2048c:short=2048:64,cuts=1 srv2:-" \
                   "short=1400:64 short=1400:64 short=1400:64,cuts=1 short=1400:64,cuts=1 short=1400:64" ;;
        headc) load "$l" "$NEW" "$CUTS" "srv:-" - ;;
        *) say "unknown load $l" ;;
    esac
done

mkdir -p "$WORK/compare"
python3 -I "$D/compare.py" "$WORK/load" "$PLANS" "$WORK/compare" > "$WORK/compare/verdicts.tsv" 2> "$WORK/compare/compare.err" ||
    say "compare rc=$?"
grep -a 'verdict' "$WORK/compare/verdicts.tsv"
say "done: our containers left: $(docker ps -q --filter "name=^$NAME-" | wc -l); card MiB: $(smi)"
date -Is > "$WORK/DONE"
