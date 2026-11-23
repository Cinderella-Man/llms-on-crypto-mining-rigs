#!/usr/bin/env bash
# C954 (Q1687 point 8): the GPU half of C910's check (fork branch pascal-p106-v5-mtp-device-mem f94aa552b, two commits
# on pascal-p106-v5 3dfad876f: --mtp-device N's copy of the output head is made as a weight buffer that the memory
# fit's dry run counts and llama_model_free frees, and its data is copied card to card through a host buffer of at most
# 16 MiB instead of one of the head's size), on a build machine with three newer NVIDIA cards that run the rig build's
# compute 6.1 PTX. Correctness and memory only, nothing timed.
#
# Loading and freeing twice in one process: llama-server's sleep mode (--sleep-idle-seconds N) frees the model and its
# contexts after N idle seconds and loads them again, in the same process, on the next request. Each arm is one server
# process that loads (cycle 1), answers, sleeps (free 1), loads again on a request (cycle 2), answers, sleeps (free 2).
#
# The replies run on the fork's Pascal matrix-vector kernels: they turn themselves on only on compute 6.1 cards, so
# they are forced on with a P106's launch shapes (5 SMs, 96 KB of shared memory per SM), as C953 ran them. Where the
# output head's copy sits does not depend on them:
ON="GGML_CUDA_MMVQ_PASCAL=2 GGML_CUDA_MMVF_PASCAL=2 GGML_CUDA_MMVQ_PASCAL_NSM=5 GGML_CUDA_MMVF_PASCAL_NSM=5 \
GGML_CUDA_MMVQ_PASCAL_KCOL_NSM=5 GGML_CUDA_MMVQ_PASCAL_KCOL_SMSM=98304"
# Each arm: Ornith-1.5-122B Pascal N2P GGUF with its v5 preset (release/ornith-v5/settings.env), the layer split changed
# to the three cards (-ts 1,1,1, so the output head sits on the last card), drafting with the MTP layer in its file as
# C899's MTP arms and C910's rig check run it (--spec-type draft-mtp --spec-draft-n-max 1), and with the memory fit on
# instead of the preset's -fit off, at log verbosity 4, so the server log shows the fit's projected memory per card
# (it needs no change on these cards, so the load is the preset's). Arms (ARMS):
#   new-unset   the branch, --mtp-device unset: the MTP layer on the last card, no copy
#   new-dev0    the branch, --mtp-device 0: the MTP layer and the head's copy on the first card
#   base-dev0   the base (3dfad876f, C908's rig tarball), --mtp-device 0
#   base-unset  the base, --mtp-device unset
# What each arm records (WORK/load/<arm>/): the server log (the fit's projection per card, the "MTP output head copied"
# line and the model buffer sizes of each load), phases.tsv (each phase's time and every card's memory use from
# nvidia-smi: before the start, up after load 1, asleep after free 1, up after load 2, asleep after free 2),
# rss.tsv (the server process's resident memory from /proc every 5 ms: run/rss_sampler.py), and c1/, c2/ (each
# cycle's replies: C953's run/replies.py, three greedy 256-token requests ROUNDS times and once with n_probs 2).
# Run it on the build machine itself, detached: setsid nohup bash run.sh > $WORK/driver.log 2>&1 < /dev/null &
# Env:
#   WORK      a folder for the outputs (mounted read-write into every container)
#   NEW, REF  bin/ folders of the two rig builds (NEW = the branch, REF = the base)
#   REPO      a copy of this repository (release/ornith-v5/settings.env, this run/ and C953's run/replies.py)
#   ORNITH    Ornith-1.5-122B Pascal N2P GGUF
#   MOUNTS    folders the containers mount read-only at the same paths (the builds, the model)
#   DOCKER_ARGS  extra `docker run` arguments (labels, --cpu-shares) the machine's policy asks for
#   PREFLIGHT    a CUDA check run before the loads as "$PREFLIGHT -g <ids>" (optional)
#   GPUS (0,1,2), TS (1,1,1), PORT (18954), IMG (nvidia/cuda:12.9.2-devel-ubuntu24.04), ARMS ("new-unset new-dev0
#   base-dev0 base-unset"), ROUNDS (1), DRAFT_N (1), SLEEP_S (15), LOAD_TIMEOUT (2400 s), NAME (c954: the containers'
#   name prefix)
# Output: WORK/load/<arm>/ as above plus command.txt and server.rc, WORK/md5.txt, WORK/DONE at the end.
set -uo pipefail
: "${WORK:?}" "${NEW:?}" "${REF:?}" "${REPO:?}" "${ORNITH:?}"
IMG=${IMG:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
GPUS=${GPUS:-0,1,2}
TS=${TS:-1,1,1}
PORT=${PORT:-18954}
ARMS=${ARMS:-new-unset new-dev0 base-dev0 base-unset}
ROUNDS=${ROUNDS:-1}
DRAFT_N=${DRAFT_N:-1}
SLEEP_S=${SLEEP_S:-15}
LOAD_TIMEOUT=${LOAD_TIMEOUT:-2400}
NAME=${NAME:-c954}
PREFLIGHT=${PREFLIGHT:-}
D=$REPO/docs/data/2026-11-23_c954_mtp_device_gpu/run
REPLIES=$REPO/docs/data/2026-11-23_c953_ornith_kcol_gpu/run/replies.py
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
md5sum "$NEW/llama-server" "$REF/llama-server" "$D"/* "$REPLIES" "$REPO/release/ornith-v5/settings.env" > "$WORK/md5.txt" 2>&1
say "start: arms '$ARMS', NEW=$NEW REF=$REF, forced: $ON"

smi() {  # every card's memory use, "index:used/total" in MiB
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits -i "$GPUS" |
        awk -F', *' '{printf "%s%s:%s/%s", (NR > 1 ? " " : ""), $1, $2, $3}'
}
mark() { printf '%s\t%s\t%s\n' "$(date +%s%3N)" "$1" "$(smi)" >> "$o/phases.tsv"; }
up() { curl -sf -m 5 "$URL/health" > /dev/null 2>&1; }
alive() { docker ps -q --filter "name=^$c\$" | grep -q .; }
asleep() {  # <n>: wait until the log shows the n-th sleep, then 10 s for the free and the cards to settle
    local t0
    t0=$(date +%s)
    while [ "$(grep -ac 'server is entering sleeping state' "$o/server.log")" -lt "$1" ]; do
        alive || return 1
        [ $(( $(date +%s) - t0 )) -lt 600 ] || return 1
        sleep 1
    done
    sleep 10
}

preset() {  # PRESET_ENV and ARGS of release/ornith-v5/settings.env: the split changed to the three cards, the fit on
    unset PRESET_MODEL PRESET_DRAFT PRESET_ENV PRESET_ARGS
    # shellcheck disable=SC1090
    . "$REPO/release/ornith-v5/settings.env"
    ARGS=$(sed -E "s/-ts [0-9,]+/-ts $TS/; s/-fit off/-fit on -lv 4/" <<< "$PRESET_ARGS")
    [[ "$ARGS" == *"-ts $TS"* && "$ARGS" == *"-fit on -lv 4"* ]] || { say "the preset's -ts or -fit off not found"; exit 1; }
    ARGS="$ARGS --spec-type draft-mtp --spec-draft-n-max $DRAFT_N --sleep-idle-seconds $SLEEP_S"
}

arm() {  # <arm> <bin folder> <extra server arguments>
    local a=$1 bin=$2 extra=$3 cpid spid="" samp="" t0 up1=0
    o=$WORK/load/$a
    c=$NAME-$a
    rm -rf "$o"; mkdir -p "$o"
    docker rm -f "$c" > /dev/null 2>&1
    printf 'bin: %s\nenv: %s %s\nargs: -m %s %s %s\n' "$bin" "$PRESET_ENV" "$ON" "$ORNITH" "$ARGS" "$extra" > "$o/command.txt"
    mark before
    say "$a: start ($extra)"
    # shellcheck disable=SC2086
    "${DR[@]}" -d --name "$c" --gpus "\"device=$GPUS\"" -p "127.0.0.1:$PORT:$PORT" "$IMG" \
        bash -c "env $PRESET_ENV $ON $bin/llama-server -m $ORNITH $ARGS $extra \
            --host 0.0.0.0 --port $PORT --no-webui > $o/server.log 2>&1; echo rc=\$? > $o/server.rc" > /dev/null
    mark started
    cpid=$(docker inspect -f '{{.State.Pid}}' "$c" 2> /dev/null)
    for _ in $(seq 1 100); do
        spid=$(pgrep -P "$cpid" -x llama-server | head -n 1)
        [ -n "$spid" ] && break
        sleep 0.1
    done
    if [ -n "$spid" ]; then
        python3 -I "$D/rss_sampler.py" "$spid" "$o/rss.tsv" > "$o/rss_sampler.out" 2>&1 &
        samp=$!
    else
        say "$a: no llama-server process found under the container's pid $cpid"
    fi
    t0=$(date +%s)
    while [ $(( $(date +%s) - t0 )) -lt "$LOAD_TIMEOUT" ]; do
        up && { up1=1; break; }
        alive || break
        sleep 1
    done
    if [ $up1 = 1 ]; then
        mark up1
        python3 -I "$REPLIES" "$URL" "$o/c1" "$ROUNDS" > "$o/c1.out" 2>&1
        mark replied1
        if asleep 1; then
            mark asleep1
            # one short request wakes the server: load 2 runs inside it
            curl -s -m "$LOAD_TIMEOUT" -H 'Content-Type: application/json' \
                -d '{"prompt": "Hello", "n_predict": 1, "temperature": 0, "cache_prompt": false}' \
                "$URL/completion" > "$o/wake.json" 2> "$o/wake.err"
            mark up2
            python3 -I "$REPLIES" "$URL" "$o/c2" "$ROUNDS" > "$o/c2.out" 2>&1
            mark replied2
            if asleep 2; then mark asleep2; else say "$a: no second sleep"; fi
        else
            say "$a: no first sleep"
        fi
    fi
    docker stop -t 60 "$c" > /dev/null 2>&1
    docker rm -f "$c" > /dev/null 2>&1
    [ -n "$samp" ] && wait "$samp"
    mark stopped
    say "$a: up=$up1 $(cat "$o/server.rc" 2>/dev/null), sleeps $(grep -ac 'server is entering sleeping state' "$o/server.log"),"\
        "copies $(grep -ac 'MTP output head copied' "$o/server.log"), asserts $(grep -ac 'GGML_ASSERT\|GGML_ABORT' "$o/server.log"),"\
        "replies $(cat "$o"/c1/replies.tsv "$o"/c2/replies.tsv 2>/dev/null | wc -l)"
}

[ -n "$PREFLIGHT" ] && { $PREFLIGHT -g "$GPUS" > "$WORK/preflight.txt" 2>&1; say "preflight rc=$?"; }
preset
for a in $ARMS; do
    case $a in
        new-unset)  arm "$a" "$NEW" "" ;;
        new-dev0)   arm "$a" "$NEW" "--mtp-device 0" ;;
        base-dev0)  arm "$a" "$REF" "--mtp-device 0" ;;
        base-unset) arm "$a" "$REF" "" ;;
        *) say "unknown arm $a" ;;
    esac
done

say "done: our containers left: $(docker ps -q --filter "name=^$NAME-" | wc -l); card MiB: $(smi)"
date -Is > "$WORK/DONE"
