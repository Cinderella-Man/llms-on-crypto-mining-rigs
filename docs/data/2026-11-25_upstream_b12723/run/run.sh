#!/usr/bin/env bash
# upstream-b12723 (2026-11-25, C1118): upstream llama.cpp's tag b12723 merged into Miner Llama's cumulative branch
# pascal-p106-v5 (at 2d839da1d, on upstream b12674) on the candidate branch pascal-p106-v5-cand-b12723, built once and
# checked on a build machine with three newer NVIDIA cards (RTX 3090) against the base's build. Correctness only:
# nothing is timed. land-v5-batch3's run.sh with these changes: the web page pinned to b12723; the model loads hold
# Swift Qwen3.8-Flash-Next Pascal Q2 GGUF (with its MTP draft) again and send two requests, as C912's did; each model
# also loads on the head with the merge's three new switches at upstream's value (UP_ENV: upstream's replay rollback,
# its fused norm and RoPE for M-RoPE models and its block sizes); the card suite gets an arm for the operators the merge
# changed (OPS_RE) on both builds and on the head with UP_ENV; and a step compares the two builds' machine code.
# CPU steps (STEPS, in this order):
#   src      a local clone of CLONE (a llama.cpp clone holding the base commit), the fork as remote "fork", the landing
#            branch fetched and checked against HEAD_SHA, its first-parent merges and diff listed; compiler caches
#            hard-linked from RIG_CCACHE and CPU_CCACHE
#   rig      the head's rig tarball: scripts/build_llamacpp_docker.sh REF with TESTS=1 (test-backend-ops and every check
#            tool), EXTRA_TARGETS and the web page pinned (HF_UI_VERSION), under LOCK; its md5 and its binaries' md5s;
#            warnings against BASE_RIG_LOG (the base tarball's build log)
#   base     the base's TESTS=1 tarball BASE_TGZ, checked against BASE_MD5, unpacked into $W/base
#   archs    test-llama-archs -b CPU -s 1 of the head's rig build against the logits the base's rig build writes in
#            the same run: every file compared bit for bit
#   cpu      cpu_tests.sh (docs/data/2026-11-05_c565_ladder4fix/run/) on two trees, base and land@base: a CPU build of
#            every target with CMAKE_ARGS (-DGGML_RPC=ON, so the RPC tests build and run), ctest without network,
#            test-llama-archs (plain, write, compare), ident (the head against the base's logits files), the hand-run
#            tests and the pipe step (C935, C1010, C1020: identity, sampled, race, streams and recurrent arms on every
#            architecture llama_pipe_init accepts)
#   compare  the two CPU builds' warnings; ctest test by test; the hand-run tests' verdicts and logs (compare_hand.py);
#            every pipe verdict of both trees
#   sass     test-backend-ops' compute 6.1 machine code (cuobjdump -sass) of both rig builds, kernel by kernel
#            (sass_compare.sh)
# GPU steps (GPU_STEPS, in this order, run beside the CPU steps from archs on, once rig and base are done):
#   gpupipe  bench/pipespec/gpu_pipe_check.sh (C1009, C1019) on the base's and the head's bin/ with its default arms:
#            the CPU step's random models pipelined against plain on one card and split over the three, greedy, sampled
#            and raced on the host, device sampling with fed tokens, the rollback check
#   load     Swift Qwen3.8-Flash-Next Pascal Q2 GGUF (with the MTP draft DRAFT), Flint Pascal Q8 GGUF and
#            Ornith-1.5-122B Pascal N2P GGUF with their v5 presets (release/*-v5/settings.env), every switch the preset
#            does not set at its default, the layer split changed to the three cards (-ts 1,1,1), the fork's
#            compute-6.1-only paths forced as on a P106 (env $PRESET_ENV $(pascal_force_preset "$PRESET_ENV"),
#            scripts/pascal_forced.sh): one llama-server per build (head, base, then head with UP_ENV), two greedy
#            requests (/completion, temperature 0, top-k 1, 256 tokens, no prompt cache, the tokens returned: C947's
#            coding request and C912's prose one), memory per card after the load and after the requests; token lists
#            compared head against base, and head with UP_ENV against head
#   bops     test-backend-ops test -b CUDA0, every case except the repeated-expert-id ones (dup=1), one card a share:
#            the first card: MUL_MAT live on the head with P106_ENV, with P106_ENV GGML_CUDA_MMVQ_PASCAL_MT=2 and with P106_ENV
#              GGML_CUDA_MMQ_J64_I64=2;
#            the second card: MUL_MAT_ID and MUL_MAT replayed from the kept record REC (C1023's run.sh, here suite_record.sh,
#              with P106_ENV) on the head and the base with --replay-allow-missing, then on the head with --cpu-check 25;
#              when every checked reference is identical and the head names cases the record lacks (NOREC), those
#              cases are recorded into REC (RESUME=1: a live check of each new case on the head, its processor
#              reference kept) and both builds replayed again without --replay-allow-missing, then the record's manifest;
#              when a reference differs, MUL_MAT_ID runs live on the head instead;
#            the third card: FLASH_ATTN_EXT unset, with GGML_CUDA_FATTN_TILE_PASCAL=2, and with every Pascal attention switch
#              (as the pull request 9 review ran it); then the operators OPS_RE (the gated delta net and its fusions, the
#              fused norm and RoPE, RoPE, softmax, the broadcast binary ops, fill, softplus) with P106_ENV on the head, the
#              base and the head with UP_ENV, each case's verdict compared between builds
# Usage: W=<work dir> CLONE=<llama.cpp clone with the base commit> BASE_TGZ=<base TESTS=1 tarball> BASE_MD5=<its md5>
#   BASE_RIG_LOG=<its build log> REPO=<folder with build_llamacpp_docker.sh cpu_tests.sh compare_hand.py
#   pascal_forced.sh suite_record.sh, pipespec/ (bench/pipespec with gpu_pipe_check.sh) and
#   release/{flash-next,flint,ornith}-v5/settings.env> FLINT_Q8=<file> ORNITH=<file> REC=<the kept record's folder>
#   [QUANT=<llama-quantize>, for a base tarball without one] [RIG_CCACHE=<dir>] [CPU_CCACHE=<dir>]
#   [JIT_SEED=<a PTX JIT cache to copy>] [LOCK=<flock file>] [PREFLIGHT=<cuda_preflight.sh>]
#   [MOUNTS='<folders the GPU containers mount at the same paths>'] [GPUS=0,1,2] [STEPS=...] [GPU_STEPS=...]
#   [PIPE_SEED=1] [RECORD_NEW=1] [JOBS=8] [THREADS=6] HEAD_SHA=<the candidate commit> FN=<file> DRAFT=<file>
#   bash run.sh (STEPS=none runs the GPU steps alone)
# Every container: labels pascal-quants=1 and cand12723=1, --cpu-shares=256; every CPU container --memory=16g and
# without network. Done when $W/DONE exists (it holds the exit code).
set -uo pipefail
W=${W:?W=<work dir>}
CLONE=${CLONE:?CLONE=<llama.cpp clone with the base commit>}
REPO=${REPO:?REPO=<folder with the scripts>}
REF=${REF:-fork/pascal-p106-v5-cand-b12723}
HEAD_SHA=${HEAD_SHA:?HEAD_SHA=<the candidate commit>}
BASE_SHA=${BASE_SHA:-2d839da1d8b7ae1dc0098acb8d4e0897d06757e7}
FORK_URL=${FORK_URL:-https://github.com/Cinderella-Man/miner-llama}
HF_UI_VERSION=${HF_UI_VERSION:-b12723}
EXTRA_TARGETS=${EXTRA_TARGETS:-}
STEPS=${STEPS:-src rig base sass archs cpu compare}
GPU_STEPS=${GPU_STEPS:-gpupipe load bops}
GPUS=${GPUS:-0,1,2}
JOBS=${JOBS:-8}
THREADS=${THREADS:-6}
PORT=${PORT:-18931}
LOAD_TIMEOUT=${LOAD_TIMEOUT:-1800}
RECORD_NEW=${RECORD_NEW:-1}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
CPU_IMAGE=${CPU_IMAGE:-pq-c565-cpu:24.04}
# shellcheck disable=SC1091
. "$REPO/pascal_forced.sh" || { echo "no $REPO/pascal_forced.sh"; exit 1; }
P106_ENV=${P106_ENV:-$PASCAL_FORCED_MMV} # the fork's matrix-vector kernels at a P106's shapes: C968's and C1023's set
# the merge's three switches at upstream's value: replay rollback (#31669), fused norm and RoPE for M-RoPE (#32500), its
# block sizes (#32500)
UP_ENV=${UP_ENV:-LLAMA_RS_REPLAY=1 GGML_CUDA_NORM_ROPE_MROPE=1 GGML_CUDA_NORM_ROPE_BS=1}
# the operators b12723 changed or that sit next to its changes, for test-backend-ops -o
OPS_RE=${OPS_RE:-GATED_DELTA_NET,GATED_DELTA_NET_CACHE_FUSION,GATED_DELTA_NET_STATE_ROWS,GDN_CONV,GDN_CONV_CACHE_FUSION,GATED_RMS_NORM,RMS_NORM,RMS_NORM_MUL_ROPE,ROPE,ROPE_SET_ROWS,SOFT_MAX,ADD,SUB,MUL,DIV,FILL,SOFTPLUS,SSM_CONV,SSM_SCAN,SSM_SCAN_ROLLBACK}
LBL="--label pascal-quants=1 --label cand12723=1"
MEM="--memory=16g --memory-swap=16g"
G="-e CUDA_VISIBLE_DEVICES= -e NVIDIA_VISIBLE_DEVICES=void"
IFS=, read -ra GA <<< "$GPUS"
mkdir -p "$W/cpu" "$W/rig" "$W/base" "$W/logs" "$W/checks"
rm -f "$W/DONE"
step()  { [[ " $STEPS " == *" $1 "* ]]; }
gstep() { [[ " $GPU_STEPS " == *" $1 "* ]]; }
locked() { if [ -n "${LOCK:-}" ]; then flock -w 21600 "$LOCK" "$@"; else "$@"; fi; }
say() { echo "$(date +%H:%M:%S) $*"; }
finish() {
    wait
    docker ps -q --filter label=cand12723=1 | xargs -r docker stop > /dev/null
    echo "=== end $(date -Is) rc=$1"; echo "$1" > "$W/DONE"; exit "$1"
}
export LC_ALL=C # sort and comm in one collation
# warning lines of a build log (gcc's "file:line:col: warning:", nvcc's "file(line): warning #N-D:", CMake's), each
# once, without line and column numbers (a moved line keeps its warning)
warnings() { grep -aE 'warning:|warning #[0-9]|CMake Warning' "$1" \
    | sed -E 's/:[0-9]+:[0-9]+: warning/: warning/; s/\([0-9]+\): warning/: warning/; s/CMakeLists.txt:[0-9]+/CMakeLists.txt/' | sort -u; }
strip() { sed 's/\x1b\[[0-9;]*m//g' "$1"; }
head_rig() { ls -d "$W"/rig/llama.cpp-*-"${HEAD_SHA:0:9}"-* 2>/dev/null | grep -v '\.tar\.gz$' | head -n 1; }
base_rig() { ls -d "$W"/base/llama.cpp-*-"${BASE_SHA:0:9}"* 2>/dev/null | grep -v '\.tar\.gz$' | head -n 1; }
echo "=== start $(date -Is): $REF ($HEAD_SHA) against $BASE_SHA; steps: $STEPS; GPU steps: $GPU_STEPS"

if step src; then
    S=$W/rig/src # where build_llamacpp_docker.sh (WORK=$W/rig) looks for its clone
    if [ ! -d "$S/.git" ]; then
        git clone -q --no-checkout "$CLONE" "$S" || finish 1
        git -C "$S" fetch -q "$CLONE" 'refs/remotes/*:refs/remotes/*'
        git -C "$S" remote set-url origin https://github.com/ggml-org/llama.cpp
    fi
    git -C "$S" remote get-url fork > /dev/null 2>&1 || git -C "$S" remote add fork "$FORK_URL"
    git -C "$S" remote set-url fork "$FORK_URL"
    git -C "$S" fetch -q fork "${REF#fork/}:refs/remotes/$REF" pascal-p106-v5:refs/remotes/fork/pascal-p106-v5 || finish 1
    got=$(git -C "$S" rev-parse "refs/remotes/$REF")
    echo "$REF = $got; fork/pascal-p106-v5 = $(git -C "$S" rev-parse refs/remotes/fork/pascal-p106-v5)"
    [ "$got" = "$HEAD_SHA" ] || { echo "$REF moved: stop"; finish 1; }
    git -C "$S" cat-file -e "$BASE_SHA^{commit}" || { echo "no base commit in $S"; finish 1; }
    { git -C "$S" log --oneline --first-parent "$BASE_SHA..$HEAD_SHA"; git -C "$S" diff --stat "$BASE_SHA" "$HEAD_SHA"; } > "$W/commits.txt"
    cat "$W/commits.txt"
    [ -d "$W/rig/ccache" ] || { [ -n "${RIG_CCACHE:-}" ] && cp -al "$RIG_CCACHE" "$W/rig/ccache"; }
    [ -d "$W/cpu/ccache" ] || { [ -n "${CPU_CCACHE:-}" ] && cp -al "$CPU_CCACHE" "$W/cpu/ccache"; }
    echo "=== sources ready $(date -Is)"
fi

if step rig; then
    rm -f "$W/rig/RC"
    locked env WORK="$W/rig" TESTS=1 JOBS="$JOBS" EXTRA_TARGETS="$EXTRA_TARGETS" \
        DOCKER_ARGS="$LBL --cpu-shares=256 $MEM -e HF_UI_VERSION=$HF_UI_VERSION" \
        bash "$REPO/build_llamacpp_docker.sh" "$REF" > "$W/logs/rig.log" 2>&1
    echo "rc=$?" | tee "$W/rig/RC"
    T=$(head_rig)
    [ -n "$T" ] && [ -f "$T.tar.gz" ] || { echo "no rig tarball"; finish 1; }
    (cd "$(dirname "$T")" && md5sum "$(basename "$T").tar.gz" && stat -c '%s bytes' "$(basename "$T").tar.gz") | tee "$W/rig/md5.txt"
    ls "$T/bin" | tr '\n' ' '; echo
    grep -a 'check tools' "$T/BUILDINFO.txt"
    (cd "$T/bin" && md5sum *) > "$W/rig/bin-md5-head.txt"
    if [ -n "${BASE_RIG_LOG:-}" ] && [ -f "$BASE_RIG_LOG" ]; then
        warnings "$BASE_RIG_LOG" > "$W/checks/rig-warnings-base.txt"
        warnings "$W/logs/rig.log" > "$W/checks/rig-warnings-head.txt"
        echo "rig warnings: base $(wc -l < "$W/checks/rig-warnings-base.txt") lines, head $(wc -l < "$W/checks/rig-warnings-head.txt"),"\
            "only in head: $(comm -13 "$W/checks/rig-warnings-base.txt" "$W/checks/rig-warnings-head.txt" | wc -l)"
        comm -13 "$W/checks/rig-warnings-base.txt" "$W/checks/rig-warnings-head.txt"
        echo "--- only in base:"; comm -23 "$W/checks/rig-warnings-base.txt" "$W/checks/rig-warnings-head.txt"
    fi
    echo "=== rig done $(date -Is)"
fi

if step base; then
    : "${BASE_TGZ:?BASE_TGZ=<base TESTS=1 tarball>}" "${BASE_MD5:?BASE_MD5=<its md5>}"
    got=$(md5sum "$BASE_TGZ" | cut -d' ' -f1)
    echo "base tarball md5 $got (expected $BASE_MD5)"
    [ "$got" = "$BASE_MD5" ] || { echo "base tarball md5 mismatch"; finish 1; }
    [ -n "$(base_rig)" ] || tar xzf "$BASE_TGZ" -C "$W/base" || finish 1
    B0=$(base_rig)
    grep -a -E '^(name|commit|check tools)' "$B0/BUILDINFO.txt"
    (cd "$B0/bin" && md5sum *) > "$W/rig/bin-md5-base.txt"
    echo "=== base done $(date -Is)"
fi

if step sass; then
    # MOUNTS first: a command's prefix assignments are expanded in order, so W="$W/sass" before it would mount sass twice
    MOUNTS="$W" W="$W/sass" BASE="$(base_rig)/bin" HEAD="$(head_rig)/bin" DOCKER_ARGS="$LBL --cpu-shares=256 $MEM" \
        bash "$REPO/sass_compare.sh" > "$W/logs/sass.log" 2>&1
    echo "sass rc=$?"; tail -n 12 "$W/logs/sass.log"
    echo "=== sass done $(date -Is)"
fi

# ---------------------------------------------------------------------------------------------------- GPU steps
gpu_chain() {
    T=$(head_rig); B0=$(base_rig)
    [ -n "$T" ] && [ -n "$B0" ] || { echo "gpu: no head or base build"; return 1; }
    mkdir -p "$W/jitcache"
    if [ -n "${JIT_SEED:-}" ] && [ -d "$JIT_SEED" ] && [ -z "$(ls -A "$W/jitcache")" ]; then cp -a "$JIT_SEED/." "$W/jitcache/"; fi
    VOL=()
    for m in ${MOUNTS:-}; do VOL+=(-v "$m:$m:ro"); done
    VOL+=(-v "$W:$W")
    DR=(docker run --rm $LBL --cpu-shares=256 "${VOL[@]}" -e "CUDA_CACHE_PATH=$W/jitcache" -e CUDA_CACHE_MAXSIZE=4294967296)

    if gstep gpupipe; then
        rm -rf "$W/gpupipe" && mkdir -p "$W/gpupipe"
        say "gpupipe start"
        # QUANT: one llama-quantize for both builds when the base tarball has none (built before C1017)
        env MODEL_SEED="${PIPE_SEED:-1}" WORK="$W/gpupipe" BUILDS="base=$B0/bin head=$T/bin" GPUS="$GPUS" MOUNTS="${MOUNTS:-}" \
            DOCKER_ARGS="$LBL --cpu-shares=256" PREFLIGHT="${PREFLIGHT:-}" QUANT="${QUANT:-}" \
            JITCACHE="$W/jitcache" NAME=c1118-pipe bash "$REPO/pipespec/gpu_pipe_check.sh" > "$W/logs/gpupipe.log" 2>&1
        say "gpupipe rc=$?"
        cut -f1-6 "$W/gpupipe/summary.tsv" 2>/dev/null
    fi

    if gstep load; then
        : "${FN:?}" "${DRAFT:?}" "${FLINT_Q8:?}" "${ORNITH:?}"
        [ -n "${PREFLIGHT:-}" ] && { $PREFLIGHT -g "$GPUS" > "$W/checks/load-preflight.txt" 2>&1; say "load preflight rc=$?"; }
        P_CODE='Write a Python function that parses an ISO 8601 date string into a datetime object, then three pytest tests for it.'
        P_PROSE='Explain in one paragraph how a hash map handles collisions, then give one example in Python.'
        model() {  # <name>: sets M_FILE M_EXTRA M_ENV M_ARGS from the model's v5 preset, the split changed to the three cards
            unset PRESET_MODEL PRESET_DRAFT PRESET_ENV PRESET_ARGS
            case $1 in
                # shellcheck disable=SC1091
                flashnext) . "$REPO/release/flash-next-v5/settings.env"; M_FILE=$FN; M_EXTRA="-md $DRAFT" ;;
                # shellcheck disable=SC1091
                flintq8) . "$REPO/release/flint-v5/settings.env"; M_FILE=$FLINT_Q8; M_EXTRA="" ;;
                # shellcheck disable=SC1091
                ornith) . "$REPO/release/ornith-v5/settings.env"; M_FILE=$ORNITH; M_EXTRA="" ;;
                *) say "unknown model $1"; return 1 ;;
            esac
            M_ENV="$PRESET_ENV $(pascal_force_preset "$PRESET_ENV")"
            M_ARGS=$(sed -E "s/-ts [0-9,]+/-ts 1,1,1/" <<< "$PRESET_ARGS")
            [[ "$M_ARGS" == *"-ts 1,1,1"* ]] || { say "$1: no -ts in the preset to change"; return 1; }
        }
        load() {  # <model> <label> <bin folder> [<extra environment>]
            local o=$W/load/$1-$2 c=c1118-$1-$2 up=0 t0 p body env_all="$M_ENV ${4:-}"
            mkdir -p "$o"
            docker rm -f "$c" > /dev/null 2>&1
            printf 'env: %s\nargs: -m %s %s %s\nbin: %s\n' "$env_all" "$M_FILE" "$M_EXTRA" "$M_ARGS" "$3" > "$o/command.txt"
            md5sum "$3/llama-server" >> "$o/command.txt"
            say "load $1-$2: $3"
            # shellcheck disable=SC2086
            "${DR[@]}" -d --name "$c" --gpus "\"device=$GPUS\"" -p "127.0.0.1:$PORT:$PORT" "$IMAGE" \
                bash -c "env $env_all $3/llama-server -m $M_FILE $M_EXTRA $M_ARGS --host 0.0.0.0 --port $PORT --metrics --no-webui \
                    > $o/server.log 2>&1; echo rc=\$? > $o/server.rc" > /dev/null
            t0=$(date +%s)
            while [ $(( $(date +%s) - t0 )) -lt "$LOAD_TIMEOUT" ]; do
                if curl -sf -m 5 "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then up=1; break; fi
                docker ps -q --filter "name=^$c\$" | grep -q . || break
                sleep 5
            done
            echo "up=$up after $(( $(date +%s) - t0 )) s" > "$o/load.txt"
            if [ $up = 1 ]; then
                sleep 20
                nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits -i "$GPUS" > "$o/mem.csv"
                for p in code prose; do
                    if [ $p = code ]; then body=$P_CODE; else body=$P_PROSE; fi
                    printf '{"prompt": "%s", "n_predict": 256, "temperature": 0, "top_k": 1, "seed": 1, "cache_prompt": false, "return_tokens": true}' "$body" |
                        curl -s -m 900 -H 'Content-Type: application/json' -d @- "http://127.0.0.1:$PORT/completion" \
                        > "$o/reply-$p.json" 2> "$o/reply-$p.err"
                    say "  $1-$2 $p: $(wc -c < "$o/reply-$p.json") B of JSON"
                done
                nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits -i "$GPUS" > "$o/mem-after.csv"
            fi
            docker stop -t 60 "$c" > /dev/null 2>&1
            docker rm -f "$c" > /dev/null 2>&1
            say "load $1-$2: $(cat "$o/load.txt")"
        }
        compare() {  # <model> <label a> <label b>
            local p
            for p in code prose; do
                python3 -I - "$W/load/$1-$2/reply-$p.json" "$W/load/$1-$3/reply-$p.json" <<'EOF' | sed "s/^/$1 $p $2-$3: /" | tee -a "$W/load/compare.txt"
import json, sys
def reply(p):
    try:
        d = json.load(open(p))
    except Exception:
        return None
    t = d.get("timings", {})
    return d.get("tokens"), d.get("content", ""), (t.get("draft_n"), t.get("draft_n_accepted"))
a, b = reply(sys.argv[1]), reply(sys.argv[2])
if a is None or b is None or a[0] is None or b[0] is None:
    print("a reply is missing or not JSON")
    sys.exit(0)
same = a[0] == b[0]
n = next((i for i, (x, y) in enumerate(zip(a[0], b[0])) if x != y), min(len(a[0]), len(b[0])))
print(f"tokens {len(a[0])} and {len(b[0])}; {'the same token ids' if same else f'first difference at token {n}'}; "
      f"text {'the same' if a[1] == b[1] else 'other'} ({len(a[1])} and {len(b[1])} chars); drafted, accepted {a[2]} and {b[2]}")
EOF
            done
        }
        for m in ${MODELS_RUN:-flashnext flintq8 ornith}; do
            model "$m" || continue
            load "$m" head "$T/bin"
            load "$m" base "$B0/bin"
            compare "$m" head base
            if [ -n "$UP_ENV" ]; then
                load "$m" headup "$T/bin" "$UP_ENV"
                compare "$m" headup head
            fi
        done
    fi

    if gstep bops; then
        o=$W/bops; mkdir -p "$o"; : > "$o/rc.txt"
        F='^(?!.*dup=1)'
        P=GGML_CUDA_FATTN_TILE_PASCAL
        [ -n "${PREFLIGHT:-}" ] && { $PREFLIGHT -g "$GPUS" > "$o/preflight.txt" 2>&1; say "bops preflight rc=$?"; }
        md5sum "$T/bin/test-backend-ops" > "$o/md5.txt"
        run_bops() {  # <card> <tag> <operators> <environment> [<bin folder>, default the head's]
            "${DR[@]}" --gpus "\"device=$1\"" "$IMAGE" \
                bash -c "env C1118_ARM=$2 $4 nice -n 19 timeout 10800 ${5:-$T/bin}/test-backend-ops test -b CUDA0 -o '$3' -p '$F' 2>&1 \
                    | sed 's/\x1b\[[0-9;]*m//g' > $o/$2.log; echo $2 rc=\${PIPESTATUS[0]} >> $o/rc.txt" > /dev/null 2>&1
            say "bops $2 done: $(grep "^$2 " "$o/rc.txt")"
        }
        # the kept record of MUL_MAT_ID and MUL_MAT (C1023): replays, the cpu-check, new cases recorded, replays again
        replay_chain() {
            local rr="$REPO/suite_record.sh" r=$W/replay ok noid nomm sh live=0
            rr_run() {  # <work dir> <environment ...>: suite_record.sh on the second card, with P106_ENV
                local wd=$1; shift
                mkdir -p "$wd/jitcache"
                [ -z "$(ls -A "$wd/jitcache")" ] && cp -a "$W/jitcache/." "$wd/jitcache/"
                env WORK="$wd" NEW="$T/bin" REC="${REC:?REC=<the kept record>}" GPU="${GA[1]}" MOUNTS="${MOUNTS:-} $W" \
                    DOCKER_ARGS="$LBL --cpu-shares=256" PREFLIGHT="${PREFLIGHT:-}" NAME=c1118-rr ON="$P106_ENV" "$@" \
                    bash "$rr" > "$wd.log" 2>&1
                say "replay $(basename "$wd") rc=$?: $(tr '\n' ';' < "$wd/bops/rc.txt" 2>/dev/null)"
            }
            rr_run "$r" STEPS=replay BUILDS="head=$T/bin base=$B0/bin" SHARES="mul_mat_id mul_mat" ALLOW_MISSING=1 CPU_CHECK=25
            # the processor's references still the record's: every checked one identical, no verdict differs
            ok=1
            for sh in mul_mat_id mul_mat; do
                strip "$r/bops/$sh-replay-head-cpucheck.log" | grep -a 'replay cpu-check:' | tee -a "$o/replay-cpucheck.txt" \
                    | grep -qE 'cpu-check: ([0-9]+) cases, \1 references identical, 0 verdicts differ$' || ok=0
            done
            noid=$(strip "$r/bops/mul_mat_id-replay-head.log" | grep -a -c ': NOREC')
            nomm=$(strip "$r/bops/mul_mat-replay-head.log" | grep -a -c ': NOREC')
            say "replay: references the record's: $ok; head cases without a record: MUL_MAT_ID $noid, MUL_MAT $nomm"
            if [ "$ok" = 1 ] && [ "$RECORD_NEW" = 1 ] && [ $((noid + nomm)) -gt 0 ]; then
                sh=""; [ "$noid" -gt 0 ] && sh="mul_mat_id"; [ "$nomm" -gt 0 ] && sh="$sh mul_mat"
                rr_run "$W/record" STEPS=record RESUME=1 SHARES="${sh# }"
                rr_run "$W/replay2" STEPS="replay manifest" BUILDS="head=$T/bin base=$B0/bin" SHARES="mul_mat_id mul_mat" \
                    ALLOW_MISSING=0 CPU_CHECK=0
            elif [ "$ok" != 1 ]; then
                live=1
            fi
            # a processor reference that differs: the record no longer fits, so MUL_MAT_ID live on the head
            [ "$live" = 1 ] && run_bops "${GA[1]}" mul_mat_id-p106 MUL_MAT_ID "$P106_ENV"
            return 0
        }
        ( run_bops "${GA[0]}" mul_mat-p106 MUL_MAT "$P106_ENV"
          run_bops "${GA[0]}" mul_mat-p106-mt2 MUL_MAT "$P106_ENV GGML_CUDA_MMVQ_PASCAL_MT=2"
          run_bops "${GA[0]}" mul_mat-p106-j64 MUL_MAT "$P106_ENV GGML_CUDA_MMQ_J64_I64=2" ) &
        ( replay_chain ) &
        ( run_bops "${GA[2]}" fa-plain FLASH_ATTN_EXT "X=1"
          run_bops "${GA[2]}" fa-forced FLASH_ATTN_EXT "$P=2"
          run_bops "${GA[2]}" fa-all FLASH_ATTN_EXT "$P=2 ${P}_LDS=1 ${P}_KDP=1 ${P}_MIN=1 GGML_CUDA_FATTN_DEC_PASCAL=4"
          run_bops "${GA[2]}" ops-head "$OPS_RE" "$P106_ENV"
          run_bops "${GA[2]}" ops-base "$OPS_RE" "$P106_ENV" "$B0/bin"
          run_bops "${GA[2]}" ops-headup "$OPS_RE" "$P106_ENV $UP_ENV" ) &
        wait
        printf 'arm\trc\tpassed\tOK\tFAIL\tnot supported\tillegal access\n' > "$o/summary.tsv"
        for t in mul_mat-p106 mul_mat-p106-mt2 mul_mat-p106-j64 mul_mat_id-p106 fa-plain fa-forced fa-all ops-head ops-base ops-headup; do
            f=$o/$t.log
            [ -f "$f" ] || continue
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$t" "$(grep "^$t " "$o/rc.txt" | sed 's/.*rc=//')" \
                "$(grep -aoE '[0-9]+/[0-9]+ tests passed' "$f" | tail -n 1)" "$(grep -a -c ': OK$' "$f")" \
                "$(grep -a -c ': FAIL' "$f")" "$(grep -a -c 'not supported' "$f")" "$(grep -a -c 'illegal memory access' "$f")" >> "$o/summary.tsv"
            grep -a ': FAIL' "$f" | grep -av 'Backend' | sed 's/^ */    /' | cut -c1-220 >> "$o/failures.txt"
        done
        for d in replay record replay2; do
            [ -f "$W/$d/bops/summary.txt" ] && { echo "--- $d"; grep -av '^    ' "$W/$d/bops/summary.txt"; }
        done
        cat "$o/summary.tsv"; cat "$o/failures.txt" 2>/dev/null | head -n 20
        # the ops arms case by case: each case's verdict on the head, the base and the head with UP_ENV
        for t in ops-head ops-base ops-headup; do
            grep -aE '^ +[A-Z_0-9]+\(.*\): (OK|FAIL|not supported)' "$o/$t.log" 2>/dev/null \
                | sed -E 's/^ +//; s/: (OK|FAIL|not supported).*$/\t\1/' | sort -u > "$o/$t.cases.tsv"
        done
        join -t$'\t' -a1 -a2 -e '-' -o 0,1.2,2.2 "$o/ops-head.cases.tsv" "$o/ops-base.cases.tsv" > "$o/ops-head-base.tsv"
        join -t$'\t' -a1 -a2 -e '-' -o 0,1.2,2.2 "$o/ops-headup.cases.tsv" "$o/ops-head.cases.tsv" > "$o/ops-headup-head.tsv"
        echo "ops, head against base: $(wc -l < "$o/ops-head-base.tsv") cases, $(awk -F'\t' '$2 == $3' "$o/ops-head-base.tsv" | wc -l) the same verdict, head only $(awk -F'\t' '$3 == "-"' "$o/ops-head-base.tsv" | wc -l), base only $(awk -F'\t' '$2 == "-"' "$o/ops-head-base.tsv" | wc -l)"
        awk -F'\t' '$2 != $3 && $2 != "-" && $3 != "-"' "$o/ops-head-base.tsv" | head -n 20
        echo "ops, head with UP_ENV against head: $(awk -F'\t' '$2 == $3' "$o/ops-headup-head.tsv" | wc -l) of $(wc -l < "$o/ops-headup-head.tsv") the same verdict"
        awk -F'\t' '$2 != $3' "$o/ops-headup-head.tsv" | head -n 20
    fi
    say "gpu done: cand12723 containers left: $(docker ps -q --filter label=cand12723=1 | wc -l); card MiB: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPUS" | paste -sd' ')"
}

if [ -n "$GPU_STEPS" ]; then
    gpu_chain > "$W/logs/gpu.out" 2>&1 &
    echo "GPU chain started in the background (log $W/logs/gpu.out), pid $!"
fi

# ---------------------------------------------------------------------------------------------------- CPU steps
if step archs; then
    T=$(head_rig); B0=$(base_rig)
    K=$W/checks/archs
    rm -rf "$K" && mkdir -p "$K"
    docker run --rm $LBL --cpu-shares=256 $MEM --network none $G \
        -v "$B0:/base:ro" -v "$T:/head:ro" -v "$K:/k" "$IMAGE" bash -c '
        mkdir -p /tmp/stub && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /tmp/stub/libcuda.so.1
        export LD_LIBRARY_PATH=/tmp/stub:${LD_LIBRARY_PATH:-}
        run() { env $3 nice -n 19 timeout 3600 $2/test-llama-archs -b CPU -s 1 2>&1 | sed "s/\x1b\[[0-9;]*m//g" > /k/$1.log; echo "$1 rc=${PIPESTATUS[0]}" >> /k/rc.txt; }
        run head-plain /head/bin ""
        run base-write /base/bin LLAMA_ARCHS_LOGITS=/k/logits-
        ls /k/logits-*.f32 | wc -l > /k/n_files.txt
        run head-compare /head/bin LLAMA_ARCHS_LOGITS=/k/logits-
        rm -f /k/logits-*.f32
        chown -R '"$(id -u):$(id -g)"' /k'
    cat "$K/rc.txt"
    echo "files written by base: $(cat "$K/n_files.txt"); compared: $(grep -ac 'logits against' "$K/head-compare.log");"\
        "differ: $(grep -a 'logits against' "$K/head-compare.log" | grep -vc ', 0 of ')"
    grep -a 'logits against' "$K/head-compare.log" | grep -v ', 0 of ' | head -n 40
    for f in head-plain base-write head-compare; do echo "$f: $(grep -a -m 1 '^Summary' "$K/$f.log")"; done
    echo "=== archs done $(date -Is)"
fi

if step cpu; then
    WORK="$W/cpu" SRC="$W/rig/src" TREES="base=$BASE_SHA land=$HEAD_SHA@base" STEPS="tree build ctest archs ident hand pipe" \
        JOBS="$JOBS" CTEST_JOBS=6 THREADS="$THREADS" LOCK="${LOCK:-}" CMAKE_ARGS="-DGGML_RPC=ON" \
        DOCKER_ARGS="$LBL $MEM" IMAGE="$CPU_IMAGE" bash "$REPO/cpu_tests.sh" > "$W/logs/cpu.log" 2>&1
    echo "cpu tests rc=$?"
    cat "$W/cpu/summary.tsv"
    echo "=== cpu done $(date -Is)"
fi

if step compare; then
    K=$W/checks
    warnings "$W/cpu/base/build.log" > "$K/cpu-warnings-base.txt"
    warnings "$W/cpu/land/build.log" > "$K/cpu-warnings-head.txt"
    echo "cpu warnings: base $(wc -l < "$K/cpu-warnings-base.txt") lines, head $(wc -l < "$K/cpu-warnings-head.txt"),"\
        "only in head: $(comm -13 "$K/cpu-warnings-base.txt" "$K/cpu-warnings-head.txt" | wc -l)"
    comm -13 "$K/cpu-warnings-base.txt" "$K/cpu-warnings-head.txt"
    for t in base land; do
        grep -aE '^ *[0-9]+/[0-9]+ Test +#[0-9]+: ' "$W/cpu/$t/ctest.log" \
            | sed -E 's/^ *[0-9]+\/[0-9]+ Test +#[0-9]+: ([^ ]+) [. ]*(\*{0,3}[A-Za-z][A-Za-z *]*[A-Za-z]).*$/\1\t\2/' | sort > "$K/ctest-$t.tsv"
    done
    echo "ctest: base $(wc -l < "$K/ctest-base.tsv") tests ($(grep -c 'Passed$' "$K/ctest-base.tsv") passed), head $(wc -l < "$K/ctest-land.tsv") ($(grep -c 'Passed$' "$K/ctest-land.tsv") passed)"
    echo "ctest, test by test, head against base: $(diff -q "$K/ctest-base.tsv" "$K/ctest-land.tsv" > /dev/null && echo same || echo DIFFERENT)"
    join -t$'\t' -a1 -a2 -e '-' -o 0,1.2,2.2 "$K/ctest-base.tsv" "$K/ctest-land.tsv" > "$K/ctest.tsv"
    awk -F'\t' '$2 != $3' "$K/ctest.tsv"
    echo "failed on head:"; awk -F'\t' '$3 != "Passed"' "$K/ctest.tsv"
    python3 -I "$REPO/compare_hand.py" "$W/cpu/base" "$W/cpu/land" "$K/hand.tsv" | tail -n 8
    echo "pipe verdicts (tree, check, verdict):"
    grep -P '\tpipe ' "$W/cpu/summary.tsv" | cut -f1-3 | sort -t$'\t' -k2,2 -k1,1
    echo "=== compare done $(date -Is)"
fi

finish 0
