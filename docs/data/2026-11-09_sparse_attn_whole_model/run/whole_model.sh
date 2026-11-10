#!/usr/bin/env bash
# whole_model.sh: Q1171's check, a changed copy of ../../2026-11-04_qsa_sparse_attn/run/long_decode.sh for a machine
# whose cards are newer than Pascal: the engine runs in a container through its compute 6.1 PTX and the requests come
# from the host, where python3 and its standard library are. Per set (plain: the model alone; draft: with the MTP
# draft) and arm one llama-server with the same build, flags and cards, and whole_requests.py (beside this script)
# against it: 128 greedy tokens after fresh prompts of DEPTHS tokens, then with PROBE=1 16 greedy tokens per depth
# with the top 10 log-probabilities of each.
# The build must hold only sm_61 code and compute 6.1 PTX: a newer card cannot run the sm_61 code, so its driver
# compiles the PTX (OUT/jit fills with the result). CUDA_FORCE_PTX_JIT=1 (PTX_JIT=1) is for a build that also holds
# the card's own code, and it stops this run: it puts cuBLAS on PTX too, cuBLAS ships none, and the first prompt
# batch aborts in cublasCreate.
# Arms: off       GGML_CUDA_FATTN_SPARSE_PASCAL=0: the dense kernels read every cell
#       on        =1: the sparse kernel takes each attention call of at most 8 query tokens once the cache holds
#                 1.5 x 2,051 cells per query token
#       fault     =1 with _FAULT=1: the sparse kernel multiplies its output by 1.01, so this arm's numbers leave the
#                 "on" arm's wherever the sparse kernel computed the attention, and nowhere else
#       offfault  =0 with _FAULT=1: the fault is the sparse kernel's own, so this is a second "off"
# Correctness only: no time is recorded.
#   BIN=<build>/bin MODEL=<first shard> DRAFT=<MTP draft> OUT=<work dir> GPUS=<the cards, comma-separated> \
#     MOUNTS="<folders the container reads>" PRESET_ENV="<the preset's variables>" \
#     setsid -f nohup bash whole_model.sh > <log> 2>&1 < /dev/null
# Env: BIN, MODEL, OUT, GPUS (docker's device list: indices or UUIDs); DRAFT (for the draft set); MOUNTS (folders
#      mounted read-only at their own paths; OUT is mounted writable); SETS (default "plain draft"), ARMS ("off on"),
#      DEPTHS (8192,32768,60000), CTX (65536), ARGS (the Flash-Next preset's flags without its nine-card split, its
#      token-embedding placement and its draft flags), DRAFT_ARGS (the preset's draft flags), PRESET_ENV (the preset's
#      variables), NSM (5: the sparse kernel splits its lists as on a P106-090), PROBE (1), PROBE_TOKENS (16),
#      EXTRA_ENV (more variables for every arm) with TAG (a suffix for those arms' names: "on" with TAG=s1 is
#      "on.s1"), PTX_JIT (0), PORT (8096), IMAGE (nvidia/cuda:12.9.2-devel-ubuntu24.04), DOCKER_ARGS (labels, CPU share)
# Output: OUT/whole.csv (one line per request), OUT/resp/<set>-<arm>-<depth>.json and probe-<set>-<arm>-<depth>.json,
#         OUT/server-<set>-<arm>.log, OUT/done
set -uo pipefail
: "${BIN:?}" "${MODEL:?}" "${OUT:?}" "${GPUS:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
SETS=${SETS:-plain draft}
ARMS=${ARMS:-off on}
DEPTHS=${DEPTHS:-8192,32768,60000}
CTX=${CTX:-65536}
ARGS=${ARGS:--ngl 99 -fa on -np 1 -bs --ctx-checkpoints 4 --cache-ram 0 --jinja --reasoning-format deepseek --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 -b 2048 -ub 512 -fit off -lm dio --no-host}
DRAFT_ARGS=${DRAFT_ARGS:---spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-n-ubatch 128}
NSM=${NSM:-5}
PROBE=${PROBE:-1}
PROBE_TOKENS=${PROBE_TOKENS:-16}
PORT=${PORT:-8096}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
CNAME=${CNAME:-sparse-whole-model}
mkdir -p "$OUT/resp" "$OUT/jit"
rm -f "$OUT/done"
vols=(-v "$OUT:$OUT")
for m in ${MOUNTS:-}; do vols+=(-v "$m:$m:ro"); done
[ -s "$OUT/whole.csv" ] || echo "set,arm,depth,prompt_tokens,prompt_md5,gen_tokens,draft_tokens,draft_accepted,text_md5" > "$OUT/whole.csv"
for set in $SETS; do
    extra=""
    if [ "$set" = draft ]; then : "${DRAFT:?}"; extra="-md $DRAFT $DRAFT_ARGS"; fi
    for arm in $ARMS; do
        case $arm in off) v=0 f=0 ;; on) v=1 f=0 ;; fault) v=1 f=1 ;; offfault) v=0 f=1 ;; *) echo "unknown arm $arm"; continue ;; esac
        # the JIT's cache is kept between loads: the same PTX compiles once per card type
        envs="CUDA_CACHE_PATH=$OUT/jit ${PRESET_ENV:-} GGML_CUDA_FATTN_SPARSE_PASCAL=$v GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=$NSM"
        [ $f = 1 ] && envs="$envs GGML_CUDA_FATTN_SPARSE_PASCAL_FAULT=1"
        [ "${PTX_JIT:-0}" = 1 ] && envs="CUDA_FORCE_PTX_JIT=1 $envs"
        [ -n "${EXTRA_ENV:-}" ] && envs="$envs $EXTRA_ENV"
        name=$arm${TAG:+.$TAG}
        eargs=(); for e in $envs; do eargs+=(-e "$e"); done
        LOG=$OUT/server-$set-$name.log
        echo "== $set $name: $envs $(date -Is)"
        docker rm -f "$CNAME" > /dev/null 2>&1
        # shellcheck disable=SC2086
        docker run -d --rm --name "$CNAME" ${DOCKER_ARGS:-} --gpus "\"device=$GPUS\"" -p "127.0.0.1:$PORT:$PORT" \
            "${vols[@]}" "${eargs[@]}" "$IMAGE" bash -c \
            "exec '$BIN/llama-server' -m '$MODEL' $extra -c $CTX $ARGS --no-webui --host 0.0.0.0 --port $PORT > '$LOG' 2>&1 < /dev/null" \
            > /dev/null
        for _ in $(seq 1 900); do
            curl -sf "http://127.0.0.1:$PORT/health" > /dev/null && break
            [ "$(docker inspect -f '{{.State.Running}}' "$CNAME" 2> /dev/null)" = true ] || break
            sleep 2
        done
        if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; then
            echo "cards (bus, MiB used): $(docker exec "$CNAME" nvidia-smi --query-gpu=pci.bus_id,memory.used --format=csv,noheader,nounits | tr '\n' ';')"
            nice -n 19 python3 -I "$HERE/whole_requests.py" "$PORT" "$set" "$name" "$DEPTHS" "$OUT/resp" "$PROBE" "$PROBE_TOKENS" | tee -a "$OUT/whole.csv"
        else
            echo "server did not start"; tail -n 20 "$LOG"
        fi
        docker stop -t 30 "$CNAME" > /dev/null 2>&1; docker rm -f "$CNAME" > /dev/null 2>&1
        # a container started with --rm is removed a moment after it stops: the next arm needs the name
        while docker container inspect "$CNAME" > /dev/null 2>&1; do sleep 1; done
        grep -E "out of memory|failed|error" "$LOG" | grep -v "unused tensor" | head -n 5
    done
done
echo "== end $(date -Is)"
touch "$OUT/done"
