#!/usr/bin/env bash
# C100 (Q640): the CPU check of LLAMA_QSA_KPOOL_DEV (the QSA k-pool tables kept on the devices) on a build machine, CPU
# only, every test in a container without network. On the CPU backend the tables' copy sits in a CPU buffer and the
# graph updates it as it updates a card's, so everything but CUDA's SET_ROWS kernel runs.
#   build   a static CPU build (GGML_CUDA=OFF, GGML_NATIVE=ON) of TIP's test-llama-archs and test-kpool-dev at
#           $WORK/cpu/tree/{src,build} (ccache: $WORK/ccache), under LOCK; binaries kept in $WORK/cpu/tip/bin
#   model   a random qwen4exp from TIP's test-llama-archs (-o)
#   kpool   test-kpool-dev on it: per-sequence and unified KV, the model's top-k (every pool kept) and 32 (the
#           selection drops pools), micro-batches of 16 and 512 tokens, SEEDS: three contexts (without the switch, with it
#           and LLAMA_QSA_KPOOL_DEV_CHECK=1, with it alone) through STEPS steps, every logit equal bit for bit
#   archs   test-llama-archs' logits files (LLAMA_ARCHS_LOGITS): BASE_BIN's build (the branch the switch was cut from)
#           writes every random model's logits, TIP compares with the switch off (every architecture), then
#           qwen4exp with LLAMA_QSA_KPOOL_DEV=1, with =1 and LLAMA_QSA_KPOOL_DEV_CHECK=1, and with =2 (the draft only)
# Usage: WORK=<dir> SRC=<llama.cpp clone holding TIP> TIP=<commit> BASE_BIN=<bin/ of the base's static CPU build>
#   [JOBS=8] [THREADS=4] [STEPS=240] [SEEDS='1 2'] [LOCK=<flock file>] [DOCKER_ARGS=...] [BASE_IMAGE=ubuntu:24.04]
#   [IMAGE=pq-c565-cpu:24.04] bash cpu_check.sh
# Output: $WORK/cpu/tip/ (tree.txt, cmake.log, build.log, bin/), $WORK/cpu/kpool/*.log, $WORK/cpu/archs/*.log,
#   $WORK/cpu/summary.tsv (check, arm, verdict, detail); done when $WORK/cpu/DONE exists
set -uo pipefail
WORK=$(realpath "${WORK:?WORK=<work dir>}")
SRC=$(realpath "${SRC:?SRC=<llama.cpp clone>}")
BASE_BIN=$(realpath "${BASE_BIN:?BASE_BIN=<bin/ of the base CPU build>}")
: "${TIP:?TIP=<commit>}"
JOBS=${JOBS:-8}
THREADS=${THREADS:-4}
STEPS=${STEPS:-240}
SEEDS=${SEEDS:-1 2}
BASE_IMAGE=${BASE_IMAGE:-ubuntu:24.04}
IMAGE=${IMAGE:-pq-c565-cpu:24.04}
C=$WORK/cpu
T=$C/tree
B=$T/build
D="--rm --label pascal-quants=1 --cpu-shares=256 --network none ${DOCKER_ARGS:-}"
mkdir -p "$C" "$T" "$WORK/ccache" "$C/kpool" "$C/archs" "$C/models"
rm -f "$C/DONE"
printf 'check\tarm\tverdict\tdetail\n' > "$C/summary.tsv"
verdict() { printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" | tee -a "$C/summary.tsv"; }
locked()  { if [ -n "${LOCK:-}" ]; then flock -w 43200 "$LOCK" "$@"; else "$@"; fi; }
# shellcheck disable=SC2086
box()     { docker run $D -v "$WORK:$WORK" -v "$BASE_BIN:$BASE_BIN:ro" "$IMAGE" "$@"; }

if ! docker image inspect "$IMAGE" > /dev/null 2>&1; then
    printf 'FROM %s\nRUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git cmake build-essential ccache python3 > /dev/null && rm -rf /var/lib/apt/lists/*\n' \
        "$BASE_IMAGE" | docker build -q --label pascal-quants=1 -t "$IMAGE" - > "$C/image.log" 2>&1 || { echo "image build failed: $C/image.log"; exit 1; }
fi
echo "=== c100 cpu check started $(date -Is), TIP ${TIP:0:9}"

R=$C/tip
mkdir -p "$R"
rm -rf "$T/src" "$B" "$R/bin" && mkdir -p "$T/src" "$R/bin"
git -C "$SRC" archive "$TIP" | tar -x -C "$T/src"
echo "commit $(git -C "$SRC" rev-parse "$TIP") $(git -C "$SRC" log -1 --format=%s "$TIP")" > "$R/tree.txt"
TARGETS="test-llama-archs test-kpool-dev"
# shellcheck disable=SC2086
locked docker run $D -v "$WORK:$WORK" -e CCACHE_DIR="$WORK/ccache" -e CCACHE_BASEDIR="$T" "$IMAGE" bash -c '
    cmake -S '"$T/src"' -B '"$B"' -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=ON \
        -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache > '"$R/cmake.log"' 2>&1 &&
    nice -n 19 cmake --build '"$B"' -j '"$JOBS"' --target '"$TARGETS"' > '"$R/build.log"' 2>&1'
rc=$?
verdict build tip "$([ $rc = 0 ] && echo OK || echo FAIL)" "rc=$rc; $(cat "$R/tree.txt"); $(grep -ac 'warning:' "$R/build.log") warning lines, $(grep -acE 'error:|undefined reference' "$R/build.log") error lines"
for t in $TARGETS; do [ -x "$B/bin/$t" ] && cp "$B/bin/$t" "$R/bin/"; done
if [ $rc != 0 ]; then
    date -Is > "$C/DONE"
    exit 1
fi

# the random qwen4exp
rm -rf "$C/models" && mkdir -p "$C/models"
box nice -n 19 "$R/bin/test-llama-archs" -a qwen4exp -s 1 -o "$C/models" > "$C/models.log" 2>&1
MODEL=$(ls "$C"/models/*.gguf 2>/dev/null | head -n 1)
verdict model qwen4exp "$([ -n "$MODEL" ] && echo OK || echo FAIL)" "${MODEL:-no file}; $(ls "$C/models" | tr '\n' ' ')"

if [ -n "$MODEL" ]; then
    for kv in seq unified; do
        for topk in 0 32; do
            for ub in 16 512; do
                for seed in $SEEDS; do
                    arm=$kv-k$topk-u$ub-s$seed
                    box nice -n 19 "$R/bin/test-kpool-dev" -m "$MODEL" -t "$THREADS" -s "$STEPS" -u "$ub" --top-k "$topk" \
                        --seed "$seed" $([ $kv = unified ] && echo --unified) > "$C/kpool/$arm.log" 2>&1
                    rc=$?
                    verdict kpool "$arm" "$([ $rc = 0 ] && echo OK || echo FAIL)" "rc=$rc; $(grep -a 'decodes, ' "$C/kpool/$arm.log" | tail -n 1); $(grep -a '^context 1:' "$C/kpool/$arm.log" | tail -n 1); $(grep -a 'checks, ' "$C/kpool/$arm.log" | tail -n 1)"
                done
            done
        done
    done
fi

# test-llama-archs' logits: the base writes, the branch compares
A=$C/archs
rm -f "$A"/*.f32 "$A"/*.log
archs() { # <tag> <bin dir> <env> [args]
    local tag=$1 bin=$2 envs=$3
    shift 3
    # shellcheck disable=SC2086
    box env LLAMA_ARCHS_LOGITS="$A/logits-" $envs nice -n 19 "$bin/test-llama-archs" -b CPU -s 1 "$@" 2>&1 | sed 's/\x1b\[[0-9;]*m//g' > "$A/$tag.log"
    echo "${PIPESTATUS[0]}"
}
rc=$(archs base "$BASE_BIN" '')
verdict archs base "$([ "$rc" = 0 ] && echo OK || echo FAIL)" "rc=$rc; $(ls "$A"/logits-*.f32 | wc -l) logits files; $(grep -a -m 1 '^Summary' "$A/base.log")"
cmp_arm() { # <tag> <env> [args]
    local tag=$1 envs=$2
    shift 2
    local rc n nd
    rc=$(archs "$tag" "$R/bin" "$envs" "$@")
    n=$(grep -ac 'logits against' "$A/$tag.log")
    nd=$(grep -a 'logits against' "$A/$tag.log" | grep -vc ', 0 of ')
    verdict archs "$tag" "$([ "$rc" = 0 ] && [ "$n" -gt 0 ] && [ "$nd" = 0 ] && echo SAME || echo DIFFERS)" \
        "rc=$rc; $n models compared, $nd differ: $(grep -a 'logits against' "$A/$tag.log" | grep -v ', 0 of ' | cut -d: -f1 | tr '\n' ' '); written by this build only: $(grep -ac 'logits_dump_or_compare: wrote' "$A/$tag.log"); $(grep -a 'kpool_dev_check: .* checked' "$A/$tag.log" | tail -n 1 | sed 's/.*kpool_dev_check: //')"
}
cmp_arm off   ''
cmp_arm on    'LLAMA_QSA_KPOOL_DEV=1' -a qwen4exp
cmp_arm check 'LLAMA_QSA_KPOOL_DEV=1 LLAMA_QSA_KPOOL_DEV_CHECK=1' -a qwen4exp
cmp_arm draft 'LLAMA_QSA_KPOOL_DEV=2' -a qwen4exp
rm -f "$A"/*.f32

date -Is > "$C/DONE"
echo "=== c100 cpu check done $(date -Is)"
