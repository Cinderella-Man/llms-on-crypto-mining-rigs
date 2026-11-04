#!/usr/bin/env bash
# Q797 and Q802 on a build machine, CPU only: block verification (LLAMA_SPEC_BLOCK_VERIFY) and the tool-call
# grammar's forced tokens appended to drafts (LLAMA_SPEC_FORCE, LLAMA_SPEC_DRAFT_KEEP), fork branch
# pascal-p106-v3-b12076-blockver, without a GPU or a real model:
#   1. build   llama-server, test-spec-block and test-llama-archs from $WORK/src (CPU backend, static); the base's
#              llama-server ($BASE_REF, the branch's parent) from $BASE_BIN, or built in a worktree $WORK/src-base
#   2. unit    test-spec-block: the paper's worked example, exactness by exact enumeration, block >= token
#              verification, the grammar's forced text
#   3. models  random-weight qwen4exp models (Flash-Next's architecture), each a target and a drafter correlated with
#              it (the same seed, other weight scales). The smoke stage's pair has Flash-Next's real tokenizer and chat
#              template, read from the header of $VOCAB (target stdev 0.1, drafter 0.13, 4,096-token context). The
#              stat stage's pair has GPT-2's tokenizer (VOCAB_STAT, 50,257 tokens: a grammar checks every token of a
#              row the sampler rejects, so a request is several times cheaper) and weights of stdev 1.0 (two drafters:
#              0.8, which accepts few, and 0.95), whose distributions are peaked: at 0.1 the top 8 of a row are near
#              uniform, and temperature 0.8 cannot be told from 1.0
#   4. smoke   llama-server arms on the same requests, draft-simple with --spec-draft-n-max 4, probabilistic draft
#              sampling, --spec-draft-p-min 0 (run/smoke.sh): plain decoding, the base build, the new build with no
#              switch, block verification, forced tokens with and without it, drafts cut to 0 and 2 tokens with forced
#              tokens, and a malformed value. Verdicts: the switches off give the base build's tokens and draft
#              counts on every request; every greedy reply (plain text, a JSON tool call and an XML one in
#              Qwen3.8's format under grammars) equals plain decoding; every grammar reply is whole; forced tokens
#              are appended and accepted (drafts cut to 0: the random drafter's tokens are rarely ones the grammar
#              allows); a malformed value stops the server
#   5. stat    sampled decoding (temperature 1, top-k 8), every arm its own seeds (run/stat.sh): $N_STAT times the
#              first five tokens after a text prompt for plain decoding, token-by-token rejection and block
#              verification, plus plain decoding at temperature 0.8 as the control the test must tell apart;
#              $N_STAT_G times a short JSON tool call under a grammar for plain decoding, rejection, block
#              verification, forced tokens and forced tokens only. run/stat_check.py: two-sample chi-square tests of
#              each arm against plain decoding (tokens 1, 2, 3, 1-2 and 1-3; the tool call's name and digit), total
#              variation, accepted drafted tokens
# Usage: WORK=<dir whose src/ holds the branch> REPO=<this repository> VOCAB=<a Flash-Next GGUF (its first file)>
#   [MOUNTS="-v <dir of VOCAB>:<same>:ro"] [BASE_BIN=<the base's CPU llama-server>] [BASE_REF=a330672c1] [JOBS=6]
#   [THREADS=6] [PORT=18191] [N_STAT=4000] [N_STAT_G=1500] [VOCAB_STAT=<a vocabulary GGUF>] [LOCK=<file>]
#   [DOCKER_ARGS=...]
#   [STEPS="build unit models smoke stat"] bash cpu_check.sh
# Output: $WORK/cpu/{bin,bin-base,models,unit,smoke,stat}/, $WORK/logs/, $WORK/cpu/summary.tsv (check, verdict,
# detail); done when $WORK/cpu/DONE exists
set -uo pipefail
WORK=$(realpath "${WORK:?WORK=<work dir>}")
REPO=$(realpath "${REPO:?REPO=<this repository>}")
VOCAB=${VOCAB:?VOCAB=<a GGUF with the tokenizer and chat template>}
BASE_REF=${BASE_REF:-a330672c1}
JOBS=${JOBS:-6}
THREADS=${THREADS:-6}
PORT=${PORT:-18191}
N_STAT=${N_STAT:-4000}
N_STAT_G=${N_STAT_G:-1500}
VOCAB_STAT=${VOCAB_STAT:-$WORK/src/models/ggml-vocab-gpt-2.gguf}
IMAGE=nvidia/cuda:12.9.2-devel-ubuntu24.04
D="--label pascal-quants=1 --cpu-shares 256 ${DOCKER_ARGS:-}"
C=$WORK/cpu
RUN=$REPO/docs/data/2026-11-04_block_verify/run
mkdir -p "$WORK/logs" "$C/bin" "$C/bin-base" "$C/models" "$C/unit" "$C/smoke" "$C/stat"
rm -f "$C/DONE"
printf 'check\tverdict\tdetail\n' > "$C/summary.tsv"
STEPS=${STEPS:-build unit models smoke stat}
has() { [[ " $STEPS " == *" $1 "* ]]; }
verdict() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" | tee -a "$C/summary.tsv"; }
locked() { if [ -n "${LOCK:-}" ]; then flock -w 10800 "$LOCK" "$@"; else "$@"; fi; }
echo "=== cpu check started $(date -Is)"

build() {
    docker run --rm $D -v "$WORK:/work" -e JOBS="$JOBS" -e BASE_REF="$BASE_REF" -e HAVE_BASE="$HAVE_BASE" \
        -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" "$IMAGE" bash -euo pipefail -c '
        export DEBIAN_FRONTEND=noninteractive CCACHE_DIR=/work/ccache-cpu
        apt-get update -qq && apt-get install -y -qq git cmake build-essential ccache > /dev/null
        git config --global --add safe.directory /work/src
        git config --global --add safe.directory /work/src-base
        cfg="-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_NATIVE=OFF -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF
             -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache"
        cd /work/src
        git log --oneline -1
        git status --short | head -20
        cmake -B build-cpu $cfg -DLLAMA_BUILD_TESTS=ON > /dev/null
        nice -n 19 cmake --build build-cpu -j "$JOBS" --target llama-server test-spec-block test-llama-archs \
            > /work/logs/cpu-build.log 2>&1 || { grep -E "error" /work/logs/cpu-build.log | head -40; exit 1; }
        cp build-cpu/bin/llama-server build-cpu/bin/test-spec-block build-cpu/bin/test-llama-archs /work/cpu/bin/
        if [ "$HAVE_BASE" != 1 ]; then
            [ -d /work/src-base ] || git worktree add -f /work/src-base "$BASE_REF" > /dev/null
            cd /work/src-base
            git checkout -q --force "$BASE_REF"
            git log --oneline -1
            cmake -B build-cpu $cfg -DLLAMA_BUILD_TESTS=OFF > /dev/null
            nice -n 19 cmake --build build-cpu -j "$JOBS" --target llama-server \
                > /work/logs/cpu-build-base.log 2>&1 || { grep -E "error" /work/logs/cpu-build-base.log | head -40; exit 1; }
            cp build-cpu/bin/llama-server /work/cpu/bin-base/
        fi
        chown -R "$HOST_UID:$HOST_GID" /work/cpu /work/logs'
}
export -f build
export WORK JOBS BASE_REF IMAGE D

if has build; then
    echo "=== 1. build $(date -Is)"
    HAVE_BASE=0
    if [ -n "${BASE_BIN:-}" ] && [ -x "$BASE_BIN" ]; then
        cp "$BASE_BIN" "$C/bin-base/llama-server" && HAVE_BASE=1
    fi
    export HAVE_BASE
    locked bash -c build > "$WORK/logs/cpu-build.console" 2>&1
    rc=$?
    if [ $rc != 0 ]; then
        verdict build FAIL "rc $rc; $(tail -n 5 "$WORK/logs/cpu-build.console" | tr '\n' ' ')"
        date -Is > "$C/DONE"; exit 1
    fi
    verdict build OK "$(grep -E '^[0-9a-f]{9} ' "$WORK/logs/cpu-build.console" | tr '\n' ';' | cut -c 1-200)$([ "$HAVE_BASE" = 1 ] && echo " base from $BASE_BIN")"
else
    verdict build SKIP "the last build's binaries"
fi

if has unit; then
    echo "=== 2. unit $(date -Is)"
    docker run --rm $D -v "$C/bin:/x:ro" -w /tmp "$IMAGE" /x/test-spec-block > "$C/unit/test-spec-block.log" 2>&1
    rc=$?
    verdict unit "$([ $rc = 0 ] && echo OK || echo FAIL)" "rc $rc; $(tail -n 1 "$C/unit/test-spec-block.log")"
fi

if has models; then
    echo "=== 3. models $(date -Is)"
    docker run --rm $D ${MOUNTS:-} -v "$C:/c" -v "$(dirname "$VOCAB_STAT"):/vs:ro" -e VOCAB_STAT="$VOCAB_STAT" \
        -e LLAMA_ARCHS_VOCAB="$VOCAB" -e LLAMA_ARCHS_N_CTX=4096 \
        -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" "$IMAGE" bash -c '
        set -e
        rm -rf /c/models/t /c/models/d /c/models/st /c/models/sd /c/models/sd2
        mkdir -p /c/models/t /c/models/d /c/models/st /c/models/sd /c/models/sd2
        /c/bin/test-llama-archs -a "^qwen4exp$" -s 1 -d 0.1  -o /c/models/t > /c/models/t.log 2>&1
        /c/bin/test-llama-archs -a "^qwen4exp$" -s 1 -d 0.13 -o /c/models/d > /c/models/d.log 2>&1
        export LLAMA_ARCHS_VOCAB=/vs/$(basename "$VOCAB_STAT")
        /c/bin/test-llama-archs -a "^qwen4exp$" -s 1 -d 1.0 -o /c/models/st > /c/models/st.log 2>&1
        /c/bin/test-llama-archs -a "^qwen4exp$" -s 1 -d 0.8 -o /c/models/sd > /c/models/sd.log 2>&1
        /c/bin/test-llama-archs -a "^qwen4exp$" -s 1 -d 0.95 -o /c/models/sd2 > /c/models/sd2.log 2>&1
        ls -la /c/models/t /c/models/d /c/models/st /c/models/sd /c/models/sd2
        chown -R "$HOST_UID:$HOST_GID" /c/models' > "$C/models/console.txt" 2>&1
    rc=$?
    T=$(ls "$C"/models/t/qwen4exp-moe.gguf 2>/dev/null || ls "$C"/models/t/qwen4exp-*.gguf 2>/dev/null | head -1)
    DR=$(ls "$C"/models/d/qwen4exp-moe.gguf 2>/dev/null || ls "$C"/models/d/qwen4exp-*.gguf 2>/dev/null | head -1)
    verdict models "$([ $rc = 0 ] && [ -n "$T" ] && [ -n "$DR" ] && echo OK || echo FAIL)" "rc $rc; target $(basename "${T:-none}"), draft $(basename "${DR:-none}")"
fi

T=$(ls "$C"/models/t/qwen4exp-moe.gguf 2>/dev/null || ls "$C"/models/t/qwen4exp-*.gguf 2>/dev/null | head -1)
DR=$(ls "$C"/models/d/qwen4exp-moe.gguf 2>/dev/null || ls "$C"/models/d/qwen4exp-*.gguf 2>/dev/null | head -1)
ST=$(ls "$C"/models/st/qwen4exp-moe.gguf 2>/dev/null || ls "$C"/models/st/qwen4exp-*.gguf 2>/dev/null | head -1)
SD=$(ls "$C"/models/sd/qwen4exp-moe.gguf 2>/dev/null || ls "$C"/models/sd/qwen4exp-*.gguf 2>/dev/null | head -1)
SD2=$(ls "$C"/models/sd2/qwen4exp-moe.gguf 2>/dev/null || ls "$C"/models/sd2/qwen4exp-*.gguf 2>/dev/null | head -1)

inner() { # <script> <log>
    docker run --rm $D -v "$C:/c" -v "$RUN:/r:ro" -e THREADS="$THREADS" -e PORT="$PORT" -e N_STAT="$N_STAT" \
        -e N_STAT_G="$N_STAT_G" -e T="/c/models/${T#$C/models/}" -e DR="/c/models/${DR#$C/models/}" \
        -e ST="/c/models/${ST#$C/models/}" -e SD="/c/models/${SD#$C/models/}" -e SD2="/c/models/${SD2#$C/models/}" \
        -e STAT_ARMS="${STAT_ARMS:-}" \
        -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" "$IMAGE" bash -c '
        export DEBIAN_FRONTEND=noninteractive
        apt-get update -qq && apt-get install -y -qq curl jq python3 > /dev/null
        bash /r/'"$1"'; rc=$?
        chown -R "$HOST_UID:$HOST_GID" /c
        exit $rc' > "$2" 2>&1
}

if has smoke; then
    echo "=== 4. smoke $(date -Is)"
    inner smoke.sh "$C/smoke-console.txt"
    S=$C/smoke-console.txt
    n_off=$(grep -c '^off R[1-6] identical' "$S")
    n_lossless=$(grep -c '^[a-z0-9-]* R[123] lossless$' "$S")
    n_lossy=$(grep -c 'NOT-LOSSLESS' "$S")
    n_json=$(grep -c '^[a-z0-9-]* R[2356] grammar ok$' "$S")
    n_json_bad=$(grep -c ' grammar BAD' "$S")
    forced=$(grep '^forced ' "$S" | tr '\n' ';')
    n_forced_ok=$(grep -c '^forced keep0: steps [1-9][0-9]*, appended [1-9][0-9]*, accepted [1-9]' "$S")
    bad=$(grep -c '^bad REFUSED' "$S")
    verdict smoke-off "$([ "$n_off" = 6 ] && echo OK || echo FAIL)" \
        "$n_off/6 replies of the new build without switches with the base build's tokens and draft counts"
    verdict smoke-lossless "$([ "$n_lossy" = 0 ] && [ "$n_lossless" -ge 21 ] && echo OK || echo FAIL)" \
        "$n_lossless greedy replies of the drafting arms equal to plain decoding, $n_lossy not"
    verdict smoke-json "$([ "$n_json_bad" = 0 ] && [ "$n_json" -ge 32 ] && echo OK || echo FAIL)" \
        "$n_json grammar replies whole (JSON or the XML tool call), $n_json_bad not"
    verdict smoke-forced "$([ "$n_forced_ok" = 1 ] && echo OK || echo FAIL)" "$forced"
    verdict smoke-bad "$([ "$bad" = 1 ] && echo OK || echo FAIL)" \
        "a malformed value $([ "$bad" = 1 ] && echo stops the server || echo did not stop the server)"
fi

if has stat; then
    echo "=== 5. stat $(date -Is)"
    inner stat.sh "$C/stat-console.txt"
    docker run --rm $D -v "$C:/c" -v "$RUN:/r:ro" "$IMAGE" bash -c '
        export DEBIAN_FRONTEND=noninteractive
        apt-get update -qq && apt-get install -y -qq python3 > /dev/null
        python3 -I /r/stat_check.py /c/stat' > "$C/stat/check.txt" 2>&1
    rc=$?
    cat "$C/stat/check.txt"
    verdict stat "$([ $rc = 0 ] && echo OK || echo FAIL)" "rc $rc; $(grep -E '^(verdict|power)' "$C/stat/check.txt" | tr '\n' ' ')"
fi

cat "$C/summary.tsv"
date -Is > "$C/DONE"
echo "=== cpu check ended $(date -Is)"
