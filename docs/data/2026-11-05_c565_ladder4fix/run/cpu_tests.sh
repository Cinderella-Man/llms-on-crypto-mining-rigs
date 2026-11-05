#!/usr/bin/env bash
# C565: the whole CPU test set of fork commits on a build machine, CPU only, every binary in a container without network.
# For each tree of TREES, in order (one tree at a time, all at the same paths $WORK/tree/{src,build}, so ccache hits):
#   tree    git archive of the commit into $WORK/tree/src, then each +commit's own patch applied on top (git apply)
#   build   a CPU build of every target (GGML_CUDA=OFF, GGML_NATIVE=ON, LLAMA_BUILD_TESTS=ON; shared libraries, upstream's
#           default, or SHARED=OFF static as the rig's builds), a fresh build folder each tree, under LOCK, going on past
#           errors so the log lists them all; a tree with ":<targets>" builds only those
#   ctest   every registered test (ctest -j CTEST_JOBS, 1,800 s each, no network) except CTEST_EXCLUDE: by default the
#           five that need upstream's downloaded model (test-download-model, test-eval-callback-download-model and their
#           users test-llama-context-model, test-thread-safety, test-eval-callback: the hand step runs them on a
#           generated model instead), test-conversion-fp8 (Python with torch and safetensors) and NET_TESTS;
#           MODEL15M=<stories15M-q4_0.gguf> puts the model where the download tests find it, PIP='<packages>' adds them
#           to the image, and an empty CTEST_EXCLUDE runs everything but NET_TESTS
#   net     NET_TESTS (default test-arg-parser, whose last case requests http://ggml.ai/ and a missing page there)
#           through ctest in a container with network
#   archs   test-llama-archs -b CPU -s 1 without LLAMA_ARCHS_LOGITS, then with it twice: the first run writes every
#           model's logits, the second compares against them; every case passes and every file is the same
#   ident   with "@<base tag>": this tree's test-llama-archs compares against the logits the base tree's archs step
#           wrote (their copy), as the identity checks of the merges do: every compared file the same
#   hand    the tests the build only compiles, on random models: test-hop-cast --unit-only; test-route-once (routers x1,
#           x30), test-bytes-switches (head as it is, --zero-tail) and test-moe-ids on a qwen4exp MoE model with 4,096
#           tokens and 32 experts, 8 used (as their own checks make it); test-thread-safety, test-llama-context -m and
#           llama-eval-callback, with ctest's arguments, on the llama-dense.gguf test-generate-models wrote (ctest runs
#           them on the downloaded stories15M model)
# TREES: '<tag>=<commit>[+<commit>...][:<targets>][@<base tag>] ...'; a tree with ":<targets>" runs only archs and ident.
# Usage: WORK=<dir> SRC=<llama.cpp clone holding the commits> TREES='...' [STEPS='tree build ctest archs ident hand']
#   [JOBS=8] [SHARED=ON] [CTEST_JOBS=8] [THREADS=8] [LOCK=<flock file>] [DOCKER_ARGS=...] [BASE_IMAGE=ubuntu:24.04]
#   [IMAGE=pq-c565-cpu:24.04] [PIP='<packages>'] [MODEL15M=<file>] [CTEST_EXCLUDE=<names, a|b>] [NET_TESTS=<names>]
#   [KEEP_BUILD=1] bash cpu_tests.sh
# Output: $WORK/<tag>/ (tree.txt, build.log, ctest.log, ctest.xml, archs/*.log, ident/*.log, hand/*.log, the logits
#   in archs/logits/), $WORK/summary.tsv (tree, check, verdict, detail); done when $WORK/DONE exists
set -uo pipefail
WORK=$(realpath "${WORK:?WORK=<work dir>}")
SRC=$(realpath "${SRC:?SRC=<llama.cpp clone>}")
: "${TREES:?TREES=<tag>=<commit>...}"
STEPS=${STEPS:-tree build ctest net archs ident hand}
JOBS=${JOBS:-8}
SHARED=${SHARED:-ON}
CTEST_JOBS=${CTEST_JOBS:-8}
THREADS=${THREADS:-8}
BASE_IMAGE=${BASE_IMAGE:-ubuntu:24.04}
IMAGE=${IMAGE:-pq-c565-cpu:24.04}
NET_TESTS=${NET_TESTS:-test-arg-parser}
X=$NET_TESTS
[ -n "${MODEL15M:-}" ] || X="test-download-model|test-eval-callback-download-model|test-llama-context-model|test-thread-safety|test-eval-callback|$X"
[ -n "${PIP:-}" ] || X="test-conversion-fp8|$X"
# the ctest step never runs NET_TESTS (the net step does)
CTEST_EXCLUDE="^(${CTEST_EXCLUDE-$X}${CTEST_EXCLUDE+|$NET_TESTS})\$"
DN="--cpu-shares=256 ${DOCKER_ARGS:-}"
D="$DN --network none"
T=$WORK/tree
mkdir -p "$WORK" "$T" "$WORK/ccache"
rm -f "$WORK/DONE"
[ -f "$WORK/summary.tsv" ] || printf 'tree\tcheck\tverdict\tdetail\n' > "$WORK/summary.tsv"
verdict() { printf '%s\t%s\t%s\t%s\n' "$tag" "$1" "$2" "$3" | tee -a "$WORK/summary.tsv"; }
locked()  { if [ -n "${LOCK:-}" ]; then flock -w 21600 "$LOCK" "$@"; else "$@"; fi; }
step()    { [[ " $STEPS " == *" $1 "* ]]; }
# shellcheck disable=SC2086
box()     { docker run --rm $D -v "$WORK:$WORK" ${MODEL15M:+-v "$(dirname "$MODEL15M"):$(dirname "$MODEL15M"):ro"} "$IMAGE" "$@"; }

# the image: the compiler and the tests' tools, nothing else (ctest's Python test needs PIP's packages)
if ! docker image inspect "$IMAGE" > /dev/null 2>&1; then
    printf 'FROM %s\nRUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git cmake build-essential ccache python3 python3-jinja2 python3-pip > /dev/null && rm -rf /var/lib/apt/lists/*\n%s\n' \
        "$BASE_IMAGE" "${PIP:+RUN pip3 install --break-system-packages --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu $PIP}" \
        | docker build -q --label pascal-quants=1 -t "$IMAGE" - > "$WORK/image.log" 2>&1 || { echo "image build failed: $WORK/image.log"; exit 1; }
fi
echo "=== cpu tests started $(date -Is), trees: $TREES, steps: $STEPS"

for spec in $TREES; do
    tag=${spec%%=*} rest=${spec#*=}
    base=""; [[ "$rest" == *@* ]] && base=${rest##*@} rest=${rest%@*}
    targets=""; [[ "$rest" == *:* ]] && targets=${rest##*:} rest=${rest%:*}
    commit=${rest%%+*} patches=""; [[ "$rest" == *+* ]] && patches=${rest#*+}
    R=$WORK/$tag
    mkdir -p "$R"
    B=$T/build
    echo "=== $tag: $commit${patches:+ + ${patches//+/ + }}${targets:+ (targets $targets)} $(date -Is)"

    if step tree; then
        rm -rf "$T/src" && mkdir -p "$T/src"
        rc=0
        git -C "$SRC" archive "$commit" | tar -x -C "$T/src" || rc=$?
        for p in ${patches//+/ }; do
            # outside any repository (no parent's found), git apply patches the folder like patch -p1
            git -C "$SRC" format-patch -1 --stdout "$p" \
                | GIT_CEILING_DIRECTORIES=$T git -C "$T/src" apply --whitespace=nowarn - || rc=$?
        done
        { echo "commit $(git -C "$SRC" rev-parse "$commit")"; for p in ${patches//+/ }; do echo "patch $(git -C "$SRC" rev-parse "$p") $(git -C "$SRC" log -1 --format=%s "$p")"; done; } > "$R/tree.txt"
        verdict tree "$([ $rc = 0 ] && echo OK || echo FAIL)" "$(tr '\n' ' ' < "$R/tree.txt")"
        [ $rc = 0 ] || continue
    fi

    if step build; then
        rm -rf "$B"
        # -k: past the first error, so one build lists them all
        locked docker run --rm $D -v "$WORK:$WORK" -e CCACHE_DIR="$WORK/ccache" -e CCACHE_BASEDIR="$T" "$IMAGE" bash -c '
            cmake -S '"$T/src"' -B '"$B"' -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=ON \
                -DBUILD_SHARED_LIBS='"$SHARED"' -DLLAMA_CURL=OFF -DCMAKE_C_COMPILER_LAUNCHER=ccache \
                -DCMAKE_CXX_COMPILER_LAUNCHER=ccache > '"$R/cmake.log"' 2>&1 &&
            nice -n 19 cmake --build '"$B"' -j '"$JOBS"' '"${targets:+--target ${targets//,/ }}"' -- -k > '"$R/build.log"' 2>&1'
        rc=$?
        errs=$(grep -aE 'error:|undefined reference' "$R/build.log" | sed -E 's/^.*(error: .*|undefined reference to .*)$/\1/' | sort | uniq -c | sort -rn)
        verdict build "$([ $rc = 0 ] && echo OK || echo FAIL)" "rc=$rc; BUILD_SHARED_LIBS=$SHARED; ${targets:-all targets}; $(grep -acE 'error:|undefined reference' "$R/build.log") error lines${errs:+: $(echo "$errs" | head -n 5 | tr -s ' ' | tr '\n' ';')}"
        # a failed build still tests what it built (ctest names any test whose binary is missing)
        [ $rc = 0 ] || [ -x "$B/bin/test-llama-archs" ] || continue
    fi

    if step ctest && [ -z "$targets" ]; then
        if [ -n "${MODEL15M:-}" ]; then mkdir -p "$B/tinyllamas" && cp "$MODEL15M" "$B/tinyllamas/"; fi
        box bash -c "cd $B && nice -n 19 ctest -j $CTEST_JOBS --timeout 1800 --output-on-failure ${CTEST_EXCLUDE:+-E '$CTEST_EXCLUDE'} \
            --output-junit $R/ctest.xml" > "$R/ctest.log" 2>&1
        rc=$?
        sline=$(grep -a 'tests passed, ' "$R/ctest.log" | tail -n 1)
        failed=$(sed -n '/The following tests FAILED:/,$p' "$R/ctest.log" | grep -aE '^\s+[0-9]+ - ' | sed -E 's/^\s+[0-9]+ - //; s/ \(.*//' | tr '\n' ' ')
        verdict ctest "$([ $rc = 0 ] && echo OK || echo FAIL)" "${sline:-no summary}${failed:+; failed: $failed}${CTEST_EXCLUDE:+; not run: $CTEST_EXCLUDE}"
    fi

    if step net && [ -z "$targets" ]; then
        # shellcheck disable=SC2086
        docker run --rm $DN -v "$WORK:$WORK" "$IMAGE" bash -c "cd $B && nice -n 19 ctest --timeout 600 --output-on-failure \
            -R '^($NET_TESTS)\$'" > "$R/net.log" 2>&1
        rc=$?
        verdict net "$([ $rc = 0 ] && echo OK || echo FAIL)" "$(grep -a 'tests passed, ' "$R/net.log" | tail -n 1); $NET_TESTS, with network"
    fi

    if step archs; then
        A=$R/archs
        rm -rf "$A" && mkdir -p "$A/logits"
        box bash -c "
            run() { env \$2 nice -n 19 $B/bin/test-llama-archs -b CPU -s 1 2>&1 | sed 's/\x1b\[[0-9;]*m//g' > $A/\$1.log; echo \"\$1 rc=\${PIPESTATUS[0]}\"; }
            run plain ''
            run write LLAMA_ARCHS_LOGITS=$A/logits/logits-
            run compare LLAMA_ARCHS_LOGITS=$A/logits/logits-" > "$A/rc.txt" 2>&1
        for r in plain write compare; do
            s=$(grep -a -m 1 '^Summary' "$A/$r.log")
            verdict "archs $r" "$([[ "$s" == *'all '*' passed'* ]] && echo OK || echo FAIL)" "${s:-no summary}; $(grep "^$r " "$A/rc.txt")"
        done
        nw=$(ls "$A/logits" | wc -l)
        nc=$(grep -ac 'logits against' "$A/compare.log")
        bad=$(grep -a 'logits against' "$A/compare.log" | grep -v ', 0 of ' | cut -d: -f1 | tr '\n' ' ')
        bw=$(grep -a 'logits against' "$A/write.log" | cut -d: -f1 | tr '\n' ' ')
        verdict "archs files" "$([ "$nw" -gt 0 ] && [ "$nc" = "$nw" ] && [ -z "$bad" ] && [ -z "$bw" ] && echo OK || echo FAIL)" \
            "$nw files written; $nc compared, differ: ${bad:-none}; compared while writing (a shared file): ${bw:-none}"
    fi

    if step ident && [ -n "$base" ]; then
        I=$R/ident
        rm -rf "$I" && mkdir -p "$I/logits"
        cp "$WORK/$base/archs/logits/"logits-*.f32 "$I/logits/"
        nb=$(ls "$I/logits" | wc -l)
        box bash -c "LLAMA_ARCHS_LOGITS=$I/logits/logits- nice -n 19 $B/bin/test-llama-archs -b CPU -s 1 2>&1 | sed 's/\x1b\[[0-9;]*m//g' \
            > $I/against-$base.log; echo rc=\${PIPESTATUS[0]}" > "$I/rc.txt" 2>&1
        nc=$(grep -ac 'logits against' "$I/against-$base.log")
        bad=$(grep -a 'logits against' "$I/against-$base.log" | grep -v ', 0 of ' | cut -d: -f1 | tr '\n' ' ')
        new=$(grep -ac 'logits_dump_or_compare: wrote' "$I/against-$base.log")
        s=$(grep -a -m 1 '^Summary' "$I/against-$base.log")
        verdict "ident vs $base" "$([ "$nc" -gt 0 ] && [ -z "$bad" ] && echo SAME || echo DIFFERS)" \
            "$base wrote $nb files; $nc compared, differ: ${bad:-none}; written by $tag only: $new; $s"
    fi

    if step hand && [ -z "$targets" ]; then
        H=$R/hand
        rm -rf "$H" && mkdir -p "$H/random"
        # <check> <binary> [args]: exit status 0 passes; the log's last line goes into the detail
        hand() {
            local n=$1 b=$2; shift 2
            box nice -n 19 "$B/bin/$b" "$@" > "$H/$n.log" 2>&1
            local rc=$?
            verdict "hand $n" "$([ $rc = 0 ] && echo OK || echo FAIL)" "$b rc=$rc: $(grep -av '^ *$' "$H/$n.log" | tail -n 1 | cut -c1-160)"
        }
        hand hop-cast test-hop-cast --unit-only
        M=$H/random/qwen4exp-moe.gguf
        docker run --rm $D -v "$WORK:$WORK" -e LLAMA_TEST_ARCHS_N_VOCAB=4096 -e LLAMA_TEST_ARCHS_N_EXPERT=32,8 "$IMAGE" \
            "$B/bin/test-llama-archs" -a qwen4exp -o "$H/random" > "$H/random/archs.log" 2>&1
        if [ -f "$M" ]; then
            hand route-once-x1   test-route-once -m "$M" -t "$THREADS" -n 16 --router-scale 1
            hand route-once-x30  test-route-once -m "$M" -t "$THREADS" -n 16 --router-scale 30
            hand bytes-switches  test-bytes-switches -m "$M" -t "$THREADS" -n 24 -g 24 -k 1024 --router-scale 30
            hand bytes-zero-tail test-bytes-switches -m "$M" -t "$THREADS" -n 24 -g 24 -k 1024 --router-scale 30 --zero-tail
            hand moe-ids         test-moe-ids -m "$M" -t "$THREADS"
        else
            verdict "hand random model" FAIL "no $M: $H/random/archs.log"
        fi
        L=$B/tests/test-models/llama-dense.gguf
        if [ -f "$L" ]; then
            hand thread-safety test-thread-safety -m "$L" -ngl 99 -p "The meaning of life is" -n 128 -c 256 -ub 32 -np 4 -t 2
            hand context-model test-llama-context -m "$L"
            hand eval-callback llama-eval-callback -m "$L" --prompt hello --seed 42 -ngl 0
        else
            verdict "hand llama-dense" FAIL "no $L (ctest's test-generate-models writes it)"
        fi
    fi

    [ -n "${KEEP_BUILD:-}" ] || rm -rf "$B"
done

date -Is > "$WORK/DONE"
echo "=== done $(date -Is)"
cat "$WORK/summary.tsv"
