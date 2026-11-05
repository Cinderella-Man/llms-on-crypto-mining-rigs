#!/usr/bin/env bash
# The v2-era ports (C524) on a build machine, CPU only: each fork branch pascal-p106-v3-b12076-<name> holds one
# feature of a v2-era branch on -ladder4 (e3b30303d), off by default, and is checked against -ladder4's own rig build:
#   prep      $ROOT/<name>/src: a clone of BASE_SRC with llama.cpp and the fork (FORK) as remotes, and a copy of
#             BASE_CCACHE in $ROOT/<name>/ccache
#   build     the rig tarball (scripts/build_llamacpp_docker.sh, TESTS=1: sm_61 code and compute_61 PTX, no AVX, with
#             test-backend-ops) and, from its build tree, the extra tests (test-llama-archs; test-quantize-fns for
#             q41g64) into the unpacked tree's bin/
#   identity  test-llama-archs -b CPU on every architecture: -ladder4's tree (BASE_TREE) writes the logits, the
#             branch's tree compares; with every switch at its default every architecture must match bit for bit
#   sass      the sm_61 SASS of every kernel in both trees' test-backend-ops (linked statically, so every ggml-cuda
#             kernel), instruction offsets stripped, one md5 per kernel (name and its count in the file): OK when every
#             kernel the two share is identical, REVIEW with the differing ones listed; new kernels are counted
#   units     q41g64: test-quantize-fns -v (every type's quantization, reference and dot-product errors, Q4_1_64's
#             among them)
#   cpubuild  toolsample-screen: llama-server as a static native CPU build into $ROOT/<name>/cpu/bin
#   screen    toolsample-screen: llama-server on MODEL (a file with an output_screen tensor) three times, the same BFCL
#             tool requests each time (bench/tool_sampling/grammar_sampling_check.py, one task per category, T 0 and
#             T 1, backend sampling): LLAMA_SCREEN_HEAD=1 LLAMA_SCREEN_HEAD_GRAMMAR=1 ("screen"), LLAMA_SCREEN_HEAD=0
#             ("full") and LLAMA_SCREEN_HEAD=1 alone ("default"); screen and default must give the same reasoning,
#             content and calls as full, and the server's log must show the screen on in tool requests with the
#             switch and never without it
# Usage: ROOT=<dir> REPO=<this repository> BASE_TREE=<-ladder4's unpacked rig tarball, extras in bin/> \
#   BASE_SRC=<a llama.cpp clone holding -ladder4> [BASE_CCACHE=<-ladder4's ccache>] [MODEL=<gguf with output_screen>] \
#   [TASKS=<BFCL tasks.jsonl of bench/bfcl_flint/prepare.py>] [BRANCHES='<name> ...'] [FORK=<fork https URL>] \
#   [JOBS=8] [THREADS=8] [PORT=18524] [LOCK=<flock file>] [DOCKER_ARGS=...] [STEPS='prep build ...'] \
#   [ONE=<work dir> ONE_BRANCH=<fork branch>] bash checks.sh
#   ONE       one work folder whose build of ONE_BRANCH holds every branch of BRANCHES (a merge, built by another script:
#             its rig tarball tree with the extras in bin/, its static native CPU build in cpu/bin): every name's steps
#             run on it (STEPS='identity sass units screen'), into $ONE/checks/c524/<step>/
# Output: $ROOT/<name>/llama.cpp-<date>-<sha>-<branch>{,.tar.gz}, $ROOT/<name>/checks/<step>/, $ROOT/<name>/logs/,
#   $ROOT/md5.txt, $ROOT/summary.tsv (branch, check, verdict, detail); done when $ROOT/DONE exists
set -uo pipefail
ROOT=$(realpath "${ROOT:?ROOT=<work root>}")
REPO=$(realpath "${REPO:?REPO=<this repository>}")
BASE_TREE=$(realpath "${BASE_TREE:?BASE_TREE=<-ladder4 rig tree>}")
BRANCHES=${BRANCHES:-toolsample-screen prefill-split f16router q41g64}
FORK=${FORK:-https://github.com/Cinderella-Man/llama.cpp}
JOBS=${JOBS:-8}
THREADS=${THREADS:-8}
PORT=${PORT:-18524}
STEPS=${STEPS:-prep build identity sass units cpubuild screen}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
D="--cpu-shares=256 ${DOCKER_ARGS:-}"
mkdir -p "$ROOT"
rm -f "$ROOT/DONE"
[ -f "$ROOT/summary.tsv" ] || printf 'branch\tcheck\tverdict\tdetail\n' > "$ROOT/summary.tsv"
locked() { if [ -n "${LOCK:-}" ]; then flock -w 21600 "$LOCK" "$@"; else "$@"; fi; }
step()   { [[ " $STEPS " == *" $1 "* ]]; }
# a container without a GPU whose CUDA backend loads against the stub driver (the rig build's binaries)
STUB='mkdir -p /tmp/stub && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /tmp/stub/libcuda.so.1 && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/tmp/stub'
echo "=== checks started $(date -Is): $BRANCHES; steps: $STEPS"

for n in $BRANCHES; do
    BR=${ONE_BRANCH:-pascal-p106-v3-b12076-$n}
    W=${ONE:-$ROOT/$n}
    S=$W/src
    CK=$W/checks${ONE:+/c524}
    mkdir -p "$W/logs" "$CK"
    verdict() { printf '%s\t%s\t%s\t%s\n' "$n" "$1" "$2" "$3" | tee -a "$ROOT/summary.tsv"; }
    tree() { local t; t=$(ls -t "$W"/llama.cpp-*-"$BR".tar.gz 2>/dev/null | head -n 1); [ -n "$t" ] && echo "$W/$(basename "$t" .tar.gz)"; }
    # shellcheck disable=SC2086
    box() { docker run --rm $D -v "$W:$W" -v "$BASE_TREE:$BASE_TREE:ro" "$@"; }
    echo "=== $n ($BR) $(date -Is)"

    if step prep; then
        if [ ! -d "$S/.git" ]; then
            git clone -q --no-checkout "${BASE_SRC:?BASE_SRC=<llama.cpp clone>}" "$S" &&
                git -C "$S" remote set-url origin https://github.com/ggml-org/llama.cpp &&
                git -C "$S" remote add fork "$FORK"
        fi
        git -C "$S" fetch -q fork "$BR"
        [ -d "$W/ccache" ] || { [ -n "${BASE_CCACHE:-}" ] && cp -a "$BASE_CCACHE" "$W/ccache"; }
        verdict prep "$(git -C "$S" rev-parse -q --verify "fork/$BR" > /dev/null && echo OK || echo FAIL)" \
            "fork/$BR at $(git -C "$S" rev-parse --short=9 "fork/$BR" 2>/dev/null)"
    fi

    if step build; then
        TESTS=1 JOBS=$JOBS WORK=$W DOCKER_ARGS="$D" locked nice -n 19 "$REPO/scripts/build_llamacpp_docker.sh" \
            "fork/$BR" > "$W/logs/rig.log" 2>&1
        rc=$?
        T=$(tree)
        if [ $rc = 0 ] && [ -n "$T" ]; then
            (cd "$W" && md5sum "$(basename "$T").tar.gz" && stat -c '%n %s bytes' "$(basename "$T").tar.gz") | tee -a "$ROOT/md5.txt"
            verdict build-rig OK "$(basename "$T"); $(grep -E '^(commit|avx|highest)' "$T/BUILDINFO.txt" | tr '\n' ' ')"
        else
            verdict build-rig FAIL "rc=$rc; $(grep -a -c -E ' error|error:' "$W/logs/rig.log") error lines in logs/rig.log"
        fi
        EXTRA=test-llama-archs
        [ "$n" = q41g64 ] && EXTRA="$EXTRA test-quantize-fns"
        # shellcheck disable=SC2086
        locked docker run --rm $D -v "$W:/work" -e JOBS="$JOBS" -e EXTRA="$EXTRA" -e HOST_UID="$(id -u)" \
            -e HOST_GID="$(id -g)" "$IMAGE" bash -euo pipefail -c '
            export DEBIAN_FRONTEND=noninteractive CCACHE_DIR=/work/ccache
            apt-get update -qq && apt-get install -y -qq git cmake build-essential ccache > /dev/null
            git config --global --add safe.directory /work/src
            cd /work/src
            cmake -B build -DLLAMA_BUILD_TESTS=ON > /dev/null
            nice -n 19 cmake --build build -j "$JOBS" --target $EXTRA > /work/logs/extra-build.log 2>&1 \
                || { tail -40 /work/logs/extra-build.log; exit 1; }
            rm -rf /work/extra && mkdir -p /work/extra
            for t in $EXTRA; do cp build/bin/$t /work/extra/; done
            chown -R "$HOST_UID:$HOST_GID" /work/extra /work/logs /work/ccache' > "$W/logs/extra.log" 2>&1
        rc=$?
        [ $rc = 0 ] && [ -n "$T" ] && cp "$W"/extra/* "$T/bin/"
        verdict build-extra "$([ $rc = 0 ] && echo OK || echo FAIL)" "rc=$rc; $EXTRA"
    fi
    T=$(tree)

    if step identity; then
        K=$CK/identity
        rm -rf "$K" && mkdir -p "$K"
        if [ -z "$T" ] || [ ! -x "$T/bin/test-llama-archs" ]; then
            verdict identity FAIL "no tree with test-llama-archs"
        else
            box -v "$K:/k" "$IMAGE" bash -c "$STUB"'
                run() { env LLAMA_ARCHS_LOGITS=/k/logits- nice -n 19 $2/test-llama-archs -b CPU -s 1 2>&1 \
                            | sed "s/\x1b\[[0-9;]*m//g" > /k/$1.log; echo "$1 rc=${PIPESTATUS[0]}"; }
                run base '"$BASE_TREE/bin"'
                ls /k/logits-*.f32 | wc -l > /k/n_base.txt
                run branch '"$T/bin"'
                rm -f /k/*.f32
                chown -R '"$(id -u):$(id -g)"' /k' > "$K/rc.txt" 2>&1
            # architectures whose models share a logits file compare against each other already in the base run
            grep 'logits against' "$K/base.log" | grep -v ', 0 of ' > "$K/base_self.txt"
            nb=$(cat "$K/n_base.txt" 2>/dev/null || echo 0)
            nc=$(grep -c 'logits against' "$K/branch.log")
            bad=$(grep 'logits against' "$K/branch.log" | grep -v ', 0 of ' | grep -vxFf "$K/base_self.txt" | cut -d: -f1 | tr '\n' ' ')
            sb=$(grep -m 1 '^Summary' "$K/base.log") sm=$(grep -m 1 '^Summary' "$K/branch.log")
            verdict identity "$([ "$nb" -gt 0 ] && [ "$nc" -gt 0 ] && [ -z "$bad" ] && [ "$sb" = "$sm" ] && echo OK || echo FAIL)" \
                "base wrote $nb logits files; $nc compared, differ: ${bad:-none}; base: $sb; branch: $sm; $(tr '\n' ' ' < "$K/rc.txt")"
        fi
    fi

    if step sass; then
        K=$CK/sass
        rm -rf "$K" && mkdir -p "$K"
        if [ -z "$T" ]; then
            verdict sass FAIL "no tree"
        else
            box -v "$K:/k" -e B="$BASE_TREE/bin/test-backend-ops" -e N="$T/bin/test-backend-ops" "$IMAGE" bash -uo pipefail -c '
                for tag in base branch; do
                    bin=$B; [ $tag = branch ] && bin=$N
                    mkdir -p /k/$tag
                    # one file per kernel: its instructions without the offsets, and its branch labels
                    nice -n 19 cuobjdump -sass -arch sm_61 "$bin" | awk -v out=/k/$tag -v idx=/k/$tag.idx "
                        /Function : / { if (f) close(f); k = \$3; c[k]++; nf++; f = sprintf(\"%s/%06d.sass\", out, nf)
                                        printf \"%06d\t%s#%d\n\", nf, k, c[k] > idx; next }
                        f && /^[ \t]+\/\*[0-9a-f]+\*\// { sub(/\/\*[0-9a-f]+\*\//, \"\"); print > f; next }
                        f && /^[ \t]+\/\* 0x[0-9a-f]+ \*\// { print > f; next }
                        f && /^[ \t]*\.L/ { print > f }"
                    (cd /k/$tag && md5sum *.sass | sed "s/\.sass\$//" | awk "{print \$2 \"\t\" \$1}" | sort) > /k/$tag.md5
                    sort /k/$tag.idx | join -t "	" - /k/$tag.md5 | cut -f2,3 | sort > /k/$tag.tsv
                    rm -rf /k/$tag /k/$tag.idx /k/$tag.md5
                done
                join -t "	" /k/base.tsv /k/branch.tsv | awk -F "\t" "\$2 == \$3" | wc -l > /k/n_same
                join -t "	" /k/base.tsv /k/branch.tsv | awk -F "\t" "\$2 != \$3 {print \$1}" > /k/differ.txt
                join -t "	" -v 2 /k/base.tsv /k/branch.tsv | cut -f1 > /k/new.txt
                join -t "	" -v 1 /k/base.tsv /k/branch.tsv | cut -f1 > /k/gone.txt
                for f in differ new gone; do sed "s/#[0-9]*\$//" /k/$f.txt | cu++filt | cut -c1-200 > /k/$f.demangled.txt; done
                chown -R '"$(id -u):$(id -g)"' /k' > "$K/console.txt" 2>&1
            nbk=$(wc -l < "$K/base.tsv" 2>/dev/null || echo 0)
            nd=$(wc -l < "$K/differ.txt" 2>/dev/null || echo 0)
            nn=$(wc -l < "$K/new.txt" 2>/dev/null || echo 0)
            ng=$(wc -l < "$K/gone.txt" 2>/dev/null || echo 0)
            v=OK
            { [ "$nd" -gt 0 ] || [ "$ng" -gt 0 ]; } && v=REVIEW
            [ "$nbk" -eq 0 ] && v=FAIL
            verdict sass "$v" "base $nbk kernels; $(cat "$K/n_same" 2>/dev/null) identical, $nd differ, $nn new, $ng gone (checks/sass/*.demangled.txt)"
        fi
    fi

    if step units && [ "$n" = q41g64 ]; then
        K=$CK/units
        rm -rf "$K" && mkdir -p "$K"
        box "$IMAGE" bash -c "$STUB; nice -n 19 $T/bin/test-quantize-fns -v" > "$K/test-quantize-fns.log" 2>&1
        rc=$?
        nq=$(grep -c '^q4_1_64 ' "$K/test-quantize-fns.log")
        verdict "unit test-quantize-fns" "$([ $rc = 0 ] && [ "$nq" -gt 0 ] && echo OK || echo FAIL)" \
            "rc=$rc; q4_1_64: $(grep '^q4_1_64 ' "$K/test-quantize-fns.log" | sed 's/  */ /g' | tr '\n' ';')"
    fi

    if step cpubuild && [ "$n" = toolsample-screen ]; then
        mkdir -p "$W/cpu"
        # shellcheck disable=SC2086
        locked docker run --rm $D -v "$W:/work" -e JOBS="$JOBS" -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
            "$IMAGE" bash -euo pipefail -c '
            export DEBIAN_FRONTEND=noninteractive CCACHE_DIR=/work/ccache
            apt-get update -qq && apt-get install -y -qq git cmake build-essential ccache > /dev/null
            git config --global --add safe.directory /work/src
            cmake -S /work/src -B /work/cpu/build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF \
                -DGGML_CUDA=OFF -DGGML_NATIVE=ON -DCMAKE_C_COMPILER_LAUNCHER=ccache \
                -DCMAKE_CXX_COMPILER_LAUNCHER=ccache > /work/logs/cpu-cmake.log 2>&1
            rc=0
            nice -n 19 cmake --build /work/cpu/build -j "$JOBS" --target llama-server > /work/logs/cpu-build.log 2>&1 || rc=$?
            if [ $rc -eq 0 ]; then rm -rf /work/cpu/bin && mkdir -p /work/cpu/bin && cp /work/cpu/build/bin/llama-server /work/cpu/bin/; fi
            chown -R "$HOST_UID:$HOST_GID" /work/cpu /work/logs /work/ccache
            exit $rc'
        rc=$?
        verdict cpubuild "$([ $rc = 0 ] && echo OK || echo FAIL)" "rc=$rc; llama-server, commit $(git -C "$S" rev-parse --short=9 HEAD)"
    fi

    if step screen && [ "$n" = toolsample-screen ]; then
        K=$CK/screen
        rm -rf "$K" && mkdir -p "$K"
        : "${MODEL:?screen needs MODEL}" "${TASKS:?screen needs TASKS}"
        name=pq-c524-screen
        r=""
        # <variant> <docker -e arguments>: one server, the requests, its log
        arm() {
            local v=$1; shift
            # shellcheck disable=SC2086
            docker run -d --name $name --network host $D -v "$W:$W" -v "$(dirname "$MODEL"):$(dirname "$MODEL"):ro" "$@" \
                "$IMAGE" nice -n 19 "$W/cpu/bin/llama-server" -m "$MODEL" -ngl 0 -t "$THREADS" -c 16384 -np 1 -bs \
                --cache-ram 0 --ctx-checkpoints 0 --host 127.0.0.1 --port "$PORT" > /dev/null
            local i
            for i in $(seq 1 900); do curl -sf "http://127.0.0.1:$PORT/health" > /dev/null && break; sleep 2; done
            nice -n 19 python3 -I "$REPO/bench/tool_sampling/grammar_sampling_check.py" run --url "http://127.0.0.1:$PORT" \
                --tasks "$TASKS" --out "$K/$v.jsonl" --per-category 1 --temps 0,1 --seeds 1 --max-tokens 1024 \
                --variants "$v:backend_sampling=true" > "$K/$v-client.log" 2>&1
            docker logs $name > "$K/$v-server.log" 2>&1
            docker rm -f $name > /dev/null
        }
        if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; then
            verdict screen FAIL "port $PORT is taken"
        else
            arm screen  -e LLAMA_SCREEN_HEAD=1 -e LLAMA_SCREEN_HEAD_GRAMMAR=1
            arm full    -e LLAMA_SCREEN_HEAD=0
            arm default -e LLAMA_SCREEN_HEAD=1
            cat "$K"/screen.jsonl "$K"/full.jsonl "$K"/default.jsonl > "$K/all.jsonl"
            G="$REPO/bench/tool_sampling/grammar_sampling_check.py"
            for a in screen default; do
                python3 -I "$G" compare "$K/all.jsonl" --a $a --b full > "$K/compare-$a.txt" 2>&1 || r="$r $a != full;"
            done
            nerr=$(grep -c '"error"' "$K/all.jsonl")
            nreq=$(wc -l < "$K/all.jsonl")
            on_s=$(grep -c 'screened output head on' "$K/screen-server.log")
            on_d=$(grep -c 'screened output head on' "$K/default-server.log")
            [ "$nerr" -gt 0 ] && r="$r $nerr requests failed;"
            [ "$on_s" -gt 0 ] || r="$r the screen never turned on with the switch;"
            [ "$on_d" -gt 0 ] && r="$r the screen turned on without the switch;"
            verdict screen "$([ -z "$r" ] && echo OK || echo FAIL)" \
                "${r:-screen = full and default = full}: $(tr '\n' ' ' < "$K/compare-screen.txt" | sed 's/  */ /g') | $(tr '\n' ' ' < "$K/compare-default.txt" | sed 's/  */ /g'); $nreq requests; screen on $on_s times with the switch, $on_d without"
        fi
    fi
done

date -Is > "$ROOT/DONE"
echo "=== done $(date -Is)"
cat "$ROOT/summary.tsv"
