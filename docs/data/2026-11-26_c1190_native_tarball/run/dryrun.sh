#!/usr/bin/env bash
# C1190: the Docker path of scripts/build_llamacpp_docker.sh before and after NATIVE=1 was added, on a machine without
# Docker: a stand-in `docker` first on PATH writes the arguments it is given and does nothing else, so neither script
# builds anything. Each case runs both scripts with the same settings, then checks that every argument but the last
# (the build steps that run in the container) is byte for byte the same, and writes the last one's diff.
#
#   OLD=<script before> NEW=<script after> OUT=<empty folder> bash dryrun.sh
set -euo pipefail
: "${OLD:?}" "${NEW:?}" "${OUT:?}"
export LC_ALL=C
mkdir -p "$OUT/stub" "$OUT/work"
cat > "$OUT/stub/docker" << 'STUB'
#!/usr/bin/env bash
# every argument, each ended by a NUL byte
printf '%s\0' "$@" > "$ARGV_OUT"
STUB
chmod +x "$OUT/stub/docker"

# case name, then the settings and arguments
run_case() {
    local name=$1 s
    shift
    for s in old new; do
        local script=$OLD
        [ "$s" = new ] && script=$NEW
        env PATH="$OUT/stub:$PATH" ARGV_OUT="$OUT/$name.$s.argv" WORK="$OUT/work" "$@" bash "$script" ${ARGS:-} \
            > /dev/null
    done
    # all but the last argument, then the last one
    for s in old new; do
        python3 -I -c 'import sys; a = open(sys.argv[1], "rb").read().split(b"\0")[:-1]; open(sys.argv[2], "wb").write(b"\0".join(a[:-1])); open(sys.argv[3], "wb").write(a[-1])' \
            "$OUT/$name.$s.argv" "$OUT/$name.$s.head" "$OUT/$name.$s.body"
    done
    if cmp -s "$OUT/$name.old.head" "$OUT/$name.new.head"; then head=same; else head=DIFFERENT; fi
    diff "$OUT/$name.old.body" "$OUT/$name.new.body" > "$OUT/$name.body.diff" || true
    printf '%s\t%s\t%s\t%s\n' "$name" "$(tr '\0' '\n' < "$OUT/$name.new.head" | grep -c .)" "$head" \
        "$(grep -c '^[<>]' "$OUT/$name.body.diff" || true)"
}

printf 'case\targuments_before_the_steps\tthose_arguments\tchanged_step_lines\n'
ARGS=fork/pascal-p106-v5-cand-b12723 run_case c1118-settings TESTS=1 HF_UI_VERSION=b12722 JOBS=8
ARGS= run_case defaults
ARGS="miner-llama-v4.0 --cuda-graphs-pascal" run_case options TESTS=1 CHECK_TOOLS=0 RPC=1 SCHED_COPIES=2 \
    PATCHES=patches/x.diff EXTRA_TARGETS=test-foo DOCKER_ARGS="--label pascal-quants=1 -e HF_UI_VERSION=b12000" \
    HF_UI_VERSION=b12674 IMAGE=nvidia/cuda:12.9.2-devel-ubuntu24.04
