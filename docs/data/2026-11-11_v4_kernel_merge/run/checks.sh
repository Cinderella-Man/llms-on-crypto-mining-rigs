#!/usr/bin/env bash
# The v4 kernel merge (2026-11-11, C796): the three read-once kernel branches that finished after the late merge (C727
# GGML_CUDA_MMVQ_PASCAL_Q6K_MT, C728 _DMT_KT, C729 _RO8, each on the day merge 18d0fa309) merged into pascal-p106-v4
# (the late merge, 08634ce00), built once and checked on a build machine, CPU only. Each branch's own static checks run on
# the merged build, and their tables are compared with the ones each branch published; the rest is the late merge's
# method (docs/data/2026-11-11_v4_late_merge/run/checks.sh). Every build and test runs in a container without GPUs, waits
# for LOCK (flock) and runs at a low CPU share (--cpu-shares=256, nice 19), so one runs at a time on a shared machine.
# Steps (STEPS, in this order by default):
#   rig     the rig tarball of BRANCH (scripts/build_llamacpp_docker.sh, TESTS=1: sm_61, no AVX, test-backend-ops in bin/)
#           in $M/main/rig, whose src/ and ccache/ it uses; its CUDA objects stay in src/build for the steps below
#   names   C770's switch list in the merged build (src/build/common/switch-names.cpp): its count, and the three branches'
#           eight names in it and in bin/llama-server
#   sass    cuobjdump -sass and -ptx of bin/llama-server of the merged tarball, of BASE_TREE (the base's rig tarball) and
#           of the three branches' own tarballs (BR_C727, BR_C728, BR_C729), anonymous-namespace hashes masked, cut into
#           functions and attributed (the day merge's sass_attrib.py attrib): every function the base's or the code of
#           the one branch that changed it, none missing, lost or merged; and the merged objects of the three branches'
#           files dumped as each branch's checks dump its own
#   sasscmp the comparisons, from the sass step's dumps: the functions attrib lists in the PTX, if any, compared in text
#           with the per-file suffix of static names masked as well (a name, not code); then each branch's own tables on
#           the merged objects, against the tables it published: C727's run/sass_check.sh (Q6_K's kernels; equal), C728's
#           run/sass_check.sh (every row it published; the rows more are C727's Q6_K and C729's 5-8 tokens) and C729's
#           run/compare_sass.py's kernel table (every row it published: code, registers, stack, local memory, spills,
#           counts; the rows more are C727's Q6_K)
#   cpu     a static native CPU build of the merged tip (GGML_CUDA=OFF, every target) in $M/cpu/tree; its test-llama-archs
#           and llama-server in $M/cpu/bin
#   off     test-llama-archs -b CPU -s 1: the base's (BASE_CPU) writes every architecture's logits, the merged one
#           compares with no switch, each branch's switch and its fault, and the three on together (CUDA switches: the
#           CPU results must be the base's)
#   pick    run/kern_pick.cu built against the base's, the three branches' and the merged mmvq-pascal-dmt.cu (nvcc, no GPU)
#           and run under each arm of PICK_ARMS on all five: every arm with only the base's switches launches the base's
#           kernels in the merged tree; under every arm each product launches in the merged tree what the branch that
#           owns it launches in its own tree (Q6_K at 2-4 tokens C727's, IQ4_XS and Q4_K at 2-4 C728's, at 5-8 C729's),
#           and Q6_K at 5-8 tokens nothing (C795)
#   emul    C727's run/q6k_emul.c: the one-token and the read-once kernels' Q6_K arithmetic emulated from the same bytes,
#           and the exact conversions
#   bitid   bench/mmvq_pascal/verify/multitoken_bitid linked against the merged build tree: modes q6k (C727), dmtkt
#           (C728), dmt8 and q5k8 (C729), the takes functions the build's own
#   record  the replay set v4kern (scripts/record_checks.sh SUITE=v4kern: the suites q6kmt, dmtkt and ro8, and the three
#           switches on together with each fault) with the merged tarball's test-backend-ops, into $M/sets
#   combo   test-llama-archs with every switch of the four merges on together against the base's logits; the late merge's
#           run/server_combo.sh (arms base bbase late latn) with the three kernel switches added to its late arms,
#           against the base's llama-server
#   combocheck  server_combo.sh's verdict read for a base that has C770: its only failing lines may be the two base
#           arms' switch line (the script wants none from a base before C770)
#   c565    2026-11-05_c565_ladder4fix/run/cpu_tests.sh, static and then shared: ctest, the network test, archs, the hand
#           tests; each build tree removed after its run
#   trim    the CPU build tree removed (the binaries stay)
#   final   the rig tarball of FINAL_BRANCH (once it is pushed at the checked commit) in the same clone and ccache, and
#           its bin/ compared with the checked tree's
# Before the c565 and final builds the chain waits (up to 2 hours) until the disk has ROOM_GB free (52 by default).
# Env: M (the work root), REPO (this repository as run), LOCK, BASE_COMMIT (08634ce00 in full), BASE_TREE (the base's
#   unpacked rig tarball), BASE_CPU (bin/ of the base's static CPU build: test-llama-archs, llama-server), BR_C727,
#   BR_C728, BR_C729 (the branches' unpacked rig tarballs), SEED_SRC (a llama.cpp clone with the fork as remote "fork",
#   https), SEED_CCACHE and SEED_CPU_CCACHE (optional), BRANCH (default pascal-p106-v4-b12246-kern), FINAL_BRANCH (default
#   pascal-p106-v4), JOBS (8), THREADS (8), DOCKER_ARGS, IMAGE, CPU_IMAGE, STEPS
# Output: $M/summary.tsv (row, check, verdict, detail), $M/sass, $M/pick, $M/sets/v4kern.tar, $M/logs; done when $M/DONE
#   exists
set -uo pipefail
M=$(realpath -m "${M:?M=<work root>}")
REPO=$(realpath "${REPO:?REPO=<this repository>}")
: "${LOCK:?LOCK=<flock file>}"
BRANCH=${BRANCH:-pascal-p106-v4-b12246-kern}
FINAL_BRANCH=${FINAL_BRANCH:-pascal-p106-v4}
BASE_COMMIT=${BASE_COMMIT:-08634ce0012d3b55f4ab2816ef077eac44c198a0}
BRANCHES="c727:pascal-p106-v4-b12246-c727 c728:pascal-p106-v4-b12246-c728 c729:pascal-p106-v4-b12246-c729"
JOBS=${JOBS:-8}
THREADS=${THREADS:-8}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
CPU_IMAGE=${CPU_IMAGE:-pq-c565-cpu:24.04}
SRV_IMAGE=${SRV_IMAGE:-pq-c735-srv:24.04}
STEPS=${STEPS:-rig names sass sasscmp cpu off pick emul bitid record combo combocheck c565 trim}
ROOM_GB=${ROOM_GB:-52}
DA="--label pascal-quants=1 --cpu-shares=256 ${DOCKER_ARGS:-}"
D=$REPO/docs/data
RUN=$D/2026-11-11_v4_kernel_merge/run
ATTRIB=$D/2026-11-11_v4_day_merge/run/sass_attrib.py
SRC=$M/main/rig/src
O=$SRC/build/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir
MASK='s/_GLOBAL__N__[0-9a-f]{8}_/_GLOBAL__N__X_/g;s/_cu_[0-9a-f]{8}/_cu_X/g;s/_INTERNAL_[0-9a-f]{8}/_INTERNAL_X/g'
mkdir -p "$M/logs"
rm -f "$M/DONE"
[ -f "$M/summary.tsv" ] || printf 'row\tcheck\tverdict\tdetail\n' > "$M/summary.tsv"
step()   { [[ " $STEPS " == *" $1 "* ]]; }
log()    { echo "$(date -Is) $*" | tee -a "$M/logs/kern.log"; }
locked() { flock -w 43200 "$LOCK" "$@"; }
add()    { printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" | tee -a "$M/summary.tsv"; }
tree()   { ls -td "$M"/main/rig/llama.cpp-*-"$1" 2> /dev/null | grep -v '\.tar\.gz$' | head -n 1; }
free_gb() { df -BG --output=avail "$M" | tail -n 1 | tr -dc '0-9'; }
# shellcheck disable=SC2086
box()    { locked docker run --rm $DA --network none -v "$M:$M" "$@"; }
room() {
    local i
    for i in $(seq 1 24); do
        [ "$(free_gb)" -ge "$ROOM_GB" ] && return 0
        [ "$i" = 1 ] && log "$1 waits for $ROOM_GB GB free ($(free_gb) GB)"
        sleep 300
    done
    add "$1" disk FAIL "under $ROOM_GB GB free for 2 hours ($(free_gb) GB): stopped"
    exit 1
}

if [ ! -d "$SRC/.git" ]; then
    mkdir -p "$M/main/rig"
    git clone -q --local --no-checkout "${SEED_SRC:?SEED_SRC=<a llama.cpp clone with the fork as remote fork>}" "$SRC" &&
        git -C "$SRC" remote add fork "$(git -C "$SEED_SRC" remote get-url fork)" &&
        git -C "$SRC" remote set-url origin "$(git -C "$SEED_SRC" remote get-url origin)"
fi
for b in "$BRANCH" ${BRANCHES//c72?:/}; do
    git -C "$SRC" fetch -q fork "$b:refs/remotes/fork/$b" || { echo "no fork/$b"; exit 1; }
done
TIP=$(git -C "$SRC" rev-parse "fork/$BRANCH")
git -C "$SRC" merge-base --is-ancestor "$BASE_COMMIT" "$TIP" || { echo "$BASE_COMMIT is not in $BRANCH"; exit 1; }
if ! docker image inspect "$CPU_IMAGE" > /dev/null 2>&1; then
    printf 'FROM ubuntu:24.04\nRUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git cmake build-essential ccache python3 python3-jinja2 python3-pip > /dev/null && rm -rf /var/lib/apt/lists/*\n' \
        | docker build -q --label pascal-quants=1 -t "$CPU_IMAGE" - > "$M/logs/image.log" 2>&1 || { echo "image build failed"; exit 1; }
fi
log "kernel merge checks: $BRANCH at ${TIP:0:9}, base ${BASE_COMMIT:0:9}, steps: $STEPS"

if step rig; then
    [ -d "$M/main/rig/ccache" ] || [ -z "${SEED_CCACHE:-}" ] || cp -al "$SEED_CCACHE" "$M/main/rig/ccache"
    locked env WORK="$M/main/rig" TESTS=1 JOBS="$JOBS" DOCKER_ARGS="$DA" nice -n 19 bash "$REPO/scripts/build_llamacpp_docker.sh" \
        "fork/$BRANCH" > "$M/logs/rig.log" 2>&1
    rc=$?
    T=$(tree "$BRANCH")
    add rig build "$([ $rc = 0 ] && [ -n "$T" ] && [ -d "$O" ] && echo OK || echo FAIL)" \
        "rc=$rc; $(basename "$T").tar.gz md5 $(md5sum < "$T.tar.gz" | cut -d' ' -f1), $(stat -c %s "$T.tar.gz") B; $(grep -E '^(commit|avx|highest)' "$T/BUILDINFO.txt" | tr '\n' ';'); test-backend-ops md5 $(md5sum < "$T/bin/test-backend-ops" | cut -d' ' -f1); $(grep -ac 'warning' "$M/logs/rig.log") warning lines"
    [ $rc = 0 ] || { date -Is > "$M/DONE"; exit 1; }
fi
T=$(tree "$BRANCH")

if step names; then
    NEW="GGML_CUDA_MMVQ_PASCAL_Q6K_MT GGML_CUDA_MMVQ_PASCAL_Q6K_MT_FAULT GGML_CUDA_MMVQ_PASCAL_DMT_KT \
GGML_CUDA_MMVQ_PASCAL_DMT_KT_M GGML_CUDA_MMVQ_PASCAL_DMT_KT_NW GGML_CUDA_MMVQ_PASCAL_DMT_KT_FAULT \
GGML_CUDA_MMVQ_PASCAL_RO8 GGML_CUDA_MMVQ_PASCAL_RO8_FAULT"
    L=$SRC/build/common/switch-names.cpp
    n=$(grep -c '^    "' "$L" 2> /dev/null)
    miss= mbin=
    for s in $NEW; do
        grep -q "^    \"$s\",\$" "$L" 2> /dev/null || miss="$miss$s "
        grep -aq "$s" "$T/bin/llama-server" || mbin="$mbin$s "
    done
    add names "switch list" "$([ -n "$n" ] && [ -z "$miss" ] && [ -z "$mbin" ] && echo OK || echo FAIL)" \
        "$n names in the merged build's list (the late merge's: 850); the three branches' eight names missing from it: ${miss:-none}, from bin/llama-server: ${mbin:-none}"
fi

if step sass; then
    S=$M/sass
    rm -rf "$S" && mkdir -p "$S/obj" "$S/c729/kern/obj" "$S/c729/out"
    # <label> <llama-server>: its SASS and PTX cut into functions
    dumpbin() {
        local d w
        d=$(dirname "$2")
        for w in sass ptx; do
            # shellcheck disable=SC2086
            locked docker run --rm $DA --network none -v "$d:$d:ro" "$IMAGE" nice -n 19 cuobjdump -$w "$2" \
                | sed -E "$MASK" | nice -n 19 python3 "$ATTRIB" dump $w > "$S/$1.$w.tsv"
        done
    }
    dumpbin kern "$T/bin/llama-server"
    dumpbin base "${BASE_TREE:?BASE_TREE=<the base rig tree>}/bin/llama-server"
    dumpbin c727 "${BR_C727:?BR_C727=<C727 rig tree>}/bin/llama-server"
    dumpbin c728 "${BR_C728:?BR_C728=<C728 rig tree>}/bin/llama-server"
    dumpbin c729 "${BR_C729:?BR_C729=<C729 rig tree>}/bin/llama-server"
    for w in sass ptx; do
        python3 "$ATTRIB" attrib "$S/kern.$w.tsv" "$S/base.$w.tsv" c727="$S/c727.$w.tsv" c728="$S/c728.$w.tsv" \
            c729="$S/c729.$w.tsv" > "$S/attrib_$w.txt" 2>&1
        add sass "llama-server $w by build" "$(tail -n 1 "$S/attrib_$w.txt" | sed 's/^verdict: //')" \
            "$(head -n 1 "$S/attrib_$w.txt"); $(sed -n '2,$p' "$S/attrib_$w.txt" | grep -v '^verdict' | sed 's/^ *//' | head -n 20 | tr '\n' ';')"
    done
    # the merged objects, dumped as each branch's checks dump its own
    # shellcheck disable=SC2086
    locked docker run --rm $DA --network none -v "$SRC:$SRC:ro" -v "$T:$T:ro" -v "$S:$S" "$IMAGE" bash -c '
        set -u
        cuobjdump --version | tail -n 1 > '"$S"'/cuobjdump.txt
        cuobjdump -res-usage '"$O"'/mmvq-pascal-dmt.cu.o > '"$S"'/obj/dmt_res.txt 2>&1
        cuobjdump -sass '"$O"'/mmvq-pascal-dmt.cu.o > '"$S"'/obj/dmt.sass
        cuobjdump -sass '"$O"'/mmvq-pascal.cu.o | gzip > '"$S"'/obj/mmvqp.sass.gz
        for o in mmvq-pascal-dmt.cu mmvq-pascal-q5k.cu; do
            nice -n 19 cuobjdump -sass '"$O"'/$o.o > '"$S"'/c729/kern/obj/$o.sass
            nice -n 19 cuobjdump -res-usage '"$O"'/$o.o > '"$S"'/c729/kern/obj/$o.res
            nice -n 19 cuobjdump -ptx '"$O"'/$o.o > '"$S"'/c729/kern/obj/$o.ptx
        done
        nice -n 19 cuobjdump -sass '"$T"'/bin/llama-server > '"$S"'/c729/kern/llama-server.sass
        md5sum '"$O"'/mmvq-pascal-dmt.cu.o '"$O"'/mmvq-pascal-q5k.cu.o '"$O"'/mmvq-pascal.cu.o > '"$S"'/obj/md5.txt
        chown -R '"$(id -u):$(id -g)"' '"$S"'' > "$M/logs/sass.log" 2>&1
    echo $? > "$S/dump_rc.txt"
    log "sass done"
fi

# the comparisons, from the sass step's dumps
if step sasscmp; then
    S=$M/sass
    rc=$(cat "$S/dump_rc.txt" 2> /dev/null || echo 1)
    for f in "$S"/obj/dmt.sass.gz "$S"/c729/kern/llama-server.sass.gz "$S"/c729/kern/obj/*.gz; do
        [ -f "$f" ] && gunzip -f "$f"
    done
    # the functions attrib lists in the PTX, if any: each one's text in the build that changed it and in the base, the
    # per-file suffix of static names masked as well (_cu_<hash>_<number>); the same text means only a name differs
    L=$S/ptx_listed
    rm -rf "$L" && mkdir -p "$L"
    sed -n 's/^  [A-Z]*: \([^:]*\): changed by \([a-z0-9]*\);.*/\1 \2/p' "$S/attrib_ptx.txt" > "$L/list.txt"
    nl=$(grep -c ' ' "$L/list.txt")
    nall=$(sed -n 's/.*; listed \([0-9]*\)$/\1/p' "$S/attrib_ptx.txt")
    if [ "$nl" -gt 0 ]; then
        cut -d' ' -f1 "$L/list.txt" > "$L/names.txt"
        for b in base $(cut -d' ' -f2 "$L/list.txt" | sort -u); do
            case $b in base) bin=$BASE_TREE ;; c727) bin=$BR_C727 ;; c728) bin=$BR_C728 ;; c729) bin=$BR_C729 ;; *) continue ;; esac
            # shellcheck disable=SC2086
            locked docker run --rm $DA --network none -v "$bin:$bin:ro" "$IMAGE" nice -n 19 cuobjdump -ptx "$bin/bin/llama-server" \
                | awk -v names="$L/names.txt" 'BEGIN { while ((getline n < names) > 0) want[n] = 1 }
                    /\.(entry|func)/ { on = 0; for (n in want) if (index($0, n)) { on = 1; cur = n } }
                    on { print cur "\t" $0 }' | sed -E "$MASK;s/_cu_X_[0-9]+/_cu_X_N/g" > "$L/$b.txt"
        done
        nsame=0 nchk=0
        while read -r f b; do
            nchk=$((nchk + 1))
            if [ -n "$(grep -F "$f	" "$L/base.txt")" ] && diff -q <(grep -F "$f	" "$L/base.txt" | cut -f2-) <(grep -F "$f	" "$L/$b.txt" | cut -f2-) > /dev/null; then
                nsame=$((nsame + 1))
            fi
        done < "$L/list.txt"
        add sass "ptx listed: names only" "$([ "$nsame" = "$nchk" ] && [ "$nchk" = "$nall" ] && echo OK || echo CHECK)" \
            "attrib listed $nall PTX functions, $nchk of them changed by one branch; with static names' per-file suffixes masked, $nsame the base's text in that branch's build (the merged build has the base's code: attrib); $(cut -d' ' -f2 "$L/list.txt" | sort | uniq -c | tr -s ' ' | tr '\n' ';')"
    fi
    # C727: its table of the Q6_K instantiations, equal to the one it published
    SASS=$S/obj nice -n 19 bash "$D/2026-11-11_c727_q6k_readonce/run/sass_check.sh" > "$S/c727_counts.tsv" 2> "$S/c727_counts.err"
    P7=$D/2026-11-11_c727_q6k_readonce/sass_counts.tsv
    d7=$(diff "$P7" "$S/c727_counts.tsv" | grep -c '^[<>]')
    add c727 "sass_check.sh on the merged object" "$([ "$rc" = 0 ] && [ "$d7" = 0 ] && [ "$(wc -l < "$S/c727_counts.tsv")" -gt 1 ] && echo SAME || echo DIFFERS)" \
        "$(($(wc -l < "$S/c727_counts.tsv") - 1)) Q6_K instantiations; lines that differ from C727's published table: $d7; with stack or local memory: $(awk -F'\t' 'NR > 1 && ($6 != 0 || $7 != 0)' "$S/c727_counts.tsv" | wc -l)"
    # C728: every row it published, the same; the rows more are other branches' instantiations
    SASS=$S/obj nice -n 19 bash "$D/2026-11-11_c728_readonce_k6144/run/sass_check.sh" > "$S/c728_counts.tsv" 2> "$S/c728_counts.err"
    P8=$D/2026-11-11_c728_readonce_k6144/sass_counts.tsv
    lost8=$(grep -vxFf "$S/c728_counts.tsv" "$P8" | wc -l)
    more8=$(grep -vxFf "$P8" "$S/c728_counts.tsv" | awk -F'\t' '!($2 == "type14" || $3 >= 5)' | wc -l)
    add c728 "sass_check.sh on the merged objects" "$([ "$rc" = 0 ] && [ "$lost8" = 0 ] && [ "$more8" = 0 ] && echo SAME || echo DIFFERS)" \
        "$(($(wc -l < "$P8") - 1)) rows published, not in the merged table as they are: $lost8; $(grep -vxFf "$P8" "$S/c728_counts.tsv" | wc -l) rows more ($(grep -vxFf "$P8" "$S/c728_counts.tsv" | awk -F'\t' '$2 == "type14"' | wc -l) Q6_K, $(grep -vxFf "$P8" "$S/c728_counts.tsv" | awk -F'\t' '$3 >= 5' | wc -l) of 5-8 tokens), others: $more8; kt with stack or local memory: $(awk -F'\t' 'NR > 1 && $1 == "mul_mat_vec_q_pascal_dmt_kt" && ($7 != 0 || $8 != 0)' "$S/c728_counts.tsv" | wc -l)"
    # C729: its kernel table on the merged objects (both sides the merged build), every row it published the same in the
    # tip's columns (code, registers, stack, local memory, spills, counts, the last two FFMA). Its script names the types
    # it knows (q4_K, q5_K, iq4_xs) and gives Q6_K by its number, 14
    python3 -I "$D/2026-11-11_c729_readonce_5to8/run/compare_sass.py" "$S/c729/kern" "$S/c729/kern" "$S/c729/out" > "$S/c729/compare.txt" 2>&1
    P9=$D/2026-11-11_c729_readonce_5to8/sass/sass_kernels.tsv
    # kernel, type, NT, R, PF, code, registers, stack, local, LDL, STL, IDP, LDS, LDG, FFMA, FADD, I2F, the last two FFMA
    cut9() { awk -F'\t' 'BEGIN { OFS = "\t" } NR > 1 { print $1, $2, $3, $4, $5, $7, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21 }' "$1" | sort; }
    cut9 "$P9" > "$S/c729/published.tsv"
    cut9 "$S/c729/out/sass_kernels.tsv" > "$S/c729/merged.tsv"
    lost9=$(comm -23 "$S/c729/published.tsv" "$S/c729/merged.tsv" | wc -l)
    more9=$(comm -13 "$S/c729/published.tsv" "$S/c729/merged.tsv" | awk -F'\t' '$2 != "14"' | wc -l)
    add c729 "compare_sass.py's kernel table on the merged objects" "$([ "$lost9" = 0 ] && [ "$more9" = 0 ] && [ -s "$S/c729/merged.tsv" ] && echo SAME || echo DIFFERS)" \
        "$(wc -l < "$S/c729/published.tsv") rows published, not in the merged table as they are: $lost9; rows more: $(comm -13 "$S/c729/published.tsv" "$S/c729/merged.tsv" | wc -l) (Q6_K, type 14: $(comm -13 "$S/c729/published.tsv" "$S/c729/merged.tsv" | awk -F'\t' '$2 == "14"' | wc -l)), others: $more9; of the $(wc -l < "$S/c729/merged.tsv") instantiations, with stack, local memory or spill instructions: $(awk -F'\t' '$8 != 0 || $9 != 0 || $10 != 0 || $11 != 0' "$S/c729/merged.tsv" | wc -l)"
fi

C=$M/cpu
if step cpu; then
    rm -rf "$C/tree" "$C/bin" && mkdir -p "$C/tree/src" "$C/bin"
    [ -d "$C/ccache" ] || { if [ -n "${SEED_CPU_CCACHE:-}" ]; then cp -al "$SEED_CPU_CCACHE" "$C/ccache"; else mkdir -p "$C/ccache"; fi; }
    git -C "$SRC" archive "$TIP" | tar -x -C "$C/tree/src"
    echo "commit $TIP $(git -C "$SRC" log -1 --format=%s "$TIP")" > "$C/tree.txt"
    box -e CCACHE_DIR="$C/ccache" -e CCACHE_BASEDIR="$C/tree" "$CPU_IMAGE" bash -c '
        cmake -S '"$C/tree/src"' -B '"$C/tree/build"' -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_NATIVE=ON \
            -DLLAMA_BUILD_TESTS=ON -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DCMAKE_C_COMPILER_LAUNCHER=ccache \
            -DCMAKE_CXX_COMPILER_LAUNCHER=ccache > '"$C/cmake.log"' 2>&1 &&
        nice -n 19 cmake --build '"$C/tree/build"' -j '"$JOBS"' -- -k > '"$C/build.log"' 2>&1'
    rc=$?
    cp "$C/tree/build/bin/test-llama-archs" "$C/tree/build/bin/llama-server" "$C/bin/" 2> /dev/null
    add cpu build "$([ $rc = 0 ] && [ -x "$C/bin/test-llama-archs" ] && [ -x "$C/bin/llama-server" ] && echo OK || echo FAIL)" \
        "rc=$rc; $(cat "$C/tree.txt"); static, every target; $(grep -ac 'warning:' "$C/build.log") warning lines, $(grep -acE 'error:|undefined reference' "$C/build.log") error lines"
fi

if step off; then
    A=$M/off
    BC=${BASE_CPU:?BASE_CPU=<bin of the base static CPU build>}
    rm -rf "$A" && mkdir -p "$A/base"
    P=GGML_CUDA_MMVQ_PASCAL
    # shellcheck disable=SC2086
    locked docker run --rm $DA --network none -v "$M:$M" -v "$BC:$BC:ro" "$CPU_IMAGE" bash -c \
        'env LLAMA_ARCHS_LOGITS='"$A"'/base/logits- nice -n 19 '"$BC"'/test-llama-archs -b CPU -s 1 2>&1 | sed "s/\x1b\[[0-9;]*m//g" > '"$A"'/base.log; exit ${PIPESTATUS[0]}'
    rc=$?
    nb=$(ls "$A"/base/logits-*.f32 2> /dev/null | wc -l)
    add off "base writes" "$([ $rc = 0 ] && [ "$nb" -gt 0 ] && echo OK || echo FAIL)" "rc=$rc; ${BC%/bin}: $nb logits files; $(grep -a -m 1 '^Summary' "$A/base.log")"
    for arm in "tip:" "q6kmt:${P}_Q6K_MT=1" "q6kmt-fault:${P}_Q6K_MT=1 ${P}_Q6K_MT_FAULT=1" "dmtkt:${P}_DMT=1 ${P}_DMT_KT=1" \
               "dmtkt-fault:${P}_DMT=1 ${P}_DMT_KT=1 ${P}_DMT_KT_FAULT=1" "ro8:${P}_RO8=1" "ro8-fault:${P}_RO8=1 ${P}_RO8_FAULT=1" \
               "kern:${P}_DMT=1 ${P}_Q5K_MT=1 ${P}_Q6K_MT=1 ${P}_DMT_KT=1 ${P}_RO8=1"; do
        tag=${arm%%:*} envs=${arm#*:}
        box "$CPU_IMAGE" bash -c \
            'env LLAMA_ARCHS_LOGITS='"$A"'/base/logits- '"$envs"' nice -n 19 '"$C"'/bin/test-llama-archs -b CPU -s 1 2>&1 | sed "s/\x1b\[[0-9;]*m//g" > '"$A/$tag.log"'; exit ${PIPESTATUS[0]}'
        rc=$?
        n=$(grep -ac 'logits against' "$A/$tag.log")
        grep -a 'logits against' "$A/$tag.log" | grep -v ', 0 of ' > "$A/$tag.differ.txt"
        bad=$(cut -d: -f1 "$A/$tag.differ.txt" | tr '\n' ' ')
        wrote=$(grep -ac 'logits_dump_or_compare: wrote' "$A/$tag.log")
        add off "$tag" "$([ "$rc" = 0 ] && [ "$n" -gt 0 ] && [ -z "$bad" ] && [ "$wrote" = 0 ] && echo SAME || echo DIFFERS)" \
            "rc=$rc; ${envs:-no switch}; against the base's $nb files: $n compared, $(wc -l < "$A/$tag.differ.txt") differ: ${bad:-none}; written by this build only: $wrote; $(grep -a -m 1 '^Summary' "$A/$tag.log")"
    done
fi

PK=$M/pick
# name:switches (the GGML_CUDA_MMVQ_PASCAL_ prefix left out); every arm runs on every tree, each tree reading its own
PICK_ARMS=${PICK_ARMS:-'none:
dmt:DMT=1
dmt-nopf:DMT=2
dmt-r2:DMT=1 DMT_R=2
dmt-r0:DMT=1 DMT_R=0
dmt-pf:DMT=1 DMT_PF=1
dmt-nw2:DMT=1 DMT_NW=2
dmt-nw1:DMT=1 DMT_NW=1
q6k:Q6K_MT=1
q6k-nopf:Q6K_MT=2
q6k-r2:Q6K_MT=1 DMT_R=2
q6k-nw1:Q6K_MT=1 DMT_NW=1
q6k-dmt:Q6K_MT=1 DMT=1
kt1:DMT=1 DMT_KT=1
kt2:DMT=1 DMT_KT=2
kt1-m2:DMT=1 DMT_KT=1 DMT_KT_M=2
kt1-nopf:DMT=2 DMT_KT=1
kt2-nw4:DMT=1 DMT_KT=2 DMT_KT_NW=4
kt1-r2:DMT=1 DMT_R=2 DMT_KT=1
kt-alone:DMT_KT=1
ro8:RO8=1
ro8-nopf:RO8=2
ro8-pf:RO8=1 DMT_PF=1
ro8-r2:RO8=1 DMT_R=2
ro8-nw2:RO8=1 DMT_NW=2
ro8-dmt:RO8=1 DMT=1
ro8-dmt-nopf:RO8=1 DMT=2
q6k-ro8:Q6K_MT=1 RO8=1
q6k-kt:Q6K_MT=1 DMT=1 DMT_KT=1
kt-ro8:DMT=1 DMT_KT=1 RO8=1
kt2-ro8:DMT=1 DMT_KT=2 RO8=1
all:DMT=1 Q6K_MT=1 DMT_KT=1 RO8=1
all-kt2:DMT=1 Q6K_MT=1 DMT_KT=2 RO8=1
all-nopf:DMT=2 Q6K_MT=2 DMT_KT=1 RO8=2
all-r2:DMT=1 Q6K_MT=1 DMT_KT=1 RO8=1 DMT_R=2
all-nw1:DMT=1 Q6K_MT=1 DMT_KT=1 RO8=1 DMT_NW=1
all-m2:DMT=1 Q6K_MT=1 DMT_KT=1 DMT_KT_M=2 RO8=1'}
if step pick; then
    rm -rf "$PK" && mkdir -p "$PK/arms"
    printf '%s\n' "$PICK_ARMS" > "$PK/arms.txt"
    cp "$RUN/kern_pick.cu" "$PK/pick.cu"
    TREES="base:$BASE_COMMIT"
    for b in $BRANCHES; do TREES="$TREES ${b%%:*}:$(git -C "$SRC" rev-parse "fork/${b#*:}")"; done
    TREES="$TREES kern:$TIP"
    for t in $TREES; do
        mkdir -p "$PK/${t%%:*}"
        git -C "$SRC" archive "${t#*:}" ggml | tar -x -C "$PK/${t%%:*}"
    done
    echo "$TREES" > "$PK/trees.txt"
    # the expressions kern_pick.cu repeats, in each tree: the switch test of ggml_cuda_mmvq_pascal_dmt_takes, Q6_K's one-warp
    # rule, the shared memory and the tiles' call
    f() { grep -cF "$2" "$PK/$1/ggml/src/ggml-cuda/mmvq-pascal-dmt.cu"; }
    T4='if (dmt_mode() == 0 || ncols < 2 || ncols > DMT_MAX_COLS || (type != GGML_TYPE_IQ4_XS && type != GGML_TYPE_Q4_K)) {'
    Q1='return type == GGML_TYPE_Q6_K && dmt_nw_env() != 1 ? nw >= 2 : nw > 0;'
    KT='if (r == 1 && dmt_kt(type, vx, vy, dst, nt, nsb, nrows, row_bytes, col_bytes_y, stride_col_dst, nw, stream)) {'
    SM0='const int smem   = act + nw * p.wstage;'
    SM1='const int smem   = act + nw * p.wstage + dmt_tail(type);'
    found="base $(f base "$T4")$(f base "$SM0") c727 $(f c727 'if (dmt_type_mode(type) == 0 || ncols < 2 || ncols > DMT_MAX_COLS ||')$(f c727 "$Q1")$(f c727 "$SM1") \
c728 $(f c728 "$T4")$(f c728 "$KT")$(f c728 "$SM0") c729 $(f c729 'if (ncols < 2 || ncols > DMT_RO8_COLS || (ncols <= DMT_MAX_COLS ? dmt_mode() : dmt_ro8()) == 0 ||')$(f c729 "$SM0") \
kern $(f kern 'if (ncols < 2 || ncols > DMT_RO8_COLS || dmt_nt_mode(type, (int) ncols) == 0 ||')$(f kern "$Q1")$(f kern "$KT")$(f kern "$SM1")"
    DEFS=$(sed -n 's/^CUDA_DEFINES = //p' "$O/flags.make")
    # shellcheck disable=SC2086
    locked docker run --rm $DA --network none -v "$M:$M" "$IMAGE" bash -c '
        set -e
        for t in base c727 c728 c729 kern; do
            W='"$PK"'/$t
            nice -n 19 nvcc -std=c++17 -O1 -Wno-deprecated-gpu-targets -gencode arch=compute_61,code=compute_61 '"$DEFS"' \
                -DTREE_$(echo $t | tr a-z A-Z) -I$W/ggml/include -I$W/ggml/src -I$W/ggml/src/ggml-cuda \
                -Xcompiler -ffunction-sections -Xcompiler -fdata-sections -Xlinker --gc-sections -o $W/pick '"$PK"'/pick.cu
        done
        while IFS=: read -r name sw; do
            envs=""
            for s in $sw; do envs="$envs GGML_CUDA_MMVQ_PASCAL_$s"; done
            for t in base c727 c728 c729 kern; do
                env $envs '"$PK"'/$t/pick > '"$PK"'/arms/$name-$t.tsv
            done
        done < '"$PK"'/arms.txt' > "$M/logs/pick.log" 2>&1
    rc=$?
    ok=$([ $rc = 0 ] && [ "$found" = "base 11 c727 111 c728 111 c729 11 kern 1111" ] && [ -s "$PK/arms/all-kern.tsv" ] && echo 1 || echo 0)
    add pick build "$([ "$ok" = 1 ] && echo OK || echo FAIL)" \
        "rc=$rc; $TREES; the expressions kern_pick.cu repeats found once each: $found; $(wc -l < "$PK/arms.txt") arms, $(tail -n +2 "$PK/arms/all-kern.tsv" 2> /dev/null | wc -l) products each"
    if [ "$ok" = 1 ]; then
        Ar=$PK/arms
        # region of a line: the branch that owns it in the merged tree
        own() { awk -F'\t' -v r="$2" 'NR > 1 { o = ($1 == "q6_K") ? ($2 <= 4 ? "c727" : "none") : ($2 <= 4 ? "c728" : "c729"); if (o == r) print }' "$1"; }
        bad= badb= nb=0 nl=0
        while IFS=: read -r name sw; do
            # 1. every line the line of the branch that owns it, in that branch's tree; Q6_K at 5-8 tokens off
            for r in c727 c728 c729; do
                d=$(diff <(own "$Ar/$name-kern.tsv" $r) <(own "$Ar/$name-$r.tsv" $r) | grep -c '^[<>]')
                [ "$d" = 0 ] || bad="$bad$name/$r($d) "
            done
            q=$(own "$Ar/$name-kern.tsv" none | awk -F'\t' '$5 != "off"' | wc -l)
            [ "$q" = 0 ] || bad="$bad$name/q6_K-5to8($q) "
            nl=$((nl + $(tail -n +2 "$Ar/$name-kern.tsv" | wc -l)))
            # 2. an arm of the base's switches alone: the merged tree's lines are the base's, every line
            only=1
            for s in $sw; do case ${s%%=*} in DMT|DMT_R|DMT_NW|DMT_PF) ;; *) only=0 ;; esac; done
            if [ $only = 1 ]; then
                nb=$((nb + 1))
                d=$(diff "$Ar/$name-kern.tsv" "$Ar/$name-base.tsv" | grep -c '^[<>]')
                [ "$d" = 0 ] || badb="$badb$name($d) "
            fi
        done < "$PK/arms.txt"
        add pick "by owner" "$([ -z "$bad" ] && echo SAME || echo DIFFERS)" \
            "$(wc -l < "$PK/arms.txt") arms, $nl product lines in all: each the line of its branch's own tree (Q6_K 2-4 tokens C727, IQ4_XS and Q4_K 2-4 C728, 5-8 C729), Q6_K at 5-8 off; differ: ${bad:-none}"
        add pick "base switches alone" "$([ -z "$badb" ] && [ "$nb" -gt 0 ] && echo SAME || echo DIFFERS)" \
            "$nb arms with only _DMT, _DMT_R, _DMT_NW, _DMT_PF (none among them): the merged tree's lines the base's; differ: ${badb:-none}"
        # what launches under all three switches at Flash-Next's widths (K = 2560 and 6144), and the tiles' share
        awk -F'\t' 'NR == 1 || $3 == 2560 || $3 == 6144' "$Ar/all-kern.tsv" > "$PK/flashnext_all.tsv"
        awk -F'\t' 'NR > 1 { k = $5 == "off" || $5 ~ /^declined/ ? $5 : $11; sub(/<.*/, "", k); n[$1 "\t" ($2 <= 4 ? "2-4" : "5-8") "\t" k]++ }
            END { for (x in n) print x "\t" n[x] }' "$Ar/all-kern.tsv" | sort > "$PK/all_kernels.tsv"
    fi
fi

if step emul; then
    E=$M/emul
    rm -rf "$E" && mkdir -p "$E"
    cp "$D/2026-11-11_c727_q6k_readonce/run/q6k_emul.c" "$E/"
    box "$CPU_IMAGE" bash -c \
        "cd $E && gcc -O2 -ffp-contract=off -o q6k_emul q6k_emul.c -lm && nice -n 19 ./q6k_emul rows > q6k_emul.tsv && nice -n 19 ./q6k_emul exact > q6k_exact.tsv" \
        > "$M/logs/emul.log" 2>&1
    rc=$?
    d=$(awk -F'\t' 'NR > 1 { o += $7; d += $8; s += $9; x += $10; c1 += $11 > 0; c2 += $12 > 0; n++ }
        END { printf "%d products, %d outputs, %d differ; %d sub-blocks, %d with x0 + x1 or scales differing; controls differ in %d and %d of %d products", n, o, d, s, x, c1, c2, n }' \
        "$E/q6k_emul.tsv" 2> /dev/null)
    ex=$(awk -F'\t' 'NR > 1 { n += $2; b += $3 } END { printf "exact checks: %d cases, %d differ", n, b }' "$E/q6k_exact.tsv" 2> /dev/null)
    same=$(cmp -s "$E/q6k_emul.tsv" "$D/2026-11-11_c727_q6k_readonce/q6k_emul.tsv" && cmp -s "$E/q6k_exact.tsv" "$D/2026-11-11_c727_q6k_readonce/q6k_exact.tsv" && echo "both tables C727's published ones" || echo "the tables differ from C727's published ones")
    add c727 emul "$([ $rc = 0 ] && awk -F'\t' 'NR > 1 { if ($8 != 0 || $10 != 0) bad = 1; c1 += $11; c2 += $12 } END { exit bad || c1 == 0 || c2 == 0 }' "$E/q6k_emul.tsv" && awk -F'\t' 'NR > 1 && $3 != 0 { bad = 1 } END { exit bad }' "$E/q6k_exact.tsv" && echo SAME || echo DIFFERS)" \
        "rc=$rc; $d; $ex; $same"
fi

if step bitid; then
    X=$REPO/bench/mmvq_pascal/build/multitoken_bitid
    rm -f "$X"
    locked env DOCKER_ARGS="$DA --network none" nice -n 19 bash "$REPO/bench/mmvq_pascal/verify/build_multitoken_bitid.sh" \
        "$SRC" "$SRC" > "$M/logs/bitid.log" 2>&1
    rc=$?
    # shellcheck disable=SC2086
    nt=$(locked docker run --rm $DA --network none -v "$M:$M:ro" -v "$REPO:$REPO:ro" "$CPU_IMAGE" bash -c "nm -C $X | grep -cE ' T ggml_cuda_mmvq_pascal_(dmt|q5k_mt)_takes\('" 2> /dev/null)
    modes=
    for m in q6k dmtkt dmt8 q5k8; do grep -aqw "$m" "$X" 2> /dev/null && modes="$modes$m "; done
    add bitid build "$([ $rc = 0 ] && [ -x "$X" ] && [ "$nt" = 2 ] && [ "$modes" = "q6k dmtkt dmt8 q5k8 " ] && echo OK || echo FAIL)" \
        "rc=$rc; multitoken_bitid md5 $(md5sum < "$X" 2> /dev/null | cut -d' ' -f1), $(stat -c %s "$X" 2> /dev/null) B; modes in it: ${modes:-none}; the two takes functions defined in it: $nt of 2; linked against ${TIP:0:9}"
fi

if step record; then
    rm -rf "$M/sets/v4kern" "$M/sets/v4kern.tar"
    mkdir -p "$M/sets"
    locked env OUT="$M/sets" SUITE=v4kern BIN="$T/bin" JOBS="$JOBS" DOCKER_ARGS="--network none ${DOCKER_ARGS:-}" \
        nice -n 19 bash "$REPO/scripts/record_checks.sh" > "$M/logs/record.log" 2>&1
    rc=$?
    MF=$M/sets/v4kern/manifest.tsv
    add record v4kern "$([ $rc = 0 ] && [ -s "$MF" ] && [ -s "$M/sets/v4kern.tar.md5" ] && echo OK || echo FAIL)" \
        "rc=$rc; $(wc -l < "$MF" 2> /dev/null) cases, $(wc -l < "$M/sets/v4kern/runs.tsv" 2> /dev/null) runs ($(cut -f5 "$M/sets/v4kern/runs.tsv" 2> /dev/null | sort | uniq -c | tr -s ' ' | tr '\n' ';')); v4kern.tar $(stat -c %s "$M/sets/v4kern.tar" 2> /dev/null) B, md5 $(cut -d' ' -f1 "$M/sets/v4kern.tar.md5" 2> /dev/null)"
fi

if step combo; then
    K=$M/combo
    BC=${BASE_CPU:?BASE_CPU=<bin of the base static CPU build>}
    rm -rf "$K" && mkdir -p "$K/off" "$K/models"
    SRV=$D/2026-11-11_v4_late_merge/run/server_combo.sh
    # every switch of the four merges on together: the host ones that compute what the base computes and the CUDA ones
    ALL="LLAMA_MTP_KV_ONLY=1 LLAMA_MTP_CATCHUP_OVERLAP=1 LLAMA_QSA_KPOOL_DEV=1 LLAMA_QSA_KPOOL_STAGE=1 \
LLAMA_QSA_POOL_MASK_DEV=1 LLAMA_QSA_INDEXER_ROWS=1 LLAMA_SAMPLING_TOP_K_ROWS=1 LLAMA_MTP_GRAPH_KEEP=1 \
LLAMA_VERIFY_PREPARE=2 LLAMA_KV_SEQ_RM_BOUND=1 LLAMA_SERVER_YIELD=2 LLAMA_NGRAM_PREFETCH=1 LLAMA_DRAFT_PREFETCH=1 \
LLAMA_DRAFT_EMBD_DEV=layer LLAMA_MTP_H_DEV=2 GGML_SCHED_PREFETCH=0 GGML_CUDA_MMVQ_PASCAL_DMT=1 \
GGML_CUDA_MMVQ_PASCAL_Q5K_MT=1 GGML_CUDA_MMVQ_PASCAL_Q3K_MT=1 GGML_CUDA_MMVQ_PASCAL_GLU_MT=1 \
GGML_CUDA_MMVF_PASCAL_EXACT=1 GGML_CUDA_MMVF_F32_MT=1 GGML_CUDA_FATTN_SPARSE_PASCAL_PROMPT=1 \
GGML_CUDA_FATTN_SPARSE_PASCAL_UNION=1 GGML_CUDA_GRAPHS_FORCE=1 GGML_CUDA_MMVQ_PASCAL_Q5K_1R=1 \
LLAMA_MTP_EMBD_FIRST=1 LLAMA_MTP_DRAFT_CAP=1 LLAMA_MTP_SKIP_LAST_ROW=1 \
GGML_CUDA_MMVQ_PASCAL_Q6K_MT=1 GGML_CUDA_MMVQ_PASCAL_DMT_KT=1 GGML_CUDA_MMVQ_PASCAL_RO8=1"
    LATE_ENV="LLAMA_MTP_EMBD_FIRST=1 LLAMA_MTP_DRAFT_CAP=1 LLAMA_MTP_SKIP_LAST_ROW=1 GGML_CUDA_MMVQ_PASCAL_DMT=1 \
GGML_CUDA_MMVQ_PASCAL_Q6K_MT=1 GGML_CUDA_MMVQ_PASCAL_DMT_KT=1 GGML_CUDA_MMVQ_PASCAL_RO8=1"
    # shellcheck disable=SC2086,SC2016
    locked docker run --rm $DA --network none -v "$M:$M" -v "$BC:$BC:ro" -v "$D:$D:ro" "$SRV_IMAGE" bash -c '
        archs() { env LLAMA_ARCHS_LOGITS='"$K"'/off/logits- "${@:3}" nice -n 19 "$2"/test-llama-archs -b CPU -s 1 2>&1 \
                    | sed "s/\x1b\[[0-9;]*m//g" > '"$K"'/off/$1.log; echo "$1 rc=${PIPESTATUS[0]}"; }
        archs base '"$BC"'
        archs all '"$C/bin"' '"$ALL"'
        rm -f '"$K"'/off/*.f32
        nice -n 19 '"$C/bin"'/test-llama-archs -a "^qwen4exp\$" -s 1 -o '"$K"'/models --mtp > '"$K"'/models/save.log 2>&1
        ARMS="base bbase late latn" LATE_ENV="'"$LATE_ENV"'" BASE='"$BC"' BIN='"$C/bin"' MODEL='"$K"'/models/qwen4exp-moe-mtp.gguf \
            OUT='"$K"'/server THREADS='"$THREADS"' SARGS="--override-kv qwen4exp.context_length=int:16384" \
            nice -n 19 bash '"$SRV"' > '"$K"'/server.console.txt 2>&1
        echo "server rc=$?"
        chown -R '"$(id -u):$(id -g)"' '"$K"'' > "$K/rc.txt" 2>&1
    grep -a 'logits against' "$K/off/base.log" | grep -v ', 0 of ' > "$K/off/base_self.txt"
    nc=$(grep -a -c 'logits against' "$K/off/all.log")
    grep -a 'logits against' "$K/off/all.log" | grep -v ', 0 of ' | grep -vxFf "$K/off/base_self.txt" > "$K/off/all.differ.txt"
    add combo "archs all switches" "$([ "$nc" -gt 0 ] && [ ! -s "$K/off/all.differ.txt" ] && grep -q '^all rc=0' "$K/rc.txt" && echo SAME || echo DIFFERS)" \
        "$nc compared, $(wc -l < "$K/off/all.differ.txt") differ against the base's logits: $(cut -d: -f1 "$K/off/all.differ.txt" | tr '\n' ' '); $(grep -a -m 1 '^Summary' "$K/off/all.log"); $(grep '^all rc=' "$K/rc.txt"); $ALL"
    add combo "server_combo late latn" "$(grep -q '^server rc=0' "$K/rc.txt" && grep -q '^PASS' "$K/server/verdict.txt" && echo OK || echo FAIL)" \
        "$(grep '^server rc=' "$K/rc.txt"); late arms + $LATE_ENV; $(grep -v '^PASS\|^FAIL' "$K/server/verdict.txt" 2> /dev/null | tr '\n' ';')"
    log "combo done"
fi

# server_combo.sh wants no switch line from the base, whose base (the day merge) had no C770; this base has C770, so
# its two arms log the line with LLAMA_SPEC_DUMP alone: every other verdict line of the script must pass
if step combocheck; then
    V=$M/combo/server/verdict.txt
    other=$(grep -v '^PASS$\|^FAIL$' "$V" | grep -vE '^(base|bbase): the base logs a switch line: switches: 1 set: LLAMA_SPEC_DUMP=[^ ]+ \(this build reads 850 LLAMA_ and GGML_ names\)$' \
        | grep -cvE '^(late|latn) = (bbase|base) \(|^(late|latn): the switch line names 21 set, every one read by this build$|^(late|latn): C744.s cap: [1-9][0-9]* drafting passes left out')
    nb=$(grep -cE '^(base|bbase): the base logs a switch line: switches: 1 set: LLAMA_SPEC_DUMP=' "$V")
    add combo "server_combo with a base that has C770" "$([ "$other" = 0 ] && [ "$nb" = 2 ] && [ "$(grep -c ' = b*base (' "$V")" = 2 ] && echo OK || echo FAIL)" \
        "the script's FAIL is its two base arms' switch line (the base, 08634ce00, has C770's line: 1 set, LLAMA_SPEC_DUMP, 850 names): $nb of 2; other lines that are not a pass: $other; $(grep -E ' = b*base \(| names 21 set' "$V" | tr '\n' ';')"
fi

if step c565; then
    for sh in OFF ON; do
        v=$([ $sh = OFF ] && echo static || echo shared)
        room "c565 $v"
        env SRC="$SRC" WORK="$M/c565-$v" TREES="kern=$TIP" STEPS='tree build ctest net archs hand' SHARED=$sh JOBS="$JOBS" \
            THREADS="$THREADS" LOCK="$LOCK" DOCKER_ARGS="--label pascal-quants=1 ${DOCKER_ARGS:-}" nice -n 19 \
            bash "$D/2026-11-05_c565_ladder4fix/run/cpu_tests.sh" > "$M/logs/c565-$v.out" 2>&1
        log "c565 $v rc=$?"
        tail -n +2 "$M/c565-$v/summary.tsv" 2> /dev/null | awk -F'\t' -v r="c565 $v" 'BEGIN { OFS = "\t" }
            NF >= 4 { d = $4; for (i = 5; i <= NF; i++) d = d " " $i; print r, $1 " " $2, $3, d; next } { print r, $1, $2, $3 }' \
            | tee -a "$M/summary.tsv"
        rm -rf "$M/c565-$v/tree"
    done
fi

if step trim; then
    rm -rf "$C/tree"
    gzip -f "$M"/sass/obj/dmt.sass "$M"/sass/c729/kern/llama-server.sass "$M"/sass/c729/kern/obj/*.sass "$M"/sass/c729/kern/obj/*.ptx 2> /dev/null
    add trim "CPU tree" OK "removed (the binaries in cpu/bin stay); $(du -sh "$M" | cut -f1) in $M; $(free_gb) GB free"
fi

if step final; then
    git -C "$SRC" fetch -q fork "$FINAL_BRANCH:refs/remotes/fork/$FINAL_BRANCH"
    F=$(git -C "$SRC" rev-parse "fork/$FINAL_BRANCH")
    if [ "$F" != "$TIP" ]; then
        add final "$FINAL_BRANCH" FAIL "fork/$FINAL_BRANCH is ${F:0:9}, not the checked ${TIP:0:9}"
    else
        room final
        locked env WORK="$M/main/rig" TESTS=1 JOBS="$JOBS" DOCKER_ARGS="$DA" nice -n 19 bash "$REPO/scripts/build_llamacpp_docker.sh" \
            "fork/$FINAL_BRANCH" > "$M/logs/final.log" 2>&1
        rc=$?
        FT=$(tree "$FINAL_BRANCH")
        dif=$(cd "$FT/bin" && for f in *; do cmp -s "$f" "$T/bin/$f" || printf '%s ' "$f"; done)
        nb=$(ls "$FT/bin" | wc -l)
        add final build "$([ $rc = 0 ] && [ -n "$FT" ] && echo OK || echo FAIL)" \
            "rc=$rc; $(basename "$FT").tar.gz md5 $(md5sum < "$FT.tar.gz" | cut -d' ' -f1), $(stat -c %s "$FT.tar.gz") B; $(grep -E '^(commit|avx|highest)' "$FT/BUILDINFO.txt" | tr '\n' ';')"
        add final "bin/ against the checked tree" "$([ -n "$FT" ] && [ -z "$dif" ] && [ "$nb" = "$(ls "$T/bin" | wc -l)" ] && echo SAME || echo DIFFERS)" \
            "$(basename "$FT") against $(basename "$T"): $nb files, differ: ${dif:-none}"
    fi
fi

log "done"
date -Is > "$M/DONE"
