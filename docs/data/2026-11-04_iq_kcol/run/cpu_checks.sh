#!/usr/bin/env bash
# The checks of the Q713 kernels (the grid types and IQ1_M in the k-column and multi-token Pascal kernels) that need no
# GPU, on any x86-64 Linux PC with Docker:
#   1. decode    bench/mmvq_pascal/verify/iq_kcol_check.sh: each type's decoding in both kernels against ggml's own
#                dequantization on the CPU (and the multi-token arithmetic bit for bit against the one-token kernels'
#                functions), then MUTATE=1, which must fail exactly iq2_s, iq3_s and iq1_m
#   2. build     the rig tarball (scripts/build_llamacpp_docker.sh, TESTS=1, compute 6.1) of the branch
#   3. regs      registers, stack frame and shared memory of every new kernel (cuobjdump -res-usage of test-backend-ops)
#                as registers.tsv: none of the kernels the launch rules pick may use local memory or a stack frame
#                (spills); the grid types' 2-row k-column kernels, which run only with GGML_CUDA_MMVQ_PASCAL_KCOL_R=2, are
#                counted apart
#   4. sass      the sm_61 SASS of every kernel against the builds of the two branches the branch joins
#                (docs/data/2026-11-04_ladder2_merge/run/sass_identity.sh): -iq2xs (the base) and -kcol at af46311af, cut
#                from -kcol at e1fa4c573 (the -kcol that -iq2xs already holds): every kernel must come out with the code
#                of one of them, every kernel -kcol changed since e1fa4c573 identical, every other kernel one of the new ones
# Env: WORK (work folder; the build clones into $WORK/src, see build_llamacpp_docker.sh), REPO (a checkout of this
# repository), TREE (the branch's source, e.g. `git archive` of it, for step 1), REF (git ref for step 2, e.g.
# fork/pascal-p106-v3-b12076-iqkcol), BASE_BUILD (the unpacked tarball of the -iq2xs branch, the base), KCOL_BUILD (of
# the -kcol branch at the merged commit), KCOL_OLD_BUILD (of -kcol at the commit -iq2xs holds), JOBS (default 6), DOCKER_ARGS (e.g. a label and CPU share a shared machine
# asks for), LOCK (a flock file that serializes builds on the machine; optional), STEPS (default
# "decode build regs sass").
# Output: $WORK/checks/{decode.txt,decode_mut.txt,registers.tsv,sass/}, $WORK/summary.tsv (one verdict per step), the
# tarball and its unpacked tree in $WORK, $WORK/md5.txt.
set -uo pipefail
WORK=$(realpath "${WORK:?}"); REPO=$(realpath "${REPO:?}")
STEPS=${STEPS:-decode build regs sass}
JOBS=${JOBS:-6}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
C=$WORK/checks
mkdir -p "$C" "$WORK/logs"
locked() { if [ -n "${LOCK:-}" ]; then flock -w 10800 "$LOCK" "$@"; else "$@"; fi; }
verdict() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" | tee -a "$WORK/summary.tsv"; }
has() { case " $STEPS " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

if has decode; then
    echo "=== 1. decode $(date -Is)"
    rm -f "$C/decode.rc" "$C/decode_mut.rc"
    export B=$REPO/bench/mmvq_pascal/verify/iq_kcol_check.sh TREE=${TREE:?} C JOBS DOCKER_ARGS=${DOCKER_ARGS:-}
    locked bash -c 'bash "$B" "$TREE" 100 > "$C/decode.txt" 2>&1; echo $? > "$C/decode.rc"
        MUTATE=1 bash "$B" "$TREE" 100 > "$C/decode_mut.txt" 2>&1; echo $? > "$C/decode_mut.rc"'
    rc=$(cat "$C/decode.rc" 2>/dev/null || echo 9)
    rcm=$(cat "$C/decode_mut.rc" 2>/dev/null || echo 9)
    verdict decode "$([ $rc = 0 ] && [ $rcm = 0 ] && echo OK || echo FAIL)" \
        "$(grep -E '^(decoding|iq1s_grid_gpu)' "$C/decode.txt" | tr '\n' ' ')/ $(grep -E '^mutation' "$C/decode_mut.txt" | tr '\n' ' ')"
    [ $rc = 0 ] || { echo "decode check failed: see $C/decode.txt"; exit 1; }
fi

if has build; then
    echo "=== 2. build $(date -Is)"
    locked env WORK="$WORK" TESTS=1 JOBS="$JOBS" DOCKER_ARGS="${DOCKER_ARGS:-}" \
        bash "$REPO/scripts/build_llamacpp_docker.sh" "${REF:?}" > "$WORK/logs/build.log" 2>&1
    rc=$?
    T=$(ls -t "$WORK"/llama.cpp-*.tar.gz 2>/dev/null | head -n 1)
    if [ $rc = 0 ] && [ -n "$T" ]; then
        (cd "$WORK" && md5sum "$(basename "$T")" > md5.txt && stat -c '%s %n' "$(basename "$T")" >> md5.txt)
        verdict build OK "$(basename "$T") $(head -n 1 "$WORK/md5.txt" | cut -d' ' -f1)"
    else
        verdict build FAIL "rc $rc; tail of logs/build.log"
        tail -n 40 "$WORK/logs/build.log"
        exit 1
    fi
fi
T=$(ls -t "$WORK"/llama.cpp-*.tar.gz 2>/dev/null | head -n 1)
NEW=${T%.tar.gz}

if has regs; then
    echo "=== 3. regs $(date -Is)"
    # the k-column kernels of the new types (kcol::ktype 10-15) and the multi-token kernels of mmvq-pascal-mtok-iq.cuh
    docker run --rm ${DOCKER_ARGS:-} -v "$NEW:/b:ro" -v "$C:/c" "$IMAGE" bash -euo pipefail -c '
        cuobjdump -res-usage -arch sm_61 /b/bin/test-backend-ops > /c/res.txt
        grep -o "Function [^:]*" /c/res.txt | cut -d" " -f2 | c++filt > /c/res.names
        grep -o "REG:[0-9]* STACK:[0-9]* SHARED:[0-9]* LOCAL:[0-9]*" /c/res.txt > /c/res.vals
        {
            printf "kernel\ttype\tgate\tSU_or_cols\tNTM_or_R\tregs\tstack\tshared\tlocal\n"
            paste -d"|" /c/res.names /c/res.vals | sed -E "s/^(void )?//" |
                grep -E "^(kcol::mul_mat_kcol<1[0-5], |mtok_iq::mul_mat_vec_iq_pascal_mtok<)" |
                sed -E "s/^kcol::mul_mat_kcol<([0-9]+), ([0-9]+), ([0-9]+)>.*\|REG:([0-9]+) STACK:([0-9]+) SHARED:([0-9]+) LOCAL:([0-9]+)$/kcol\t\1\t-\t\2\t\3\t\4\t\5\t\6\t\7/;
                        s/^mtok_iq::mul_mat_vec_iq_pascal_mtok<\(ggml_type\)([0-9]+), (true|false), ([0-9]+), ([0-9]+)>.*\|REG:([0-9]+) STACK:([0-9]+) SHARED:([0-9]+) LOCAL:([0-9]+)$/mtok\t\1\t\2\t\3\t\4\t\5\t\6\t\7\t\8/" |
                awk -F"\t" -v OFS="\t" "{ if (\$1 == \"kcol\") { split(\"iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s iq1_m\", n, \" \"); \$2 = n[\$2 - 9] }
                    else { m[16] = \"iq2_xxs\"; m[17] = \"iq2_xs\"; m[18] = \"iq3_xxs\"; m[21] = \"iq3_s\"; m[22] = \"iq2_s\"; m[29] = \"iq1_m\"; \$2 = m[\$2] }
                    print }" | sort -t"	" -k1,1 -k2,2 -k3,3 -k4,4n -k5,5n
        } > /c/registers.tsv
        rm /c/res.names /c/res.vals'
    n=$(($(wc -l < "$C/registers.tsv") - 1))
    # R = 2 k-column kernels of the 2-byte-aligned grid types: only with GGML_CUDA_MMVQ_PASCAL_KCOL_R=2
    r2() { awk -F'\t' 'NR > 1 && $1 == "kcol" && $2 != "iq1_m" && $5 == 2' "$C/registers.tsv"; }
    nl=$(awk -F'\t' 'NR > 1 && !($1 == "kcol" && $2 != "iq1_m" && $5 == 2) && ($9 > 0 || $7 > 0)' "$C/registers.tsv" | wc -l)
    nr2=$(r2 | awk -F'\t' '$9 > 0 || $7 > 0' | wc -l)
    verdict regs "$([ "$n" = 204 ] && [ "$nl" = 0 ] && echo OK || echo REVIEW)" \
        "$n new kernels (96 k-column, 108 multi-token), $nl of the default ones with a stack frame or local memory, $nr2 of the 40 forced-R=2 grid-type k-column ones (stack $(r2 | awk -F'\t' '{ print $7 }' | sort -n | uniq | tr '\n' ' ')bytes); registers $(awk -F'\t' 'NR > 1 { print $6 }' "$C/registers.tsv" | sort -n | sed -n '1p;$p' | tr '\n' ' ')(min, max)"
fi

if has sass; then
    echo "=== 4. sass $(date -Is)"
    WORK=$C DOCKER_ARGS="${DOCKER_ARGS:-}" bash "$REPO/docs/data/2026-11-04_ladder2_merge/run/sass_identity.sh" "$NEW" \
        "iq2xs=${BASE_BUILD:?}" "kcol:kcol_old=${KCOL_BUILD:?}" "kcol_old=${KCOL_OLD_BUILD:?}" > "$WORK/logs/sass.log" 2>&1
    rc=$?
    # every kernel of the two branches must be in the new build with identical code ("as base" or carried from kcol),
    # and every other kernel of the new build must be new ("NEW in merge") and one of Q713's
    v=$(docker run --rm -i ${DOCKER_ARGS:-} -v "$C/sass:/s:ro" python:3.12-slim-bookworm python3 -I - /s/kernels.tsv /s/carried.tsv <<'EOF'
import collections, re, sys
rows = [l.rstrip("\n").split("\t") for l in open(sys.argv[1])][1:]
# kcol_old is a reference tree only: what -kcol changed since it must be carried unchanged
carried = [l.rstrip("\n").split("\t") for l in open(sys.argv[2])][1:]
not_carried = sum(1 for r in carried if r[0] == "kcol" and r[2] != "identical")
cls = collections.Counter(r[6] for r in rows)
new = [r[0] for r in rows if r[6] == "NEW in merge"]
other = [n for n in new if not re.search(r"(mul_mat_kcol|quantize_records)ILi1[0-5]E|mul_mat_vec_iq_pascal_mtok", n)]
bad = sum(v for k, v in cls.items() if k not in ("as base", "NEW in merge") and not k.startswith("as kcol"))
ok = bad == 0 and not other and not_carried == 0
print(("OK" if ok else "REVIEW") + f"\t{len(rows)} kernels: " + ", ".join(f"{v} {k}" for k, v in sorted(cls.items()))
      + f"; {len(other)} new kernels not of Q713; {not_carried} kernels -kcol changed not carried identically")
EOF
)
    [ $rc = 0 ] || v="FAIL	rc $rc"
    verdict sass "${v%%	*}" "${v#*	}"
fi
echo "=== end $(date -Is)"
touch "$WORK/DONE"
