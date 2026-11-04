#!/usr/bin/env bash
# The checks of the Pascal IQ2_XXS / IQ2_XS / IQ3_S kernels (Q222) that need no GPU, on any x86-64 Linux PC with Docker:
#   1. decode    bench/mmvq_pascal/verify/iq_unit_check.sh: each type's decoding against ggml's own dequantization on the
#                CPU (200 trials of 2 cases per type), then MUTATE=1, which must fail exactly iq2_xxs and iq3_s
#   2. build     the rig tarball (scripts/build_llamacpp_docker.sh, TESTS=1, compute 6.1) of the branch
#   3. regs      registers, stack, shared and local memory of every grid-type kernel (cuobjdump -res-usage of
#                test-backend-ops) as registers.tsv: no kernel may use local memory (spills)
#   4. sass      the sm_61 SASS of every kernel of the new build against a build of the tree it was cut from
#                (docs/data/2026-11-04_ladder2_merge/run/sass_identity.sh): every kernel of the old build must come out
#                with identical machine code, and the new kernels must all be of the three new types
# Env: WORK (work folder; the build clones into $WORK/src, see build_llamacpp_docker.sh), REPO (a checkout of this
# repository), TREE (the branch's source, e.g. `git archive` of it, for step 1), REF (git ref for step 2, e.g.
# fork/pascal-p106-v3-b12076-iq2xs), BASE_BUILD (the unpacked tarball of the base tree for step 4), JOBS (default 6),
# DOCKER_ARGS (e.g. a label and CPU share a shared machine asks for), LOCK (a flock file that serializes builds on the
# machine; optional), STEPS (default "decode build regs sass").
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
    # both runs under one hold of the lock; the second reuses the first's ggml build
    rm -f "$C/decode.rc" "$C/decode_mut.rc"
    export B=$REPO/bench/mmvq_pascal/verify/iq_unit_check.sh TREE=${TREE:?} C JOBS DOCKER_ARGS=${DOCKER_ARGS:-}
    locked bash -c 'bash "$B" "$TREE" 200 > "$C/decode.txt" 2>&1; echo $? > "$C/decode.rc"
        MUTATE=1 bash "$B" "$TREE" 200 > "$C/decode_mut.txt" 2>&1; echo $? > "$C/decode_mut.rc"'
    rc=$(cat "$C/decode.rc" 2>/dev/null || echo 9)
    rcm=$(cat "$C/decode_mut.rc" 2>/dev/null || echo 9)
    verdict decode "$([ $rc = 0 ] && [ $rcm = 0 ] && echo OK || echo FAIL)" \
        "$(grep -E '^decoding' "$C/decode.txt" | tr '\n' ' ')/ $(grep -E '^mutation' "$C/decode_mut.txt" | tr '\n' ' ')"
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
    docker run --rm ${DOCKER_ARGS:-} -v "$NEW:/b:ro" -v "$C:/c" "$IMAGE" bash -euo pipefail -c '
        cuobjdump -res-usage -arch sm_61 /b/bin/test-backend-ops > /c/res.txt
        grep -o "Function [^:]*" /c/res.txt | cut -d" " -f2 | c++filt > /c/res.names
        grep -o "REG:[0-9]* STACK:[0-9]* SHARED:[0-9]* LOCAL:[0-9]*" /c/res.txt > /c/res.vals
        {
            printf "kernel\ttype\tR\tC\tstaged\tfk\tsa\tSU\ttile\tregs\tstack\tshared\tlocal\n"
            paste -d"|" /c/res.names /c/res.vals | sed -E "s/^(void )?(\(anonymous namespace\)::)?//" | grep -E "^mul_mat_vec_q_pascal(_mc)?<\(ggml_type\)(16|17|18|21|22)," |
            sed -E "s/^(mul_mat_vec_q_pascal(_mc)?)<\(ggml_type\)([0-9]+), (.*)>\(.*\|REG:([0-9]+) STACK:([0-9]+) SHARED:([0-9]+) LOCAL:([0-9]+)$/\1\t\3\t\4\t\5\t\6\t\7\t\8/" |
            awk -F"\t" -v OFS="\t" "{ split(\$3, a, \", \"); tn[16]=\"iq2_xxs\"; tn[17]=\"iq2_xs\"; tn[18]=\"iq3_xxs\"; tn[21]=\"iq3_s\"; tn[22]=\"iq2_s\";
                mc = \$1 ~ /_mc\$/; R=a[1]; Cc=a[2]; st=a[3]; fk=a[4]; sa = mc ? a[5] : \"-\"; su = mc ? a[6] : a[5]; ti = mc ? a[7] : a[6];
                print \$1, tn[\$2], R, Cc, st, fk, sa, su, ti, \$4, \$5, \$6, \$7 }" | sort -t"	" -k2,2 -k1,1 -k6,6n -k8,8n
        } > /c/registers.tsv
        rm /c/res.names /c/res.vals'
    n=$(($(wc -l < "$C/registers.tsv") - 1))
    nl=$(awk -F'\t' 'NR > 1 && $13 > 0' "$C/registers.tsv" | wc -l)
    ns=$(awk -F'\t' 'NR > 1 && $11 > 0' "$C/registers.tsv" | wc -l)
    verdict regs "$([ "$n" -gt 0 ] && [ "$nl" = 0 ] && echo OK || echo REVIEW)" \
        "$n grid-type kernels, $nl with local memory, $ns with a stack frame; registers $(awk -F'\t' 'NR > 1 { print $10 }' "$C/registers.tsv" | sort -n | sed -n '1p;$p' | tr '\n' ' ')(min, max)"
fi

if has sass; then
    echo "=== 4. sass $(date -Is)"
    WORK=$C DOCKER_ARGS="${DOCKER_ARGS:-}" bash "$REPO/docs/data/2026-11-04_ladder2_merge/run/sass_identity.sh" "$NEW" \
        "base=${BASE_BUILD:?}" > "$WORK/logs/sass.log" 2>&1
    rc=$?
    # every kernel of the base build must be in the new one with identical code ("as base"), and every other kernel
    # of the new build must be new ("NEW in merge") and of the new types (ggml_type 16 IQ2_XXS, 17 IQ2_XS, 21 IQ3_S)
    nbase=$(zcat "$C/sass/base.sass.gz" | grep "Function : " | sort -u | wc -l)
    v=$(python3 -I - "$C/sass/kernels.tsv" "${nbase:-0}" <<'EOF'
import collections, re, sys
rows = [l.rstrip("\n").split("\t") for l in open(sys.argv[1])][1:]
cls = collections.Counter(r[6] for r in rows)
new = [r[0] for r in rows if r[6] == "NEW in merge"]
other = [n for n in new if not re.search(r"ggml_type(16|17|21)E", n)]
bad = sum(v for k, v in cls.items() if k not in ("as base", "NEW in merge"))
ok = bad == 0 and not other
print(("OK" if ok else "REVIEW") + f"\t{len(rows)} kernels: {cls['as base']} identical to the base (which has {sys.argv[2]} kernel names), "
      f"{len(new)} new ({len(other)} not of the new types), {bad} changed")
EOF
)
    [ $rc = 0 ] || v="FAIL	rc $rc"
    verdict sass "${v%%	*}" "${v#*	}"
fi
echo "=== end $(date -Is)"
touch "$WORK/DONE"
