#!/usr/bin/env bash
# The CPU check of the sparse decode attention (no GPU needed): the tree's fattn-sparse-pascal-impl.cuh must pass every
# case of build/fasp_check (per-token and union kernel); the kernel's four fault codes and six source mutations of the
# header (two of them in the union kernel's warp map) must each fail at least one case. The switch's rule (fasp_check
# --rule) must pass, and six mutations of it (three of the union rule) must fail. Writes build/check.txt (every run's
# table) and prints the verdicts.
# Usage: SRC=<llama.cpp tree> bench/fattn_sparse/check.sh   (env IMAGE, DOCKER_ARGS, JOBS as for build.sh; THREADS: CPU
# threads of ggml's reference, default 4)
set -euo pipefail
cd "$(dirname "$0")"
SRC=$(realpath "${SRC:?SRC=<llama.cpp source tree>}")
H=$SRC/ggml/src/ggml-cuda/fattn-sparse-pascal-impl.cuh
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
THREADS=${THREADS:-4}

# name|sed expression on the header
muts=(
    "Q8_0 rows read without the 2-byte scale|s/(const int8_t \*) (blk + 2 + i0 % 32)/(const int8_t *) (blk + i0 % 32)/"
    "merge adds the splits' sums unscaled|s/l = fmaf(f, FASP_LDCG(ps + D + 1), l);/l = l + FASP_LDCG(ps + D + 1);/"
    "every split but the first skips its first entry|s/begin = s\*share < count ? s\*share : count;/begin = s*share + (s > 0) < count ? s*share + (s > 0) : count;/"
    "scores mapped to the wrong key of the tile|s/return i \/ FASP_HPW;/return i % FASP_KPR;/"
    "union warps on the wrong token of their list|s/r.iq    = j\*tpb + wy \/ nw;/r.iq    = j*tpb + wy % nw;/"
    "union lists of the first sequence for every sequence|s/r.list  = (r.i3 % m_ne3)\*n_lists + j;/r.list  = j;/"
)
# mutations of the switch's rule (fasp_rule_takes), checked with --rule
rmuts=(
    "prompt calls by cells per query token|s/!((double) n_kv < r.prompt_min_ratio\*n_kv_max)/!((double) n_kv < r.prompt_min_ratio*n_kv_max*n_q)/"
    "prompt calls taken with the prompt rule off|s/if (r.prompt == 0 || n_q > r.prompt_nq) {/if (n_q > r.prompt_nq) {/"
    "a call of exactly _NQ tokens by the prompt rule|s/if (n_q <= r.nq) {/if (n_q < r.nq) {/"
    "union lists counted per query token|s/r.union_min_ratio\*n_kv_max\*((n_q + tpb - 1)\/tpb)/r.union_min_ratio*n_kv_max*n_q/"
    "the union rule with _UNION off|s/if (r.uni != 0 \&\& tpb > 1) {/if (tpb > 1) {/"
    "one-token calls by the union rule|s/if (r.uni != 0 \&\& tpb > 1) {/if (r.uni != 0 \&\& tpb > 0) {/"
)

./build.sh fasp_check
for i in "${!muts[@]}"; do
    mkdir -p build/mut$i
    sed "${muts[$i]#*|}" "$H" > build/mut$i/fattn-sparse-pascal-impl.cuh
    if cmp -s "$H" build/mut$i/fattn-sparse-pascal-impl.cuh; then
        echo "mutation $i (${muts[$i]%%|*}) did not apply"
        exit 1
    fi
    ./build.sh fasp_check_mut$i /w/build/mut$i > /dev/null
done
for i in "${!rmuts[@]}"; do
    mkdir -p build/rmut$i
    sed "${rmuts[$i]#*|}" "$H" > build/rmut$i/fattn-sparse-pascal-impl.cuh
    if cmp -s "$H" build/rmut$i/fattn-sparse-pascal-impl.cuh; then
        echo "rule mutation $i (${rmuts[$i]%%|*}) did not apply"
        exit 1
    fi
    ./build.sh fasp_check_rmut$i /w/build/rmut$i > /dev/null
done

run() {
    docker run --rm ${DOCKER_ARGS:-} -v "$PWD:/w" -w /w "$IMAGE" "$@" 2>/dev/null | grep -v "^$\|^==\|NVIDIA\|CUDA Version\|Container image\|governed\|By pulling\|https://\|copy of this\|WARNING\|Use the NVIDIA"
}

: > build/check.txt
verdicts=()
# want 0: every case passes; want 1: at least one case fails
check() {
    local label=$1 want=$2
    shift 2
    echo "== $label" >> build/check.txt
    set +e
    run bash -c "$* --threads $THREADS; echo rc=\$?" >> build/check.txt
    set -e
    local rc
    rc=$(tail -1 build/check.txt | sed 's/rc=//')
    local fails
    fails=$(grep -c "FAIL$" <(sed -n "/^== $label\$/,/^rc=/p" build/check.txt) || true)
    local what="$fails failing cases"
    if [[ " $* " == *" --rule "* ]]; then
        what=$(sed -n "/^== $label\$/,/^rc=/p" build/check.txt | grep -o "fasp_check --rule: [A-Za-z]*" | tail -n 1)
    fi
    if { [ "$want" = 0 ] && [ "$rc" = 0 ]; } || { [ "$want" = 1 ] && [ "$rc" = 1 ]; }; then
        verdicts+=("ok    $label: $what")
    else
        verdicts+=("WRONG $label: rc=$rc, $what")
    fi
}

check "clean header" 0 build/fasp_check
for f in 1 2 3 4; do
    check "fault $f" 1 build/fasp_check --fault $f
done
for i in "${!muts[@]}"; do
    check "mutation $i: ${muts[$i]%%|*}" 1 build/fasp_check_mut$i
done
check "the switch's rule" 0 build/fasp_check --rule
for i in "${!rmuts[@]}"; do
    check "rule mutation $i: ${rmuts[$i]%%|*}" 1 build/fasp_check_rmut$i --rule
done
printf '%s\n' "${verdicts[@]}" | tee -a build/check.txt
! printf '%s\n' "${verdicts[@]}" | grep -q "^WRONG"
