#!/usr/bin/env bash
# Q671: checks that each arm's test-backend-ops (build_timing_arms.sh) carries its flag: per arm, the registers of the grid
# types' 2-4-token kernels, the multi-token Q2_0 kernels and the k-column kernels in the binary's sm_61 code
# (cuobjdump -res-usage), and the number of `LDG.E.CG` instructions in the binary. CPU only, no GPU.
# Usage: WORK=<build_timing_arms.sh output dir> ARMS='base iqmc2 ...' [DOCKER_ARGS=...] arm_binary_check.sh
set -euo pipefail
WORK=$(realpath "${WORK:?WORK=<build_timing_arms.sh output dir>}")
ARMS=${ARMS:?ARMS=<arm names>}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
printf 'arm\tgrid_mc_regs_max\tgrid_mc_regs_mean\tmtok_regs_max\tmtok_regs_mean\tkcol_regs_max\tldg_cg_in_binary\n'
for arm in $ARMS; do
    docker run --rm --cpu-shares 256 ${DOCKER_ARGS:-} -v "$WORK/timing/$arm:/a:ro" -e ARM="$arm" "$IMAGE" bash -c '
        res=$(nice -n 19 cuobjdump -res-usage -arch sm_61 /a/bin/test-backend-ops 2>/dev/null)
        regs() { # <name regex>
            echo "$res" | awk -v re="$1" "/^ *Function /{ ok = (\$2 ~ re) } ok && /REG:/{ split(\$1, a, \":\"); print a[2] }" | \
                awk "{ s += \$1; if (\$1 > m) m = \$1; n++ } END { if (n) printf \"%d\t%.1f\", m, s/n; else printf \"-\t-\" }"
        }
        grid=$(regs "mul_mat_vec_q_pascal_mcIL9ggml_type(18|22)")
        mtok=$(regs "mul_mat_vec_q2_0_pascal_mtok")
        kcol=$(regs "mul_mat_kcol" | cut -f1)
        cg=$(nice -n 19 cuobjdump -sass -arch sm_61 /a/bin/test-backend-ops 2>/dev/null | grep -c "LDG.E.CG" || true)
        printf "%s\t%s\t%s\t%s\t%s\n" "$ARM" "$grid" "$mtok" "$kcol" "$cg"' 2>&1 | grep -P "^$arm\t"
done
