#!/usr/bin/env bash
# upstream-b12723: every instance of the two kernels b12723 renamed (gated_delta_net_cuda gained the template argument
# `replay`, rms_norm_mul_rope_f32 `supports_mrope`) against its base twin: the head's instance with the new argument at
# false and the base's otherwise equal one, through sass_pair.py. Reads $W/base.sass and $W/head.sass (cuobjdump -sass
# of test-backend-ops), writes $W/sass_pairs.tsv.
set -uo pipefail
: "${W:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
out=$W/sass_pairs.tsv
printf 'head kernel\tinstructions\tdiffering\tdiffering past parameter offsets\n' > "$out"
for S in 16 32 64 128; do for K in 0 1; do for KEEP in 0 1; do
    b="_Z20gated_delta_net_cudaILi${S}ELb${K}ELb${KEEP}ELi4EEvPKfS1_S1_S1_S1_S1_PfS2_llllllllllll5uint3S3_fliPKil"
    h="_Z20gated_delta_net_cudaILi${S}ELb${K}ELb${KEEP}ELb0ELi4EEvPKfS1_S1_S1_S1_S1_PfS2_llllllllllll5uint3S3_fliPKil15gdn_replay_args"
    python3 -I "$HERE/sass_pair.py" "$W/base.sass" "$W/head.sass" "$b" "$h" -q >> "$out"
done; done; done
for BS in 256 1024; do for FF in 0 1; do
    b="_Z21rms_norm_mul_rope_f32ILi${BS}ELb${FF}EfEvPKfPT1_iiillllllfS1_lll5uint3S4_S4_S4_iPKifff14rope_corr_dimsfS1_PKlib"
    h="_Z21rms_norm_mul_rope_f32ILi${BS}ELb${FF}EfLb0EEvPKfPT1_iiillllllfS1_lll5uint3S4_S4_S4_iPKifff14rope_corr_dimsfS1_14mrope_sectionsPKlibbb"
    python3 -I "$HERE/sass_pair.py" "$W/base.sass" "$W/head.sass" "$b" "$h" -q >> "$out"
    b="_Z21rms_norm_mul_rope_f32ILi${BS}ELb${FF}E6__halfEvPKfPT1_iiillllllfS2_lll5uint3S5_S5_S5_iPKifff14rope_corr_dimsfS2_PKlib"
    h="_Z21rms_norm_mul_rope_f32ILi${BS}ELb${FF}E6__halfLb0EEvPKfPT1_iiillllllfS2_lll5uint3S5_S5_S5_iPKifff14rope_corr_dimsfS2_14mrope_sectionsPKlibbb"
    python3 -I "$HERE/sass_pair.py" "$W/base.sass" "$W/head.sass" "$b" "$h" -q >> "$out"
done; done
cat "$out"
