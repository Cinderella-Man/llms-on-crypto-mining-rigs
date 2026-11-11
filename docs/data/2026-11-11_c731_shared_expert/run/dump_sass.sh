#!/usr/bin/env bash
# dump_sass.sh: C731's dump of the sm_61 code of the kernels that take Qwen3.8-Flash-Next's shared expert, from a
# Miner Llama build tree (no GPU: cuobjdump runs in the CUDA devel image without --gpus). For each object, its
# registers and spills (-res-usage) and its machine code (-sass), gzipped, plus the objects' md5.
#   mmvq-pascal-kcol-<type>.cu.o  the k-column kernels (plain products of 2-8 tokens under the preset; one token for
#                                 Q3_K, IQ4_NL and short Q8_0 rows)
#   mmvq-pascal-mtok.cu.o         the Q2_0 expert kernel of 2-8 tokens, with the routed + shared expert launch
#   mmvq-pascal.cu.o              the one-token Pascal kernels (and the multi-token mul_mat_vec_q_pascal_mc and _mt)
#   mmvq.cu.o                     llama.cpp's mul_mat_vec_q (the fused Q3_K gate/up of one token)
#   mmvf.cu.o                     llama.cpp's mul_mat_vec_f (the F32 gate_inp_shexp of 1-3 tokens)
# Usage: BUILD=<build dir> OUT=<dir> [IMAGE=nvidia/cuda:12.9.2-devel-ubuntu24.04] [DOCKER_ARGS=...] bash dump_sass.sh
set -euo pipefail
: "${BUILD:?BUILD=<build dir>}" "${OUT:?OUT=<dir>}"
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
O=$BUILD/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir
mkdir -p "$OUT"
OBJS="mmvq-pascal-mtok.cu.o mmvq-pascal.cu.o mmvq.cu.o mmvf.cu.o"
for t in q2_0 q3_K q4_K q5_K q6_K iq4_xs q4_0 q5_0 iq4_nl q8_0; do OBJS="$OBJS mmvq-pascal-kcol-$t.cu.o"; done
(cd "$O" && md5sum $OBJS) > "$OUT/obj.md5"
# shellcheck disable=SC2086
docker run --rm ${DOCKER_ARGS:-} -v "$O:$O:ro" -v "$OUT:$OUT" "$IMAGE" sh -c "
    for o in $OBJS; do
        b=\${o%.cu.o}
        cuobjdump -res-usage $O/\$o > $OUT/\$b.res.txt 2>&1
        cuobjdump -sass $O/\$o | gzip > $OUT/\$b.sass.gz
    done
    chown -R $(id -u):$(id -g) $OUT"
