#!/usr/bin/env bash
# Q167's valid1 (docs/data/2026-10-27_ornith_alloc) with a Q5_0 output layer, composed from Q164's files c and d
# (docs/data/2026-10-27_lowbit/run/quantize.sh FILE=c, FILE=d) and a one-tensor GGUF of the Q8_0's output layer at Q5_0
# (docs/data/2026-10-27_ornith_kld/run/head_q5_0.py). Every tensor comes byte for byte from one of them
# (scripts/compose_gguf.py, first matching rule wins):
#   - routed experts' gate/up of layers 24-47 from file d (N3P), their down of layers 0-31 from file d (N2P), with the
#     regexes of alloc.py's switches gu3, gu4, gu5, dn0, dn1, dn2, dn3;
#   - output.weight from the Q5_0 head;
#   - everything else from file c: gate/up N2P in layers 0-23 and the MTP block, down N3P in 32-47 and the MTP block,
#     dense tensors Q5_0, attention k/v and the token embedding Q8_0.
# Runs on the quality machine (QM) in Docker (CPU, low share, no network) with a gguf-py that knows N2P/N3P; the
# machine that runs this script only copies compose_gguf.py over ssh.
#   QM=<ssh alias> FILE_C=<file c> FILE_D=<file d> HEAD=<output-q5_0.gguf> OUT=<new .gguf> GGUFPY=<gguf-py folder> \
#     [RUN=<folder on QM for compose_gguf.py, default OUT's>] bash compose.sh
set -euo pipefail
QM=${QM:?ssh alias of the quality machine}
FILE_C=${FILE_C:?file c}; FILE_D=${FILE_D:?file d}; HEAD=${HEAD:?one-tensor GGUF of output.weight at Q5_0}
OUT=${OUT:?output file}; GGUFPY=${GGUFPY:?gguf-py folder that knows N2P/N3P}
RUN=${RUN:-$(dirname "$OUT")}
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../../.." && pwd)
IMG_PY=pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

ssh "$QM" "mkdir -p $RUN $(dirname "$OUT") && cat > $RUN/compose_gguf.py" < "$REPO/scripts/compose_gguf.py"
m="-v $(dirname "$OUT"):$(dirname "$OUT")"
case " $m " in *" -v $RUN:$RUN"*) ;; *) m="$m -v $RUN:$RUN" ;; esac
for p in "$FILE_C" "$FILE_D" "$HEAD" "$GGUFPY/x"; do
    d=$(dirname "$p"); case " $m " in *" -v $d:$d"*) ;; *) m="$m -v $d:$d:ro" ;; esac
done
gu="^blk\\.(?:$(seq -s'|' 24 47))\\.ffn_(?:gate|up)_exps\\.weight\$"
dn="^blk\\.(?:$(seq -s'|' 0 31))\\.ffn_down_exps\\.weight\$"
hd="^output\\.weight\$"
ssh "$QM" "docker run --rm --label pascal-quants=1 --cpu-shares 256 --network none $m -e PYTHONPATH=$GGUFPY $IMG_PY \
    nice -n 19 python $RUN/compose_gguf.py $FILE_C $OUT.part '$gu=$FILE_D' '$dn=$FILE_D' '$hd=$HEAD' && mv $OUT.part $OUT"
echo "composed: $OUT"
