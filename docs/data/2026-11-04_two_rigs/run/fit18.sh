#!/usr/bin/env bash
# fit18.sh: the file-choice fit (bench/file_choice/fit.py, Q811) on 18 cards, rig 2's nine first (llama.cpp lists RPC
# devices before local ones), for the IQ3_S files and Swift IQ3_XXS; then fit18.md, one line per file, cache type,
# context and draft with the smallest free MiB on any card at each micro-batch (whole layers).
#   FILES="swift-iq3_s=S1.gguf,S2.gguf ista-iq3_s=... swift-iq3_xxs=..." OUT=dir bash fit18.sh
# Reads GGUF headers only (~2 min on one core).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../../.." && pwd)
: "${FILES:?}" "${OUT:?}"
mkdir -p "$OUT"
F=6001,6003,6003,6003,6003,6003,6001,6007,6007  # free MiB per card once the CUDA context is up (Q213's nine cards)
# shellcheck disable=SC2086
nice -n 19 python3 -I "$REPO/bench/file_choice/fit.py" --cards 18 --free "$F,$F" --drafts none,dflash2,mtp $FILES \
    > "$OUT/fit18.tsv"
{
    echo "| File | Cache | Context | Draft | -ub 64 | -ub 128 | -ub 256 | -ub 512 | -ts at the largest -ub with 150 MiB left |"
    echo "|---|---|---|---|---|---|---|---|---|"
    awk -F '\t' 'NR > 1 && $6 == "whole" {
        k = $1 "|" $3 "|" $2 "|" $5
        if (!(k in seen)) { seen[k] = 1; order[++n] = k }
        free[k, $4] = $7
        if ($7 + 0 >= 150 && $4 + 0 > best[k] + 0) { best[k] = $4; ts[k] = $8 }
    } END {
        for (i = 1; i <= n; i++) {
            k = order[i]; split(k, p, "|")
            line = "| " p[1] " | " p[2] " | " p[3] " | " p[4]
            for (u = 64; u <= 512; u *= 2) line = line " | " ((k, u) in free ? free[k, u] : "")
            line = line " | " (k in ts ? "ub " best[k] ": `" ts[k] "`" : "no fit") " |"
            print line
        }
    }' "$OUT/fit18.tsv"
} > "$OUT/fit18.md"
wc -l "$OUT/fit18.tsv" "$OUT/fit18.md"
