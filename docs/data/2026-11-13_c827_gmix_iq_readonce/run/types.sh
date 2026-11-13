#!/usr/bin/env bash
# C827: the tensor types of gmix-q2 against pmix-q2-fast, from the GGUF headers only (CPU, nice 19).
# GMIX and PMIX are the directories holding each file's shards 1 and 2 (shard 3 is the shared n-gram table);
# header copies (truncated files with the full header) work too. OUT is where the tables go.
set -euo pipefail
: "${GMIX:?}" "${PMIX:?}" "${OUT:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT" && cd "$OUT"
nice -n 19 python3 "$HERE/gguf_types.py" "$GMIX"/*-0000[12]-of-00003.gguf > gmix.tsv
nice -n 19 python3 "$HERE/gguf_types.py" "$PMIX"/*-0000[12]-of-00003.gguf > pmix.tsv
# one line per tensor: name, pmix-q2-fast's type, shape, gmix-q2's type
join -t $'\t' <(tail -n +2 pmix.tsv | cut -f2,3,4 | sort) <(tail -n +2 gmix.tsv | cut -f2,3 | sort) > both.tsv
{ printf 'tensor\tpmix_q2_fast\tshape\tgmix_q2\n'; awk -F'\t' '$2 != $4' both.tsv | sort -V; } > changed.tsv
{ printf 'type\tpmix_q2_fast\tgmix_q2\n'
  join -t $'\t' -a1 -a2 -e 0 -o 0,1.2,2.2 \
    <(tail -n +2 pmix.tsv | cut -f3 | sort | uniq -c | awk '{print $2"\t"$1}' | sort) \
    <(tail -n +2 gmix.tsv | cut -f3 | sort | uniq -c | awk '{print $2"\t"$1}' | sort); } > type_counts.tsv
wc -l both.tsv changed.tsv type_counts.tsv
