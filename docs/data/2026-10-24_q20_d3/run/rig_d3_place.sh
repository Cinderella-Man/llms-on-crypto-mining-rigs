#!/bin/bash
# Q20/D3 on the rig: where the head runs. The e2e run put the head on card 1 next to Flint's first stage;
# one guess costs 5.3-7.3 ms there. Same requests and identity checks as the e2e (bench/pipespec/rig_server.sh: chat,
# code, edit 512 tokens, tool 1,536, T 0 and the makers' sampling, depth 3, -c 8192), configs 3s (copy lookup +
# continuations + store) and 3ds (the same, then the head), K2h-r16. Run from anywhere (RIG: the rig's ssh alias, default rig; OUT: local output dir):
#
#   T=<rc4 build on the rig> [T_MD5=<md5>] HEAD9=<nine-card head> [HEAD6=<head for b, default HEAD9>]
#   STORE_SRC=<local .pstore> [STEPS='install a b c fetch clean'] docs/data/2026-10-24_q20_d3/run/rig_d3_place.sh
#
# Designed on 2026-10-25 and never run, held: heads and store must be built on the strict data list, so there are no
# defaults for them (the R2 heads and the 13.8 M-token store were trained/built on excluded data).
#
#   a  nine cards -ts 6,6,6,6,6,6,6,5,2 (llama assigns 7/6/6/6/6/6/6/5/1), HEAD9 on card 8 (one layer + the output head)
#   b  six cards for Flint (-dev CUDA0..CUDA5 -ts 9,8,8,8,8,8) plus card 6 holding nothing but HEAD6 ("pre" window:
#      the newest token has only its embedding when the head starts). A 0 in -ts would not do: llama puts the output
#      head on the last device, so the model is limited to six devices instead
#   c  the control for a: nine cards as in a, HEAD9 on card 0 as in the e2e, with the embedding table in host memory
# The head needs the embedding table on its own card or in host memory (the drafter refuses to load otherwise), so all
# three run with LLAMA_INPUT_ON_GPU=0, the 3s baselines included. Each TAG runs no depth 0: its T 0 reference is the
# e2e run's no-guessing answers (REF9, REF6), so its "x" columns are against that run's no-guessing speed (table
# on card 0); compare 3ds with the 3s of the same TAG. ~33 min: a ~13, b ~12, c ~7, install ~1.
#   install  check T and the heads and store from the e2e (relayed if missing), copy bench/pipespec and the references
#   fetch    /tmp/d3place/pipespec/out back into $OUT; clean removes /tmp/d3place on the rig
set -u
D=$(cd "$(dirname "$0")" && pwd)
R=$(cd "$D/../../../.." && pwd)   # this repository
RIG=${RIG:-rig}
OUT=${OUT:-$PWD/rig-out}; mkdir -p "$OUT"
T=${T:?T=<rc4 build on the rig>}
T_MD5=${T_MD5:-}
SRC=${SRC:-.}   # local directory holding $T.tar.gz
HEADS=${HEADS:?HEADS=<local directory of exported heads, NAME.gguf>}
HEAD9=${HEAD9:?HEAD9=<nine-card head trained on the strict data list>}
HEAD6=${HEAD6:-$HEAD9}
STORE_SRC=${STORE_SRC:?STORE_SRC=<local .pstore built on the strict data list>}
STORE=${STORE:-/opt/llama-builds/stores/$(basename "$STORE_SRC")}
K2H=${MODEL:-/mnt/models/flint/flint-q113-K2h-r16.gguf}
STEPS=${STEPS:-install a b c fetch clean}
B=/opt/llama-builds/$T/bin
Q=/opt/llama-builds/q20
W=/tmp/d3place
# the e2e run's no-guessing answers (Flint alone, this repository's prompts): the T 0 reference
REF9=${REF9:?REF9=<results.jsonl of a no-guessing rig_server.sh run, nine cards, release split>}
REF6=${REF6:?REF6=<results.jsonl of a no-guessing rig_server.sh run, six cards>}
has() { case " $STEPS " in *" $1 "*) return 0;; esac; return 1; }
RIGSSH="ssh -c aes128-gcm@openssh.com -o Compression=no $RIG"
relay() { # local file, rig file, md5
    cat "$1" | $RIGSSH "mkdir -p $(dirname "$2") && cat > $2.part && echo '$3  $2.part' | md5sum -c --quiet && mv $2.part $2" \
        || { echo "relay of $1 failed"; return 1; }
}
serve() { # tag gpus ts configs head draft_args extra ref
    echo "== run $1: configs $4, GPUs $2${3:+, -ts $3}, head $5 ${6:-(card 0)} ${7:+, $7} $(date +%T)"
    ssh "$RIG" "env BIN=$B MODEL=$K2H TAG=$1 GPUS=$2 ${3:+TS=$3} DEPTHS='$4' GUESS=cont CTX=8192 MAX_TOKENS=512 \
             TOOL_MAX_TOKENS=1536 SEED=1 STORE=$STORE DRAFT=$Q/$5.gguf DRAFT_ARGS='$6' EXTRA_ARGS='-lv 4 $7' \
             ENVS='LLAMA_INPUT_ON_GPU=0' REF=$W/$8 bash $W/pipespec/rig_server.sh > $W/$1.out 2>&1; \
             grep -h -m1 -o 'staircase drafter .*' $W/pipespec/out/$1/server-d*.log | cut -c1-300; tail -6 $W/$1.out"
}

{
echo "== $(date -Is) D3 placement: build $T, steps: $STEPS"
if has install; then
    if ! ssh "$RIG" "[ -x $B/llama-server ]"; then
        [ -n "$T_MD5" ] || { echo "T_MD5 needed to relay $T"; exit 1; }
        cat "$SRC/$T.tar.gz" | $RIGSSH "cat > /opt/llama-builds/$T.tar.gz"
        ssh "$RIG" "cd /opt/llama-builds && echo '$T_MD5  $T.tar.gz' | md5sum -c && tar xzf $T.tar.gz && rm $T.tar.gz" || { echo "md5 or unpack failed"; exit 1; }
    fi
    ssh "$RIG" "grep commit /opt/llama-builds/$T/BUILDINFO.txt; $B/llama-server --help 2>/dev/null | grep -c -e '--pipe-draft-dev' | sed 's/^/--pipe-draft-dev: /'"
    for h in $(echo $HEAD9 $HEAD6 | tr " " "\n" | awk "!s[\$0]++"); do
        m=$(md5sum "$HEADS/$h.gguf" | cut -d' ' -f1)
        ssh "$RIG" "echo '$m  $Q/$h.gguf' | md5sum -c --quiet 2> /dev/null" && echo "$h.gguf on the rig ($m)" || relay "$HEADS/$h.gguf" "$Q/$h.gguf" "$m" || exit 1
    done
    sm=$(md5sum "$STORE_SRC" | cut -d' ' -f1); [ -n "$sm" ] || { echo "no $STORE_SRC"; exit 1; }
    ssh "$RIG" "echo '$sm  $STORE' | md5sum -c --quiet 2> /dev/null" && echo "store $STORE on the rig ($sm)" || relay "$STORE_SRC" "$STORE" "$sm" || exit 1
    ssh "$RIG" "rm -rf $W && mkdir -p $W" && scp -q -r "$R/bench/pipespec" "$RIG:$W/" && scp -q "$REF9" "$RIG:$W/ref-9l.jsonl" && scp -q "$REF6" "$RIG:$W/ref-6c.jsonl"
    ssh "$RIG" "ls $W; [ -f $K2H ] && echo 'model ok'"
fi
has a && serve d3place-a-9l-card8 0,1,2,3,4,5,6,7,8 6,6,6,6,6,6,6,5,2 '3s 3ds' $HEAD9 '--pipe-draft-dev 8' '' ref-9l.jsonl
has b && serve d3place-b-6c-spare 0,1,2,3,4,5,6 9,8,8,8,8,8 '3s 3ds' $HEAD6 '--pipe-draft-dev 6' '-dev CUDA0,CUDA1,CUDA2,CUDA3,CUDA4,CUDA5' ref-6c.jsonl
has c && serve d3place-c-9l-card0-host 0,1,2,3,4,5,6,7,8 6,6,6,6,6,6,6,5,2 '3ds' $HEAD9 '' '' ref-9l.jsonl
if has fetch; then
    mkdir -p "$OUT"
    ssh "$RIG" "tar czf - -C $W pipespec/out \$(cd $W && ls *.out 2>/dev/null)" | tar xzf - -C "$OUT"
    for f in "$OUT"/pipespec/out/d3place-*/summary.tsv; do
        [ -f "$f" ] && { echo "== $(basename "$(dirname "$f")")"; column -t -s "$(printf '\t')" < "$f"; }
    done
fi
has clean && ssh "$RIG" "rm -rf $W" && echo "== $W removed on the rig"
echo "== done $(date +%T)"
} 2>&1 | tee -a "$OUT/rig_d3_place.md"
