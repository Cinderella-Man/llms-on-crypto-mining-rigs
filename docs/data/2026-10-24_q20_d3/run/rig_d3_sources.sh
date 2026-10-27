#!/bin/bash
# Designed on 2026-10-25 and never run, held: heads and store must be built on the strict data list, so HEAD and STORE_SRC
# have no defaults (the R2 heads and the 13.8 M-token store were trained/built on excluded data, not for release).
# Q20/D3 on the rig: what each guess source is worth (decides whether the answer store ships). K2h-r16,
# nine cards -ts 6,6,6,6,6,6,6,5,2, the head (HEAD) on card 0 as in the e2e, depth 3, the e2e's requests (chat, code, edit 512
# tokens, tool 1,536, T 0 and the makers' sampling) and identity checks; T 0 reference = the e2e run's no-guessing
# answers (REF9), so no depth-0 run here. Run from anywhere (RIG: the rig's ssh alias, default rig; OUT: local output dir):
#
#   T=<rc4 build> [T_MD5=<md5>] [CONFIGS='3ds 3s 3d 3c 3'] [STEPS='install run fetch clean']
#   HEAD=<head> STORE_SRC=<local .pstore> [STORE=<its path on the rig>] docs/data/2026-10-24_q20_d3/run/rig_d3_sources.sh
#
# STORE defaults to /opt/llama-builds/stores/<the file name of STORE_SRC>. install
# relays it (md5-checked) unless the rig already has a file with the same md5 at STORE.
#
# Configs (bench/pipespec/rig_server.sh letters), most important first so a cut-off loses the least:
#   3ds  all three: copy lookup, the request's continuations, the store, then the head
#   3s   copying + store: copy lookup, continuations, store (the release's cont with --pipe-store)
#   3d   copying + head, no store: copy lookup, then the head (staircase without --pipe-store)
#   3c   copying alone: the copy lookup only (--pipe-guess copy): the baseline for 3d
#   3    copying + continuations, no store (the release's default cont): the baseline for 3s
# The head's worth with the store is 3ds against 3s, without it 3d against 3c; the store's worth is 3s against 3; the
# continuations' worth 3 against 3c. ~33 min (five servers, ~6.7 min each); CONFIGS='3ds 3s 3d 3c' is ~27 min.
set -u
D=$(cd "$(dirname "$0")" && pwd)
R=$(cd "$D/../../../.." && pwd)   # this repository
RIG=${RIG:-rig}
OUT=${OUT:-$PWD/rig-out}; mkdir -p "$OUT"
T=${T:?T=<rc4 build on the rig>}
T_MD5=${T_MD5:-}
SRC=${SRC:-.}   # local directory holding $T.tar.gz
HEADS=${HEADS:?HEADS=<local directory of exported heads, NAME.gguf>}
HEAD=${HEAD:?HEAD=<drafter name: a head trained on the strict data list>}
CONFIGS=${CONFIGS:-3ds 3s 3d 3c 3}
STORE_SRC=${STORE_SRC:?STORE_SRC=<local .pstore built on the strict data list>}
STORE=${STORE:-/opt/llama-builds/stores/$(basename "$STORE_SRC")}                          # on the rig
K2H=${MODEL:-/mnt/models/flint/flint-q113-K2h-r16.gguf}
STEPS=${STEPS:-install run fetch clean}
B=/opt/llama-builds/$T/bin
Q=/opt/llama-builds/q20
W=/tmp/d3src
TAG=d3src-$HEAD-9l
# the e2e run's no-guessing answers (Flint alone, this repository's prompts): the T 0 reference
REF9=${REF9:?REF9=<results.jsonl of a no-guessing rig_server.sh run, nine cards, release split>}
has() { case " $STEPS " in *" $1 "*) return 0;; esac; return 1; }
RIGSSH="ssh -c aes128-gcm@openssh.com -o Compression=no $RIG"
relay() { # local file, rig file, md5
    cat "$1" | $RIGSSH "mkdir -p $(dirname "$2") && cat > $2.part && echo '$3  $2.part' | md5sum -c --quiet && mv $2.part $2" \
        || { echo "relay of $1 failed"; return 1; }
}

{
echo "== $(date -Is) D3 guess sources: build $T, head $HEAD, configs: $CONFIGS, steps: $STEPS"
if has install; then
    if ! ssh "$RIG" "[ -x $B/llama-server ]"; then
        [ -n "$T_MD5" ] || { echo "T_MD5 needed to relay $T"; exit 1; }
        cat "$SRC/$T.tar.gz" | $RIGSSH "cat > /opt/llama-builds/$T.tar.gz"
        ssh "$RIG" "cd /opt/llama-builds && echo '$T_MD5  $T.tar.gz' | md5sum -c && tar xzf $T.tar.gz && rm $T.tar.gz" || { echo "md5 or unpack failed"; exit 1; }
    fi
    ssh "$RIG" "grep commit /opt/llama-builds/$T/BUILDINFO.txt"
    m=$(md5sum "$HEADS/$HEAD.gguf" | cut -d' ' -f1)
    ssh "$RIG" "echo '$m  $Q/$HEAD.gguf' | md5sum -c --quiet 2> /dev/null" && echo "$HEAD.gguf on the rig ($m)" || relay "$HEADS/$HEAD.gguf" "$Q/$HEAD.gguf" "$m" || exit 1
    sm=$(md5sum "$STORE_SRC" | cut -d' ' -f1)
    [ -n "$sm" ] || { echo "no $STORE_SRC"; exit 1; }
    ssh "$RIG" "echo '$sm  $STORE' | md5sum -c --quiet 2> /dev/null" && echo "store $STORE on the rig ($sm)" \
        || { relay "$STORE_SRC" "$STORE" "$sm" && echo "store relayed to $STORE ($sm)"; } || exit 1
    ssh "$RIG" "rm -rf $W && mkdir -p $W" && scp -q -r "$R/bench/pipespec" "$RIG:$W/" && scp -q "$REF9" "$RIG:$W/ref-9l.jsonl"
    ssh "$RIG" "[ -f $K2H ] && echo 'model ok'"
fi
if has run; then
    echo "== run $TAG: configs $CONFIGS $(date +%T)"
    ssh "$RIG" "env BIN=$B MODEL=$K2H TAG=$TAG GPUS=0,1,2,3,4,5,6,7,8 TS=6,6,6,6,6,6,6,5,2 DEPTHS='$CONFIGS' GUESS=cont CTX=8192 \
             MAX_TOKENS=512 TOOL_MAX_TOKENS=1536 SEED=1 STORE=$STORE DRAFT=$Q/$HEAD.gguf EXTRA_ARGS='-lv 4' \
             REF=$W/ref-9l.jsonl bash $W/pipespec/rig_server.sh > $W/$TAG.out 2>&1; \
             grep -h -m1 -o 'staircase drafter .*' $W/pipespec/out/$TAG/server-d3ds.log | cut -c1-300; tail -14 $W/$TAG.out"
fi
if has fetch; then
    mkdir -p "$OUT"
    ssh "$RIG" "tar czf - -C $W pipespec/out \$(cd $W && ls *.out 2>/dev/null)" | tar xzf - -C "$OUT"
    [ -f "$OUT/pipespec/out/$TAG/summary.tsv" ] && { echo "== $TAG"; column -t -s "$(printf '\t')" < "$OUT/pipespec/out/$TAG/summary.tsv"; }
fi
has clean && ssh "$RIG" "rm -rf $W" && echo "== $W removed on the rig"
echo "== done $(date +%T)"
} 2>&1 | tee -a "$OUT/rig_d3_sources.md"
