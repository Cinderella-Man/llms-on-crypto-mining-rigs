#!/bin/bash
# Designed on 2026-10-25 and never run, held: heads and store must be built on the strict data list, so HEAD and STORE_SRC
# have no defaults (the R2 heads and the 13.8 M-token store were trained/built on excluded data, not for release).
# Q20/D3 on the rig: Flint Q8_0 served at the release context, nine cards, the same flow as the e2e run
# (bench/pipespec/rig_server.sh: chat, code, edit 512 tokens, tool 1,536, T 0 and the makers' sampling, depth 3,
# identity per config). Q8_0 is what the heads were trained on (the dump's teacher), so this is their home ground.
# Run from anywhere (RIG: the rig's ssh alias, default rig; OUT: local output dir):
#
#   T=<rc4 build> [T_MD5=<md5>] HEAD=<head> STORE_SRC=<local .pstore> [STEPS='install run fetch clean']
#   docs/data/2026-10-24_q20_d3/run/rig_d3_q8.sh
#
# Model Q8 (Flint Q8_0, 42.3 GB, ~3-4 min per load from a USB stick), -c 262144,
# -ts 6,6,5,5,6,5,5,6,6 (the model page's full-context split), GPUs 0-8, the head (HEAD) behind the store.
#   run, part 1: configs 0 (no guessing, the reference) and 3s (copy lookup + continuations + store), while
#            nvidia-smi samples every card's memory every 5 s. Part 2 picks the head's card from the free memory under
#            the 3s server: card 0 if it has >= HEAD_MB (600 MiB: 0.27 GB of head, its KV ring, store and graphs, with
#            room) free, as in the e2e; else the card with the most free memory, with the embedding table in host
#            memory for that config (LLAMA_INPUT_ON_GPU=0: the head needs the table on its card or on the host); else
#            no card fits and 3ds is skipped (said in the log). Part 2: config 3ds (the store, then the head), added to
#            part 1's results (APPEND), so the summary has all three with identity at T 0 against no guessing and at
#            T 1 between 3s and 3ds. ~30 min (three loads ~11, generation and prompts ~19).
# Nothing in the store or the head is specific to Q8_0: the store holds token ids, the head reads float hidden states
# and derives its window from the split the run gets.
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
HEAD_MB=${HEAD_MB:-600}
STORE_SRC=${STORE_SRC:?STORE_SRC=<local .pstore built on the strict data list>}
STORE=${STORE:-/opt/llama-builds/stores/$(basename "$STORE_SRC")}
Q8=${Q8:?Q8=<Flint Q8_0 GGUF on the rig>}
TS=6,6,5,5,6,5,5,6,6
STEPS=${STEPS:-install run fetch clean}
B=/opt/llama-builds/$T/bin
Q=/opt/llama-builds/q20
W=/tmp/d3q8
TAG=d3q8-$HEAD-9c
has() { case " $STEPS " in *" $1 "*) return 0;; esac; return 1; }
RIGSSH="ssh -c aes128-gcm@openssh.com -o Compression=no $RIG"
relay() { # local file, rig file, md5
    cat "$1" | $RIGSSH "mkdir -p $(dirname "$2") && cat > $2.part && echo '$3  $2.part' | md5sum -c --quiet && mv $2.part $2" \
        || { echo "relay of $1 failed"; return 1; }
}
serve() { # configs append envs draft_args
    ssh "$RIG" "env BIN=$B MODEL=$Q8 TAG=$TAG GPUS=0,1,2,3,4,5,6,7,8 TS=$TS DEPTHS='$1' APPEND=$2 GUESS=cont CTX=262144 \
             MAX_TOKENS=512 TOOL_MAX_TOKENS=1536 SEED=1 STORE=$STORE DRAFT=$Q/$HEAD.gguf DRAFT_ARGS='$4' \
             EXTRA_ARGS='-lv 4' ENVS='$3' bash $W/pipespec/rig_server.sh >> $W/$TAG.out 2>&1; tail -14 $W/$TAG.out"
}

{
echo "== $(date -Is) D3 Q8_0: build $T, head $HEAD, steps: $STEPS"
if has install; then
    if ! ssh "$RIG" "[ -x $B/llama-server ]"; then
        [ -n "$T_MD5" ] || { echo "T_MD5 needed to relay $T"; exit 1; }
        cat "$SRC/$T.tar.gz" | $RIGSSH "cat > /opt/llama-builds/$T.tar.gz"
        ssh "$RIG" "cd /opt/llama-builds && echo '$T_MD5  $T.tar.gz' | md5sum -c && tar xzf $T.tar.gz && rm $T.tar.gz" || { echo "md5 or unpack failed"; exit 1; }
    fi
    ssh "$RIG" "grep commit /opt/llama-builds/$T/BUILDINFO.txt"
    m=$(md5sum "$HEADS/$HEAD.gguf" | cut -d' ' -f1)
    ssh "$RIG" "echo '$m  $Q/$HEAD.gguf' | md5sum -c --quiet 2> /dev/null" && echo "$HEAD.gguf on the rig ($m)" || relay "$HEADS/$HEAD.gguf" "$Q/$HEAD.gguf" "$m" || exit 1
    sm=$(md5sum "$STORE_SRC" | cut -d' ' -f1); [ -n "$sm" ] || { echo "no $STORE_SRC"; exit 1; }
    ssh "$RIG" "echo '$sm  $STORE' | md5sum -c --quiet 2> /dev/null" && echo "store $STORE on the rig ($sm)" || relay "$STORE_SRC" "$STORE" "$sm" || exit 1
    ssh "$RIG" "rm -rf $W && mkdir -p $W" && scp -q -r "$R/bench/pipespec" "$RIG:$W/"
    ssh "$RIG" "ls -la $Q8 | cut -c1-120 || echo 'MODEL MISSING'; free -m | sed -n 2p"
fi
if has run; then
    echo "== part 1: configs 0 3s, memory sampled every 5 s $(date +%T)"
    ssh "$RIG" "setsid nohup nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits -l 5 > $W/mem.csv 2> /dev/null < /dev/null & echo \$! > $W/mem.pid"
    serve '0 3s' 0 '' ''
    ssh "$RIG" "kill \$(cat $W/mem.pid) 2> /dev/null"
    # per card: the most memory used while the servers ran, and what that leaves free
    free_mb=$(ssh "$RIG" "awk -F', *' '{ if (\$2 > u[\$1]) u[\$1] = \$2; t[\$1] = \$3 } END { for (i = 0; i < 9; i++) printf \"%d \", t[i] - u[i] }' $W/mem.csv")
    echo "free MiB per card at peak (cards 0-8): $free_mb"
    read -r -a fr <<< "$free_mb"
    dev=-1
    if [ "${fr[0]:-0}" -ge "$HEAD_MB" ]; then dev=0
    else
        best=-1; bmb=0
        for i in 1 2 3 4 5 6 7 8; do [ "${fr[$i]:-0}" -gt "$bmb" ] && { bmb=${fr[$i]}; best=$i; }; done
        [ "$bmb" -ge "$HEAD_MB" ] && dev=$best
    fi
    if [ "$dev" = 0 ]; then
        echo "== part 2: config 3ds, $HEAD on card 0 (table on card 0, as in the e2e) $(date +%T)"
        serve '3ds' 1 '' ''
    elif [ "$dev" -gt 0 ]; then
        echo "== part 2: config 3ds, $HEAD on card $dev (card 0 has ${fr[0]} MiB free), table in host memory $(date +%T)"
        serve '3ds' 1 'LLAMA_INPUT_ON_GPU=0' "--pipe-draft-dev $dev"
    else
        echo "== part 2 skipped: no card has $HEAD_MB MiB free at -c 262144 (free per card: $free_mb); 0 and 3s stand alone"
    fi
    ssh "$RIG" "grep -h -m1 -o 'staircase drafter .*' $W/pipespec/out/$TAG/server-d3ds.log 2> /dev/null | cut -c1-300"
fi
if has fetch; then
    mkdir -p "$OUT"
    ssh "$RIG" "tar czf - -C $W pipespec/out mem.csv \$(cd $W && ls *.out 2>/dev/null)" | tar xzf - -C "$OUT"
    [ -f "$OUT/pipespec/out/$TAG/summary.tsv" ] && { echo "== $TAG"; column -t -s "$(printf '\t')" < "$OUT/pipespec/out/$TAG/summary.tsv"; }
    mv -f "$OUT/mem.csv" "$OUT/$TAG-mem.csv" 2> /dev/null
fi
has clean && ssh "$RIG" "rm -rf $W" && echo "== $W removed on the rig"
echo "== done $(date +%T)"
} 2>&1 | tee -a "$OUT/rig_d3_q8.md"
