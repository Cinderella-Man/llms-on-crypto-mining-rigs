# wait_gpus for cards shared with small jobs, appended to a copy of FN2's run/common.sh
# (../../2026-11-03_flash_next_quality/run/common.sh), whose wait_gpus it replaces: other containers may hold the
# cards, and each card must have at least MIN_FREE_MIB free (one number, or one per card in the order given), checked
# every minute; then the CUDA pre-flight as before. The split stays FN2's (-fit on, which sizes itself to the free
# memory at load), so sharing changes the speed of a run, not its KLD.
wait_gpus() {
    local want=${1//,/ } n=0 i g m free short mins
    IFS=, read -r -a mins <<< "${MIN_FREE_MIB:?}"
    while :; do
        short="" i=0
        for g in $want; do
            free=$(nvidia-smi -i "$g" --query-gpu=memory.free --format=csv,noheader,nounits 2> /dev/null | tr -d ' ')
            case $free in ''|*[!0-9]*) log "GPU $g: nvidia-smi cannot read its free memory"; return 1 ;; esac
            m=${mins[$i]:-${mins[0]}}
            [ "$free" -ge "$m" ] || short="$short gpu$g:${free}<${m}MiB"
            i=$((i + 1))
        done
        if [ -z "$short" ]; then
            log "GPUs $1: enough free memory, shared with: $(busy_gpus | tr '\n' ' ')"
            preflight_gpus "$1" || return 1
            return 0
        fi
        [ $((n % 10)) -eq 0 ] && log "waiting for free memory on GPUs $1:$short"
        n=$((n + 1))
        sleep 60
    done
}
