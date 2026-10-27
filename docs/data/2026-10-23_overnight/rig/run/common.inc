# Overnight rig queue, 2026-10-23: sourced by every job (kept in /tmp/rq/jobs/, which the queue never runs since it
# only picks *.sh). SC holds copies of the repository scripts the jobs call: bench/eagle3_flint/verify_cost.sh,
# bench/eagle3_flint/draft_cost.sh, bench/tool_sampling/tool_request_ab.sh and scripts/server_bench.sh.
FORK=/opt/llama-builds/llama.cpp-20261023-eeaa59660-pascal-p106/bin
STOCK=/opt/llama-builds/llama.cpp-20261022-d77256999/bin
S4L0=/mnt/models2/flint/flint-S4L0.gguf
SC=/tmp/rq/scripts
OUTD=/tmp/rq/out
mkdir -p $OUTD
