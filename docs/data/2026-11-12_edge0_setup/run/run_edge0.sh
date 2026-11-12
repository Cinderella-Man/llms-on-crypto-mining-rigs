#!/usr/bin/env bash
# Serve one Edge0 model folder on one NVIDIA GPU: edge0-infer in a container, which starts edge0-strata, waits until
# the model is loaded and then answers OpenAI- and Anthropic-style requests on 127.0.0.1 of the host.
#
#   WORK=/path/to/edge0 GPU=<index or GPU-uuid> CUDA_DIR=/usr/local/cuda-13.4 bash run_edge0.sh
#
# Optional: MODEL (folder name under $WORK/models, default Edge0-145B-q3k), PORT (default 8000), NAME (container name),
#           PLE_IO (ram = load the 51 GB embedding table into memory, the choice when it sits on a hard disk;
#                   direct = read rows from disk as needed, for a table on an SSD; default ram),
#           DU (q8 or q4: the expert projections' format; the launcher's default when unset),
#           CONTEXT (tokens; the launcher's default is 131072), IMAGE (the build image),
#           EXTRA_MOUNTS (e.g. "-v /mnt/a:/mnt/a -v /mnt/b:/mnt/b" for disks the model folder links to),
#           STRATA_ARGS (words for the engine's own command line, each passed with --strata-arg).
# Watch:    docker logs -f $NAME      (the line "Listening: http://127.0.0.1:$PORT model=..." means ready)
# Stop:     docker stop -t 150 $NAME  (the launcher lets requests finish, then stops the engine)
#
# Memory: with PLE_IO=ram the host needs about 48 GiB for the table on top of the experts that do not fit the card
# (the maker gives 31 GiB for Edge0 145B Q3_K beside a 24 GB card).
set -eu
WORK=${WORK:?set WORK}; GPU=${GPU:?set GPU to an index or a GPU-uuid}
MODEL=${MODEL:-Edge0-145B-q3k}; PORT=${PORT:-8000}; NAME=${NAME:-edge0-serve}
PLE_IO=${PLE_IO:-ram}; DU=${DU:-}; CONTEXT=${CONTEXT:-}; IMAGE=${IMAGE:-edge0-build:24.04}
CUDA_DIR=${CUDA_DIR:-}; EXTRA_MOUNTS=${EXTRA_MOUNTS:-}; STRATA_ARGS=${STRATA_ARGS:-}
SRC=$WORK/src/Edge0; DIR=$WORK/models/$MODEL; HOME_DIR=$WORK/run/$NAME
[ -x "$SRC/edge0-infer/build/cpu/bin/edge0-infer" ] || { echo "edge0-infer is not built under $SRC" >&2; exit 1; }
[ -f "$DIR/model.json" ] || { echo "no model folder at $DIR" >&2; exit 1; }
mkdir -p "$HOME_DIR"

ARGS="serve --model $DIR --host 127.0.0.1 --port $PORT --device cuda:0 --model-option ple-io=$PLE_IO"
[ -z "$DU" ] || ARGS="$ARGS --model-option du=$DU"
[ -z "$CONTEXT" ] || ARGS="$ARGS --context $CONTEXT"
for w in $STRATA_ARGS; do ARGS="$ARGS --strata-arg $w"; done

# The card is cuda:0 inside the container whichever card GPU names. Host networking with a loopback listener keeps
# the server reachable from this machine only, without an API key.
# shellcheck disable=SC2086
docker run -d --name "$NAME" --label pascal-quants=1 --label edge0-serve=1 \
  --gpus "device=$GPU" --network host --ulimit memlock=-1:-1 \
  -v "$WORK:$WORK" ${CUDA_DIR:+-v "$CUDA_DIR:$CUDA_DIR:ro"} $EXTRA_MOUNTS \
  -e HOME="$HOME_DIR" -e CUDA_DIR="$CUDA_DIR" -w "$SRC/edge0-infer" --entrypoint bash "$IMAGE" -c '
    [ -z "$CUDA_DIR" ] || export PATH="$CUDA_DIR/bin:$PATH" LD_LIBRARY_PATH="$CUDA_DIR/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    exec build/cpu/bin/edge0-infer '"$ARGS"'
  '
echo "started $NAME: docker logs -f $NAME"
