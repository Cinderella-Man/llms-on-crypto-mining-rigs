# The session sandbox: the pinned python:3.12-slim image (CPython 3.12.15, the approved python-stdlib source) with the
# documentation folders, pip, ensurepip, IDLE and turtledemo removed, plus the tool runner. Whatever a tool output
# quotes then comes from the project in /work, CPython's own library or the tool runner. Sessions run it with
# --network none, a read-only root, a private /tmp and /work as the only writable folder (agent.py).
# Build: docker build --label pascal-quants=1 -t agx-sandbox:py312 bench/agentic_data/sandbox
FROM python:3.12-slim@sha256:cb33f7f771ecec2ef1225cf39b159591a040391bd73e4c01f00fdfc4c47611d2
RUN rm -rf /usr/share/doc /usr/share/man /usr/share/info /usr/share/common-licenses /usr/share/locale \
      /usr/local/lib/python3.12/site-packages/pip /usr/local/lib/python3.12/site-packages/pip-* \
      /usr/local/lib/python3.12/ensurepip /usr/local/bin/pip /usr/local/bin/pip3 /usr/local/bin/pip3.12 \
      /usr/local/lib/python3.12/idlelib /usr/local/lib/python3.12/turtledemo \
 && mkdir -p /work /opt/agx
COPY tools.py /opt/agx/tools.py
WORKDIR /work
