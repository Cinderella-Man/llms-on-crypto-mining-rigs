# The agent check's sandbox image for machines that run attempts in Docker (the quant machine): the fixed in-sandbox
# paths of common.py with pi 0.87.1 (locked as its own installer locks it), node 24, Python 3.12 with pytest, and the
# shared jest node_modules. Build from bench/agent_check:
#   docker build --label pascal-quants=1 -t agent-check:0.87.1 -f docker/Dockerfile .
FROM node:24-bookworm-slim AS node

FROM python:3.12-slim-bookworm
COPY --from=node /usr/local/bin/node /opt/node/bin/node
COPY --from=node /usr/local/lib/node_modules/npm /opt/node/lib/node_modules/npm
RUN ln -s ../lib/node_modules/npm/bin/npm-cli.js /opt/node/bin/npm \
 && ln -s ../lib/node_modules/npm/bin/npx-cli.js /opt/node/bin/npx \
 && apt-get update && apt-get install -y --no-install-recommends git ripgrep ca-certificates \
 && rm -rf /var/lib/apt/lists/*
ENV PATH=/opt/ac-py/bin:/opt/node/bin:/usr/local/bin:/usr/bin:/bin
COPY docker/pi-package.json /opt/pi/package.json
COPY docker/pi-package-lock.json /opt/pi/package-lock.json
RUN cd /opt/pi && npm ci --no-audit --no-fund --ignore-scripts \
 && test -f /opt/pi/node_modules/@earendil-works/pi-coding-agent/dist/bundle/cli.js
COPY env/package.json env/package-lock.json /opt/ac-node/
RUN cd /opt/ac-node && npm ci --no-audit --no-fund --ignore-scripts
COPY env/requirements.txt /tmp/requirements.txt
RUN python3 -m venv /opt/ac-py && /opt/ac-py/bin/pip install --no-cache-dir -r /tmp/requirements.txt \
 && mkdir -p /work /session /opt/pi-agent /home/agent
WORKDIR /work
