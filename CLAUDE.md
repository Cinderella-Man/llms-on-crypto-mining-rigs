# llms-on-crypto-mining-rigs

Running LLMs on old crypto-mining hardware. The current rig is a Manli M-P106L9-N6G with nine
NVIDIA P106-090 6 GB cards; its hardware spec is in `docs/mining_rig.md`. Read it before
reasoning about performance: every GPU sits on a PCIe 1.0 x1 link, there is no GPU peer-to-peer, each
card has 5 SMs, and the host has 2 Celeron cores and 4 GB of RAM.

## Working on the rig

- The rig is reached through the SSH alias `rig`, non-interactively: `ssh rig '<command>'`. Each
  contributor defines that alias for their own machine, so the real hostname never enters this repo.
- Build CUDA code with CUDA 12.x for `sm_61`. CUDA 12.x is the last toolkit line and the NVIDIA 580
  driver branch the last driver line that support Pascal.
- Remove anything you leave on the rig, such as test binaries in `/tmp`.

## This repo is public

- Never commit hostnames, tailnet or domain names, usernames, IP or MAC addresses, GPU UUIDs or serial
  numbers, whether in docs, scripts, logs or commit messages.
- Keep raw command output out of the repo; summarize what it shows instead.
- When documenting hardware, prefer measurements taken on the rig over spec sheets and GPU databases.
- Write docs as finished specs: state the correct facts plainly. Don't add review commentary,
  verification markers, or notes about what an earlier version or another source claimed.
