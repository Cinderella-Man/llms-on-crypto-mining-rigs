#!/usr/bin/env python3
"""C1025's rig half (Q1758): what upstream's NVML probe at CUDA start costs a P106 rig, on one build, two arms.

  unset   GGML_CUDA_NVML_PROBE unset: no P106 has fast BF16, so the build skips the probe
  probe   GGML_CUDA_NVML_PROBE=1: the probe runs as upstream b12674 runs it (dlopen of libnvidia-ml.so.1, nvmlInit)

Runs on the rig itself (python3 -I, standard library only), detached; rig_probe.sh starts, waits for and collects it.
  (1) the device list: REPS pairs of `llama-server -lv 4 --list-devices` with the preset's environment, the order of
      each pair alternating (unset first, then probe first): the process's wall time and peak resident memory
      (getrusage of the child), and the build's probe line (its own time to open and initialise NVML and to shut it
      down, and the resident memory it added);
  (2) one load each way of MODEL on the preset's line (PRESET_ENV, PRESET_ARGS) with -lv 4 first: from the start to
      /health, the log's own timestamps of "found N CUDA devices", the first card's line and the probe line, the
      server's peak resident memory (VmHWM at /health and at exit), the host's MemAvailable before the load and its
      lowest value during it (sampled every 0.25 s), and one 16-token greedy reply, compared between the arms.
A load stops if MemAvailable falls below --min-avail-mb (the 4 GB host). Writes results.csv, summary.md and logs/.
"""
import argparse, json, os, re, shlex, signal, subprocess, sys, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--bin", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--env", default="", help="the preset's PRESET_ENV")
ap.add_argument("--args", default="", help="the preset's PRESET_ARGS")
ap.add_argument("--out", required=True)
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--loads", default="unset probe")
ap.add_argument("--port", type=int, default=18925)
ap.add_argument("--min-avail-mb", type=int, default=60)
ap.add_argument("--load-timeout", type=int, default=1800)
a = ap.parse_args()
os.makedirs(os.path.join(a.out, "logs"), exist_ok=True)
ARM_ENV = {"unset": {}, "probe": {"GGML_CUDA_NVML_PROBE": "1"}}


def env_for(arm):
    e = dict(os.environ)
    e.pop("GGML_CUDA_NVML_PROBE", None)
    for kv in shlex.split(a.env):
        k, _, v = kv.partition("=")
        e[k] = v
    e.update(ARM_ENV[arm])
    return e


def mem_available_kib():
    with open("/proc/meminfo") as f:
        for l in f:
            if l.startswith("MemAvailable:"):
                return int(l.split()[1])
    return -1


def status_kib(pid, key):
    try:
        with open(f"/proc/{pid}/status") as f:
            for l in f:
                if l.startswith(key + ":"):
                    return int(l.split()[1])
    except OSError:
        pass
    return -1


TS = re.compile(r"^(\d+)\.(\d\d)\.(\d\d\d)\.(\d\d\d) ")


def log_s(line):
    """llama-server's log prefix m.ss.mmm.uuu, in seconds from the log's start; None without one."""
    m = TS.match(line)
    if not m:
        return None
    mi, s, ms, us = (int(x) for x in m.groups())
    return mi * 60 + s + ms / 1e3 + us / 1e6


PROBE = re.compile(r"NVML probe ran \((.*?)\): library (\S+), nvmlInit (\S+); open and init ([0-9.]+) ms, shutdown and close "
                   r"([0-9.]+) ms; host RSS ([-+0-9.]+) MiB after init, ([-+0-9.]+) MiB after close")


def read_log(path):
    r = {"probe": "-", "open_init_ms": "", "close_ms": "", "rss_init_mib": "", "rss_close_mib": "",
         "t_found_s": "", "t_device0_s": "", "t_probe_line_s": ""}
    with open(path, errors="replace") as f:
        for l in f:
            if "NVML probe" in l:
                m = PROBE.search(l)
                if m:
                    r.update(probe=f"ran ({m.group(1)}; library {m.group(2)}, init {m.group(3)})", open_init_ms=m.group(4),
                             close_ms=m.group(5), rss_init_mib=m.group(6), rss_close_mib=m.group(7))
                elif "skipped" in l:
                    r["probe"] = "skipped " + l.split("skipped", 1)[1].strip()
                t = log_s(l)
                r["t_probe_line_s"] = "" if t is None else f"{t:.6f}"
            elif "CUDA devices" in l and "found" in l and not r["t_found_s"]:
                t = log_s(l)
                r["t_found_s"] = "" if t is None else f"{t:.6f}"
            elif re.search(r"Device 0: ", l) and not r["t_device0_s"]:
                t = log_s(l)
                r["t_device0_s"] = "" if t is None else f"{t:.6f}"
    return r


rows = []
COLS = ["phase", "rep", "arm", "rc", "wall_s", "maxrss_kib", "vmhwm_health_kib", "health_s", "memavail_before_kib",
        "memavail_min_kib", "stopped", "probe", "open_init_ms", "close_ms", "rss_init_mib", "rss_close_mib", "t_found_s",
        "t_device0_s", "t_probe_line_s", "reply_tokens"]


def save():
    with open(os.path.join(a.out, "results.csv"), "w") as f:
        f.write(",".join(COLS) + "\n")
        for r in rows:
            f.write(",".join('"%s"' % str(r.get(c, "")).replace('"', "'") for c in COLS) + "\n")


def run_list(rep, arm):
    log = os.path.join(a.out, "logs", f"list-{rep}-{arm}.log")
    t0 = time.monotonic()
    with open(log, "w") as lf:
        p = subprocess.Popen([a.bin, "-lv", "4", "--list-devices"], stdout=lf, stderr=subprocess.STDOUT, env=env_for(arm))
        _, st, ru = os.wait4(p.pid, 0)
    wall = time.monotonic() - t0
    r = {"phase": "list", "rep": rep, "arm": arm, "rc": os.waitstatus_to_exitcode(st), "wall_s": f"{wall:.4f}",
         "maxrss_kib": ru.ru_maxrss}
    r.update(read_log(log))
    rows.append(r)
    save()


def post(path, body, timeout):
    req = urllib.request.Request(f"http://127.0.0.1:{a.port}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def run_load(arm):
    log = os.path.join(a.out, "logs", f"load-{arm}.log")
    # -lv 4 first: the library's INFO lines (the probe's among them) print only from verbosity 4, and CUDA starts while
    # the later arguments are read
    cmd = [a.bin, "-lv", "4", "-m", a.model] + shlex.split(a.args) + ["--host", "127.0.0.1", "--port", str(a.port)]
    with open(os.path.join(a.out, "logs", f"load-{arm}.cmd"), "w") as f:
        f.write(" ".join(f"{k}={v}" for k, v in sorted(ARM_ENV[arm].items())) + " " + a.env + " " + " ".join(cmd) + "\n")
    before = mem_available_kib()
    low, stopped, health, hwm, tokens = before, "", None, -1, ""
    t0 = time.monotonic()
    done = []  # (status, rusage) once the server has exited; reaped here so that its rusage is kept

    def exited(block=False):
        if not done:
            pid, st, ru = os.wait4(p.pid, 0 if block else os.WNOHANG)
            if pid != 0:
                done.append((st, ru))
        return bool(done)

    with open(log, "w") as lf:
        p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env_for(arm))
        while time.monotonic() - t0 < a.load_timeout:
            if exited():
                stopped = "exited"
                break
            m = mem_available_kib()
            low = min(low, m)
            if m < a.min_avail_mb * 1024:
                stopped = f"MemAvailable {m} KiB"
                break
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{a.port}/health", timeout=2) as resp:
                    if resp.status == 200:
                        health = time.monotonic() - t0
                        break
            except Exception:
                pass
            time.sleep(0.25)
        else:
            stopped = "timeout"
        if health is not None:
            hwm = status_kib(p.pid, "VmHWM")
            try:
                d = post("/completion", {"prompt": "The capital of France is", "n_predict": 16, "temperature": 0, "top_k": 1,
                                         "seed": 1, "cache_prompt": False, "return_tokens": True}, 300)
                tokens = " ".join(str(t) for t in d.get("tokens", []))
            except Exception as e:
                tokens = f"error {e}"
            low = min(low, mem_available_kib())
        if not exited():
            p.send_signal(signal.SIGTERM)
            for _ in range(240):
                if exited():
                    break
                time.sleep(0.25)
            if not exited():
                p.kill()
        exited(block=True)
        st, ru = done[0]
        rc, maxrss = os.waitstatus_to_exitcode(st), ru.ru_maxrss
    r = {"phase": "load", "rep": 1, "arm": arm, "rc": rc, "wall_s": f"{time.monotonic() - t0:.3f}", "maxrss_kib": maxrss,
         "vmhwm_health_kib": hwm, "health_s": "" if health is None else f"{health:.3f}", "memavail_before_kib": before,
         "memavail_min_kib": low, "stopped": stopped, "reply_tokens": tokens}
    r.update(read_log(log))
    rows.append(r)
    save()
    time.sleep(10)  # the cards and the host settle before the next process


for rep in range(1, a.reps + 1):
    for arm in (("unset", "probe") if rep % 2 else ("probe", "unset")):
        run_list(rep, arm)
for arm in a.loads.split():
    run_load(arm)

# summary: each run by value, nothing averaged
with open(os.path.join(a.out, "summary.md"), "w") as f:
    f.write("| phase | rep | arm | rc | wall s | peak RSS KiB | probe | open+init ms | close ms | RSS after init MiB | "
            "after close MiB | found s | device 0 s | /health s | VmHWM at /health KiB | MemAvailable before / lowest KiB | "
            "stopped | reply tokens |\n|" + "---|" * 18 + "\n")
    for r in rows:
        f.write(f"| {r['phase']} | {r['rep']} | {r['arm']} | {r['rc']} | {r['wall_s']} | {r['maxrss_kib']} | {r['probe']} | "
                f"{r['open_init_ms']} | {r['close_ms']} | {r['rss_init_mib']} | {r['rss_close_mib']} | {r['t_found_s']} | "
                f"{r['t_device0_s']} | {r.get('health_s', '')} | {r.get('vmhwm_health_kib', '')} | "
                f"{r.get('memavail_before_kib', '')} / {r.get('memavail_min_kib', '')} | {r.get('stopped', '')} | "
                f"{r.get('reply_tokens', '')} |\n")
    loads = [r for r in rows if r["phase"] == "load"]
    if len(loads) == 2:
        same = loads[0]["reply_tokens"] == loads[1]["reply_tokens"] and loads[0]["reply_tokens"] != ""
        f.write(f"\nThe two loads' greedy replies: {'the same token ids' if same else 'NOT the same'}.\n")
print(open(os.path.join(a.out, "summary.md")).read())
