#!/usr/bin/env python3
# A stand-in for llama-server in lightprof_rig.sh's dry run (dryrun.sh), so the rig script's logic runs on any CPU: it
# answers /health and /completion like the real server (the same JSON fields, made-up timings), and with
# GGML_CUDA_PROFILE set it writes a synthetic profile in the mode GGML_CUDA_PROFILE_MODE names, from prof_summary.py's
# self-test generator (three cards, a known token; PROF_SUMMARY_DIR names the folder holding prof_summary.py).
# The token time it reports grows with the mode by FAKE_COST (group 30 %, sample 3 %, layer 1 %, graph 0.2 %).
import hashlib, http.server, json, os, signal, sys

sys.path.insert(0, os.environ["PROF_SUMMARY_DIR"])
import prof_summary as ps  # noqa: E402

port = int(sys.argv[sys.argv.index("--port") + 1])
prof = os.environ.get("GGML_CUDA_PROFILE", "")
mode = os.environ.get("GGML_CUDA_PROFILE_MODE", "group") if prof else ""
kind = mode.split("=")[0]
cost = {"": 1.0, "graph": 1.002, "layer": 1.01, "sample": 1.03, "group": 1.30}[kind]
cards = ps.synth_truth()
if prof:
    n = int(mode.split("=")[1]) if "=" in mode else 4
    ps.synth_write(prof, kind, cards, 8.0, 40, sample_n=n, hop_us=50.0, host_us=200.0)
ms_per_token = (sum(us for g in cards for _, _, us in g) / 1000 + 0.3) * cost


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def reply(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/health":
            self.reply({"status": "ok"})
        else:
            self.send_error(404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        n = int(body.get("n_predict", 16))
        text = hashlib.sha256(f"{body.get('prompt')}|{n}".encode()).hexdigest() * (1 + n // 64)
        self.reply({"content": text, "timings": {"prompt_n": 61, "prompt_ms": 4100.0, "predicted_n": n,
                                                 "predicted_ms": n * ms_per_token,
                                                 "predicted_per_token_ms": ms_per_token}})


signal.signal(signal.SIGINT, lambda *a: sys.exit(0))
signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))
print(f"fake llama-server on 127.0.0.1:{port}, profile {prof or 'off'} {mode}", flush=True)
http.server.HTTPServer(("127.0.0.1", port), Handler).serve_forever()
