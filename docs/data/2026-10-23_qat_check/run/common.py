# Q74 (2026-10-23), the check of Flint's INT4 files: see ../README.md for the order of the steps. Env: QAT_DATA
# (fetched slices, default ../data), FLINT_INT4_GGUF (the makers' Q4_1 GGUF, read in place).
"""Loaders shared by the comparison scripts (no network access). Reads the slices fetch.py wrote under $QAT_DATA
(default: data/ next to this run/ folder) and the makers' Q4_1 GGUF named by $FLINT_INT4_GGUF."""
import json
import os

import numpy as np

QAT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("QAT_DATA", os.path.join(QAT, "data"))
GGUF_PATH = os.environ.get("FLINT_INT4_GGUF", "flint-int4-q4_1.gguf")

_logs = {}


def _log(repo):
    if repo not in _logs:
        _logs[repo] = json.load(open(os.path.join(DATA, repo, "slices", "slices.json")))
    return _logs[repo]


def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def load_st(repo, key):
    """repo: 'int4' or 'bf16'; key: '<tensor>@<i0>-<i1>' as written by fetch.py. Returns float32 or uint8 array."""
    e = _log(repo)[key]
    raw = np.fromfile(e["file"], dtype=np.uint8)
    if e["dtype"] == "BF16":
        return bf16_to_f32(raw.view("<u2")).reshape(e["shape"])
    if e["dtype"] == "U8":
        return raw.reshape(e["shape"])
    raise ValueError(e["dtype"])


def mlx_unpack4(u8):
    """MLX affine 4-bit, stored as bytes of little-endian uint32 words: element 2k = low nibble of byte k, 2k+1 = high."""
    q = np.empty(u8.shape[:-1] + (u8.shape[-1] * 2,), dtype=np.uint8)
    q[..., 0::2] = u8 & 15
    q[..., 1::2] = u8 >> 4
    return q


def mlx_int4(name, i0, i1):
    """codes [.., out, in], scale and bias [.., out, in/64] (float32, exact BF16 values)."""
    k = f"@{i0}-{i1}"
    q = mlx_unpack4(load_st("int4", name + ".weight" + k))
    return q, load_st("int4", name + ".scales" + k), load_st("int4", name + ".biases" + k)


_gidx = None


def gguf_index():
    global _gidx
    if _gidx is None:
        d = json.load(open(os.path.join(DATA, "gguf", "index.json")))
        d["by_name"] = {t["name"]: t for t in d["tensors"]}
        _gidx = d
    return _gidx


def gguf_q4_1(name, i0, i1):
    """Rows along the outermost ggml axis range [i0, i1) of a Q4_1 tensor.
    Returns codes [n, ..., ne0] (uint8), d and m [n, ..., ne0/32] as float32 from FP16."""
    g = gguf_index()
    t = g["by_name"][name]
    assert t["type"] == 3, t
    ne = t["ne"]
    inner = int(np.prod(ne[:-1]))  # elements per outermost index
    nb = inner // 32
    row_bytes = nb * 20
    with open(GGUF_PATH, "rb") as f:
        f.seek(g["data_start"] + t["offset"] + i0 * row_bytes)
        raw = np.frombuffer(f.read((i1 - i0) * row_bytes), dtype=np.uint8)
    blk = raw.reshape(-1, 20)
    d = blk[:, 0:2].copy().view("<f2").astype(np.float32)[:, 0]
    m = blk[:, 2:4].copy().view("<f2").astype(np.float32)[:, 0]
    qs = blk[:, 4:]
    q = np.empty((blk.shape[0], 32), dtype=np.uint8)
    q[:, :16] = qs & 15
    q[:, 16:] = qs >> 4
    shape = [i1 - i0] + list(reversed(ne[:-1]))  # numpy order: outer..., ne0 last
    return (q.reshape(shape), d.reshape(shape[:-1] + [ne[0] // 32]), m.reshape(shape[:-1] + [ne[0] // 32]))


def gguf_f32(name, i0, i1):
    g = gguf_index()
    t = g["by_name"][name]
    assert t["type"] == 0, t
    ne = t["ne"]
    inner = int(np.prod(ne[:-1]))
    with open(GGUF_PATH, "rb") as f:
        f.seek(g["data_start"] + t["offset"] + i0 * inner * 4)
        a = np.frombuffer(f.read((i1 - i0) * inner * 4), dtype="<f4")
    return a.reshape([i1 - i0] + list(reversed(ne[:-1])))
