"""The staircase guesser (Q20/D1): a small transformer that guesses Flint's next token in a pipeline from, for every
earlier token, the deepest Flint hidden state that exists when the guess is made.

Rows. Each position contributes one row: Flint's (frozen) input embedding of its token, plus one projected Flint hidden
state, the deepest one available (an "anchor": the input of layer 9, 17, 25, 33 or 41, i.e. a six-card split's card
boundaries, or the final hidden state), plus an embedding of which anchor it is. A profile says which anchor each row
gets: committed positions (every token older than the guesses in flight) use one anchor, and the newest W positions (a
"window") use the anchors of a staircase, e.g. head A on card 1 after its stage: newest L9, then L17, L33, L41.

Training computes, per sequence, every committed row once (causal attention) and, for every target position, its window
rows, which attend to the committed rows older than the window and causally to each other. The newest window row
predicts the next token; the loss is the forward KL divergence to Flint's sampler distribution (top-k 20, top-p 0.95,
min-p 0 at temperature 1) over a 32k vocabulary plus one "other" class, plus cross-entropy on the sampled token.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

ANCHOR_IDS = {"emb": 0, "L9": 1, "L17": 2, "L25": 3, "L33": 4, "L41": 5, "full": 6}

# anchor per window offset (0 = the newest token), committed anchor for older tokens
PROFILES = {
    # A: on card 1 after its stage (six cards, section 3.1 of pipeline_drafter.md), plus the after-miss case
    "A": {"committed": "full", "windows": {"stair": ["L9", "L17", "L33", "L41"], "miss": ["L9"]}},
    # B: on a spare card before the cycle (pre-step) and late in the cycle, plus the after-miss case
    "B": {"committed": "full", "windows": {"pre": ["emb", "L9", "L25", "L33", "L41"], "late": ["emb", "L17", "L33", "L41"],
                                           "miss": ["emb"]}},
    # U: every older token at full depth, the newest at card 1's output (A's inputs after a miss)
    "U": {"committed": "full", "windows": {"miss": ["L9"]}},
    # C: card 1's output for every token, nothing from other cards
    "C": {"committed": "L9", "windows": {"plain": ["L9"]}},
    # T: tokens only
    "T": {"committed": "emb", "windows": {"plain": ["emb"]}},
}
SIX_CARD_LAYERS = (9, 17, 25, 33, 41)
N_LAYERS = 49
FULL = 49   # anchor index of the final state; 0 is the embedding alone, n the input of layer n


def anchor_index(name):
    """Depth of an anchor name: 0 for "emb", n for "Ln" (the input of layer n), 49 for "full"."""
    return 0 if name == "emb" else FULL if name == "full" else int(name[1:])


# Depth buckets of the general head: layer inputs 1-12, 13-20, 21-28, 29-36 and 37-48 share the projection of the
# six-card anchor inside them (L9, L17, L25, L33, L41), the final state keeps its own; a per-depth embedding tells the
# head which layer each input comes from.
BUCKET = [-1] + [0] * 12 + [1] * 8 + [2] * 8 + [3] * 8 + [4] * 12 + [5]


def split_depths(split, kind="stair", layer_ms=None, head_ms=2.5, hop_ms=0.3, max_w=12):
    """Depths (layers done: the input of the next layer; 0 = not yet on card 1) of the tokens in flight for a layer split
    (layers per card, the output head on the last card), newest first, up to the first complete token. With every card
    busy a token enters card 1 once per cycle (slowest stage plus a hop), so the token m cycles older has finished every
    card that ends within m cycles. kind "stair": a head on card 1 right after card 1's stage; "pre": a spare card at the
    start of a cycle; "miss": the first guess after a miss (newest token through card 1, every older one complete).
    Per-layer time as in Q87's simulator: 21.3 ms / 49 layers on six cards, 24.9 ms on nine, extrapolated."""
    n = len(split)
    if layer_ms is None:
        layer_ms = max(0.3, 0.435 + (n - 6) * 0.024)
    stages = [layer_ms * x for x in split]
    stages[-1] += head_ms
    cycle = max(stages) + hop_ms
    ends, t = [], 0.0
    for s in stages:
        t += s
        ends.append(t)
        t += hop_ms
    done = [sum(split[:k + 1]) for k in range(n)]
    if kind == "miss":
        return [done[0]]
    t0 = stages[0] if kind == "stair" else 0.0
    out = []
    for m in range(max_w):
        ks = [j for j in range(n) if ends[j] <= t0 + m * cycle + 1e-9]
        if ks and ks[-1] == n - 1:
            break
        out.append(done[ks[-1]] if ks else 0)
    return out


def depth_name(depth, recorded):
    """The deepest recorded anchor at or below a depth: "full" for a complete token, "Ln" for the deepest recorded layer
    input n <= depth, "emb" if none."""
    if depth >= FULL:
        return "full"
    below = [x for x in recorded if x <= depth]
    return f"L{max(below)}" if below else "emb"


def even_split(n, light_last=0):
    """Layers per card for n cards, as even as possible; light_last > 0 gives the last card (which holds the output
    head) only that many layers."""
    rest, cards = (N_LAYERS - light_last, n - 1) if light_last else (N_LAYERS, n)
    split = [rest // cards] * cards
    for i in range(rest - sum(split)):
        split[i] += 1
    return split + ([light_last] if light_last else [])


def staircase(split, layer_ms=None, head_ms=2.5, hop_ms=0.3, max_w=8):
    """The window of a head on card 1 right after card 1's stage, for a layer split, newest token first, each depth
    mapped to the nearest lower six-card anchor (the embedding below layer 9)."""
    return [depth_name(x, SIX_CARD_LAYERS) for x in split_depths(split, "stair", layer_ms, head_ms, hop_ms, max_w)]


# Layer splits an owner may run (layers per card; the output head on the last card) and their card-1 staircases.
SPLITS = {"c5": [10, 10, 10, 10, 9], "c6": [9, 8, 8, 8, 8, 8], "c7": [7] * 7, "c8": [7, 6, 6, 6, 6, 6, 6, 6],
          "c9": [6, 6, 6, 6, 5, 5, 5, 5, 5], "c9light": [6, 6, 6, 6, 6, 6, 6, 5, 2]}
SPLIT_WINDOWS = {name: staircase(split) for name, split in SPLITS.items()}
# Extra windows a head is scored on without training on them: the splits' card-1 staircases, a spare card's pre-step
# and late windows, and the after-miss window with the newest token as its embedding (card 1 below layer 9).
SCORE_EXTRA = {"A": {"pre": ["emb", "L9", "L25", "L33", "L41"], "late": ["emb", "L17", "L33", "L41"],
                     "miss_emb": ["emb"], **SPLIT_WINDOWS},
               "B": {"stair": ["L9", "L17", "L33", "L41"], **SPLIT_WINDOWS}}


def rope_tables(pos, dim, theta=1e5):
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, device=pos.device, dtype=torch.float32) / dim))
    ang = pos.float()[..., None] * inv
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):
    # x [..., L, H, D]; cos/sin [..., L, D/2]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c, s = cos[..., None, :], sin[..., None, :]
    out = torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), -1)
    return out.flatten(-2).type_as(x)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight).type_as(x)


class Block(nn.Module):
    def __init__(self, d, n_head, n_kv, head_dim, ffn, dropout=0.0):
        super().__init__()
        self.n_head, self.n_kv, self.hd = n_head, n_kv, head_dim
        self.drop = nn.Dropout(dropout)
        self.attn_norm = RMSNorm(d)
        self.wq = nn.Linear(d, n_head * head_dim, bias=False)
        self.wk = nn.Linear(d, n_kv * head_dim, bias=False)
        self.wv = nn.Linear(d, n_kv * head_dim, bias=False)
        self.wo = nn.Linear(n_head * head_dim, d, bias=False)
        self.mlp_norm = RMSNorm(d)
        self.w1 = nn.Linear(d, ffn, bias=False)
        self.w3 = nn.Linear(d, ffn, bias=False)
        self.w2 = nn.Linear(ffn, d, bias=False)
        nn.init.zeros_(self.wo.weight)
        nn.init.zeros_(self.w2.weight)

    def qkv(self, x, cos, sin):
        h = self.attn_norm(x)
        L = x.shape[:-1]
        q = apply_rope(self.wq(h).view(*L, self.n_head, self.hd), cos, sin)
        k = apply_rope(self.wk(h).view(*L, self.n_kv, self.hd), cos, sin)
        v = self.wv(h).view(*L, self.n_kv, self.hd)
        return q, k, v

    def mlp(self, x):
        h = self.mlp_norm(x)
        return x + self.drop(self.w2(F.silu(self.w1(h)) * self.w3(h)))

    def committed(self, x, cos, sin, ctx=0):
        """Causal self-attention over the committed rows x [N, d]; returns the new rows and their keys/values. ctx > 0:
        each row attends to the last ctx rows only (itself included), the engine's ring of committed positions
        (--pipe-draft-ctx), computed in blocks of ctx rows."""
        q, k, v = self.qkv(x, cos, sin)
        rep = self.n_head // self.n_kv
        N = x.shape[0]
        qt = q.transpose(0, 1)[None]
        kt, vt = k.repeat_interleave(rep, 1).transpose(0, 1)[None], v.repeat_interleave(rep, 1).transpose(0, 1)[None]
        if not ctx or N <= ctx:
            o = F.scaled_dot_product_attention(qt, kt, vt, is_causal=True)
        else:
            outs = []
            for s in range(0, N, ctx):
                e, k0 = min(N, s + ctx), max(0, s - ctx + 1)
                qp = torch.arange(s, e, device=x.device)[:, None]
                kp = torch.arange(k0, e, device=x.device)[None, :]
                outs.append(F.scaled_dot_product_attention(qt[:, :, s:e], kt[:, :, k0:e], vt[:, :, k0:e],
                                                           attn_mask=(kp <= qp) & (kp > qp - ctx)))
            o = torch.cat(outs, 2)
        o = o[0].transpose(0, 1).reshape(N, -1)
        return self.mlp(x + self.drop(self.wo(o))), k, v

    def window(self, xw, cos, sin, kc, vc, limit, chunk=256, ctx=0):
        """Window rows xw [T, W, d] (offset 0 = newest) of T targets. Row (t, j) attends to committed keys at positions
        <= limit[t] (and > limit[t] - ctx with ctx > 0) and to the window rows (t, j') with j' >= j (older or
        itself). A chunk's scores take chunk x keys x W x heads values, so long sequences get smaller chunks."""
        q, k, v = self.qkv(xw, cos, sin)                     # [T, W, H, D], [T, W, Hkv, D]
        T, W = xw.shape[:2]
        rep = self.n_head // self.n_kv
        span = min(kc.shape[0], ctx) if ctx else kc.shape[0]
        chunk = max(16, min(chunk, (256 * 8192) // max(1, span)))
        outs = []
        lim_all = limit.tolist()
        for t0 in range(0, T, chunk):
            t1 = min(T, t0 + chunk)
            n_keys = max(1, max(lim_all[t0:t1]) + 1) if SLICE_KEYS else kc.shape[0]   # later keys are masked anyway
            k0 = max(0, min(lim_all[t0:t1]) - ctx + 1) if ctx and SLICE_KEYS else 0   # and so are older ones
            kc_, vc_ = kc[k0:n_keys], vc[k0:n_keys]
            fn = lambda q_, k_, v_, lim, kc_, vc_: _window_attn(q_, k_, v_, lim, kc_, vc_, rep, k0, ctx)   # noqa: E731
            if torch.is_grad_enabled():
                o = checkpoint(fn, q[t0:t1], k[t0:t1], v[t0:t1], limit[t0:t1], kc_, vc_, use_reentrant=False)
            else:
                o = fn(q[t0:t1], k[t0:t1], v[t0:t1], limit[t0:t1], kc_, vc_)
            outs.append(o)
        o = torch.cat(outs, 0).reshape(T, W, -1)
        return self.mlp(xw + self.drop(self.wo(o)))


SLICE_KEYS = True   # window attention reads committed keys only up to each chunk's last limit (check_train.py compares)


def _window_attn(q, k, v, limit, kc, vc, rep, k0=0, ctx=0):
    """kc, vc: the committed keys and values from position k0 on."""
    T, W, H, D = q.shape
    N = kc.shape[0]
    scale = 1.0 / math.sqrt(D)
    kcr = kc.repeat_interleave(rep, 1)                       # [N, H, D]
    vcr = vc.repeat_interleave(rep, 1)
    s_c = torch.einsum("twhd,nhd->thwn", q, kcr) * scale     # [T, H, W, N]
    pos = torch.arange(k0, k0 + N, device=q.device)
    ok_c = pos[None, :] <= limit[:, None]                    # [T, N]
    if ctx:
        ok_c &= pos[None, :] > limit[:, None] - ctx
    s_c = s_c.masked_fill(~ok_c[:, None, None, :], float("-inf"))
    kw = k.repeat_interleave(rep, 2)                         # [T, W, H, D]
    vw = v.repeat_interleave(rep, 2)
    s_w = torch.einsum("tjhd,tkhd->thjk", q, kw) * scale     # [T, H, W, W]
    j = torch.arange(W, device=q.device)
    ok_w = j[None, :] >= j[:, None]                          # key offset >= query offset
    s_w = s_w.masked_fill(~ok_w[None, None], float("-inf"))
    p = torch.softmax(torch.cat((s_c, s_w), -1).float(), -1).type_as(q)
    o = torch.einsum("thwn,nhd->twhd", p[..., :N], vcr) + torch.einsum("thjk,tkhd->tjhd", p[..., N:], vw)
    return o.reshape(T, W, H * D)


class Drafter(nn.Module):
    def __init__(self, emb, head_rows, norm_w, n_layer=2, n_head=16, n_kv=4, head_dim=128, ffn=4096, fuse=False,
                 noise=0.0, dropout=0.0, general=False):
        """emb: Flint's input embedding table [n_vocab, d] (frozen); head_rows: Flint's output-head rows for the draft
        vocabulary [V, d] (an "other" class is added); norm_w: Flint's output-norm weight. fuse: every row also gets
        the shallower anchors its token has reached (zero-initialized projections, so a warm start is unchanged);
        noise: uniform noise of this amplitude after each anchor's RMS norm while training; dropout: after attention
        and the FFN. general: anchors are any layer input (anchor ids = depths 0..49, see anchor_index), projected by the
        six-card projection of their depth bucket plus a per-depth embedding (init_general copies a six-card head into
        it)."""
        super().__init__()
        d = emb.shape[1]
        self.fuse, self.noise, self.general = fuse, noise, general
        if general:
            self.depth_emb = nn.Embedding(FULL + 1, d)
            nn.init.zeros_(self.depth_emb.weight)
            self.register_buffer("bucket", torch.tensor(BUCKET), persistent=False)
        if fuse:   # L9, L17, L25, L33, L41
            self.fuse_norm = nn.ModuleList(RMSNorm(d) for _ in range(5))
            self.w_fuse = nn.ModuleList(nn.Linear(d, d, bias=False) for _ in range(5))
            for w in self.w_fuse:
                nn.init.zeros_(w.weight)
        self.register_buffer("emb", emb, persistent=False)
        self.emb_norm = RMSNorm(d)
        self.w_emb = nn.Linear(d, d, bias=False)
        self.anc_norm = nn.ModuleList(RMSNorm(d) for _ in range(6))
        self.w_anc = nn.ModuleList(nn.Linear(d, d, bias=False) for _ in range(6))
        self.anchor_emb = nn.Embedding(7, d)
        nn.init.zeros_(self.anchor_emb.weight)
        self.blocks = nn.ModuleList(Block(d, n_head, n_kv, head_dim, ffn, dropout) for _ in range(n_layer))
        self.norm = RMSNorm(d)
        # Initialization: with the blocks' output projections at zero, a row's prediction starts as Flint's output head
        # on its anchor. Layer inputs are residual-stream states, so an identity projection gives the logit lens of
        # that layer; the final hidden state is dumped after Flint's output norm, so its projection divides that norm's
        # weight out again and the final norm (Flint's weight) restores it: a row with the final state reproduces
        # Flint's own distribution. The embedding enters at a tenth of the anchor's scale.
        with torch.no_grad():
            self.norm.weight.copy_(norm_w)
            for w in self.w_anc[:5]:
                nn.init.eye_(w.weight)
            safe = torch.where(norm_w.abs() < 1e-3, torch.full_like(norm_w, 1e-3), norm_w)
            self.w_anc[5].weight.copy_(torch.diag(1.0 / safe))
            nn.init.eye_(self.w_emb.weight)
            self.w_emb.weight.mul_(0.1)
        self.head = nn.Linear(d, head_rows.shape[0] + 1, bias=True)
        with torch.no_grad():
            self.head.weight[:-1].copy_(head_rows)
            self.head.weight[-1].zero_()
            self.head.bias.zero_()
            self.head.bias[-1] = -4.0

    def _noisy(self, x):
        if self.training and self.noise > 0:
            return x + (torch.rand_like(x) * 2 - 1) * self.noise
        return x

    def rows(self, tokens, anchor, h, extra=None):
        """Input rows: tokens [...], anchor ids [...] (0..6), h [..., d] the anchor's hidden state (ignored for 0).
        extra (with fuse): [(k, h_k [..., d], mask [...])] shallower anchors (k = 0..4 for L9..L41) to add where mask."""
        x = self.w_emb(self.emb_norm(self.emb[tokens].to(h.dtype)))
        flat_a, flat_h = anchor.reshape(-1), h.reshape(-1, h.shape[-1])
        proj = self.bucket[flat_a] if self.general else flat_a - 1      # projection index, -1 for the embedding alone
        add = torch.zeros_like(flat_h)
        for p in range(6):
            idx = (proj == p).nonzero(as_tuple=True)[0]
            if idx.numel():
                add.index_copy_(0, idx, self.w_anc[p](self._noisy(self.anc_norm[p](flat_h[idx]))))
        if self.fuse and extra:
            for k, hk, mask in extra:
                idx = mask.reshape(-1).nonzero(as_tuple=True)[0]
                if idx.numel():
                    fk = self.w_fuse[k](self._noisy(self.fuse_norm[k](hk.reshape(-1, hk.shape[-1])[idx])))
                    add = add.index_add(0, idx, fk)
        depth = self.depth_emb(anchor) if self.general else self.anchor_emb(anchor)
        return x + add.view_as(x) + depth.to(h.dtype)

    def anchor_id(self, name):
        """Anchor id of a name: its depth for the general head, else the six-card anchor id."""
        return anchor_index(name) if self.general else ANCHOR_IDS[name]

    def init_general(self, sd):
        """Fill the per-depth embedding from a six-card head's anchor embedding (state dict sd): each depth gets its
        bucket's anchor row, so the general head starts as the six-card head on six-card depths. A general head's own
        state dict (with depth_emb) is left as loaded."""
        if not self.general or "anchor_emb.weight" not in sd or "depth_emb.weight" in sd:
            return
        old = sd["anchor_emb.weight"].to(self.depth_emb.weight)
        with torch.no_grad():
            for depth, b in enumerate(BUCKET):
                self.depth_emb.weight[depth] = old[0 if b < 0 else b + 1]

    def forward(self, tokens, committed_h, committed_anchor, windows, return_feats=False, committed_extra=None, ctx=0):
        """tokens [N]; committed_h [N, d]; committed_anchor: int; windows: {name: (target positions [T],
        anchors [W], window hidden states [T, W, d][, extra])}, extra as in rows() for the window rows;
        committed_extra likewise for the committed rows; ctx > 0: attend to the last ctx committed positions only.
        Returns {name: logits [T, V+1]} of each window's newest row (for a plain window, of the committed rows at the
        target positions)."""
        N = tokens.shape[0]
        dev = committed_h.device
        pos = torch.arange(N, device=dev)
        if isinstance(committed_anchor, str):
            committed_anchor = self.anchor_id(committed_anchor)
        xc = self.rows(tokens, torch.full((N,), committed_anchor, device=dev, dtype=torch.long), committed_h,
                       committed_extra)
        cos_c, sin_c = rope_tables(pos, self.blocks[0].hd)
        state = {}
        for name, win in windows.items():
            tpos, anchors, hw = win[:3]
            if hw is None:
                state[name] = None
                continue
            W = len(anchors)
            wpos = tpos[:, None] - torch.arange(W, device=dev)[None, :]
            a = torch.tensor([self.anchor_id(x) for x in anchors], device=dev).expand(tpos.shape[0], W)
            cos, sin = rope_tables(wpos, self.blocks[0].hd)
            state[name] = [self.rows(tokens[wpos], a, hw, win[3] if len(win) > 3 else None), cos, sin, tpos - W]
        for blk in self.blocks:
            xc_new, kc, vc = blk.committed(xc, cos_c, sin_c, ctx)
            for name, st in state.items():
                if st is not None:
                    st[0] = blk.window(st[0], st[1], st[2], kc, vc, st[3], ctx=ctx)
            xc = xc_new
        out = {}
        for name, win in windows.items():
            tpos = win[0]
            h = xc[tpos] if state[name] is None else state[name][0][:, 0]
            f = self.norm(h)
            out[name] = (self.head(f).float(), f) if return_feats else self.head(f).float()
        return out


def feature_loss(f, target):
    """EAGLE-style regression of the newest row's normalized output onto Flint's final hidden state of the same token
    (which Flint's own head turns into the next-token distribution), relative to the target's mean square."""
    f, target = f.float(), target.float()
    return ((f - target) ** 2).mean() / (target ** 2).mean().clamp_min(1e-8)


def _mean(x, weight):
    return x.mean() if weight is None else (x * weight).sum() / weight.sum().clamp_min(1e-6)


def kl_loss(logits, p_draft, sampled, ce_weight=0.2, weight=None):
    """Forward KL from Flint's kept distribution (p_draft [T, k] over draft ids in sampled-order columns, given as
    (ids [T, k], probs [T, k])) to the guesser, plus cross-entropy on the sampled token's draft id; weight [T]: a
    weighted mean over targets."""
    ids, probs = p_draft
    logq = torch.log_softmax(logits, -1)
    lq = logq.gather(1, ids)
    kl = (probs * (torch.log(probs.clamp_min(1e-12)) - lq)).sum(-1)
    ce = -logq.gather(1, sampled[:, None])[:, 0]
    return _mean(kl, weight) + ce_weight * _mean(ce, weight), kl.mean().detach(), ce.mean().detach()


def lk_loss(logits, p_draft, sampled, eta=3.0, ce_weight=0.2, weight=None):
    """Acceptance-matched hybrid loss (LK losses, arXiv 2602.23881): per position lambda * KL + (1 - lambda) * (1 -
    coupled match), lambda = exp(-eta * coupled) with the match held constant, so KL leads while acceptance is low and
    the coupled match itself takes over as it rises; plus cross-entropy on the sampled token."""
    ids, probs = p_draft
    logq = torch.log_softmax(logits, -1)
    kl = (probs * (torch.log(probs.clamp_min(1e-12)) - logq.gather(1, ids))).sum(-1)
    c = coupled_and_top(logits, ids, probs)[0]
    lam = torch.exp(-eta * c.detach())
    ce = -logq.gather(1, sampled[:, None])[:, 0]
    return (_mean(lam * kl + (1 - lam) * (1 - c), weight) + ce_weight * _mean(ce, weight), kl.mean().detach(),
            ce.mean().detach())


def uniform_match(logits, top_ids_draft, p_kept, alpha=1.0, top_m=16):
    """Per target: the match probability when the guess and Flint's sample share one uniform u, as llama.cpp's device
    sampler draws them: Flint takes the first of its kept tokens (descending probability) whose cumulative probability
    exceeds u; the guesser does the same over its top_m candidates of q^alpha, and takes its best real token when u
    lands in its tail or on the "other" class. Sum over tokens of the overlap of the two tokens' u-intervals."""
    q = torch.softmax(alpha * logits.float(), -1)
    other = q.shape[1] - 1
    qv, qi = q.topk(top_m, -1)                                          # [T, m], descending
    q_hi = qv.cumsum(-1)
    q_lo = q_hi - qv
    real = qi != other
    best = torch.where(real, torch.arange(top_m, device=q.device), top_m).min(-1).values.clamp(max=top_m - 1)
    best_tok = qi.gather(1, best[:, None])                              # [T, 1]
    eff = torch.where(real, qi, best_tok)                               # "other" counts for the best real token
    p_hi = p_kept.cumsum(-1)
    p_lo = p_hi - p_kept
    ov = (torch.minimum(p_hi[:, :, None], q_hi[:, None, :]) - torch.maximum(p_lo[:, :, None], q_lo[:, None, :])).clamp_min(0)
    tid = top_ids_draft
    same = (tid[:, :, None] == eff[:, None, :]) & (tid != other)[:, :, None] & (p_kept > 0)[:, :, None]
    match = (ov * same).sum((1, 2))
    tail_lo = q_hi[:, -1:]                                              # [T, 1]: u beyond the top_m mass
    tail_ov = (p_hi.clamp(max=1.0) - torch.maximum(p_lo, tail_lo)).clamp_min(0)
    match = match + (tail_ov * ((tid == best_tok) & (tid != other) & (p_kept > 0))).sum(-1)
    return match


def coupled_and_top(logits, top_ids_draft, p_kept, alpha=1.0):
    """Per target: the Gumbel-coupled match probability (pipeline_sim.coupled_match) between Flint's kept distribution
    p_kept [T, k] (zeros for dropped tokens) and the guesser's distribution over the draft vocabulary, and the
    probability that the guesser's top token is Flint's sample. top_ids_draft [T, k]: draft id of each of Flint's
    top-k tokens (the "other" id when outside the vocabulary). alpha sharpens the guesser's distribution (q^alpha,
    renormalized) before the coupled draw; the top token does not change."""
    q = torch.softmax(alpha * logits.float(), -1)
    other = q.shape[1] - 1
    in_vocab = top_ids_draft != other
    qk = q.gather(1, top_ids_draft) * in_vocab * (p_kept > 0)          # guesser mass on each kept token
    q_out = (1 - qk.sum(-1)).clamp_min(0)                               # everything else, "other" included
    pi = p_kept[:, :, None]
    pj = p_kept[:, None, :]
    qi = qk[:, :, None]
    qj = qk[:, None, :]
    valid = (pi > 0) & (qi > 0)
    ratio = torch.maximum(pj / pi.clamp_min(1e-30), qj / qi.clamp_min(1e-30)) * (pj > 0)
    s = q_out[:, None] / qk.clamp_min(1e-30) + ratio.sum(-1)
    coupled = torch.where(valid[:, :, 0], 1.0 / s, torch.zeros_like(s)).sum(-1)
    best = q.argmax(-1)
    top = (p_kept * (top_ids_draft == best[:, None]) * in_vocab).sum(-1)
    return coupled, top, q.max(-1).values
