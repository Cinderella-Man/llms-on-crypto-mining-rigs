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
# Extra windows a head is scored on without training on them. Nine-card staircases on card 1 after its stage, with
# each card boundary mapped to the nearest lower dumped layer (card 1's own output, layer 6, maps to the embedding):
# 6/6/6/6/5/5/5/5/5 (cycle 5.34 ms) and 6/6/6/6/6/6/6/5/2 (cycle 3.82 ms), from the stage times in section 1.
NINE = ["emb", "L9", "L17", "L33", "L41"]
NINE_LIGHT = ["emb", "L9", "L17", "L25", "L33", "L41", "L41"]
SCORE_EXTRA = {"A": {"pre": ["emb", "L9", "L25", "L33", "L41"], "late": ["emb", "L17", "L33", "L41"],
                     "nine": NINE, "nine_light": NINE_LIGHT, "miss_emb": ["emb"]},
               "B": {"stair": ["L9", "L17", "L33", "L41"], "nine": NINE, "nine_light": NINE_LIGHT}}


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
    def __init__(self, d, n_head, n_kv, head_dim, ffn):
        super().__init__()
        self.n_head, self.n_kv, self.hd = n_head, n_kv, head_dim
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
        return x + self.w2(F.silu(self.w1(h)) * self.w3(h))

    def committed(self, x, cos, sin):
        """Causal self-attention over the committed rows x [N, d]; returns the new rows and their keys/values."""
        q, k, v = self.qkv(x, cos, sin)
        rep = self.n_head // self.n_kv
        o = F.scaled_dot_product_attention(q.transpose(0, 1)[None], k.repeat_interleave(rep, 1).transpose(0, 1)[None],
                                           v.repeat_interleave(rep, 1).transpose(0, 1)[None], is_causal=True)
        o = o[0].transpose(0, 1).reshape(x.shape[0], -1)
        return self.mlp(x + self.wo(o)), k, v

    def window(self, xw, cos, sin, kc, vc, limit, chunk=256):
        """Window rows xw [T, W, d] (offset 0 = newest) of T targets. Row (t, j) attends to committed keys at positions
        <= limit[t] and to the window rows (t, j') with j' >= j (older or itself)."""
        q, k, v = self.qkv(xw, cos, sin)                     # [T, W, H, D], [T, W, Hkv, D]
        T, W = xw.shape[:2]
        rep = self.n_head // self.n_kv
        outs = []
        for t0 in range(0, T, chunk):
            t1 = min(T, t0 + chunk)
            fn = lambda q_, k_, v_, lim, kc_, vc_: _window_attn(q_, k_, v_, lim, kc_, vc_, rep)   # noqa: E731
            if torch.is_grad_enabled():
                o = checkpoint(fn, q[t0:t1], k[t0:t1], v[t0:t1], limit[t0:t1], kc, vc, use_reentrant=False)
            else:
                o = fn(q[t0:t1], k[t0:t1], v[t0:t1], limit[t0:t1], kc, vc)
            outs.append(o)
        o = torch.cat(outs, 0).reshape(T, W, -1)
        return self.mlp(xw + self.wo(o))


def _window_attn(q, k, v, limit, kc, vc, rep):
    T, W, H, D = q.shape
    N = kc.shape[0]
    scale = 1.0 / math.sqrt(D)
    kcr = kc.repeat_interleave(rep, 1)                       # [N, H, D]
    vcr = vc.repeat_interleave(rep, 1)
    s_c = torch.einsum("twhd,nhd->thwn", q, kcr) * scale     # [T, H, W, N]
    pos = torch.arange(N, device=q.device)
    ok_c = pos[None, :] <= limit[:, None]                    # [T, N]
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
    def __init__(self, emb, head_rows, norm_w, n_layer=2, n_head=16, n_kv=4, head_dim=128, ffn=4096):
        """emb: Flint's input embedding table [n_vocab, d] (frozen); head_rows: Flint's output-head rows for the draft
        vocabulary [V, d] (an "other" class is added); norm_w: Flint's output-norm weight."""
        super().__init__()
        d = emb.shape[1]
        self.register_buffer("emb", emb, persistent=False)
        self.emb_norm = RMSNorm(d)
        self.w_emb = nn.Linear(d, d, bias=False)
        self.anc_norm = nn.ModuleList(RMSNorm(d) for _ in range(6))
        self.w_anc = nn.ModuleList(nn.Linear(d, d, bias=False) for _ in range(6))
        self.anchor_emb = nn.Embedding(7, d)
        nn.init.zeros_(self.anchor_emb.weight)
        self.blocks = nn.ModuleList(Block(d, n_head, n_kv, head_dim, ffn) for _ in range(n_layer))
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

    def rows(self, tokens, anchor, h):
        """Input rows: tokens [...], anchor ids [...] (0..6), h [..., d] the anchor's hidden state (ignored for 0)."""
        x = self.w_emb(self.emb_norm(self.emb[tokens].to(h.dtype)))
        flat_a, flat_h = anchor.reshape(-1), h.reshape(-1, h.shape[-1])
        add = torch.zeros_like(flat_h)
        for a in range(1, 7):
            idx = (flat_a == a).nonzero(as_tuple=True)[0]
            if idx.numel():
                add.index_copy_(0, idx, self.w_anc[a - 1](self.anc_norm[a - 1](flat_h[idx])))
        return x + add.view_as(x) + self.anchor_emb(anchor).to(h.dtype)

    def forward(self, tokens, committed_h, committed_anchor, windows, return_feats=False):
        """tokens [N]; committed_h [N, d]; committed_anchor: int; windows: {name: (target positions [T],
        anchors [W], window hidden states [T, W, d])}. Returns {name: logits [T, V+1]} of each window's newest row
        (for a plain window, of the committed rows at the target positions)."""
        N = tokens.shape[0]
        dev = committed_h.device
        pos = torch.arange(N, device=dev)
        xc = self.rows(tokens, torch.full((N,), committed_anchor, device=dev, dtype=torch.long), committed_h)
        cos_c, sin_c = rope_tables(pos, self.blocks[0].hd)
        state = {}
        for name, (tpos, anchors, hw) in windows.items():
            if hw is None:
                state[name] = None
                continue
            W = len(anchors)
            wpos = tpos[:, None] - torch.arange(W, device=dev)[None, :]
            a = torch.tensor([ANCHOR_IDS[x] for x in anchors], device=dev).expand(tpos.shape[0], W)
            cos, sin = rope_tables(wpos, self.blocks[0].hd)
            state[name] = [self.rows(tokens[wpos], a, hw), cos, sin, tpos - W]
        for blk in self.blocks:
            xc_new, kc, vc = blk.committed(xc, cos_c, sin_c)
            for name, st in state.items():
                if st is not None:
                    st[0] = blk.window(st[0], st[1], st[2], kc, vc, st[3])
            xc = xc_new
        out = {}
        for name, (tpos, anchors, hw) in windows.items():
            h = xc[tpos] if state[name] is None else state[name][0][:, 0]
            f = self.norm(h)
            out[name] = (self.head(f).float(), f) if return_feats else self.head(f).float()
        return out


def feature_loss(f, target):
    """EAGLE-style regression of the newest row's normalized output onto Flint's final hidden state of the same token
    (which Flint's own head turns into the next-token distribution), relative to the target's mean square."""
    f, target = f.float(), target.float()
    return ((f - target) ** 2).mean() / (target ** 2).mean().clamp_min(1e-8)


def kl_loss(logits, p_draft, sampled, ce_weight=0.2):
    """Forward KL from Flint's kept distribution (p_draft [T, k] over draft ids in sampled-order columns, given as
    (ids [T, k], probs [T, k])) to the guesser, plus cross-entropy on the sampled token's draft id."""
    ids, probs = p_draft
    logq = torch.log_softmax(logits, -1)
    lq = logq.gather(1, ids)
    kl = (probs * (torch.log(probs.clamp_min(1e-12)) - lq)).sum(-1)
    ce = -logq.gather(1, sampled[:, None])[:, 0]
    return kl.mean() + ce_weight * ce.mean(), kl.mean().detach(), ce.mean().detach()


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
