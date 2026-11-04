// Viterbi encoder of a tail-biting bitshift trellis, for the Pascal trellis proxy (registry Q929). CPU only.
//
// A sequence of T steps carries V weights per step. Step t's state s_t is L bits: s_t = ((s_{t-1} << kV) | b_t) mod 2^L,
// b_t the kV new bits of step t, so the top M = L - kV bits of s_t are the low M bits of s_{t-1}. The decoded value of
// weight v of step t is d * tab[v][s_t] (tab: V arrays of 2^L floats, built by the caller from the code's codebook and
// index function). Tail-biting: s_0's top M bits are s_{T-1}'s low M bits, so no start state is stored. A block has Q
// sequences that share one scale d.
//
// The search minimizes sum_i w_i (x_i - d c_i)^2. Tail-biting follows QTIP: one free pass over the sequence rotated by
// T/2 fixes the M bits at the wrap, then a pass constrained to them. The scale: for each g of a grid, d = g * rms(x),
// keep the best; then least-squares refits of d on the chosen path, each followed by a new search (refine).
//
//   gcc -O3 -march=x86-64-v3 -fPIC -shared -o libptq.so ptq.c -lm
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define PTQ_INF 1e30f

typedef struct {
    int L, kV, V, T;
    const float * tab;
    float * cost, * cnew, * err, * m;
    uint16_t * bp;
} ptq_ctx;

// One pass in the order t = (r + i) % T; start < 0: free start and end, else the wrap bits fixed to start.
static float ptq_pass(ptq_ctx * c, const float * x, const float * w, float d, int start, int r, uint32_t * st) {
    const int L = c->L, kV = c->kV, V = c->V, T = c->T;
    const int S = 1 << L, M = L - kV, U = 1 << M, B = 1 << kV;
    float * cost = c->cost, * cnew = c->cnew, * err = c->err, * m = c->m;
    for (int i = 0; i < T; i++) {
        const int t = (r + i) % T;
        {
            const float xv = x[t*V], wv = w[t*V];
            const float * tv = c->tab;
            for (int s = 0; s < S; s++) { const float e = xv - d*tv[s]; err[s] = wv*e*e; }
        }
        for (int v = 1; v < V; v++) {
            const float xv = x[t*V + v], wv = w[t*V + v];
            const float * tv = c->tab + (size_t) v*S;
            for (int s = 0; s < S; s++) { const float e = xv - d*tv[s]; err[s] += wv*e*e; }
        }
        if (i == 0) {
            if (start < 0) {
                memcpy(cost, err, S*sizeof(float));
            } else {
                for (int s = 0; s < S; s++) cost[s] = (s >> kV) == start ? err[s] : PTQ_INF;
            }
            continue;
        }
        uint16_t * bpi = c->bp + (size_t) i*U;
        memcpy(m, cost, U*sizeof(float));
        memset(bpi, 0, U*sizeof(uint16_t));
        for (int hi = 1; hi < B; hi++) {
            const float * cc = cost + ((size_t) hi << M);
            for (int u = 0; u < U; u++) {
                const int lt = cc[u] < m[u];
                m[u]   = lt ? cc[u] : m[u];
                bpi[u] = lt ? (uint16_t) hi : bpi[u];
            }
        }
        for (int u = 0; u < U; u++) {
            const float mu = m[u];
            float * cn = cnew + ((size_t) u << kV);
            const float * e = err + ((size_t) u << kV);
            for (int b = 0; b < B; b++) cn[b] = e[b] + mu;
        }
        float * tmp = cost; cost = cnew; cnew = tmp;
    }
    float best = PTQ_INF * 4;
    int bs = 0;
    for (int s = 0; s < S; s++) {
        if (start >= 0 && (s & (U - 1)) != start) continue;
        if (cost[s] < best) { best = cost[s]; bs = s; }
    }
    uint32_t s = (uint32_t) bs;
    st[(r + T - 1) % T] = s;
    for (int i = T - 1; i >= 1; i--) {
        const uint32_t u  = s >> kV;
        const uint32_t hi = c->bp[(size_t) i*U + u];
        s = (hi << M) | u;
        st[(r + i - 1) % T] = s;
    }
    return best;
}

static float ptq_tailbiting(ptq_ctx * c, const float * x, const float * w, float d, uint32_t * st) {
    const int U = 1 << (c->L - c->kV);
    ptq_pass(c, x, w, d, -1, c->T/2, st);
    const int ustar = (int) (st[c->T - 1] & (uint32_t) (U - 1));
    return ptq_pass(c, x, w, d, ustar, 0, st);
}

// Least-squares scale and weighted error of the path st over the n = Q*T*V values of a block.
static void ptq_fit(const ptq_ctx * c, int Q, const float * x, const float * w, const uint32_t * st, float * d, float * e) {
    const int S = 1 << c->L, V = c->V, T = c->T;
    double sxc = 0, scc = 0, sxx = 0;
    for (int q = 0; q < Q; q++) {
        for (int t = 0; t < T; t++) {
            const uint32_t s = st[q*T + t];
            for (int v = 0; v < V; v++) {
                const int i = (q*T + t)*V + v;
                const double cv = c->tab[(size_t) v*S + s];
                sxc += w[i]*x[i]*cv; scc += w[i]*cv*cv; sxx += w[i]*x[i]*x[i];
            }
        }
    }
    *d = scc > 0 ? (float) (sxc/scc) : 0.0f;
    *e = (float) (sxx - 2*(*d)*sxc + (double) (*d)*(*d)*scc);
}

// Encode nb blocks of Q sequences of T steps of V values (x, w in sequence order, nb*Q*T*V each).
// Out: d_out[nb] (the scale, before any FP16 rounding), st_out[nb*Q*T] (states), cv_out[nb*Q*T*V] (tab values
// along the path, unit scale). Returns 0, or -1 if memory runs out.
int ptq_encode(int L, int kV, int V, int T, int Q, const float * tab, int nb, const float * x, const float * w,
               const float * grid, int ng, int refine, float * d_out, uint32_t * st_out, float * cv_out) {
    const int S = 1 << L, U = 1 << (L - kV), n = Q*T*V;
    ptq_ctx c = { L, kV, V, T, tab, NULL, NULL, NULL, NULL, NULL };
    c.cost = (float *) malloc(S*sizeof(float));
    c.cnew = (float *) malloc(S*sizeof(float));
    c.err  = (float *) malloc(S*sizeof(float));
    c.m    = (float *) malloc(U*sizeof(float));
    c.bp   = (uint16_t *) malloc((size_t) T*U*sizeof(uint16_t));
    uint32_t * tst = (uint32_t *) malloc((size_t) Q*T*sizeof(uint32_t));
    if (!c.cost || !c.cnew || !c.err || !c.m || !c.bp || !tst) return -1;
    for (int b = 0; b < nb; b++) {
        const float * xb = x + (size_t) b*n;
        const float * wb = w + (size_t) b*n;
        uint32_t * sb = st_out + (size_t) b*Q*T;
        double ss = 0;
        for (int i = 0; i < n; i++) ss += (double) xb[i]*xb[i];
        const float rms = (float) sqrt(ss/n);
        float bd = 0.0f, be = PTQ_INF;
        if (rms == 0.0f) {
            memset(sb, 0, (size_t) Q*T*sizeof(uint32_t));
        } else {
            float bcost = PTQ_INF * 4;
            for (int g = 0; g < ng; g++) {
                const float d = rms*grid[g];
                float cost = 0;
                for (int q = 0; q < Q; q++) cost += ptq_tailbiting(&c, xb + q*T*V, wb + q*T*V, d, tst + q*T);
                if (cost < bcost) { bcost = cost; memcpy(sb, tst, (size_t) Q*T*sizeof(uint32_t)); }
            }
            ptq_fit(&c, Q, xb, wb, sb, &bd, &be);
            for (int it = 0; it < refine && bd > 0; it++) {
                for (int q = 0; q < Q; q++) ptq_tailbiting(&c, xb + q*T*V, wb + q*T*V, bd, tst + q*T);
                float d2, e2;
                ptq_fit(&c, Q, xb, wb, tst, &d2, &e2);
                if (e2 < be) { be = e2; bd = d2; memcpy(sb, tst, (size_t) Q*T*sizeof(uint32_t)); } else break;
            }
        }
        d_out[b] = bd;
        float * cb = cv_out + (size_t) b*n;
        for (int q = 0; q < Q; q++)
            for (int t = 0; t < T; t++)
                for (int v = 0; v < V; v++) cb[(q*T + t)*V + v] = tab[(size_t) v*S + sb[q*T + t]];
    }
    free(c.cost); free(c.cnew); free(c.err); free(c.m); free(c.bp); free(tst);
    return 0;
}
