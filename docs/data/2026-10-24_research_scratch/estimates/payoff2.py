# 2026-10-24 estimate: payoff.py with separate recovery shares for routed experts and always-read tensors (the
# mid and head-like cases in docs/research/notes/gptq_body.md). Run from this folder: it reads payoff.py's group table.
exec(open('payoff.py').read().split("E5 = 0.027")[0].replace("print(","(lambda *a,**k:None)("))
E5_exp, E5_aro = 0.016, 0.011   # est.: routed down 2-45 ~0.011 + gate/up 9-48 ~0.005; always-read Q5_0 groups ~0.011
T0, K2H, BYTES = 26.5, 0.109, 2.139
expert_groups = {'gu9-48->Q4_1', 'down2-45->Q4_1', 'gu2-8'}
for re_, ra in ((0.0,0.0),(0.15,0.25),(0.15,0.4),(0.15,0.65),(0.4,0.65)):
    base = K2H - re_*E5_exp - ra*E5_aro
    for target in (0.109, 0.117):
        kld, mb, ms, used = base, 0, 0, []
        cand = sorted(groups, key=lambda g: (1-(re_ if g[0] in expert_groups else ra))*g[2]/(g[1]*g[3]))
        for g in cand:
            c = (1-(re_ if g[0] in expert_groups else ra))*g[2]
            if kld + c <= target + 1e-9:
                kld += c; mb += g[1]; ms += g[1]*g[3]; used.append(g[0])
        print(f'r_exp={re_:.2f} r_aro={ra:.2f} target {target}: base {base:.4f} -> {kld:.4f}; -{mb:.0f} MB ({BYTES-mb/1000:.3f} GB), -{ms:.2f} ms, +{100*(T0/(T0-ms)-1):.1f} %; {used}')
