# 2026-10-24 estimate: the GPTQ payoff model for K2h's body in docs/research/notes/gptq_body.md ("Expected gain
# (est.)"): Q77's per-group KLD costs of Q8_0 -> imatrix round-to-nearest Q5_0 on G3, ms saved per MB at the Pascal
# kernels' per-format rates, and the groups a recovery share r lets K2h drop at KLD 0.109 and 0.117. No input files.
# Payoff model for GPTQ on K2h's body (est.). Costs: Q77 per-group KLD of Q8_0 -> imatrix-RTN Q5_0 on G3.
MS_PER_MB_8to5 = (1.0625/150 - 0.6875/138) / 0.375   # ms per MB saved (ps/w / B/w) -> 0.0056
MS_PER_MB_5to41 = (0.6875/138 - 0.625/145) / 0.0625    # 0.0107
print(f'ms per MB: Q8->Q5 {MS_PER_MB_8to5:.4f}, Q5->Q4_1 {MS_PER_MB_5to41:.4f}')
groups = [  # name, MB saved, RTN KLD cost, ms per MB
 ('q9-16',25.2,.0050,MS_PER_MB_8to5),('q17-24',25.2,.0039,MS_PER_MB_8to5),('o0-8',28.3,.0105,MS_PER_MB_8to5),
 ('o9-16',25.2,.0090,MS_PER_MB_8to5),('o17-24',25.2,.0079,MS_PER_MB_8to5),('o25-32',25.2,.0030,MS_PER_MB_8to5),
 ('kv0-8',7.1,.0064,MS_PER_MB_8to5),('kv9-16',6.3,.0070,MS_PER_MB_8to5),('kv17-24',6.3,.0031,MS_PER_MB_8to5),
 ('kv25-32',6.3,.0014,MS_PER_MB_8to5),('sh2-8',16.5,.0073,MS_PER_MB_8to5),('sh9-16',16.5,.0071,MS_PER_MB_8to5),
 ('sh17-24',18.9,.0046,MS_PER_MB_8to5),('sh25-32',18.9,.0021,MS_PER_MB_8to5),('dense0-1',28.3,.0072,MS_PER_MB_8to5),
 ('gu2-8',33.0,.0060,MS_PER_MB_8to5),
 ('gu9-48->Q4_1',31.5,.022,MS_PER_MB_5to41),('down2-45->Q4_1',17.3,.009,MS_PER_MB_5to41),
]
print('RTN KLD per ms saved:')
for g in sorted(groups, key=lambda g: g[2]/(g[1]*g[3])):
    print(f'  {g[0]:16s} {g[1]:5.1f} MB {g[1]*g[3]:.3f} ms  {g[2]/(g[1]*g[3]):.3f}/ms')
E5 = 0.027   # est. excess of K2h's existing Q5_0 body tensors over Q8_0
T0 = 26.5    # est. K2h ms per token on 6 cards
K2H, BYTES = 0.109, 2.139
for r in (0.0, 0.15, 0.4, 0.65):
    base = K2H - r*E5
    for target in (0.109, 0.117):
        kld, mb, ms, used = base, 0, 0, []
        for g in sorted(groups, key=lambda g: g[2]/(g[1]*g[3])):
            c = (1-r)*g[2]
            if kld + c <= target + 1e-9:
                kld += c; mb += g[1]; ms += g[1]*g[3]; used.append(g[0])
        print(f'r={r:.2f} target {target}: base {base:.4f} -> {kld:.4f}; -{mb:.0f} MB ({BYTES-mb/1000:.3f} GB/token), -{ms:.2f} ms, +{100*(T0/(T0-ms)-1):.1f} % tok/s; groups {used}')
