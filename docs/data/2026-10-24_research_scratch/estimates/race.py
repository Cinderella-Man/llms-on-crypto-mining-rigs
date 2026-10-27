# 2026-10-24 estimate: the shortest of k reasoning lengths when one prompt's lengths are log-normal around their mean
# (Monte Carlo, CV between samples): the "Expected gain (est.)" table of the first-finish race in
# docs/research/notes/unconventional_ideas.md. No input files.
import random, math
random.seed(1)
for cv in (0.3,0.45,0.6,0.8):
    s2=math.log(1+cv*cv); s=math.sqrt(s2); mu=-s2/2  # mean 1
    out=[]
    for k in (2,3,4):
        n=200000; acc=0.0
        for _ in range(n):
            acc+=min(math.exp(random.gauss(mu,s)) for _ in range(k))
        out.append("k=%d %.2f"%(k,acc/n))
    print("cv",cv," ".join(out))
