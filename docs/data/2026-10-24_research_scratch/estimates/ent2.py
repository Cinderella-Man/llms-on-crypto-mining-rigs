# 2026-10-24 estimate: ent.py in pure Python (no numpy), as a cross-check.
import random, math, collections
random.seed(0)
def ent(c):
    n=sum(c.values()); return -sum(v/n*math.log2(v/n) for v in c.values())
def t4():
    # student t with 4 dof
    z=random.gauss(0,1); chi=sum(random.gauss(0,1)**2 for _ in range(4)); return z/math.sqrt(chi/4)
gens={"gauss":lambda: random.gauss(0,1),"t4":t4,"laplace":lambda: (random.expovariate(1) * (1 if random.random()<0.5 else -1))}
for name,g in gens.items():
    C={k:collections.Counter() for k in ("q8","q6","q5","q4")}
    for _ in range(20000):
        w=[g() for _ in range(32)]
        mx=max(w,key=abs); am=abs(mx)
        for x in w:
            C["q8"][round(x/am*127)]+=1
            C["q6"][min(63,max(0,round(x/(mx/-32))+32))]+=1
            C["q5"][min(31,max(0,round(x/(mx/-16))+16))]+=1
            C["q4"][min(15,max(0,round(x/(mx/-8))+8))]+=1
    print(name, " ".join("%s %.2f"%(k,ent(v)) for k,v in C.items()))
