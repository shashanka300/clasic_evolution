# Paired-by-seed comparison: recipe (*_swept) vs tuned stepwise + revival (*_rev_tuned). Run from repo root.
import json,collections,statistics as st,math
P={"gmm-ema":("gmm_ema_swept","online_em_rev_tuned","online_em_tuned"),
   "hmm-ema":("hmm_ema_swept","stepwise_em_rev_tuned","stepwise_em_tuned"),
   "lda-scaling":("lda_ema_swept","online_lda_rev_tuned","online_lda_tuned"),
   "dictlearn-scaling":("dl_ema_swept","stepwise_dl_rev_tuned","stepwise_dl_tuned")}
EP=[1,2,3,5,10,25]
tot=collections.Counter()
for tr,(a,b,c0) in P.items():
    d=collections.defaultdict(dict)
    for l in open(f"results/{tr}/runs.jsonl"):
        r=json.loads(l)
        if r["method"] not in (a,b,c0): continue
        cur={c["epoch"]:c["test_loss"] for c in r["extra"].get("curve") or []}
        cur[r["epochs"]]=r["test_loss"]
        d[(r["extra"]["group"],r["method"])][r["seed"]]=cur
    groups=sorted({g for g,_ in d})
    print("===",tr,"  rel diff % (recipe - rev_tuned)/rev_tuned, t-stat over seeds; neg = recipe better")
    for g in groups:
        out=[]
        for e in EP:
            A=d[(g,a)];B=d[(g,b)]
            s=[k for k in A if k in B and e in A[k] and e in B[k]]
            diffs=[A[k][e]-B[k][e] for k in s]
            m=st.mean(diffs); sd=st.stdev(diffs) if len(diffs)>1 else 0
            t=m/(sd/math.sqrt(len(diffs))) if sd>0 else float('inf')*(1 if m>0 else -1) if m!=0 else 0
            ref=st.mean(B[k][e] for k in s)
            rel=100*m/ref
            v="W" if t<-2.78 else ("L" if t>2.78 else "=")   # 4 dof, p<.05 two-sided
            tot[(tr,v)]+=1
            out.append(f"{rel:+6.2f}%{v}")
        print(f"  {g:28s} "+" ".join(out))
print("epochs",EP)
for k,v in sorted(tot.items()): print(k,v)
