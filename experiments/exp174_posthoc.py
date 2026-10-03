"""exp174_posthoc.py — POST-HOC DESCRIPTIVE readouts from exp174_rows.jsonl.
NOT registered in PREREG_exp174.md. Written after the frozen verdicts were
seen (2026-10-02). Nothing here can change a verdict. No model is loaded."""
import json, numpy as np
rows=[json.loads(l) for l in open("/Users/macn/Documents/embeddingexp/exp174_rows.jsonl")]
print("rows", len(rows))
L5=["4","8","12","16","20"]
def dz(x,y):
    d=np.array(x)-np.array(y); return d.mean()/d.std(ddof=1)
cell={}
for r in rows:
    k=(r["item"], "intact" if r["order"]=="intact" else "shuf", r["pass"])
    cell.setdefault(k,[]).append(r)
items=sorted({r["item"] for r in rows}); meta={r["item"]:(r["triple"],r["valence"]) for r in rows}
def val(it,o,p,key,L=None):
    rs=cell[(it,o,p)]
    return np.mean([r[key] if L is None else r[key][L] for r in rs])
tri={}
for it in items: tri.setdefault(meta[it][0],{})[meta[it][1]]=it
T=sorted(tri)
print("\nPOST-HOC DESCRIPTIVE (not registered). UP_literal readout, cell means x1000, pass 1")
for L in L5:
    out=[]
    for v in ("happy","neutral","sad"):
        for o in ("intact","shuf"):
            out.append(f"{v[:3]}-{o[:3]} {1000*np.mean([val(tri[t][v],o,1,'UP_literal',L) for t in T]):+7.2f}")
    print(f" L{L:>2}: "+"  ".join(out))
print("\nThe crossing Niamh named (text coherence): sad-INTACT vs happy-SCRAMBLED on UP_literal, d_z by triple (positive = the coherent sad text reads more UP)")
print("  "+"  ".join(f"L{L} {dz([val(tri[t]['sad'],'intact',1,'UP_literal',L) for t in T],[val(tri[t]['happy'],'shuf',1,'UP_literal',L) for t in T]):+.2f}" for L in L5))
print("\nPass effect on UP_literal readout (pass2 - pass1, intact), d_z over 120 sentences; and on C")
print("  UP: "+"  ".join(f"L{L} {dz([val(i,'intact',2,'UP_literal',L) for i in items],[val(i,'intact',1,'UP_literal',L) for i in items]):+.2f}" for L in L5))
print(f"  C: {dz([val(i,'intact',2,'C') for i in items],[val(i,'intact',1,'C') for i in items]):+.2f}")
print("\nWithin pass 1 only: across sentences, corr( C(intact)-C(scrambled), UP(intact)-UP(scrambled) )")
dC=np.array([val(i,'intact',1,'C')-val(i,'shuf',1,'C') for i in items])
for L in L5:
    dU=np.array([val(i,'intact',1,'UP_literal',L)-val(i,'shuf',1,'UP_literal',L) for i in items])
    print(f"  L{L}: r = {np.corrcoef(dC,dU)[0,1]:+.3f}")
print("\nScramble effect on path length and straightness, pass 1 (intact - scrambled) d_z")
print(f"  path {dz([val(i,'intact',1,'path') for i in items],[val(i,'shuf',1,'path') for i in items]):+.2f}  straight {dz([val(i,'intact',1,'straight') for i in items],[val(i,'shuf',1,'straight') for i in items]):+.2f}")
print("  norm: "+"  ".join(f"L{L} {dz([val(i,'intact',1,'norm',L) for i in items],[val(i,'shuf',1,'norm',L) for i in items]):+.2f}" for L in L5))
print("\nValence effect on C (happy vs sad, intact p1, by triple) d_z:", round(dz([val(tri[t]['happy'],'intact',1,'C') for t in T],[val(tri[t]['sad'],'intact',1,'C') for t in T]),2))
