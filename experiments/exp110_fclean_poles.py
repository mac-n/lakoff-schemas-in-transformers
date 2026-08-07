"""
exp110_fclean_poles.py
Look at both poles of f_clean -- the orthogonal frequency-residual direction
(regression of log-rank on MEANINGFUL words; ~orthogonal to PC1/markedness).
What actually sits at each end?
"""
import numpy as np
import gensim.downloader as api
from scipy.stats import spearmanr

def unit(v): return v / (np.linalg.norm(v) + 1e-12)

print("Loading GloVe...")
kv = api.load("glove-wiki-gigaword-300"); mu = kv.vectors.mean(axis=0)
rank = {w: i for i, w in enumerate(kv.index_to_key)}
S = kv.vectors[:100000] - mu
ev, evec = np.linalg.eigh((S.T @ S) / S.shape[0]); pc1 = unit(evec[:, -1])

meaningful = [w for w in kv.index_to_key[:50000] if w.isalpha() and w.islower() and 4 <= len(w) <= 15]
Xm = np.stack([kv[w] - mu for w in meaningful])
ym = np.log1p(np.array([rank[w] for w in meaningful], float)); ym -= ym.mean()
wfr, *_ = np.linalg.lstsq(Xm, ym, rcond=None); f_clean = unit(wfr)

print(f"  meaningful words: {len(meaningful)}")
print(f"  cos(f_clean, PC1) = {f_clean @ pc1:+.3f}")

proj = Xm @ f_clean
rho, _ = spearmanr(proj, [rank[w] for w in meaningful])
print(f"  spearman(f_clean projection, rank) over meaningful = {rho:+.3f}")
order = np.argsort(proj)
# orient: which end is rarer?
rare_end  = [meaningful[i] for i in (order[::-1] if rho > 0 else order)][:35]
common_end = [meaningful[i] for i in (order if rho > 0 else order[::-1])][:35]

print("\n  f_clean  COMMON-leaning pole (low rank):")
for i in range(0, 35, 7): print("   ", common_end[i:i+7])
print("\n  f_clean  RARE-leaning pole (high rank):")
for i in range(0, 35, 7): print("   ", rare_end[i:i+7])

# for contrast: PC1 poles over the same meaningful set
pp = Xm @ pc1; po = np.argsort(pp)
print("\n  (contrast) PC1 pole A:", [meaningful[i] for i in po[::-1][:18]])
print("  (contrast) PC1 pole B:", [meaningful[i] for i in po[:18]])
