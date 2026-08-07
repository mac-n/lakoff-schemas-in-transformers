"""
exp101_what_is_pc1.py

Niamh's challenge: calling hardness an 'anisotropy artifact' assumes global PC1 is
nuisance. Is it? And which end of hardness lines up with which end of PC1?

Does, concretely:
  1. Recompute global PC1 (centered top-100k GloVe), same as exp100.
  2. Show the words at each pole of PC1 (what IS this direction?).
  3. Project the hard-pole and soft-pole anchor words onto PC1 (sign/direction).
  4. Test the frequency hypothesis: is PC1 just frequency rank?
  5. For reference, do the same pole-word readout for the hardness axis itself.
"""
import numpy as np
import gensim.downloader as api
from scipy.stats import spearmanr

def unit(v):
    return v / np.linalg.norm(v)

print("Loading GloVe...")
wv = api.load("glove-wiki-gigaword-300")
mu = wv.vectors.mean(axis=0)

# global PC1 (centered, top-100k) -- identical to exp100
X = wv.vectors[:100000] - mu
cov = (X.T @ X) / X.shape[0]
evals, evecs = np.linalg.eigh(cov)
pc1 = unit(evecs[:, -1])

# hardness axis (exp90 canonical): hard - soft
hard_pole = ["hard", "firm", "rigid", "solid", "stiff"]
soft_pole = ["soft", "mushy", "pliable", "flimsy", "squishy"]
pairs = [("hard","soft"),("firm","mushy"),("rigid","pliable"),("solid","flimsy")]
hardness = unit(np.stack([wv[a]-wv[b] for a,b in pairs]).mean(axis=0))

# Orient PC1 so its '+' end is whatever; we'll read both ends explicitly.
# Project a frequent slice (centered + normalized) onto PC1 and onto hardness.
N = 50000
vocab = wv.index_to_key[:N]
M = wv.vectors[:N] - mu
M = M / np.linalg.norm(M, axis=1, keepdims=True)
proj_pc1 = M @ pc1
proj_h = M @ hardness
order = np.argsort(proj_pc1)

print("\n" + "="*74)
print(f"GLOBAL PC1  (explains {evals[-1]/evals.sum()*100:.1f}% of centered variance)")
print("="*74)
print("\n  PC1(+) pole — top 30 words:")
print("   ", [vocab[i] for i in order[::-1][:30]])
print("\n  PC1(-) pole — bottom 30 words:")
print("   ", [vocab[i] for i in order[:30]])

print("\n" + "="*74)
print("FREQUENCY TEST: is PC1 just frequency rank?")
print("="*74)
ranks = np.arange(N)  # 0 = most frequent
rho, p = spearmanr(proj_pc1, ranks)
print(f"  spearman(PC1 projection, frequency rank) over top-{N}: rho = {rho:+.3f}")
print("  (|rho| near 1 => PC1 is essentially frequency; near 0 => it is not)")

print("\n" + "="*74)
print("WHICH END OF HARDNESS <-> WHICH END OF PC1")
print("="*74)
print(f"  cos(hardness, PC1) = {hardness @ pc1:+.3f}   (hardness = hard - soft)")
print("\n  PC1 projection of each anchor word (centered+normalized cos with PC1):")
print("    HARD-pole words:")
for w in hard_pole:
    if w in wv:
        v = unit(wv[w]-mu)
        print(f"      {w:<10} cos(.,PC1)={v@pc1:+.3f}   cos(.,hardness)={v@hardness:+.3f}")
print("    SOFT-pole words:")
for w in soft_pole:
    if w in wv:
        v = unit(wv[w]-mu)
        print(f"      {w:<10} cos(.,PC1)={v@pc1:+.3f}   cos(.,hardness)={v@hardness:+.3f}")

print("\n" + "="*74)
print("FOR REFERENCE: poles of the HARDNESS axis itself (top-50k)")
print("="*74)
oh = np.argsort(proj_h)
print("\n  HARD pole — top 30:")
print("   ", [vocab[i] for i in oh[::-1][:30]])
print("\n  SOFT pole — bottom 30:")
print("   ", [vocab[i] for i in oh[:30]])
