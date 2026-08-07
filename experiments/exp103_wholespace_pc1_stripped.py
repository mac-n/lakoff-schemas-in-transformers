"""
exp103_wholespace_pc1_stripped.py

Niamh's correction: stop measuring against the hand-curated 85-word 'cognitive
sample' (circular). Measure over the WHOLE vocabulary, with just PC1 stripped.

Metric (non-circular, whole-space):
  For unit vectors x_i = normalize( strip_PC1( w_i - mu ) ) over the ENTIRE vocab,
  and a unit contrast axis u (also PC1-stripped),
      varfrac(u) = mean_i (x_i . u)^2  =  u^T M u,   M = mean_i x_i x_i^T
  i.e. the fraction of the space's directional variance that lies along u.

Calibration:
  - random axis baseline:  E[varfrac] = 1/(d-1)  (d=300, PC1 removed)
  - eigenvalue spectrum of M: what a genuine dominant direction explains.
  Each contrast axis is placed in the PC spectrum ("behaves like the ~k-th PC").

Substrates: GloVe and fastText. PC1's frequency-rank correlation is reported so
we know what 'PC1' is in each (it is frequency in GloVe, not in fastText).
"""
import numpy as np
import gensim.downloader as api
from scipy.stats import spearmanr
import gc

CONTRASTS = [
    ("soft->hard","soft","hard"), ("warm->cold","warm","cold"),
    ("up->down","up","down"), ("forward->back","forward","back"),
    ("past->future","past","future"), ("chaos->order","chaos","order"),
    ("darkness->light","darkness","light"),
]
NAMES = [c[0] for c in CONTRASTS]

def unit(v):
    n = np.linalg.norm(v); return v/n if n > 1e-12 else v

def analyse(kv, label):
    print("\n" + "#"*78)
    print(f"# {label}   (vocab {len(kv.index_to_key):,}, dim {kv.vectors.shape[1]})")
    print("#"*78)
    V = kv.vectors
    N, d = V.shape
    mu = V.mean(axis=0)

    # PC1 from covariance of centered vectors
    C = V - mu
    cov = (C.T @ C) / N
    evals, evecs = np.linalg.eigh(cov)
    pc1 = unit(evecs[:, -1])

    # what is PC1? correlate with frequency rank over own top-50k
    n50 = min(50000, N)
    proj = (V[:n50] - mu) @ pc1
    rho, _ = spearmanr(proj, np.arange(n50))
    print(f"\n  PC1 explains {evals[-1]/evals.sum()*100:.1f}% of centered variance; "
          f"spearman(PC1, freq-rank) = {rho:+.3f}")
    print(f"  ({'PC1 IS frequency -> stripping removes the confound' if abs(rho)>0.5 else 'PC1 is NOT frequency here -> stripping removes top semantic variance'})")

    # accumulate M = mean x x^T over the WHOLE vocab, batched (x = unit, PC1-stripped)
    M = np.zeros((d, d))
    for s in range(0, N, 50000):
        b = V[s:s+50000] - mu
        b = b - np.outer(b @ pc1, pc1)
        nb = np.linalg.norm(b, axis=1, keepdims=True)
        nb[nb < 1e-12] = 1.0
        b = b / nb
        M += b.T @ b
    M /= N

    # spectrum of M (directional variance fractions) for calibration
    mvals = np.linalg.eigvalsh(M)[::-1]   # descending
    rand_base = 1.0 / (d - 1)
    print(f"\n  CALIBRATION (varfrac a single direction can explain):")
    print(f"    random axis baseline       : {rand_base:.4f}")
    print(f"    top directional PCs of space: " +
          ", ".join(f"PC{i+1}={mvals[i]:.4f}" for i in range(5)))

    # contrast axes (PC1-stripped, unit) and their whole-space variance fraction
    print(f"\n  WHOLE-SPACE VARIANCE FRACTION explained by each contrast axis:")
    print(f"    {'axis':<16}{'varfrac':>9}{'x random':>10}   behaves like")
    rows = []
    for name, a, b in CONTRASTS:
        if a not in kv or b not in kv:
            print(f"    {name:<16}  (OOV)"); continue
        u = unit((kv[b]-kv[a]) - np.dot(kv[b]-kv[a], pc1)*pc1)
        vf = float(u @ M @ u)
        like = int(np.searchsorted(-mvals, -vf))  # rank among PCs
        rows.append((name, u, vf))
        print(f"    {name:<16}{vf:>9.4f}{vf/rand_base:>9.1f}x   ~PC{like+1} of {d-1}")

    # pairwise cosines among the axes (PC1-stripped)
    print(f"\n  pairwise cosines (PC1-stripped):")
    print("        " + "".join(f"{n.split('->')[1][:6]:>8}" for n,_,_ in rows))
    for n1,u1,_ in rows:
        print(f"  {n1.split('->')[1][:5]:<5} " + "".join(f"{u1@u2:>8.2f}" for _,u2,_ in rows))
    return rows

print("Loading GloVe...")
g = api.load("glove-wiki-gigaword-300")
analyse(g, "GLOVE")
del g; gc.collect()

print("\nLoading fastText...")
f = api.load("fasttext-wiki-news-subwords-300")
analyse(f, "FASTTEXT")
del f; gc.collect()

print("\n" + "="*78)
print("READING")
print("="*78)
print("""  - varfrac tells you, over the ENTIRE vocabulary, what fraction of directional
    variance lies along each contrast. Compare to the random baseline and to the
    top PCs. An axis that explains ~random is not a real axis of the space; an axis
    near the top-PC range is a genuine major direction.""")
