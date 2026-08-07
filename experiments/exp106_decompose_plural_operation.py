"""
exp106_decompose_plural_operation.py

Niamh: "could I see the 300x300 matrix? what if we decompose it?"

Fits the PLURAL operation (b ~ a W) in full 300-d space (mean-centered + PC1/freq
stripped), saves the matrix, and decomposes it. The raw lstsq map is ill-conditioned
(exp105), so we also fit a ridge-regularized map and decompose THAT.

Decompositions:
  - eigenvalue spectrum of W   (how identity-like? rotations = complex pairs?)
  - SVD of the ACTION  E = W - I   -> is the operation LOW-RANK? (the big question)
  - top singular directions of E, named by nearest words (what it reads / writes)
  - symmetric vs antisymmetric energy of E  (stretch vs rotation-generator)

Outputs:
  results_exp106_W_plural_raw.npy / .txt     (the matrix you asked to see)
  results_exp106_W_plural_ridge.npy          (the well-conditioned one we decompose)
  results_exp106.txt                          (this run's printed analysis)
"""
import numpy as np
import gensim.downloader as api

rng = np.random.default_rng(0)
def unit_rows(X): return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)

print("Loading GloVe...")
kv = api.load("glove-wiki-gigaword-300")
vocab = set(kv.index_to_key); mu = kv.vectors.mean(axis=0)
S = kv.vectors[:100000] - mu
ev, evec = np.linalg.eigh((S.T @ S) / S.shape[0])
pc1 = evec[:, -1]; pc1 = pc1 / np.linalg.norm(pc1)
d = 300

def vec(w):
    v = kv[w] - mu
    return v - np.dot(v, pc1) * pc1

pairs = [(w, w + "s") for w in kv.index_to_key[:80000]
         if w.isalpha() and len(w) >= 3 and not w.endswith("s") and (w + "s") in vocab]
rng.shuffle(pairs)
cut = int(0.85 * len(pairs)); tr, te = pairs[:cut], pairs[cut:]
A = np.stack([vec(a) for a, b in tr]); B = np.stack([vec(b) for a, b in tr])
Ate = np.stack([vec(a) for a, b in te]); Bte = unit_rows(np.stack([vec(b) for a, b in te]))
print(f"  plural pairs: {len(pairs)} (train {len(tr)})")

def heldout_cos(W): return float(np.mean(np.sum(unit_rows(Ate @ W) * Bte, axis=1)))
def inv_err(W):
    Wb = np.linalg.solve(W.T @ W + 1e-6*np.eye(d), W.T)  # right-ish inverse via lstsq
    return np.linalg.norm(W @ Wb - np.eye(d)) / np.sqrt(d)

# raw least squares (the matrix she saw)
W_raw, *_ = np.linalg.lstsq(A, B, rcond=None)
np.save("results_exp106_W_plural_raw.npy", W_raw)
np.savetxt("results_exp106_W_plural_raw.txt", W_raw, fmt="%.4f")

# ridge sweep -- fit the ACTION E directly (shrink E->0, i.e. W->I in unconstrained dirs)
#   minimise ||A E - (B - A)||^2 + lambda ||E||^2 ;  W = I + E
AtA = A.T @ A; At_dB = A.T @ (B - A)
lam_max = np.linalg.eigvalsh(AtA)[-1]
print(f"\n  identity held-out cos (baseline): {float(np.mean(np.sum(unit_rows(Ate)*Bte,axis=1))):.3f}")
print(f"  raw lstsq:  held-out cos {heldout_cos(W_raw):.3f}   ||W||_F {np.linalg.norm(W_raw):.1f}")
print("\n  ridge sweep on the ACTION E (lambda as fraction of largest eigenvalue of A^T A):")
chosen = None
for frac in [1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
    lam = frac * lam_max
    E = np.linalg.solve(AtA + lam*np.eye(d), At_dB)
    W = np.eye(d) + E
    print(f"    frac={frac:<6}  held-out cos {heldout_cos(W):.3f}   ||E||_F {np.linalg.norm(E):.2f}")
    if frac == 1e-2:
        chosen = (frac, W)
frac, Wr = chosen
np.save("results_exp106_W_plural_ridge.npy", Wr)
print(f"\n  -> decomposing the ridge map at frac={frac} (||W||_F={np.linalg.norm(Wr):.2f})")

# ---------- DECOMPOSITION ----------
print("\n" + "="*70); print("EIGENVALUES of W  (W = I means identity; deviation = the operation)"); print("="*70)
eigs = np.linalg.eigvals(Wr)
re, im = eigs.real, eigs.imag
print(f"  {d} eigenvalues. |lambda-1| summary: "
      f"min {np.min(np.abs(eigs-1)):.3f}  median {np.median(np.abs(eigs-1)):.3f}  max {np.max(np.abs(eigs-1)):.3f}")
print(f"  near-identity (|lambda-1|<0.05): {(np.abs(eigs-1)<0.05).sum()} / {d}")
print(f"  complex (|Im|>1e-6): {(np.abs(im)>1e-6).sum()} eigenvalues "
      f"({(np.abs(im)>1e-6).sum()//2} rotation pairs)")
print(f"  real part range: [{re.min():.3f}, {re.max():.3f}]")

print("\n" + "="*70); print("SVD of the ACTION  E = W - I   (is the operation LOW-RANK?)"); print("="*70)
E = Wr - np.eye(d)
U, s, Vt = np.linalg.svd(E)
energy = s**2 / (s**2).sum()
cum = np.cumsum(energy)
print(f"  ||E||_F = {np.linalg.norm(E):.3f}")
print(f"  top singular values: {np.round(s[:12],3)}")
for thr in [0.5, 0.9, 0.95]:
    print(f"    #components for {int(thr*100)}% of action energy: {int(np.searchsorted(cum, thr)+1)} / {d}")

# name the top action directions by nearest words
N = 50000
voc = kv.index_to_key[:N]
Mv = unit_rows(np.stack([vec(w) for w in voc]))
print("\n  TOP ACTION DIRECTIONS (E = sum sigma_k * u_k v_k^T):")
for k in range(4):
    vin = Vt[k]; uout = U[:, k]
    si = Mv @ vin; so = Mv @ uout
    print(f"\n  dir {k+1}  (sigma={s[k]:.3f})")
    print(f"    READS  (input v_{k+1}, +): {[voc[i] for i in np.argsort(si)[::-1][:8]]}")
    print(f"    READS  (input v_{k+1}, -): {[voc[i] for i in np.argsort(si)[:8]]}")
    print(f"    WRITES (output u_{k+1}, +): {[voc[i] for i in np.argsort(so)[::-1][:8]]}")

print("\n" + "="*70); print("SYMMETRIC vs ANTISYMMETRIC part of E  (stretch vs rotation-generator)"); print("="*70)
Sym = (E + E.T)/2; Anti = (E - E.T)/2
es = np.linalg.norm(Sym)**2; ea = np.linalg.norm(Anti)**2
print(f"  symmetric energy  (stretch / scaling):        {es/(es+ea):.1%}")
print(f"  antisymmetric energy (rotation generator):    {ea/(es+ea):.1%}")
print("""
  - If E is low-rank, the operation acts in a few interpretable directions.
  - Antisymmetric-heavy => the operation is mostly a rotation (a Lie generator);
    symmetric-heavy => mostly stretching/projection along directions.
Files written: results_exp106_W_plural_raw.{npy,txt}, results_exp106_W_plural_ridge.npy
""")
