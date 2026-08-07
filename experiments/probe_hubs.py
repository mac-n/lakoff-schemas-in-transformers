"""
Quick probe: are the hub features (12466, 14242, 27311, 19569, 31468) aligned
with anisotropy / salience-style directions in the SAE decoder?

Tests:
  1. Cosine of each hub to the mean decoder direction (= the anisotropy axis)
  2. Cosine of each hub to top 10 PCs of the decoder cloud
  3. Cosine of hubs to each other (mutually parallel = one common direction)
  4. Same in PC-stripped space (what the pipeline actually sees)

If hubs are salience/anisotropy features:
  - High cosine to mean direction
  - High cosine to top PCs (especially PCs 3-10, since 1-2 were stripped)
  - Mutually highly parallel
"""
import numpy as np
from sklearn.decomposition import PCA
from sparsify import Sae

HUBS = [12466, 14242, 27311, 19569, 31468]
LAYER = 6

print("Loading SAE...")
sae = Sae.load_from_hub("EleutherAI/sae-pythia-160m-32k", hookpoint=f"layers.{LAYER}.mlp")
W = sae.W_dec.detach().cpu().numpy().astype(np.float32)
W = W / np.linalg.norm(W, axis=1, keepdims=True)
print(f"decoder: {W.shape}")

mean_dir = W.mean(axis=0)
mean_dir_n = mean_dir / np.linalg.norm(mean_dir)
print(f"\n||mean decoder direction|| = {np.linalg.norm(mean_dir):.4f}")
print(f"  (large = high anisotropy in decoder cloud; small = isotropic)")

print("\n=== Cosine of each hub to mean decoder direction ===")
for h in HUBS:
    c = float(W[h] @ mean_dir_n)
    print(f"  feat {h:>6d}:  cos(W[h], mean_dir) = {c:+.4f}")

print("\n=== PCA on raw decoder cloud (top 10 PCs) ===")
pca = PCA(n_components=10).fit(W - W.mean(axis=0, keepdims=True))
print(f"  explained variance ratio: {pca.explained_variance_ratio_}")

print("\n=== Cosine of each hub to each top PC (raw decoder) ===")
print(f"  {'feat':>6}  " + "  ".join(f"PC{i+1:>2}" for i in range(10)))
for h in HUBS:
    cos_to_pcs = [float(W[h] @ pc / np.linalg.norm(pc)) for pc in pca.components_]
    print(f"  {h:>6d}  " + "  ".join(f"{c:>+.2f}" for c in cos_to_pcs))

print("\n=== Cosines between hubs (raw decoder) ===")
H = W[HUBS]
M = H @ H.T
print(f"  {'':>8}" + "  ".join(f"{h:>7d}" for h in HUBS))
for i, h in enumerate(HUBS):
    print(f"  {h:>7d} " + "  ".join(f"{M[i, j]:>+.4f}" for j in range(len(HUBS))))

print("\n=== After stripping top 2 PCs (what the pipeline sees) ===")
Wc = W - W.mean(axis=0, keepdims=True)
top2 = pca.components_[:2]
W_stripped = Wc - (Wc @ top2.T) @ top2
W_stripped = W_stripped / np.linalg.norm(W_stripped, axis=1, keepdims=True)

mean_dir_s = W_stripped.mean(axis=0)
print(f"  ||mean decoder direction|| (post-strip) = {np.linalg.norm(mean_dir_s):.4f}")

print(f"\n  Cosines between hubs (post-strip):")
H_s = W_stripped[HUBS]
M_s = H_s @ H_s.T
print(f"  {'':>8}" + "  ".join(f"{h:>7d}" for h in HUBS))
for i, h in enumerate(HUBS):
    print(f"  {h:>7d} " + "  ".join(f"{M_s[i, j]:>+.4f}" for j in range(len(HUBS))))

print("\n=== PCs 3-10 of the post-strip decoder (the residual anisotropy) ===")
pca_s = PCA(n_components=10).fit(W_stripped)
print(f"  explained variance ratio (post-strip): {pca_s.explained_variance_ratio_}")
print(f"  PCs 3-10 cosine to hubs (post-strip):")
print(f"  {'feat':>6}  " + "  ".join(f"PC{i+1:>2}" for i in range(10)))
for h in HUBS:
    cos_to_pcs = [float(W_stripped[h] @ pc / np.linalg.norm(pc) / (np.linalg.norm(W_stripped[h]) + 1e-12))
                  for pc in pca_s.components_]
    print(f"  {h:>6d}  " + "  ".join(f"{c:>+.2f}" for c in cos_to_pcs))

print("\n=== How many features each hub is the nearest neighbor of (k=10 search) ===")
# Use raw cosine sim. For each feature, find its top 10 NNs. Count how often each hub appears.
from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=11).fit(W_stripped)
_, idx = nn.kneighbors(W_stripped)
nn_list = idx[:, 1:].reshape(-1)  # drop self
from collections import Counter
counts = Counter(nn_list.tolist())
total = W.shape[0]
print(f"  Hub feature   NN-count   pct-of-features")
for h in HUBS:
    n = counts.get(h, 0)
    print(f"  {h:>11d}   {n:>8d}   {100*n/total:>6.2f}%")

print(f"\n  For comparison, top 10 most-popular NN features overall:")
for feat, n in counts.most_common(10):
    print(f"  feat {feat:>6d}: NN of {n:>5d} features ({100*n/total:>5.2f}%)")
