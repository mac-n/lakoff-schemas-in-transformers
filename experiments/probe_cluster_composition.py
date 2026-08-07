"""
Look at the actual composition of the top difference cluster.
Earlier claim that "many sources → one target = hub-and-spoke" was based on 6
example edges out of 55. If hubs are only NN-of ~10 features, the rest of the
cluster's edges must point to DIFFERENT targets — which would mean the cluster
is a real schema-shaped recurring operator, not hub-and-spoke.

Reconstructs the differences exactly as schema_operators.py did and inspects:
  - How many distinct sources, distinct targets in top cluster?
  - Are sources concentrated or spread? Are targets concentrated or spread?
  - What does the centroid direction look like geometrically?
"""
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sparsify import Sae

LAYER = 6
N_DOMAINS = 60
STRIP_TOP_PCS = 2
K_NEIGHBORS = 10
N_DIFF_CLUSTERS = 1000
SEED = 0

print("Loading SAE...")
sae = Sae.load_from_hub("EleutherAI/sae-pythia-160m-32k", hookpoint=f"layers.{LAYER}.mlp")
W_raw = sae.W_dec.detach().cpu().numpy().astype(np.float32)
W_raw = W_raw / np.linalg.norm(W_raw, axis=1, keepdims=True)

print("Stripping distractors (with post-strip renormalization)...")
Wc = W_raw - W_raw.mean(axis=0, keepdims=True)
pcs = PCA(n_components=STRIP_TOP_PCS).fit(Wc).components_
Wc = Wc - (Wc @ pcs.T) @ pcs
norms = np.linalg.norm(Wc, axis=1, keepdims=True)
W = Wc / np.clip(norms, 1e-8, None)

print("k-NN sampling...")
n = W.shape[0]
nn = NearestNeighbors(n_neighbors=K_NEIGHBORS + 1).fit(W)
_, idx = nn.kneighbors(W)
pair_i = np.repeat(np.arange(n), K_NEIGHBORS)
pair_j = idx[:, 1:].reshape(-1)
d = W[pair_j] - W[pair_i]
norms = np.linalg.norm(d, axis=1, keepdims=True)
keep = norms[:, 0] > 1e-8
d = d[keep] / norms[keep]
pair_i, pair_j = pair_i[keep], pair_j[keep]

print(f"sampled {len(d):,} differences")

print("Clustering differences...")
km = MiniBatchKMeans(n_clusters=N_DIFF_CLUSTERS, random_state=SEED, n_init=3, batch_size=8192)
diff_labels = km.fit_predict(d)
centroids = km.cluster_centers_
centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)

# Find cluster 33 (or whatever's currently top scoring)
print("\nReproducing top-15 cluster ranking...")
# domain labels
km_d = MiniBatchKMeans(n_clusters=N_DOMAINS, random_state=SEED, n_init=3, batch_size=4096)
domain_labels = km_d.fit_predict(W_raw)

def entropy(counts):
    p = counts / counts.sum()
    p = p[p > 0]
    if len(p) <= 1:
        return 0.0
    return float(-(p * np.log(p)).sum() / np.log(len(p)))

di = domain_labels[pair_i]
dj = domain_labels[pair_j]

scored = []
for c in range(N_DIFF_CLUSTERS):
    m = diff_labels == c
    size = int(m.sum())
    if size < 50:
        continue
    tight = float((d[m] @ centroids[c]).mean())
    h = entropy(np.bincount(di[m], minlength=domain_labels.max() + 1))
    scored.append((c, size, tight, h, tight * h))

scored.sort(key=lambda r: r[4], reverse=True)

print(f"\nTop cluster: {scored[0][0]}, size={scored[0][1]}, tight={scored[0][2]:.3f}, dom_H={scored[0][3]:.3f}")

for top_idx in range(3):
    cluster_id = scored[top_idx][0]
    m = np.where(diff_labels == cluster_id)[0]
    sources = pair_i[m]
    targets = pair_j[m]

    print(f"\n=== Cluster {cluster_id} composition ===")
    print(f"  total edges:        {len(m)}")
    print(f"  distinct sources:   {len(set(sources.tolist()))}")
    print(f"  distinct targets:   {len(set(targets.tolist()))}")
    print(f"  ratio targets/edges: {len(set(targets.tolist()))/len(m):.3f}")
    print(f"  (if hub-and-spoke: ratio ~0.02; if real operator: ratio ~1.0)")

    # Top 10 most common targets
    from collections import Counter
    target_counts = Counter(targets.tolist())
    print(f"\n  Top 10 most common targets in this cluster:")
    for t, n_t in target_counts.most_common(10):
        print(f"    feat {t:>6d}: appears as target {n_t:>3d} times")

    # Top 10 most common sources
    source_counts = Counter(sources.tolist())
    print(f"\n  Top 10 most common sources:")
    for s, n_s in source_counts.most_common(10):
        print(f"    feat {s:>6d}: appears as source {n_s:>3d} times")
