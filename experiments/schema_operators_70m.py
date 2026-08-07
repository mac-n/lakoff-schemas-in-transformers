"""
Port of schema_operators.py to Pythia 70m residual-stream SAE at layer 5
(the concentrated layer per exp7c / exp8). Same pipeline, different substrate.

Why this substrate:
  - Neuronpedia HAS auto-interp descriptions for these SAEs (via SAE Lens), so
    we can actually look up what the hub/cluster features fire on, unlike the
    Pythia 160m MLP SAEs.
  - We already characterized this layer in exp8: PC0 = legal/government register,
    PC6 = biomedical, PC8 = LaTeX/math. So we have a baseline interpretation of
    what the dominant geometry looks like.

Includes:
  - the post-strip renormalization fix from the 160m run
  - automatic Neuronpedia lookup for the hub features in the top clusters
"""
import json
import re
import time

import numpy as np
import requests
from sae_lens import SAE
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

# CONFIG
RELEASE           = "pythia-70m-deduped-res-sm"
SAE_ID            = "blocks.5.hook_resid_post"
NEURONPEDIA_MODEL = "pythia-70m-deduped"
NEURONPEDIA_SAE   = "5-res-sm"
N_DOMAINS         = 60
STRIP_TOP_PCS     = 2
K_NEIGHBORS       = 10
N_DIFF_CLUSTERS   = 1000
MIN_CLUSTER_SIZE  = 50
SEED              = 0
API_DELAY         = 0.3
TOP_HUBS_TO_LOOKUP = 8  # hubs to query on Neuronpedia


# ---------------------------------------------------------------------- LOAD
print(f"Loading SAE: {RELEASE} / {SAE_ID}...")
sae_res = SAE.from_pretrained(release=RELEASE, sae_id=SAE_ID, device="cpu")
sae = sae_res[0] if isinstance(sae_res, tuple) else sae_res
W_raw = sae.W_dec.detach().cpu().numpy().astype(np.float32)
W_raw = W_raw / np.linalg.norm(W_raw, axis=1, keepdims=True)
print(f"decoder loaded: {W_raw.shape}  (Pythia 70m res-sm L5)")


# ---------------------------------------------------------------------- DOMAINS
km_d = MiniBatchKMeans(n_clusters=N_DOMAINS, random_state=SEED, n_init=3, batch_size=4096)
domain_labels = km_d.fit_predict(W_raw)


# ---------------------------------------------------------------------- STRIP + RENORM
Wc = W_raw - W_raw.mean(axis=0, keepdims=True)
if STRIP_TOP_PCS > 0:
    pcs = PCA(n_components=STRIP_TOP_PCS).fit(Wc).components_
    Wc = Wc - (Wc @ pcs.T) @ pcs
norms = np.linalg.norm(Wc, axis=1, keepdims=True)
W = Wc / np.clip(norms, 1e-8, None)


# ---------------------------------------------------------------------- SAMPLE
print(f"k-NN sampling (k={K_NEIGHBORS})...")
n = W.shape[0]
nn = NearestNeighbors(n_neighbors=K_NEIGHBORS + 1).fit(W)
_, idx = nn.kneighbors(W)
pair_i = np.repeat(np.arange(n), K_NEIGHBORS)
pair_j = idx[:, 1:].reshape(-1)
d = W[pair_j] - W[pair_i]
norms_d = np.linalg.norm(d, axis=1, keepdims=True)
keep = norms_d[:, 0] > 1e-8
d = d[keep] / norms_d[keep]
pair_i, pair_j = pair_i[keep], pair_j[keep]
print(f"sampled {len(d):,} difference vectors")


# ---------------------------------------------------------------------- CLUSTER
print("Clustering differences...")
km = MiniBatchKMeans(n_clusters=N_DIFF_CLUSTERS, random_state=SEED, n_init=3, batch_size=8192)
diff_labels = km.fit_predict(d)
centroids = km.cluster_centers_
centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)


# ---------------------------------------------------------------------- SCORE
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
    if size < MIN_CLUSTER_SIZE:
        continue
    tight = float((d[m] @ centroids[c]).mean())
    h = entropy(np.bincount(di[m], minlength=domain_labels.max() + 1))
    n_dpairs = len(set(zip(di[m].tolist(), dj[m].tolist())))
    scored.append({
        "cluster": c, "size": size, "tightness": tight,
        "domain_entropy": h, "n_domain_pairs": n_dpairs, "score": tight * h
    })
scored.sort(key=lambda r: r["score"], reverse=True)


# ---------------------------------------------------------------------- REPORT
print(f"\n{'cluster':>8} {'size':>6} {'tight':>6} {'dom_H':>6} {'#dpairs':>8} {'score':>6}")
for r in scored[:15]:
    print(f"{r['cluster']:>8} {r['size']:>6} {r['tightness']:>6.3f} "
          f"{r['domain_entropy']:>6.3f} {r['n_domain_pairs']:>8} {r['score']:>6.3f}")


# ---------------------------------------------------------------------- COMPOSITION ANALYSIS
# For each top cluster: is it hub-and-spoke or schema-shaped?
print("\n=== Cluster composition (hub-and-spoke vs schema-shaped) ===")
from collections import Counter
hub_features = []  # collect distinct hub targets for Neuronpedia lookup
schema_clusters = []  # collect clusters that look schema-shaped

for r in scored[:10]:
    cluster_id = r["cluster"]
    m = np.where(diff_labels == cluster_id)[0]
    sources, targets = pair_i[m], pair_j[m]
    n_distinct_targets = len(set(targets.tolist()))
    target_ratio = n_distinct_targets / len(m)
    target_counts = Counter(targets.tolist())
    top_target, top_target_n = target_counts.most_common(1)[0]
    shape = "hub-and-spoke" if target_ratio < 0.1 else "mixed" if target_ratio < 0.5 else "schema-shaped"
    print(f"  cluster {cluster_id:>4}: size={r['size']:>3}  tight={r['tightness']:.3f}  "
          f"dom_H={r['domain_entropy']:.3f}  "
          f"n_targets={n_distinct_targets:>3}  ratio={target_ratio:.3f}  → {shape}  "
          f"(top target feat {top_target}: {top_target_n}/{len(m)} edges)")
    if shape == "hub-and-spoke":
        hub_features.append((top_target, cluster_id, top_target_n))
    else:
        schema_clusters.append((cluster_id, r))


# ---------------------------------------------------------------------- NEURONPEDIA LOOKUP
def fetch_feature(feat_idx):
    url = f"https://www.neuronpedia.org/api/feature/{NEURONPEDIA_MODEL}/{NEURONPEDIA_SAE}/{feat_idx}"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return None
        try:
            return r.json()
        except json.JSONDecodeError:
            clean = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', r.text)
            return json.loads(clean, strict=False)
    except Exception as e:
        return {"_error": str(e)}


def feature_summary(d):
    if d is None:
        return "[no data]"
    if "_error" in d:
        return f"[error: {d['_error']}]"
    expls = d.get("explanations") or []
    if expls:
        return expls[0].get("description", "").strip()[:200]
    return "(no auto-interp description)"


print(f"\n=== Looking up hub features on Neuronpedia ===")
unique_hubs = list({h[0] for h in hub_features})[:TOP_HUBS_TO_LOOKUP]
print(f"Querying {len(unique_hubs)} unique hub features: {unique_hubs}")
for h_feat in unique_hubs:
    data = fetch_feature(h_feat)
    desc = feature_summary(data)
    print(f"\n  feat {h_feat}:")
    print(f"    {desc}")
    print(f"    https://www.neuronpedia.org/{NEURONPEDIA_MODEL}/{NEURONPEDIA_SAE}/{h_feat}")
    time.sleep(API_DELAY)


# If any clusters look schema-shaped, look those up too
if schema_clusters:
    print(f"\n=== Schema-shaped clusters (rare/interesting) ===")
    for cluster_id, r in schema_clusters[:5]:
        m = np.where(diff_labels == cluster_id)[0]
        print(f"\n  cluster {cluster_id} examples (first 8 distinct edges):")
        seen_edges = set()
        for k in m:
            edge = (int(pair_i[k]), int(pair_j[k]))
            if edge in seen_edges:
                continue
            seen_edges.add(edge)
            if len(seen_edges) > 8:
                break
            print(f"    feat {edge[0]:>6d}  ->  feat {edge[1]:>6d}")
else:
    print("\n=== No schema-shaped clusters in top 10 ===")
    print("All top-scoring clusters were hub-and-spoke. This matches the 160m result.")


# ---------------------------------------------------------------------- SAVE FOR FOLLOWUP
np.savez(
    "/Users/macn/Documents/embeddingexp/schema_operators_70m_results.npz",
    centroids=centroids,
    pair_i=pair_i,
    pair_j=pair_j,
    diff_labels=diff_labels,
    domain_labels=domain_labels,
    scored=np.array([(r["cluster"], r["size"], r["tightness"], r["domain_entropy"], r["score"]) for r in scored]),
)
print("\nSaved results to schema_operators_70m_results.npz")
