"""
SAE schema-operator mining
==========================================================================
The relocation: schemas are not nodes (features) and not the variance-axes of
the feature cloud -- PCA already showed those come back as salient *content*.
A schema is an EDGE: a recurring difference-vector that turns one concept-state
into another. king->queen is the gender edge; the hope is that MORE, CONTAINER,
UP, FORCE etc. live as recurring edges too.

The trap (why "cluster all differences" alone fails): the difference between two
random features is dominated by their biggest *content* gap, so the difference-
clusters that recur most strongly are frequent CONTENT relations (capital-of,
plural-of, gender). The salient stuff wins the frequency contest again.

The discriminator that isolates a schema from a content-relation:
CROSS-DOMAIN RECURRENCE. A content relation lives in one domain pair
(countries->cities, always). A schema is the same operator firing across
semantically unrelated domains -- exactly Niamh's UP result, where UP recurred
across happy / health / power / quantity. So we don't just cluster differences;
we score each difference-cluster by the DOMAIN-DIVERSITY of the pairs that make
it up. Tight cluster + diverse domains = schema candidate. Tight + one domain =
content relation, discard.

Pipeline:
  1. load decoder, unit-normalise (feature directions on the 768-sphere)
  2. define "domains" = k-means clusters of the features themselves (offline
     proxy for semantic domain, no Neuronpedia calls needed)
  3. strip distractor directions (mean + top global PCs = the anisotropy /
     salient-content axes; the feature-space analog of Li et al's word-length
     strip). See note in strip_distractors about the proper LDA version.
  4. sample many random pairs, compute normalised difference vectors
  5. cluster the difference directions -> recurring transformations
  6. per cluster: tightness, size, domain-entropy -> rank schema candidates
  7. read them: print feature indices + Neuronpedia URLs for top clusters
  8. validate: geometric steering proxy now; causal steering = next step
  9. converge: cosine the cluster centroids against your hand-built UP/LIGHT
     contrast vectors -- the "two roads meet" test

Repo: EleutherAI/sae-pythia-160m-32k  (decoder 32768 x 768)
pip install sae scikit-learn numpy
(If the sae loading API has drifted, hand the LOAD block to Claude Code --
 everything below it is pure numpy/sklearn and takes W_dec as an argument.)
"""

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
LAYER             = 6        # pythia-160m has 12 layers; sweep later
N_DOMAINS         = 60       # coarse semantic domains (k-means on features)
STRIP_TOP_PCS     = 2        # distractor strip: 0 = none, 2-3 = remove anisotropy
N_PAIRS           = 1_000_000  # cap on sampled difference vectors (knn usually well under this)
K_NEIGHBORS       = 10       # difference each feature against its k nearest neighbours
N_DIFF_CLUSTERS   = 1000     # many small clusters to catch recurring directions
MIN_CLUSTER_SIZE  = 50       # ignore clusters smaller than this
SEED              = 0
NEURONPEDIA_MODEL = "pythia-160m-deduped"   # for the lookup URL pattern


# ----------------------------------------------------------------------
# LOAD  (the only block that touches the network / the sae library)
# ----------------------------------------------------------------------
def load_decoder(layer=LAYER):
    from sparsify import Sae  # EleutherAI (package name eai-sparsify, imports as sparsify)
    sae = Sae.load_from_hub("EleutherAI/sae-pythia-160m-32k",
                            hookpoint=f"layers.{layer}.mlp")
    W = sae.W_dec.detach().cpu().numpy().astype(np.float32)
    W = W / np.linalg.norm(W, axis=1, keepdims=True)   # unit feature directions
    print(f"decoder loaded: {W.shape}  (layer {layer})")
    return W


# ----------------------------------------------------------------------
# 2. DOMAINS  (offline proxy: which coarse semantic region each feature is in)
# ----------------------------------------------------------------------
def define_domains(W, n_domains=N_DOMAINS, seed=SEED):
    km = MiniBatchKMeans(n_clusters=n_domains, random_state=seed,
                         n_init=3, batch_size=4096)
    labels = km.fit_predict(W)
    return labels


# ----------------------------------------------------------------------
# 3. STRIP DISTRACTORS
# ----------------------------------------------------------------------
# Removes the global mean and the top few PCs of the feature cloud. Those top
# PCs are the high-variance "salient content" / residual-stream anisotropy axes
# -- the same thing PCA kept handing back. Pulling them out before differencing
# is the feature-space analog of Li et al projecting out word length with LDA.
#
# PROPER VERSION (if you have labels): if you can tag features with token-length
# / log-frequency of their top-activating token, run LDA with those as classes
# and project out the discriminant directions instead of blind top-PCs. Blind
# top-PC strip is the cheap honest first pass; it can also remove real structure,
# so try STRIP_TOP_PCS in {0, 2, 3} and see what's robust.
def strip_distractors(W, n_top_pcs=STRIP_TOP_PCS):
    mu = W.mean(axis=0, keepdims=True)
    Wc = W - mu
    if n_top_pcs > 0:
        pcs = PCA(n_components=n_top_pcs).fit(Wc).components_   # (k, d)
        Wc = Wc - (Wc @ pcs.T) @ pcs                            # project out
    # RENORMALIZE: post-strip vectors have varying norms because features
    # that were mostly along the stripped PCs become short. Without this,
    # Euclidean k-NN treats short features as all-near-each-other and creates
    # hub-and-spoke artifacts. Renormalizing makes Euclidean k-NN equivalent
    # to cosine k-NN, which is what we want for direction-only comparison.
    norms = np.linalg.norm(Wc, axis=1, keepdims=True)
    Wc = Wc / np.clip(norms, 1e-8, None)
    return Wc


# ----------------------------------------------------------------------
# 4. SAMPLE DIFFERENCE VECTORS
# ----------------------------------------------------------------------
# Random-pair differencing FAILS for schema mining: a schema lives in specific
# near-neighbour correspondences (happy->happier), and random pairs almost never
# hit those -- you just recover giant dense-region->dense-region content hops
# that swamp the quiet operator. So difference each feature against its k nearest
# neighbours. This is also how Li et al found parallelograms: among related
# concepts, never random ones. (mode='random' kept only as a baseline/null.)
def sample_differences(W, mode="knn", k=10, n_pairs=N_PAIRS, seed=SEED):
    rng = np.random.default_rng(seed)
    n = W.shape[0]
    if mode == "knn":
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=k + 1).fit(W)
        _, idx = nn.kneighbors(W)                    # (n, k+1), col0 is self
        i = np.repeat(np.arange(n), k)
        j = idx[:, 1:].reshape(-1)                   # each feat -> its k NNs
        if n_pairs and len(i) > n_pairs:             # subsample if huge
            sel = rng.choice(len(i), n_pairs, replace=False)
            i, j = i[sel], j[sel]
    else:  # 'random' baseline
        i = rng.integers(0, n, size=n_pairs)
        j = rng.integers(0, n, size=n_pairs)
        keep = i != j
        i, j = i[keep], j[keep]
    d = W[j] - W[i]                                  # the transformation i -> j
    norms = np.linalg.norm(d, axis=1, keepdims=True)
    keep = norms[:, 0] > 1e-8
    d, i, j = d[keep], i[keep], j[keep]
    d = d / norms[keep]                              # direction only
    return d.astype(np.float32), i, j


# ----------------------------------------------------------------------
# 5. CLUSTER THE DIFFERENCE DIRECTIONS
# ----------------------------------------------------------------------
def cluster_differences(d, n_clusters=N_DIFF_CLUSTERS, seed=SEED):
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed,
                         n_init=3, batch_size=8192)
    labels = km.fit_predict(d)
    centroids = km.cluster_centers_
    centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
    return labels, centroids


# ----------------------------------------------------------------------
# 6. SCORE  (tightness, size, cross-domain entropy)
# ----------------------------------------------------------------------
def _entropy(counts):
    p = counts / counts.sum()
    p = p[p > 0]
    h = -(p * np.log(p)).sum()
    return h / np.log(len(p)) if len(p) > 1 else 0.0   # normalised to [0,1]

def score_clusters(diff_labels, centroids, d, pair_i, pair_j,
                   domain_labels, n_clusters=N_DIFF_CLUSTERS,
                   min_size=MIN_CLUSTER_SIZE):
    rows = []
    di = domain_labels[pair_i]
    dj = domain_labels[pair_j]
    for c in range(n_clusters):
        m = diff_labels == c
        size = int(m.sum())
        if size < min_size:
            continue
        # tightness: how parallel are the edges in this cluster
        tight = float((d[m] @ centroids[c]).mean())
        # SCHEMA DISCRIMINATOR: entropy over SOURCE domains. A content relation
        # has one source domain (entropy ~0); a schema fires from many (high).
        # (Using the union of source+target would wrongly score a one-relation
        # A->B as max-entropy because it touches 2 domains -- that was the bug.)
        h = _entropy(np.bincount(di[m], minlength=domain_labels.max() + 1))
        # how many *distinct domain-pairs* the operator connects (secondary)
        n_dpairs = len(set(zip(di[m].tolist(), dj[m].tolist())))
        rows.append(dict(cluster=c, size=size, tightness=tight,
                         domain_entropy=h, n_domain_pairs=n_dpairs,
                         score=tight * h))
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows


# ----------------------------------------------------------------------
# 7. READ THE TOP OPERATORS
# ----------------------------------------------------------------------
# Schema candidates = high tightness AND high domain_entropy. A content relation
# will be tight but low-entropy; pure noise will be high-entropy but not tight.
def report(rows, diff_labels, pair_i, pair_j, top_k=15, examples=6):
    print(f"\n{'cluster':>8} {'size':>6} {'tight':>6} {'dom_H':>6} "
          f"{'#dpairs':>8} {'score':>6}")
    for r in rows[:top_k]:
        print(f"{r['cluster']:>8} {r['size']:>6} {r['tightness']:>6.3f} "
              f"{r['domain_entropy']:>6.3f} {r['n_domain_pairs']:>8} "
              f"{r['score']:>6.3f}")
    print("\n--- example edges for the top clusters (look these up) ---")
    for r in rows[:5]:
        c = r["cluster"]
        m = np.where(diff_labels == c)[0]
        sel = m[:examples]
        print(f"\ncluster {c}  (tight={r['tightness']:.3f} "
              f"dom_H={r['domain_entropy']:.3f}):")
        for k in sel:
            i, j = int(pair_i[k]), int(pair_j[k])
            print(f"   feat {i:6d}  ->  feat {j:6d}")
        print(f"   Neuronpedia: https://neuronpedia.org/{NEURONPEDIA_MODEL}/"
              f"{LAYER}-mlp-sae/<FEATURE_INDEX>")


# ----------------------------------------------------------------------
# 8. STEERING VALIDATION
# ----------------------------------------------------------------------
# Cheap geometric proxy (runnable now, no model): for held-out source features,
# the nearest feature to (f_src + v) should be a CONSISTENT semantic shift, and
# -- if v is a schema -- it should work across many domains, not one. We report
# the domain spread of the source features whose nearest-after-v neighbour is a
# clean hit. Broad spread = the operator generalises = schema-like.
def geometric_steer(W_raw, v, domain_labels, n_probe=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    v = v / np.linalg.norm(v)
    src = rng.choice(W_raw.shape[0], size=n_probe, replace=False)
    moved = W_raw[src] + v
    moved = moved / np.linalg.norm(moved, axis=1, keepdims=True)
    sims = moved @ W_raw.T               # (n_probe, n_features)
    sims[np.arange(n_probe), src] = -1   # don't let it map to itself
    nn = sims.argmax(axis=1)
    hit = sims.max(axis=1) > 0.5         # landed near a real feature
    src_domains_hit = domain_labels[src[hit]]
    spread = _entropy(np.bincount(src_domains_hit,
                                  minlength=domain_labels.max() + 1))
    print(f"  steer: {hit.sum()}/{n_probe} clean landings, "
          f"source-domain spread (entropy) = {spread:.3f}")
    return src[hit], nn[hit], spread
#
# CAUSAL VERSION (the real test, heavier -- hand to Claude Code):
#   load pythia-160m, hook layers.{LAYER}, add alpha*v to the residual stream at
#   that layer, generate, and check whether the output shifts the SAME way
#   (more / contained / upward) across prompts from unrelated domains. A steering
#   vector that generalises across domains is the thing correlation can't fake.


# ----------------------------------------------------------------------
# 9. CONVERGENCE WITH YOUR CONTRAST VECTORS  (the unfakeable result)
# ----------------------------------------------------------------------
# If the blind difference-cluster centroids cosine-align with the UP/LIGHT
# vectors you built BY HAND, that's the same primitives arriving two independent
# ways. Save your contrast vectors as a (k, 768) .npy with a names list.
def convergence(centroids, contrast_path="contrast_vectors.npy",
                names_path="contrast_names.txt"):
    try:
        C = np.load(contrast_path)
        names = open(names_path).read().split()
    except FileNotFoundError:
        print("\n(convergence test skipped: no contrast_vectors.npy found)")
        return
    C = C / np.linalg.norm(C, axis=1, keepdims=True)
    sims = C @ centroids.T                       # (k_contrast, n_clusters)
    print("\n--- convergence: best matching difference-cluster per contrast ---")
    for r, name in enumerate(names):
        c = int(sims[r].argmax())
        print(f"  {name:12s}  ->  cluster {c:4d}   cosine {sims[r, c]:.3f}")


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    W_raw = load_decoder(LAYER)
    domains = define_domains(W_raw)
    W = strip_distractors(W_raw)
    d, pi, pj = sample_differences(W, mode="knn", k=K_NEIGHBORS)
    print(f"sampled {len(d):,} difference vectors (knn, k={K_NEIGHBORS})")
    diff_labels, centroids = cluster_differences(d)
    rows = score_clusters(diff_labels, centroids, d, pi, pj, domains)
    report(rows, diff_labels, pi, pj)
    # NOTE: edges are directional, so a real operator usually shows up TWICE,
    # as +v and -v (two clusters with near-identical scores, centroids ~antiparallel).
    # Fold them together when reading -- they're one schema, both polarities.
    if rows:
        top_v = centroids[rows[0]["cluster"]]
        print("\nsteering the top schema candidate:")
        geometric_steer(W_raw, top_v, domains)
    convergence(centroids)
    #
    # WHAT A HIT LOOKS LIKE:
    #   a cluster that is tight (>~0.5), large, and high domain-entropy (>~0.7),
    #   whose example edges -- when you read the feature labels on Neuronpedia --
    #   are the SAME shift (more/contained/up) across unrelated concept families,
    #   and whose centroid cosine-matches one of your hand-built contrast vectors.
    #
    # WHAT A NULL LOOKS LIKE:
    #   tight clusters are all low-entropy content relations; the high-entropy
    #   clusters are all loose (noise). i.e. tightness and cross-domain
    #   recurrence never co-occur. That would say schemas are not recoverable as
    #   linear edges at this layer -- push to nonlinear (diffusion maps on the
    #   edge set) or to the causal-steering version before concluding anything.


if __name__ == "__main__":
    main()
