"""
exp93 — PCA oracle at 24 axes (quick).

Comparison: greedy 24-axis hybrid reached 51.66% on cognitive test sample.
What does PCA-24 capture on the same sample? That's the variance-maximizing
24-axis basis — the upper bound for 24-axis explanatory power.
"""
import numpy as np
import gensim.downloader as api


def unit(v):
    return v / np.linalg.norm(v)


def gs_orthogonalize(axes):
    out = []
    for v in axes:
        u = v.copy()
        for p in out:
            u = u - (u @ p) * p
        n = np.linalg.norm(u)
        if n < 1e-10:
            continue
        out.append(u / n)
    return out


def coverage_in(v, gs_basis):
    v_residual = v.copy()
    for u_gs in gs_basis:
        v_residual = v_residual - (v_residual @ u_gs) * u_gs
    return float(np.sqrt(max(0, 1 - np.linalg.norm(v_residual) ** 2)))


print("Loading GloVe...")
wv = api.load("glove-wiki-gigaword-300")
mu = wv.vectors.mean(axis=0)


def get_deanisotropized(word):
    if word not in wv.key_to_index:
        return None
    v = wv[word] - mu
    n = np.linalg.norm(v)
    if n < 1e-10:
        return None
    return v / n


# Same cognitive test categories as exp92
test_categories = {
    "EMOTIONS_AFFECTIVE": [
        "happiness", "sadness", "anger", "envy", "jealousy", "pride", "humility",
        "contentment", "longing", "delight", "melancholy", "rage", "elation",
        "despair", "serenity", "anguish", "disgust",
    ],
    "AGENTIVE_STATES": [
        "ambition", "determination", "resignation", "willpower", "discipline",
        "procrastination", "perseverance", "complacency", "vigilance",
        "diligence", "industriousness", "negligence", "carelessness",
    ],
    "SOCIAL_RELATIONAL": [
        "trust", "betrayal", "friendship", "enmity", "loyalty", "rivalry",
        "respect", "contempt", "admiration", "scorn", "gratitude", "resentment",
    ],
    "EPISTEMIC_STATES": [
        "knowledge", "ignorance", "belief", "doubt", "uncertainty",
        "conviction", "skepticism", "confidence", "hesitation",
        "speculation", "intuition", "memory", "forgetting",
    ],
    "CONCRETE_NOUNS": [
        "chair", "table", "dog", "stone", "tree", "river", "mountain",
        "hammer", "rope", "lamp", "cup", "window",
    ],
    "ABSTRACT_FORMAL": [
        "theorem", "philosophy", "ontology", "epistemology", "axiom",
        "principle", "framework", "paradigm", "schema", "abstraction",
    ],
    "MODAL_HYPOTHETICAL": [
        "hypothetical", "imaginary", "fictional", "speculative", "conjectural",
        "perhaps", "supposedly", "allegedly", "putative",
    ],
}

test_vectors = []
for cat, words in test_categories.items():
    for w in words:
        v = get_deanisotropized(w)
        if v is not None:
            test_vectors.append(v)
test_vectors = np.stack(test_vectors)
print(f"Test sample: {len(test_vectors)} cognitive words")


# PCA on deanisotropized 50K sample
print("\nComputing PCA on 50K deanisotropized random vocab...")
np.random.seed(42)
sample_idx = np.random.choice(len(wv.vectors), 50000, replace=False)
sample = wv.vectors[sample_idx] - mu

U, S, Vt = np.linalg.svd(sample, full_matrices=False)
var_ratios = (S ** 2) / (S ** 2).sum()

# Coverage for different PCA sizes
print(f"\n{'PCA size':<10} {'coverage':>10} {'cumulative variance ratio':>30}")
print("-" * 55)
for k in [10, 12, 13, 19, 24, 30, 40, 50]:
    pca_basis = [Vt[i] for i in range(k)]
    gs_pca = gs_orthogonalize(pca_basis)
    cs = []
    for v in test_vectors:
        cs.append(coverage_in(v, gs_pca))
    mean_cov = np.mean(cs)
    cum_var = var_ratios[:k].sum()
    print(f"  PCA-{k:<6} {mean_cov * 100:>8.2f}%  {cum_var * 100:>22.2f}%")

print(f"\nReference: greedy-24 hybrid on same sample = 51.66%")
print(f"Reference: Cog-13b on same sample = 39.27%")
