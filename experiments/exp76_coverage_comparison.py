"""
exp76 — Comprehensive coverage comparison.

Tests where our 10-axis basis sits relative to:
  1. Our 10-axis cognitive basis (C, W, ATT, INT, R, D, IO, DV, MB, EV)
  2. 10-axis Lakoff image-schema basis (UD, IO, FB, LD, PATH, EXIST, FORCE, BAL, DIFF, COH)
  3. 10-axis random orthogonal bases (averaged over 20 seeds, anisotropy-cancelled)
  4. First 10 PCs of deanisotropized GloVe — oracle upper bound for 10-axis
  5. Our 10 + ABSTRACT_CONCRETE + MODAL_STATUS = 12 axes — tests Niamh's hypothesis
  6. First 12 PCs of deanisotropized GloVe — oracle upper bound for 12-axis

All bases compared with anisotropy correction applied uniformly.

Test categories include the existing exp75 ones plus new MODAL_HYPOTHETICAL,
MODAL_ACTUAL, and ABSTRACT_FORMAL_EXTENDED — categories that should specifically
light up if abstract/concrete and modal-status are real missing primitives.
"""
import numpy as np
import gensim.downloader as api
import sys
sys.path.insert(0, "/Users/macn/Documents/embeddingexp")

from project_axis_vocabulary import (
    TARGET_REWARD_COMPOSITE_PAIRS, TARGET_WEIGHT_PAIRS,
    ATTENTION_CLEAN_PAIRS, INTENTION_CLEAN_PAIRS,
    TARGET_EQUILIBRIUM_RUNAWAY_PAIRS, TARGET_SURPRISAL_PAIRS,
    TARGET_DECISION_VERDICT_PAIRS, TARGET_MARKOV_BLANKET_PAIRS,
    TARGET_EPISTEMIC_VALUE_PAIRS,
)
from lakoff_canonical_vocabulary import (
    UP_DOWN_MML, IN_OUT_MML_CLEAN, FORWARD_BACK_MML, PATH_MOTION_MML,
    LIGHT_DARK_MML, EXISTENCE_MML, FORCE_MML, BALANCE_MML,
    DIFFICULTY_BURDEN_MML,
)
from exp52_target_axis_validation import (
    VALENCE_PAIRS, AROUSAL_PAIRS, COHERENCE_PAIRS,
)


def unit(v):
    return v / np.linalg.norm(v)


def build_axis_from_pairs(wv, pairs, deanisotropy_mean=None):
    """Build axis from anchor offsets. Optionally deanisotropize anchors first."""
    offs = []
    for a, c in pairs:
        if a in wv.key_to_index and c in wv.key_to_index:
            va = wv[a]
            vc = wv[c]
            if deanisotropy_mean is not None:
                va = va - deanisotropy_mean
                vc = vc - deanisotropy_mean
            offs.append(va - vc)
    if not offs:
        return None
    raw = np.stack(offs).mean(axis=0)
    return unit(raw)


def build_R_residualized(wv, deanisotropy_mean):
    eq_raw = build_axis_from_pairs(wv, TARGET_EQUILIBRIUM_RUNAWAY_PAIRS, deanisotropy_mean)
    valence = build_axis_from_pairs(wv, VALENCE_PAIRS, deanisotropy_mean)
    arousal = build_axis_from_pairs(wv, AROUSAL_PAIRS, deanisotropy_mean)
    r = eq_raw - (eq_raw @ valence) * valence
    r = r - (r @ arousal) * arousal
    return unit(r)


def gs_orthogonalize(axes_list):
    """Gram-Schmidt orthonormalize a list of axis vectors."""
    out = []
    for v in axes_list:
        u = v.copy()
        for prev in out:
            u = u - (u @ prev) * prev
        n = np.linalg.norm(u)
        if n < 1e-10:
            continue  # axis collapsed against previous
        out.append(u / n)
    return out


def coverage_in_basis(v, gs_basis_list):
    """Magnitude of v captured by orthonormal basis. Returns sqrt(variance)."""
    v_residual = v.copy()
    for u_gs in gs_basis_list:
        v_residual = v_residual - (v_residual @ u_gs) * u_gs
    explained = float(np.sqrt(max(0, 1 - np.linalg.norm(v_residual) ** 2)))
    return explained


# ============================================================================
# Load GloVe and compute deanisotropization mean
# ============================================================================
print("Loading glove-wiki-gigaword-300...")
wv = api.load("glove-wiki-gigaword-300")
print(f"  vocab: {len(wv.key_to_index)}")

# Compute mean over the full vocab — the dominant anisotropy direction
print("Computing anisotropy mean over full vocab...")
all_vecs = wv.vectors  # shape (400000, 300)
mu = all_vecs.mean(axis=0)
print(f"  mu norm: {np.linalg.norm(mu):.4f}  (vs typical word vector norm ~6)")


def get_deanisotropized(word):
    """Return unit-normalized deanisotropized word vector, or None if OOV."""
    if word not in wv.key_to_index:
        return None
    v = wv[word] - mu
    n = np.linalg.norm(v)
    if n < 1e-10:
        return None
    return v / n


# ============================================================================
# Build all bases
# ============================================================================
print("\nBuilding bases...")

# Basis 1: our 10-axis cognitive basis
print("  Basis 1: 10-axis cognitive (C, W, ATT, INT, R, D, IO, DV, MB, EV)")
cognitive_axes = [
    build_axis_from_pairs(wv, TARGET_REWARD_COMPOSITE_PAIRS, mu),
    build_axis_from_pairs(wv, TARGET_WEIGHT_PAIRS, mu),
    build_axis_from_pairs(wv, ATTENTION_CLEAN_PAIRS, mu),
    build_axis_from_pairs(wv, INTENTION_CLEAN_PAIRS, mu),
    build_R_residualized(wv, mu),
    build_axis_from_pairs(wv, TARGET_SURPRISAL_PAIRS, mu),
    build_axis_from_pairs(wv, IN_OUT_MML_CLEAN, mu),
    build_axis_from_pairs(wv, TARGET_DECISION_VERDICT_PAIRS, mu),
    build_axis_from_pairs(wv, TARGET_MARKOV_BLANKET_PAIRS, mu),
    build_axis_from_pairs(wv, TARGET_EPISTEMIC_VALUE_PAIRS, mu),
]
gs_cognitive_10 = gs_orthogonalize(cognitive_axes)
print(f"    {len(gs_cognitive_10)} axes after GS")

# Basis 2: 10-axis Lakoff image-schema basis
print("  Basis 2: 10-axis Lakoff (UD, IO, FB, LD, PATH, EXIST, FORCE, BAL, DIFF, COH)")
lakoff_axes = [
    build_axis_from_pairs(wv, UP_DOWN_MML, mu),
    build_axis_from_pairs(wv, IN_OUT_MML_CLEAN, mu),
    build_axis_from_pairs(wv, FORWARD_BACK_MML, mu),
    build_axis_from_pairs(wv, LIGHT_DARK_MML, mu),
    build_axis_from_pairs(wv, PATH_MOTION_MML, mu),
    build_axis_from_pairs(wv, EXISTENCE_MML, mu),
    build_axis_from_pairs(wv, FORCE_MML, mu),
    build_axis_from_pairs(wv, BALANCE_MML, mu),
    build_axis_from_pairs(wv, DIFFICULTY_BURDEN_MML, mu),
    build_axis_from_pairs(wv, COHERENCE_PAIRS, mu),
]
gs_lakoff_10 = gs_orthogonalize(lakoff_axes)
print(f"    {len(gs_lakoff_10)} axes after GS")

# Basis 5: 10 cognitive + ABSTRACT_CONCRETE + MODAL_STATUS = 12 axes
print("  Basis 5: 12-axis (cognitive + ABSTRACT_CONCRETE + MODAL_STATUS)")

ABSTRACT_CONCRETE_PAIRS = [
    ("abstract", "concrete"),       ("theoretical", "practical"),
    ("conceptual", "physical"),     ("general", "specific"),
    ("idea", "object"),             ("principle", "instance"),
    ("intangible", "tangible"),     ("notion", "thing"),
    ("categorical", "particular"),  ("ideal", "material"),
]
MODAL_STATUS_PAIRS = [
    ("hypothetical", "actual"),     ("imagined", "observed"),
    ("fictional", "factual"),       ("counterfactual", "real"),
    ("possible", "actual"),         ("could", "can"),           # cleaner per Niamh
    ("might", "is"),                ("simulated", "real"),
    ("speculative", "established"), ("theoretical", "empirical"),
]

abstract_axis = build_axis_from_pairs(wv, ABSTRACT_CONCRETE_PAIRS, mu)
modal_axis = build_axis_from_pairs(wv, MODAL_STATUS_PAIRS, mu)
print(f"    ABSTRACT_CONCRETE built from {sum(1 for a,c in ABSTRACT_CONCRETE_PAIRS if a in wv.key_to_index and c in wv.key_to_index)}/{len(ABSTRACT_CONCRETE_PAIRS)} pairs")
print(f"    MODAL_STATUS    built from {sum(1 for a,c in MODAL_STATUS_PAIRS    if a in wv.key_to_index and c in wv.key_to_index)}/{len(MODAL_STATUS_PAIRS)} pairs")
cognitive_12_axes = cognitive_axes + [abstract_axis, modal_axis]
gs_cognitive_12 = gs_orthogonalize(cognitive_12_axes)
print(f"    {len(gs_cognitive_12)} axes after GS")


# Basis 4 & 6: PCA on deanisotropized GloVe vocab
print("  Bases 4 & 6: PCA on deanisotropized GloVe vocab...")
all_vecs_da = all_vecs - mu  # deanisotropized
# Use sample for speed; 50K is plenty for top-12 PCs
sample_size = 50000
np.random.seed(42)
sample_idx = np.random.choice(len(all_vecs_da), sample_size, replace=False)
sample = all_vecs_da[sample_idx]
# PCA via SVD
U, S, Vt = np.linalg.svd(sample, full_matrices=False)
# Vt rows are principal components in 300-D
pcs_10 = [Vt[i] for i in range(10)]
pcs_12 = [Vt[i] for i in range(12)]
gs_pca_10 = gs_orthogonalize(pcs_10)  # already orthonormal but be safe
gs_pca_12 = gs_orthogonalize(pcs_12)
print(f"    PC variance ratios (top 12): "
      f"{[f'{(s**2 / (S**2).sum()) * 100:.1f}%' for s in S[:12]]}")
print(f"    First 10 PCs cumulative variance: {((S[:10]**2).sum() / (S**2).sum()) * 100:.1f}%")
print(f"    First 12 PCs cumulative variance: {((S[:12]**2).sum() / (S**2).sum()) * 100:.1f}%")


# Basis 3: random orthogonal 10-axis bases (anisotropy-aware)
print("  Basis 3: 20 random anisotropy-orthogonal 10-axis bases...")
random_bases = []
np.random.seed(0)
mu_unit = unit(mu)
for seed_i in range(20):
    # Generate random 10 directions, project out anisotropy mean, GS-orthogonalize
    random_dirs = np.random.randn(10, 300)
    # Project each out of anisotropy direction
    random_dirs = random_dirs - (random_dirs @ mu_unit)[:, None] * mu_unit
    random_bases.append(gs_orthogonalize([random_dirs[i] for i in range(10)]))


# ============================================================================
# Test categories
# ============================================================================
test_categories = {
    "EMOTIONS_AFFECTIVE": [
        "happiness", "sadness", "anger", "disgust", "envy", "jealousy",
        "pride", "humility", "contentment", "longing", "yearning", "delight",
        "melancholy", "rage", "elation", "despair", "serenity", "anguish",
    ],
    "AGENTIVE_STATES": [
        "ambition", "determination", "resignation", "indecision",
        "willpower", "discipline", "procrastination", "perseverance",
        "complacency", "vigilance", "diligence", "industriousness",
        "negligence", "carelessness",
    ],
    "SOCIAL_RELATIONAL": [
        "trust", "betrayal", "friendship", "enmity", "loyalty", "rivalry",
        "respect", "contempt", "admiration", "scorn", "gratitude",
        "resentment", "cooperation", "competition",
    ],
    "EPISTEMIC_STATES": [
        "knowledge", "ignorance", "belief", "doubt", "uncertainty",
        "conviction", "skepticism", "confidence", "hesitation",
        "speculation", "intuition", "memory", "forgetting",
    ],
    "CONCRETE_NOUNS_ANIMATE": [
        "dog", "cat", "horse", "elephant", "sparrow", "frog",
        "tiger", "rabbit", "bee", "spider",
    ],
    "CONCRETE_NOUNS_INANIMATE": [
        "chair", "table", "lamp", "cup", "hammer", "rope",
        "stone", "tree", "river", "mountain", "window", "shovel",
    ],
    "ABSTRACT_FORMAL": [
        "logic", "theorem", "mathematics", "geometry", "algebra",
        "philosophy", "ontology", "epistemology", "definition", "axiom",
    ],
    "ABSTRACT_FORMAL_EXTENDED": [
        "theory", "concept", "abstraction", "generality", "principle",
        "framework", "paradigm", "model", "notion", "schema",
    ],
    "MODAL_HYPOTHETICAL": [
        "hypothetical", "imaginary", "fictional", "supposed", "conjectural",
        "perhaps", "presumably", "allegedly", "supposedly", "putative",
    ],
    "MODAL_ACTUAL": [
        "fact", "evidence", "reality", "observation", "actually",
        "definitely", "certainly", "demonstrably", "factual", "empirical",
    ],
}


# ============================================================================
# Compute coverage across all bases × all categories
# ============================================================================
def coverage_for_category(words, gs_basis):
    cs = []
    for w in words:
        v = get_deanisotropized(w)
        if v is None:
            continue
        cs.append(coverage_in_basis(v, gs_basis))
    return cs


def random_coverage_for_category(words, random_bases_list):
    """Average coverage over multiple random bases. Returns (mean, std)."""
    per_basis_means = []
    for rb in random_bases_list:
        cs = coverage_for_category(words, rb)
        if cs:
            per_basis_means.append(np.mean(cs))
    if not per_basis_means:
        return None, None
    return float(np.mean(per_basis_means)), float(np.std(per_basis_means))


print("\n" + "=" * 110)
print("COVERAGE BY BASIS × CATEGORY (anisotropy-corrected)")
print("=" * 110)

bases_info = [
    ("Cognitive-10", gs_cognitive_10),
    ("Lakoff-10",    gs_lakoff_10),
    ("PCA-10",       gs_pca_10),
    ("Cognitive-12", gs_cognitive_12),
    ("PCA-12",       gs_pca_12),
]

print(f"\n{'category':<28} "
      f"{'Cog-10':>8} {'Lak-10':>8} {'Rand-10':>10} {'PCA-10':>8} "
      f"{'Cog-12':>8} {'PCA-12':>8}")
print("-" * 95)

results = {}
for cat, words in test_categories.items():
    row = f"{cat:<28} "
    cat_results = {}
    for name, basis in bases_info:
        if name.startswith("Rand"):
            continue
        cs = coverage_for_category(words, basis)
        if cs:
            m = np.mean(cs)
            cat_results[name] = m
        else:
            m = float("nan")
        if name == "PCA-10":  # insert random column here for layout
            rand_m, rand_s = random_coverage_for_category(words, random_bases)
            if rand_m is not None:
                row += f"{rand_m * 100:>7.1f}%±{rand_s * 100:.1f} "
                cat_results["Random-10"] = rand_m
            else:
                row += f"{'N/A':>10}"
        row += f"{m * 100:>7.1f}% "
    results[cat] = cat_results
    print(row)


# ============================================================================
# Summary / interpretation
# ============================================================================
print("\n" + "=" * 110)
print("SUMMARY — across all categories, mean coverage by basis:")
print("=" * 110)

basis_means = {}
for name in ["Cognitive-10", "Lakoff-10", "Random-10", "PCA-10",
             "Cognitive-12", "PCA-12"]:
    vals = [r[name] for r in results.values() if name in r]
    if vals:
        basis_means[name] = float(np.mean(vals))
        print(f"  {name:<14}  {basis_means[name] * 100:>6.2f}%")

print("\nKEY COMPARISONS:")
if "Cognitive-10" in basis_means and "PCA-10" in basis_means:
    gap_10 = basis_means["PCA-10"] - basis_means["Cognitive-10"]
    pct_of_oracle_10 = basis_means["Cognitive-10"] / basis_means["PCA-10"] * 100
    print(f"  Cognitive-10 captures {pct_of_oracle_10:.1f}% of oracle-10's coverage")
    print(f"  (gap = {gap_10 * 100:+.2f}pp — how much we're below oracle 10-axis)")

if "Cognitive-10" in basis_means and "Lakoff-10" in basis_means:
    diff = basis_means["Cognitive-10"] - basis_means["Lakoff-10"]
    print(f"  Cognitive-10 vs Lakoff-10:  {diff * 100:+.2f}pp")

if "Cognitive-12" in basis_means and "Cognitive-10" in basis_means:
    diff = basis_means["Cognitive-12"] - basis_means["Cognitive-10"]
    print(f"  Cognitive-12 vs Cognitive-10:  {diff * 100:+.2f}pp")
    print(f"    (does adding ABSTRACT+MODAL substantively help?)")

if "Cognitive-12" in basis_means and "PCA-12" in basis_means:
    pct_of_oracle_12 = basis_means["Cognitive-12"] / basis_means["PCA-12"] * 100
    print(f"  Cognitive-12 captures {pct_of_oracle_12:.1f}% of oracle-12's coverage")

if "Cognitive-10" in basis_means and "Random-10" in basis_means:
    ratio = basis_means["Cognitive-10"] / basis_means["Random-10"]
    print(f"  Cognitive-10 / Random-10 ratio: {ratio:.2f}x")


# ============================================================================
# Per-category light-up patterns (does abstract/modal addition help specifically?)
# ============================================================================
print("\n" + "=" * 110)
print("PER-CATEGORY: does Cognitive-12 light up specifically on new categories?")
print("=" * 110)
print(f"\n{'category':<28} {'Cog-10':>8} {'Cog-12':>8} {'Δ':>8}")
for cat, r in results.items():
    if "Cognitive-10" in r and "Cognitive-12" in r:
        delta = (r["Cognitive-12"] - r["Cognitive-10"]) * 100
        print(f"  {cat:<26} {r['Cognitive-10'] * 100:>6.1f}% "
              f"{r['Cognitive-12'] * 100:>6.1f}%  {delta:>+5.2f}pp")


# Save
np.savez(
    "/Users/macn/Documents/embeddingexp/exp76_results.npz",
    basis_means={k: float(v) for k, v in basis_means.items()},
)
print("\nSaved exp76_results.npz")
