"""
exp82 — PCA on our 12-axis basis (Niamh's queued move).

What's the internal eigenstructure of the basis when actual word vectors
are projected into it?

Procedure:
  1. Build 12 axes, GS-orthogonalize.
  2. Project a representative word sample onto the 12-D basis subspace.
  3. PCA on the resulting (N, 12) matrix.
  4. Report variance per PC, top axis-loadings per PC, semantic identity per PC.

Interpretation:
  - If PC1 captures most variance with clean loadings on a few axes, that's
    a "super-axis" — a natural primary direction within our basis subspace.
  - PC loading structure tells us which basis axes co-vary in actual word
    use, suggesting they might be aspects of the same underlying primitive.
  - If variance is spread evenly across 12 PCs, the basis is genuinely
    multi-dimensional (no super-axes).
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
    ABSTRACT_CONCRETE_PAIRS, REAL_IMAGINARY_PAIRS,
)
from lakoff_canonical_vocabulary import IN_OUT_MML_CLEAN
from exp52_target_axis_validation import VALENCE_PAIRS, AROUSAL_PAIRS


def unit(v):
    return v / np.linalg.norm(v)


def build_axis(wv, pairs):
    offs = [wv[a] - wv[c] for a, c in pairs
            if a in wv.key_to_index and c in wv.key_to_index]
    return unit(np.stack(offs).mean(axis=0))


def build_R(wv):
    eq = build_axis(wv, TARGET_EQUILIBRIUM_RUNAWAY_PAIRS)
    v = build_axis(wv, VALENCE_PAIRS)
    a = build_axis(wv, AROUSAL_PAIRS)
    r = eq - (eq @ v) * v
    r = r - (r @ a) * a
    return unit(r)


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


print("Loading GloVe...")
wv = api.load("glove-wiki-gigaword-300")
mu = wv.vectors.mean(axis=0)

print("Building 12-axis basis...")
axis_names = ["C", "W", "ATT", "INT", "R", "D", "IO", "DV", "MB", "EV", "ABS", "REAL_IMAG"]
raw_axes = [
    build_axis(wv, TARGET_REWARD_COMPOSITE_PAIRS),
    build_axis(wv, TARGET_WEIGHT_PAIRS),
    build_axis(wv, ATTENTION_CLEAN_PAIRS),
    build_axis(wv, INTENTION_CLEAN_PAIRS),
    build_R(wv),
    build_axis(wv, TARGET_SURPRISAL_PAIRS),
    build_axis(wv, IN_OUT_MML_CLEAN),
    build_axis(wv, TARGET_DECISION_VERDICT_PAIRS),
    build_axis(wv, TARGET_MARKOV_BLANKET_PAIRS),
    build_axis(wv, TARGET_EPISTEMIC_VALUE_PAIRS),
    build_axis(wv, ABSTRACT_CONCRETE_PAIRS),
    build_axis(wv, REAL_IMAGINARY_PAIRS),
]
gs_basis = gs_orthogonalize(raw_axes)
B = np.stack(gs_basis)  # (12, 300), orthonormal
print(f"  GS basis: {len(gs_basis)} axes")


# ============================================================================
# Project a representative word sample onto the 12-axis basis
# ============================================================================
print("\nProjecting vocabulary onto 12-axis basis...")
# Deanisotropize and unit-normalize the vocab sample
np.random.seed(42)
sample_size = 50000
sample_idx = np.random.choice(len(wv.vectors), sample_size, replace=False)
sample_words = [wv.index_to_key[i] for i in sample_idx]
sample_vecs = wv.vectors[sample_idx] - mu  # deanisotropize
norms = np.linalg.norm(sample_vecs, axis=1, keepdims=True)
norms[norms < 1e-10] = 1
sample_vecs = sample_vecs / norms

# Project onto 12 axes → (50000, 12)
projections = sample_vecs @ B.T

print(f"  Projections shape: {projections.shape}")
print(f"  Mean projection magnitude per axis:")
for i, n in enumerate(axis_names):
    print(f"    {n:<12}  {np.abs(projections[:, i]).mean():.4f}  "
          f"(std: {projections[:, i].std():.4f})")


# ============================================================================
# PCA on projections
# ============================================================================
print("\n" + "=" * 80)
print("PCA on (50000 words × 12 basis axes) projection matrix")
print("=" * 80)

# Center first
proj_centered = projections - projections.mean(axis=0, keepdims=True)
# PCA via SVD
U, S, Vt = np.linalg.svd(proj_centered, full_matrices=False)
# Vt is (12, 12) — each row is a PC direction in the 12-D basis space
variance = S ** 2 / proj_centered.shape[0]
var_ratio = variance / variance.sum()

print(f"\n{'PC':<6}  {'variance':>10}  {'cumulative':>10}")
cum = 0
for k in range(12):
    cum += var_ratio[k]
    print(f"  PC{k+1:<3} {var_ratio[k] * 100:>8.2f}%  {cum * 100:>9.2f}%")


# ============================================================================
# PC loadings on basis axes
# ============================================================================
print("\n" + "=" * 80)
print("PC loadings on basis axes (top contributors per PC)")
print("=" * 80)

for k in range(12):
    pc = Vt[k]
    # Top 5 axes by absolute loading
    top_idx = np.argsort(np.abs(pc))[::-1][:5]
    parts = []
    for idx in top_idx:
        sign = "+" if pc[idx] >= 0 else "−"
        parts.append(f"{sign}{abs(pc[idx]):.2f}·{axis_names[idx]}")
    print(f"  PC{k+1:<3} (var {var_ratio[k]*100:>5.2f}%):  " + "  ".join(parts))


# ============================================================================
# PC pole vocabulary — what do the PCs MEAN semantically?
# ============================================================================
print("\n" + "=" * 80)
print("PC semantic content — top words on each PC (back in word-vector space)")
print("=" * 80)

# Each PC is a direction in the 12-D basis space; convert to 300-D direction
pc_in_300d = Vt @ B  # (12, 300)
for k in range(12):
    direction = pc_in_300d[k]
    direction_unit = unit(direction)
    print(f"\n--- PC{k+1} (var {var_ratio[k]*100:.2f}%) ---")
    print(f"  Axis loadings: ", end="")
    top_idx = np.argsort(np.abs(Vt[k]))[::-1][:4]
    parts = []
    for idx in top_idx:
        sign = "+" if Vt[k][idx] >= 0 else "−"
        parts.append(f"{sign}{abs(Vt[k][idx]):.2f}·{axis_names[idx]}")
    print(", ".join(parts))

    print("  Positive pole (top 8):")
    for w, s in wv.similar_by_vector(direction_unit.astype(np.float32), topn=8):
        print(f"    {w:25s}  {s:+.4f}")
    print("  Negative pole (top 8):")
    for w, s in wv.similar_by_vector((-direction_unit).astype(np.float32), topn=8):
        print(f"    {w:25s}  {s:+.4f}")


np.savez("/Users/macn/Documents/embeddingexp/exp82_results.npz",
         axis_names=np.array(axis_names),
         pc_loadings=Vt,
         variance_ratios=var_ratio,
         pc_in_300d=pc_in_300d)
print("\nSaved exp82_results.npz")
