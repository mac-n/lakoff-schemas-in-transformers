"""
exp78 — Are our 10 cognitive axes orthogonal to the 10-PC oracle subspace?

If the cognitive-primitives basis and the PCA-oracle basis span largely-orthogonal
subspaces, then:
  - Our basis isn't underperforming PCA for the same structure
  - We're capturing content invisible to variance-maximization
  - The "16pp coverage gap" is largely the PCA capturing tokenization/register
    artifacts that our basis correctly ignores

Tests:
  1 — Full 10×10 cross-cosine matrix (cognitive × PC)
  2 — Principal angles between the two 10-D subspaces (via SVD of cross-cosine)
  3 — For each cognitive axis: fraction of its variance in the 10-PC subspace
  4 — For each PC: fraction of its variance in the 10-cognitive subspace
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

print("Building 10 cognitive axes (deanisotropized via offsets)...")
# Anchor offsets cancel anisotropy by construction; no explicit mu subtract needed.
# But for fair comparison to PCs computed on deanisotropized vocab, we use raw.
cognitive_axes = [
    ("C",   build_axis(wv, TARGET_REWARD_COMPOSITE_PAIRS)),
    ("W",   build_axis(wv, TARGET_WEIGHT_PAIRS)),
    ("ATT", build_axis(wv, ATTENTION_CLEAN_PAIRS)),
    ("INT", build_axis(wv, INTENTION_CLEAN_PAIRS)),
    ("R",   build_R(wv)),
    ("D",   build_axis(wv, TARGET_SURPRISAL_PAIRS)),
    ("IO",  build_axis(wv, IN_OUT_MML_CLEAN)),
    ("DV",  build_axis(wv, TARGET_DECISION_VERDICT_PAIRS)),
    ("MB",  build_axis(wv, TARGET_MARKOV_BLANKET_PAIRS)),
    ("EV",  build_axis(wv, TARGET_EPISTEMIC_VALUE_PAIRS)),
]
cog_names = [n for n, _ in cognitive_axes]
cog_vecs = [v for _, v in cognitive_axes]
gs_cog = gs_orthogonalize(cog_vecs)
COG = np.stack(gs_cog)  # 10 × 300

print("Computing top 10 PCs of deanisotropized GloVe...")
mu = wv.vectors.mean(axis=0)
da_vecs = wv.vectors - mu
np.random.seed(42)
sample = da_vecs[np.random.choice(len(da_vecs), 50000, replace=False)]
_, S, Vt = np.linalg.svd(sample, full_matrices=False)
PCS = Vt[:10]  # 10 × 300, orthonormal


# ============================================================================
# TEST 1 — full 10×10 cross-cosine matrix
# ============================================================================
print("\n" + "=" * 88)
print("TEST 1 — cross-cosine matrix: rows = cognitive axes, cols = PCs")
print("=" * 88)
print()

cross = COG @ PCS.T  # 10 × 10 matrix of cos(cog_i, PC_j)

print(f"{'':<6}" + "".join(f"{f'PC{i+1}':>8}" for i in range(10)))
for i, name in enumerate(cog_names):
    row = f"{name:<6}"
    for j in range(10):
        row += f"{cross[i, j]:>+8.3f}"
    print(row)

print()
print("Max |cos| per cognitive axis:")
for i, name in enumerate(cog_names):
    best_pc = int(np.argmax(np.abs(cross[i])))
    print(f"  {name:<4} → PC{best_pc + 1}  ({cross[i, best_pc]:+.4f})")

print("\nMax |cos| per PC:")
for j in range(10):
    best_cog = int(np.argmax(np.abs(cross[:, j])))
    print(f"  PC{j+1:<2} → {cog_names[best_cog]:<4}  ({cross[best_cog, j]:+.4f})")

print(f"\nGlobal: max |cos| = {np.abs(cross).max():.4f}, "
      f"mean |cos| = {np.abs(cross).mean():.4f}")
print(f"Fraction of cross-cosines with |cos| > 0.3: "
      f"{(np.abs(cross) > 0.3).sum()}/{cross.size}")
print(f"Fraction of cross-cosines with |cos| > 0.2: "
      f"{(np.abs(cross) > 0.2).sum()}/{cross.size}")


# ============================================================================
# TEST 2 — Principal angles between the two 10-D subspaces
# ============================================================================
print("\n" + "=" * 88)
print("TEST 2 — Principal angles between cognitive-10 subspace and PC-10 subspace")
print("=" * 88)
print()
print("For two orthonormal bases A, B: SVD of A @ B.T gives singular values")
print("which are the COSINES of the principal angles between the two subspaces.")
print()
print("  All singular values near 0  →  subspaces are orthogonal")
print("  All near 1                  →  subspaces coincide")
print("  Mix                         →  partial overlap with specific aligned dirs")

U_pa, sv_pa, Vt_pa = np.linalg.svd(cross)
principal_angles_cos = sv_pa
principal_angles_deg = np.degrees(np.arccos(np.clip(sv_pa, -1, 1)))

print(f"\n{'rank':>4}  {'cos':>8}  {'angle (deg)':>12}  interpretation")
print("  " + "-" * 60)
for k in range(10):
    if sv_pa[k] > 0.7:
        interp = "near-aligned"
    elif sv_pa[k] > 0.4:
        interp = "partially aligned"
    elif sv_pa[k] > 0.2:
        interp = "weakly aligned"
    else:
        interp = "near-orthogonal"
    print(f"  {k+1:>4}  {sv_pa[k]:>+8.4f}  {principal_angles_deg[k]:>10.2f}°    {interp}")

print(f"\nSummary:")
print(f"  Subspaces have {(sv_pa > 0.5).sum()} principal directions with cos > 0.5 "
      f"(angle < 60°)")
print(f"  Subspaces have {(sv_pa > 0.7).sum()} principal directions with cos > 0.7 "
      f"(angle < 45°)")
print(f"  Subspaces have {(sv_pa < 0.3).sum()} principal directions with cos < 0.3 "
      f"(near-orthogonal)")
print(f"  Mean principal-angle cos: {sv_pa.mean():.4f}")
print(f"  Median principal-angle cos: {np.median(sv_pa):.4f}")


# ============================================================================
# TEST 3 — Variance of each cognitive axis explained by 10-PC subspace
# ============================================================================
print("\n" + "=" * 88)
print("TEST 3 — For each cognitive axis: fraction explained by 10-PC subspace")
print("=" * 88)
print()
print("If cognitive axes live in non-PC subspace, this should be LOW.")
print()

cog_in_pc = (cross ** 2).sum(axis=1)  # row-wise sum of squared cosines
print(f"{'axis':<6}  {'variance in PC10 subspace':>26}")
for i, name in enumerate(cog_names):
    print(f"  {name:<4}  {cog_in_pc[i] * 100:>20.2f}%")
print(f"\nMean: {cog_in_pc.mean() * 100:.2f}%")
print("(For comparison: a random 10-D subspace would capture ~10/290 ≈ 3.4%)")


# ============================================================================
# TEST 4 — Variance of each PC explained by 10-cognitive subspace
# ============================================================================
print("\n" + "=" * 88)
print("TEST 4 — For each PC: fraction explained by 10-cognitive subspace")
print("=" * 88)
print()

pc_in_cog = (cross ** 2).sum(axis=0)  # col-wise
print(f"{'PC':<6}  {'variance in COG10 subspace':>26}")
for j in range(10):
    print(f"  PC{j+1:<3} {pc_in_cog[j] * 100:>20.2f}%")
print(f"\nMean: {pc_in_cog.mean() * 100:.2f}%")


# ============================================================================
# Headline summary
# ============================================================================
print("\n" + "=" * 88)
print("HEADLINE")
print("=" * 88)
print()
total_overlap = (sv_pa ** 2).sum()
print(f"Total subspace overlap (sum of squared principal-angle cosines): "
      f"{total_overlap:.3f} out of 10 max")
print(f"  → 10 = subspaces coincide; 0 = subspaces orthogonal")
print(f"  → our value: {total_overlap:.3f}")
print()
print(f"Average cognitive axis is {cog_in_pc.mean() * 100:.1f}% in the PC subspace")
print(f"Average PC is {pc_in_cog.mean() * 100:.1f}% in the cognitive subspace")
print(f"(Random baseline for both: ~3.4%)")
print()
if cog_in_pc.mean() < 0.15:
    print("→ STRONG orthogonality: cognitive and PC subspaces are largely separate.")
    print("  Coverage gap is mostly PCs capturing non-cognitive variance.")
elif cog_in_pc.mean() < 0.30:
    print("→ MODERATE separation: substantial cognitive content NOT in PCs.")
elif cog_in_pc.mean() < 0.50:
    print("→ PARTIAL overlap: significant cognitive content lives in PC subspace,")
    print("  but also significant content outside it.")
else:
    print("→ HEAVY overlap: cognitive subspace is largely WITHIN PC subspace.")

np.savez("/Users/macn/Documents/embeddingexp/exp78_results.npz",
         cross_cosine_matrix=cross,
         principal_angle_cosines=sv_pa,
         cog_axes_in_pc_variance=cog_in_pc,
         pc_axes_in_cog_variance=pc_in_cog)
print("\nSaved exp78_results.npz")
