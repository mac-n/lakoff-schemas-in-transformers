"""
exp88 — Test EVALUATIVE = unit(VAL + UD) as a single composite primitive.

VAL and UD overlap at cos +0.60 — they may be two lexical surfaces of one
foundational primitive. Test by building EVALUATIVE explicitly and seeing:

  1. Where does EVALUATIVE sit relative to the 13 cognitive axes?
  2. What does pole vocab look like?
  3. Coverage of EVALUATIVE alone vs (VAL+UD as two axes)
  4. Can the basis be restructured around EVALUATIVE?

Two restructured-basis options:
  Basis-13-EVAL: EVALUATIVE + C_resid + W + ATT + INT_resid + R + D + IO +
                 DV_resid + MB + EV + ABS + REAL_IMAG_resid (replace UD)
  Basis-12-EVAL: same as above but drop C_resid (since C overlaps EVALUATIVE
                 substantially already)
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
from lakoff_canonical_vocabulary import IN_OUT_MML_CLEAN, UP_DOWN_MML
from exp52_target_axis_validation import VALENCE_PAIRS, AROUSAL_PAIRS


def unit(v):
    return v / np.linalg.norm(v)


def cos(a, b):
    return float(unit(a) @ unit(b))


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


def residualize(axis, against):
    return unit(axis - (axis @ against) * against)


def coverage_in(v, gs_basis):
    v_residual = v.copy()
    for u_gs in gs_basis:
        v_residual = v_residual - (v_residual @ u_gs) * u_gs
    return float(np.sqrt(max(0, 1 - np.linalg.norm(v_residual) ** 2)))


print("Loading GloVe...")
wv = api.load("glove-wiki-gigaword-300")
mu = wv.vectors.mean(axis=0)

print("Building all axes including EVALUATIVE = unit(VAL + UD)...")
VALENCE = build_axis(wv, VALENCE_PAIRS)
UD = build_axis(wv, UP_DOWN_MML)
EVALUATIVE = unit(VALENCE + UD)

C   = build_axis(wv, TARGET_REWARD_COMPOSITE_PAIRS)
W   = build_axis(wv, TARGET_WEIGHT_PAIRS)
ATT = build_axis(wv, ATTENTION_CLEAN_PAIRS)
INT = build_axis(wv, INTENTION_CLEAN_PAIRS)
R = build_R(wv)
D = build_axis(wv, TARGET_SURPRISAL_PAIRS)
IO = build_axis(wv, IN_OUT_MML_CLEAN)
DV = build_axis(wv, TARGET_DECISION_VERDICT_PAIRS)
MB = build_axis(wv, TARGET_MARKOV_BLANKET_PAIRS)
EV = build_axis(wv, TARGET_EPISTEMIC_VALUE_PAIRS)
ABS = build_axis(wv, ABSTRACT_CONCRETE_PAIRS)
REAL_IMAG = build_axis(wv, REAL_IMAGINARY_PAIRS)


# ============================================================================
# TEST 1 — Where does EVALUATIVE sit relative to the 13 axes?
# ============================================================================
print("\n" + "=" * 78)
print("TEST 1 — cos(EVALUATIVE, all 13 axes)")
print("=" * 78)

basis_13 = [("UD", UD), ("C", C), ("W", W), ("ATT", ATT), ("INT", INT),
            ("R", R), ("D", D), ("IO", IO), ("DV", DV), ("MB", MB),
            ("EV", EV), ("ABS", ABS), ("REAL_IMAG", REAL_IMAG)]

print(f"\n{'axis':<12}  {'cos(EVAL, .)':>13}")
print("-" * 28)
for name, vec in basis_13:
    print(f"  {name:<10}  {cos(EVALUATIVE, vec):>+13.4f}")

print(f"\nFor reference:")
print(f"  cos(EVAL, VALENCE) = {cos(EVALUATIVE, VALENCE):+.4f}")
print(f"  cos(EVAL, UD)      = {cos(EVALUATIVE, UD):+.4f}")
print(f"  (both should be ~+0.89, since EVAL = unit(VAL+UD) and cos(VAL,UD)=0.60)")


# ============================================================================
# TEST 2 — Pole vocabulary of EVALUATIVE
# ============================================================================
print("\n" + "=" * 78)
print("TEST 2 — EVALUATIVE pole vocabulary")
print("=" * 78)

print("\nEVALUATIVE positive pole (top 15):")
for w, s in wv.similar_by_vector(EVALUATIVE.astype(np.float32), topn=15):
    print(f"  {w:25s}  {s:+.4f}")

print("\nEVALUATIVE negative pole (top 15):")
for w, s in wv.similar_by_vector((-EVALUATIVE).astype(np.float32), topn=15):
    print(f"  {w:25s}  {s:+.4f}")


# ============================================================================
# TEST 3 — Coverage of bases with EVALUATIVE
# ============================================================================
print("\n" + "=" * 78)
print("TEST 3 — Coverage comparison")
print("=" * 78)

# Build candidate bases
# Original 13-axis with UD + C_resid
basis_13b = [UD, residualize(C, UD), W, ATT, residualize(INT, UD), R, D, IO,
             residualize(DV, UD), MB, EV, ABS, residualize(REAL_IMAG, UD)]
gs_13b = gs_orthogonalize(basis_13b)

# Basis-13-EVAL: replace UD with EVALUATIVE, keep C_resid (now residualized
# against EVALUATIVE rather than UD)
C_resid_eval = residualize(C, EVALUATIVE)
INT_resid_eval = residualize(INT, EVALUATIVE)
DV_resid_eval = residualize(DV, EVALUATIVE)
RI_resid_eval = residualize(REAL_IMAG, EVALUATIVE)
basis_13_eval = [EVALUATIVE, C_resid_eval, W, ATT, INT_resid_eval, R, D, IO,
                 DV_resid_eval, MB, EV, ABS, RI_resid_eval]
gs_13_eval = gs_orthogonalize(basis_13_eval)

# Basis-12-EVAL: drop C_resid since EVALUATIVE already captures most of C's content
basis_12_eval = [EVALUATIVE, W, ATT, INT_resid_eval, R, D, IO,
                 DV_resid_eval, MB, EV, ABS, RI_resid_eval]
gs_12_eval = gs_orthogonalize(basis_12_eval)

# EVALUATIVE alone
gs_eval_alone = gs_orthogonalize([EVALUATIVE])

# VAL + UD as 2 axes (for direct comparison)
gs_val_ud = gs_orthogonalize([VALENCE, UD])


def get_deanisotropized(word):
    if word not in wv.key_to_index:
        return None
    v = wv[word] - mu
    n = np.linalg.norm(v)
    if n < 1e-10:
        return None
    return v / n


test_categories = {
    "EMOTIONS_AFFECTIVE": [
        "happiness", "sadness", "anger", "envy", "jealousy", "pride", "humility",
        "contentment", "longing", "delight", "melancholy", "rage", "elation",
    ],
    "AGENTIVE_STATES": [
        "ambition", "determination", "resignation", "willpower", "discipline",
        "procrastination", "perseverance", "complacency", "vigilance",
    ],
    "EPISTEMIC_STATES": [
        "knowledge", "ignorance", "belief", "doubt", "uncertainty",
        "conviction", "skepticism", "confidence", "speculation",
    ],
    "CONCRETE_NOUNS": [
        "chair", "table", "dog", "stone", "tree", "river", "mountain",
        "hammer", "rope", "lamp",
    ],
    "ABSTRACT_FORMAL": [
        "theorem", "philosophy", "ontology", "epistemology", "axiom",
        "principle", "framework", "paradigm",
    ],
    "MODAL_HYPOTHETICAL": [
        "hypothetical", "imaginary", "fictional", "speculative", "conjectural",
        "perhaps", "supposedly",
    ],
}


def avg_coverage_cat(gs_basis, words):
    cs = []
    for w in words:
        v = get_deanisotropized(w)
        if v is None:
            continue
        cs.append(coverage_in(v, gs_basis))
    return np.mean(cs) if cs else None


bases = [
    ("EVAL alone", gs_eval_alone, 1),
    ("VAL+UD (2)", gs_val_ud, 2),
    ("Basis-12-EVAL", gs_12_eval, 12),
    ("Basis-13-EVAL", gs_13_eval, 13),
    ("Basis-13b (orig)", gs_13b, 13),
]

print(f"\n{'category':<26} " + " ".join(f"{n:>16}" for n, _, _ in bases))
all_results = {n: [] for n, _, _ in bases}
for cat, words in test_categories.items():
    row = f"  {cat:<24}"
    for name, basis, _ in bases:
        m = avg_coverage_cat(basis, words)
        if m is not None:
            for w in words:
                v = get_deanisotropized(w)
                if v is not None:
                    all_results[name].append(coverage_in(v, basis))
            row += f"  {m * 100:>14.1f}%"
        else:
            row += f"  {'N/A':>15}"
    print(row)

print("-" * 110)
overall_row = f"  {'OVERALL MEAN':<24}"
for name, _, _ in bases:
    if all_results[name]:
        overall_row += f"  {np.mean(all_results[name]) * 100:>14.2f}%"
print(overall_row)

# Also: max off-diagonal for each multi-axis basis
print("\nMax off-diagonal cross-bleed for each basis:")
for name, basis, k in bases:
    if k < 2:
        continue
    M = np.zeros((len(basis), len(basis)))
    for i in range(len(basis)):
        for j in range(len(basis)):
            M[i, j] = basis[i] @ basis[j] if i != j else 1
    abs_off = np.abs(M - np.eye(len(basis)))
    flat = abs_off[np.triu_indices(len(basis), k=1)]
    print(f"  {name:<18} max = {flat.max():.3f}, mean = {flat.mean():.3f}, "
          f"pairs > 0.35 = {(flat > 0.35).sum()}/{len(flat)}")


np.savez("/Users/macn/Documents/embeddingexp/exp88_results.npz",
         EVALUATIVE=EVALUATIVE)
print("\nSaved exp88_results.npz")
