"""
exp84 — Add UD as 13th primitive + refactor high-UD-overlap axes via residualization.

Niamh's hypothesis: UD is a primitive (the most explanatory single one, but
not the only). The four high-UD-overlap axes (C +0.48, INT +0.30, DV +0.36,
REAL_IMAG −0.29) have been the iteration-magnets all session, suggesting
they've been chasing UD-content. Refactor by residualizing them against UD.

Tests:
  1 — Build target_UD; cross-bleed with all 12 axes (confirm pattern)
  2 — Residualize C, INT, DV, REAL_IMAGINARY against UD. Compute cos(A, A_resid)
       — how much did residualization change each axis?
  3 — Pole vocabulary of residualized axes — still domain-coherent?
  4 — Cross-bleed: residualized axes vs each other vs unchanged axes vs UD
  5 — Probe tests: do residualized axes still discriminate their domains?
  6 — Two 13-axis configurations:
       Basis-13a: original 12 + UD (with high-overlap pairs noted)
       Basis-13b: UD + residualized C/INT/DV/REAL_IMAG + unchanged 8 axes
       Compare max off-diagonal magnitude and coverage
  7 — Re-PCA on Basis-13b: is PC1 now UD-dominated?
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


def coverage_in(v, gs_basis):
    v_residual = v.copy()
    for u_gs in gs_basis:
        v_residual = v_residual - (v_residual @ u_gs) * u_gs
    return float(np.sqrt(max(0, 1 - np.linalg.norm(v_residual) ** 2)))


print("Loading GloVe...")
wv = api.load("glove-wiki-gigaword-300")
mu = wv.vectors.mean(axis=0)


print("Building 12 cognitive axes + UD...")
C   = build_axis(wv, TARGET_REWARD_COMPOSITE_PAIRS)
W   = build_axis(wv, TARGET_WEIGHT_PAIRS)
ATT = build_axis(wv, ATTENTION_CLEAN_PAIRS)
INT = build_axis(wv, INTENTION_CLEAN_PAIRS)
R   = build_R(wv)
D   = build_axis(wv, TARGET_SURPRISAL_PAIRS)
IO  = build_axis(wv, IN_OUT_MML_CLEAN)
DV  = build_axis(wv, TARGET_DECISION_VERDICT_PAIRS)
MB  = build_axis(wv, TARGET_MARKOV_BLANKET_PAIRS)
EV  = build_axis(wv, TARGET_EPISTEMIC_VALUE_PAIRS)
ABS = build_axis(wv, ABSTRACT_CONCRETE_PAIRS)
REAL_IMAG = build_axis(wv, REAL_IMAGINARY_PAIRS)
UD  = build_axis(wv, UP_DOWN_MML)


# ============================================================================
# TEST 1 — cos(UD, all 12 axes) — confirm pattern
# ============================================================================
print("\n" + "=" * 78)
print("TEST 1 — cos(UD, 12-axis basis)")
print("=" * 78)

axes_12 = [("C", C), ("W", W), ("ATT", ATT), ("INT", INT), ("R", R),
           ("D", D), ("IO", IO), ("DV", DV), ("MB", MB), ("EV", EV),
           ("ABS", ABS), ("REAL_IMAG", REAL_IMAG)]
print(f"\n{'axis':<12}  {'cos(UD, .)':>12}  refactor candidate?")
print("-" * 50)
for name, vec in axes_12:
    c = cos(UD, vec)
    flag = " ← refactor" if abs(c) > 0.25 else ""
    print(f"{name:<12}  {c:>+12.4f}{flag}")


# ============================================================================
# TEST 2 — Residualize trouble axes against UD
# ============================================================================
print("\n" + "=" * 78)
print("TEST 2 — Residualize C, INT, DV, REAL_IMAG against UD")
print("=" * 78)

def residualize(axis, against):
    return unit(axis - (axis @ against) * against)

C_resid = residualize(C, UD)
INT_resid = residualize(INT, UD)
DV_resid = residualize(DV, UD)
REAL_IMAG_resid = residualize(REAL_IMAG, UD)

print(f"\n{'axis':<14}  {'cos(orig, resid)':>16}  interpretation")
print("-" * 60)
for name, orig, resid in [("C", C, C_resid), ("INT", INT, INT_resid),
                           ("DV", DV, DV_resid),
                           ("REAL_IMAG", REAL_IMAG, REAL_IMAG_resid)]:
    c = cos(orig, resid)
    if c > 0.95:
        interp = "essentially unchanged"
    elif c > 0.85:
        interp = "modest shift"
    elif c > 0.7:
        interp = "substantive shift"
    else:
        interp = "major restructuring"
    print(f"  {name:<12}  {c:>+16.4f}  {interp}")

# Verify residualization made them orthogonal to UD
print(f"\nVerification (should all be ~0):")
for name, resid in [("C_resid", C_resid), ("INT_resid", INT_resid),
                     ("DV_resid", DV_resid), ("REAL_IMAG_resid", REAL_IMAG_resid)]:
    print(f"  cos({name}, UD) = {cos(resid, UD):+.6f}")


# ============================================================================
# TEST 3 — Pole vocabulary of residualized axes
# ============================================================================
print("\n" + "=" * 78)
print("TEST 3 — Pole vocabulary of residualized axes")
print("=" * 78)

for name, resid in [("C_resid", C_resid), ("INT_resid", INT_resid),
                     ("DV_resid", DV_resid), ("REAL_IMAG_resid", REAL_IMAG_resid)]:
    print(f"\n--- {name} positive pole (top 10) ---")
    for w, s in wv.similar_by_vector(resid.astype(np.float32), topn=10):
        print(f"  {w:25s}  {s:+.4f}")
    print(f"--- {name} negative pole (top 10) ---")
    for w, s in wv.similar_by_vector((-resid).astype(np.float32), topn=10):
        print(f"  {w:25s}  {s:+.4f}")


# ============================================================================
# TEST 4 — Cross-bleed of refactored basis (Basis-13b)
# ============================================================================
print("\n" + "=" * 78)
print("TEST 4 — Cross-bleed of Basis-13b (UD + residualized + unchanged 8)")
print("=" * 78)

basis_13b_names = ["UD", "C_resid", "W", "ATT", "INT_resid", "R", "D", "IO",
                   "DV_resid", "MB", "EV", "ABS", "REAL_IMAG_resid"]
basis_13b_vecs = [UD, C_resid, W, ATT, INT_resid, R, D, IO,
                  DV_resid, MB, EV, ABS, REAL_IMAG_resid]

M_13b = np.zeros((13, 13))
for i, vi in enumerate(basis_13b_vecs):
    for j, vj in enumerate(basis_13b_vecs):
        M_13b[i, j] = cos(vi, vj) if i != j else 1.0

print(f"\n{'':<10}" + "".join(f"{n[:5]:>7}" for n in basis_13b_names))
for i, ni in enumerate(basis_13b_names):
    row = f"{ni[:10]:<10}"
    for j in range(13):
        if i == j:
            row += f"  1.000"
        else:
            row += f" {M_13b[i, j]:>+6.3f}"
    print(row)

abs_off = np.abs(M_13b - np.eye(13))
flat = abs_off[np.triu_indices(13, k=1)]
print(f"\nBasis-13b off-diagonal: max = {flat.max():.3f}, "
      f"mean = {flat.mean():.3f}, median = {np.median(flat):.3f}")
print(f"Pairs > 0.35: {(flat > 0.35).sum()}/{len(flat)}")
print(f"Pairs > 0.25: {(flat > 0.25).sum()}/{len(flat)}")

# Find the largest off-diagonal
i_max, j_max = np.unravel_index(np.argmax(abs_off), abs_off.shape)
print(f"Largest off-diagonal: {basis_13b_names[i_max]} ↔ {basis_13b_names[j_max]} = "
      f"{M_13b[i_max, j_max]:+.4f}")

# Compare to Basis-13a (original 12 + UD)
print(f"\n--- Comparison to Basis-13a (original 12 + UD, no residualization) ---")
basis_13a_names = [n for n, _ in axes_12] + ["UD"]
basis_13a_vecs = [v for _, v in axes_12] + [UD]
M_13a = np.zeros((13, 13))
for i, vi in enumerate(basis_13a_vecs):
    for j, vj in enumerate(basis_13a_vecs):
        M_13a[i, j] = cos(vi, vj) if i != j else 1.0
abs_off_13a = np.abs(M_13a - np.eye(13))
flat_13a = abs_off_13a[np.triu_indices(13, k=1)]
print(f"Basis-13a off-diagonal: max = {flat_13a.max():.3f}, "
      f"mean = {flat_13a.mean():.3f}")
print(f"Basis-13a pairs > 0.35: {(flat_13a > 0.35).sum()}")


# ============================================================================
# TEST 5 — Probe tests: do residualized axes still discriminate?
# ============================================================================
print("\n" + "=" * 78)
print("TEST 5 — Probe tests for residualized axes")
print("=" * 78)

probe_tests = {
    "C_resid (value)": (C_resid, [
        ("happiness", "suffering"),
        ("success", "failure"),
        ("joy", "despair"),
        ("blessing", "curse"),
        ("flourishing", "ruin"),  # close to anchor
    ]),
    "INT_resid (commitment)": (INT_resid, [
        ("resolved", "vacillating"),
        ("steadfast", "wavering"),
        ("committed", "hesitant"),  # close to anchor
        ("decisive", "indecisive"),
        ("determined", "irresolute"),
    ]),
    "DV_resid (verdict)": (DV_resid, [
        ("endorsed", "rejected"),
        ("approved", "denied"),
        ("granted", "refused"),
        ("ratified", "vetoed"),
        ("confirmed", "dismissed"),
    ]),
    "REAL_IMAG_resid (real/imaginary)": (REAL_IMAG_resid, [
        ("imagination", "perception"),
        ("fantasy", "memory"),
        ("fiction", "history"),
        ("myth", "fact"),
        ("supposition", "evidence"),
    ]),
}

for axis_name, (axis_vec, pairs) in probe_tests.items():
    print(f"\n--- {axis_name} ---")
    print(f"{'positive':<14} {'negative':<14}  {'cos(pos)':>9}  {'cos(neg)':>9}  Δ")
    for pos, neg in pairs:
        if pos not in wv.key_to_index or neg not in wv.key_to_index:
            print(f"  ({pos}, {neg}) — OOV")
            continue
        cp = cos(unit(wv[pos]), axis_vec)
        cn = cos(unit(wv[neg]), axis_vec)
        delta = cp - cn
        marker = "  ← consistent" if delta > 0 else "  ← reversed"
        print(f"  {pos:<12}  {neg:<12}  {cp:>+9.4f}  {cn:>+9.4f}  {delta:>+.4f}{marker}")


# ============================================================================
# TEST 6 — Coverage comparison: Basis-13a vs Basis-13b
# ============================================================================
print("\n" + "=" * 78)
print("TEST 6 — Coverage: 12-axis vs Basis-13a vs Basis-13b")
print("=" * 78)


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

basis_12_vecs = [v for _, v in axes_12]
gs_12 = gs_orthogonalize(basis_12_vecs)
gs_13a = gs_orthogonalize(basis_13a_vecs)
gs_13b = gs_orthogonalize(basis_13b_vecs)

print(f"\n{'category':<28} {'12-axis':>9} {'13a':>9} {'13b':>9}")
print("-" * 60)
all_12, all_13a, all_13b = [], [], []
for cat, words in test_categories.items():
    cs_12, cs_13a, cs_13b = [], [], []
    for w in words:
        v = get_deanisotropized(w)
        if v is None:
            continue
        cs_12.append(coverage_in(v, gs_12))
        cs_13a.append(coverage_in(v, gs_13a))
        cs_13b.append(coverage_in(v, gs_13b))
    if not cs_12:
        continue
    all_12.extend(cs_12)
    all_13a.extend(cs_13a)
    all_13b.extend(cs_13b)
    print(f"  {cat:<26} {np.mean(cs_12) * 100:>7.1f}% "
          f"{np.mean(cs_13a) * 100:>7.1f}% {np.mean(cs_13b) * 100:>7.1f}%")

print(f"\n  {'OVERALL MEAN':<26} {np.mean(all_12) * 100:>7.2f}% "
      f"{np.mean(all_13a) * 100:>7.2f}% {np.mean(all_13b) * 100:>7.2f}%")
print(f"  Δ from 12-axis:                            "
      f"{(np.mean(all_13a) - np.mean(all_12)) * 100:>+6.2f}pp "
      f"{(np.mean(all_13b) - np.mean(all_12)) * 100:>+6.2f}pp")


# ============================================================================
# TEST 7 — Re-PCA on Basis-13b: is PC1 dominated by UD?
# ============================================================================
print("\n" + "=" * 78)
print("TEST 7 — Re-PCA on Basis-13b (50K word sample)")
print("=" * 78)

B13b = np.stack(gs_13b)  # (13, 300)
np.random.seed(42)
sample_idx = np.random.choice(len(wv.vectors), 50000, replace=False)
sample = wv.vectors[sample_idx] - mu
norms = np.linalg.norm(sample, axis=1, keepdims=True)
norms[norms < 1e-10] = 1
sample = sample / norms

projections = sample @ B13b.T  # (50000, 13)
proj_centered = projections - projections.mean(axis=0, keepdims=True)
_, S, Vt = np.linalg.svd(proj_centered, full_matrices=False)
var_ratio = (S ** 2) / (S ** 2).sum()

print(f"\nVariance per PC (Basis-13b):")
for k in range(13):
    print(f"  PC{k+1:<3}  {var_ratio[k] * 100:>6.2f}%")

print(f"\nTop axis-loadings per PC (Basis-13b):")
for k in range(13):
    pc = Vt[k]
    top_idx = np.argsort(np.abs(pc))[::-1][:4]
    parts = []
    for idx in top_idx:
        sign = "+" if pc[idx] >= 0 else "−"
        parts.append(f"{sign}{abs(pc[idx]):.2f}·{basis_13b_names[idx]}")
    print(f"  PC{k+1:<3} (var {var_ratio[k] * 100:>5.2f}%):  " + " ".join(parts))


np.savez("/Users/macn/Documents/embeddingexp/exp84_results.npz",
         UD=UD,
         C_resid=C_resid, INT_resid=INT_resid,
         DV_resid=DV_resid, REAL_IMAG_resid=REAL_IMAG_resid,
         basis_13b_names=np.array(basis_13b_names),
         M_13b=M_13b)
print("\nSaved exp84_results.npz")
