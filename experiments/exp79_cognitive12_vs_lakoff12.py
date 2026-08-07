"""
exp79 — Our 12-axis basis vs Lakoff-12 basis.

  Our-12   = Cognitive-10 + ABSTRACT_CONCRETE + MODAL_STATUS
  Lakoff-12 = 9 image schemas (UD, IO, FB, LD, PATH, EXIST, FORCE, BAL, DIFF)
              + 3 Lakoff-adjacent cluster axes (COH, SUC, LOSS)

Tests:
  1 — Coverage by category for each (anisotropy-corrected)
  2 — Subspace orthogonality: principal angles between Our-12 and Lakoff-12
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
    SUCCESS_FAILURE_PAIRS, LOSS_PAIRS,
)


def unit(v):
    return v / np.linalg.norm(v)


def build_axis(wv, pairs, mu=None):
    offs = []
    for a, c in pairs:
        if a in wv.key_to_index and c in wv.key_to_index:
            va = wv[a]
            vc = wv[c]
            if mu is not None:
                va = va - mu
                vc = vc - mu
            offs.append(va - vc)
    return unit(np.stack(offs).mean(axis=0))


def build_R(wv, mu):
    eq = build_axis(wv, TARGET_EQUILIBRIUM_RUNAWAY_PAIRS, mu)
    v = build_axis(wv, VALENCE_PAIRS, mu)
    a = build_axis(wv, AROUSAL_PAIRS, mu)
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


def coverage_in_basis(v, gs_basis_list):
    v_residual = v.copy()
    for u_gs in gs_basis_list:
        v_residual = v_residual - (v_residual @ u_gs) * u_gs
    return float(np.sqrt(max(0, 1 - np.linalg.norm(v_residual) ** 2)))


print("Loading GloVe...")
wv = api.load("glove-wiki-gigaword-300")
mu = wv.vectors.mean(axis=0)

# ============================================================================
# Build Our-12
# ============================================================================
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
    ("possible", "actual"),         ("could", "can"),
    ("might", "is"),                ("simulated", "real"),
    ("speculative", "established"), ("theoretical", "empirical"),
]

print("Building Our-12...")
our_axes = [
    ("C",   build_axis(wv, TARGET_REWARD_COMPOSITE_PAIRS, mu)),
    ("W",   build_axis(wv, TARGET_WEIGHT_PAIRS, mu)),
    ("ATT", build_axis(wv, ATTENTION_CLEAN_PAIRS, mu)),
    ("INT", build_axis(wv, INTENTION_CLEAN_PAIRS, mu)),
    ("R",   build_R(wv, mu)),
    ("D",   build_axis(wv, TARGET_SURPRISAL_PAIRS, mu)),
    ("IO",  build_axis(wv, IN_OUT_MML_CLEAN, mu)),
    ("DV",  build_axis(wv, TARGET_DECISION_VERDICT_PAIRS, mu)),
    ("MB",  build_axis(wv, TARGET_MARKOV_BLANKET_PAIRS, mu)),
    ("EV",  build_axis(wv, TARGET_EPISTEMIC_VALUE_PAIRS, mu)),
    ("ABS", build_axis(wv, ABSTRACT_CONCRETE_PAIRS, mu)),
    ("MOD", build_axis(wv, MODAL_STATUS_PAIRS, mu)),
]
gs_ours = gs_orthogonalize([v for _, v in our_axes])
OURS = np.stack(gs_ours)

# ============================================================================
# Build Lakoff-12
# ============================================================================
print("Building Lakoff-12 (9 image schemas + COH + SUC + LOSS)...")
lakoff_axes = [
    ("UD",     build_axis(wv, UP_DOWN_MML, mu)),
    ("IO",     build_axis(wv, IN_OUT_MML_CLEAN, mu)),
    ("FB",     build_axis(wv, FORWARD_BACK_MML, mu)),
    ("LD",     build_axis(wv, LIGHT_DARK_MML, mu)),
    ("PATH",   build_axis(wv, PATH_MOTION_MML, mu)),
    ("EXIST",  build_axis(wv, EXISTENCE_MML, mu)),
    ("FORCE",  build_axis(wv, FORCE_MML, mu)),
    ("BAL",    build_axis(wv, BALANCE_MML, mu)),
    ("DIFF",   build_axis(wv, DIFFICULTY_BURDEN_MML, mu)),
    ("COH",    build_axis(wv, COHERENCE_PAIRS, mu)),
    ("SUC",    build_axis(wv, SUCCESS_FAILURE_PAIRS, mu)),
    ("LOSS",   build_axis(wv, LOSS_PAIRS, mu)),
]
gs_lakoff = gs_orthogonalize([v for _, v in lakoff_axes])
LAKOFF = np.stack(gs_lakoff)

print(f"Our-12 GS: {len(gs_ours)} axes")
print(f"Lakoff-12 GS: {len(gs_lakoff)} axes")


def get_deanisotropized(word):
    if word not in wv.key_to_index:
        return None
    v = wv[word] - mu
    n = np.linalg.norm(v)
    if n < 1e-10:
        return None
    return v / n


# ============================================================================
# Test categories (from exp76)
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
# TEST 1 — Coverage comparison
# ============================================================================
print("\n" + "=" * 84)
print("TEST 1 — coverage by category")
print("=" * 84)
print(f"\n{'category':<28}  {'Our-12':>8}  {'Lakoff-12':>10}  {'Δ':>6}")
print("-" * 62)
all_ours = []
all_lakoff = []
for cat, words in test_categories.items():
    ours_cs = []
    lakoff_cs = []
    for w in words:
        v = get_deanisotropized(w)
        if v is None:
            continue
        ours_cs.append(coverage_in_basis(v, gs_ours))
        lakoff_cs.append(coverage_in_basis(v, gs_lakoff))
    if not ours_cs:
        continue
    om = np.mean(ours_cs)
    lm = np.mean(lakoff_cs)
    all_ours.extend(ours_cs)
    all_lakoff.extend(lakoff_cs)
    delta = (om - lm) * 100
    print(f"{cat:<28}  {om * 100:>7.1f}% {lm * 100:>9.1f}% {delta:>+5.2f}pp")

print(f"\n{'OVERALL MEAN':<28}  "
      f"{np.mean(all_ours) * 100:>7.2f}% "
      f"{np.mean(all_lakoff) * 100:>9.2f}% "
      f"{(np.mean(all_ours) - np.mean(all_lakoff)) * 100:>+5.2f}pp")


# ============================================================================
# TEST 2 — Subspace orthogonality between Our-12 and Lakoff-12
# ============================================================================
print("\n" + "=" * 84)
print("TEST 2 — Principal angles between Our-12 and Lakoff-12 subspaces")
print("=" * 84)

cross = OURS @ LAKOFF.T  # 12×12

print("\nCross-cosine matrix (Our-12 rows × Lakoff-12 cols):")
our_names = [n for n, _ in our_axes]
lakoff_names = [n for n, _ in lakoff_axes]
print(f"{'':<6}" + "".join(f"{n:>7}" for n in lakoff_names))
for i in range(12):
    row = f"{our_names[i]:<6}"
    for j in range(12):
        row += f"{cross[i, j]:>+7.3f}"
    print(row)

# Strongest alignments
print("\nMax |cos| per Our axis (the Lakoff axis it most aligns with):")
for i in range(12):
    best_j = int(np.argmax(np.abs(cross[i])))
    print(f"  {our_names[i]:<6} → {lakoff_names[best_j]:<6}  "
          f"({cross[i, best_j]:+.4f})")

# Principal angles
U_pa, sv, Vt_pa = np.linalg.svd(cross)
print(f"\n{'rank':>4}  {'cos':>8}  {'angle':>8}  interpretation")
print("  " + "-" * 50)
for k in range(12):
    if sv[k] > 0.7:
        interp = "near-aligned"
    elif sv[k] > 0.4:
        interp = "partially aligned"
    elif sv[k] > 0.2:
        interp = "weakly aligned"
    else:
        interp = "near-orthogonal"
    print(f"  {k+1:>4}  {sv[k]:>+8.4f}  {np.degrees(np.arccos(np.clip(sv[k], -1, 1))):>6.2f}°  "
          f"{interp}")

print(f"\nSummary:")
print(f"  Subspaces have {(sv > 0.7).sum()} principal directions with cos > 0.7 "
      f"(near-aligned)")
print(f"  Subspaces have {(sv > 0.5).sum()} principal directions with cos > 0.5")
print(f"  Subspaces have {(sv < 0.3).sum()} principal directions with cos < 0.3 "
      f"(near-orthogonal)")
print(f"  Mean principal-angle cos: {sv.mean():.4f}")
print(f"  Total subspace overlap: {(sv ** 2).sum():.3f} out of 12 max")
print(f"  (overlap fraction: {(sv ** 2).sum() / 12 * 100:.1f}%)")

# Per-axis variance in other subspace
our_in_lakoff = (cross ** 2).sum(axis=1)
lakoff_in_ours = (cross ** 2).sum(axis=0)
print(f"\n{'Our axis':<8}  {'% in Lakoff-12 subspace':>22}")
for i, n in enumerate(our_names):
    print(f"  {n:<6}  {our_in_lakoff[i] * 100:>18.2f}%")
print(f"\nMean: {our_in_lakoff.mean() * 100:.2f}%")
print(f"(Random baseline for 12-D / 290-D: ~{12/290 * 100:.1f}%)")

print(f"\n{'Lakoff axis':<10}  {'% in Our-12 subspace':>22}")
for j, n in enumerate(lakoff_names):
    print(f"  {n:<8}  {lakoff_in_ours[j] * 100:>18.2f}%")
print(f"\nMean: {lakoff_in_ours.mean() * 100:.2f}%")

np.savez("/Users/macn/Documents/embeddingexp/exp79_results.npz",
         cross_cosine=cross,
         principal_angle_cosines=sv,
         our_in_lakoff=our_in_lakoff,
         lakoff_in_ours=lakoff_in_ours,
         mean_ours=float(np.mean(all_ours)),
         mean_lakoff=float(np.mean(all_lakoff)))
print("\nSaved exp79_results.npz")
